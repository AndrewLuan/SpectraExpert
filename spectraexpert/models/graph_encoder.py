"""
功能概述：以纯 PyTorch 两轮 edge-conditioned GIN 将变长二维分子图编码为检索向量。
主要职责：GraphEncoderOutput 暴露 0/1/2-hop 节点状态、融合 backbone 与归一化 embedding；
_EdgeConditionedGIN 用 source node 加键 embedding 构造消息，并以 index_add_ 累加到目标节点；
GraphEncoder 验证 GraphBatch、独立嵌入各 categorical 字段，对每尺度做 mean/sum readout 后融合。
代码库关系：严格消费 molecular_graph.GraphBatch 与 structure_features 的字段顺序、键类型 ID；
由 StructureAwareRetriever 的候选结构分支使用，不读取 batch 中的 SMILES、fingerprint 或描述符值。
输入输出与副作用：接收包含至少一个非空图的同设备 long tensor batch，返回保留梯度的浮点张量；
支持孤立节点及完全无边图，拒绝缺失 graph id、跨图边和非法形状，避免空池化 NaN；不读写文件。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from spectraexpert.data.molecular_graph import GraphBatch
from spectraexpert.data.structure_features import (
    BOND_TYPE_TO_ID,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)


@dataclass
class GraphEncoderOutput:
    node_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    backbone: torch.Tensor
    embedding: torch.Tensor


class _EdgeConditionedGIN(nn.Module):
    """执行一轮 GINE 更新：MLP((1+eps)h + sum ReLU(h_source + edge))。"""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        aggregate = nodes.new_zeros(nodes.shape)
        aggregate.index_add_(0, target, F.relu(nodes[source] + edges))
        return self.update((1 + self.eps) * nodes + aggregate)


class GraphEncoder(nn.Module):
    """恰好两轮消息传递、三个尺度 mean/sum readout 的分子图编码器。"""

    def __init__(self, hidden_dim: int, projection_dim: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        if num_layers != 2:
            raise ValueError("GraphEncoder requires exactly two (2) message-passing layers")
        if hidden_dim <= 0 or projection_dim <= 0:
            raise ValueError("hidden_dim and projection_dim must be positive")
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.num_layers = num_layers
        self.dropout = dropout
        # Embedding 容量按 structure_features.atom_features 的截断上界配置；不重编码字段。
        node_sizes = (119, 9, 13, 7, 9, 2, 2)
        edge_sizes = (max(BOND_TYPE_TO_ID.values()) + 1, 2, 2)
        self.node_embeddings = nn.ModuleList(nn.Embedding(size, hidden_dim) for size in node_sizes)
        self.edge_embeddings = nn.ModuleList(nn.Embedding(size, hidden_dim) for size in edge_sizes)
        self.layers = nn.ModuleList(_EdgeConditionedGIN(hidden_dim, dropout) for _ in range(2))
        self.readout = nn.Sequential(nn.Linear(hidden_dim * 6, hidden_dim), nn.GELU())
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, projection_dim),
        )

    def _validate_batch(self, graphs: GraphBatch) -> int:
        """检查结构 tensor 的形状和 membership；图数来自 mol_ids，不能用最大 id 猜测。"""
        nodes, edges, indices, membership = (
            graphs.node_features, graphs.edge_features, graphs.edge_index, graphs.graph_index,
        )
        device = self.node_embeddings[0].weight.device
        if any(value.dtype != torch.long or value.device != device
               for value in (nodes, edges, indices, membership)):
            raise ValueError("graph feature/index tensors must be long and on the encoder device")
        if graphs.mol_ids.ndim != 1 or graphs.mol_ids.numel() == 0:
            raise ValueError("GraphBatch requires at least one graph in mol_ids")
        count = graphs.mol_ids.numel()
        if nodes.ndim != 2 or nodes.shape[1] != len(NODE_FEATURE_NAMES) or nodes.shape[0] == 0:
            raise ValueError("node_features must contain nonempty schema-aligned nodes")
        if membership.shape != (nodes.shape[0],):
            raise ValueError("graph_index must have one entry per node")
        if indices.ndim != 2 or indices.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if edges.shape != (indices.shape[1], len(EDGE_FEATURE_NAMES)):
            raise ValueError("edge_features must have shape (E, schema fields)")
        if (membership < 0).any() or (membership >= count).any():
            raise ValueError("graph_index contains an out-of-range graph id")
        if (torch.bincount(membership, minlength=count) == 0).any():
            raise ValueError("GraphBatch cannot contain empty graphs or missing graph ids")
        if indices.numel():
            if (indices < 0).any() or (indices >= nodes.shape[0]).any():
                raise ValueError("edge_index contains an out-of-range node id")
            if (membership[indices[0]] != membership[indices[1]]).any():
                raise ValueError("edges cannot connect different graphs")
        return count

    def forward(self, graphs: GraphBatch) -> GraphEncoderOutput:
        count = self._validate_batch(graphs)
        nodes = sum(embedding(graphs.node_features[:, i])
                    for i, embedding in enumerate(self.node_embeddings))
        edges = sum(embedding(graphs.edge_features[:, i])
                    for i, embedding in enumerate(self.edge_embeddings))
        states = [nodes]
        for layer in self.layers:
            nodes = layer(nodes, graphs.edge_index, edges)
            states.append(nodes)
        sizes = nodes.new_zeros((count, 1))
        sizes.index_add_(0, graphs.graph_index, nodes.new_ones((nodes.shape[0], 1)))
        readouts = []
        for state in states:
            total = state.new_zeros((count, self.hidden_dim))
            total.index_add_(0, graphs.graph_index, state)
            readouts.extend((total / sizes, total))
        backbone = self.readout(torch.cat(readouts, dim=1))
        return GraphEncoderOutput(tuple(states), backbone, F.normalize(self.projection(backbone), dim=1))


__all__ = ["GraphEncoder", "GraphEncoderOutput"]

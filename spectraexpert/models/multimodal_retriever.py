"""
功能概述：将共享权重的两份光谱视图与候选分子图对齐到相同维度的归一化检索空间。
主要职责：MultimodalOutput 保存三个 embedding、对应 backbone 与辅助 logits；
StructureAwareRetriever.forward 对齐 batch 并用两个光谱 backbone 的均值预测描述符、元素和
可选 fingerprint；encode_spectrum 是只接收光谱的查询入口，encode_graph 是候选编码入口；
predict_auxiliary 返回光谱预测描述符和可选 fingerprint 概率，to_config/from_config 提供配置恢复。
代码库关系：组合 SpectrumTransformer 与 GraphEncoder，不使用结构标签作为光谱分支输入；
固定十键 model_config 供 checkpoint 重建，限制未序列化的 dropout/MLP 参数必须使用默认值。
输入输出与副作用：训练接收两个 (B,L) spectrum 和 B 个图，维度不一致或配置不可恢复时报 ValueError；
配置仅含标量，返回副本，不存模块对象，不执行文件或网络访问。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from spectraexpert.data.molecular_graph import GraphBatch
from spectraexpert.models.graph_encoder import GraphEncoder
from spectraexpert.models.spectrum_encoder import SpectrumTransformer


@dataclass
class MultimodalOutput:
    spectrum_view1_embedding: torch.Tensor
    spectrum_view2_embedding: torch.Tensor
    graph_embedding: torch.Tensor
    spectrum_view1_features: torch.Tensor
    spectrum_view2_features: torch.Tensor
    graph_features: torch.Tensor
    descriptor_prediction: torch.Tensor
    element_presence_logits: torch.Tensor
    fingerprint_logits: torch.Tensor | None


class StructureAwareRetriever(nn.Module):
    def __init__(self, spectrum_encoder: SpectrumTransformer, graph_encoder: GraphEncoder,
                 descriptor_dim: int = 15, predict_fingerprint: bool = False):
        super().__init__()
        if spectrum_encoder.projection_dim != graph_encoder.projection_dim:
            raise ValueError("spectrum and graph projection dimensions must match")
        if (spectrum_encoder.mlp_ratio != 4.0 or spectrum_encoder.dropout != 0.1
                or graph_encoder.dropout != 0.1 or graph_encoder.num_layers != 2):
            raise ValueError("fixed checkpoint schema requires spectrum mlp_ratio=4.0, "
                             "spectrum/graph dropout=0.1 and graph num_layers=2")
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        self.spectrum_encoder = spectrum_encoder
        self.graph_encoder = graph_encoder
        self.descriptor_head = nn.Linear(spectrum_encoder.embed_dim, descriptor_dim)
        self.element_head = nn.Linear(spectrum_encoder.embed_dim, 5)
        self.fingerprint_head = (nn.Linear(spectrum_encoder.embed_dim, 2048)
                                 if predict_fingerprint else None)
        self.model_config = {
            "signal_length": int(spectrum_encoder.signal_length),
            "patch_size": int(spectrum_encoder.patch_size),
            "stride": int(spectrum_encoder.stride),
            "embed_dim": int(spectrum_encoder.embed_dim),
            "depth": int(spectrum_encoder.depth),
            "num_heads": int(spectrum_encoder.num_heads),
            "projection_dim": int(spectrum_encoder.projection_dim),
            "graph_hidden_dim": int(graph_encoder.hidden_dim),
            "descriptor_dim": int(descriptor_dim),
            "predict_fingerprint": bool(predict_fingerprint),
        }

    def to_config(self) -> dict[str, object]:
        return dict(self.model_config)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "StructureAwareRetriever":
        spectrum = SpectrumTransformer(**{
            key: int(config[key]) for key in (
                "signal_length", "patch_size", "stride", "embed_dim", "depth", "num_heads", "projection_dim",
            )
        })
        graph = GraphEncoder(hidden_dim=int(config["graph_hidden_dim"]),
                             projection_dim=int(config["projection_dim"]), num_layers=2)
        return cls(spectrum, graph, descriptor_dim=int(config["descriptor_dim"]),
                   predict_fingerprint=bool(config["predict_fingerprint"]))

    def forward(self, spectrum_view1: torch.Tensor, spectrum_view2: torch.Tensor,
                graphs: GraphBatch) -> MultimodalOutput:
        if (spectrum_view1.ndim != 2 or spectrum_view2.shape != spectrum_view1.shape
                or spectrum_view1.shape[0] != graphs.mol_ids.numel()):
            raise ValueError("spectrum views must share shape (B,L) and match the graph batch size")
        h1 = self.spectrum_encoder.encode_backbone(spectrum_view1)
        h2 = self.spectrum_encoder.encode_backbone(spectrum_view2)
        graph_output = self.graph_encoder(graphs)
        h_mean = (h1 + h2) / 2
        return MultimodalOutput(
            spectrum_view1_embedding=self.spectrum_encoder.project(h1),
            spectrum_view2_embedding=self.spectrum_encoder.project(h2),
            graph_embedding=graph_output.embedding,
            spectrum_view1_features=h1,
            spectrum_view2_features=h2,
            graph_features=graph_output.backbone,
            descriptor_prediction=self.descriptor_head(h_mean),
            element_presence_logits=self.element_head(h_mean),
            fingerprint_logits=(self.fingerprint_head(h_mean) if self.fingerprint_head is not None else None),
        )

    def encode_spectrum(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.spectrum_encoder(spectrum)

    def encode_graph(self, graphs: GraphBatch) -> torch.Tensor:
        return self.graph_encoder(graphs).embedding

    def predict_auxiliary(self, spectrum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.spectrum_encoder.encode_backbone(spectrum)
        fingerprint = (torch.sigmoid(self.fingerprint_head(features))
                       if self.fingerprint_head is not None else None)
        return self.descriptor_head(features), fingerprint


__all__ = ["MultimodalOutput", "StructureAwareRetriever"]

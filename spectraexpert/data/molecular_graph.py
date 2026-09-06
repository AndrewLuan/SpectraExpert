"""
功能概述：把 canonical SMILES 转成与结构指标对齐的二维重原子图，并将变长图无 padding 拼批。
主要职责：``MoleculeGraph`` 保存单分子节点、双向边、Morgan 位向量、描述符和兼容性 metadata；
``featurize_smiles`` 先规范化 SMILES 再图化；``GraphBatch`` 表示拼接后的图集合并支持 ``to``；
``collate_graphs`` 累加节点 offset、生成 graph membership 并兼容全部无键分子。
代码库关系：唯一依赖 ``spectraexpert.data.structure_features`` 的 schema、键映射和特征函数；后续
GraphEncoder、dataset collate 与离线缓存均消费这里的 dataclass，不能另建图字段或平行映射。
输入输出与副作用：输入 SMILES、整数 mol_id 或 ``MoleculeGraph`` 序列，输出 CPU tensor dataclass；
显式氢由 RDKit 移除并只保留为原子 H-count 字段，非法/空 SMILES 和空 batch 明确抛 ``ValueError``。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import torch
from rdkit import Chem

from spectraexpert.data.structure_features import (
    EDGE_FEATURE_NAMES,
    StructureFeatureConfig,
    atom_features,
    bond_features,
    molecular_descriptors,
    morgan_fingerprint,
)


@dataclass(frozen=True)
class MoleculeGraph:
    """一个 canonical SMILES 对应的二维重原子图及离线结构标签。"""

    node_features: torch.LongTensor
    edge_index: torch.LongTensor
    edge_features: torch.LongTensor
    fingerprint: torch.BoolTensor
    descriptors: torch.FloatTensor
    mol_id: int
    canonical_smiles: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class GraphBatch:
    """通过节点拼接和 edge offset 表示的无 padding 分子图批次。"""

    node_features: torch.LongTensor
    edge_index: torch.LongTensor
    edge_features: torch.LongTensor
    graph_index: torch.LongTensor
    fingerprints: torch.BoolTensor
    descriptors: torch.FloatTensor
    mol_ids: torch.LongTensor
    canonical_smiles: tuple[str, ...]

    def to(self, device: torch.device | str) -> "GraphBatch":
        """迁移批次的全部 tensor，同时原样保留 canonical SMILES。"""

        return dataclasses.replace(
            self,
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            graph_index=self.graph_index.to(device),
            fingerprints=self.fingerprints.to(device),
            descriptors=self.descriptors.to(device),
            mol_ids=self.mol_ids.to(device),
        )


def _canonicalize_heavy_molecule(smiles: str) -> tuple[str, Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"unable to parse SMILES: {smiles!r}")
    molecule = Chem.RemoveAllHs(molecule)
    if molecule.GetNumAtoms() == 0:
        raise ValueError("SMILES must contain at least one heavy atom")
    canonical_smiles = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    canonical_molecule = Chem.MolFromSmiles(canonical_smiles)
    if canonical_molecule is None or canonical_molecule.GetNumAtoms() == 0:
        raise ValueError(f"unable to canonicalize SMILES: {smiles!r}")
    return canonical_smiles, canonical_molecule


def featurize_smiles(
    smiles: str,
    mol_id: int,
    config: StructureFeatureConfig = StructureFeatureConfig(),
) -> MoleculeGraph:
    """先 canonicalize，再生成七列节点、三列双向边和两类结构监督标签。"""

    canonical_smiles, molecule = _canonicalize_heavy_molecule(smiles)
    node_features = torch.tensor(
        [atom_features(atom) for atom in molecule.GetAtoms()], dtype=torch.long
    )

    directed_edges: list[tuple[int, int]] = []
    directed_edge_features: list[tuple[int, int, int]] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        directed_edges.extend(((begin, end), (end, begin)))
        directed_edge_features.extend((features, features))

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(directed_edge_features, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_features = torch.empty((0, len(EDGE_FEATURE_NAMES)), dtype=torch.long)

    return MoleculeGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        fingerprint=morgan_fingerprint(molecule, config=config),
        descriptors=molecular_descriptors(molecule),
        mol_id=int(mol_id),
        canonical_smiles=canonical_smiles,
        metadata=config.to_metadata(),
    )


def collate_graphs(graphs: Sequence[MoleculeGraph]) -> GraphBatch:
    """按输入顺序拼接变长图，累加 edge node offset 并记录每个节点所属图。"""

    graph_list = list(graphs)
    if not graph_list:
        raise ValueError("collate_graphs requires at least one graph")

    node_chunks: list[torch.Tensor] = []
    edge_index_chunks: list[torch.Tensor] = []
    edge_feature_chunks: list[torch.Tensor] = []
    graph_index_chunks: list[torch.Tensor] = []
    node_offset = 0
    for graph_number, graph in enumerate(graph_list):
        node_chunks.append(graph.node_features)
        graph_index_chunks.append(
            torch.full(
                (graph.node_features.shape[0],), graph_number, dtype=torch.long
            )
        )
        if graph.edge_index.shape[1] > 0:
            edge_index_chunks.append(graph.edge_index + node_offset)
            edge_feature_chunks.append(graph.edge_features)
        node_offset += graph.node_features.shape[0]

    if edge_index_chunks:
        edge_index = torch.cat(edge_index_chunks, dim=1)
        edge_features = torch.cat(edge_feature_chunks, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_features = torch.empty((0, len(EDGE_FEATURE_NAMES)), dtype=torch.long)

    return GraphBatch(
        node_features=torch.cat(node_chunks, dim=0),
        edge_index=edge_index,
        edge_features=edge_features,
        graph_index=torch.cat(graph_index_chunks, dim=0),
        fingerprints=torch.stack([graph.fingerprint for graph in graph_list]),
        descriptors=torch.stack([graph.descriptors for graph in graph_list]),
        mol_ids=torch.tensor([graph.mol_id for graph in graph_list], dtype=torch.long),
        canonical_smiles=tuple(graph.canonical_smiles for graph in graph_list),
    )


__all__ = ["GraphBatch", "MoleculeGraph", "collate_graphs", "featurize_smiles"]

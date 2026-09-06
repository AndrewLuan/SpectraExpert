#!/usr/bin/env python3
"""
功能概述：从调用方显式指定的候选 SMILES Parquet 与 Task 6 feature cache 构建单一版本化图索引。
主要职责：``build_index_from_paths`` 用 Task 7 v2 checkpoint snapshot 恢复 graph encoder 和严格
feature config，验证 SMILES 的唯一 mol_id/canonical identity 与 cache source hash，再通过 DataLoader
分 batch 调用共享 ``build_graph_index``；``_CandidateGraphDataset`` 只保存已验证 MoleculeGraph 顺序；
``_parser/main`` 暴露必填 ``--checkpoint/--smiles-path/--feature-cache/--output`` 和 batch/device 选项。
代码库关系：输入缓存由 ``scripts/cache_molecular_features.py`` 生成并由
``load_feature_cache`` 严格恢复；编码和 NPZ 发布委托 ``spectraexpert.retrieval.graph_index``，因此 CLI、
测试与 Stage 1 使用完全相同的 checkpoint SHA、model/feature/source metadata 契约。
输入输出与副作用：只读取四个显式路径，不扫描目录、不读取 spectra 或真实大型数据目录；``mol_id``
在转换前逐值要求 Python/NumPy exact integer 且位于 int64 范围；目标已存在、重复/非 canonical 结构、
cache provenance 不匹配、空输入或非法 batch 会在编码/写入前失败。成功时写一个原子 NPZ；
checkpoint 只面向可信本地文件，底层 Task 7 loader 的 ``weights_only=False`` 安全边界保持不变。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from spectraexpert.data.molecular_graph import MoleculeGraph, collate_graphs
from spectraexpert.data.structure_dataset import (
    canonical_smiles_sha256,
    canonicalize_smiles,
    load_feature_cache,
)
from spectraexpert.data.structure_features import StructureFeatureConfig
from spectraexpert.retrieval.graph_index import GraphIndex, build_graph_index
from spectraexpert.training.checkpoints import load_checkpoint, read_checkpoint_snapshot


class _CandidateGraphDataset(Dataset):
    """按显式 SMILES 行序暴露已验证图，并携带 index builder 要求的审计 metadata。"""

    def __init__(self, graphs: list[MoleculeGraph], feature_config: StructureFeatureConfig,
                 source_smiles_sha256: str) -> None:
        self.graphs = tuple(graphs)
        self.feature_config = feature_config
        self.source_smiles_sha256 = source_smiles_sha256

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> MoleculeGraph:
        return self.graphs[index]


def _candidate_records(path: Path) -> tuple[list[int], list[str]]:
    """读取单一 Parquet 并返回严格 int64 ID 与 canonical SMILES，拒绝任意重复 identity。"""
    if not path.is_file():
        raise FileNotFoundError(f"candidate SMILES Parquet not found: {path}")
    frame = pd.read_parquet(path)
    if "mol_id" not in frame:
        raise ValueError("candidate SMILES Parquet requires mol_id")
    smiles_column = ("smiles_canonical" if "smiles_canonical" in frame
                     else "smiles" if "smiles" in frame else None)
    if smiles_column is None:
        raise ValueError("candidate SMILES Parquet requires smiles_canonical or smiles")
    if frame.empty:
        raise ValueError("candidate SMILES Parquet is empty")
    mol_ids = []
    minimum, maximum = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    for raw in frame["mol_id"].array:
        if (isinstance(raw, (bool, np.bool_))
                or not isinstance(raw, (int, np.integer))):
            raise ValueError("candidate mol_id must be exact int64-compatible integer")
        value = int(raw)
        if value < minimum or value > maximum:
            raise ValueError("candidate mol_id must be exact int64-compatible integer")
        mol_ids.append(value)
    if len(set(mol_ids)) != len(mol_ids):
        raise ValueError("duplicate mol_id in candidate SMILES Parquet")
    canonical = [canonicalize_smiles(value, mol_id=identifier)
                 for identifier, value in zip(mol_ids, frame[smiles_column].tolist())]
    if smiles_column == "smiles_canonical" and canonical != frame[smiles_column].tolist():
        raise ValueError("smiles_canonical column contains non-canonical SMILES")
    if len(set(canonical)) != len(canonical):
        raise ValueError("duplicate canonical SMILES in candidate Parquet")
    return mol_ids, canonical


def build_index_from_paths(checkpoint_path: str | Path, smiles_path: str | Path,
                           feature_cache_path: str | Path, output_path: str | Path, *,
                           batch_size: int = 1024, device: str | None = None) -> GraphIndex:
    """校验四个显式输入并用同一 checkpoint snapshot 分批构建 v2 图索引。"""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"graph index already exists: {destination}")
    mol_ids, canonical = _candidate_records(Path(smiles_path))
    source_hash = canonical_smiles_sha256(mol_ids, canonical)
    snapshot = read_checkpoint_snapshot(checkpoint_path)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(snapshot, expected_mode="spectrum_graph_v2",
                                     device=resolved_device)
    try:
        feature_config = StructureFeatureConfig(**payload["feature_config"])
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint feature_config cannot construct Task 2 schema") from error
    graph_by_id = load_feature_cache(feature_cache_path, expected_config=feature_config,
        expected_source_smiles_sha256=source_hash, required_mol_ids=mol_ids)
    graphs = []
    for identifier, identity in zip(mol_ids, canonical):
        graph = graph_by_id[identifier]
        if graph.canonical_smiles != identity:
            raise ValueError(f"source_smiles_sha256 identity mismatch for mol_id={identifier}")
        graphs.append(graph)
    dataset = _CandidateGraphDataset(graphs, feature_config, source_hash)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    return build_graph_index(model, loader, snapshot, destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a versioned graph candidate index")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--smiles-path", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    index = build_index_from_paths(args.checkpoint, args.smiles_path, args.feature_cache,
                                   args.output, batch_size=args.batch_size, device=args.device)
    print(f"indexed {len(index.mol_ids)} molecular graphs to {args.output} "
          f"(checkpoint_sha256={index.metadata['checkpoint_sha256']})")


if __name__ == "__main__":
    main()

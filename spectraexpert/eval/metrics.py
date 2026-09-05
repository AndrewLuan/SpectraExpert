"""
功能概述：为光谱检索候选提供版本化的分子结构质量指标，包括 GED、nGED 与 Morgan
Tanimoto，避免旧版键阶截断把芳香键误判为单键。
主要职责：``smiles_to_nx`` 依版本把 SMILES 转为重原子图；``graph_edit_distance`` 分别执行
legacy 首个上界和 chemical 完整搜索语义；``normalized_graph_edit_distance`` 用两图节点/边总数
归一化；``pairwise_tanimoto`` 计算可微批量交并比；``tanimoto_similarity`` 与
``compute_retrieval_metrics`` 为评估报告提供单对和候选列表聚合结果。
代码库关系：chemical 图键类别与 Morgan 指纹唯一复用
``spectraexpert.data.structure_features`` 的 ``bond_features``、``BOND_TYPE_TO_ID`` 和默认
``StructureFeatureConfig``；``spectraexpert.eval.evaluate`` 消费本模块，不应另行定义键类别或
Morgan 参数。
输入输出与副作用：公开 GED 接收两个 SMILES，非法 SMILES、超时或未完成搜索返回 ``None``；
legacy 仅为历史复现而返回 generator 首项，chemical 仅在 generator 耗尽后返回最小值。本模块
不读取数据或写入状态；GED 在隔离 fork worker 中运行，父进程在给定 wall-clock deadline 后终止
worker 并回收其资源，因此主线程与后台线程都不会被长期不 yield 的搜索阻塞。
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Literal, Optional

import networkx as nx
import torch
from rdkit import Chem

from spectraexpert.data.structure_features import bond_features, morgan_fingerprint


GEDVersion = Literal["legacy_ged_v1", "chemical_ged_v2"]
DEFAULT_GED_VERSION: GEDVersion = "chemical_ged_v2"


def _validate_version(version: str) -> GEDVersion:
    if version not in {"legacy_ged_v1", "chemical_ged_v2"}:
        raise ValueError(
            "version must be 'legacy_ged_v1' or 'chemical_ged_v2', "
            f"got {version!r}"
        )
    return version  # type: ignore[return-value]


def _bond_type_id(bond: Chem.Bond, version: GEDVersion) -> int:
    """返回历史截断键阶或 Task 2 的五类键类型 ID。"""

    if version == "legacy_ged_v1":
        return int(bond.GetBondTypeAsDouble())
    return bond_features(bond)[0]


def _molecule_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """解析至少含一个原子的 SMILES；空字符串和无效输入统一返回 ``None``。"""

    try:
        molecule = Chem.MolFromSmiles(smiles)
    except (TypeError, ValueError):
        return None
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    return molecule


def smiles_to_nx(
    smiles: str,
    include_h: bool = False,
    version: GEDVersion = DEFAULT_GED_VERSION,
) -> Optional[nx.Graph]:
    """将 SMILES 转为指定 GED schema 的 NetworkX 图，解析失败返回 ``None``。"""

    version = _validate_version(version)
    molecule = _molecule_from_smiles(smiles)
    if molecule is None:
        return None
    if not include_h:
        molecule = Chem.RemoveHs(molecule)
    if molecule.GetNumAtoms() == 0:
        return None

    graph = nx.Graph()
    for atom in molecule.GetAtoms():
        graph.add_node(atom.GetIdx(), atomic_num=int(atom.GetAtomicNum()))
    for bond in molecule.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=_bond_type_id(bond, version),
        )
    return graph


def _node_substitution_cost(left: dict, right: dict) -> float:
    return float(left.get("atomic_num") != right.get("atomic_num"))


def _node_insertion_or_deletion_cost(_: dict) -> float:
    return 1.0


def _edge_substitution_cost(left: dict, right: dict) -> float:
    return float(left.get("bond_type") != right.get("bond_type"))


def _edge_insertion_or_deletion_cost(_: dict) -> float:
    return 1.0


def _consume_optimized_ged(
    left: nx.Graph,
    right: nx.Graph,
    version: GEDVersion,
) -> Optional[float]:
    """在 worker 内完整消费 generator，复现 legacy 或 chemical 的版本化语义。"""

    distances = nx.optimize_graph_edit_distance(
        left,
        right,
        node_subst_cost=_node_substitution_cost,
        node_del_cost=_node_insertion_or_deletion_cost,
        node_ins_cost=_node_insertion_or_deletion_cost,
        edge_subst_cost=_edge_substitution_cost,
        edge_del_cost=_edge_insertion_or_deletion_cost,
        edge_ins_cost=_edge_insertion_or_deletion_cost,
    )
    if version == "legacy_ged_v1":
        return next((float(distance) for distance in distances), None)

    minimum: Optional[float] = None
    for distance in distances:
        value = float(distance)
        minimum = value if minimum is None else min(minimum, value)
    return minimum


def _ged_worker(
    connection: object,
    left: nx.Graph,
    right: nx.Graph,
    version: GEDVersion,
) -> None:
    """在独立进程中执行搜索；普通搜索异常以 ``None`` 回传而不捕获中断信号。"""

    try:
        result = _consume_optimized_ged(left, right, version)
    except Exception:
        result = None
    try:
        connection.send(result)  # type: ignore[attr-defined]
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()  # type: ignore[attr-defined]


def _graph_edit_distance_from_graphs(
    left: nx.Graph,
    right: nx.Graph,
    timeout: float,
    version: GEDVersion,
) -> Optional[float]:
    """在可终止 worker 中执行版本化 GED，超过 deadline 或异常均返回 ``None``。"""

    if timeout <= 0:
        return None
    deadline = time.monotonic() + timeout
    process: Optional[mp.Process] = None
    process_started = False
    parent_connection = None
    child_connection = None
    try:
        context = mp.get_context("fork")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_ged_worker,
            args=(child_connection, left, right, version),
        )
        process.start()
        process_started = True
        child_connection.close()
        child_connection = None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        process.join(remaining)
        if process.is_alive() or time.monotonic() > deadline:
            return None
        if not parent_connection.poll():
            return None
        return parent_connection.recv()
    except (EOFError, OSError, ValueError):
        return None
    finally:
        if child_connection is not None:
            child_connection.close()
        if parent_connection is not None:
            parent_connection.close()
        if process is not None and process_started:
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()


def graph_edit_distance(
    smiles_a: str,
    smiles_b: str,
    timeout: float = 10.0,
    include_h: bool = False,
    version: GEDVersion = DEFAULT_GED_VERSION,
) -> Optional[float]:
    """返回版本化 GED；chemical 只报告完整 bounded 搜索的最小值。"""

    version = _validate_version(version)
    left = smiles_to_nx(smiles_a, include_h=include_h, version=version)
    right = smiles_to_nx(smiles_b, include_h=include_h, version=version)
    if left is None or right is None:
        return None
    return _graph_edit_distance_from_graphs(left, right, timeout, version)


def normalized_graph_edit_distance(
    smiles_a: str,
    smiles_b: str,
    timeout: float = 10.0,
    include_h: bool = False,
    version: GEDVersion = DEFAULT_GED_VERSION,
) -> Optional[float]:
    """以 ``|Vq| + |Vc| + |Eq| + |Ec|`` 固定分母返回 nGED。"""

    version = _validate_version(version)
    left = smiles_to_nx(smiles_a, include_h=include_h, version=version)
    right = smiles_to_nx(smiles_b, include_h=include_h, version=version)
    if left is None or right is None:
        return None
    denominator = len(left) + len(right) + left.number_of_edges() + right.number_of_edges()
    if denominator == 0:
        return None
    distance = _graph_edit_distance_from_graphs(left, right, timeout, version)
    return None if distance is None else distance / denominator


def batch_graph_edit_distance(
    ground_truth: str,
    candidates: list[str],
    timeout: float = 10.0,
    include_h: bool = False,
    version: GEDVersion = DEFAULT_GED_VERSION,
) -> list[Optional[float]]:
    """对候选列表逐项计算同一 GED schema 的有界距离。"""

    version = _validate_version(version)
    ground_truth_graph = smiles_to_nx(
        ground_truth, include_h=include_h, version=version
    )
    if ground_truth_graph is None:
        return [None] * len(candidates)

    results: list[Optional[float]] = []
    for candidate in candidates:
        candidate_graph = smiles_to_nx(candidate, include_h=include_h, version=version)
        if candidate_graph is None:
            results.append(None)
            continue
        results.append(
            _graph_edit_distance_from_graphs(
                ground_truth_graph, candidate_graph, timeout, version
            )
        )
    return results


def pairwise_tanimoto(fingerprints: torch.Tensor) -> torch.Tensor:
    """返回二维 fingerprint tensor 的可微 NxN Tanimoto 相似度矩阵。"""

    if fingerprints.ndim != 2:
        raise ValueError("fingerprints must be a two-dimensional tensor")
    bits = fingerprints.float()
    intersection = bits @ bits.T
    union = bits.sum(1, keepdim=True) + bits.sum(1).unsqueeze(0) - intersection
    return torch.where(
        union > 0,
        intersection / union.clamp_min(1),
        torch.ones_like(union),
    )


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> Optional[float]:
    """用 Task 2 默认 Morgan schema 返回两条 SMILES 的 Tanimoto 相似度。"""

    molecule_a = _molecule_from_smiles(smiles_a)
    molecule_b = _molecule_from_smiles(smiles_b)
    if molecule_a is None or molecule_b is None:
        return None
    try:
        fingerprints = torch.stack(
            [morgan_fingerprint(molecule_a), morgan_fingerprint(molecule_b)]
        )
    except ValueError:
        return None
    return float(pairwise_tanimoto(fingerprints)[0, 1])


def compute_retrieval_metrics(
    ground_truth: str,
    candidate_smiles: list[str],
    candidate_similarities: list[float],
) -> dict:
    """聚合按检索 rank 排列的候选 GED 与 Task 2 Morgan Tanimoto 指标。"""

    if not candidate_smiles:
        return {
            "top1_smiles": "",
            "top1_sim": 0.0,
            "top1_ged": None,
            "min_ged_top5": None,
            "min_ged_top10": None,
            "min_ged_all": None,
            "rank_of_min_ged": -1,
            "best_ged": None,
            "best_ged_smiles": "",
            "top1_tanimoto": None,
            "tanimoto_top5": None,
            "tanimoto_top10": None,
            "num_valid_candidates": 0,
        }

    top1_ged = graph_edit_distance(ground_truth, candidate_smiles[0], timeout=10.0)
    all_ged = batch_graph_edit_distance(ground_truth, candidate_smiles, timeout=10.0)
    valid_ged = [
        (index, distance)
        for index, distance in enumerate(all_ged)
        if distance is not None
    ]

    top1_tanimoto = tanimoto_similarity(ground_truth, candidate_smiles[0])
    tanimoto_top5 = [
        tanimoto_similarity(ground_truth, candidate) for candidate in candidate_smiles[:5]
    ]
    tanimoto_top10 = [
        tanimoto_similarity(ground_truth, candidate) for candidate in candidate_smiles[:10]
    ]

    def minimum_distance(limit: int) -> Optional[float]:
        distances = [distance for index, distance in valid_ged if index < limit]
        return min(distances) if distances else None

    min_ged_all = min((distance for _, distance in valid_ged), default=None)
    if min_ged_all is None:
        rank_of_min_ged = -1
        best_ged_smiles = ""
    else:
        best_index, _ = next(
            (index, distance)
            for index, distance in valid_ged
            if distance == min_ged_all
        )
        rank_of_min_ged = best_index + 1
        best_ged_smiles = candidate_smiles[best_index]

    return {
        "top1_smiles": candidate_smiles[0],
        "top1_sim": float(candidate_similarities[0]),
        "top1_ged": top1_ged,
        "min_ged_top5": minimum_distance(5),
        "min_ged_top10": minimum_distance(10),
        "min_ged_all": min_ged_all,
        "rank_of_min_ged": rank_of_min_ged,
        "best_ged": min_ged_all,
        "best_ged_smiles": best_ged_smiles,
        "top1_tanimoto": top1_tanimoto,
        "tanimoto_top5": [value for value in tanimoto_top5 if value is not None],
        "tanimoto_top10": [value for value in tanimoto_top10 if value is not None],
        "num_valid_candidates": len(valid_ged),
    }


__all__ = [
    "DEFAULT_GED_VERSION",
    "GEDVersion",
    "batch_graph_edit_distance",
    "compute_retrieval_metrics",
    "graph_edit_distance",
    "normalized_graph_edit_distance",
    "pairwise_tanimoto",
    "smiles_to_nx",
    "tanimoto_similarity",
]

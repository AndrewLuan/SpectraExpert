"""
功能概述：为光谱检索候选提供版本化的分子结构质量指标，包括 GED、nGED 与 Morgan
Tanimoto，避免旧版键阶截断把芳香键误判为单键。
主要职责：``smiles_to_nx`` 依版本把 SMILES 转为重原子图；``graph_edit_distance`` 分别执行
legacy 首个上界和 chemical 完整搜索语义；``normalized_graph_edit_distance`` 用两图节点/边总数
归一化；``pairwise_tanimoto`` 计算可微批量交并比；``tanimoto_similarity`` 与
``compute_retrieval_metrics`` 保留旧报告契约；``rank_structure_candidates`` 在既有排序之后执行
identity 排除并记录逐候选化学质量，``aggregate_structure_retrieval`` 汇总完整 Top-K 验收指标。
代码库关系：chemical 图键类别与 Morgan 指纹唯一复用
``spectraexpert.data.structure_features`` 的 ``bond_features``、``BOND_TYPE_TO_ID`` 和默认
``StructureFeatureConfig``；``spectraexpert.eval.evaluate`` 消费本模块，不应另行定义键类别或
Morgan 参数。
输入输出与副作用：公开 GED 接收两个 SMILES，非法 SMILES、超时或未完成搜索返回 ``None``；
legacy 仅为历史复现而返回 generator 首项，chemical 仅在 generator 耗尽后返回最小值。本模块
不读取数据或写入状态；GED 的解析、构图和搜索都在隔离 spawn worker 中运行，父进程在 deadline
后以有限 terminate/kill/join 回收资源，因此主线程与后台线程均不会被长期工作阻塞。
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from collections.abc import Sequence
from typing import Literal, Optional

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr

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
    if molecule is None or molecule.GetNumHeavyAtoms() == 0:
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
    if molecule.GetNumHeavyAtoms() == 0:
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
    search_factory: Optional[Callable[..., object]] = None,
) -> Optional[float]:
    """在 worker 内完整消费 generator，复现 legacy 或 chemical 的版本化语义。"""

    factory = search_factory or nx.optimize_graph_edit_distance
    distances = factory(
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


def _worker_metric(
    smiles_a: str,
    smiles_b: str,
    include_h: bool,
    version: str,
    normalized: bool,
    search_factory: Optional[Callable[..., object]] = None,
) -> Optional[float]:
    """在 worker 中从 SMILES 完成解析、构图、GED 及可选 nGED。"""

    validated_version = _validate_version(version)
    left = smiles_to_nx(smiles_a, include_h=include_h, version=validated_version)
    right = smiles_to_nx(smiles_b, include_h=include_h, version=validated_version)
    if left is None or right is None:
        return None
    distance = _consume_optimized_ged(
        left, right, validated_version, search_factory=search_factory
    )
    if distance is None or not normalized:
        return distance
    denominator = len(left) + len(right) + left.number_of_edges() + right.number_of_edges()
    return None if denominator == 0 else distance / denominator


def _ged_worker(
    connection: object,
    smiles_a: str,
    smiles_b: str,
    include_h: bool,
    version: str,
    normalized: bool,
) -> None:
    """在独立进程中执行搜索；普通搜索异常以 ``None`` 回传而不捕获中断信号。"""

    try:
        result = _worker_metric(smiles_a, smiles_b, include_h, version, normalized)
    except Exception:
        result = None
    try:
        connection.send(result)  # type: ignore[attr-defined]
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()  # type: ignore[attr-defined]


_GED_WORKER_TARGET: Callable[..., None] = _ged_worker
_TERMINATE_JOIN_SECONDS = 0.05
_KILL_JOIN_SECONDS = 0.05


def _cleanup_worker_process(process: mp.Process, started: bool) -> None:
    """只用有限 join 回收 worker；SIGTERM 无效时升级为 kill。"""

    if not started:
        return
    if process.is_alive():
        process.terminate()
        process.join(_TERMINATE_JOIN_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(_KILL_JOIN_SECONDS)
    if not process.is_alive():
        process.join(0)
        process.close()


def _run_ged_worker(
    smiles_a: str,
    smiles_b: str,
    timeout: float,
    include_h: bool,
    version: GEDVersion,
    normalized: bool,
) -> Optional[float]:
    """以 spawn worker 执行完整 GED 路径，并在解析前建立 wall-clock deadline。"""

    if timeout <= 0:
        return None
    deadline = time.monotonic() + timeout
    process: Optional[mp.Process] = None
    process_started = False
    parent_connection = None
    child_connection = None
    try:
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_GED_WORKER_TARGET,
            args=(
                child_connection,
                smiles_a,
                smiles_b,
                include_h,
                version,
                normalized,
            ),
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
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not parent_connection.poll(remaining):
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
            _cleanup_worker_process(process, process_started)


def graph_edit_distance(
    smiles_a: str,
    smiles_b: str,
    timeout: float = 10.0,
    include_h: bool = False,
    version: GEDVersion = DEFAULT_GED_VERSION,
) -> Optional[float]:
    """返回版本化 GED；chemical 只报告完整 bounded 搜索的最小值。"""

    validated_version = _validate_version(version)
    if timeout <= 0:
        return None
    return _run_ged_worker(
        smiles_a, smiles_b, timeout, include_h, validated_version, normalized=False
    )


def normalized_graph_edit_distance(
    smiles_a: str,
    smiles_b: str,
    timeout: float = 10.0,
    include_h: bool = False,
    version: GEDVersion = DEFAULT_GED_VERSION,
) -> Optional[float]:
    """以 ``|Vq| + |Vc| + |Eq| + |Ec|`` 固定分母返回 nGED。"""

    validated_version = _validate_version(version)
    if timeout <= 0:
        return None
    return _run_ged_worker(
        smiles_a, smiles_b, timeout, include_h, validated_version, normalized=True
    )


def batch_graph_edit_distance(
    ground_truth: str,
    candidates: list[str],
    timeout: float = 10.0,
    include_h: bool = False,
    version: GEDVersion = DEFAULT_GED_VERSION,
) -> list[Optional[float]]:
    """对候选列表逐项计算同一 GED schema 的有界距离。"""

    validated_version = _validate_version(version)
    if timeout <= 0:
        return [None] * len(candidates)
    return [
        _run_ged_worker(
            ground_truth,
            candidate,
            timeout,
            include_h,
            validated_version,
            normalized=False,
        )
        for candidate in candidates
    ]


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


def _canonical_molecule(smiles: str) -> tuple[str, Chem.Mol]:
    """返回用于 identity 与组成审计的 canonical SMILES 和重原子分子。"""

    molecule = _molecule_from_smiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid non-empty molecular SMILES: {smiles!r}")
    heavy = Chem.RemoveAllHs(molecule)
    return Chem.MolToSmiles(heavy, canonical=True), heavy


def _scaffold_identity(canonical_smiles: str, molecule: Chem.Mol) -> str:
    """返回 Bemis–Murcko scaffold；无环分子回退 canonical identity。"""

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule, includeChirality=True
    )
    return f"scaffold:{scaffold}" if scaffold else f"identity:{canonical_smiles}"


def _finite_scores(values: Sequence[float], name: str) -> list[float]:
    """把一维有限分数序列规范为 Python float，供严格 JSON 序列化。"""

    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def rank_structure_candidates(
    query_smiles: str,
    candidate_smiles: Sequence[str],
    scores: Sequence[float],
    *,
    cosine_scores: Sequence[float] | None = None,
    exclude_identity: bool = False,
    top_k: int | None = None,
    ged_timeout: float = 10.0,
    training_atomic_numbers: Sequence[int] = (1, 6, 7, 8, 9),
) -> dict[str, object]:
    """评估已有 rank，不自行排序；真实结构仅在 retrieval 已完成后进入本函数。"""

    if type(exclude_identity) is not bool:
        raise ValueError("exclude_identity must be bool")
    if top_k is not None and (
        not isinstance(top_k, (int, np.integer))
        or isinstance(top_k, (bool, np.bool_))
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer or None")
    identities = list(candidate_smiles)
    ranked_scores = _finite_scores(scores, "scores")
    embedding_scores = _finite_scores(
        ranked_scores if cosine_scores is None else cosine_scores,
        "cosine_scores",
    )
    if len(identities) != len(ranked_scores) or len(identities) != len(embedding_scores):
        raise ValueError("candidate_smiles, scores, and cosine_scores must have equal length")
    query_identity, query_molecule = _canonical_molecule(query_smiles)
    query_atoms = {atom.GetAtomicNum() for atom in query_molecule.GetAtoms()}
    query_scaffold = _scaffold_identity(query_identity, query_molecule)
    query_weight = float(Descriptors.MolWt(query_molecule))
    query_heavy_atoms = int(query_molecule.GetNumHeavyAtoms())
    trained_elements = {int(value) for value in training_atomic_numbers}
    if not trained_elements or any(value < 1 or value > 118 for value in trained_elements):
        raise ValueError("training_atomic_numbers must contain atomic numbers in [1, 118]")

    retained: list[tuple[int, str, Chem.Mol, float, float]] = []
    for source_rank, (identity, score, cosine) in enumerate(
        zip(identities, ranked_scores, embedding_scores), start=1
    ):
        canonical, molecule = _canonical_molecule(identity)
        if exclude_identity and canonical == query_identity:
            continue
        retained.append((source_rank, canonical, molecule, score, cosine))
        if top_k is not None and len(retained) == top_k:
            break

    candidates: list[dict[str, object]] = []
    for rank, (source_rank, canonical, molecule, score, cosine) in enumerate(
        retained, start=1
    ):
        legacy_ged = graph_edit_distance(
            query_identity,
            canonical,
            timeout=ged_timeout,
            version="legacy_ged_v1",
        )
        chemical_ged = graph_edit_distance(
            query_identity,
            canonical,
            timeout=ged_timeout,
            version="chemical_ged_v2",
        )
        chemical_nged = normalized_graph_edit_distance(
            query_identity,
            canonical,
            timeout=ged_timeout,
            version="chemical_ged_v2",
        )
        tanimoto = tanimoto_similarity(query_identity, canonical)
        candidate_atoms = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
        union = query_atoms | candidate_atoms
        unseen_atoms = [
            atom.GetAtomicNum()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() not in trained_elements
        ]
        candidates.append(
            {
                "rank": rank,
                "source_rank": source_rank,
                "smiles": canonical,
                "score": score,
                "cosine_score": cosine,
                "tanimoto": tanimoto,
                "legacy_ged_v1": legacy_ged,
                "chemical_ged_v2": chemical_ged,
                "chemical_nged_v2": chemical_nged,
                "heavy_atom_gap": abs(
                    query_heavy_atoms - int(molecule.GetNumHeavyAtoms())
                ),
                "element_jaccard": (
                    len(query_atoms & candidate_atoms) / len(union) if union else None
                ),
                "molecular_weight_error": abs(
                    query_weight - float(Descriptors.MolWt(molecule))
                ),
                "scaffold_match": (
                    query_scaffold == _scaffold_identity(canonical, molecule)
                ),
                "exact_match": query_identity == canonical,
                "unseen_element_fraction": (
                    len(unseen_atoms) / molecule.GetNumAtoms()
                    if molecule.GetNumAtoms()
                    else None
                ),
            }
        )

    top1 = candidates[0] if candidates else None
    return {
        "ground_truth_smiles": query_identity,
        "identity_excluded": exclude_identity,
        "top1_smiles": "" if top1 is None else top1["smiles"],
        "top1_tanimoto": None if top1 is None else top1["tanimoto"],
        "legacy_ged_v1": None if top1 is None else top1["legacy_ged_v1"],
        "chemical_ged_v2": None if top1 is None else top1["chemical_ged_v2"],
        "chemical_nged_v2": None if top1 is None else top1["chemical_nged_v2"],
        "candidates": candidates,
    }


def _mean_valid(values: Sequence[object]) -> float | None:
    """对非 ``None`` 的有限数取均值；无可计算值时返回 ``None``。"""

    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def aggregate_structure_retrieval(
    query_results: Sequence[dict[str, object]],
) -> dict[str, object]:
    """按 query 等权汇总统一 Top-K 结构质量、召回、相关性、覆盖与延迟。"""

    rows = list(query_results)
    if not rows:
        raise ValueError("query_results must not be empty")
    candidate_rows = [list(row.get("candidates", [])) for row in rows]

    def per_query_mean(field: str, limit: int) -> list[float | None]:
        return [
            _mean_valid([candidate.get(field) for candidate in candidates[:limit]])
            for candidates in candidate_rows
        ]

    def per_query_best(field: str, limit: int, *, minimum: bool = False) -> list[float | None]:
        values: list[float | None] = []
        for candidates in candidate_rows:
            valid = [
                float(candidate[field])
                for candidate in candidates[:limit]
                if candidate.get(field) is not None
            ]
            values.append((min(valid) if minimum else max(valid)) if valid else None)
        return values

    def recall(field: str, limit: int) -> float:
        return float(np.mean([
            any(bool(candidate.get(field)) for candidate in candidates[:limit])
            for candidates in candidate_rows
        ]))

    top1 = [candidates[0] if candidates else {} for candidates in candidate_rows]
    cosine_values: list[float] = []
    tanimoto_values: list[float] = []
    for candidates in candidate_rows:
        for candidate in candidates:
            if candidate.get("tanimoto") is not None:
                cosine_values.append(float(candidate["cosine_score"]))
                tanimoto_values.append(float(candidate["tanimoto"]))
    if (
        len(cosine_values) < 2
        or len(set(cosine_values)) < 2
        or len(set(tanimoto_values)) < 2
    ):
        correlation = None
    else:
        statistic = float(spearmanr(cosine_values, tanimoto_values).statistic)
        correlation = statistic if np.isfinite(statistic) else None

    structure_recall: dict[str, dict[str, float]] = {}
    for threshold in (0.3, 0.5):
        structure_recall[str(threshold)] = {
            str(limit): float(np.mean([
                any(
                    candidate.get("tanimoto") is not None
                    and float(candidate["tanimoto"]) >= threshold
                    for candidate in candidates[:limit]
                )
                for candidates in candidate_rows
            ]))
            for limit in (1, 5, 10)
        }

    unseen = [
        candidate.get("unseen_element_fraction")
        for candidates in candidate_rows
        for candidate in candidates
    ]
    latencies = [row.get("latency_seconds") for row in rows]
    aggregate: dict[str, object] = {
        "query_count": len(rows),
        "top1_tanimoto": _mean_valid([candidate.get("tanimoto") for candidate in top1]),
        "mean_top5_tanimoto": _mean_valid(per_query_mean("tanimoto", 5)),
        "best_at_5_tanimoto": _mean_valid(per_query_best("tanimoto", 5)),
        "best_at_10_tanimoto": _mean_valid(per_query_best("tanimoto", 10)),
        "top1_legacy_ged_v1": _mean_valid([
            candidate.get("legacy_ged_v1") for candidate in top1
        ]),
        "top1_chemical_ged_v2": _mean_valid([
            candidate.get("chemical_ged_v2") for candidate in top1
        ]),
        "top1_chemical_nged_v2": _mean_valid([
            candidate.get("chemical_nged_v2") for candidate in top1
        ]),
        "best_at_5_chemical_ged_v2": _mean_valid(
            per_query_best("chemical_ged_v2", 5, minimum=True)
        ),
        "best_at_10_chemical_ged_v2": _mean_valid(
            per_query_best("chemical_ged_v2", 10, minimum=True)
        ),
        "best_at_5_chemical_nged_v2": _mean_valid(
            per_query_best("chemical_nged_v2", 5, minimum=True)
        ),
        "best_at_10_chemical_nged_v2": _mean_valid(
            per_query_best("chemical_nged_v2", 10, minimum=True)
        ),
        "best_at_5_legacy_ged_v1": _mean_valid(
            per_query_best("legacy_ged_v1", 5, minimum=True)
        ),
        "best_at_10_legacy_ged_v1": _mean_valid(
            per_query_best("legacy_ged_v1", 10, minimum=True)
        ),
        "top1_heavy_atom_gap": _mean_valid([
            candidate.get("heavy_atom_gap") for candidate in top1
        ]),
        "top1_element_jaccard": _mean_valid([
            candidate.get("element_jaccard") for candidate in top1
        ]),
        "top1_molecular_weight_error": _mean_valid([
            candidate.get("molecular_weight_error") for candidate in top1
        ]),
        "embedding_tanimoto_spearman": correlation,
        "candidate_unseen_element_fraction": _mean_valid(unseen),
        "mean_latency_seconds": _mean_valid(latencies),
        "total_latency_seconds": (
            float(sum(float(value) for value in latencies if value is not None))
        ),
        "structure_recall_at_1": {
            threshold: values["1"] for threshold, values in structure_recall.items()
        },
        "structure_recall_at_5": {
            threshold: values["5"] for threshold, values in structure_recall.items()
        },
        "structure_recall_at_10": {
            threshold: values["10"] for threshold, values in structure_recall.items()
        },
    }
    for limit in (1, 5, 10):
        aggregate[f"exact_match_recall_at_{limit}"] = recall("exact_match", limit)
        aggregate[f"scaffold_recall_at_{limit}"] = recall("scaffold_match", limit)
        for threshold in ("0.3", "0.5"):
            aggregate[
                f"structure_recall_at_{limit}_tanimoto_{threshold}"
            ] = structure_recall[threshold][str(limit)]
    return aggregate


__all__ = [
    "DEFAULT_GED_VERSION",
    "GEDVersion",
    "batch_graph_edit_distance",
    "aggregate_structure_retrieval",
    "compute_retrieval_metrics",
    "graph_edit_distance",
    "normalized_graph_edit_distance",
    "pairwise_tanimoto",
    "rank_structure_candidates",
    "smiles_to_nx",
    "tanimoto_similarity",
]

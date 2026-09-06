"""
功能概述：统一运行 KNN、``spectral_only_v1`` 与 ``spectrum_graph_v2`` 的冻结集结构检索评测，
保证三种模式共享既有 cosine/Stage 1 排序接口，真实结构只在 ranking 完成后进入 metrics。
主要职责：``paired_bootstrap`` 生成固定协议置信区间；``candidate_corpus_audit`` 以 canonical identity
审计完整唯一候选库与实际训练元素词表；``evaluate_ranked_queries`` 编排纯 spectrum
查询和逐条/聚合指标；``run_evaluation`` 从显式目录构造 baseline 或 Task 8 Stage1Retrieval；
``compare_evaluation_results`` 先核对内容 hash/结构指标协议，再按 query block 重采样并重算 pooled
Spearman，生成可审计的 v2-baseline 配对统计；``main`` 提供 CLI。
代码库关系：结构质量完全委托 Task 3 ``eval.metrics``，legacy cosine 委托 ``SpectralEncoder.retrieve``，
v1/v2 模型检索委托 Task 8 ``Stage1Retrieval.run``；不维护任何平行 GED、Tanimoto 或排序算法。
输入输出与副作用：查询模型仅接收有限二维 spectra 的逐行一维数组；读取显式 Parquet、manifest、
checkpoint/index，输出严格 JSON。SMILES 只在检索返回后用于 identity 排除和 metrics；不会训练模型、
修改输入数据或访问未指定目录。baseline 对比要求 query mol_id 顺序完全一致，否则 fail fast。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr

from spectraexpert.data.structure_dataset import (
    canonical_smiles_sha256,
    canonicalize_smiles,
    scaffold_split,
)
from spectraexpert.data.structure_features import StructureFeatureConfig
from spectraexpert.eval import metrics as structure_metrics
from spectraexpert.inference import SpectralEncoder
from spectraexpert.pipeline.stage1_retrieval import Stage1Retrieval


@dataclass(frozen=True)
class _DatasetBundle:
    """评测数据及其可比较性 provenance；对外兼容入口仍只返回三元组。"""

    mol_ids: np.ndarray
    spectra: np.ndarray
    smiles: list[str]
    all_mol_ids: np.ndarray
    all_smiles: list[str]
    source_sha256: dict[str, str]
    source_smiles_sha256: str
    split_manifest_sha256: str | None
    manifest: dict[str, object] | None
    training_atomic_numbers: list[int]


def _file_sha256(path: Path) -> str:
    """流式计算显式评测输入文件的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_numbers(smiles: Sequence[str]) -> list[int]:
    """从实际 canonical 训练 identity 推导元素词表。"""

    values: set[int] = set()
    for identity in smiles:
        molecule = Chem.MolFromSmiles(identity)
        if molecule is None:
            raise ValueError(f"invalid canonical SMILES: {identity!r}")
        values.update(atom.GetAtomicNum() for atom in molecule.GetAtoms())
    if not values:
        raise ValueError("training split must contain at least one atom")
    return sorted(values)


def _spectrum_corpus_sha256(
    mol_ids: Sequence[int], canonical_smiles: Sequence[str], spectra: np.ndarray
) -> str:
    """按实际行序绑定 mol_id、canonical identity、float32 shape 与模型所见谱值。"""

    values = np.asarray(spectra, dtype=np.float32)
    identifiers = [int(value) for value in mol_ids]
    identities = [canonicalize_smiles(value) for value in canonical_smiles]
    if values.ndim != 2 or len(values) != len(identifiers) or len(values) != len(identities):
        raise ValueError("spectrum corpus IDs, identities, and values must be aligned")
    digest = hashlib.sha256()
    header = {
        "schema_version": "ordered_spectrum_corpus_v1",
        "dtype": "float32-le",
        "shape": list(values.shape),
        "records": list(zip(identifiers, identities)),
    }
    digest.update(json.dumps(
        header, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    little_endian = np.ascontiguousarray(values.astype("<f4", copy=False))
    digest.update(little_endian.tobytes(order="C"))
    return digest.hexdigest()


def _structure_metric_protocol(ged_timeout: float) -> dict[str, object]:
    """记录会改变结构指标语义的 GED 与 Morgan 完整固定配置。"""

    timeout = float(ged_timeout)
    if not np.isfinite(timeout) or timeout < 0:
        raise ValueError("ged_timeout must be finite and non-negative")
    return {
        "schema_version": "structure_retrieval_metrics_v1",
        "ged_timeout_seconds": timeout,
        "ged_include_h": False,
        "ged_versions": ["legacy_ged_v1", "chemical_ged_v2"],
        "normalized_ged": "chemical_nged_v2",
        "morgan": StructureFeatureConfig().to_metadata(),
    }


def candidate_corpus_audit(
    candidate_smiles: Sequence[str],
    *,
    training_atomic_numbers: Sequence[int],
    training_element_source: str,
) -> dict[str, object]:
    """按 canonical unique corpus 计算 identity multiplicity 与未覆盖元素审计。"""

    canonical = [canonicalize_smiles(value) for value in candidate_smiles]
    if not canonical:
        raise ValueError("candidate corpus must not be empty")
    counts = Counter(canonical)
    unique = sorted(counts)
    trained = sorted({int(value) for value in training_atomic_numbers})
    if not trained or any(value < 1 or value > 118 for value in trained):
        raise ValueError("training atomic numbers must be non-empty values in [1, 118]")
    if not isinstance(training_element_source, str) or not training_element_source:
        raise ValueError("training element source must be a non-empty string")
    unseen_molecules = 0
    unseen_atoms = 0
    total_atoms = 0
    for identity in unique:
        molecule = Chem.MolFromSmiles(identity)
        if molecule is None:
            raise ValueError(f"invalid canonical candidate SMILES: {identity!r}")
        atomic_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
        total_atoms += len(atomic_numbers)
        molecule_unseen = sum(value not in trained for value in atomic_numbers)
        unseen_atoms += molecule_unseen
        unseen_molecules += int(molecule_unseen > 0)
    identity_hash = hashlib.sha256(
        json.dumps(unique, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "candidate_corpus_audit_v1",
        "canonical_smiles": canonical,
        "canonical_unique_smiles": unique,
        "canonical_identity_sha256": identity_hash,
        "record_count": len(canonical),
        "unique_identity_count": len(unique),
        "max_identity_multiplicity": max(counts.values()),
        "training_atomic_numbers": trained,
        "training_element_source": training_element_source,
        "unseen_element_molecule_count": unseen_molecules,
        "unseen_element_molecule_fraction": unseen_molecules / len(unique),
        "unseen_element_atom_count": unseen_atoms,
        "total_heavy_atom_count": total_atoms,
        "unseen_element_atom_fraction": unseen_atoms / total_atoms,
    }


def paired_bootstrap(
    values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """对逐 query 差值 ``values-baseline`` 做 percentile paired bootstrap。"""

    first = np.asarray(values, dtype=np.float64)
    second = np.asarray(baseline_values, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape or not first.size:
        raise ValueError("paired bootstrap inputs must be non-empty equal-length vectors")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("paired bootstrap inputs must contain only finite values")
    if (
        not isinstance(n_resamples, (int, np.integer))
        or isinstance(n_resamples, (bool, np.bool_))
        or n_resamples <= 0
    ):
        raise ValueError("n_resamples must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    differences = first - second
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, len(differences), size=(int(n_resamples), len(differences))
    )
    resampled_means = differences[indices].mean(axis=1)
    lower, upper = np.percentile(resampled_means, [2.5, 97.5])
    return {
        "mean_delta": float(differences.mean()),
        "ci_2_5": float(lower),
        "ci_97_5": float(upper),
        "n_pairs": int(len(differences)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def _validate_query_inputs(query_inputs: Sequence[str]) -> list[str]:
    """拒绝任何真实结构 query 字段，审计配置只允许唯一的 ``spectrum``。"""

    values = list(query_inputs)
    if values != ["spectrum"]:
        raise ValueError("query model inputs may contain only spectrum")
    return values


def _spearman_or_none(first: Sequence[object], second: Sequence[object]) -> float | None:
    """常量或少于两对时返回不可用，避免把 SciPy warning 当作评测结论。"""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if (
        left.shape != right.shape
        or left.ndim != 1
        or len(left) < 2
        or len(np.unique(left)) < 2
        or len(np.unique(right)) < 2
    ):
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def _paired_pooled_spearman_bootstrap(
    current_pairs: Sequence[Mapping[str, object]],
    baseline_pairs: Sequence[Mapping[str, object]],
    query_ids: Sequence[int],
    *,
    current_point: object,
    baseline_point: object,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """按 query pair block 重采样，每次重算 pooled rho 差，与报告点估计保持同一统计量。"""

    identifiers = [int(value) for value in query_ids]
    current_by_query = {identifier: [] for identifier in identifiers}
    baseline_by_query = {identifier: [] for identifier in identifiers}
    for pair in current_pairs:
        identifier = int(pair["query_mol_id"])
        if identifier not in current_by_query:
            raise ValueError("sampled pair references unknown query_mol_id")
        current_by_query[identifier].append(pair)
    for pair in baseline_pairs:
        identifier = int(pair["query_mol_id"])
        if identifier not in baseline_by_query:
            raise ValueError("sampled pair references unknown query_mol_id")
        baseline_by_query[identifier].append(pair)
    if any(not current_by_query[value] or not baseline_by_query[value] for value in identifiers):
        return {
            "available": False, "mean_delta": None, "ci_2_5": None,
            "ci_97_5": None, "n_pairs": 0, "n_total": len(identifiers),
            "n_resamples": n_resamples, "valid_resamples": 0, "seed": seed,
        }

    current_rho = _spearman_or_none(
        [pair["cosine_score"] for pair in current_pairs],
        [pair["tanimoto"] for pair in current_pairs],
    )
    baseline_rho = _spearman_or_none(
        [pair["cosine_score"] for pair in baseline_pairs],
        [pair["tanimoto"] for pair in baseline_pairs],
    )
    try:
        declared_current = float(current_point)
        declared_baseline = float(baseline_point)
    except (TypeError, ValueError):
        declared_current = declared_baseline = float("nan")
    if current_rho is None or baseline_rho is None:
        return {
            "available": False, "mean_delta": None, "ci_2_5": None,
            "ci_97_5": None, "n_pairs": 0, "n_total": len(identifiers),
            "n_resamples": n_resamples, "valid_resamples": 0, "seed": seed,
        }
    if (
        not np.isfinite([declared_current, declared_baseline]).all()
        or not np.isclose(declared_current, current_rho, rtol=0.0, atol=1e-12)
        or not np.isclose(declared_baseline, baseline_rho, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("aggregate pooled sampled-pair Spearman is inconsistent")

    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(identifiers), size=(n_resamples, len(identifiers)))
    differences: list[float] = []
    for sample in samples:
        selected = [identifiers[index] for index in sample]
        current_block = [pair for identifier in selected for pair in current_by_query[identifier]]
        baseline_block = [pair for identifier in selected for pair in baseline_by_query[identifier]]
        current_sample_rho = _spearman_or_none(
            [pair["cosine_score"] for pair in current_block],
            [pair["tanimoto"] for pair in current_block],
        )
        baseline_sample_rho = _spearman_or_none(
            [pair["cosine_score"] for pair in baseline_block],
            [pair["tanimoto"] for pair in baseline_block],
        )
        if current_sample_rho is not None and baseline_sample_rho is not None:
            differences.append(current_sample_rho - baseline_sample_rho)
    if not differences:
        return {
            "available": False, "mean_delta": None, "ci_2_5": None,
            "ci_97_5": None, "n_pairs": 0, "n_total": len(identifiers),
            "n_resamples": n_resamples, "valid_resamples": 0, "seed": seed,
        }
    lower, upper = np.percentile(np.asarray(differences, dtype=np.float64), [2.5, 97.5])
    return {
        "available": True,
        "mean_delta": declared_current - declared_baseline,
        "ci_2_5": float(lower),
        "ci_97_5": float(upper),
        "n_pairs": len(identifiers),
        "n_total": len(identifiers),
        "n_resamples": n_resamples,
        "valid_resamples": len(differences),
        "seed": seed,
        "resampling_unit": "query_pair_block",
        "statistic": "pooled_sampled_pair_spearman",
    }


def _unpack_ranking(ranking: object) -> tuple[list[str], list[float], list[float]]:
    """规范 Task 8 Stage1Result 或轻量 KNN mapping，不改变上游给出的 rank。"""

    if isinstance(ranking, Mapping):
        smiles = list(ranking.get("smiles", []))
        scores = list(ranking.get("scores", []))
        cosine = list(ranking.get("cosine_scores", scores))
    elif hasattr(ranking, "candidates"):
        candidates = list(ranking.candidates)
        smiles = [candidate.smiles for candidate in candidates]
        scores = [candidate.similarity for candidate in candidates]
        cosine = [candidate.embedding_similarity for candidate in candidates]
    else:
        raise TypeError("retriever must return a ranking mapping or Stage1Result")
    return [str(value) for value in smiles], [float(value) for value in scores], [
        float(value) for value in cosine
    ]


def evaluate_ranked_queries(
    *,
    query_mol_ids: np.ndarray,
    query_spectra: np.ndarray,
    query_smiles: Sequence[str],
    retrieve: Callable[[np.ndarray], object],
    top_k: int,
    exclude_identity: bool = False,
    query_inputs: Sequence[str] = ("spectrum",),
    ged_timeout: float = 10.0,
    training_atomic_numbers: Sequence[int] = (1, 6, 7, 8, 9),
    candidate_corpus_smiles: Sequence[str] | None = None,
    score_sampled_candidates: Callable[[np.ndarray, object, Sequence[str]], Sequence[float]] | None = None,
    sampled_pairs_per_query: int = 32,
    sampled_pair_seed: int = 0,
    eligible_candidate_count: int | None = None,
    split_manifest_sha256: str | None = None,
    candidate_corpus_sha256: str | None = None,
    candidate_corpus_kind: str = "unverified",
    candidate_source_smiles_sha256: str | None = None,
    candidate_spectrum_corpus_sha256: str | None = None,
) -> dict[str, object]:
    """先以 spectrum 完成所有单条 ranking，再把真实 SMILES 交给结构 metrics。"""

    inputs = _validate_query_inputs(query_inputs)
    identifiers = np.asarray(query_mol_ids)
    spectra = np.asarray(query_spectra, dtype=np.float32)
    identities = list(query_smiles)
    if identifiers.ndim != 1 or spectra.ndim != 2:
        raise ValueError("query mol_ids/spectra must be one- and two-dimensional")
    if len(identifiers) != len(spectra) or len(identities) != len(spectra) or not len(spectra):
        raise ValueError("query mol_ids, spectra, and SMILES must be non-empty and aligned")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("query mol_ids must be unique")
    if not np.isfinite(spectra).all():
        raise ValueError("query spectra must contain only finite values")
    if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, (bool, np.bool_)) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if (candidate_corpus_smiles is None) != (score_sampled_candidates is None):
        raise ValueError("fixed sampled pairs require candidate corpus and scoring callback together")
    if (
        not isinstance(sampled_pairs_per_query, (int, np.integer))
        or isinstance(sampled_pairs_per_query, (bool, np.bool_))
        or sampled_pairs_per_query <= 0
    ):
        raise ValueError("sampled_pairs_per_query must be a positive integer")
    if not isinstance(sampled_pair_seed, (int, np.integer)) or isinstance(sampled_pair_seed, (bool, np.bool_)):
        raise ValueError("sampled_pair_seed must be an integer")
    canonical_corpus = (
        None
        if candidate_corpus_smiles is None
        else [canonicalize_smiles(value) for value in candidate_corpus_smiles]
    )
    sampled_corpus = None if canonical_corpus is None else sorted(set(canonical_corpus))
    if sampled_corpus == []:
        raise ValueError("candidate corpus for sampled pairs must not be empty")

    rows: list[dict[str, object]] = []
    sampled_pairs: list[dict[str, object]] = []
    for index, (identifier, spectrum) in enumerate(zip(identifiers, spectra)):
        started = time.perf_counter()
        ranking = retrieve(np.ascontiguousarray(spectrum))
        latency = time.perf_counter() - started
        ranked_smiles, ranked_scores, cosine_scores = _unpack_ranking(ranking)
        query_pairs: list[dict[str, object]] = []
        if sampled_corpus is not None and score_sampled_candidates is not None:
            seed_material = f"{int(sampled_pair_seed)}:{int(identifier)}".encode("utf-8")
            query_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            generator = np.random.default_rng(query_seed)
            count = min(int(sampled_pairs_per_query), len(sampled_corpus))
            selected_indices = sorted(
                generator.choice(len(sampled_corpus), size=count, replace=False).tolist()
            )
            selected = [sampled_corpus[value] for value in selected_indices]
            sampled_scores = [
                float(value)
                for value in score_sampled_candidates(
                    np.ascontiguousarray(spectrum), ranking, selected
                )
            ]
            if len(sampled_scores) != len(selected) or not np.isfinite(sampled_scores).all():
                raise ValueError("sampled candidate scorer returned invalid scores")
            identity = identities[index]
            for candidate, score in zip(selected, sampled_scores):
                query_pairs.append({
                    "pair_id": f"{int(identifier)}:{candidate}",
                    "query_mol_id": int(identifier),
                    "candidate_smiles": candidate,
                    "cosine_score": score,
                    "tanimoto": float(structure_metrics.tanimoto_similarity(identity, candidate)),
                })
            sampled_pairs.extend(query_pairs)
        else:
            identity = identities[index]
        row_eligible_count = eligible_candidate_count
        if canonical_corpus is not None:
            row_eligible_count = len(canonical_corpus)
            if exclude_identity:
                row_eligible_count -= canonical_corpus.count(canonicalize_smiles(identity))
        metrics = structure_metrics.rank_structure_candidates(
            str(identity),
            ranked_smiles,
            ranked_scores,
            cosine_scores=cosine_scores,
            exclude_identity=exclude_identity,
            top_k=int(top_k),
            ged_timeout=ged_timeout,
            training_atomic_numbers=training_atomic_numbers,
            eligible_candidate_count=row_eligible_count,
        )
        pair_correlation = _spearman_or_none(
            [pair["cosine_score"] for pair in query_pairs],
            [pair["tanimoto"] for pair in query_pairs],
        )
        rows.append(
            {
                "query_mol_id": int(identifier),
                "ground_truth_smiles": metrics.pop("ground_truth_smiles"),
                "latency_seconds": float(latency),
                "sampled_pair_spearman": pair_correlation,
                **metrics,
            }
        )
    aggregate = structure_metrics.aggregate_structure_retrieval(rows)
    global_correlation = _spearman_or_none(
        [pair["cosine_score"] for pair in sampled_pairs],
        [pair["tanimoto"] for pair in sampled_pairs],
    )
    aggregate["embedding_tanimoto_spearman"] = global_correlation
    query_identity_hash = canonical_smiles_sha256(
        identifiers.astype(np.int64).tolist(),
        [canonicalize_smiles(value) for value in identities],
    )
    query_spectrum_hash = _spectrum_corpus_sha256(
        identifiers.astype(np.int64).tolist(), identities, spectra
    )
    candidate_identity_hash = (
        None
        if sampled_corpus is None
        else hashlib.sha256(
            json.dumps(sampled_corpus, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    if sampled_corpus is not None and candidate_source_smiles_sha256 is None:
        candidate_source_smiles_sha256 = canonical_smiles_sha256(
            list(range(len(sampled_corpus))), sampled_corpus
        )
    return {
        "config": {
            "query_inputs": inputs,
            "top_k": int(top_k),
            "exclude_identity": exclude_identity,
            "ged_versions": ["legacy_ged_v1", "chemical_ged_v2"],
            "normalized_ged": "chemical_nged_v2",
            "query_identity_sha256": query_identity_hash,
            "query_source_sha256": None,
            "query_spectrum_corpus_sha256": query_spectrum_hash,
            "split_manifest_sha256": split_manifest_sha256,
            "candidate_identity_sha256": candidate_identity_hash,
            "candidate_source_smiles_sha256": candidate_source_smiles_sha256,
            "candidate_corpus_kind": candidate_corpus_kind,
            "candidate_corpus_sha256": candidate_corpus_sha256,
            "candidate_spectrum_corpus_sha256": candidate_spectrum_corpus_sha256,
            "candidate_unique_identities": (
                None if sampled_corpus is None else len(sampled_corpus)
            ),
            "training_atomic_numbers": sorted({int(value) for value in training_atomic_numbers}),
            "structure_metric_protocol": _structure_metric_protocol(ged_timeout),
            "sampled_pair_protocol": (
                None
                if sampled_corpus is None
                else {
                    "schema_version": "fixed_sampled_pairs_v1",
                    "seed": int(sampled_pair_seed),
                    "pairs_per_query": int(sampled_pairs_per_query),
                    "candidate_identity_sha256": candidate_identity_hash,
                    "duplicate_identity_embedding": "first_mol_id_sorted_source_record",
                }
            ),
        },
        "queries": rows,
        "sampled_pairs": sampled_pairs,
        "aggregate": aggregate,
    }


def _load_dataset_bundle(data_dir: str | Path, *, frozen_test_split: bool) -> _DatasetBundle:
    """严格验证三件套、canonical identity 与 frozen split 后返回 provenance bundle。"""

    directory = Path(data_dir)
    paths = {
        name: directory / name
        for name in ("spectra.parquet", "smiles.parquet", "wavenumber.parquet")
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"data directory requires dataset files: {missing}")
    spectra = pd.read_parquet(paths["spectra.parquet"])
    smiles = pd.read_parquet(paths["smiles.parquet"])
    axis = pd.read_parquet(paths["wavenumber.parquet"])
    if list(axis.columns) != ["wavenumber"] or axis.empty:
        raise ValueError("wavenumber.parquet must contain one non-empty wavenumber column")
    axis_values = axis["wavenumber"].to_numpy(dtype=np.float32)
    if not np.isfinite(axis_values).all() or (
        len(axis_values) > 1 and not np.all(np.diff(axis_values) > 0)
    ):
        raise ValueError("wavenumber axis must be finite and strictly increasing")
    smiles_column = (
        "smiles_canonical"
        if "smiles_canonical" in smiles
        else "smiles"
        if "smiles" in smiles
        else None
    )
    if "mol_id" not in spectra or "mol_id" not in smiles or smiles_column is None:
        raise ValueError("dataset requires mol_id and canonical SMILES columns")
    if spectra["mol_id"].duplicated().any() or smiles["mol_id"].duplicated().any():
        raise ValueError("dataset mol_id must be unique in both tables")
    spectral_columns = [column for column in spectra if column != "mol_id"]
    if len(spectral_columns) != len(axis_values):
        raise ValueError("wavenumber/intensity length mismatch")
    selected_smiles = smiles[["mol_id", smiles_column]].rename(
        columns={smiles_column: "smiles_canonical"}
    )
    joined = spectra.merge(selected_smiles, on="mol_id", how="outer", validate="one_to_one", indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError("spectra and SMILES mol_id sets must match exactly")
    joined = joined.drop(columns="_merge").sort_values("mol_id", kind="stable")
    joined["smiles_canonical"] = [
        canonicalize_smiles(value, mol_id=int(identifier))
        for identifier, value in zip(joined["mol_id"], joined["smiles_canonical"])
    ]
    all_ids = joined["mol_id"].to_numpy(dtype=np.int64)
    all_smiles = joined["smiles_canonical"].astype(str).tolist()
    source_hashes = {name: _file_sha256(path) for name, path in paths.items()}
    source_smiles_hash = canonical_smiles_sha256(all_ids, all_smiles)
    manifest: dict[str, object] | None = None
    manifest_hash: str | None = None
    training_atomic_numbers = _atomic_numbers(all_smiles)
    if frozen_test_split:
        manifest_path = directory / "split_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("frozen query dataset requires split_manifest.json")
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("query split_manifest.json is invalid") from error
        if not isinstance(loaded, dict):
            raise ValueError("query split manifest must be a mapping")
        if loaded.get("schema_version") != "scaffold_split_v1":
            raise ValueError("query split manifest schema_version mismatch")
        fractions = loaded.get("fractions")
        seed = loaded.get("seed")
        if not isinstance(fractions, list) or len(fractions) != 3:
            raise ValueError("query split manifest fractions are invalid")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("query split manifest seed is invalid")
        manifest = scaffold_split(
            all_smiles,
            fractions=tuple(fractions),
            seed=seed,
            mol_ids=all_ids,
            manifest_path=manifest_path,
        )
        declared_hashes = manifest.get("source_sha256")
        if not isinstance(declared_hashes, dict) or declared_hashes != source_hashes:
            raise ValueError("query split manifest source_sha256 mismatch")
        if manifest.get("source_smiles_sha256") != source_smiles_hash:
            raise ValueError("query split manifest source_smiles_sha256 mismatch")
        test_ids = manifest.get("test")
        train_ids = manifest.get("train")
        if not isinstance(test_ids, list) or not test_ids:
            raise ValueError("query split manifest requires a non-empty test list")
        if not isinstance(train_ids, list) or not train_ids:
            raise ValueError("query split manifest requires a non-empty train list")
        identity_by_id = dict(zip(all_ids.tolist(), all_smiles))
        training_atomic_numbers = _atomic_numbers([identity_by_id[value] for value in train_ids])
        manifest_hash = _file_sha256(manifest_path)
        joined = joined[joined["mol_id"].isin(test_ids)]
        if set(joined["mol_id"].tolist()) != set(test_ids):
            raise ValueError("query split manifest references unknown mol_id")
        order = {identifier: index for index, identifier in enumerate(test_ids)}
        joined = joined.assign(_order=joined["mol_id"].map(order)).sort_values("_order").drop(columns="_order")
    values = joined[spectral_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("dataset spectra must contain only finite values")
    return _DatasetBundle(
        mol_ids=joined["mol_id"].to_numpy(dtype=np.int64),
        spectra=values,
        smiles=joined["smiles_canonical"].astype(str).tolist(),
        all_mol_ids=all_ids,
        all_smiles=all_smiles,
        source_sha256=source_hashes,
        source_smiles_sha256=source_smiles_hash,
        split_manifest_sha256=manifest_hash,
        manifest=manifest,
        training_atomic_numbers=training_atomic_numbers,
    )


def _load_aligned_dataset(
    data_dir: str | Path, *, frozen_test_split: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """兼容入口：返回严格验证 bundle 的 mol_id、spectra 与 canonical SMILES。"""

    bundle = _load_dataset_bundle(data_dir, frozen_test_split=frozen_test_split)
    return bundle.mol_ids, bundle.spectra, bundle.smiles


def _normalise_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """按 Task 8 legacy query 约定标准化 candidate spectrum。"""

    deviation = float(spectrum.std())
    if deviation > 1e-10:
        return ((spectrum - spectrum.mean()) / deviation).astype(np.float32)
    return (spectrum - spectrum.mean()).astype(np.float32)


def _stage1_mapping(stage1: Stage1Retrieval) -> Callable[[np.ndarray], object]:
    """把 Task 8 ``run`` 原样暴露为纯 spectrum callable。"""

    return stage1.run


def _sampled_embedding_scorer(
    candidate_smiles: Sequence[str], candidate_embeddings: np.ndarray
) -> Callable[[np.ndarray, object, Sequence[str]], Sequence[float]]:
    """以 canonical identity 选定固定候选，并复用 legacy cosine API 计算未排序 pair 分数。"""

    identities = [canonicalize_smiles(value) for value in candidate_smiles]
    embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(identities):
        raise ValueError("candidate identities and embeddings must be aligned")
    first_index: dict[str, int] = {}
    for index, identity in enumerate(identities):
        first_index.setdefault(identity, index)

    def score(
        spectrum: np.ndarray, ranking: object, selected: Sequence[str]
    ) -> Sequence[float]:
        del spectrum
        if isinstance(ranking, Mapping):
            query = ranking.get("query_embedding")
        else:
            query = getattr(ranking, "query_embedding", None)
        if query is None:
            raise ValueError("ranking must expose query_embedding for sampled pair audit")
        try:
            selected_indices = [first_index[canonicalize_smiles(value)] for value in selected]
        except KeyError as error:
            raise ValueError("sampled candidate identity is absent from embedding corpus") from error
        order, ranked_scores = SpectralEncoder.retrieve(
            np.asarray(query, dtype=np.float32),
            embeddings[selected_indices],
            topk=len(selected_indices),
        )
        unsorted = np.empty(len(selected_indices), dtype=np.float32)
        unsorted[order] = ranked_scores
        return unsorted.tolist()

    return score


def _validate_checkpoint_provenance(payload: Mapping[str, Any], query: _DatasetBundle) -> None:
    """要求 checkpoint 训练数据 provenance 与冻结 query 三件套/manifest 完全一致。"""

    if payload.get("split_manifest_sha256") != query.split_manifest_sha256:
        raise ValueError("checkpoint split_manifest_sha256 does not match frozen query manifest")
    if payload.get("source_sha256") != query.source_sha256:
        raise ValueError("checkpoint source_sha256 does not match frozen query dataset")
    if payload.get("source_smiles_sha256") != query.source_smiles_sha256:
        raise ValueError("checkpoint source_smiles_sha256 does not match frozen query dataset")


def compare_evaluation_results(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, object]:
    """先验证数据/协议完全可比，再配对统计；任一 GED 缺失不阻断其他字段。"""

    current_config = current.get("config")
    baseline_config = baseline.get("config")
    if not isinstance(current_config, Mapping) or not isinstance(baseline_config, Mapping):
        raise ValueError("paired evaluation requires complete config provenance")
    comparable_fields = (
        "query_inputs",
        "top_k",
        "exclude_identity",
        "ged_versions",
        "normalized_ged",
        "query_identity_sha256",
        "query_source_sha256",
        "query_spectrum_corpus_sha256",
        "split_manifest_sha256",
        "candidate_identity_sha256",
        "candidate_source_smiles_sha256",
        "candidate_corpus_sha256",
        "candidate_unique_identities",
        "training_atomic_numbers",
        "sampled_pair_protocol",
        "structure_metric_protocol",
    )
    for field in comparable_fields:
        if field not in current_config or field not in baseline_config:
            raise ValueError(f"paired evaluation missing comparable config field {field}")
        if current_config[field] != baseline_config[field]:
            raise ValueError(f"paired evaluation {field} mismatch")
    verified_hash_fields = (
        "query_spectrum_corpus_sha256",
        "split_manifest_sha256",
        "candidate_identity_sha256",
        "candidate_source_smiles_sha256",
        "candidate_corpus_sha256",
    )
    for field in verified_hash_fields:
        value = current_config[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"paired evaluation {field} requires a verified non-null SHA-256")
    metric_protocol = current_config["structure_metric_protocol"]
    if not isinstance(metric_protocol, Mapping):
        raise ValueError("paired evaluation structure_metric_protocol must be a mapping")
    try:
        expected_metric_protocol = _structure_metric_protocol(
            float(metric_protocol["ged_timeout_seconds"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("paired evaluation structure_metric_protocol is invalid") from error
    if metric_protocol != expected_metric_protocol:
        raise ValueError("paired evaluation structure_metric_protocol is invalid")
    sampled_protocol = current_config["sampled_pair_protocol"]
    if sampled_protocol is not None:
        if (
            not isinstance(sampled_protocol, Mapping)
            or sampled_protocol.get("candidate_identity_sha256")
            != current_config["candidate_identity_sha256"]
        ):
            raise ValueError("paired evaluation sampled_pair_protocol provenance is invalid")
    current_kind = current_config.get("candidate_corpus_kind")
    baseline_kind = baseline_config.get("candidate_corpus_kind")
    if current_kind not in {"spectrum", "graph_index"}:
        raise ValueError("paired evaluation candidate_corpus_kind is unverified")
    if baseline_kind not in {"spectrum", "graph_index"}:
        raise ValueError("paired evaluation baseline candidate_corpus_kind is unverified")
    if current_kind == baseline_kind == "spectrum":
        current_spectrum_hash = current_config.get("candidate_spectrum_corpus_sha256")
        baseline_spectrum_hash = baseline_config.get("candidate_spectrum_corpus_sha256")
        if current_spectrum_hash != baseline_spectrum_hash:
            raise ValueError("paired evaluation candidate spectrum corpus mismatch")
        if (
            not isinstance(current_spectrum_hash, str)
            or len(current_spectrum_hash) != 64
            or any(character not in "0123456789abcdef" for character in current_spectrum_hash)
        ):
            raise ValueError(
                "paired evaluation candidate_spectrum_corpus_sha256 requires verified non-null SHA-256"
            )
    _validate_query_inputs(current_config["query_inputs"])
    if current_config["exclude_identity"] is not baseline_config["exclude_identity"]:
        raise ValueError("paired evaluation exclude_identity mismatch")

    current_rows = list(current.get("queries", []))
    baseline_rows = list(baseline.get("queries", []))
    current_ids = [row.get("query_mol_id") for row in current_rows]
    baseline_ids = [row.get("query_mol_id") for row in baseline_rows]
    if current_ids != baseline_ids or not current_ids:
        raise ValueError("paired evaluation requires identical non-empty query mol_id order")
    for current_row, baseline_row in zip(current_rows, baseline_rows):
        current_identity = current_row.get("ground_truth_smiles")
        baseline_identity = baseline_row.get("ground_truth_smiles")
        try:
            canonical_current = canonicalize_smiles(current_identity)
            canonical_baseline = canonicalize_smiles(baseline_identity)
        except (TypeError, ValueError) as error:
            raise ValueError("paired evaluation query identity is invalid") from error
        if current_identity != canonical_current or baseline_identity != canonical_baseline:
            raise ValueError("paired evaluation query identity must be canonical")
        if canonical_current != canonical_baseline:
            raise ValueError("paired evaluation query identity mismatch")
    expected_query_hash = canonical_smiles_sha256(
        [int(value) for value in current_ids],
        [str(row["ground_truth_smiles"]) for row in current_rows],
    )
    if current_config["query_identity_sha256"] != expected_query_hash:
        raise ValueError("paired evaluation query identity hash is inconsistent")

    def require_depth(row: Mapping[str, Any], limit: int) -> None:
        requested = row.get("requested_top_k")
        observed = row.get("observed_depth")
        eligible = row.get("eligible_candidate_count")
        if not all(isinstance(value, int) and not isinstance(value, bool)
                   for value in (requested, observed, eligible)):
            raise ValueError("paired Recall@10 requires auditable observed depth")
        if observed != len(row.get("candidates", [])) or eligible < observed:
            raise ValueError("paired Recall@10 observed depth metadata is inconsistent")
        if requested != current_config["top_k"]:
            raise ValueError("paired evaluation top_k does not match query depth metadata")
        if row.get("identity_excluded") is not current_config["exclude_identity"]:
            raise ValueError("paired evaluation exclude_identity does not match query metadata")
        if requested < limit or observed < min(limit, eligible):
            raise ValueError("paired Recall@10 unavailable at actual observed depth")

    for row in current_rows + baseline_rows:
        require_depth(row, 10)

    def paired_values(
        first_rows: list[Mapping[str, Any]],
        second_rows: list[Mapping[str, Any]],
        field: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        pairs: list[tuple[float, float]] = []
        for first_row, second_row in zip(first_rows, second_rows):
            try:
                first_value = float(first_row.get(field))
                second_value = float(second_row.get(field))
            except (TypeError, ValueError):
                continue
            if np.isfinite(first_value) and np.isfinite(second_value):
                pairs.append((first_value, second_value))
        array = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
        return array[:, 0], array[:, 1]

    def optional_bootstrap(
        first: np.ndarray, second: np.ndarray, *, n_total: int
    ) -> dict[str, object]:
        if not len(first):
            return {
                "available": False,
                "mean_delta": None,
                "ci_2_5": None,
                "ci_97_5": None,
                "n_pairs": 0,
                "n_total": n_total,
                "n_resamples": 10_000,
                "seed": 0,
            }
        return {
            "available": True,
            "n_total": n_total,
            **paired_bootstrap(first, second),
        }

    current_tanimoto, baseline_tanimoto = paired_values(
        current_rows, baseline_rows, "top1_tanimoto"
    )
    baseline_nged, current_nged = paired_values(
        baseline_rows, current_rows, "chemical_nged_v2"
    )
    tanimoto = optional_bootstrap(current_tanimoto, baseline_tanimoto, n_total=len(current_rows))
    nged = optional_bootstrap(baseline_nged, current_nged, n_total=len(current_rows))
    current_exact = np.asarray([
        any(bool(candidate.get("exact_match")) for candidate in row.get("candidates", [])[:10])
        for row in current_rows
    ], dtype=np.float64)
    baseline_exact = np.asarray([
        any(bool(candidate.get("exact_match")) for candidate in row.get("candidates", [])[:10])
        for row in baseline_rows
    ], dtype=np.float64)
    exact_recall = optional_bootstrap(current_exact, baseline_exact, n_total=len(current_rows))

    current_pairs = list(current.get("sampled_pairs", []))
    baseline_pairs = list(baseline.get("sampled_pairs", []))
    if bool(current_pairs) is not (current_config["sampled_pair_protocol"] is not None):
        raise ValueError("paired evaluation sampled_pair_protocol does not match sampled pairs")
    if bool(baseline_pairs) is not (baseline_config["sampled_pair_protocol"] is not None):
        raise ValueError("paired evaluation sampled_pair_protocol does not match sampled pairs")
    current_pair_ids = [pair.get("pair_id") for pair in current_pairs]
    baseline_pair_ids = [pair.get("pair_id") for pair in baseline_pairs]
    if current_pair_ids != baseline_pair_ids:
        raise ValueError("paired evaluation sampled pair identity mismatch")
    for first, second in zip(current_pairs, baseline_pairs):
        identity_fields = ("pair_id", "query_mol_id", "candidate_smiles", "tanimoto")
        if any(first.get(field) != second.get(field) for field in identity_fields):
            raise ValueError("paired evaluation sampled pair identity mismatch")
    current_aggregate = current.get("aggregate")
    baseline_aggregate = baseline.get("aggregate")
    correlation = _paired_pooled_spearman_bootstrap(
        current_pairs,
        baseline_pairs,
        [int(value) for value in current_ids],
        current_point=(
            current_aggregate.get("embedding_tanimoto_spearman")
            if isinstance(current_aggregate, Mapping) else None
        ),
        baseline_point=(
            baseline_aggregate.get("embedding_tanimoto_spearman")
            if isinstance(baseline_aggregate, Mapping) else None
        ),
    )
    return {
        "positive_delta_means_improvement": True,
        "top1_tanimoto_v2_minus_baseline": tanimoto,
        "top1_chemical_nged_baseline_minus_v2": nged,
        "exact_match_recall_at_10_v2_minus_baseline": exact_recall,
        "embedding_tanimoto_spearman_v2_minus_baseline": correlation,
        "current_aggregate": current.get("aggregate"),
        "baseline_aggregate": baseline.get("aggregate"),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """写入拒绝 NaN/Infinity 的 UTF-8 审计 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    """校验 mode 所需显式路径，构造已有 retrieval API 并生成单次 JSON 报告。"""

    _validate_query_inputs(args.query_model_inputs)
    query = _load_dataset_bundle(args.query_data_dir, frozen_test_split=True)
    mode = args.mode
    candidate_data_required = mode in {"knn", "spectral_only_v1"}
    if candidate_data_required and args.candidate_data_dir is None:
        raise ValueError(f"{mode} requires --candidate-data-dir")
    if mode != "knn" and args.checkpoint is None:
        raise ValueError(f"{mode} requires --checkpoint")
    if mode == "spectrum_graph_v2" and args.candidate_index is None:
        raise ValueError("spectrum_graph_v2 requires --candidate-index")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if mode == "knn":
            candidate = _load_dataset_bundle(
                args.candidate_data_dir, frozen_test_split=False
            )
            audit = candidate_corpus_audit(
                candidate.smiles,
                training_atomic_numbers=query.training_atomic_numbers,
                training_element_source=f"split_manifest:{query.split_manifest_sha256}:train",
            )
            identity_allowance = int(audit["max_identity_multiplicity"])
            retrieval_top_k = args.top_k + identity_allowance if args.exclude_identity else args.top_k

            def retrieve(spectrum: np.ndarray) -> Mapping[str, Sequence[object]]:
                indices, scores = SpectralEncoder.retrieve(
                    spectrum, candidate.spectra, topk=retrieval_top_k
                )
                return {
                    "smiles": [candidate.smiles[index] for index in indices],
                    "scores": scores.tolist(),
                    "cosine_scores": scores.tolist(),
                    "query_embedding": spectrum,
                }
            sampled_scorer = _sampled_embedding_scorer(candidate.smiles, candidate.spectra)
            checkpoint_schema = None
            candidate_source_smiles_sha256 = candidate.source_smiles_sha256
            candidate_corpus_kind = "spectrum"
            candidate_corpus_hash = candidate.source_smiles_sha256
            candidate_spectrum_hash = _spectrum_corpus_sha256(
                candidate.mol_ids, candidate.smiles, candidate.spectra
            )

        elif mode == "spectral_only_v1":
            candidate = _load_dataset_bundle(
                args.candidate_data_dir, frozen_test_split=False
            )
            audit = candidate_corpus_audit(
                candidate.smiles,
                training_atomic_numbers=query.training_atomic_numbers,
                training_element_source=f"split_manifest:{query.split_manifest_sha256}:train",
            )
            identity_allowance = int(audit["max_identity_multiplicity"])
            retrieval_top_k = args.top_k + identity_allowance if args.exclude_identity else args.top_k
            from spectraexpert.inference import VersionedSpectralEncoder

            encoder = VersionedSpectralEncoder.from_checkpoint(args.checkpoint, device=args.device)
            _validate_checkpoint_provenance(encoder.payload, query)
            embeddings = np.stack([
                encoder.encode_spectrum(
                    _normalise_spectrum(spectrum) if encoder.normalize_spectrum else spectrum
                )
                for spectrum in candidate.spectra
            ]).astype(np.float32)
            temporary = tempfile.TemporaryDirectory(prefix="spectraexpert-eval-")
            cache_path = Path(temporary.name) / "legacy_embeddings.npz"
            np.savez(
                cache_path,
                embeddings=embeddings,
                mol_ids=candidate.mol_ids,
                smiles=np.asarray(candidate.smiles),
            )
            stage1 = Stage1Retrieval({
                "pipeline": {"top_k": retrieval_top_k, "device": args.device},
                "stage1": {
                    "retrieval_mode": "spectral_only_v1",
                    "checkpoint_format": "versioned",
                    "checkpoint": str(args.checkpoint),
                    "embeddings_cache": str(cache_path),
                    "data_dir": str(args.query_data_dir),
                    "normalize_spectrum": encoder.normalize_spectrum,
                },
            })
            retrieve = _stage1_mapping(stage1)
            sampled_scorer = _sampled_embedding_scorer(candidate.smiles, embeddings)
            checkpoint_schema = "spectral_only_v1"
            candidate_source_smiles_sha256 = candidate.source_smiles_sha256
            candidate_corpus_kind = "spectrum"
            candidate_corpus_hash = candidate.source_smiles_sha256
            candidate_spectrum_hash = _spectrum_corpus_sha256(
                candidate.mol_ids, candidate.smiles, candidate.spectra
            )
        else:
            retrieval_top_k = args.top_k + 1 if args.exclude_identity else args.top_k
            stage1 = Stage1Retrieval({
                "pipeline": {"top_k": retrieval_top_k, "device": args.device},
                "stage1": {
                    "retrieval_mode": "spectrum_graph_v2",
                    "checkpoint": str(args.checkpoint),
                    "graph_index": str(args.candidate_index),
                    "data_dir": str(args.query_data_dir),
                    "normalize_spectrum": True,
                    "rerank": {
                        "enabled": False,
                        "top_m": 100,
                        "fingerprint_weight": 0.0,
                        "descriptor_weight": 0.0,
                    },
                },
            })
            _validate_checkpoint_provenance(stage1.checkpoint_payload, query)
            candidate_smiles = [str(value) for value in stage1.candidate_smiles]
            audit = candidate_corpus_audit(
                candidate_smiles,
                training_atomic_numbers=query.training_atomic_numbers,
                training_element_source=f"split_manifest:{query.split_manifest_sha256}:train",
            )
            retrieve = _stage1_mapping(stage1)
            sampled_scorer = _sampled_embedding_scorer(
                candidate_smiles, stage1.candidate_embeddings
            )
            checkpoint_schema = "spectrum_graph_v2"
            candidate_source_smiles_sha256 = stage1.candidate_source_smiles_sha256
            candidate_corpus_kind = "graph_index"
            candidate_corpus_hash = stage1.candidate_source_smiles_sha256
            candidate_spectrum_hash = None

        result = evaluate_ranked_queries(
            query_mol_ids=query.mol_ids,
            query_spectra=query.spectra,
            query_smiles=query.smiles,
            retrieve=retrieve,
            top_k=args.top_k,
            exclude_identity=args.exclude_identity,
            query_inputs=args.query_model_inputs,
            ged_timeout=args.ged_timeout,
            training_atomic_numbers=query.training_atomic_numbers,
            candidate_corpus_smiles=audit["canonical_smiles"],
            score_sampled_candidates=sampled_scorer,
            sampled_pairs_per_query=getattr(args, "sampled_pairs_per_query", 32),
            sampled_pair_seed=getattr(args, "sampled_pair_seed", 0),
            eligible_candidate_count=int(audit["record_count"]),
            split_manifest_sha256=query.split_manifest_sha256,
            candidate_corpus_sha256=candidate_corpus_hash,
            candidate_corpus_kind=candidate_corpus_kind,
            candidate_source_smiles_sha256=candidate_source_smiles_sha256,
            candidate_spectrum_corpus_sha256=candidate_spectrum_hash,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    result["config"].update({
        "mode": mode,
        "query_data_dir": str(args.query_data_dir),
        "candidate_data_dir": (
            None if args.candidate_data_dir is None else str(args.candidate_data_dir)
        ),
        "candidate_index": None if args.candidate_index is None else str(args.candidate_index),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "checkpoint_schema": checkpoint_schema,
        "query_identity_sha256": canonical_smiles_sha256(query.mol_ids, query.smiles),
        "query_source_sha256": query.source_sha256,
        "split_manifest_sha256": query.split_manifest_sha256,
        "candidate_identity_sha256": audit["canonical_identity_sha256"],
        "candidate_source_smiles_sha256": candidate_source_smiles_sha256,
        "candidate_corpus_kind": candidate_corpus_kind,
        "candidate_corpus_sha256": candidate_corpus_hash,
        "candidate_spectrum_corpus_sha256": candidate_spectrum_hash,
        "candidate_unique_identities": audit["unique_identity_count"],
        "training_atomic_numbers": query.training_atomic_numbers,
    })
    result["candidate_corpus_audit"] = {
        key: value for key, value in audit.items()
        if key not in {"canonical_smiles", "canonical_unique_smiles"}
    }
    if args.baseline_result is not None:
        baseline = json.loads(Path(args.baseline_result).read_text(encoding="utf-8"))
        result["paired_bootstrap"] = compare_evaluation_results(result, baseline)
        result["config"]["baseline_result"] = str(args.baseline_result)
    _write_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    """建立明确 mode/path、纯 spectrum 审计与可选 paired baseline 的 CLI。"""

    parser = argparse.ArgumentParser(description="Evaluate structure-aware spectral retrieval")
    parser.add_argument("--mode", choices=("knn", "spectral_only_v1", "spectrum_graph_v2"), required=True)
    parser.add_argument("--query-data-dir", type=Path, required=True)
    parser.add_argument("--candidate-data-dir", type=Path)
    parser.add_argument("--candidate-index", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--exclude-identity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-model-inputs", nargs="+", default=["spectrum"])
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--ged-timeout", type=float, default=10.0)
    parser.add_argument("--sampled-pairs-per-query", type=int, default=32)
    parser.add_argument("--sampled-pair-seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """解析 CLI、运行评测并输出聚合摘要；详情写入 ``--output``。"""

    args = build_parser().parse_args(argv)
    result = run_evaluation(args)
    print(json.dumps(result["aggregate"], ensure_ascii=False, sort_keys=True, allow_nan=False))


rank_structure_candidates = structure_metrics.rank_structure_candidates


if __name__ == "__main__":
    main()


__all__ = [
    "build_parser",
    "candidate_corpus_audit",
    "compare_evaluation_results",
    "evaluate_ranked_queries",
    "paired_bootstrap",
    "rank_structure_candidates",
    "run_evaluation",
]

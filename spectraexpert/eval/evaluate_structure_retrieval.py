"""
功能概述：统一运行 KNN、``spectral_only_v1`` 与 ``spectrum_graph_v2`` 的冻结集结构检索评测，
保证三种模式共享既有 cosine/Stage 1 排序接口，真实结构只在 ranking 完成后进入 metrics。
主要职责：``paired_bootstrap`` 生成固定协议置信区间；``evaluate_ranked_queries`` 编排纯 spectrum
查询和逐条/聚合指标；``run_evaluation`` 从显式目录构造 baseline 或 Task 8 Stage1Retrieval；
``compare_evaluation_results`` 对同序 query 生成可审计的 v2-baseline 配对统计；``main`` 提供 CLI。
代码库关系：结构质量完全委托 Task 3 ``eval.metrics``，legacy cosine 委托 ``SpectralEncoder.retrieve``，
v1/v2 模型检索委托 Task 8 ``Stage1Retrieval.run``；不维护任何平行 GED、Tanimoto 或排序算法。
输入输出与副作用：查询模型仅接收有限二维 spectra 的逐行一维数组；读取显式 Parquet、manifest、
checkpoint/index，输出严格 JSON。SMILES 只在检索返回后用于 identity 排除和 metrics；不会训练模型、
修改输入数据或访问未指定目录。baseline 对比要求 query mol_id 顺序完全一致，否则 fail fast。
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spectraexpert.eval import metrics as structure_metrics
from spectraexpert.inference import SpectralEncoder
from spectraexpert.pipeline.stage1_retrieval import Stage1Retrieval


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

    rows: list[dict[str, object]] = []
    for identifier, spectrum, identity in zip(identifiers, spectra, identities):
        started = time.perf_counter()
        ranking = retrieve(np.ascontiguousarray(spectrum))
        latency = time.perf_counter() - started
        ranked_smiles, ranked_scores, cosine_scores = _unpack_ranking(ranking)
        metrics = structure_metrics.rank_structure_candidates(
            str(identity),
            ranked_smiles,
            ranked_scores,
            cosine_scores=cosine_scores,
            exclude_identity=exclude_identity,
            top_k=int(top_k),
            ged_timeout=ged_timeout,
            training_atomic_numbers=training_atomic_numbers,
        )
        rows.append(
            {
                "query_mol_id": int(identifier),
                "ground_truth_smiles": metrics.pop("ground_truth_smiles"),
                "latency_seconds": float(latency),
                **metrics,
            }
        )
    return {
        "config": {
            "query_inputs": inputs,
            "top_k": int(top_k),
            "exclude_identity": exclude_identity,
            "ged_versions": ["legacy_ged_v1", "chemical_ged_v2"],
            "normalized_ged": "chemical_nged_v2",
        },
        "queries": rows,
        "aggregate": structure_metrics.aggregate_structure_retrieval(rows),
    }


def _load_aligned_dataset(
    data_dir: str | Path, *, frozen_test_split: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """从显式目录按 mol_id 一对一 join；query 可受同目录 frozen manifest 限定。"""

    directory = Path(data_dir)
    spectra_path = directory / "spectra.parquet"
    smiles_path = directory / "smiles.parquet"
    if not spectra_path.is_file() or not smiles_path.is_file():
        raise FileNotFoundError("data directory requires spectra.parquet and smiles.parquet")
    spectra = pd.read_parquet(spectra_path)
    smiles = pd.read_parquet(smiles_path)
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
    if not spectral_columns:
        raise ValueError("spectra.parquet contains no spectrum values")
    selected_smiles = smiles[["mol_id", smiles_column]].rename(
        columns={smiles_column: "smiles_canonical"}
    )
    joined = spectra.merge(selected_smiles, on="mol_id", how="outer", validate="one_to_one", indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError("spectra and SMILES mol_id sets must match exactly")
    joined = joined.drop(columns="_merge").sort_values("mol_id", kind="stable")
    if frozen_test_split:
        manifest_path = directory / "split_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("frozen query dataset requires split_manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("query split_manifest.json is invalid") from error
        test_ids = manifest.get("test")
        if not isinstance(test_ids, list) or not test_ids:
            raise ValueError("query split manifest requires a non-empty test list")
        if len(test_ids) != len(set(test_ids)):
            raise ValueError("query split manifest contains duplicate test mol_id")
        joined = joined[joined["mol_id"].isin(test_ids)]
        if set(joined["mol_id"].tolist()) != set(test_ids):
            raise ValueError("query split manifest references unknown mol_id")
        order = {identifier: index for index, identifier in enumerate(test_ids)}
        joined = joined.assign(_order=joined["mol_id"].map(order)).sort_values("_order").drop(columns="_order")
    values = joined[spectral_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("dataset spectra must contain only finite values")
    return (
        joined["mol_id"].to_numpy(dtype=np.int64),
        values,
        joined["smiles_canonical"].astype(str).tolist(),
    )


def _normalise_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """按 Task 8 legacy query 约定标准化 candidate spectrum。"""

    deviation = float(spectrum.std())
    if deviation > 1e-10:
        return ((spectrum - spectrum.mean()) / deviation).astype(np.float32)
    return (spectrum - spectrum.mean()).astype(np.float32)


def _stage1_mapping(stage1: Stage1Retrieval) -> Callable[[np.ndarray], object]:
    """把 Task 8 ``run`` 原样暴露为纯 spectrum callable。"""

    return stage1.run


def compare_evaluation_results(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, object]:
    """按 mol_id 严格配对 Top-1 Tanimoto/nGED，方向统一为正值代表 v2 改善。"""

    current_rows = list(current.get("queries", []))
    baseline_rows = list(baseline.get("queries", []))
    current_ids = [row.get("query_mol_id") for row in current_rows]
    baseline_ids = [row.get("query_mol_id") for row in baseline_rows]
    if current_ids != baseline_ids or not current_ids:
        raise ValueError("paired evaluation requires identical non-empty query mol_id order")

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
        if not pairs:
            raise ValueError(f"paired evaluation field {field} has no jointly computable pairs")
        array = np.asarray(pairs, dtype=np.float64)
        return array[:, 0], array[:, 1]

    current_tanimoto, baseline_tanimoto = paired_values(
        current_rows, baseline_rows, "top1_tanimoto"
    )
    baseline_nged, current_nged = paired_values(
        baseline_rows, current_rows, "chemical_nged_v2"
    )
    tanimoto = paired_bootstrap(current_tanimoto, baseline_tanimoto)
    nged = paired_bootstrap(baseline_nged, current_nged)
    current_exact = np.asarray([
        any(bool(candidate.get("exact_match")) for candidate in row.get("candidates", [])[:10])
        for row in current_rows
    ], dtype=np.float64)
    baseline_exact = np.asarray([
        any(bool(candidate.get("exact_match")) for candidate in row.get("candidates", [])[:10])
        for row in baseline_rows
    ], dtype=np.float64)
    exact_recall = paired_bootstrap(current_exact, baseline_exact)
    return {
        "positive_delta_means_improvement": True,
        "top1_tanimoto_v2_minus_baseline": tanimoto,
        "top1_chemical_nged_baseline_minus_v2": nged,
        "exact_match_recall_at_10_v2_minus_baseline": exact_recall,
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
    query_ids, query_spectra, query_smiles = _load_aligned_dataset(
        args.query_data_dir, frozen_test_split=True
    )
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
            _, candidate_spectra, candidate_smiles = _load_aligned_dataset(
                args.candidate_data_dir, frozen_test_split=False
            )
            identity_allowance = max(Counter(candidate_smiles).values())
            retrieval_top_k = args.top_k + identity_allowance if args.exclude_identity else args.top_k

            def retrieve(spectrum: np.ndarray) -> Mapping[str, Sequence[object]]:
                indices, scores = SpectralEncoder.retrieve(
                    spectrum, candidate_spectra, topk=retrieval_top_k
                )
                return {
                    "smiles": [candidate_smiles[index] for index in indices],
                    "scores": scores.tolist(),
                    "cosine_scores": scores.tolist(),
                }

        elif mode == "spectral_only_v1":
            candidate_ids, candidate_spectra, candidate_smiles = _load_aligned_dataset(
                args.candidate_data_dir, frozen_test_split=False
            )
            identity_allowance = max(Counter(candidate_smiles).values())
            retrieval_top_k = args.top_k + identity_allowance if args.exclude_identity else args.top_k
            encoder = SpectralEncoder(str(args.checkpoint), device=args.device)
            embeddings = np.stack([
                encoder.encode_spectrum(_normalise_spectrum(spectrum))
                for spectrum in candidate_spectra
            ]).astype(np.float32)
            temporary = tempfile.TemporaryDirectory(prefix="spectraexpert-eval-")
            cache_path = Path(temporary.name) / "legacy_embeddings.npz"
            np.savez(
                cache_path,
                embeddings=embeddings,
                mol_ids=candidate_ids,
                smiles=np.asarray(candidate_smiles),
            )
            stage1 = Stage1Retrieval({
                "pipeline": {"top_k": retrieval_top_k, "device": args.device},
                "stage1": {
                    "retrieval_mode": "spectral_only_v1",
                    "checkpoint": str(args.checkpoint),
                    "embeddings_cache": str(cache_path),
                    "data_dir": str(args.query_data_dir),
                    "normalize_spectrum": True,
                },
            })
            retrieve = _stage1_mapping(stage1)
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
            retrieve = _stage1_mapping(stage1)

        result = evaluate_ranked_queries(
            query_mol_ids=query_ids,
            query_spectra=query_spectra,
            query_smiles=query_smiles,
            retrieve=retrieve,
            top_k=args.top_k,
            exclude_identity=args.exclude_identity,
            query_inputs=args.query_model_inputs,
            ged_timeout=args.ged_timeout,
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
    })
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
    "compare_evaluation_results",
    "evaluate_ranked_queries",
    "paired_bootstrap",
    "rank_structure_candidates",
    "run_evaluation",
]

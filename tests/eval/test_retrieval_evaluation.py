"""
功能概述：验证统一结构检索评测的逐查询指标、聚合统计、paired bootstrap 与防泄漏边界。
主要职责：锁定 identity 排除发生在排序后、候选审计字段完整、Top-K/Recall/Spearman 聚合语义、
bootstrap 默认可复现，以及查询检索器永远只收到 spectrum 的契约。
代码库关系：直接消费 ``spectraexpert.eval.metrics`` 的结构指标入口与
``spectraexpert.eval.evaluate_structure_retrieval`` 的评测编排；Task 3 已单独覆盖 GED 搜索本身，
本文件只用确定性替身隔离其进程开销，不复制结构算法。
输入输出与副作用：仅构造小型 SMILES、NumPy 数组和内存检索器；不读取真实数据、不写评测文件、
不启动训练或服务器任务。若真实 SMILES 提前进入 query model、统计键缺失或默认随机性漂移则失败。
"""

from __future__ import annotations

import argparse
import numpy as np
import pytest

import spectraexpert.eval.metrics as structure_metrics
from spectraexpert.eval.evaluate_structure_retrieval import (
    compare_evaluation_results,
    evaluate_ranked_queries,
    paired_bootstrap,
    run_evaluation,
)


def _fast_structure_metrics(monkeypatch):
    tanimoto = {
        ("CCO", "CCO"): 1.0,
        ("CCO", "CCCO"): 0.6,
        ("CCO", "c1ccccc1"): 0.1,
    }
    monkeypatch.setattr(
        structure_metrics,
        "tanimoto_similarity",
        lambda left, right: tanimoto.get((left, right), 0.0),
    )
    monkeypatch.setattr(
        structure_metrics,
        "graph_edit_distance",
        lambda left, right, **kwargs: (
            0.0 if left == right else 2.0 if kwargs["version"] == "chemical_ged_v2" else 3.0
        ),
    )
    monkeypatch.setattr(
        structure_metrics,
        "normalized_graph_edit_distance",
        lambda left, right, **kwargs: 0.0 if left == right else 0.25,
    )


def test_retrieval_metrics_report_identity_excluded_structure_quality(monkeypatch):
    _fast_structure_metrics(monkeypatch)

    metrics = structure_metrics.rank_structure_candidates(
        query_smiles="CCO",
        candidate_smiles=["CCO", "CCCO", "c1ccccc1"],
        scores=[1.0, 0.8, 0.2],
        exclude_identity=True,
    )

    assert metrics["top1_smiles"] == "CCCO"
    assert metrics["chemical_ged_v2"] == 2.0
    assert metrics["legacy_ged_v1"] == 3.0
    assert metrics["chemical_nged_v2"] == 0.25
    candidate = metrics["candidates"][0]
    assert candidate["rank"] == 1
    assert candidate["source_rank"] == 2
    assert candidate["smiles"] == "CCCO"
    assert candidate["cosine_score"] == pytest.approx(0.8)
    assert candidate["tanimoto"] == pytest.approx(0.6)
    assert candidate["legacy_ged_v1"] == 3.0
    assert candidate["chemical_ged_v2"] == 2.0
    assert candidate["chemical_nged_v2"] == 0.25
    assert candidate["heavy_atom_gap"] == 1
    assert candidate["element_jaccard"] == 1.0
    assert candidate["molecular_weight_error"] == pytest.approx(14.027, abs=0.01)
    assert candidate["scaffold_match"] is False


def test_aggregate_reports_full_topk_metrics_and_latency(monkeypatch):
    _fast_structure_metrics(monkeypatch)
    first = structure_metrics.rank_structure_candidates(
        "CCO", ["CCO", "CCCO", "c1ccccc1"], [0.9, 0.8, 0.2],
        exclude_identity=False,
    )
    second = structure_metrics.rank_structure_candidates(
        "CCO", ["CCCO", "CCO", "c1ccccc1"], [0.9, 0.8, 0.2],
        exclude_identity=False,
    )
    first["latency_seconds"] = 0.1
    second["latency_seconds"] = 0.3

    aggregate = structure_metrics.aggregate_structure_retrieval([first, second])

    assert aggregate["top1_tanimoto"] == pytest.approx(0.8)
    assert aggregate["mean_top5_tanimoto"] == pytest.approx((1.7 / 3 + 1.7 / 3) / 2)
    assert aggregate["best_at_5_tanimoto"] == 1.0
    assert aggregate["best_at_10_tanimoto"] == 1.0
    assert aggregate["exact_match_recall_at_1"] == 0.5
    assert aggregate["exact_match_recall_at_5"] == 1.0
    assert aggregate["scaffold_recall_at_1"] == 0.5
    assert aggregate["structure_recall_at_1"]["0.5"] == 1.0
    assert aggregate["embedding_tanimoto_spearman"] is not None
    assert aggregate["mean_latency_seconds"] == pytest.approx(0.2)


def test_paired_bootstrap_is_reproducible():
    result1 = paired_bootstrap(
        np.array([0.3, 0.4, 0.5]),
        np.array([0.1, 0.2, 0.2]),
        seed=0,
        n_resamples=2000,
    )
    result2 = paired_bootstrap(
        np.array([0.3, 0.4, 0.5]),
        np.array([0.1, 0.2, 0.2]),
        seed=0,
        n_resamples=2000,
    )

    assert result1 == result2
    assert result1["mean_delta"] > 0
    assert result1["n_resamples"] == 2000
    assert result1["seed"] == 0
    assert set(result1) == {"mean_delta", "ci_2_5", "ci_97_5", "n_pairs", "n_resamples", "seed"}


def test_default_paired_bootstrap_uses_fixed_protocol():
    result = paired_bootstrap(np.array([0.4, 0.5]), np.array([0.2, 0.3]))

    assert result["n_resamples"] == 10_000
    assert result["seed"] == 0


def test_query_model_receives_only_spectrum_before_ground_truth_metrics(monkeypatch):
    _fast_structure_metrics(monkeypatch)
    calls = []

    def retrieve(spectrum):
        calls.append(spectrum.copy())
        return {
            "smiles": ["CCCO", "CCO"],
            "scores": [0.8, 0.7],
            "cosine_scores": [0.8, 0.7],
        }

    result = evaluate_ranked_queries(
        query_mol_ids=np.array([7]),
        query_spectra=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        query_smiles=["CCO"],
        retrieve=retrieve,
        top_k=2,
        exclude_identity=False,
    )

    assert len(calls) == 1
    assert calls[0].tolist() == [1.0, 2.0, 3.0]
    assert result["config"]["query_inputs"] == ["spectrum"]
    assert result["queries"][0]["ground_truth_smiles"] == "CCO"
    assert result["queries"][0]["top1_smiles"] == "CCCO"


@pytest.mark.parametrize(
    "bad_inputs",
    [["spectrum", "smiles"], ["descriptor"], ["molecular_weight"]],
)
def test_query_structure_fields_are_rejected_as_model_inputs(bad_inputs):
    with pytest.raises(ValueError, match="only.*spectrum"):
        evaluate_ranked_queries(
            query_mol_ids=np.array([7]),
            query_spectra=np.array([[1.0, 2.0]], dtype=np.float32),
            query_smiles=["CCO"],
            retrieve=lambda spectrum: {"smiles": ["CC"], "scores": [0.5]},
            top_k=1,
            query_inputs=bad_inputs,
        )


def test_cli_rejects_query_structure_inputs_before_reading_any_dataset(tmp_path):
    args = argparse.Namespace(
        mode="knn",
        query_data_dir=tmp_path / "missing-query",
        candidate_data_dir=tmp_path / "missing-candidates",
        candidate_index=None,
        checkpoint=None,
        top_k=10,
        exclude_identity=False,
        output=tmp_path / "must-not-exist.json",
        query_model_inputs=["spectrum", "smiles"],
        baseline_result=None,
        ged_timeout=10.0,
        device=None,
    )

    with pytest.raises(ValueError, match="only.*spectrum"):
        run_evaluation(args)

    assert not args.output.exists()


def test_paired_report_skips_jointly_unavailable_ged_but_keeps_tanimoto():
    current = {
        "queries": [
            {"query_mol_id": 1, "top1_tanimoto": 0.6, "chemical_nged_v2": 0.2,
             "candidates": [{"exact_match": True}]},
            {"query_mol_id": 2, "top1_tanimoto": 0.5, "chemical_nged_v2": None,
             "candidates": [{"exact_match": True}]},
        ]
    }
    baseline = {
        "queries": [
            {"query_mol_id": 1, "top1_tanimoto": 0.4, "chemical_nged_v2": 0.3,
             "candidates": [{"exact_match": True}]},
            {"query_mol_id": 2, "top1_tanimoto": 0.3, "chemical_nged_v2": 0.4,
             "candidates": [{"exact_match": False}]},
        ]
    }

    report = compare_evaluation_results(current, baseline)

    assert report["top1_tanimoto_v2_minus_baseline"]["n_pairs"] == 2
    assert report["top1_chemical_nged_baseline_minus_v2"]["n_pairs"] == 1
    assert report["top1_chemical_nged_baseline_minus_v2"]["mean_delta"] == pytest.approx(0.1)
    assert report["exact_match_recall_at_10_v2_minus_baseline"]["mean_delta"] == 0.5

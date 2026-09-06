"""
功能概述：复现 Task 9 独立审查发现的 checkpoint、冻结协议、公平相关性与缺失统计反例。
主要职责：验证 Task 7 spectral-only checkpoint 可端到端评测；manifest hash/partition 严格校验；
paired report 对 query identity、候选全集和协议 fail closed；固定 sampled pairs、canonical alias 预取、
GED coverage/深度可用性、完整候选库未知元素审计，以及同一 chemical GED 只搜索一次。
代码库关系：使用真实 ``training.checkpoints.save_checkpoint``、``SpectrumTransformer``、小型 Parquet
与 Task 6 split helper；评测行为消费 ``evaluate_structure_retrieval`` 与 ``eval.metrics``，不建立测试专用
生产入口。Task 3 的 GED 搜索本身由原测试覆盖，本文件只对调用次数和缺失值聚合施加确定性替身。
输入输出与副作用：仅在 pytest 临时目录写小型 checkpoint、Parquet、manifest 和 JSON；不访问真实数据、
GPU 或服务器。若模型看见 query 结构、paired 配置不可比或不完整统计被伪装为完整结论则失败。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import spectraexpert.eval.metrics as structure_metrics
from spectraexpert.data.structure_dataset import canonical_smiles_sha256, scaffold_split
from spectraexpert.data.structure_features import DescriptorNormalizer
from spectraexpert.eval.evaluate_structure_retrieval import (
    _load_aligned_dataset,
    build_parser,
    candidate_corpus_audit,
    compare_evaluation_results,
    evaluate_ranked_queries,
    run_evaluation,
)
from spectraexpert.models.spectrum_encoder import SpectrumTransformer
from spectraexpert.training.checkpoints import save_checkpoint


SMILES = ["CC", "CCC", "CCCC", "CCO", "CCN", "CO", "CN", "C=C", "C#N", "c1ccccc1"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_dataset(root: Path, smiles: list[str] = SMILES) -> dict[str, object]:
    root.mkdir()
    ids = list(range(len(smiles)))
    spectra = np.arange(len(smiles) * 17, dtype=np.float32).reshape(len(smiles), 17)
    pd.DataFrame(
        {"mol_id": ids} | {f"w_{index}": spectra[:, index] for index in range(17)}
    ).to_parquet(root / "spectra.parquet", index=False)
    pd.DataFrame({"mol_id": ids, "smiles_canonical": smiles}).to_parquet(
        root / "smiles.parquet", index=False
    )
    pd.DataFrame({"wavenumber": np.linspace(500, 4000, 17, dtype=np.float32)}).to_parquet(
        root / "wavenumber.parquet", index=False
    )
    manifest_path = root / "split_manifest.json"
    manifest = scaffold_split(smiles, mol_ids=ids, seed=0, manifest_path=manifest_path)
    source_hashes = {
        name: _sha256(root / name)
        for name in ("spectra.parquet", "smiles.parquet", "wavenumber.parquet")
    }
    manifest["source_sha256"] = source_hashes
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return manifest


def _save_spectral_checkpoint(root: Path, manifest: dict[str, object]) -> Path:
    model = SpectrumTransformer(
        signal_length=17,
        patch_size=4,
        stride=2,
        embed_dim=8,
        depth=1,
        num_heads=2,
        projection_dim=4,
    )
    path = root / "spectral_only.pt"
    save_checkpoint(
        path,
        model,
        descriptor_stats=DescriptorNormalizer(torch.zeros(15), torch.ones(15)),
        training_mode="spectral_only_v1",
        split_manifest_sha256=_sha256(root / "split_manifest.json"),
        source_sha256=manifest["source_sha256"],
        source_smiles_sha256=manifest["source_smiles_sha256"],
        loss_weights={
            "lambda_aug": 1.0,
            "lambda_struct": 0.0,
            "lambda_aux": 0.0,
            "lambda_fingerprint": 0.0,
        },
        train_config={"world_size": 1, "normalize": True},
    )
    return path


def _minimal_query(
    mol_id: int,
    identity: str,
    *,
    exact: bool = False,
    nged: float | None = 0.2,
) -> dict[str, object]:
    return {
        "query_mol_id": mol_id,
        "ground_truth_smiles": identity,
        "identity_excluded": True,
        "top1_tanimoto": 0.5,
        "chemical_nged_v2": nged,
        "requested_top_k": 10,
        "observed_depth": 10,
        "eligible_candidate_count": 20,
        "candidates": [{"exact_match": exact}] * 10,
        "sampled_pair_spearman": 0.4,
    }


def _comparable_report(identity: str = "CCO") -> dict[str, object]:
    report = {
        "config": {
            "query_inputs": ["spectrum"],
            "top_k": 10,
            "exclude_identity": True,
            "ged_versions": ["legacy_ged_v1", "chemical_ged_v2"],
            "normalized_ged": "chemical_nged_v2",
            "query_identity_sha256": "query-hash",
            "query_source_sha256": {"spectra.parquet": "s", "smiles.parquet": "m", "wavenumber.parquet": "w"},
            "split_manifest_sha256": "split-hash",
            "candidate_identity_sha256": "candidate-hash",
            "candidate_source_smiles_sha256": "candidate-source-hash",
            "candidate_unique_identities": 20,
            "training_atomic_numbers": [6, 8],
            "sampled_pair_protocol": {
                "schema_version": "fixed_sampled_pairs_v1", "seed": 0,
                "pairs_per_query": 4,
                "duplicate_identity_embedding": "first_mol_id_sorted_source_record",
            },
        },
        "queries": [_minimal_query(1, identity)],
        "sampled_pairs": [
            {"pair_id": "1:CC", "query_mol_id": 1, "candidate_smiles": "CC", "cosine_score": 0.2, "tanimoto": 0.3},
            {"pair_id": "1:CO", "query_mol_id": 1, "candidate_smiles": "CO", "cosine_score": 0.4, "tanimoto": 0.5},
        ],
        "aggregate": {"embedding_tanimoto_spearman": 1.0},
    }
    report["config"]["query_identity_sha256"] = canonical_smiles_sha256([1], [identity])
    return report


def test_versioned_spectral_checkpoint_runs_unified_evaluation(tmp_path):
    data_dir = tmp_path / "data"
    manifest = _write_dataset(data_dir)
    checkpoint = _save_spectral_checkpoint(data_dir, manifest)
    output = tmp_path / "result.json"
    args = build_parser().parse_args([
        "--mode", "spectral_only_v1",
        "--query-data-dir", str(data_dir),
        "--candidate-data-dir", str(data_dir),
        "--checkpoint", str(checkpoint),
        "--top-k", "1",
        "--ged-timeout", "0",
        "--sampled-pairs-per-query", "2",
        "--output", str(output),
    ])

    result = run_evaluation(args)

    assert output.is_file()
    assert result["config"]["checkpoint_schema"] == "spectral_only_v1"
    assert result["config"]["query_inputs"] == ["spectrum"]
    assert len(result["queries"]) == len(manifest["test"])


@pytest.mark.parametrize("tamper", ["source_hash", "overlap", "missing_coverage"])
def test_frozen_manifest_rejects_hash_or_partition_tampering(tmp_path, tamper):
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)
    path = data_dir / "split_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "source_hash":
        manifest["source_sha256"]["spectra.parquet"] = "wrong"
    elif tamper == "overlap":
        manifest["test"].append(manifest["train"][0])
    else:
        manifest["train"].pop()
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source_sha256|partition|split manifest"):
        _load_aligned_dataset(data_dir, frozen_test_split=True)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda report: report["queries"][0].update(ground_truth_smiles="CCCC"), "identity"),
        (lambda report: report["config"].update(candidate_identity_sha256="other"), "candidate"),
        (lambda report: report["config"].update(exclude_identity=False), "exclude_identity"),
        (lambda report: report["config"].update(top_k=5), "top_k"),
        (lambda report: report["config"].update(split_manifest_sha256="other"), "split_manifest"),
        (lambda report: report["config"].update(training_atomic_numbers=[6]), "training_atomic_numbers"),
    ],
)
def test_paired_comparison_fails_closed_for_incomparable_protocol(mutator, message):
    current = _comparable_report()
    baseline = _comparable_report()
    mutator(baseline)

    with pytest.raises(ValueError, match=message):
        compare_evaluation_results(current, baseline)


def test_sampled_pair_correlation_uses_same_pairs_when_topk_rankings_differ(monkeypatch):
    monkeypatch.setattr(structure_metrics, "graph_edit_distance", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(structure_metrics, "tanimoto_similarity", lambda left, right: {
        "CC": 0.2, "CO": 0.5, "CN": 0.8,
    }[right])

    def run(order, score_sign):
        def retrieve(spectrum):
            return {"smiles": order, "scores": [0.9, 0.8, 0.7], "query_embedding": spectrum}

        def score_pairs(spectrum, ranking, candidates):
            base = {"CC": 0.1, "CO": 0.4, "CN": 0.7}
            return [score_sign * base[candidate] for candidate in candidates]

        return evaluate_ranked_queries(
            query_mol_ids=np.array([7, 8]),
            query_spectra=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            query_smiles=["CCO", "CCO"],
            retrieve=retrieve,
            top_k=10,
            candidate_corpus_smiles=["CN", "CC", "CO"],
            score_sampled_candidates=score_pairs,
            sampled_pairs_per_query=3,
            sampled_pair_seed=0,
            ged_timeout=0,
        )

    first = run(["CC", "CO", "CN"], 1.0)
    second = run(["CN", "CO", "CC"], -1.0)

    assert [pair["pair_id"] for pair in first["sampled_pairs"]] == [
        pair["pair_id"] for pair in second["sampled_pairs"]
    ]
    assert first["config"]["sampled_pair_protocol"]["schema_version"] == "fixed_sampled_pairs_v1"
    assert first["aggregate"]["embedding_tanimoto_spearman"] == pytest.approx(1.0)
    assert second["aggregate"]["embedding_tanimoto_spearman"] == pytest.approx(-1.0)
    comparison = compare_evaluation_results(first, second)
    assert comparison["embedding_tanimoto_spearman_v2_minus_baseline"]["mean_delta"] == pytest.approx(2.0)


def test_candidate_aliases_are_counted_by_canonical_identity():
    audit = candidate_corpus_audit(
        ["CCO", "OCC", "[H]OC([H])([H])C([H])([H])[H]", "CCCO"],
        training_atomic_numbers=[6, 8],
        training_element_source="manifest:train",
    )

    assert audit["max_identity_multiplicity"] == 3
    assert audit["unique_identity_count"] == 2
    assert audit["canonical_smiles"] == ["CCO", "CCO", "CCO", "CCCO"]


def test_ged_aggregates_report_coverage_and_do_not_publish_partial_best():
    row = {
        "requested_top_k": 5,
        "observed_depth": 2,
        "eligible_candidate_count": 2,
        "latency_seconds": 0.1,
        "candidates": [
            {"tanimoto": 0.7, "legacy_ged_v1": 9.0, "chemical_ged_v2": 8.0,
             "chemical_nged_v2": 0.8, "exact_match": False, "scaffold_match": False,
             "cosine_score": 0.9, "heavy_atom_gap": 1, "element_jaccard": 1,
             "molecular_weight_error": 1, "unseen_element_fraction": 0},
            {"tanimoto": 0.6, "legacy_ged_v1": None, "chemical_ged_v2": None,
             "chemical_nged_v2": None, "exact_match": False, "scaffold_match": False,
             "cosine_score": 0.8, "heavy_atom_gap": 1, "element_jaccard": 1,
             "molecular_weight_error": 1, "unseen_element_fraction": 0},
        ],
    }

    aggregate = structure_metrics.aggregate_structure_retrieval([row])

    assert aggregate["best_at_5_chemical_ged_v2"] is None
    assert aggregate["ged_coverage"]["best_at_5_chemical_ged_v2"] == {
        "completed": 1, "total": 2, "coverage": 0.5,
    }
    assert aggregate["best_at_5_tanimoto"] == 0.7


def test_all_ged_timeouts_leave_other_paired_statistics_available():
    current = _comparable_report()
    baseline = _comparable_report()
    current["queries"][0]["chemical_nged_v2"] = None
    baseline["queries"][0]["chemical_nged_v2"] = None

    comparison = compare_evaluation_results(current, baseline)

    assert comparison["top1_tanimoto_v2_minus_baseline"]["available"] is True
    assert comparison["top1_chemical_nged_baseline_minus_v2"] == {
        "available": False,
        "mean_delta": None,
        "ci_2_5": None,
        "ci_97_5": None,
        "n_pairs": 0,
        "n_total": 1,
        "n_resamples": 10_000,
        "seed": 0,
    }


def test_metrics_beyond_observed_topk_are_null_and_not_bootstrapped():
    row = {
        "requested_top_k": 2,
        "observed_depth": 2,
        "eligible_candidate_count": 20,
        "latency_seconds": 0.1,
        "candidates": [
            {"tanimoto": 0.7, "legacy_ged_v1": 1.0, "chemical_ged_v2": 1.0,
             "chemical_nged_v2": 0.1, "exact_match": True, "scaffold_match": True,
             "cosine_score": 0.9, "heavy_atom_gap": 0, "element_jaccard": 1,
             "molecular_weight_error": 0, "unseen_element_fraction": 0},
        ] * 2,
    }

    aggregate = structure_metrics.aggregate_structure_retrieval([row])

    assert aggregate["exact_match_recall_at_1"] == 1.0
    assert aggregate["exact_match_recall_at_5"] is None
    assert aggregate["scaffold_recall_at_10"] is None
    assert aggregate["best_at_5_tanimoto"] is None
    report = _comparable_report()
    report["config"]["top_k"] = 2
    report["queries"][0].update(requested_top_k=2, observed_depth=2)
    with pytest.raises(ValueError, match="observed depth|Recall@10"):
        compare_evaluation_results(report, report)


def test_candidate_unseen_elements_use_training_vocab_and_unique_full_corpus():
    audit = candidate_corpus_audit(
        ["CC", "CCl", "ClC", "CBr"],
        training_atomic_numbers=[6],
        training_element_source="split:train",
    )

    assert audit["training_atomic_numbers"] == [6]
    assert audit["training_element_source"] == "split:train"
    assert audit["unique_identity_count"] == 3
    assert audit["unseen_element_molecule_count"] == 2
    assert audit["unseen_element_molecule_fraction"] == pytest.approx(2 / 3)
    assert audit["unseen_element_atom_count"] == 2
    assert audit["total_heavy_atom_count"] == 6
    assert audit["unseen_element_atom_fraction"] == pytest.approx(1 / 3)


def test_chemical_raw_and_normalized_ged_share_one_search(monkeypatch):
    calls = []

    def ged(left, right, **kwargs):
        calls.append(kwargs["version"])
        return 2.0 if kwargs["version"] == "chemical_ged_v2" else 3.0

    monkeypatch.setattr(structure_metrics, "graph_edit_distance", ged)
    monkeypatch.setattr(
        structure_metrics,
        "normalized_graph_edit_distance",
        lambda *args, **kwargs: pytest.fail("normalized GED must reuse completed chemical raw GED"),
    )
    monkeypatch.setattr(structure_metrics, "tanimoto_similarity", lambda *args: 0.5)

    result = structure_metrics.rank_structure_candidates("CCO", ["CCCO"], [0.8])

    assert calls == ["legacy_ged_v1", "chemical_ged_v2"]
    assert result["chemical_nged_v2"] == pytest.approx(2 / 12)

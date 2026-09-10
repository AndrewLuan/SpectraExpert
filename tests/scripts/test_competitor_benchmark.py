"""功能概述：验证竞争方法统一评分入口对检索与生成结果采用完全相同的身份和结构口径。
主要职责：test_identity_normalization 覆盖 canonical/group 规范化；两个 score_result 测试构造
最小冻结 query/pool，校验 Top-K、top-1、无效生成和汇总字段，不启动昂贵 GED 搜索。
代码库关系：直接测试 scripts.score_competitor_benchmark；通过 monkeypatch 固定 GED 返回值，
因此测试聚焦 adapter 协议解析和聚合契约，GED 化学语义由 eval metrics 自身测试负责。
输入输出与副作用：只在 pytest 的 tmp_path 中写临时 JSONL；不读取真实数据、不调用模型或网络。
"""

from __future__ import annotations

import json

import pytest

from scripts import score_competitor_benchmark as scorer


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def frozen_rows():
    queries = [
        {"query_id": "q0000", "mol_id": 7, "target_smiles": "CCO", "group_hash": "g1"}
    ]
    pool = [
        {"candidate_id": "m00007", "mol_id": 7, "smiles": "CCO", "group_hash": "g1"},
        {"candidate_id": "m00008", "mol_id": 8, "smiles": "CCN", "group_hash": "g2"},
    ]
    return queries, pool


def patch_metrics(monkeypatch):
    monkeypatch.setattr(scorer, "tanimoto_similarity", lambda left, right: 1.0 if left == right else 0.25)
    monkeypatch.setattr(
        scorer,
        "graph_edit_distance",
        lambda left, right, timeout, version: 0.0 if left == right else 2.0,
    )


def test_identity_normalization():
    assert scorer.canonical("C(C)O") == scorer.canonical("CCO")
    assert scorer.group_identity("CCO.Cl") == scorer.group_identity("CCO")
    assert scorer.canonical("not-smiles") is None


def test_score_retrieval_uses_frozen_candidate_identity(tmp_path, monkeypatch, frozen_rows):
    patch_metrics(monkeypatch)
    queries, pool = frozen_rows
    ranked = [
        {"candidate_id": "m00007", "candidate_smiles": "CCO", "score": 1.0}
    ] + [
        {"candidate_id": f"extra-{index}", "candidate_smiles": "C", "score": 0.0}
        for index in range(24)
    ]
    extended_pool = pool + [
        {"candidate_id": f"extra-{index}", "mol_id": 100 + index, "smiles": "C", "group_hash": "x0"}
        for index in range(24)
    ]
    result = tmp_path / "retrieval.jsonl"
    write_jsonl(result, [{"query_id": "q0000", "method": "demo", "arm": "retrieval",
                          "modality": "raman", "ranked": ranked}])
    rows, summary = scorer.score_result("demo", result, queries, extended_pool, 1.0, 1)
    assert rows[0]["canonical_exact"] and rows[0]["group_exact"]
    assert summary["canonical_hit_at_25"] == 1.0
    assert summary["chemical_ged_v2"]["mean"] == 0.0


def test_score_generation_keeps_invalid_top1_and_scores_beam(tmp_path, monkeypatch, frozen_rows):
    patch_metrics(monkeypatch)
    queries, pool = frozen_rows
    result = tmp_path / "generation.jsonl"
    write_jsonl(result, [{"query_id": "q0000", "method": "demo", "arm": "generation",
                          "modality": "raman", "greedy_smiles": "not-smiles",
                          "beam_smiles": ["not-smiles", "CCO"]}])
    rows, summary = scorer.score_result("demo", result, queries, pool, 1.0, 1)
    assert not rows[0]["valid"]
    assert not rows[0]["canonical_hit_at_1"] and rows[0]["canonical_hit_at_5"]
    assert not rows[0]["group_hit_at_1"] and rows[0]["group_hit_at_5"]
    assert summary["tanimoto"]["n"] == 0

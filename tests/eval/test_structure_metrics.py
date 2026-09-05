"""
功能概述：锁定分子结构评估的版本化 GED、归一化 GED 与批量 Tanimoto 公开契约。
主要职责：验证 legacy GED 保留旧芳香键截断与首个 generator 上界语义，chemical GED 使用
可区分芳香键的 schema 且只在完整搜索后返回最小值；验证 nGED 的节点边分母和 tensor
Tanimoto 的对称、单位对角线与交并比，并覆盖未完成搜索不得泄漏中间上界和过滤无效候选后的
原始检索 rank 映射。本轮还验证任意调用线程的硬 timeout、搜索异常和空 SMILES 在所有公开
指标及聚合路径中的一致无效语义。
代码库关系：直接消费 ``spectraexpert.eval.metrics``，并以 Task 2 的
``spectraexpert.data.structure_features`` 键类型和 Morgan schema 为唯一事实来源；评估脚本通过
这些函数比较检索候选的分子结构。
输入输出与副作用：测试仅构造小型 SMILES、内存 generator 与布尔 tensor，不读取数据集或写入
持久化状态；若 GED 搜索语义、归一化分母或相似度矩阵计算偏离契约，则 pytest 明确失败。
"""

import time
import threading

import pytest
import torch

import spectraexpert.eval.metrics as metrics
from spectraexpert.eval.metrics import (
    graph_edit_distance,
)


def test_chemical_ged_distinguishes_benzene_from_cyclohexane():
    assert graph_edit_distance("c1ccccc1", "C1CCCCC1", version="legacy_ged_v1") == 0.0
    assert (
        graph_edit_distance("c1ccccc1", "C1CCCCC1", version="chemical_ged_v2")
        > 0.0
    )


def test_chemical_ged_returns_minimum_completed_search_value(monkeypatch):
    monkeypatch.setattr(
        metrics.nx,
        "optimize_graph_edit_distance",
        lambda *args, **kwargs: iter([5.0, 3.0, 2.0]),
    )

    assert graph_edit_distance("CC", "CO", version="chemical_ged_v2") == 2.0
    assert graph_edit_distance("CC", "CO", version="legacy_ged_v1") == 5.0


def test_normalized_ged_uses_nodes_and_edges_denominator():
    raw = graph_edit_distance("CC", "CO", version="chemical_ged_v2")

    normalized = metrics.normalized_graph_edit_distance(
        "CC", "CO", version="chemical_ged_v2"
    )

    assert raw is not None
    assert normalized == pytest.approx(raw / 6.0)


def test_pairwise_tanimoto_is_symmetric_with_unit_diagonal():
    fingerprints = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)

    similarity = metrics.pairwise_tanimoto(fingerprints)

    assert torch.allclose(similarity, similarity.T)
    assert torch.allclose(similarity.diag(), torch.ones(2))
    assert similarity[0, 1].item() == pytest.approx(1 / 3)


def test_chemical_ged_returns_none_when_bounded_search_does_not_complete(monkeypatch):
    def unfinished_search(*args, **kwargs):
        yield 5.0
        time.sleep(0.05)
        yield 3.0

    monkeypatch.setattr(metrics.nx, "optimize_graph_edit_distance", unfinished_search)

    assert graph_edit_distance("CC", "CO", timeout=0.01) is None


def test_retrieval_metrics_keeps_original_rank_after_invalid_candidate():
    result = metrics.compute_retrieval_metrics(
        "CC",
        ["not-a-smiles", "CC", "CO"],
        [0.9, 0.8, 0.7],
    )

    assert result["rank_of_min_ged"] == 2
    assert result["best_ged_smiles"] == "CC"


def test_chemical_ged_with_zero_timeout_returns_none_immediately():
    started = time.monotonic()

    result = graph_edit_distance("CC", "CO", timeout=0.0)

    assert result is None
    assert time.monotonic() - started < 0.05


def test_chemical_ged_timeout_is_bounded_inside_background_thread(monkeypatch):
    def generator_that_blocks_before_yield(*args, **kwargs):
        time.sleep(0.5)
        if False:
            yield 1.0

    monkeypatch.setattr(
        metrics.nx, "optimize_graph_edit_distance", generator_that_blocks_before_yield
    )
    results = []
    worker = threading.Thread(
        target=lambda: results.append(graph_edit_distance("CC", "CO", timeout=0.02))
    )

    started = time.monotonic()
    worker.start()
    worker.join(0.2)

    assert not worker.is_alive()
    assert results == [None]
    assert time.monotonic() - started < 0.2


def test_chemical_ged_returns_none_when_generator_raises_after_yield(monkeypatch):
    def generator_that_raises(*args, **kwargs):
        yield 5.0
        raise RuntimeError("search failed")

    monkeypatch.setattr(metrics.nx, "optimize_graph_edit_distance", generator_that_raises)

    assert graph_edit_distance("CC", "CO") is None


@pytest.mark.parametrize(
    "metric",
    [
        metrics.graph_edit_distance,
        metrics.normalized_graph_edit_distance,
        metrics.tanimoto_similarity,
    ],
)
def test_empty_smiles_is_invalid_for_public_pairwise_metrics(metric):
    assert metric("", "CC") is None


def test_retrieval_metrics_skips_empty_smiles_without_raising():
    try:
        result = metrics.compute_retrieval_metrics("", ["", "CC"], [0.9, 0.8])
    except Exception as error:
        pytest.fail(f"empty SMILES must not make aggregation raise: {error!r}")

    assert result["top1_ged"] is None
    assert result["top1_tanimoto"] is None
    assert result["num_valid_candidates"] == 0

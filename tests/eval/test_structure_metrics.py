"""
功能概述：锁定分子结构评估的版本化 GED、归一化 GED 与批量 Tanimoto 公开契约。
主要职责：验证 legacy GED 保留旧芳香键截断与首个 generator 上界语义，chemical GED 使用
可区分芳香键的 schema 且只在完整搜索后返回最小值；验证 nGED 的节点边分母和 tensor
Tanimoto 的对称、单位对角线与交并比，并覆盖未完成搜索不得泄漏中间上界和过滤无效候选后的
原始检索 rank 映射。本轮还验证任意调用线程的硬 timeout、搜索异常和空 SMILES 在所有公开
指标及聚合路径中的一致无效语义。Fix Round 2 进一步验证 spawn worker 覆盖解析至搜索的
deadline、SIGTERM 后 kill 清理，以及零重原子 SMILES 的统一无效语义。
Fix Round 3 还锁定零超时时先验证版本、但绝不解析 SMILES 的公开 API 顺序契约。
代码库关系：直接消费 ``spectraexpert.eval.metrics``，并以 Task 2 的
``spectraexpert.data.structure_features`` 键类型和 Morgan schema 为唯一事实来源；评估脚本通过
这些函数比较检索候选的分子结构。
输入输出与副作用：测试仅构造小型 SMILES、内存 generator 与布尔 tensor，不读取数据集或写入
持久化状态；若 GED 搜索语义、归一化分母或相似度矩阵计算偏离契约，则 pytest 明确失败。
"""

import time
import threading
import multiprocessing as mp

import pytest
import torch

import spectraexpert.eval.metrics as metrics
from tests.eval import worker_helpers
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
        metrics, "_GED_WORKER_TARGET", worker_helpers.completed_values_worker, raising=False
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
    monkeypatch.setattr(
        metrics, "_GED_WORKER_TARGET", worker_helpers.blocking_worker, raising=False
    )

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
    monkeypatch.setattr(
        metrics, "_GED_WORKER_TARGET", worker_helpers.blocking_worker, raising=False
    )
    results = []
    worker = threading.Thread(
        target=lambda: results.append(graph_edit_distance("CC", "CO", timeout=0.05))
    )

    started = time.monotonic()
    worker.start()
    worker.join(0.5)

    assert not worker.is_alive()
    assert results == [None]
    assert time.monotonic() - started < 0.5


def test_spawn_worker_completes_chemical_ged_inside_background_thread(monkeypatch):
    monkeypatch.setattr(
        metrics, "_GED_WORKER_TARGET", worker_helpers.completed_values_worker, raising=False
    )
    results = []
    worker = threading.Thread(
        target=lambda: results.append(graph_edit_distance("CC", "CO", timeout=10.0))
    )

    worker.start()
    worker.join(12.0)

    assert not worker.is_alive()
    assert results == [2.0]


def test_chemical_ged_returns_none_when_generator_raises_after_yield(monkeypatch):
    monkeypatch.setattr(
        metrics, "_GED_WORKER_TARGET", worker_helpers.raising_values_worker, raising=False
    )

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


def test_timeout_zero_returns_before_parent_smiles_parsing(monkeypatch):
    original_parser = metrics.smiles_to_nx

    def slow_parser(*args, **kwargs):
        time.sleep(0.1)
        return original_parser(*args, **kwargs)

    monkeypatch.setattr(metrics, "smiles_to_nx", slow_parser)
    started = time.monotonic()

    assert graph_edit_distance("CC", "CO", timeout=0.0) is None
    assert time.monotonic() - started < 0.05


@pytest.mark.parametrize("hydrogen_smiles", ["[H]", "[H][H]"])
def test_zero_heavy_atom_smiles_are_invalid_for_all_structure_metrics(hydrogen_smiles):
    assert graph_edit_distance(hydrogen_smiles, "CC") is None
    assert metrics.normalized_graph_edit_distance(hydrogen_smiles, "CC") is None
    assert metrics.tanimoto_similarity(hydrogen_smiles, "CC") is None

    aggregate = metrics.compute_retrieval_metrics(
        hydrogen_smiles, [hydrogen_smiles, "CC"], [0.9, 0.8]
    )
    assert aggregate["top1_ged"] is None
    assert aggregate["top1_tanimoto"] is None
    assert aggregate["num_valid_candidates"] == 0


def test_sigterm_ignoring_spawn_worker_is_killed_without_residual_child(monkeypatch):
    monkeypatch.setattr(
        metrics, "_GED_WORKER_TARGET", worker_helpers.ignore_sigterm_worker, raising=False
    )
    existing_pids = {child.pid for child in mp.active_children()}
    started = time.monotonic()

    result = graph_edit_distance("CC", "CO", timeout=0.05)

    assert result is None
    assert time.monotonic() - started < 0.5
    assert {child.pid for child in mp.active_children()} <= existing_pids


@pytest.mark.parametrize(
    ("metric", "args"),
    [
        (metrics.graph_edit_distance, ("CC", "CO")),
        (metrics.normalized_graph_edit_distance, ("CC", "CO")),
        (metrics.batch_graph_edit_distance, ("CC", ["CO"])),
    ],
)
def test_zero_timeout_rejects_invalid_ged_version(metric, args):
    with pytest.raises(ValueError, match="version must be"):
        metric(*args, timeout=0.0, version="unsupported_ged_v0")


@pytest.mark.parametrize(
    ("metric", "args", "expected"),
    [
        (metrics.graph_edit_distance, ("CC", "CO"), None),
        (metrics.normalized_graph_edit_distance, ("CC", "CO"), None),
        (metrics.batch_graph_edit_distance, ("CC", ["CO"]), [None]),
    ],
)
def test_zero_timeout_valid_ged_version_never_parses_smiles(
    monkeypatch, metric, args, expected
):
    def parsing_must_not_run(*args, **kwargs):
        pytest.fail("zero-timeout GED must return before parsing SMILES")

    monkeypatch.setattr(metrics, "smiles_to_nx", parsing_must_not_run)

    assert metric(*args, timeout=0.0, version="chemical_ged_v2") == expected

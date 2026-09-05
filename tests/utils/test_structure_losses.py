"""
功能概述：验证结构感知训练目标在动态 batch、重复 canonical identity 和稀疏结构标签下的数值契约。
主要职责：分别锁定动态 NT-Xent、multi-positive soft target、Morgan Tanimoto 关系分布，以及
``StructureAwareObjective`` 对跨模态、增强、结构和辅助监督项的组合、梯度与 fail-fast 行为。
代码库关系：直接消费 ``spectraexpert.utils.structure_losses`` 和 Task 4 的 ``MultimodalOutput``；
使用手工小张量而不复制 Task 2 的 Morgan 生成 schema，也不读取任何真实光谱或分子数据。
输入输出与副作用：测试仅在当前设备内创建小型 tensor 并执行反向传播；不初始化分布式进程组，
不写 checkpoint。非法温度、无匹配 identity、shape 错误或非有限 loss 必须抛出可定位异常。
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from spectraexpert.models import MultimodalOutput
from spectraexpert.utils.structure_losses import (
    StructureAwareObjective,
    dynamic_nt_xent,
    multi_positive_cross_entropy,
    relational_structure_loss,
)


def test_multi_positive_identical_structures_are_not_negatives():
    logits = torch.tensor(
        [[4.0, 4.0, 0.0], [4.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        requires_grad=True,
    )
    ids = torch.tensor([11, 11, 12])

    loss = multi_positive_cross_entropy(logits, ids, ids)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad[0, 1].item() < 0.0
    assert logits.grad[0, 2].item() > 0.0


def test_multi_positive_rejects_rows_without_any_positive():
    with pytest.raises(ValueError, match=r"no positive.*\[1\]"):
        multi_positive_cross_entropy(
            torch.zeros(2, 2),
            torch.tensor([4, 99]),
            torch.tensor([4, 5]),
        )


def test_structure_loss_rewards_tanimoto_ordering():
    fingerprints = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], dtype=torch.bool
    )
    aligned = torch.tensor(
        [[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    reversed_similarity = torch.tensor(
        [[1.0, 0.0, 0.8], [0.0, 1.0, 0.8], [0.8, 0.8, 1.0]]
    )

    assert relational_structure_loss(aligned, fingerprints) < relational_structure_loss(
        reversed_similarity, fingerprints
    )


def test_structure_loss_handles_zero_fingerprint_union_and_singleton_diagonal_mask():
    similarities = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    zero_fingerprints = torch.zeros(2, 4, dtype=torch.bool)

    loss = relational_structure_loss(
        similarities,
        zero_fingerprints,
        exclude_diagonal=True,
    )
    loss.backward()

    assert loss.dtype == torch.float64
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert similarities.grad is not None
    singleton = torch.ones(1, 1, requires_grad=True)
    singleton_loss = relational_structure_loss(
        singleton,
        torch.zeros(1, 4, dtype=torch.bool),
        exclude_diagonal=True,
    )
    singleton_loss.backward()
    assert singleton_loss.item() == 0.0
    assert singleton.grad is not None


def test_dynamic_nt_xent_accepts_last_short_batch_and_backpropagates():
    first_raw = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    second_raw = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    first = F.normalize(first_raw, dim=1)
    second = F.normalize(second_raw, dim=1)

    loss = dynamic_nt_xent(first, second, 0.1)
    loss.backward()

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert loss.dtype == torch.float64
    assert first_raw.grad is not None and first_raw.grad.abs().sum() > 0
    assert second_raw.grad is not None and second_raw.grad.abs().sum() > 0


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan"), float("inf")])
def test_loss_temperatures_must_be_positive_and_finite(temperature):
    embeddings = F.normalize(torch.randn(2, 4), dim=1)
    fingerprints = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)

    with pytest.raises(ValueError, match="temperature"):
        dynamic_nt_xent(embeddings, embeddings, temperature)
    with pytest.raises(ValueError, match="temperature"):
        relational_structure_loss(
            embeddings @ embeddings.T,
            fingerprints,
            model_temperature=temperature,
        )
    with pytest.raises(ValueError, match="temperature"):
        StructureAwareObjective(temperature=temperature)


def _multimodal_output(dtype: torch.dtype = torch.float64):
    torch.manual_seed(7)
    spectrum1_raw = torch.randn(3, 5, dtype=dtype, requires_grad=True)
    spectrum2_raw = torch.randn(3, 5, dtype=dtype, requires_grad=True)
    graph_raw = torch.randn(3, 5, dtype=dtype, requires_grad=True)
    descriptor_prediction = torch.randn(3, 2, dtype=dtype, requires_grad=True)
    presence_logits = torch.randn(3, 2, dtype=dtype, requires_grad=True)
    fingerprint_logits = torch.randn(3, 4, dtype=dtype, requires_grad=True)
    output = MultimodalOutput(
        spectrum_view1_embedding=F.normalize(spectrum1_raw, dim=1),
        spectrum_view2_embedding=F.normalize(spectrum2_raw, dim=1),
        graph_embedding=F.normalize(graph_raw, dim=1),
        spectrum_view1_features=torch.zeros(3, 2, dtype=dtype),
        spectrum_view2_features=torch.zeros(3, 2, dtype=dtype),
        graph_features=torch.zeros(3, 2, dtype=dtype),
        descriptor_prediction=descriptor_prediction,
        element_presence_logits=presence_logits,
        fingerprint_logits=fingerprint_logits,
    )
    leaves = (
        spectrum1_raw,
        spectrum2_raw,
        graph_raw,
        descriptor_prediction,
        presence_logits,
        fingerprint_logits,
    )
    return output, leaves


def test_objective_combines_all_terms_in_output_dtype_and_backpropagates():
    output, leaves = _multimodal_output()
    fingerprints = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.bool
    )
    descriptor_targets = torch.tensor(
        [[0.0, 1.0], [0.5, -0.5], [1.0, 0.0]], dtype=torch.float32
    )
    element_presence = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.bool)
    identity_ids = torch.tensor([11, 11, 12])
    objective = StructureAwareObjective(
        temperature=0.2,
        structure_temperature=0.3,
        lambda_aug=0.4,
        lambda_struct=0.6,
        lambda_aux=0.7,
        lambda_fingerprint=0.8,
    )

    losses = objective(
        output,
        fingerprints,
        descriptor_targets,
        element_presence,
        identity_ids,
    )
    expected_total = (
        losses["cross"]
        + 0.4 * losses["aug"]
        + 0.6 * losses["struct"]
        + 0.7 * (losses["descriptor"] + losses["presence"])
        + 0.8 * losses["fingerprint"]
    )

    assert set(losses) == {
        "total",
        "cross",
        "aug",
        "struct",
        "descriptor",
        "presence",
        "fingerprint",
    }
    torch.testing.assert_close(losses["total"], expected_total)
    assert all(loss.ndim == 0 and loss.dtype == torch.float64 for loss in losses.values())
    assert all(torch.isfinite(loss) for loss in losses.values())
    losses["total"].backward()
    assert all(leaf.grad is not None and leaf.grad.abs().sum() > 0 for leaf in leaves)


def test_objective_without_fingerprint_head_returns_differentiable_zero_component():
    output, _ = _multimodal_output(dtype=torch.float32)
    output.fingerprint_logits = None
    objective = StructureAwareObjective(lambda_fingerprint=3.0)

    losses = objective(
        output,
        torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.bool),
        torch.zeros(3, 2),
        torch.zeros(3, 2, dtype=torch.bool),
        torch.arange(3),
    )

    assert losses["fingerprint"].item() == 0.0
    assert losses["fingerprint"].dtype == torch.float32
    losses["total"].backward()


def test_objective_rejects_non_finite_component_loss():
    output, _ = _multimodal_output(dtype=torch.float32)
    bad_targets = torch.zeros(3, 2)
    bad_targets[1, 0] = math.nan

    with pytest.raises(FloatingPointError, match="descriptor"):
        StructureAwareObjective()(
            output,
            torch.tensor(
                [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool
            ),
            bad_targets,
            torch.zeros(3, 2, dtype=torch.bool),
            torch.arange(3),
        )

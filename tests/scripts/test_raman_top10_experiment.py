"""功能概述：检验 Top10 新实验的结构排名和矩形多正样本损失契约。
主要职责：确认稳定同分排序、重复测量的组等权统计，以及额外参考不会产生无正样本
反向项或把同身份记录误当负样本；不加载真实数据、不访问 GPU 或修改实验结果。
代码库关系：直接调用 scripts.raman_top10_experiment 的纯数值入口，使用合成小矩阵。
"""
import numpy as np
import torch
import pytest

from scripts.raman_top10_experiment import retrieval, contrastive


def test_ranking_group_weights_and_stable_ties():
    query = np.array([[1., 0.], [1., 0.], [0., 1.]])
    reference = np.array([[1., 0.], [1., 0.], [0., 1.]] + [[-1., -1.]] * 9)
    metrics, ranks, top = retrieval(query, reference, [0, 0, 0], ['a', 'a', 'b'])
    assert ranks.tolist() == [1, 1, 2]
    assert top[0, :2].tolist() == [0, 1]
    assert metrics['exact@1'] == pytest.approx(2/3)
    assert metrics['group_exact@1'] == .5
    assert metrics['exact@10'] == 1.


def test_extra_references_support_gradients_and_multi_positive():
    left = torch.eye(2, requires_grad=True)
    right = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [-1., -1.]], requires_grad=True)
    loss = contrastive(left, right, torch.tensor([5, 8]), torch.tensor([5, 8, 5, 9]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(left.grad).all() and torch.isfinite(right.grad).all()
    wrong = contrastive(left, right, torch.tensor([5, 8]), torch.tensor([5, 8, 9, 9]))
    assert loss < wrong
    with pytest.raises(ValueError, match='no positive'):
        contrastive(left, right, torch.tensor([5, 77]), torch.tensor([5, 8, 5, 9]))

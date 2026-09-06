"""功能概述：用可手算的检索结果验证验证集指标及身份屏蔽。
主要职责：测试真实排序聚合逻辑、无候选错误和模型训练状态恢复。
代码库关系：validate_retrieval 消费 batch 中光谱、图和固定标签；此处使用确定性编码器隔离排序。
输入输出与副作用：仅 CPU 小矩阵，无磁盘和外部服务操作。
"""
import pytest
import torch
from spectraexpert.training.validation import validate_retrieval


class KnownEncoder(torch.nn.Module):
    def encode_spectrum(self, spectrum):
        return spectrum

    def encode_graph(self, graph):
        return graph


def batch():
    return dict(spectrum=torch.eye(3), graph=torch.eye(3),
        fingerprint=torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.bool),
        identity_id=torch.tensor([4, 4, 9]), heavy_atoms=torch.tensor([1., 1., 3.]))


def test_validation_exact_retrieval_and_restore_training_state():
    model = KnownEncoder().train()
    metrics = validate_retrieval(model, [batch()], "cpu")
    assert metrics["top1_tanimoto"] == 1.
    assert metrics["best5_tanimoto"] == 1.
    assert metrics["exact_recall_at_1"] == 1.
    assert metrics["exact_recall_at_5"] == 1.
    assert metrics["mean_heavy_atom_gap"] == 0.
    assert model.training


def test_exclude_identity_masks_every_duplicate_and_short_top5():
    result = validate_retrieval(KnownEncoder(), [batch()], "cpu", exclude_identity=True)
    assert result["top1_tanimoto"] == 0.
    assert result["best5_tanimoto"] == 0.
    assert result["exact_recall_at_5"] == 0.
    assert result["mean_heavy_atom_gap"] == 2.


def test_empty_validation_and_no_eligible_candidates_fail_clearly():
    model = KnownEncoder().train()
    with pytest.raises(ValueError, match="empty"):
        validate_retrieval(model, [], "cpu")
    row = batch()
    row["identity_id"] = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="eligible"):
        validate_retrieval(model, [row], "cpu", exclude_identity=True)
    assert model.training

"""功能概述：验证训练 checkpoint 的版本隔离、严格重建和原子写入契约。
主要职责：通过真实模型保存/恢复验证权重、优化器与 RNG；篡改配置验证字段级拒绝。
代码库关系：消费 training.checkpoints，避免旧 checkpoint 被静默解释为新模型。
输入输出与副作用：只在 pytest 临时目录创建小型模型文件，不访问实验数据。
"""
import random

import numpy as np
import pytest
import torch

from spectraexpert.data.structure_features import DescriptorNormalizer
from spectraexpert.models.multimodal_retriever import StructureAwareRetriever
from spectraexpert.training.checkpoints import load_checkpoint, save_checkpoint


def make_model():
    return StructureAwareRetriever.from_config(dict(signal_length=65, patch_size=8,
        stride=4, embed_dim=16, depth=1, num_heads=4, projection_dim=8,
        graph_hidden_dim=16, descriptor_dim=15, predict_fingerprint=False))


def test_checkpoint_roundtrip_records_metadata_and_rng(tmp_path):
    model = make_model().eval()
    optimizer = torch.optim.Adam(model.parameters())
    stats = DescriptorNormalizer(torch.zeros(15), torch.ones(15))
    path = save_checkpoint(tmp_path / "model.pt", model, optimizer, epoch=2,
                           validation={"top1_tanimoto": .4}, descriptor_stats=stats,
                           split_manifest_sha256="abc", source_sha256={"spectra": "123"})
    expected_rng = (random.random(), np.random.rand(), torch.rand(2))
    restored, payload = load_checkpoint(path, restore_rng=True)
    assert payload["schema_version"] == "spectrum_graph_v2"
    assert payload["training_mode"] == "spectrum_graph_v2"
    assert payload["feature_config"]["morgan_radius"] == 2
    assert payload["feature_config"]["use_chirality"] is False
    assert payload["selection_metric"] == "top1_tanimoto"
    assert payload["optimizer_state_dict"] is not None
    assert payload["split_manifest_sha256"] == "abc"
    assert payload["source_sha256"] == {"spectra": "123"}
    assert random.random() == expected_rng[0]
    assert np.random.rand() == expected_rng[1]
    assert torch.equal(torch.rand(2), expected_rng[2])
    x = torch.randn(2, 65)
    assert torch.equal(model.encode_spectrum(x), restored.eval().encode_spectrum(x))


@pytest.mark.parametrize("field,value,expected", [
    ("schema_version", "legacy", "schema_version"),
    ("training_mode", "spectral_only_v1", "training_mode"),
    ("model_config", {"projection_dim": 33}, "model_config"),
    ("feature_config", {"morgan_radius": 3}, "feature_config"),
    ("descriptor_stats", {"mean": torch.zeros(15), "std": torch.zeros(15)}, "descriptor_stats"),
    ("epoch", -1, "epoch"),
    ("global_steps", "2", "global_steps"),
    ("rng_states", [], "rng_states"),
    ("selection_exclude_identity", False, "selection_exclude_identity"),
])
def test_checkpoint_rejects_incompatible_metadata(tmp_path, field, value, expected):
    path = save_checkpoint(tmp_path / "model.pt", make_model(), descriptor_stats=
                           DescriptorNormalizer(torch.zeros(15), torch.ones(15)))
    payload = torch.load(path, weights_only=False)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=expected):
        load_checkpoint(path)


def test_checkpoint_rejects_wrong_projection_and_unknown_config_field(tmp_path):
    path = save_checkpoint(tmp_path / "model.pt", make_model(), descriptor_stats=
                           DescriptorNormalizer(torch.zeros(15), torch.ones(15)))
    with pytest.raises(ValueError, match="projection_dim"):
        load_checkpoint(path, expected_model_config={"projection_dim": 99})
    payload = torch.load(path, weights_only=False)
    payload["model_config"]["ignored_typo"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match="model_config"):
        load_checkpoint(path)


def test_atomic_save_preserves_existing_checkpoint_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    path.write_bytes(b"previous")
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(path, make_model(), descriptor_stats=
                        DescriptorNormalizer(torch.zeros(15), torch.ones(15)))
    assert path.read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.pt"]

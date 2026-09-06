"""
功能概述：验证安装后的 ``spectraexpert`` 包不再依赖历史 ``proj.spectra_expert`` 命名空间，并为
旧版余弦检索 API 建立最小、可重复的行为基线。
主要职责：``test_public_modules_import_without_legacy_proj_namespace`` 覆盖训练、管线、评估和推理
公开模块；检索测试锁定精确余弦分数、稳定并列排序、dtype 与非法输入；checkpoint 测试锁定 legacy
裸 state dict 的参数恢复、带 args/prefix 的恢复、单谱编码和歧义架构的明确失败。
代码库关系：由 ``pytest.ini`` 收集，直接消费 ``spectraexpert.inference``；它保护
``train_contrastive``、pipeline 和 eval 的统一绝对包导入迁移。
输入输出与副作用：构造内存中的 float32 向量，不读取 checkpoint 或数据集；导入失败或检索排序偏离
契约时以 pytest assertion 失败，不修改项目文件。
"""

import importlib

import numpy as np
import pytest
import torch

from spectraexpert.models.spectrum_encoder import SpectrumTransformer


def test_public_modules_import_without_legacy_proj_namespace():
    for name in (
        "spectraexpert.train_contrastive",
        "spectraexpert.pipeline",
        "spectraexpert.eval.metrics",
        "spectraexpert.inference",
    ):
        importlib.import_module(name)


def test_retrieve_returns_exact_stable_cosine_neighbors_with_expected_dtypes():
    from spectraexpert.inference import SpectralEncoder

    query = np.array([1.0, 0.0], dtype=np.float32)
    database = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32
    )

    indices, scores = SpectralEncoder.retrieve(query, database, topk=4)

    assert indices.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(scores, [1.0, 1.0, 0.8, 0.0], rtol=0, atol=1e-6)
    assert indices.dtype == np.int64
    assert scores.dtype == np.float32


@pytest.mark.parametrize(
    ("query", "database", "topk"),
    [
        (np.array([np.nan], dtype=np.float32), np.array([[1.0]], dtype=np.float32), 1),
        (np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32), 1),
        (np.array([1.0], dtype=np.float32), np.array([[1.0, 0.0]], dtype=np.float32), 1),
        (np.array([1.0], dtype=np.float32), np.array([[1.0]], dtype=np.float32), 0),
    ],
)
def test_retrieve_rejects_nonfinite_or_invalid_shapes(query, database, topk):
    from spectraexpert.inference import SpectralEncoder

    with pytest.raises(ValueError):
        SpectralEncoder.retrieve(query, database, topk)


def test_bare_default_training_state_dict_recovers_legacy_architecture(tmp_path):
    from spectraexpert.inference import SpectralEncoder

    trained_model = SpectrumTransformer(
        signal_length=3501,
        patch_size=16,
        stride=8,
        embed_dim=256,
        depth=6,
        num_heads=8,
        projection_dim=256,
    )
    checkpoint_path = tmp_path / "legacy-bare-state-dict.pth"
    torch.save(trained_model.state_dict(), checkpoint_path)

    encoder = SpectralEncoder(str(checkpoint_path), device="cpu")

    assert encoder.model.patch_embed.signal_length == 3501
    assert encoder.model.patch_embed.stride == 8
    assert encoder.model.encoder.num_layers == 6
    assert encoder.model.projection[-1].out_features == 256


def test_checkpoint_args_and_simclr_prefix_encode_finite_spectrum(tmp_path):
    from spectraexpert.inference import SpectralEncoder

    model_args = {
        "signal_length": 8,
        "patch_size": 2,
        "stride": 2,
        "embed_dim": 4,
        "depth": 1,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "projection_dim": 3,
    }
    trained_model = SpectrumTransformer(**model_args).eval()
    checkpoint_path = tmp_path / "prefixed-checkpoint.pth"
    torch.save(
        {
            "args": model_args,
            "model_state_dict": {
                f"module.encoder.{key}": value
                for key, value in trained_model.state_dict().items()
            },
        },
        checkpoint_path,
    )

    encoder = SpectralEncoder(str(checkpoint_path), device="cpu")
    embedding = encoder.encode_spectrum(np.arange(8, dtype=np.float32))

    assert embedding.shape == (3,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    np.testing.assert_allclose(np.linalg.norm(embedding), 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="finite"):
        encoder.encode_spectrum(np.array([0.0, np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="one-dimensional"):
        encoder.encode_spectrum(np.ones((1, 8), dtype=np.float32))


def test_bare_state_dict_with_ambiguous_architecture_requires_metadata(tmp_path):
    from spectraexpert.inference import SpectralEncoder

    ambiguous_model = SpectrumTransformer(
        signal_length=8,
        patch_size=2,
        stride=2,
        embed_dim=4,
        depth=1,
        num_heads=2,
        projection_dim=3,
    )
    checkpoint_path = tmp_path / "ambiguous-bare-state-dict.pth"
    torch.save(ambiguous_model.state_dict(), checkpoint_path)

    with pytest.raises(ValueError, match="cannot infer"):
        SpectralEncoder(str(checkpoint_path), device="cpu")

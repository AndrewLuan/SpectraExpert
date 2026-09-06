"""
功能概述：验证 Stage 1 的 ``spectrum_graph_v2`` 模式能以纯光谱 query 检索离线分子图候选。
主要职责：用 Task 7 真实 65 点 checkpoint、真实 graph encoder 和 Task 8 NPZ index 构建 fixture，
覆盖训练/推理 exact bool 归一化绑定、稳定 top-k、Candidate provenance、loss-gated 辅助重排、
checkpoint/index 不匹配、v2 默认 artifact 路径与显式 mode。
代码库关系：连接 ``StructureAwareSpectralEncoder``、``GraphIndex``、PipelineConfig 和
``Stage1Retrieval``；legacy 模式仍由旧 wrapper/cache 路径负责，不能静默回退到本模式。
输入输出与副作用：只在 pytest 临时目录写小 checkpoint、index 与 65 点 wavenumber Parquet；autouse
fixture 固定并恢复 Python/NumPy/Torch RNG，避免随机模型影响其他测试；不 mock encoder/index loader，
不读取真实 QM9S 数据，不传 SMILES、formula、descriptor 或 fingerprint 到 query。
"""

from __future__ import annotations

import inspect
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from spectraexpert.data.molecular_graph import collate_graphs, featurize_smiles
from spectraexpert.data.structure_dataset import canonical_smiles_sha256
from spectraexpert.data.structure_features import DescriptorNormalizer, StructureFeatureConfig
from spectraexpert.models import GraphEncoder, StructureAwareRetriever
from spectraexpert.models.spectrum_encoder import SpectrumTransformer
from spectraexpert.pipeline.config import PipelineConfig
from spectraexpert.pipeline.data_types import Candidate
from spectraexpert.pipeline.stage1_retrieval import Stage1Retrieval
from spectraexpert.retrieval import GraphIndex
from spectraexpert.training.checkpoints import save_checkpoint


@pytest.fixture(autouse=True)
def _isolated_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(802)
    np.random.seed(802)
    torch.manual_seed(802)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def _model(*, predict_fingerprint: bool = True) -> StructureAwareRetriever:
    return StructureAwareRetriever(
        SpectrumTransformer(signal_length=65, patch_size=8, stride=4, embed_dim=16,
                            depth=1, num_heads=4, projection_dim=8),
        GraphEncoder(hidden_dim=16, projection_dim=8),
        descriptor_dim=15,
        predict_fingerprint=predict_fingerprint,
    ).eval()


def _v2_config(tmp_path: Path, *, predict_fingerprint: bool = True,
               checkpoint_normalize: object = True,
               config_normalize: object = True,
               lambda_aux: float = 1.0) -> dict:
    model = _model(predict_fingerprint=predict_fingerprint)
    checkpoint = tmp_path / "model.pt"
    normalizer = DescriptorNormalizer(torch.zeros(15), torch.ones(15))
    save_checkpoint(checkpoint, model, descriptor_stats=normalizer,
                    loss_weights={"lambda_aux": lambda_aux,
                                  "lambda_fingerprint": 1.0 if predict_fingerprint else 0.0},
                    train_config={"normalize": checkpoint_normalize})
    graphs = [featurize_smiles("CC", 1), featurize_smiles("CO", 2), featurize_smiles("CN", 3)]
    batch = collate_graphs(graphs)
    with torch.inference_mode():
        embeddings = model.encode_graph(batch).numpy()
    source_hash = canonical_smiles_sha256(
        batch.mol_ids.tolist(), list(batch.canonical_smiles)
    )
    index = GraphIndex.from_arrays(
        embeddings,
        batch.mol_ids.numpy(),
        np.asarray(batch.canonical_smiles),
        batch.fingerprints.numpy(),
        batch.descriptors.numpy(),
        checkpoint_sha256=__import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        feature_metadata=StructureFeatureConfig().to_metadata(),
        model_config=model.to_config(),
        source_smiles_sha256=source_hash,
        descriptor_mean=normalizer.mean.numpy(),
        descriptor_std=normalizer.std.numpy(),
        loss_weights={"lambda_aux": lambda_aux,
                      "lambda_fingerprint": 1.0 if predict_fingerprint else 0.0},
    )
    index_path = tmp_path / "index.npz"
    index.save(index_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"wavenumber": np.linspace(500, 4000, 65)}).to_parquet(
        data_dir / "wavenumber.parquet", index=False
    )
    return {
        "pipeline": {"top_k": 2, "device": "cpu"},
        "stage1": {
            "retrieval_mode": "spectrum_graph_v2",
            "checkpoint": str(checkpoint),
            "graph_index": str(index_path),
            "data_dir": str(data_dir),
            "normalize_spectrum": config_normalize,
            "rerank": {
                "enabled": False,
                "top_m": 3,
                "fingerprint_weight": 0.0,
                "descriptor_weight": 0.0,
            },
        },
        "stage2": {"peak_detection": {}},
    }


def _legacy_config(tmp_path: Path) -> dict:
    args = {
        "signal_length": 65,
        "patch_size": 8,
        "stride": 4,
        "embed_dim": 16,
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "projection_dim": 8,
    }
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"args": args, "model_state_dict": SpectrumTransformer(**args).state_dict()}, checkpoint)
    cache = tmp_path / "legacy-index.npz"
    np.savez_compressed(cache, embeddings=np.eye(2, 8, dtype=np.float32),
                        mol_ids=np.asarray([1, 2], dtype=np.int64),
                        smiles=np.asarray(["CC", "CO"]))
    data_dir = tmp_path / "legacy-data"
    data_dir.mkdir()
    pd.DataFrame({"wavenumber": np.linspace(500, 4000, 65)}).to_parquet(
        data_dir / "wavenumber.parquet", index=False
    )
    return {
        "pipeline": {"top_k": 2, "device": "cpu"},
        "stage1": {
            "retrieval_mode": "spectral_only_v1",
            "checkpoint": str(checkpoint),
            "embeddings_cache": str(cache),
            "data_dir": str(data_dir),
            "normalize_spectrum": False,
        },
        "stage2": {"peak_detection": {}},
    }


def test_stage1_v2_run_accepts_only_spectrum_and_records_provenance(tmp_path):
    config = _v2_config(tmp_path)
    spectrum = np.random.default_rng(0).normal(size=65).astype(np.float32)
    stage = Stage1Retrieval(config)
    result = stage.run(spectrum)

    assert list(inspect.signature(stage.run).parameters) == ["spectrum"]
    assert result.query_embedding.shape == (8,)
    assert len(result.candidates) == config["pipeline"]["top_k"]
    assert all(candidate.retrieval_mode == "spectrum_graph_v2" for candidate in result.candidates)
    assert all(candidate.embedding_similarity is not None for candidate in result.candidates)
    assert all(candidate.similarity == candidate.embedding_similarity for candidate in result.candidates)
    assert [candidate.rank for candidate in result.candidates] == [1, 2]


def test_stage1_v2_normalizes_query_once_before_both_embedding_and_auxiliary(tmp_path):
    config = _v2_config(tmp_path)
    spectrum = np.linspace(-3.0, 4.0, 65, dtype=np.float32)
    stage = Stage1Retrieval(config)
    result = stage.run(spectrum)
    normalized = ((spectrum - spectrum.mean()) / spectrum.std()).astype(np.float32)
    expected = stage._encoder.encode_spectrum(normalized)
    np.testing.assert_allclose(result.query_embedding, expected, rtol=1e-6, atol=1e-6)


def test_stage1_v2_false_normalize_uses_raw_query_once(tmp_path):
    """checkpoint/config 均为 False 时不得标准化 raw spectrum。"""
    config = _v2_config(tmp_path, checkpoint_normalize=False, config_normalize=False)
    spectrum = np.linspace(-3.0, 4.0, 65, dtype=np.float32)
    stage = Stage1Retrieval(config)
    result = stage.run(spectrum)
    expected = stage._encoder.encode_spectrum(spectrum)
    normalized = stage._encoder.encode_spectrum(
        ((spectrum - spectrum.mean()) / spectrum.std()).astype(np.float32)
    )
    np.testing.assert_allclose(result.query_embedding, expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(result.query_embedding, normalized)


@pytest.mark.parametrize(
    ("checkpoint_normalize", "config_normalize"),
    [(False, True), (True, False), (True, "false")],
)
def test_stage1_v2_rejects_normalize_mismatch_or_non_bool(
    tmp_path, checkpoint_normalize, config_normalize
):
    """Stage1 必须 exact bool 且与 checkpoint 训练预处理一致。"""
    config = _v2_config(
        tmp_path,
        checkpoint_normalize=checkpoint_normalize,
        config_normalize=config_normalize,
    )
    with pytest.raises(ValueError, match="normalize_spectrum"):
        Stage1Retrieval(config)


def test_stage1_optional_rerank_only_consumes_spectrum_predictions(tmp_path):
    config = _v2_config(tmp_path)
    config["stage1"]["rerank"] = {
        "enabled": True,
        "top_m": 3,
        "fingerprint_weight": 0.2,
        "descriptor_weight": 0.1,
    }
    result = Stage1Retrieval(config).run(
        np.random.default_rng(1).normal(size=65).astype(np.float32)
    )
    assert len(result.candidates) == 2
    assert all(np.isfinite(candidate.similarity) for candidate in result.candidates)
    assert all(np.isfinite(candidate.embedding_similarity) for candidate in result.candidates)
    assert any(candidate.similarity != candidate.embedding_similarity
               for candidate in result.candidates)


def test_stage1_rejects_positive_fingerprint_weight_when_checkpoint_has_no_head(tmp_path):
    config = _v2_config(tmp_path, predict_fingerprint=False)
    config["stage1"]["rerank"] = {
        "enabled": True,
        "top_m": 3,
        "fingerprint_weight": 0.1,
        "descriptor_weight": 0.0,
    }
    with pytest.raises(ValueError, match="fingerprint head"):
        Stage1Retrieval(config)


def test_stage1_rejects_positive_descriptor_weight_when_checkpoint_loss_is_zero(tmp_path):
    """物理 descriptor head 存在但 lambda_aux=0 时不得启用 descriptor rerank。"""
    config = _v2_config(tmp_path, lambda_aux=0.0)
    config["stage1"]["rerank"] = {
        "enabled": True,
        "top_m": 3,
        "fingerprint_weight": 0.0,
        "descriptor_weight": 0.1,
    }
    with pytest.raises(ValueError, match="descriptor.*trained|lambda_aux"):
        Stage1Retrieval(config)


def test_stage1_rejects_checkpoint_index_mismatch_and_never_falls_back(tmp_path):
    config = _v2_config(tmp_path)
    replacement = tmp_path / "replacement.pt"
    save_checkpoint(replacement, _model(),
                    descriptor_stats=DescriptorNormalizer(torch.zeros(15), torch.ones(15)),
                    loss_weights={"lambda_aux": 1.0, "lambda_fingerprint": 1.0},
                    train_config={"normalize": True})
    config["stage1"]["checkpoint"] = str(replacement)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        Stage1Retrieval(config)
    config["stage1"]["graph_index"] = str(tmp_path / "missing.npz")
    with pytest.raises(FileNotFoundError, match="Graph index"):
        Stage1Retrieval(config)


def test_stage1_legacy_mode_stays_compatible_and_rejects_v2_checkpoint(tmp_path):
    config = _legacy_config(tmp_path)
    result = Stage1Retrieval(config).run(np.linspace(-1, 1, 65, dtype=np.float32))
    assert len(result.candidates) == 2
    assert all(candidate.retrieval_mode == "spectral_only_v1" for candidate in result.candidates)
    assert all(candidate.similarity == candidate.embedding_similarity
               for candidate in result.candidates)

    v2_checkpoint = tmp_path / "v2.pt"
    save_checkpoint(v2_checkpoint, _model(),
                    descriptor_stats=DescriptorNormalizer(torch.zeros(15), torch.ones(15)))
    config["stage1"]["checkpoint"] = str(v2_checkpoint)
    with pytest.raises(ValueError, match="legacy.*v2|v2.*legacy"):
        Stage1Retrieval(config)


def test_pipeline_config_defaults_to_v2_and_rejects_unknown_retrieval_mode():
    assert PipelineConfig.defaults().get("stage1", "retrieval_mode") == "spectrum_graph_v2"
    default_checkpoint = PipelineConfig.defaults().get("stage1", "checkpoint")
    assert default_checkpoint.endswith(
        "outputs/structure_aware/best.pt"
    )
    assert Path(default_checkpoint).parents[2] == Path(__file__).resolve().parents[2]
    assert PipelineConfig.defaults().get("stage1", "graph_index").endswith(
        "outputs/structure_aware/graph_index.npz"
    )
    config = PipelineConfig.from_dict({"stage1": {"retrieval_mode": "auto"}})
    assert any("retrieval_mode" in warning for warning in config.warnings)
    assert Candidate(1, 1, "CC", 0.5).retrieval_mode == "spectral_only_v1"
    assert Candidate(1, 1, "CC", 0.5).embedding_similarity is None
    nonfinite = PipelineConfig.from_dict(
        {"stage1": {"rerank": {"fingerprint_weight": float("nan")}}}
    )
    assert any("fingerprint_weight" in warning for warning in nonfinite.warnings)
    string_false = PipelineConfig.from_dict(
        {"stage1": {"normalize_spectrum": "false"}}
    )
    assert any("normalize_spectrum" in warning for warning in string_false.warnings)

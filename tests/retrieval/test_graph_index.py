"""
功能概述：验证 Task 8 图候选索引模块能够作为独立、版本化的检索边界导入。
主要职责：首个 TDD 用例锁定 ``spectraexpert.retrieval.graph_index`` 模块必须存在，后续用例将在
同一文件覆盖索引持久化、严格 metadata 校验、稳定检索与可选光谱预测辅助重排。
代码库关系：目标模块消费 Task 2 结构特征 schema、Task 6 图缓存与 Task 7 v2 checkpoint；测试仅
使用临时小数组和小模型，不读取项目真实数据，也不启动训练或网络访问。
输入输出与副作用：只在临时目录写小产物；autouse fixture 固定并恢复 Python/NumPy/Torch RNG，
避免随机小模型推进全局状态、影响其他测试；不访问真实数据或外部服务。
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from spectraexpert.data.molecular_graph import collate_graphs, featurize_smiles
from spectraexpert.data.structure_dataset import canonical_smiles_sha256
from spectraexpert.data.structure_features import DescriptorNormalizer, StructureFeatureConfig
from spectraexpert.models import GraphEncoder, StructureAwareRetriever
from spectraexpert.models.spectrum_encoder import SpectrumTransformer
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


def test_graph_index_module_is_publicly_importable():
    """删除 Task 8 模块会破坏公开导入边界。"""

    assert importlib.util.find_spec("spectraexpert.retrieval.graph_index") is not None


def test_graph_index_module_exports_versioned_index_contract():
    """删除任何公开索引类型会使 Stage 1 与构建 CLI 无法共享同一契约。"""

    module = importlib.import_module("spectraexpert.retrieval.graph_index")
    assert hasattr(module, "GraphIndex")
    assert hasattr(module, "GraphSearchResult")
    assert hasattr(module, "RerankConfig")


def _model(*, predict_fingerprint: bool = True) -> StructureAwareRetriever:
    return StructureAwareRetriever(
        SpectrumTransformer(
            signal_length=65,
            patch_size=8,
            stride=4,
            embed_dim=16,
            depth=1,
            num_heads=4,
            projection_dim=8,
        ),
        GraphEncoder(hidden_dim=16, projection_dim=8),
        descriptor_dim=15,
        predict_fingerprint=predict_fingerprint,
    ).eval()


def _checkpoint(tmp_path: Path, *, predict_fingerprint: bool = True) -> tuple[Path, StructureAwareRetriever]:
    model = _model(predict_fingerprint=predict_fingerprint)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model,
        descriptor_stats=DescriptorNormalizer(torch.arange(15).float(), torch.full((15,), 2.0)),
        source_sha256={"smiles": "source-file"},
        source_smiles_sha256="source-smiles",
        train_config={"normalize": True},
    )
    return path, model


def _index_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.8, 0.2]], dtype=np.float32)
    mol_ids = np.asarray([1, 2, 3], dtype=np.int64)
    smiles = np.asarray(["CC", "CO", "CN"])
    fingerprints = np.zeros((3, 2048), dtype=np.uint8)
    fingerprints[0, [0, 2]] = 1
    fingerprints[1, [0, 1]] = 1
    fingerprints[2, [3, 4]] = 1
    descriptors = np.zeros((3, 15), dtype=np.float32)
    descriptors[:, :2] = [[2.0, 0.0], [1.0, 1.0], [3.0, 3.0]]
    return embeddings, mol_ids, smiles, fingerprints, descriptors


def _index(**overrides):
    from spectraexpert.retrieval import GraphIndex

    embeddings, mol_ids, smiles, fingerprints, descriptors = _index_arrays()
    values = {
        "embeddings": embeddings,
        "mol_ids": mol_ids,
        "smiles": smiles,
        "fingerprints": fingerprints,
        "descriptors": descriptors,
        "checkpoint_sha256": "a" * 64,
        "feature_metadata": StructureFeatureConfig().to_metadata(),
        "model_config": {
            "signal_length": 65,
            "patch_size": 8,
            "stride": 4,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 4,
            "projection_dim": 2,
            "graph_hidden_dim": 16,
            "descriptor_dim": 15,
            "predict_fingerprint": True,
        },
        "source_smiles_sha256": canonical_smiles_sha256(mol_ids.tolist(), smiles.tolist()),
        "descriptor_mean": np.zeros(15, dtype=np.float32),
        "descriptor_std": np.ones(15, dtype=np.float32),
    }
    values.update(overrides)
    return GraphIndex.from_arrays(**values)


def test_graph_index_round_trip_uses_non_object_arrays_and_validates_all_metadata(tmp_path):
    from spectraexpert.retrieval import GraphIndex

    index = _index()
    path = tmp_path / "index.npz"
    index.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "embeddings", "mol_ids", "smiles", "fingerprints_packed", "descriptors", "metadata"
        }
        assert all(archive[name].dtype != object for name in archive.files)
    loaded = GraphIndex.load(
        path,
        expected_checkpoint_sha256="a" * 64,
        expected_feature_metadata=StructureFeatureConfig().to_metadata(),
        expected_model_config=index.metadata["model_config"],
        expected_source_smiles_sha256=index.metadata["source_smiles_sha256"],
        expected_descriptor_normalizer=index.metadata["descriptor_normalizer"],
    )
    np.testing.assert_allclose(loaded.embeddings, index.embeddings)
    np.testing.assert_array_equal(loaded.mol_ids, index.mol_ids)
    np.testing.assert_array_equal(loaded.smiles, index.smiles)
    np.testing.assert_array_equal(loaded.fingerprints, index.fingerprints)
    np.testing.assert_allclose(loaded.descriptors, index.descriptors)
    assert loaded.metadata == index.metadata

    mismatch_cases = {
        "expected_checkpoint_sha256": ("b" * 64, "checkpoint_sha256"),
        "expected_feature_metadata": (
            StructureFeatureConfig(morgan_radius=3).to_metadata(), "feature_config"
        ),
        "expected_model_config": (
            {**index.metadata["model_config"], "projection_dim": 7}, "model_config"
        ),
        "expected_source_smiles_sha256": ("c" * 64, "source_smiles_sha256"),
        "expected_descriptor_normalizer": (
            {"mean": [2.0] * 15, "std": [1.0] * 15}, "descriptor_normalizer"
        ),
    }
    for argument, (wrong, message) in mismatch_cases.items():
        with pytest.raises(ValueError, match=message):
            GraphIndex.load(path, **{argument: wrong})


def test_graph_index_rejects_duplicate_id_and_canonical_identity():
    _, mol_ids, smiles, _, _ = _index_arrays()
    with pytest.raises(ValueError, match="duplicate mol_id"):
        _index(mol_ids=np.asarray([1, 1, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="duplicate canonical SMILES"):
        _index(smiles=np.asarray(["CC", "CO", "CC"]))
    with pytest.raises(ValueError, match="canonical SMILES"):
        _index(smiles=np.asarray(["C(C)", "CO", "CN"]))
    assert canonical_smiles_sha256(mol_ids.tolist(), smiles.tolist())


def test_graph_index_rejects_unknown_model_fields_and_non_exact_feature_types():
    model_config = dict(_index().metadata["model_config"])
    model_config["future_field"] = 1
    with pytest.raises(ValueError, match="model_config.*unexpected"):
        _index(model_config=model_config)
    feature = StructureFeatureConfig().to_metadata()
    feature["morgan_radius"] = True
    with pytest.raises(ValueError, match="feature_config.*morgan_radius"):
        _index(feature_metadata=feature)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("embeddings", np.ones((3, 3), dtype=np.float32), "projection_dim"),
        ("embeddings", np.asarray([[1.0, 0.0], [np.nan, 0.0], [1.0, 0.0]]), "finite"),
        ("embeddings", np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), "zero"),
        ("fingerprints", np.zeros((3, 8), dtype=np.uint8), "fingerprint"),
        ("fingerprints", np.full((3, 2048), 2, dtype=np.uint8), "binary"),
        ("descriptors", np.zeros((3, 14), dtype=np.float32), "descriptor"),
        ("descriptors", np.full((3, 15), np.inf, dtype=np.float32), "finite"),
    ],
)
def test_graph_index_array_validation_fails_before_persistence(field, value, message):
    with pytest.raises(ValueError, match=message):
        _index(**{field: value})


def test_cosine_search_is_stable_finite_float32_and_caps_topk():
    index = _index()
    result = index.search(np.asarray([0.0, 1.0], dtype=np.float32), topk=99)
    assert result.indices.tolist() == [1, 2, 0]
    assert result.indices.dtype == np.int64
    assert result.scores.dtype == result.embedding_similarities.dtype == np.float32
    assert np.isfinite(result.scores).all()
    assert result.metadata == {"query_structure_source": "none", "reranked": False}
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones(3, dtype=np.float32), topk=1)
    with pytest.raises(ValueError, match="finite"):
        index.search(np.asarray([np.nan, 1.0], dtype=np.float32), topk=1)
    with pytest.raises(ValueError, match="zero"):
        index.search(np.zeros(2, dtype=np.float32), topk=1)
    with pytest.raises(ValueError, match="positive integer"):
        index.search(np.ones(2, dtype=np.float32), topk=0)


def test_optional_rerank_uses_spectrum_predictions_and_exact_formula():
    from spectraexpert.retrieval import RerankConfig

    index = _index()
    predicted_fingerprint = np.zeros(2048, dtype=np.float32)
    predicted_fingerprint[[0, 1]] = 1.0
    predicted_descriptors = np.zeros(15, dtype=np.float32)
    predicted_descriptors[:2] = [1.0, 1.0]
    result = index.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        topk=2,
        rerank=RerankConfig(
            top_m=3, fingerprint_weight=1.0, descriptor_weight=0.3
        ),
        predicted_fingerprint=predicted_fingerprint,
        predicted_descriptors=predicted_descriptors,
    )
    assert result.indices.tolist()[0] == 1
    cosine = float(np.dot(index.embeddings[1], np.asarray([1.0, 0.0], dtype=np.float32)))
    expected = cosine + 1.0 - 0.0
    assert result.scores[0] == pytest.approx(expected, abs=1e-6)
    assert result.embedding_similarities[0] == pytest.approx(cosine, abs=1e-6)
    assert result.metadata == {
        "query_structure_source": "predicted_from_spectrum",
        "reranked": True,
    }


def test_zero_weight_rerank_needs_no_auxiliary_head_but_positive_weights_do():
    from spectraexpert.retrieval import RerankConfig

    index = _index(
        model_config={**_index().metadata["model_config"], "predict_fingerprint": False}
    )
    zero = index.search(
        np.asarray([0.0, 1.0], dtype=np.float32),
        topk=2,
        rerank=RerankConfig(enabled=True, top_m=2),
    )
    assert zero.indices.tolist() == [1, 2]
    assert zero.metadata["query_structure_source"] == "none"
    with pytest.raises(ValueError, match="fingerprint head"):
        index.search(
            np.asarray([0.0, 1.0], dtype=np.float32),
            topk=1,
            rerank=RerankConfig(enabled=True, top_m=2, fingerprint_weight=0.1),
            predicted_fingerprint=np.zeros(2048, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="predicted_descriptors"):
        index.search(
            np.asarray([0.0, 1.0], dtype=np.float32),
            topk=1,
            rerank=RerankConfig(enabled=True, top_m=2, descriptor_weight=0.1),
        )


def test_corrupt_or_future_index_metadata_is_rejected_without_pickle(tmp_path):
    from spectraexpert.retrieval import GraphIndex

    path = tmp_path / "index.npz"
    _index().save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(arrays["metadata"].tobytes().decode("utf-8"))
    metadata["schema_version"] = "legacy"
    arrays["metadata"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="schema_version"):
        GraphIndex.load(path)


def test_index_claiming_normalized_rejects_non_unit_embedding_rows(tmp_path):
    from spectraexpert.retrieval import GraphIndex

    path = tmp_path / "index.npz"
    _index().save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["embeddings"][0] *= 0.5
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="normalized|unit"):
        GraphIndex.load(path)


def test_atomic_save_preserves_existing_index_on_write_failure(tmp_path, monkeypatch):
    import spectraexpert.retrieval.graph_index as graph_index

    path = tmp_path / "index.npz"
    path.write_bytes(b"existing")

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(graph_index.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk full"):
        _index().save(path)
    assert path.read_bytes() == b"existing"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["index.npz"]


def test_structure_aware_encoder_restores_one_snapshot_and_accepts_only_spectrum(tmp_path):
    from spectraexpert.inference import StructureAwareSpectralEncoder

    checkpoint, model = _checkpoint(tmp_path)
    encoder = StructureAwareSpectralEncoder.from_checkpoint(checkpoint, device="cpu")
    query = np.random.default_rng(0).normal(size=65).astype(np.float32)
    actual = encoder.encode_spectrum(query)
    with torch.inference_mode():
        expected = model.encode_spectrum(torch.from_numpy(query).unsqueeze(0)).squeeze(0).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert list(inspect.signature(encoder.encode_spectrum).parameters) == ["spectrum"]
    assert encoder.checkpoint_sha256 == __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    assert encoder.model_config["signal_length"] == 65
    descriptor, fingerprint = encoder.predict_auxiliary(query)
    assert descriptor.shape == (15,)
    assert fingerprint is not None and fingerprint.shape == (2048,)
    with pytest.raises(ValueError, match="signal length"):
        encoder.encode_spectrum(np.ones(3501, dtype=np.float32))
    with pytest.raises(ValueError, match="one-dimensional"):
        encoder.encode_spectrum(np.ones((1, 65), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        encoder.encode_spectrum(np.full(65, np.nan, dtype=np.float32))


class _GraphDataset(Dataset):
    def __init__(self):
        self.graphs = [featurize_smiles("CC", 1), featurize_smiles("CO", 2)]
        self.feature_config = StructureFeatureConfig()
        self.source_smiles_sha256 = canonical_smiles_sha256([1, 2], ["CC", "CO"])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def test_build_graph_index_encodes_graph_batches_and_binds_checkpoint_metadata(tmp_path):
    from spectraexpert.retrieval.graph_index import GraphIndex, build_graph_index

    checkpoint, model = _checkpoint(tmp_path)
    loader = DataLoader(_GraphDataset(), batch_size=1, shuffle=False, collate_fn=collate_graphs)
    output = tmp_path / "graphs.npz"
    index = build_graph_index(model, loader, checkpoint, output)
    loaded = GraphIndex.load(
        output,
        expected_checkpoint_sha256=__import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        expected_feature_metadata=StructureFeatureConfig().to_metadata(),
        expected_model_config=model.to_config(),
    )
    with torch.inference_mode():
        expected = model.encode_graph(collate_graphs(_GraphDataset().graphs)).numpy()
    np.testing.assert_allclose(index.embeddings, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(loaded.embeddings, expected, rtol=1e-6, atol=1e-6)
    assert loaded.metadata["capabilities"] == {
        "descriptor_normalizer": True,
        "descriptor_rerank": True,
        "fingerprint_rerank": True,
    }


def test_from_encoded_batches_accepts_brief_checkpoint_and_dataset_contract(tmp_path):
    from spectraexpert.retrieval import GraphIndex

    checkpoint, model = _checkpoint(tmp_path)
    dataset = _GraphDataset()
    with torch.inference_mode():
        embeddings = model.encode_graph(collate_graphs(dataset.graphs)).numpy()
    index = GraphIndex.from_encoded_batches(
        embedding_batches=[embeddings[:1], embeddings[1:]],
        mol_id_batches=[np.asarray([1], dtype=np.int64), np.asarray([2], dtype=np.int64)],
        smiles=["CC", "CO"],
        checkpoint_path=checkpoint,
        dataset=dataset,
    )
    assert index.mol_ids.tolist() == [1, 2]
    assert index.metadata["model_config"] == model.to_config()
    assert index.metadata["source_smiles_sha256"] == dataset.source_smiles_sha256


def test_build_graph_index_cli_helper_consumes_explicit_smiles_and_feature_cache(tmp_path):
    from scripts.build_graph_index import build_index_from_paths
    from scripts.cache_molecular_features import build_feature_cache
    from spectraexpert.retrieval import GraphIndex

    checkpoint, model = _checkpoint(tmp_path)
    smiles_path = tmp_path / "candidates.parquet"
    pd.DataFrame({"mol_id": [4, 5], "smiles_canonical": ["CC", "CO"]}).to_parquet(
        smiles_path, index=False
    )
    feature_cache = tmp_path / "features.parquet"
    build_feature_cache(smiles_path, feature_cache, batch_size=1)
    output = tmp_path / "index.npz"

    built = build_index_from_paths(
        checkpoint_path=checkpoint,
        smiles_path=smiles_path,
        feature_cache_path=feature_cache,
        output_path=output,
        batch_size=1,
        device="cpu",
    )
    loaded = GraphIndex.load(
        output,
        expected_checkpoint_sha256=__import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        expected_model_config=model.to_config(),
        expected_feature_metadata=StructureFeatureConfig().to_metadata(),
    )
    assert built.mol_ids.tolist() == loaded.mol_ids.tolist() == [4, 5]
    assert loaded.smiles.tolist() == ["CC", "CO"]
    with pytest.raises(FileExistsError, match="already exists"):
        build_index_from_paths(checkpoint, smiles_path, feature_cache, output, batch_size=1)


def test_build_graph_index_cli_helper_rejects_smiles_cache_source_mismatch(tmp_path):
    from scripts.build_graph_index import build_index_from_paths
    from scripts.cache_molecular_features import build_feature_cache

    checkpoint, _ = _checkpoint(tmp_path)
    cached_smiles = tmp_path / "cached.parquet"
    requested_smiles = tmp_path / "requested.parquet"
    pd.DataFrame({"mol_id": [1], "smiles_canonical": ["CC"]}).to_parquet(cached_smiles, index=False)
    pd.DataFrame({"mol_id": [1], "smiles_canonical": ["CO"]}).to_parquet(requested_smiles, index=False)
    feature_cache = tmp_path / "features.parquet"
    build_feature_cache(cached_smiles, feature_cache)
    with pytest.raises(ValueError, match="source_smiles_sha256"):
        build_index_from_paths(
            checkpoint, requested_smiles, feature_cache, tmp_path / "index.npz", batch_size=1
        )
    assert not (tmp_path / "index.npz").exists()

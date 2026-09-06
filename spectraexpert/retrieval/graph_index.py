"""
功能概述：定义 Task 8 唯一的 ``spectrum_graph_index_v2`` 候选索引格式，并提供稳定 cosine
检索与只使用光谱辅助头预测的可选重排；候选 embedding 只能由离线分子图编码得到。
主要职责：``GraphIndex.from_arrays`` 校验并归一化图 embedding、唯一 mol_id/canonical SMILES、
Morgan 位向量、原始 descriptors 及完整 checkpoint/feature/model/source metadata；``save/load``
用无 object array、``allow_pickle=False`` 的单一 NPZ 原子持久化并逐字段复核；``search`` 先稳定
选 cosine Top-M，再按 ``cosine + wf*soft_tanimoto - wd*descriptor_mae`` 稳定重排；
``build_graph_index`` 分批迁移 ``GraphBatch`` 到模型设备，用 checkpoint 对应 graph encoder 编码。
代码库关系：复用 Task 2 ``StructureFeatureConfig`` 和 canonical identity hash、Task 6 GraphBatch/cache
字段、Task 7 checkpoint snapshot/严格 loader；由 ``scripts/build_graph_index.py`` 与 Stage 1 v2 模式
消费，legacy embedding cache 不经过本模块，因而不能和 v2 schema 混用。
输入输出与副作用：构造和查询只操作有限、形状严格的 NumPy 数组；指纹保存为 little-endian packed
uint8，metadata 保存为 canonical JSON UTF-8 bytes。写文件使用同目录临时文件、fsync 和 replace，
失败不覆盖既有目标；加载前检查 ZIP 原始 member 清单，并先校验 metadata/调用方期望再物化大数组。
重排能力同时绑定物理 head 与 checkpoint 正有限 loss 权重。重排只接受调用方从 query spectrum
得到的预测，不接收 query SMILES、真实
fingerprint、分子式或 descriptors；零权重不要求相应辅助 head，正权重缺能力或预测时 fail fast。
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from spectraexpert.data.molecular_graph import GraphBatch
from spectraexpert.data.structure_dataset import canonical_smiles_sha256
from spectraexpert.data.structure_features import StructureFeatureConfig
from spectraexpert.models.multimodal_retriever import StructureAwareRetriever
from spectraexpert.training.checkpoints import load_checkpoint, read_checkpoint_snapshot


INDEX_SCHEMA_VERSION = "spectrum_graph_index_v2"
_METADATA_FIELDS = {
    "schema_version", "checkpoint_sha256", "feature_config", "model_config",
    "source_smiles_sha256", "canonical_smiles_sha256", "embedding_dim", "normalized",
    "record_count", "fingerprint_bits", "descriptor_dim", "descriptor_normalizer", "loss_weights",
    "capabilities",
}
_CAPABILITY_FIELDS = {"descriptor_normalizer", "descriptor_rerank", "fingerprint_rerank"}
_MINIMAL_MODEL_FIELDS = {"projection_dim", "descriptor_dim", "predict_fingerprint"}
_V2_MODEL_FIELDS = {
    "signal_length", "patch_size", "stride", "embed_dim", "depth", "num_heads",
    "projection_dim", "graph_hidden_dim", "descriptor_dim", "predict_fingerprint",
}
_RERANK_ENABLED_UNSET = object()


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("index metadata must be finite JSON data") from error


def _strict_json(data: bytes) -> object:
    try:
        return json.loads(data.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("corrupt graph index metadata") from error


def _validate_feature_metadata(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("feature_config must be a mapping")
    metadata = dict(value)
    if set(metadata) != set(StructureFeatureConfig().to_metadata()):
        raise ValueError("feature_config has missing or unexpected fields")
    try:
        config = StructureFeatureConfig(**metadata)
    except (TypeError, ValueError) as error:
        raise ValueError(f"feature_config is invalid: {error}") from error
    if config.to_metadata() != metadata:
        raise ValueError("feature_config failed strict roundtrip")
    if config.schema_version != "metric_aligned_graph_v1":
        raise ValueError("feature_config.schema_version is unsupported")
    if type(config.morgan_radius) is not int or config.morgan_radius <= 0:
        raise ValueError("feature_config.morgan_radius must be a positive integer")
    if type(config.morgan_bits) is not int or config.morgan_bits <= 0:
        raise ValueError("feature_config.morgan_bits must be a positive integer")
    if (type(config.schema_version) is not str or type(config.use_bond_types) is not bool
            or type(config.use_chirality) is not bool):
        raise ValueError("feature_config boolean fields are invalid")
    return metadata


def _validate_model_config(value: Mapping[str, object], embedding_dim: int,
                           descriptor_dim: int) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("model_config must be a mapping")
    config = dict(value)
    if set(config) not in (_MINIMAL_MODEL_FIELDS, _V2_MODEL_FIELDS):
        raise ValueError("model_config has missing or unexpected fields")
    for field in set(config) - {"predict_fingerprint"}:
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"model_config.{field} must be a positive integer")
    if type(config["projection_dim"]) is not int or config["projection_dim"] != embedding_dim:
        raise ValueError("model_config.projection_dim does not match embeddings")
    if type(config["descriptor_dim"]) is not int or config["descriptor_dim"] != descriptor_dim:
        raise ValueError("model_config.descriptor_dim does not match descriptors")
    if type(config["predict_fingerprint"]) is not bool:
        raise ValueError("model_config.predict_fingerprint must be bool")
    _canonical_json(config)
    return config


def _validate_loss_weights(value: Mapping[str, object]) -> dict[str, float]:
    """验证 checkpoint objective 权重，并保留全部有限非负审计字段。"""
    if not isinstance(value, Mapping):
        raise ValueError("loss_weights must be a mapping")
    weights = dict(value)
    if not {"lambda_aux", "lambda_fingerprint"}.issubset(weights):
        raise ValueError("loss_weights requires lambda_aux and lambda_fingerprint")
    normalized: dict[str, float] = {}
    for name, raw in weights.items():
        if type(name) is not str or isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, float, np.number)
        ):
            raise ValueError(f"loss_weights.{name} must be a finite nonnegative number")
        if np.iscomplexobj(raw):
            raise ValueError(f"loss_weights.{name} must be a finite nonnegative number")
        value_float = float(raw)
        if not np.isfinite(value_float) or value_float < 0:
            raise ValueError(f"loss_weights.{name} must be a finite nonnegative number")
        normalized[name] = value_float
    return normalized


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, init=False)
class RerankConfig:
    """控制 cosine Top-M 重排；省略 enabled 时仅由非零权重启用，显式 False 始终关闭。"""

    enabled: bool = False
    top_m: int = 100
    fingerprint_weight: float = 0.0
    descriptor_weight: float = 0.0

    def __init__(self, enabled: bool | object = _RERANK_ENABLED_UNSET, top_m: int = 100,
                 fingerprint_weight: float = 0.0, descriptor_weight: float = 0.0) -> None:
        weights = (fingerprint_weight, descriptor_weight)
        inferred = any(
            isinstance(value, (int, float, np.number))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value) and value > 0
            for value in weights
        )
        resolved = inferred if enabled is _RERANK_ENABLED_UNSET else enabled
        object.__setattr__(self, "enabled", resolved)
        object.__setattr__(self, "top_m", top_m)
        object.__setattr__(self, "fingerprint_weight", fingerprint_weight)
        object.__setattr__(self, "descriptor_weight", descriptor_weight)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("rerank.enabled must be bool")
        _positive_integer(self.top_m, "rerank.top_m")
        for name in ("fingerprint_weight", "descriptor_weight"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"rerank.{name} must be a finite nonnegative number")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"rerank.{name} must be a finite nonnegative number")


@dataclass(frozen=True)
class GraphSearchResult:
    """保存稳定排序的候选下标、最终分数、原 cosine 分数与 query provenance。"""

    indices: np.ndarray
    scores: np.ndarray
    embedding_similarities: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class GraphIndex:
    """内存中的只读 v2 图候选索引及其完整兼容性 metadata。"""

    embeddings: np.ndarray
    mol_ids: np.ndarray
    smiles: np.ndarray
    fingerprints: np.ndarray
    descriptors: np.ndarray
    metadata: dict[str, object]

    @classmethod
    def from_arrays(cls, embeddings: np.ndarray, mol_ids: np.ndarray,
                    smiles: Sequence[str] | np.ndarray, fingerprints: np.ndarray,
                    descriptors: np.ndarray, *, checkpoint_sha256: str,
                    feature_metadata: Mapping[str, object],
                    model_config: Mapping[str, object] | None = None,
                    source_smiles_sha256: str | None = None,
                    descriptor_mean: np.ndarray | Sequence[float] | None = None,
                    descriptor_std: np.ndarray | Sequence[float] | None = None,
                    loss_weights: Mapping[str, object] | None = None) -> "GraphIndex":
        """验证候选全集并将图 embedding 归一化为可稳定点积检索的只读数组。"""

        raw_embeddings = np.asarray(embeddings)
        if raw_embeddings.ndim != 2 or not raw_embeddings.shape[0] or not raw_embeddings.shape[1]:
            raise ValueError("embeddings must be a non-empty two-dimensional array")
        try:
            embedding_array = raw_embeddings.astype(np.float32, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("embeddings must be numeric") from error
        if not np.isfinite(embedding_array).all():
            raise ValueError("embeddings must contain only finite values")
        norms = np.linalg.norm(embedding_array, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("embeddings must not contain zero-norm rows")
        embedding_array /= norms[:, None]

        count, embedding_dim = embedding_array.shape
        raw_ids = np.asarray(mol_ids)
        if raw_ids.ndim != 1 or raw_ids.shape[0] != count:
            raise ValueError("mol_ids must have one value per embedding")
        if raw_ids.dtype.kind not in "iu" or (
            raw_ids.dtype.kind == "u" and raw_ids.size and raw_ids.max() > np.iinfo(np.int64).max
        ):
            raise ValueError("mol_ids must be int64-compatible integers")
        id_array = raw_ids.astype(np.int64, copy=True)
        if np.unique(id_array).size != count:
            raise ValueError("duplicate mol_id in graph index")

        raw_smiles = np.asarray(smiles)
        if raw_smiles.ndim != 1 or raw_smiles.shape[0] != count or raw_smiles.dtype.kind not in "US":
            raise ValueError("canonical SMILES must be a one-dimensional string array")
        smiles_values = [str(value) for value in raw_smiles.tolist()]
        from spectraexpert.data.structure_dataset import canonicalize_smiles
        canonical = [canonicalize_smiles(value, mol_id=int(identifier))
                     for identifier, value in zip(id_array, smiles_values)]
        if canonical != smiles_values:
            raise ValueError("canonical SMILES array contains non-canonical values")
        if len(set(canonical)) != count:
            raise ValueError("duplicate canonical SMILES in graph index")
        smiles_array = np.asarray(canonical, dtype=f"<U{max(len(value) for value in canonical)}")

        feature = _validate_feature_metadata(feature_metadata)
        raw_fingerprints = np.asarray(fingerprints)
        if raw_fingerprints.shape != (count, int(feature["morgan_bits"])):
            raise ValueError("fingerprint shape does not match feature_config")
        if raw_fingerprints.dtype.kind not in "biu" or not np.isin(raw_fingerprints, (0, 1)).all():
            raise ValueError("fingerprints must contain binary values")
        fingerprint_array = raw_fingerprints.astype(np.uint8, copy=True)

        raw_descriptors = np.asarray(descriptors)
        if raw_descriptors.ndim != 2 or raw_descriptors.shape[0] != count or not raw_descriptors.shape[1]:
            raise ValueError("descriptors must be a non-empty two-dimensional array")
        try:
            descriptor_array = raw_descriptors.astype(np.float32, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("descriptors must be numeric") from error
        if not np.isfinite(descriptor_array).all():
            raise ValueError("descriptors must contain only finite values")
        descriptor_dim = descriptor_array.shape[1]

        if model_config is None:
            model_config = {"projection_dim": embedding_dim, "descriptor_dim": descriptor_dim,
                            "predict_fingerprint": True}
        model = _validate_model_config(model_config, embedding_dim, descriptor_dim)
        if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
            raise ValueError("checkpoint_sha256 must be a non-empty string")
        identity_hash = canonical_smiles_sha256(id_array.tolist(), canonical)
        if source_smiles_sha256 is None:
            source_smiles_sha256 = identity_hash
        if not isinstance(source_smiles_sha256, str) or source_smiles_sha256 != identity_hash:
            raise ValueError("source_smiles_sha256 does not match canonical candidate records")

        mean = (np.zeros(descriptor_dim, dtype=np.float32) if descriptor_mean is None
                else np.asarray(descriptor_mean, dtype=np.float32))
        std = (np.ones(descriptor_dim, dtype=np.float32) if descriptor_std is None
               else np.asarray(descriptor_std, dtype=np.float32))
        if mean.shape != (descriptor_dim,) or std.shape != (descriptor_dim,):
            raise ValueError("descriptor_normalizer shape does not match descriptors")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("descriptor_normalizer must contain finite mean and positive std")
        weights = _validate_loss_weights(loss_weights or {})
        capabilities = {
            "descriptor_normalizer": True,
            "descriptor_rerank": weights["lambda_aux"] > 0,
            "fingerprint_rerank": (
                bool(model["predict_fingerprint"]) and weights["lambda_fingerprint"] > 0
            ),
        }
        metadata = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "feature_config": feature,
            "model_config": model,
            "source_smiles_sha256": source_smiles_sha256,
            "canonical_smiles_sha256": identity_hash,
            "embedding_dim": embedding_dim,
            "normalized": True,
            "record_count": count,
            "fingerprint_bits": fingerprint_array.shape[1],
            "descriptor_dim": descriptor_dim,
            "descriptor_normalizer": {"mean": mean.astype(float).tolist(),
                                      "std": std.astype(float).tolist()},
            "loss_weights": weights,
            "capabilities": capabilities,
        }
        _canonical_json(metadata)
        return cls(_readonly(embedding_array), _readonly(id_array), _readonly(smiles_array),
                   _readonly(fingerprint_array), _readonly(descriptor_array), metadata)

    @classmethod
    def from_encoded_batches(cls, *, embedding_batches: Iterable[np.ndarray],
                             mol_id_batches: Iterable[np.ndarray], smiles: Sequence[str],
                             fingerprint_batches: Iterable[np.ndarray] | None = None,
                             descriptor_batches: Iterable[np.ndarray] | None = None,
                             checkpoint_sha256: str | None = None,
                             feature_metadata: Mapping[str, object] | None = None,
                             model_config: Mapping[str, object] | None = None,
                             source_smiles_sha256: str | None = None,
                             descriptor_mean: np.ndarray | None = None,
                             descriptor_std: np.ndarray | None = None,
                             loss_weights: Mapping[str, object] | None = None,
                             checkpoint_path: str | Path | None = None,
                             dataset: object | None = None) -> "GraphIndex":
        """合并编码 batch；可直接消费完整数组 metadata，或按 brief 从 checkpoint/dataset 恢复。"""
        try:
            embeddings = np.concatenate(list(embedding_batches), axis=0)
            mol_ids = np.concatenate(list(mol_id_batches), axis=0)
        except ValueError as error:
            raise ValueError("encoded graph batches are empty or inconsistent") from error
        if checkpoint_path is not None:
            if dataset is None or any(value is not None for value in (
                fingerprint_batches, descriptor_batches, checkpoint_sha256, feature_metadata,
                model_config, source_smiles_sha256, descriptor_mean, descriptor_std, loss_weights,
            )):
                raise ValueError("checkpoint_path mode requires only dataset plus encoded batches")
            snapshot = read_checkpoint_snapshot(checkpoint_path)
            _, payload = load_checkpoint(snapshot, expected_mode="spectrum_graph_v2")
            graphs = getattr(dataset, "graphs", None)
            if not isinstance(graphs, (list, tuple)) or len(graphs) != len(mol_ids):
                raise ValueError("dataset.graphs must align one-to-one with encoded batches")
            for offset, (graph, identifier, identity) in enumerate(zip(graphs, mol_ids, smiles)):
                if (not hasattr(graph, "fingerprint") or not hasattr(graph, "descriptors")
                        or int(graph.mol_id) != int(identifier)
                        or graph.canonical_smiles != identity):
                    raise ValueError(f"dataset graph alignment mismatch at row {offset}")
            fingerprints = np.stack([graph.fingerprint.detach().cpu().numpy() for graph in graphs])
            descriptors = np.stack([graph.descriptors.detach().cpu().numpy() for graph in graphs])
            checkpoint_sha256 = snapshot.sha256
            feature_metadata = payload["feature_config"]
            model_config = payload["model_config"]
            source_smiles_sha256 = canonical_smiles_sha256(mol_ids.tolist(), smiles)
            dataset_source = getattr(dataset, "source_smiles_sha256", None)
            if type(dataset_source) is not str:
                raise ValueError("source_smiles_sha256: dataset provenance is required")
            if dataset_source != source_smiles_sha256:
                raise ValueError("source_smiles_sha256: dataset records mismatch")
            dataset_feature = getattr(dataset, "feature_config", None)
            if type(dataset_feature) is not StructureFeatureConfig:
                raise ValueError("feature_config: verified dataset provenance is required")
            if dataset_feature.to_metadata() != feature_metadata:
                raise ValueError("feature_config: dataset/checkpoint mismatch")
            descriptor_mean = payload["descriptor_stats"]["mean"].detach().cpu().numpy()
            descriptor_std = payload["descriptor_stats"]["std"].detach().cpu().numpy()
            loss_weights = payload["loss_weights"]
        else:
            required = (fingerprint_batches, descriptor_batches, checkpoint_sha256,
                        feature_metadata, model_config, source_smiles_sha256,
                        descriptor_mean, descriptor_std, loss_weights)
            if any(value is None for value in required):
                raise ValueError("encoded batches require complete index metadata")
            try:
                fingerprints = np.concatenate(list(fingerprint_batches), axis=0)
                descriptors = np.concatenate(list(descriptor_batches), axis=0)
            except ValueError as error:
                raise ValueError("encoded graph batches are empty or inconsistent") from error
        return cls.from_arrays(embeddings, mol_ids, smiles, fingerprints, descriptors,
            checkpoint_sha256=checkpoint_sha256, feature_metadata=feature_metadata,
            model_config=model_config, source_smiles_sha256=source_smiles_sha256,
            descriptor_mean=descriptor_mean, descriptor_std=descriptor_std,
            loss_weights=loss_weights)

    def save(self, path: str | Path) -> Path:
        """用无 pickle 单 NPZ 原子发布索引；任意写入失败保留既有目标字节。"""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                                  dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, embeddings=self.embeddings, mol_ids=self.mol_ids,
                    smiles=self.smiles,
                    fingerprints_packed=np.packbits(self.fingerprints, axis=1, bitorder="little"),
                    descriptors=self.descriptors,
                    metadata=np.frombuffer(_canonical_json(self.metadata), dtype=np.uint8))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return destination

    @classmethod
    def load(cls, path: str | Path, *, expected_checkpoint_sha256: str | None = None,
             expected_feature_metadata: Mapping[str, object] | None = None,
             expected_model_config: Mapping[str, object] | None = None,
             expected_source_smiles_sha256: str | None = None,
             expected_descriptor_normalizer: Mapping[str, object] | None = None,
             expected_loss_weights: Mapping[str, object] | None = None) -> "GraphIndex":
        """安全加载 v2 NPZ，并在返回数组前验证 schema、内容 hash 和所有期望 metadata。"""
        source = Path(path)
        required = {"embeddings", "mol_ids", "smiles", "fingerprints_packed",
                    "descriptors", "metadata"}
        expected_members = {f"{name}.npy" for name in required}
        try:
            with zipfile.ZipFile(source, "r") as raw_archive:
                member_names = raw_archive.namelist()
            if (len(member_names) != len(expected_members)
                    or len(set(member_names)) != len(member_names)
                    or set(member_names) != expected_members):
                raise ValueError("graph index NPZ members must be exact and unique")
            with np.load(source, allow_pickle=False) as archive:
                if set(archive.files) != required:
                    raise ValueError("graph index has missing or unexpected arrays")
                metadata_array = archive["metadata"].copy()
                if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
                    raise ValueError("corrupt graph index metadata array")
                metadata = _strict_json(metadata_array.tobytes())
                if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
                    raise ValueError("graph index metadata has missing or unexpected fields")
                if metadata["schema_version"] != INDEX_SCHEMA_VERSION:
                    raise ValueError("schema_version: legacy or future graph index is incompatible")
                capabilities = metadata.get("capabilities")
                if (not isinstance(capabilities, dict) or set(capabilities) != _CAPABILITY_FIELDS
                        or any(type(value) is not bool for value in capabilities.values())):
                    raise ValueError("capabilities metadata is invalid")
                normalizer = metadata.get("descriptor_normalizer")
                if not isinstance(normalizer, dict) or set(normalizer) != {"mean", "std"}:
                    raise ValueError("descriptor_normalizer metadata is invalid")
                weights = _validate_loss_weights(metadata.get("loss_weights"))
                embedding_dim = metadata.get("embedding_dim")
                descriptor_dim = metadata.get("descriptor_dim")
                record_count = metadata.get("record_count")
                fingerprint_bits = metadata.get("fingerprint_bits")
                for field, value in (
                    ("embedding_dim", embedding_dim),
                    ("descriptor_dim", descriptor_dim),
                    ("record_count", record_count),
                    ("fingerprint_bits", fingerprint_bits),
                ):
                    if type(value) is not int or value <= 0:
                        raise ValueError(f"{field} metadata is invalid")
                feature = _validate_feature_metadata(metadata.get("feature_config"))
                model = _validate_model_config(
                    metadata.get("model_config"), embedding_dim, descriptor_dim
                )
                if fingerprint_bits != feature["morgan_bits"]:
                    raise ValueError("fingerprint_bits metadata does not match feature_config")
                if type(metadata.get("checkpoint_sha256")) is not str or not metadata["checkpoint_sha256"]:
                    raise ValueError("checkpoint_sha256 metadata is invalid")
                if type(metadata.get("source_smiles_sha256")) is not str:
                    raise ValueError("source_smiles_sha256 metadata is invalid")
                if type(metadata.get("canonical_smiles_sha256")) is not str:
                    raise ValueError("canonical_smiles_sha256 metadata is invalid")
                if metadata.get("normalized") is not True:
                    raise ValueError("normalized metadata must be true")
                try:
                    mean = np.asarray(normalizer["mean"], dtype=np.float32)
                    std = np.asarray(normalizer["std"], dtype=np.float32)
                except (TypeError, ValueError) as error:
                    raise ValueError("descriptor_normalizer metadata is invalid") from error
                if (mean.shape != (descriptor_dim,) or std.shape != (descriptor_dim,)
                        or not np.isfinite(mean).all() or not np.isfinite(std).all()
                        or np.any(std <= 0)):
                    raise ValueError("descriptor_normalizer metadata is invalid")
                derived_capabilities = {
                    "descriptor_normalizer": True,
                    "descriptor_rerank": weights["lambda_aux"] > 0,
                    "fingerprint_rerank": (
                        model["predict_fingerprint"] and weights["lambda_fingerprint"] > 0
                    ),
                }
                if capabilities != derived_capabilities:
                    raise ValueError("capabilities metadata conflicts with model/loss_weights")
                expected = {
                    "checkpoint_sha256": expected_checkpoint_sha256,
                    "feature_config": (None if expected_feature_metadata is None
                                       else dict(expected_feature_metadata)),
                    "model_config": (None if expected_model_config is None
                                     else dict(expected_model_config)),
                    "source_smiles_sha256": expected_source_smiles_sha256,
                    "descriptor_normalizer": (
                        None if expected_descriptor_normalizer is None
                        else dict(expected_descriptor_normalizer)
                    ),
                    "loss_weights": (None if expected_loss_weights is None
                                     else _validate_loss_weights(expected_loss_weights)),
                }
                for field, value in expected.items():
                    if value is not None and metadata[field] != value:
                        raise ValueError(f"{field}: graph index metadata mismatch")
                arrays = {name: archive[name].copy() for name in required - {"metadata"}}
        except ValueError:
            raise
        except (OSError, KeyError, AttributeError, zipfile.BadZipFile) as error:
            raise ValueError(f"corrupt graph index: {source}") from error
        if (arrays["embeddings"].dtype != np.float32 or arrays["mol_ids"].dtype != np.int64
                or arrays["smiles"].dtype.kind != "U" or arrays["fingerprints_packed"].dtype != np.uint8
                or arrays["descriptors"].dtype != np.float32):
            raise ValueError("graph index array dtype mismatch")
        if metadata.get("normalized") is not True:
            raise ValueError("normalized metadata must be true")
        embedding_norms = np.linalg.norm(arrays["embeddings"], axis=1)
        if (not np.isfinite(embedding_norms).all()
                or not np.allclose(embedding_norms, 1.0, rtol=1e-5, atol=1e-6)):
            raise ValueError("normalized graph index requires unit-norm embedding rows")
        fingerprint_bits = metadata.get("fingerprint_bits")
        if type(fingerprint_bits) is not int or fingerprint_bits <= 0:
            raise ValueError("fingerprint_bits metadata is invalid")
        packed = arrays["fingerprints_packed"]
        if packed.ndim != 2 or packed.shape[1] != (fingerprint_bits + 7) // 8:
            raise ValueError("packed fingerprint shape mismatch")
        fingerprints = np.unpackbits(packed, axis=1, bitorder="little")[:, :fingerprint_bits]
        rebuilt = cls.from_arrays(arrays["embeddings"], arrays["mol_ids"], arrays["smiles"],
            fingerprints, arrays["descriptors"], checkpoint_sha256=metadata.get("checkpoint_sha256"),
            feature_metadata=metadata.get("feature_config"), model_config=metadata.get("model_config"),
            source_smiles_sha256=metadata.get("source_smiles_sha256"),
            descriptor_mean=normalizer["mean"], descriptor_std=normalizer["std"],
            loss_weights=weights)
        if rebuilt.metadata != metadata:
            raise ValueError("graph index metadata does not match stored arrays")
        return rebuilt

    def search(self, query_embedding: np.ndarray, topk: int, *, rerank: RerankConfig | None = None,
               predicted_fingerprint: np.ndarray | None = None,
               predicted_descriptors: np.ndarray | None = None) -> GraphSearchResult:
        """稳定检索；辅助项只能来自 spectrum head 的预测，候选真实结构只作为比较目标。"""
        topk = _positive_integer(topk, "topk")
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query embedding must be one-dimensional")
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("query embedding dimension does not match graph index")
        if not np.isfinite(query).all():
            raise ValueError("query embedding must contain only finite values")
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm <= np.finfo(np.float32).eps:
            raise ValueError("query embedding must not have zero norm")
        query = query / norm
        cosine = (self.embeddings @ query).astype(np.float32, copy=False)
        if not np.isfinite(cosine).all():
            raise ValueError("cosine scores must be finite")
        base_order = np.argsort(-cosine, kind="stable")
        config = rerank or RerankConfig()
        if not isinstance(config, RerankConfig):
            raise TypeError("rerank must be a RerankConfig")
        output_count = min(topk, len(base_order))
        if not config.enabled:
            chosen = base_order[:output_count]
            return GraphSearchResult(_readonly(chosen.astype(np.int64)),
                _readonly(cosine[chosen].astype(np.float32)),
                _readonly(cosine[chosen].astype(np.float32)),
                {"query_structure_source": "none", "reranked": False})
        if config.top_m < output_count:
            raise ValueError("rerank.top_m must be at least the requested output count")
        subset = base_order[:min(config.top_m, len(base_order))]
        final = cosine[subset].astype(np.float32, copy=True)
        used_prediction = False
        if config.fingerprint_weight > 0:
            if not self.metadata["capabilities"]["fingerprint_rerank"]:
                raise ValueError("checkpoint has no fingerprint head for reranking")
            if predicted_fingerprint is None:
                raise ValueError("predicted_fingerprint is required for positive weight")
            fingerprint = np.asarray(predicted_fingerprint, dtype=np.float32)
            if fingerprint.shape != (self.fingerprints.shape[1],) or not np.isfinite(fingerprint).all():
                raise ValueError("predicted_fingerprint has invalid shape or non-finite values")
            if np.any((fingerprint < 0) | (fingerprint > 1)):
                raise ValueError("predicted_fingerprint probabilities must be in [0, 1]")
            candidates = self.fingerprints[subset].astype(np.float32)
            intersection = candidates @ fingerprint
            union = candidates.sum(axis=1) + fingerprint.sum() - intersection
            tanimoto = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            final += np.float32(config.fingerprint_weight) * tanimoto.astype(np.float32)
            used_prediction = True
        if config.descriptor_weight > 0:
            if not self.metadata["capabilities"]["descriptor_rerank"]:
                raise ValueError("checkpoint has no descriptor head for reranking")
            if predicted_descriptors is None:
                raise ValueError("predicted_descriptors is required for positive weight")
            descriptor = np.asarray(predicted_descriptors, dtype=np.float32)
            if descriptor.shape != (self.descriptors.shape[1],) or not np.isfinite(descriptor).all():
                raise ValueError("predicted_descriptors has invalid shape or non-finite values")
            normalizer = self.metadata["descriptor_normalizer"]
            mean = np.asarray(normalizer["mean"], dtype=np.float32)
            std = np.asarray(normalizer["std"], dtype=np.float32)
            standardized = (self.descriptors[subset] - mean) / std
            mae = np.mean(np.abs(standardized - descriptor), axis=1, dtype=np.float32)
            final -= np.float32(config.descriptor_weight) * mae
            used_prediction = True
        if not np.isfinite(final).all():
            raise ValueError("rerank scores must be finite")
        reranked = np.argsort(-final, kind="stable")[:output_count]
        chosen = subset[reranked]
        return GraphSearchResult(_readonly(chosen.astype(np.int64)),
            _readonly(final[reranked].astype(np.float32)),
            _readonly(cosine[chosen].astype(np.float32)),
            {"query_structure_source": "predicted_from_spectrum" if used_prediction else "none",
             "reranked": True})


def _same_model_weights(first: StructureAwareRetriever, second: StructureAwareRetriever) -> bool:
    first_state, second_state = first.state_dict(), second.state_dict()
    return first_state.keys() == second_state.keys() and all(
        torch.equal(first_state[key].detach().cpu(), second_state[key].detach().cpu())
        for key in first_state)


def build_graph_index(model: StructureAwareRetriever, dataloader: Iterable[GraphBatch],
                      checkpoint_path: str | Path, output_path: str | Path) -> GraphIndex:
    """从 GraphBatch 分批编码候选，并把权重、normalizer、feature/source hash 绑定到单一 NPZ。"""
    if not isinstance(model, StructureAwareRetriever):
        raise TypeError("build_graph_index requires StructureAwareRetriever")
    snapshot = read_checkpoint_snapshot(checkpoint_path)
    checkpoint_model, payload = load_checkpoint(snapshot, expected_mode="spectrum_graph_v2",
                                                expected_model_config=model.to_config())
    if not _same_model_weights(model, checkpoint_model):
        raise ValueError("model weights do not match checkpoint_sha256")
    device = next(model.parameters()).device
    embedding_batches: list[np.ndarray] = []
    mol_id_batches: list[np.ndarray] = []
    fingerprint_batches: list[np.ndarray] = []
    descriptor_batches: list[np.ndarray] = []
    smiles: list[str] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for raw_batch in dataloader:
                graph_batch = raw_batch.get("graph") if isinstance(raw_batch, Mapping) else raw_batch
                if not isinstance(graph_batch, GraphBatch):
                    raise TypeError("graph index dataloader must yield GraphBatch or {'graph': GraphBatch}")
                encoded = model.encode_graph(graph_batch.to(device))
                if encoded.ndim != 2 or encoded.shape[0] != graph_batch.mol_ids.numel():
                    raise ValueError("graph encoder returned incompatible batch shape")
                embedding_batches.append(encoded.detach().cpu().numpy().astype(np.float32))
                mol_id_batches.append(graph_batch.mol_ids.detach().cpu().numpy().astype(np.int64))
                fingerprint_batches.append(graph_batch.fingerprints.detach().cpu().numpy().astype(np.uint8))
                descriptor_batches.append(graph_batch.descriptors.detach().cpu().numpy().astype(np.float32))
                smiles.extend(graph_batch.canonical_smiles)
    finally:
        model.train(was_training)
    if not embedding_batches:
        raise ValueError("graph index dataloader is empty")
    ids = np.concatenate(mol_id_batches)
    source_hash = canonical_smiles_sha256(ids.tolist(), smiles)
    dataset = getattr(dataloader, "dataset", None)
    dataset_feature = getattr(dataset, "feature_config", None)
    if type(dataset_feature) is not StructureFeatureConfig:
        raise ValueError("feature_config: verified dataloader dataset provenance is required")
    if dataset_feature.to_metadata() != payload["feature_config"]:
        raise ValueError("feature_config: dataloader/checkpoint mismatch")
    dataset_source = getattr(dataset, "source_smiles_sha256", None)
    if type(dataset_source) is not str:
        raise ValueError("source_smiles_sha256: dataloader dataset provenance is required")
    if dataset_source != source_hash:
        raise ValueError("source_smiles_sha256: dataloader records mismatch")
    stats = payload["descriptor_stats"]
    index = GraphIndex.from_encoded_batches(embedding_batches=embedding_batches,
        mol_id_batches=mol_id_batches, smiles=smiles, fingerprint_batches=fingerprint_batches,
        descriptor_batches=descriptor_batches, checkpoint_sha256=snapshot.sha256,
        feature_metadata=payload["feature_config"], model_config=payload["model_config"],
        source_smiles_sha256=source_hash,
        descriptor_mean=stats["mean"].detach().cpu().numpy(),
        descriptor_std=stats["std"].detach().cpu().numpy(),
        loss_weights=payload["loss_weights"])
    index.save(output_path)
    return index


__all__ = ["GraphIndex", "GraphSearchResult", "INDEX_SCHEMA_VERSION", "RerankConfig",
           "build_graph_index"]

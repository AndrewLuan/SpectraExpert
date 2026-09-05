"""功能概述：原子保存两种显式训练模式的完整状态，严格阻止配置、数据及版本混用。
主要职责：model_config/rebuild_model 序列化和严格重建模型；capture_rng/restore_rng_state
保存 Python、NumPy、Torch 和 CUDA RNG；save_checkpoint 以同目录临时文件原子替换；
load_checkpoint 检查字段后恢复模型、可选 optimizer/scheduler 和指定 rank 的 RNG；
copy_checkpoint 严格验证后原子保留历史 best 的完整 payload。
MODES/SPECTRUM_KEYS 固定版本和光谱配置键；_compare/_validate_stats 执行字段与统计量校验，
_atomic_save 在目标目录创建临时文件并在持久化成功后替换目标。
代码库关系：消费 Task 4 StructureAwareRetriever 与 SpectrumTransformer、Task 2 normalizer；
由 train_structure_aware 使用。历史无 schema checkpoint 仍只能由 legacy wrapper 加载。
输入输出与副作用：只加载可信本地 torch checkpoint（weights_only=False）；错误报告字段路径；
失败不会覆盖原文件。epoch 为已完成的零基 epoch，RNG 必须在重建模型后恢复。
"""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from spectraexpert.data.structure_features import DescriptorNormalizer, StructureFeatureConfig
from spectraexpert.models.multimodal_retriever import StructureAwareRetriever
from spectraexpert.models.spectrum_encoder import SpectrumTransformer


MODES = ("spectral_only_v1", "spectrum_graph_v2")
SPECTRUM_KEYS = ("signal_length", "patch_size", "stride", "embed_dim", "depth", "num_heads", "projection_dim")


def model_config(model):
    if isinstance(model, StructureAwareRetriever):
        return model.to_config()
    if isinstance(model, SpectrumTransformer):
        if model.dropout != .1 or model.mlp_ratio != 4.:
            raise ValueError("model_config requires default dropout and mlp_ratio")
        return {key: getattr(model, key) for key in SPECTRUM_KEYS}
    raise ValueError("model_config: unsupported model type")


def rebuild_model(mode, config):
    if mode not in MODES:
        raise ValueError(f"training_mode: unsupported {mode!r}")
    expected_keys = set(SPECTRUM_KEYS)
    if mode == "spectrum_graph_v2":
        expected_keys |= {"graph_hidden_dim", "descriptor_dim", "predict_fingerprint"}
    if not isinstance(config, dict) or set(config) != expected_keys:
        raise ValueError("model_config: missing or unexpected fields")
    for key, value in config.items():
        expected_type = bool if key == "predict_fingerprint" else int
        if type(value) is not expected_type or (expected_type is int and value <= 0):
            raise ValueError(f"model_config.{key}: invalid type/value")
    try:
        model = (StructureAwareRetriever.from_config(config) if mode == "spectrum_graph_v2"
                 else SpectrumTransformer(**config))
    except (ValueError, AssertionError, KeyError, TypeError) as error:
        raise ValueError(f"model_config: {error}") from error
    if model_config(model) != config:
        raise ValueError("model_config: strict roundtrip mismatch")
    return model


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("rng_states.cuda: device count mismatch")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _compare(expected, actual, prefix):
    if not isinstance(actual, dict):
        raise ValueError(f"{prefix}: expected mapping")
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise ValueError(f"{prefix}.{key}: expected {value!r}, got {actual.get(key)!r}")


def _validate_stats(stats, dimension):
    if not isinstance(stats, dict) or set(stats) != {"mean", "std"}:
        raise ValueError("descriptor_stats: requires mean/std")
    for name in ("mean", "std"):
        value = stats[name]
        if not isinstance(value, torch.Tensor) or value.shape != (dimension,) or not torch.isfinite(value).all():
            raise ValueError(f"descriptor_stats.{name}: invalid shape/value")
    if not (stats["std"] > 0).all():
        raise ValueError("descriptor_stats.std: must be positive")


def save_checkpoint(path, model, optimizer=None, *, epoch=0, global_steps=0,
                    validation=None, descriptor_stats: DescriptorNormalizer,
                    training_mode=None, feature_config=None, scheduler=None,
                    selection_metric="top1_tanimoto", seed=0, split_manifest_sha256=None,
                    source_sha256=None, source_smiles_sha256=None, loss_weights=None,
                    best_score=float("-inf"), bad_epochs=0, rng_states=None, train_config=None):
    mode = training_mode or ("spectrum_graph_v2" if isinstance(model, StructureAwareRetriever) else "spectral_only_v1")
    actual_mode = "spectrum_graph_v2" if isinstance(model, StructureAwareRetriever) else "spectral_only_v1"
    if mode not in MODES or mode != actual_mode:
        raise ValueError("training_mode: model does not match mode")
    config = model_config(model)
    stats = {"mean": descriptor_stats.mean.detach().cpu(), "std": descriptor_stats.std.detach().cpu()}
    _validate_stats(stats, config.get("descriptor_dim", 15))
    payload = dict(schema_version=mode, training_mode=mode,
        candidate_modality="graph" if mode == "spectrum_graph_v2" else "spectrum",
        model_config=config, feature_config=(feature_config or StructureFeatureConfig()).to_metadata(),
        descriptor_stats=stats, model_state_dict=model.state_dict(),
        optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
        scheduler_state_dict=None if scheduler is None else scheduler.state_dict(),
        epoch=int(epoch), global_steps=int(global_steps), validation=validation or {},
        selection_metric=selection_metric, selection_exclude_identity=True,
        seed=int(seed), split_manifest_sha256=split_manifest_sha256,
        source_sha256=source_sha256 or {}, source_smiles_sha256=source_smiles_sha256,
        loss_weights=loss_weights or {}, best_score=float(best_score), bad_epochs=int(bad_epochs),
        rng_states=rng_states if rng_states is not None else [capture_rng()], train_config=train_config or {})
    return _atomic_save(path, payload)


def _atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def copy_checkpoint(source, destination, **validation_options):
    """验证来源后原子复制完整历史 checkpoint；不把 last 权重误记为 best。"""
    _, payload = load_checkpoint(source, **validation_options)
    return _atomic_save(destination, payload)


def load_checkpoint(path, *, model=None, optimizer=None, scheduler=None, device="cpu",
                    expected_mode=None, expected_model_config=None, expected_feature_config=None,
                    expected_metadata=None, restore_rng=False, rank=0):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") not in MODES:
        raise ValueError("schema_version: unsupported checkpoint; use legacy wrapper for historical files")
    mode = payload["schema_version"]
    if payload.get("training_mode") != mode or (expected_mode is not None and expected_mode != mode):
        raise ValueError("training_mode: checkpoint mismatch")
    required = {"model_config", "feature_config", "descriptor_stats", "model_state_dict",
        "optimizer_state_dict", "scheduler_state_dict", "epoch", "global_steps", "validation",
        "selection_metric", "selection_exclude_identity", "seed", "split_manifest_sha256", "source_sha256", "source_smiles_sha256",
        "loss_weights", "best_score", "bad_epochs", "rng_states", "train_config", "candidate_modality"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    for key in ("epoch", "global_steps", "bad_epochs", "seed"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError(f"{key}: expected nonnegative integer")
    states = payload["rng_states"]
    if not isinstance(states, list) or not states or len(states) != payload["train_config"].get("world_size", 1):
        raise ValueError("rng_states: expected one RNG state per rank")
    for state in states:
        if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch", "cuda"}:
            raise ValueError("rng_states: missing or unexpected RNG fields")
    if payload["candidate_modality"] != ("graph" if mode == "spectrum_graph_v2" else "spectrum"):
        raise ValueError("candidate_modality: mismatch")
    feature = (expected_feature_config or StructureFeatureConfig()).to_metadata()
    if set(payload["feature_config"]) != set(feature):
        raise ValueError("feature_config: missing or unexpected fields")
    _compare(feature, payload["feature_config"], "feature_config")
    if expected_model_config is not None:
        _compare(expected_model_config, payload["model_config"], "model_config")
    if expected_metadata is not None:
        _compare(expected_metadata, payload, "checkpoint")
    reconstructed = rebuild_model(mode, payload["model_config"])
    _validate_stats(payload["descriptor_stats"], payload["model_config"].get("descriptor_dim", 15))
    if payload["selection_metric"] != "top1_tanimoto":
        raise ValueError("selection_metric: requires top1_tanimoto")
    if payload["selection_exclude_identity"] is not True:
        raise ValueError("selection_exclude_identity: requires True for fair ablation selection")
    if model is None:
        model = reconstructed
    elif model_config(model) != payload["model_config"]:
        raise ValueError("model_config: target model mismatch")
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(f"model_state_dict: {error}") from error
    model.to(device)
    for name, component in (("optimizer", optimizer), ("scheduler", scheduler)):
        if component is not None:
            state = payload[f"{name}_state_dict"]
            if state is None:
                raise ValueError(f"{name}_state_dict: absent; cannot resume")
            component.load_state_dict(state)
    if optimizer is not None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
    if restore_rng:
        if rank >= len(payload["rng_states"]):
            raise ValueError("rng_states: missing rank")
        restore_rng_state(payload["rng_states"][rank])
    return model, payload

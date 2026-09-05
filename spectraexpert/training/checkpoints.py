"""功能概述：原子保存两种显式训练模式的完整状态，严格阻止配置、数据及版本混用。
主要职责：CheckpointSnapshot/read_checkpoint_snapshot 捕获一次读取的不可变 bytes，hash 和
load(BytesIO) 使用相同内容；materialize_checkpoint/copy_checkpoint 验证后原子物化原字节。
model_config/rebuild_model 序列化和严格重建模型；capture_rng/restore_rng_state 保存各 RNG，
恢复前先对称校验 CUDA 可用性及状态数量；validate_best_payload 统一验证 best 的有限等分数、
bad_epochs=0 与选择元数据，copy_checkpoint(require_best=True) 在复制历史 best 前调用它。
save_checkpoint 原子保存状态，load_checkpoint 预检后恢复模型、optimizer/scheduler 与 RNG。
_validate_training_bindings 拒绝重复参数、全冻结模型和不完整 trainable 参数覆盖，检查 scheduler
归属；_preflight_states 在写入前检查状态存在性和参数组布局；_apply_states 执行状态与设备恢复；
_validate_standard_loaders/_require_torch_implementation 检查 public/private 恢复及 extra-state
动态方法的 torch.* 来源并拒绝实例覆盖和 hooks；_scheduler_tree 递归检查组合 scheduler。
允许官方 ReduceLROnPlateau/SequentialLR/ChainedScheduler 覆盖；_transactional_apply 先 deepcopy/probe，
再备份调用方状态进行事务式应用，失败恢复三者。入口 RNG 保护涵盖所有校验、probe 与应用。
MODES/SPECTRUM_KEYS 固定版本和光谱配置键；_compare/_validate_stats 执行字段与统计量校验，
_atomic_save/_atomic_write 在目标目录创建临时文件，flush/fsync 成功后 replace 目标。
代码库关系：消费 Task 4 StructureAwareRetriever 与 SpectrumTransformer、Task 2 normalizer；
由 train_structure_aware 使用。历史无 schema checkpoint 仍只能由 legacy wrapper 加载。
输入输出与副作用：只加载可信本地 torch checkpoint（weights_only=False）；错误报告字段路径；
事务只支持使用框架恢复实现且无用户扩展/load-state hooks 的对象，不承诺任意自定义副作用或
导致回滚也无法执行的硬件故障；显式恢复需要 probe 和回滚快照的额外内存，probe 随后释放。
epoch 为已完成的零基 epoch；默认保持入口 RNG，restore_rng=True 成功时才采用 checkpoint RNG。
"""
from __future__ import annotations

import os
import copy
import hashlib
import io
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from spectraexpert.data.structure_features import DescriptorNormalizer, StructureFeatureConfig
from spectraexpert.models.multimodal_retriever import StructureAwareRetriever
from spectraexpert.models.spectrum_encoder import SpectrumTransformer


MODES = ("spectral_only_v1", "spectrum_graph_v2")
SPECTRUM_KEYS = ("signal_length", "patch_size", "stride", "embed_dim", "depth", "num_heads", "projection_dim")


@dataclass(frozen=True)
class CheckpointSnapshot:
    """同一次文件读取的不可变字节；digest 和反序列化均只使用这份内容。"""
    content: bytes

    def __post_init__(self):
        if not isinstance(self.content, bytes):
            raise TypeError("CheckpointSnapshot requires immutable bytes")

    @property
    def sha256(self):
        return hashlib.sha256(self.content).hexdigest()


def read_checkpoint_snapshot(path):
    return path if isinstance(path, CheckpointSnapshot) else CheckpointSnapshot(Path(path).read_bytes())


def materialize_checkpoint(snapshot, destination):
    """按原字节原子落盘；replace 替换目标 symlink 本身，不写入它指向的文件。"""
    if not isinstance(snapshot, CheckpointSnapshot):
        raise TypeError("materialize_checkpoint requires an immutable CheckpointSnapshot")
    return _atomic_write(destination, lambda handle: handle.write(snapshot.content))


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
    cuda_states = state["cuda"]
    available = torch.cuda.is_available()
    if (not isinstance(cuda_states, list)
            or (available and (not cuda_states or len(cuda_states) != torch.cuda.device_count()))
            or (not available and cuda_states)):
        raise ValueError("rng_states.cuda: device count mismatch")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def validate_best_payload(payload):
    """在物化任何 best 前验证统一的选择不变量；普通 last 不要求分数等于 best。"""
    metric = payload.get("selection_metric")
    validation = payload.get("validation")
    score = validation.get(metric) if isinstance(validation, dict) else None
    best = payload.get("best_score")
    if (metric != "top1_tanimoto" or payload.get("selection_exclude_identity") is not True
            or payload.get("bad_epochs") != 0
            or type(score) not in (int, float) or type(best) not in (int, float)
            or not np.isfinite(score) or not np.isfinite(best) or score != best):
        raise ValueError("best selection metadata inconsistent: requires finite equal score/best_score and bad_epochs=0")


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
    return _atomic_write(path, lambda handle: torch.save(payload, handle))


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def copy_checkpoint(source, destination, *, require_best=False, **validation_options):
    """验证来源后原子复制完整历史 checkpoint；不把 last 权重误记为 best。"""
    snapshot = read_checkpoint_snapshot(source)
    _, payload = load_checkpoint(snapshot, **validation_options)
    if require_best:
        validate_best_payload(payload)
    return materialize_checkpoint(snapshot, destination)


def _validate_training_bindings(model, optimizer, scheduler):
    if optimizer is not None:
        if model is None:
            raise ValueError("optimizer requires an explicitly supplied model")
        parameters = list(model.parameters())
        model_ids = {id(parameter) for parameter in parameters}
        required_ids = {id(parameter) for parameter in parameters if parameter.requires_grad}
        if not required_ids:
            raise ValueError("model must have at least one requires_grad parameter")
        flattened = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        optimizer_ids = {id(parameter) for parameter in flattened}
        if len(flattened) != len(optimizer_ids):
            raise ValueError("optimizer contains duplicate parameter objects")
        if optimizer_ids - model_ids:
            raise ValueError("optimizer parameters contain objects outside the supplied model")
        if optimizer_ids - required_ids:
            raise ValueError("optimizer parameters include non-trainable model parameters")
        if required_ids - optimizer_ids:
            raise ValueError("optimizer parameters do not cover all model requires_grad parameters")
    if scheduler is not None:
        if optimizer is None:
            raise ValueError("scheduler requires an explicitly supplied optimizer")
        for current in _scheduler_tree(scheduler):
            if getattr(current, "optimizer", None) is not optimizer:
                raise ValueError("scheduler must belong to the supplied optimizer")


def _scheduler_tree(scheduler):
    pending, seen = [scheduler], set()
    while pending:
        current = pending.pop()
        if id(current) not in seen:
            seen.add(id(current))
            yield current
            pending.extend(getattr(current, "_schedulers", []))


def _require_torch_implementation(component, name, methods):
    for method in methods:
        implementation = getattr(type(component), method, None)
        if (method in component.__dict__ or (implementation is not None
                and not getattr(implementation, "__module__", "").startswith("torch."))):
            raise ValueError(f"unsupported {name} {method} override")


def _validate_standard_loaders(model, optimizer, scheduler):
    """验证实际动态调用点；官方 torch.* 覆盖受支持，用户扩展及实例覆盖被拒绝。"""
    if model is not None:
        for name, module in model.named_modules():
            if module._load_state_dict_pre_hooks or module._load_state_dict_post_hooks:
                raise ValueError(f"unsupported model load-state hook: {name or '<root>'}")
            _require_torch_implementation(module, f"model.{name}",
                ("load_state_dict", "_load_from_state_dict", "set_extra_state", "__setstate__"))
    if optimizer is not None:
        if (optimizer._optimizer_load_state_dict_pre_hooks
                or optimizer._optimizer_load_state_dict_post_hooks):
            raise ValueError("unsupported optimizer load-state hook")
        _require_torch_implementation(optimizer, "optimizer", ("load_state_dict", "__setstate__"))
    if scheduler is not None:
        for current in _scheduler_tree(scheduler):
            _require_torch_implementation(current, "scheduler",
                ("load_state_dict", "__setstate__", "_init_is_better", "_init_scale_fn"))


def _preflight_states(payload, optimizer, scheduler):
    if not isinstance(payload["model_state_dict"], dict):
        raise ValueError("model_state_dict: requires a mapping")
    for name, component in (("optimizer", optimizer), ("scheduler", scheduler)):
        if component is not None and not isinstance(payload[f"{name}_state_dict"], dict):
            raise ValueError(f"{name}_state_dict: absent or invalid; cannot resume")
    if optimizer is not None:
        state = payload["optimizer_state_dict"]
        groups = state.get("param_groups")
        if not isinstance(state.get("state"), dict) or not isinstance(groups, list):
            raise ValueError("optimizer_state_dict: requires state and param_groups")
        if len(groups) != len(optimizer.param_groups):
            raise ValueError("optimizer_state_dict.param_groups: group count mismatch")
        for index, (saved, current) in enumerate(zip(groups, optimizer.param_groups)):
            if (not isinstance(saved, dict) or not isinstance(saved.get("params"), list)
                    or len(saved["params"]) != len(current["params"])):
                raise ValueError(f"optimizer_state_dict.param_groups[{index}]: parameter count mismatch")


def _apply_states(model, optimizer, scheduler, payload, device):
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(f"model_state_dict: {error}") from error
    model.to(device)
    _validate_training_bindings(model, optimizer, scheduler)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    _validate_training_bindings(model, optimizer, scheduler)


def _transactional_apply(model, optimizer, scheduler, payload, device, rng_state=None):
    # 整体 deepcopy 保留 probe 内 scheduler→optimizer→model 参数的对象关系。
    probe = copy.deepcopy((model, optimizer, scheduler))
    try:
        _apply_states(*probe, payload, device)
    finally:
        del probe
    components = (model, optimizer, scheduler)
    memo = {id(component): component for component in components if component is not None}
    previous = [copy.deepcopy(component.state_dict(), memo.copy()) if component is not None else None
                for component in components]
    original_groups = (copy.deepcopy(optimizer.param_groups,
        {id(parameter): parameter for parameter in model.parameters()}) if optimizer is not None else None)
    # 记录注册对象与设备，回滚 device migration 时仍保持 optimizer 引用原 Parameter。
    layout = [(module, mapping, name, tensor, tensor.device)
              for module in model.modules() for mapping in ("_parameters", "_buffers")
              for name, tensor in getattr(module, mapping).items() if tensor is not None]
    original_rng = capture_rng() if rng_state is not None else None
    scheduler_keys = set(scheduler.__dict__) if scheduler is not None else set()
    try:
        _apply_states(model, optimizer, scheduler, payload, device)
        if rng_state is not None:
            restore_rng_state(rng_state)
    except Exception:
        for module, mapping, name, tensor, original_device in layout:
            getattr(module, mapping)[name] = tensor
            tensor.data = tensor.data.to(original_device)
            if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None:
                tensor.grad.data = tensor.grad.data.to(original_device)
        # 入口已拒绝 hooks/覆盖；标准加载保留 training、requires_grad 和梯度值。
        # 使用框架基础方法恢复，回滚仍依赖设备及底层张量复制可用。
        torch.nn.Module.load_state_dict(model, previous[0], strict=True)
        if optimizer is not None:
            optimizer.param_groups = original_groups
            torch.optim.Optimizer.load_state_dict(optimizer, previous[1])
        if scheduler is not None:
            scheduler.optimizer = optimizer
            for key in set(scheduler.__dict__) - scheduler_keys:
                del scheduler.__dict__[key]
            type(scheduler).load_state_dict(scheduler, previous[2])
        if original_rng is not None:
            restore_rng_state(original_rng)
        raise


def load_checkpoint(path, *, model=None, optimizer=None, scheduler=None, device="cpu",
                    expected_mode=None, expected_model_config=None, expected_feature_config=None,
                    expected_metadata=None, restore_rng=False, rank=0):
    entry_rng = capture_rng()
    try:
        _validate_standard_loaders(model, optimizer, scheduler)
        result = _load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
            device=device, expected_mode=expected_mode, expected_model_config=expected_model_config,
            expected_feature_config=expected_feature_config, expected_metadata=expected_metadata,
            restore_rng=restore_rng, rank=rank)
        if not restore_rng:
            restore_rng_state(entry_rng)
        return result
    except BaseException:
        restore_rng_state(entry_rng)
        raise


def _load_checkpoint(path, *, model, optimizer, scheduler, device, expected_mode,
                     expected_model_config, expected_feature_config, expected_metadata,
                     restore_rng, rank):
    _validate_training_bindings(model, optimizer, scheduler)
    snapshot = read_checkpoint_snapshot(path)
    payload = torch.load(io.BytesIO(snapshot.content), map_location="cpu", weights_only=False)
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
    _validate_stats(payload["descriptor_stats"], payload["model_config"].get("descriptor_dim", 15))
    if payload["selection_metric"] != "top1_tanimoto":
        raise ValueError("selection_metric: requires top1_tanimoto")
    if payload["selection_exclude_identity"] is not True:
        raise ValueError("selection_exclude_identity: requires True for fair ablation selection")
    _preflight_states(payload, optimizer, scheduler)
    reconstructed = rebuild_model(mode, payload["model_config"])
    if model is not None and model_config(model) != payload["model_config"]:
        raise ValueError("model_config: target model mismatch")
    rng_state = None
    if restore_rng:
        if rank < 0 or rank >= len(payload["rng_states"]):
            raise ValueError("rng_states: missing rank")
        rng_state = payload["rng_states"][rank]
        current_rng = capture_rng()
        try:
            restore_rng_state(rng_state)
        finally:
            restore_rng_state(current_rng)
    if model is None:
        model = reconstructed
        _apply_states(model, None, None, payload, device)
        if rng_state is not None:
            restore_rng_state(rng_state)
    else:
        del reconstructed
        _transactional_apply(model, optimizer, scheduler, payload, device, rng_state)
    return model, payload

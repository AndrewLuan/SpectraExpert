"""功能概述：原子保存两种显式训练模式的完整状态，严格阻止配置、数据及版本混用。
主要职责：CheckpointSnapshot/read_checkpoint_snapshot 使 load(BytesIO) 和 hash 共用一次读取的
不可变 bytes；materialize_checkpoint/copy_checkpoint 验证后原子物化原字节。model_config/
rebuild_model 负责严格配置重建，MODES/SPECTRUM_KEYS 固定版本及字段；_compare/_validate_stats
校验元数据/normalizer，validate_best_payload 统一验证有限且相等的 best 分数与选择规则。
capture_rng/restore_rng_state 管理各 RNG，写入前对称检查 CUDA topology；load_checkpoint 保证
默认保持入口 RNG，显式恢复成功才采用 checkpoint RNG，任何异常恢复入口 RNG。
_registered_tree 只读取注册字典并拒绝循环，_model_tensors 得到持久化键；不调用 named_modules。
_require_torch_implementation 用静态方法解析拒绝用户 state save/load/extra-state 覆盖及 descriptor；
_validate_standard_loaders 检查 registered save/load hooks 和 exact Parameter/Tensor 类型。
COPY_PROTOCOL 保留旧扩展拒绝兼容性，但不再调用任何调用方对象 copy 协议或建立 deepcopy probe。
_scheduler_tree/_validate_training_bindings 验证组合 scheduler、参数对象唯一性和完整 trainable 覆盖。
_safe_clone 只克隆受限纯数据与 exact Tensor；_preflight_states/_validate_scheduler_state/
_validate_state_layout 校验来源
状态字段、参数组、参数 ID 和张量布局；_apply_states 调用标准框架加载方法并恢复设备。
_transactional_apply 直接克隆注册参数/缓冲区/梯度并保存模式与 flags；_backup_optimizer 保留
Parameter 引用并手工复制 state/groups；scheduler 备份只保存 safe-clone 后的状态及子对象引用。
真实应用失败时回到原注册对象/存储并原位恢复，已有 state_dict 存储视图也保持原值；
不实例化或 deepcopy 调用方模型/optimizer/scheduler。
save_checkpoint/_atomic_save/_atomic_write 以同目录临时文件、flush/fsync、replace 原子保存。
代码库关系：消费 Task 4 StructureAwareRetriever 与 SpectrumTransformer、Task 2 normalizer；
由 train_structure_aware 使用。历史无 schema checkpoint 仍只能由 legacy wrapper 加载。
输入输出与副作用：只加载可信本地 torch checkpoint（weights_only=False）；错误报告字段路径；
事务只支持使用框架恢复实现且无用户扩展/load-state hooks 的对象，不承诺任意自定义副作用或
导致回滚也无法执行的硬件故障。支持标准项目模型的注册模块/参数/缓冲区、框架 optimizer/
scheduler 与纯数据状态；任意额外用户对象或伪造来源不属于契约，loader 不是 Python 对象沙箱。
epoch 为已完成的零基 epoch；默认保持入口 RNG，restore_rng=True 成功时才采用 checkpoint RNG。
"""
from __future__ import annotations

import os
import inspect
import hashlib
import io
import random
import tempfile
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from types import FunctionType, BuiltinFunctionType
from pathlib import Path

import numpy as np
import torch

from spectraexpert.data.structure_features import DescriptorNormalizer, StructureFeatureConfig
from spectraexpert.models.multimodal_retriever import StructureAwareRetriever
from spectraexpert.models.spectrum_encoder import SpectrumTransformer


MODES = ("spectral_only_v1", "spectrum_graph_v2")
SPECTRUM_KEYS = ("signal_length", "patch_size", "stride", "embed_dim", "depth", "num_heads", "projection_dim")
COPY_PROTOCOL = ("__deepcopy__", "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__")


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
        parameters = [tensor for _, module in _registered_tree(model)
                      for tensor in object.__getattribute__(module, "_parameters").values() if tensor is not None]
        model_ids = {id(parameter) for parameter in parameters}
        required_ids = {id(parameter) for parameter in parameters if parameter.requires_grad}
        if not required_ids:
            raise ValueError("model must have at least one requires_grad parameter")
        groups = object.__getattribute__(optimizer, "__dict__")["param_groups"]
        if type(groups) is not list or any(type(group) is not dict or type(group.get("params")) is not list for group in groups):
            raise ValueError("optimizer: unsupported parameter group container")
        flattened = [parameter for group in groups for parameter in group["params"]]
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
            if object.__getattribute__(current, "__dict__").get("optimizer") is not optimizer:
                raise ValueError("scheduler must belong to the supplied optimizer")


def _scheduler_tree(scheduler):
    pending, seen = [scheduler], set()
    while pending:
        current = pending.pop()
        if id(current) not in seen:
            seen.add(id(current))
            yield current
            children = object.__getattribute__(current, "__dict__").get("_schedulers", [])
            if type(children) not in (list, tuple):
                raise ValueError("scheduler: unsupported child container")
            pending.extend(children)


def _registered_tree(model):
    """只读注册字典；允许共享子模块的不同路径，拒绝祖先循环。"""
    pending = [("", model, frozenset())]
    while pending:
        name, module, ancestors = pending.pop()
        if id(module) in ancestors:
            raise ValueError("model registration cycle")
        if not isinstance(module, torch.nn.Module):
            raise ValueError("model registration requires Module")
        yield name, module
        children = object.__getattribute__(module, "_modules")
        if type(children) not in (dict, OrderedDict):
            raise ValueError("model registration requires standard dict")
        for key, child in reversed(list(children.items())):
            if child is not None:
                pending.append((f"{name}.{key}" if name else key, child, ancestors | {id(module)}))


def _require_torch_implementation(component, name, methods):
    missing = object()
    for method in methods:
        implementation = inspect.getattr_static(type(component), method, missing)
        source = (object.__getattribute__(implementation, "__module__")
                  if type(implementation) in (FunctionType, BuiltinFunctionType) else "")
        if (method in object.__getattribute__(component, "__dict__") or (implementation is not missing
                and implementation is not inspect.getattr_static(object, method, missing)
                and not (type(source) is str and source.startswith("torch.")))):
            raise ValueError(f"unsupported {name} {method} override")


def _validate_standard_loaders(model, optimizer, scheduler):
    """静态验证注册对象的保存/加载入口，不调用用户 descriptor 或遍历任意附加属性。"""
    if model is not None:
        for name, module in _registered_tree(model):
            values = object.__getattribute__(module, "__dict__")
            if any(values.get(key) for key in ("_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
                                               "_state_dict_pre_hooks", "_state_dict_hooks")):
                raise ValueError(f"unsupported model state hook: {name or '<root>'}")
            _require_torch_implementation(module, f"model.{name}",
                ("state_dict", "load_state_dict", "_save_to_state_dict", "_load_from_state_dict",
                 "get_extra_state", "set_extra_state") + COPY_PROTOCOL)
            for mapping, expected_type in (("_parameters", torch.nn.Parameter), ("_buffers", torch.Tensor)):
                registered = object.__getattribute__(module, mapping)
                if type(registered) not in (dict, OrderedDict):
                    raise ValueError("model registration requires standard dict")
                for key, tensor in registered.items():
                    if tensor is not None and type(tensor) is not expected_type:
                        raise ValueError(f"model {mapping[1:]} {name}.{key}: unsupported tensor type")
                    if tensor is not None and tensor.grad is not None and type(tensor.grad) is not torch.Tensor:
                        raise ValueError("model gradient: unsupported tensor type")
    if optimizer is not None:
        values = object.__getattribute__(optimizer, "__dict__")
        if any(values.get(key) for key in ("_optimizer_load_state_dict_pre_hooks", "_optimizer_load_state_dict_post_hooks",
                                           "_optimizer_state_dict_pre_hooks", "_optimizer_state_dict_post_hooks")):
            raise ValueError("unsupported optimizer state hook")
        _require_torch_implementation(optimizer, "optimizer", ("state_dict", "load_state_dict") + COPY_PROTOCOL)
    if scheduler is not None:
        for current in _scheduler_tree(scheduler):
            _require_torch_implementation(current, "scheduler",
                ("state_dict", "load_state_dict", "_init_is_better", "_init_scale_fn") + COPY_PROTOCOL)


def _safe_clone(value, path="state", ancestors=frozenset()):
    """纯数据复制：不调用对象 copy/reduce 协议，不接受用户容器或 Tensor 子类。"""
    kind = type(value)
    if kind in (type(None), bool, int, float, str, bytes):
        return value
    if kind is torch.Tensor:
        if value.layout != torch.strided:
            raise ValueError(f"{path}: unsupported tensor layout")
        return value.detach().clone()
    if kind not in (dict, OrderedDict, list, tuple) or id(value) in ancestors:
        raise ValueError(f"{path}: unsupported state data type or cycle")
    ancestors = ancestors | {id(value)}
    if kind in (dict, OrderedDict):
        result = kind()
        for key, item in value.items():
            if type(key) not in (str, int, float, bool, type(None)):
                raise ValueError(f"{path}: unsupported state key")
            result[key] = _safe_clone(item, f"{path}.{key}", ancestors)
        if kind is OrderedDict and "_metadata" in object.__getattribute__(value, "__dict__"):
            result._metadata = _safe_clone(value._metadata, f"{path}._metadata", ancestors)
        return result
    return kind(_safe_clone(item, f"{path}[{index}]", ancestors) for index, item in enumerate(value))


def _model_tensors(model):
    result = {}
    for prefix, module in _registered_tree(model):
        nonpersistent = object.__getattribute__(module, "_non_persistent_buffers_set")
        for mapping in ("_parameters", "_buffers"):
            for name, tensor in object.__getattribute__(module, mapping).items():
                if tensor is not None and not (mapping == "_buffers" and name in nonpersistent):
                    result[f"{prefix}.{name}" if prefix else name] = tensor
    return result


def _preflight_states(payload, model, optimizer, scheduler):
    if not isinstance(payload["model_state_dict"], dict):
        raise ValueError("model_state_dict: requires a mapping")
    for name, component in (("optimizer", optimizer), ("scheduler", scheduler)):
        if component is not None and not isinstance(payload[f"{name}_state_dict"], dict):
            raise ValueError(f"{name}_state_dict: absent or invalid; cannot resume")
    for name, component in (("model", model), ("optimizer", optimizer), ("scheduler", scheduler)):
        if component is not None:
            payload[f"{name}_state_dict"] = _safe_clone(payload[f"{name}_state_dict"], f"{name}_state_dict")
    expected = _model_tensors(model)
    state = payload["model_state_dict"]
    if state.keys() != expected.keys():
        raise ValueError("model_state_dict: missing or unexpected keys")
    for key, tensor in expected.items():
        saved = state[key]
        if (type(saved) is not torch.Tensor or saved.shape != tensor.shape
                or saved.dtype != tensor.dtype or saved.layout != tensor.layout):
            raise ValueError(f"model_state_dict.{key}: shape/dtype/layout mismatch")
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
        ids = [key for group in groups for key in group["params"]]
        if any(type(key) is not int for key in ids) or len(ids) != len(set(ids)) or set(state["state"]) - set(ids):
            raise ValueError("optimizer_state_dict: invalid parameter IDs")
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        for key, parameter in zip(ids, parameters):
            values = state["state"].get(key, {})
            if type(values) is not dict:
                raise ValueError("optimizer_state_dict: parameter state requires dict")
            for name, value in values.items():
                if type(value) is torch.Tensor and (name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer")):
                    if value.shape != parameter.shape or value.dtype != parameter.dtype:
                        raise ValueError(f"optimizer_state_dict.{name}: shape/dtype mismatch")
                if name == "step" and type(value) is torch.Tensor and value.numel() != 1:
                    raise ValueError("optimizer_state_dict.step: requires scalar")
    if scheduler is not None:
        _validate_scheduler_state(payload["scheduler_state_dict"], scheduler)


def _validate_scheduler_state(state, scheduler):
    current = _safe_clone(scheduler.state_dict(), "scheduler backup")
    if state.keys() != current.keys() or "optimizer" in state:
        raise ValueError("scheduler_state_dict: field layout mismatch")
    _validate_state_layout(state, current, "scheduler_state_dict")
    children = object.__getattribute__(scheduler, "__dict__").get("_schedulers", [])
    if children:
        saved_children = state.get("_schedulers")
        if type(saved_children) is not list or len(saved_children) != len(children):
            raise ValueError("scheduler_state_dict: child count mismatch")
        for child_state, child in zip(saved_children, children):
            _validate_scheduler_state(child_state, child)


def _validate_state_layout(saved, current, path):
    if type(saved) is not type(current):
        raise ValueError(f"{path}: type layout mismatch")
    if type(current) in (dict, OrderedDict):
        if saved.keys() != current.keys():
            raise ValueError(f"{path}: field layout mismatch")
        for key in current:
            _validate_state_layout(saved[key], current[key], f"{path}.{key}")
    elif type(current) in (list, tuple):
        if len(saved) != len(current):
            raise ValueError(f"{path}: length layout mismatch")
        for index, (first, second) in enumerate(zip(saved, current)):
            _validate_state_layout(first, second, f"{path}[{index}]")
    elif type(current) is torch.Tensor:
        if saved.shape != current.shape or saved.dtype != current.dtype or saved.layout != current.layout:
            raise ValueError(f"{path}: tensor shape/dtype/layout mismatch")


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
    # 只备份注册张量与纯数据；不实例化、不 deepcopy 调用方组件。
    layout, modes = [], []
    for _, module in _registered_tree(model):
        modes.append((module, object.__getattribute__(module, "training")))
        for mapping in ("_parameters", "_buffers"):
            for name, tensor in object.__getattribute__(module, mapping).items():
                if tensor is not None:
                    grad = tensor.grad
                    data = tensor.detach()
                    grad_data = None if grad is None else grad.detach()
                    layout.append((module, mapping, name, tensor, data, data.clone(), tensor.requires_grad,
                                   grad, grad_data, None if grad_data is None else grad_data.clone()))
    optimizer_backup = _backup_optimizer(optimizer) if optimizer is not None else None
    scheduler_backup = []
    if scheduler is not None:
        for current in _scheduler_tree(scheduler):
            values = object.__getattribute__(current, "__dict__")
            scheduler_backup.append((current, _safe_clone(current.state_dict(), "scheduler backup"),
                                     set(values), values.get("_schedulers")))
    try:
        _apply_states(model, optimizer, scheduler, payload, device)
        if rng_state is not None:
            restore_rng_state(rng_state)
    except BaseException:
        for module, mapping, name, tensor, data, saved_data, requires_grad, grad, grad_data, saved_grad in layout:
            object.__getattribute__(module, mapping)[name] = tensor
            tensor.data = data
            data.copy_(saved_data)
            tensor.requires_grad_(requires_grad)
            tensor.grad = grad
            if grad is not None:
                grad.data = grad_data
                grad_data.copy_(saved_grad)
        for module, training in modes:
            object.__setattr__(module, "training", training)
        if optimizer is not None:
            values = object.__getattribute__(optimizer, "__dict__")
            values["state"], values["param_groups"] = optimizer_backup
        for current, state, keys, children in scheduler_backup:
            values = object.__getattribute__(current, "__dict__")
            for key in set(values) - keys:
                del values[key]
            values["optimizer"] = optimizer
            if children is not None:
                values["_schedulers"] = children
            current.load_state_dict(state)
        raise


def _backup_optimizer(optimizer):
    values = object.__getattribute__(optimizer, "__dict__")
    current_state = values["state"]
    if type(current_state) not in (dict, defaultdict):
        raise ValueError("optimizer backup: unsupported state mapping")
    if type(current_state) is defaultdict and current_state.default_factory not in (dict, None):
        raise ValueError("optimizer backup: unsupported default factory")
    state = defaultdict(current_state.default_factory) if type(current_state) is defaultdict else {}
    groups = []
    parameter_ids = set()
    for group in values["param_groups"]:
        copied = _safe_clone({key: value for key, value in group.items() if key != "params"}, "optimizer group")
        copied["params"] = list(group["params"])
        groups.append(copied)
        parameter_ids.update(id(parameter) for parameter in group["params"])
    for parameter, entry in current_state.items():
        if type(parameter) is not torch.nn.Parameter or id(parameter) not in parameter_ids:
            raise ValueError("optimizer backup: external state parameter")
        state[parameter] = _safe_clone(entry, "optimizer backup")
    return state, groups


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
    reconstructed = rebuild_model(mode, payload["model_config"])
    if model is not None and model_config(model) != payload["model_config"]:
        raise ValueError("model_config: target model mismatch")
    _preflight_states(payload, reconstructed if model is None else model, optimizer, scheduler)
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

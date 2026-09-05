"""功能概述：验证训练 checkpoint 的版本隔离、严格重建和原子写入契约。
主要职责：通过真实模型保存/恢复验证权重、优化器与 RNG；篡改配置验证字段级拒绝；
检查 optimizer 参数对象及 scheduler 归属，拒绝时不改变模型，正确恢复后能更新同一模型。
覆盖单/多参数组重复、空 trainable、缺失状态与组布局，递归拒绝加载 hooks 及实例/子类覆盖。
检查调用方对象不被 deepcopy、底层张量复制中断后的三者事务回滚和入口 RNG 保持，以及不可变 checkpoint
bytes；同时确认梯度、requires_grad 和各子模块训练模式保持不变，故障注入由 pytest 撤销。
覆盖 nested private/extra-state 和 optimizer/scheduler 恢复扩展的入口拒绝，官方 plateau/组合
scheduler 的恢复、续跑与回滚；以 fake CUDA RNG 后端验证数量对称预检和不恢复时的入口保护。
对 root/nested module、optimizer、scheduler/子 scheduler 的五种 copy 协议作实例/子类矩阵，
确认任何 deepcopy 前拒绝，用户回调未运行，调用方权重/状态与入口 RNG 不变。
构造/newargs/named_modules 探针确认这些入口不被调用；descriptor、save hooks/extra-state 与
Parameter 子类静态拒绝；纯数据和张量/scheduler 布局预检在任何调用方状态写入之前完成。
代码库关系：消费 training.checkpoints，避免旧 checkpoint 被静默解释为新模型。
输入输出与副作用：只在 pytest 临时目录创建小型模型文件，不访问实验数据。
"""
import random
import copy
import types

import numpy as np
import pytest
import torch

from spectraexpert.data.structure_features import DescriptorNormalizer
from spectraexpert.models.multimodal_retriever import StructureAwareRetriever
from spectraexpert.training.checkpoints import capture_rng, load_checkpoint, save_checkpoint


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


def make_optimizer_checkpoint(tmp_path):
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    scheduler.step()
    return save_checkpoint(tmp_path / "trained.pt", model, optimizer, scheduler=scheduler,
        descriptor_stats=DescriptorNormalizer(torch.zeros(15), torch.ones(15)))


def test_optimizer_without_explicit_model_is_rejected(tmp_path):
    path = make_optimizer_checkpoint(tmp_path)
    optimizer = torch.optim.Adam(make_model().parameters())
    with pytest.raises(ValueError, match="optimizer.*model"):
        load_checkpoint(path, optimizer=optimizer)
    assert not optimizer.state


@pytest.mark.parametrize("binding", ["other_model", "external_parameter", "incomplete"])
def test_optimizer_invalid_parameter_binding_rejected_before_model_mutation(tmp_path, binding):
    path = make_optimizer_checkpoint(tmp_path)
    target = make_model()
    parameters = list(target.parameters())
    if binding == "other_model":
        parameters = list(make_model().parameters())
    elif binding == "external_parameter":
        parameters.append(torch.nn.Parameter(torch.ones(1)))
    else:
        parameters = parameters[:-1]
    optimizer = torch.optim.Adam(parameters)
    before = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="optimizer.*parameters"):
        load_checkpoint(path, model=target, optimizer=optimizer)
    assert all(torch.equal(before[key], value) for key, value in target.state_dict().items())
    assert not optimizer.state


@pytest.mark.parametrize("binding", ["missing_optimizer", "other_optimizer"])
def test_scheduler_requires_its_own_explicit_optimizer(tmp_path, binding):
    path = make_optimizer_checkpoint(tmp_path)
    target = make_model()
    optimizer = torch.optim.Adam(target.parameters())
    other_optimizer = torch.optim.Adam(target.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(other_optimizer, step_size=1)
    before = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="scheduler.*optimizer"):
        load_checkpoint(path, model=target,
            optimizer=optimizer if binding == "other_optimizer" else None, scheduler=scheduler)
    assert all(torch.equal(before[key], value) for key, value in target.state_dict().items())
    assert scheduler.last_epoch == 0


def test_correct_optimizer_scheduler_binding_resumes_and_updates_returned_model(tmp_path):
    path = make_optimizer_checkpoint(tmp_path)
    target = make_model()
    optimizer = torch.optim.Adam(target.parameters(), lr=.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.1)
    restored, _ = load_checkpoint(path, model=target, optimizer=optimizer, scheduler=scheduler)
    assert restored is target
    assert optimizer.state
    assert optimizer.param_groups[0]["lr"] == .0005
    assert scheduler.last_epoch == 1
    before = {key: value.clone() for key, value in restored.state_dict().items()}
    sum(parameter.square().sum() for parameter in restored.parameters()).backward()
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == .00025
    assert any(not torch.equal(before[key], value) for key, value in target.state_dict().items())


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_nested_equal(first, second)
    else:
        assert actual == expected


def training_bundle():
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.2)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer, scheduler


@pytest.mark.parametrize("groups", [1, 2])
def test_optimizer_duplicate_parameter_objects_are_rejected(tmp_path, groups):
    path = make_optimizer_checkpoint(tmp_path)
    model = make_model()
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam([{"params": parameters}] if groups == 1 else
        [{"params": parameters[:2]}, {"params": parameters[2:]}])
    optimizer.param_groups[-1]["params"].append(parameters[0])
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="optimizer.*duplicate"):
        load_checkpoint(path, model=model, optimizer=optimizer)
    assert_nested_equal(model.state_dict(), before)


def test_optimizer_requires_nonempty_trainable_model(tmp_path):
    path = make_optimizer_checkpoint(tmp_path)
    model = make_model().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError, match="requires_grad"):
        load_checkpoint(path, model=model, optimizer=optimizer)


@pytest.mark.parametrize("corruption", ["missing_optimizer", "missing_scheduler", "group_count", "group_size", "scheduler_binding"])
def test_state_preflight_failure_preserves_all_caller_states(tmp_path, corruption):
    path = make_optimizer_checkpoint(tmp_path)
    payload = torch.load(path, weights_only=False)
    if corruption == "missing_optimizer":
        payload["optimizer_state_dict"] = None
    elif corruption == "missing_scheduler":
        payload["scheduler_state_dict"] = None
    elif corruption == "group_count":
        payload["optimizer_state_dict"]["param_groups"].append(copy.deepcopy(payload["optimizer_state_dict"]["param_groups"][0]))
    elif corruption == "scheduler_binding":
        payload["scheduler_state_dict"]["optimizer"] = None
    else:
        payload["optimizer_state_dict"]["param_groups"][0]["params"].pop()
    torch.save(payload, path)
    model, optimizer, scheduler = training_bundle()
    before = copy.deepcopy([component.state_dict() for component in (model, optimizer, scheduler)])
    with pytest.raises(ValueError):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert_nested_equal([component.state_dict() for component in (model, optimizer, scheduler)], before)


@pytest.mark.parametrize("component_name", ["model", "optimizer", "scheduler"])
@pytest.mark.parametrize("override_kind", ["instance", "subclass"])
def test_custom_load_overrides_rejected_before_calling_user_code(tmp_path, component_name, override_kind):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    components = dict(model=model, optimizer=optimizer, scheduler=scheduler)
    failing_component = components[component_name]
    def unsupported_load(self, *args, **kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError("unsupported user code executed")
    if override_kind == "instance":
        failing_component.load_state_dict = types.MethodType(unsupported_load, failing_component)
    else:
        failing_component.__class__ = type("CustomLoader", (type(failing_component),), {"load_state_dict": unsupported_load})
    before = copy.deepcopy([component.state_dict() for component in components.values()])
    before_rng = capture_rng()
    with pytest.raises(ValueError, match="unsupported.*load_state_dict"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    after = [component.state_dict() for component in components.values()]
    if component_name == "scheduler":
        before[-1].pop("load_state_dict", None)
        after[-1].pop("load_state_dict", None)
    assert_nested_equal(after, before)
    assert scheduler.optimizer is optimizer
    assert_nested_equal(capture_rng(), before_rng)


def test_multiple_optimizer_groups_restore_and_step(tmp_path):
    source = make_model()
    params = list(source.parameters())
    optimizer = torch.optim.Adam([{"params": params[:3], "lr": .001}, {"params": params[3:], "lr": .002}])
    sum(parameter.square().sum() for parameter in params).backward()
    optimizer.step()
    path = save_checkpoint(tmp_path / "groups.pt", source, optimizer,
                           descriptor_stats=DescriptorNormalizer(torch.zeros(15), torch.ones(15)))
    target = make_model()
    params = list(target.parameters())
    restored_optimizer = torch.optim.Adam([{"params": params[:3]}, {"params": params[3:]}])
    restored, _ = load_checkpoint(path, model=target, optimizer=restored_optimizer)
    before = copy.deepcopy(restored.state_dict())
    sum(parameter.square().sum() for parameter in restored.parameters()).backward()
    restored_optimizer.step()
    assert restored is target
    assert [group["lr"] for group in restored_optimizer.param_groups] == [.001, .002]
    assert any(not torch.equal(before[key], value) for key, value in restored.state_dict().items())


def test_checkpoint_snapshot_rejects_mutable_content():
    from spectraexpert.training.checkpoints import CheckpointSnapshot
    with pytest.raises(TypeError, match="immutable bytes"):
        CheckpointSnapshot(bytearray(b"mutable"))


def test_standard_application_copy_failure_rolls_back_execution_state_and_rng(tmp_path, monkeypatch):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    model.eval()
    model.spectrum_encoder.train()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = copy.deepcopy([component.state_dict() for component in (model, optimizer, scheduler)])
    live_state_views = model.state_dict()
    modes = [module.training for module in model.modules()]
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    failing_parameter = list(model.parameters())[1]
    original_copy = torch.Tensor.copy_
    failed = False
    def fail_once(self, *args, **kwargs):
        nonlocal failed
        if self is failing_parameter and not failed:
            failed = True
            random.random()
            np.random.rand()
            torch.rand(1)
            raise RuntimeError("injected tensor copy failure")
        return original_copy(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "copy_", fail_once)
    before_rng = capture_rng()
    with pytest.raises(ValueError, match="injected tensor copy"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert_nested_equal([component.state_dict() for component in (model, optimizer, scheduler)], before)
    assert_nested_equal(live_state_views, before[0])
    assert_nested_equal([parameter.grad for parameter in model.parameters()], gradients)
    assert [module.training for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == requires_grad
    assert_nested_equal(capture_rng(), before_rng)


@pytest.mark.parametrize("hook_kind", ["module_pre", "module_post", "optimizer_pre", "optimizer_post"])
@pytest.mark.parametrize("restore", [False, True])
def test_registered_load_hooks_are_rejected_without_rng_changes(tmp_path, hook_kind, restore):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    def unsupported_hook(*args, **kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError("registered hook executed")
    owner = model.spectrum_encoder.projection[0] if hook_kind.startswith("module") else optimizer
    register = owner.register_load_state_dict_pre_hook if hook_kind.endswith("pre") else owner.register_load_state_dict_post_hook
    register(unsupported_hook)
    before = copy.deepcopy([component.state_dict() for component in (model, optimizer, scheduler)])
    before_rng = capture_rng()
    with pytest.raises(ValueError, match="unsupported.*hook"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=restore)
    assert_nested_equal([component.state_dict() for component in (model, optimizer, scheduler)], before)
    assert_nested_equal(capture_rng(), before_rng)


@pytest.mark.parametrize("restore", [False, True])
def test_loading_never_deepcopies_caller_objects(tmp_path, monkeypatch, restore):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    before = copy.deepcopy([component.state_dict() for component in (model, optimizer, scheduler)])
    original_deepcopy = copy.deepcopy
    caller_ids = {id(component) for component in list(model.modules()) + [optimizer, scheduler]}
    def forbid_objects(value, *args, **kwargs):
        if id(value) in caller_ids or (isinstance(value, tuple) and any(id(item) in caller_ids for item in value)):
            raise RuntimeError("caller object copied")
        return original_deepcopy(value, *args, **kwargs)
    monkeypatch.setattr(copy, "deepcopy", forbid_objects)
    before_rng = capture_rng()
    restored, payload = load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=restore)
    assert restored is model
    assert_nested_equal(model.state_dict(), payload["model_state_dict"])
    assert_nested_equal(capture_rng(), payload["rng_states"][0] if restore else before_rng)


def test_inference_load_without_rng_restore_preserves_entry_rng_and_execution_flags(tmp_path):
    path = make_optimizer_checkpoint(tmp_path)
    model = make_model().eval()
    model.spectrum_encoder.train()
    next(model.parameters()).requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    before_rng = capture_rng()
    restored, _ = load_checkpoint(path, model=model, restore_rng=False)
    assert restored is model
    assert [module.training for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert_nested_equal([parameter.grad for parameter in model.parameters()], gradients)
    assert_nested_equal(capture_rng(), before_rng)


@pytest.mark.parametrize("extension", ["_load_from_state_dict", "set_extra_state", "__setstate__"])
@pytest.mark.parametrize("override_kind", ["instance", "subclass"])
def test_dynamic_restore_extensions_rejected_before_user_code(tmp_path, extension, override_kind):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    owner = optimizer if extension == "__setstate__" else model.spectrum_encoder.projection[0]
    def unsupported_extension(*args, **kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError("dynamic extension executed")
    if override_kind == "instance":
        setattr(owner, extension, types.MethodType(unsupported_extension, owner))
    else:
        owner.__class__ = type("CustomRestoreExtension", (type(owner),), {extension: unsupported_extension})
    before = copy.deepcopy([component.state_dict() for component in (model, optimizer, scheduler)])
    before_rng = capture_rng()
    with pytest.raises(ValueError, match=f"unsupported.*{extension}"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert_nested_equal([component.state_dict() for component in (model, optimizer, scheduler)], before)
    assert_nested_equal(capture_rng(), before_rng)


@pytest.mark.parametrize("kind", ["plateau", "sequential", "chained"])
def test_official_scheduler_overrides_restore_and_continue(tmp_path, kind):
    def make_scheduler(optimizer):
        schedulers = torch.optim.lr_scheduler
        if kind == "plateau":
            return schedulers.ReduceLROnPlateau(optimizer, patience=0, factor=.5)
        children = [schedulers.StepLR(optimizer, step_size=1, gamma=.5),
                    schedulers.ExponentialLR(optimizer, gamma=.8)]
        return (schedulers.SequentialLR(optimizer, children, milestones=[2]) if kind == "sequential"
                else schedulers.ChainedScheduler(children, optimizer=optimizer))
    source = make_model()
    optimizer = torch.optim.Adam(source.parameters(), lr=.02)
    scheduler = make_scheduler(optimizer)
    for _ in range(3):
        optimizer.zero_grad()
        sum(parameter.square().sum() for parameter in source.parameters()).backward()
        optimizer.step()
        scheduler.step(1.) if kind == "plateau" else scheduler.step()
    path = save_checkpoint(tmp_path / "official.pt", source, optimizer, scheduler=scheduler,
                           descriptor_stats=DescriptorNormalizer(torch.zeros(15), torch.ones(15)))
    model = make_model()
    restored_optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    restored_scheduler = make_scheduler(restored_optimizer)
    before_rng = capture_rng()
    restored, _ = load_checkpoint(path, model=model, optimizer=restored_optimizer, scheduler=restored_scheduler)
    assert restored is model
    assert_nested_equal(restored_scheduler.state_dict(), scheduler.state_dict())
    assert_nested_equal(capture_rng(), before_rng)
    for current_optimizer, current_scheduler in ((optimizer, scheduler), (restored_optimizer, restored_scheduler)):
        current_optimizer.step()
        current_scheduler.step(1.) if kind == "plateau" else current_scheduler.step()
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert_nested_equal(restored_scheduler.state_dict(), scheduler.state_dict())


@pytest.mark.parametrize("available,saved_count", [(True, 0), (True, 2), (False, 1), (True, 1)])
@pytest.mark.parametrize("restore", [False, True])
def test_cuda_rng_topology_checked_symmetrically_without_real_cuda(tmp_path, monkeypatch, available, saved_count, restore):
    path = make_optimizer_checkpoint(tmp_path)
    payload = torch.load(path, weights_only=False)
    payload["rng_states"][0]["cuda"] = [torch.tensor([42], dtype=torch.uint8) for _ in range(saved_count)]
    torch.save(payload, path)
    model, optimizer, scheduler = training_bundle()
    fake_cuda = [torch.tensor([17], dtype=torch.uint8)] if available else []
    def set_cuda(states):
        fake_cuda[:] = [state.clone() for state in states]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1 if available else 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [state.clone() for state in fake_cuda])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_cuda)
    before = copy.deepcopy([component.state_dict() for component in (model, optimizer, scheduler)])
    before_rng = capture_rng()
    mismatch = (available and saved_count != 1) or (not available and saved_count != 0)
    if restore and mismatch:
        with pytest.raises(ValueError, match="rng_states.cuda"):
            load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=True)
        assert_nested_equal([component.state_dict() for component in (model, optimizer, scheduler)], before)
    else:
        restored, _ = load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=restore)
        assert restored is model
    assert_nested_equal(capture_rng(), payload["rng_states"][0] if restore and not mismatch else before_rng)


@pytest.mark.parametrize("extension", ["_init_is_better", "__setstate__", "child_load"])
def test_scheduler_restore_extensions_rejected_recursively(tmp_path, extension):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, _ = training_bundle()
    if extension == "child_load":
        child = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([child], optimizer=optimizer)
        owner, method = child, "load_state_dict"
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        owner, method = scheduler, extension
    def unsupported_extension(*args, **kwargs):
        raise RuntimeError("scheduler restore extension executed")
    owner.__class__ = type("CustomSchedulerExtension", (type(owner),), {method: unsupported_extension})
    before = copy.deepcopy([component.state_dict() for component in (model, optimizer, scheduler)])
    before_rng = capture_rng()
    with pytest.raises(ValueError, match=f"unsupported.*{method}"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert_nested_equal([component.state_dict() for component in (model, optimizer, scheduler)], before)
    assert_nested_equal(capture_rng(), before_rng)


def test_sequential_scheduler_rollback_preserves_child_objects(tmp_path, monkeypatch):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, _ = training_bundle()
    children = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5),
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.8)]
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, children, milestones=[2])
    payload = torch.load(path, weights_only=False)
    payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
    before = copy.deepcopy([component.state_dict() for component in (model, optimizer, scheduler)])
    failing_parameter = list(model.parameters())[1]
    original_copy = torch.Tensor.copy_
    failed = False
    def fail_once(self, *args, **kwargs):
        nonlocal failed
        if self is failing_parameter and not failed:
            failed = True
            raise RuntimeError("sequential rollback injected copy failure")
        return original_copy(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "copy_", fail_once)
    with pytest.raises(ValueError, match="injected copy failure"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert scheduler._schedulers == children
    assert all(child.optimizer is optimizer for child in children)
    assert_nested_equal([component.state_dict() for component in (model, optimizer, scheduler)], before)


def test_cuda_topology_failure_precedes_all_rng_writes(monkeypatch):
    from spectraexpert.training.checkpoints import restore_rng_state
    entry = capture_rng()
    invalid = copy.deepcopy(entry)
    invalid["cuda"] = [torch.tensor([42], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    def forbidden_write(*args, **kwargs):
        raise RuntimeError("RNG written before topology validation")
    monkeypatch.setattr(random, "setstate", forbidden_write)
    monkeypatch.setattr(np.random, "set_state", forbidden_write)
    monkeypatch.setattr(torch, "set_rng_state", forbidden_write)
    with pytest.raises(ValueError, match="rng_states.cuda"):
        restore_rng_state(invalid)


@pytest.mark.parametrize("owner_kind", ["root", "nested", "optimizer", "scheduler", "child_scheduler"])
@pytest.mark.parametrize("method", ["__deepcopy__", "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__"])
@pytest.mark.parametrize("override_kind", ["instance", "subclass"])
def test_copy_protocol_extensions_rejected_before_any_copy(tmp_path, owner_kind, method, override_kind):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    owners = dict(root=model, nested=model.spectrum_encoder.projection[0], optimizer=optimizer, scheduler=scheduler)
    if owner_kind == "child_scheduler":
        child = scheduler
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([child], optimizer=optimizer)
        owners[owner_kind] = child
    owner = owners[owner_kind]
    before = copy.deepcopy([component.state_dict() for component in (model, optimizer, scheduler)])
    def unsafe_copy(*args, **kwargs):
        owner.copy_hook_ran = True
        with torch.no_grad():
            next(model.parameters()).fill_(99)
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError("unsafe copy protocol executed")
    if override_kind == "instance":
        setattr(owner, method, types.MethodType(unsafe_copy, owner))
    else:
        owner.__class__ = type("CustomCopyProtocol", (type(owner),), {method: unsafe_copy})
    before_rng = capture_rng()
    with pytest.raises(ValueError, match=f"unsupported.*{method}"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    def without_hook(value):
        if isinstance(value, dict):
            return {key: without_hook(item) for key, item in value.items() if key != method}
        if isinstance(value, list):
            return [without_hook(item) for item in value]
        return value
    assert not hasattr(owner, "copy_hook_ran")
    assert_nested_equal(without_hook([component.state_dict() for component in (model, optimizer, scheduler)]), before)
    assert_nested_equal(capture_rng(), before_rng)


@pytest.mark.parametrize("method", ["__getnewargs_ex__", "__getnewargs__", "__new__", "named_modules"])
def test_unused_construction_and_traversal_extensions_never_called(tmp_path, method):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    def forbidden(*args, **kwargs):
        model.extension_called = True
        raise RuntimeError("construction or traversal extension called")
    model.__class__ = type("UnusedExtension", (type(model),), {method: forbidden})
    restored, _ = load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert restored is model
    assert not hasattr(model, "extension_called")


def test_descriptor_method_is_rejected_without_evaluation(tmp_path):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    def forbidden_descriptor(self):
        model.descriptor_called = True
        raise RuntimeError("descriptor evaluated")
    model.__class__ = type("DescriptorLoader", (type(model),), {"load_state_dict": property(forbidden_descriptor)})
    before = copy.deepcopy([component.state_dict() for component in (model, optimizer, scheduler)])
    before_rng = capture_rng()
    with pytest.raises(ValueError, match="unsupported.*load_state_dict"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert not hasattr(model, "descriptor_called")
    assert_nested_equal([component.state_dict() for component in (model, optimizer, scheduler)], before)
    assert_nested_equal(capture_rng(), before_rng)


def test_parameter_subclass_rejected_without_constructing_it(tmp_path):
    path = make_optimizer_checkpoint(tmp_path)
    model = make_model()
    parameter = next(model.parameters())
    def forbidden_new(*args, **kwargs):
        raise RuntimeError("Parameter subclass constructed")
    parameter.__class__ = type("UserParameter", (torch.nn.Parameter,), {"__new__": forbidden_new})
    before = parameter.detach().clone()
    before_rng = capture_rng()
    with pytest.raises(ValueError, match="parameter.*type"):
        load_checkpoint(path, model=model)
    assert torch.equal(parameter, before)
    assert_nested_equal(capture_rng(), before_rng)


@pytest.mark.parametrize("method", ["state_dict", "_save_to_state_dict", "get_extra_state"])
def test_model_save_extensions_rejected_before_backup(tmp_path, method):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    before = copy.deepcopy(model.state_dict())
    owner = model.spectrum_encoder.projection[0]
    def forbidden(*args, **kwargs):
        owner.save_extension_called = True
        raise RuntimeError("save extension called")
    owner.__class__ = type("SaveExtension", (type(owner),), {method: forbidden})
    before_rng = capture_rng()
    with pytest.raises(ValueError, match=f"unsupported.*{method}"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert not hasattr(owner, "save_extension_called")
    owner.__class__ = torch.nn.Linear
    assert_nested_equal(model.state_dict(), before)
    assert_nested_equal(capture_rng(), before_rng)


@pytest.mark.parametrize("kind", ["module_pre", "module_post", "optimizer_pre", "optimizer_post"])
def test_registered_save_hooks_rejected_before_backup(tmp_path, kind):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    before = copy.deepcopy([component.state_dict() for component in (model, optimizer, scheduler)])
    owner = model if kind.startswith("module") else optimizer
    def forbidden(*args, **kwargs):
        raise RuntimeError("save hook called")
    register = owner.register_state_dict_pre_hook if kind.endswith("pre") else owner.register_state_dict_post_hook
    handle = register(forbidden)
    before_rng = capture_rng()
    with pytest.raises(ValueError, match="unsupported.*hook"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    handle.remove()
    assert_nested_equal([component.state_dict() for component in (model, optimizer, scheduler)], before)
    assert_nested_equal(capture_rng(), before_rng)


def test_method_descriptor_is_not_executed_during_static_inspection(tmp_path):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    class LoaderDescriptor:
        def __get__(self, instance, owner):
            model.descriptor_called = True
            raise RuntimeError("class-level descriptor invoked")
    model.__class__ = type("DescriptorMethod", (type(model),), {"load_state_dict": LoaderDescriptor()})
    before_rng = capture_rng()
    with pytest.raises(ValueError, match="unsupported.*load_state_dict"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert not hasattr(model, "descriptor_called")
    assert_nested_equal(capture_rng(), before_rng)


@pytest.mark.parametrize("owner_kind", ["optimizer", "scheduler"])
def test_non_data_state_rejected_without_copying_user_object(tmp_path, owner_kind):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    class UserState:
        def __deepcopy__(self, memo):
            raise RuntimeError("user state copied")
    user_state = UserState()
    if owner_kind == "optimizer":
        optimizer.param_groups[0]["custom"] = user_state
    else:
        scheduler.custom = user_state
    before_model = copy.deepcopy(model.state_dict())
    before_rng = capture_rng()
    with pytest.raises(ValueError, match="unsupported state data"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert_nested_equal(model.state_dict(), before_model)
    assert_nested_equal(capture_rng(), before_rng)


@pytest.mark.parametrize("corruption", ["model_shape", "optimizer_shape", "scheduler_layout"])
def test_tensor_layout_preflight_rejects_before_any_application(tmp_path, corruption, monkeypatch):
    path = make_optimizer_checkpoint(tmp_path)
    payload = torch.load(path, weights_only=False)
    if corruption == "model_shape":
        key = next(iter(payload["model_state_dict"]))
        payload["model_state_dict"][key] = torch.zeros(1)
    elif corruption == "optimizer_shape":
        state = next(iter(payload["optimizer_state_dict"]["state"].values()))
        state["exp_avg"] = torch.zeros(1)
    else:
        payload["scheduler_state_dict"]["gamma"] = {"not": "a scalar"}
    torch.save(payload, path)
    model, optimizer, scheduler = training_bundle()
    original_copy = torch.Tensor.copy_
    ids = {id(parameter) for parameter in model.parameters()}
    copied_target = False
    def forbid_target_copy(self, *args, **kwargs):
        nonlocal copied_target
        if id(self) in ids:
            copied_target = True
            raise RuntimeError("target mutated before layout validation")
        return original_copy(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "copy_", forbid_target_copy)
    before_rng = capture_rng()
    expected = "scheduler_state_dict.*layout" if corruption == "scheduler_layout" else "shape"
    with pytest.raises(ValueError, match=expected):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert not copied_target
    assert_nested_equal(capture_rng(), before_rng)


@pytest.mark.parametrize("container", ["parameter_group", "scheduler_children"])
def test_user_state_containers_rejected_before_iteration(tmp_path, container):
    path = make_optimizer_checkpoint(tmp_path)
    model, optimizer, scheduler = training_bundle()
    calls = []
    if container == "parameter_group":
        class UserGroup(dict):
            def items(self):
                calls.append("items")
                raise RuntimeError("user container iterated")
        optimizer.param_groups[0] = UserGroup(optimizer.param_groups[0])
    else:
        class UserChildren(list):
            def __iter__(self):
                calls.append("iter")
                raise RuntimeError("user container iterated")
        scheduler._schedulers = UserChildren()
    before_rng = capture_rng()
    with pytest.raises(ValueError, match="unsupported.*container"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler)
    assert calls == []
    assert_nested_equal(capture_rng(), before_rng)

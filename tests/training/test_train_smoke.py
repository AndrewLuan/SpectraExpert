"""功能概述：用临时三件套 Parquet 运行单 CPU 的两种训练模式和恢复流程。
主要职责：fixture 固定分子、光谱和 split；验证 train-only 统计量、graph 梯度更新、
v1 独立候选光谱模式、相同 seed 的连续/恢复训练、来源哈希和空训练保护；恢复到已有
目录时拒绝 last 内容冲突并保持原文件，同时接受同路径或逐字节相同的 checkpoint。
原子替换来源文件验证已加载快照不会混入替换后的状态，并验证 symlink 物化和 best 来源复用。
真正 best 状态要求分数与 best_score 相同且 bad_epochs 为零；已有 best 的字节冲突先拒绝，
等字节链接按快照物化，平分后的 last 仍保留独立历史 best。
已有或 sibling historical best 的分数不一致、NaN、选择键缺失均在写入前拒绝，目录原字节不变。
两进程 gloo 用固定 CPU topology/fake CUDA RNG 仅破坏 rank 1，确认相同应用错误无 timeout，
新目录不创建、已有目录的文件名/字节/symlink 类型不变；正常 DDP resume 仍逐参数一致。
weakref 与训练帧检查确认快照和完整 resume payload 不驻留 epoch 循环；物化失败也清空列表及循环变量。
代码库关系：覆盖 Task 4–6 到 train_structure_aware 的真实集成；不读取项目大型数据目录。
输入输出与副作用：写入 tmp_path 内微型数据、日志、best/last checkpoint。
"""
import dataclasses
import copy
import hashlib
import json
import shutil
import os
import weakref
import inspect
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spectraexpert.data.structure_dataset import canonical_smiles_sha256
from spectraexpert.train_structure_aware import TrainConfig, train


@pytest.fixture
def tiny_dataset_dir(tmp_path):
    torch.set_num_threads(1)
    directory = tmp_path / "data"
    directory.mkdir()
    smiles = ["C", "CC", "CCC", "CCO", "CCN", "COC", "CO", "CN", "C=C", "C#N", "O", "N"]
    values = np.stack([np.random.default_rng(i).normal(size=65).astype(np.float32) for i in range(12)])
    frame = pd.DataFrame(values, columns=[f"i{i}" for i in range(65)])
    frame.insert(0, "mol_id", range(12))
    frame.to_parquet(directory / "spectra.parquet")
    pd.DataFrame({"mol_id": range(12), "smiles": smiles}).to_parquet(directory / "smiles.parquet")
    pd.DataFrame({"wavenumber": np.linspace(500, 4000, 65)}).to_parquet(directory / "wavenumber.parquet")
    manifest = dict(schema_version="scaffold_split_v1", seed=0,
        source_smiles_sha256=canonical_smiles_sha256(list(range(12)), smiles),
        train=list(range(8)), validation=[8, 9], test=[10, 11],
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.parquet")})
    (directory / "split.json").write_text(json.dumps(manifest))
    return directory


@pytest.mark.parametrize("mode", ["spectral_only_v1", "spectrum_graph_v2"])
def test_one_epoch_smoke_writes_best_and_last(tiny_dataset_dir, tmp_path, mode):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / mode,
                            training_mode=mode, batch_size=3)
    result = train(config)
    assert result["global_steps"] == 3
    checkpoint = torch.load(config.output_dir / "last.pt", weights_only=False)
    assert (config.output_dir / "best.pt").is_file()
    assert checkpoint["schema_version"] == mode
    assert checkpoint["candidate_modality"] == ("spectrum" if mode == "spectral_only_v1" else "graph")
    # Eight train molecules have C counts 1,2,3,2,2,2,1,1; val/test must not enter fit.
    assert checkpoint["descriptor_stats"]["mean"][0].item() == 1.75
    keys = checkpoint["model_state_dict"]
    assert any(k.startswith("graph_encoder.") for k in keys) == (mode == "spectrum_graph_v2")
    log = json.loads((config.output_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert set(log["losses"]) == {"total", "aug", "cross", "struct", "descriptor", "presence", "fingerprint"}
    assert log["validation"]["top1_tanimoto"] >= 0
    # Val is C=C versus C#N: their Morgan bits are disjoint. Selection must exclude self.
    assert log["validation"]["top1_tanimoto"] == 0.
    assert checkpoint["selection_exclude_identity"] is True
    if mode == "spectral_only_v1":
        assert log["validation"]["inclusive_top1_tanimoto"] == 1.
    if mode == "spectrum_graph_v2":
        assert result["graph_gradient_norm"] > 0


@pytest.mark.parametrize("mode", ["spectral_only_v1", "spectrum_graph_v2"])
def test_resume_matches_uninterrupted_seeded_training(tiny_dataset_dir, tmp_path, mode):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "full", training_mode=mode, epochs=2)
    train(config)
    partial = dataclasses.replace(config, output_dir=tmp_path / "resume", epochs=1)
    train(partial)
    resumed = dataclasses.replace(partial, epochs=2, resume_from=partial.output_dir / "last.pt")
    result = train(resumed)
    a = torch.load(config.output_dir / "last.pt", weights_only=False)
    b = torch.load(partial.output_dir / "last.pt", weights_only=False)
    assert result["global_steps"] == a["global_steps"] == b["global_steps"]
    assert a["epoch"] == b["epoch"] == 1
    for key in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][key], b["model_state_dict"][key]), key


def test_resume_rejects_loss_and_manifest_changes(tiny_dataset_dir, tmp_path):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "run")
    train(config)
    resumed = dataclasses.replace(config, epochs=2, resume_from=config.output_dir / "last.pt", lambda_struct=.2)
    with pytest.raises(ValueError, match="loss_weights"):
        train(resumed)
    manifest = tiny_dataset_dir / "split.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="split_manifest_sha256"):
        train(dataclasses.replace(config, epochs=2, resume_from=config.output_dir / "last.pt"))


def test_rejects_empty_training_selection(tiny_dataset_dir, tmp_path):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "run", max_samples=0)
    with pytest.raises(ValueError, match="max_samples"):
        train(config)


def test_selection_without_neighbors_fails_before_creating_run(tiny_dataset_dir, tmp_path):
    manifest_path = tiny_dataset_dir / "split.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(validation=[8], test=[9, 10, 11])
    manifest_path.write_text(json.dumps(manifest))
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "invalid-val")
    with pytest.raises(ValueError, match="eligible"):
        train(config)
    assert not config.output_dir.exists()


@pytest.mark.parametrize("weights", [(0., 0., 0.), (1., 1., .25)])
def test_v2_ablation_and_optional_fingerprint_smoke(tiny_dataset_dir, tmp_path, weights):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "ablation",
        lambda_struct=weights[0], lambda_aux=weights[1], lambda_fingerprint=weights[2])
    result = train(config)
    payload = torch.load(config.output_dir / "last.pt", weights_only=False)
    assert result["graph_gradient_norm"] > 0
    assert payload["model_config"]["predict_fingerprint"] == (weights[2] > 0)
    assert any(key.startswith("fingerprint_head.") for key in payload["model_state_dict"]) == (weights[2] > 0)
    assert np.isfinite(list(result["history"][0]["losses"].values())).all()


def test_resume_new_output_keeps_historical_best(tiny_dataset_dir, tmp_path):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "first",
                             training_mode="spectral_only_v1", patience=2)
    train(config)
    resumed = dataclasses.replace(config, epochs=4, patience=2, output_dir=tmp_path / "new",
                                  resume_from=config.output_dir / "last.pt")
    result = train(resumed)
    best = torch.load(resumed.output_dir / "best.pt", weights_only=False)
    last = torch.load(resumed.output_dir / "last.pt", weights_only=False)
    assert best["epoch"] == 0
    assert last["epoch"] == 2
    assert result["global_steps"] == 6


def test_completed_resume_new_directory_preserves_best_and_last(tiny_dataset_dir, tmp_path):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "first",
                             training_mode="spectral_only_v1", epochs=2)
    train(config)
    resumed = dataclasses.replace(config, output_dir=tmp_path / "new",
                                  resume_from=config.output_dir / "last.pt")
    result = train(resumed)
    assert result["history"] == []
    assert torch.load(resumed.output_dir / "last.pt", weights_only=False)["epoch"] == 1
    assert torch.load(resumed.output_dir / "best.pt", weights_only=False)["epoch"] == 0


def test_resume_rejects_existing_best_from_different_mode(tiny_dataset_dir, tmp_path):
    first = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "v1",
                            training_mode="spectral_only_v1")
    second = dataclasses.replace(first, output_dir=tmp_path / "v2", training_mode="spectrum_graph_v2")
    train(first)
    train(second)
    with pytest.raises(ValueError, match="training_mode"):
        train(dataclasses.replace(first, output_dir=second.output_dir, epochs=2,
                                  resume_from=first.output_dir / "last.pt"))


@pytest.mark.parametrize("keep_best", [True, False])
def test_resume_rejects_conflicting_destination_last_without_writing(tiny_dataset_dir, tmp_path, keep_best):
    source = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source",
                              training_mode="spectral_only_v1", epochs=2)
    destination = dataclasses.replace(source, output_dir=tmp_path / "destination", epochs=3)
    train(source)
    train(destination)
    if not keep_best:
        (destination.output_dir / "best.pt").unlink()
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in destination.output_dir.iterdir()}
    with pytest.raises(ValueError, match="last.pt.*conflict"):
        train(dataclasses.replace(source, output_dir=destination.output_dir,
                                  resume_from=source.output_dir / "last.pt"))
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in destination.output_dir.iterdir()}
    assert after == before


def test_resume_accepts_identical_checkpoint_in_different_directory(tiny_dataset_dir, tmp_path):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source", epochs=2)
    train(config)
    destination = tmp_path / "destination"
    destination.mkdir()
    for name in ("best.pt", "last.pt"):
        shutil.copyfile(config.output_dir / name, destination / name)
    result = train(dataclasses.replace(config, output_dir=destination,
                                      resume_from=config.output_dir / "last.pt"))
    assert result["history"] == []
    assert result["global_steps"] == 4
    assert (destination / "last.pt").read_bytes() == (config.output_dir / "last.pt").read_bytes()


@pytest.mark.parametrize("same_destination", [False, True])
def test_resume_uses_loaded_snapshot_when_source_is_atomically_replaced(tiny_dataset_dir, tmp_path, monkeypatch, same_destination):
    import spectraexpert.train_structure_aware as entry
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source", epochs=2)
    train(config)
    replacement = dataclasses.replace(config, output_dir=tmp_path / "replacement", epochs=3)
    train(replacement)
    original_bytes = (config.output_dir / "last.pt").read_bytes()
    replacement_bytes = (replacement.output_dir / "last.pt").read_bytes()
    destination = config.output_dir if same_destination else tmp_path / "destination"
    if not same_destination:
        destination.mkdir()
        for name in ("best.pt", "last.pt"):
            shutil.copyfile(config.output_dir / name, destination / name)
    original_loader = entry.load_checkpoint
    replaced = False
    def replace_after_loading(*args, **kwargs):
        nonlocal replaced
        result = original_loader(*args, **kwargs)
        if not replaced:
            replaced = True
            os.replace(replacement.output_dir / "last.pt", config.output_dir / "last.pt")
        return result
    monkeypatch.setattr(entry, "load_checkpoint", replace_after_loading)
    resumed = dataclasses.replace(config, output_dir=destination, resume_from=config.output_dir / "last.pt")
    if same_destination:
        with pytest.raises(ValueError, match="last.pt.*conflict"):
            train(resumed)
        assert (destination / "last.pt").read_bytes() == replacement_bytes
    else:
        result = train(resumed)
        assert result["global_steps"] == 4
        assert (destination / "last.pt").read_bytes() == original_bytes


def test_resume_materializes_destination_symlink_as_snapshot_file(tiny_dataset_dir, tmp_path):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source", epochs=2)
    train(config)
    destination = tmp_path / "destination"
    destination.mkdir()
    shutil.copyfile(config.output_dir / "best.pt", destination / "best.pt")
    (destination / "last.pt").symlink_to(config.output_dir / "last.pt")
    expected = (config.output_dir / "last.pt").read_bytes()
    result = train(dataclasses.replace(config, output_dir=destination, resume_from=destination / "last.pt"))
    assert result["global_steps"] == 4
    assert not (destination / "last.pt").is_symlink()
    assert (destination / "last.pt").read_bytes() == expected


def test_resume_from_best_uses_snapshot_for_both_delivered_files(tiny_dataset_dir, tmp_path, monkeypatch):
    import spectraexpert.train_structure_aware as entry
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source")
    train(config)
    source = config.output_dir / "best.pt"
    expected = source.read_bytes()
    changed = torch.load(source, weights_only=False)
    first_key = next(iter(changed["model_state_dict"]))
    changed["model_state_dict"][first_key] += 1
    replacement = tmp_path / "replacement.pt"
    torch.save(changed, replacement)
    original_loader = entry.load_checkpoint
    replaced = False
    def replace_after_loading(*args, **kwargs):
        nonlocal replaced
        result = original_loader(*args, **kwargs)
        if not replaced:
            replaced = True
            os.replace(replacement, source)
        return result
    monkeypatch.setattr(entry, "load_checkpoint", replace_after_loading)
    destination = tmp_path / "destination"
    train(dataclasses.replace(config, output_dir=destination, resume_from=source))
    assert (destination / "best.pt").read_bytes() == expected
    assert (destination / "last.pt").read_bytes() == expected


def test_existing_best_with_same_score_but_different_weights_is_rejected(tiny_dataset_dir, tmp_path):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source")
    train(config)
    destination = tmp_path / "destination"
    destination.mkdir()
    shutil.copyfile(config.output_dir / "last.pt", destination / "last.pt")
    changed = torch.load(config.output_dir / "best.pt", weights_only=False)
    changed["model_state_dict"][next(iter(changed["model_state_dict"]))] += 1
    torch.save(changed, destination / "best.pt")
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    with pytest.raises(ValueError, match="best.pt.*conflict"):
        train(dataclasses.replace(config, output_dir=destination, resume_from=config.output_dir / "last.pt"))
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before


@pytest.mark.parametrize("replace_alias", [False, True])
def test_existing_best_alias_is_materialized_or_rejected_if_replaced(tiny_dataset_dir, tmp_path, monkeypatch, replace_alias):
    import spectraexpert.train_structure_aware as entry
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source")
    train(config)
    source = config.output_dir / "best.pt"
    expected = source.read_bytes()
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "best.pt").symlink_to(source)
    shutil.copyfile(config.output_dir / "last.pt", destination / "last.pt")
    changed = torch.load(source, weights_only=False)
    changed["model_state_dict"][next(iter(changed["model_state_dict"]))] += 1
    replacement = tmp_path / "replacement.pt"
    torch.save(changed, replacement)
    replacement_bytes = replacement.read_bytes()
    original_loader = entry.load_checkpoint
    replaced = False
    def replace_after_loading(*args, **kwargs):
        nonlocal replaced
        result = original_loader(*args, **kwargs)
        if replace_alias and not replaced:
            replaced = True
            os.replace(replacement, source)
        return result
    monkeypatch.setattr(entry, "load_checkpoint", replace_after_loading)
    resumed = dataclasses.replace(config, output_dir=destination, resume_from=source)
    if replace_alias:
        with pytest.raises(ValueError, match="best.pt.*conflict"):
            train(resumed)
        assert (destination / "best.pt").is_symlink()
        assert (destination / "best.pt").read_bytes() == replacement_bytes
        assert (destination / "last.pt").read_bytes() == expected
    else:
        train(resumed)
        assert not (destination / "best.pt").is_symlink()
        assert (destination / "best.pt").read_bytes() == expected


def test_inconsistent_best_selection_metadata_is_rejected(tiny_dataset_dir, tmp_path):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source")
    train(config)
    source = config.output_dir / "last.pt"
    changed = torch.load(source, weights_only=False)
    changed["validation"]["top1_tanimoto"] = .25
    torch.save(changed, source)
    with pytest.raises(ValueError, match="selection.*inconsistent"):
        train(dataclasses.replace(config, output_dir=tmp_path / "destination", resume_from=source))


@pytest.mark.parametrize("location", ["existing", "sibling"])
@pytest.mark.parametrize("corruption", ["different_score", "nan_score", "missing_metric", "nan_best"])
def test_historical_best_invariant_rejected_before_any_write(tiny_dataset_dir, tmp_path, location, corruption):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source", epochs=2)
    train(config)
    last = torch.load(config.output_dir / "last.pt", weights_only=False)
    assert last["bad_epochs"] > 0
    destination = tmp_path / "destination"
    destination.mkdir()
    shutil.copyfile(config.output_dir / "last.pt", destination / "last.pt")
    best_path = destination / "best.pt" if location == "existing" else config.output_dir / "best.pt"
    payload = torch.load(config.output_dir / "best.pt", weights_only=False)
    if corruption == "different_score":
        payload["validation"]["top1_tanimoto"] = -.25
    elif corruption == "nan_score":
        payload["validation"]["top1_tanimoto"] = float("nan")
    elif corruption == "missing_metric":
        payload["validation"].pop("top1_tanimoto")
    else:
        payload["best_score"] = float("nan")
    torch.save(payload, best_path)
    before = {str(p): p.read_bytes() for directory in (destination, config.output_dir) for p in directory.iterdir()}
    with pytest.raises(ValueError):
        train(dataclasses.replace(config, output_dir=destination, resume_from=config.output_dir / "last.pt"))
    assert {str(p): p.read_bytes() for directory in (destination, config.output_dir) for p in directory.iterdir()} == before


def test_resume_cuda_rng_mismatch_rejected_before_checkpoint_materialization(tiny_dataset_dir, tmp_path):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source")
    train(config)
    source = config.output_dir / "last.pt"
    payload = torch.load(source, weights_only=False)
    payload["rng_states"][0]["cuda"] = [torch.tensor([42], dtype=torch.uint8)]
    torch.save(payload, source)
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="rng_states.cuda"):
        train(dataclasses.replace(config, output_dir=destination, resume_from=source))
    assert not destination.exists()


def test_resume_releases_snapshots_and_large_payload_before_training(tiny_dataset_dir, tmp_path, monkeypatch):
    import spectraexpert.train_structure_aware as entry
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source", epochs=2)
    train(config)
    references = []
    original_read = entry.read_checkpoint_snapshot
    def record_snapshot(*args, **kwargs):
        snapshot = original_read(*args, **kwargs)
        references.append(weakref.ref(snapshot))
        return snapshot
    original_validate = entry.validate_retrieval
    def validate_after_release(*args, **kwargs):
        assert references and all(reference() is None for reference in references)
        frame = inspect.currentframe().f_back
        assert "payload" not in frame.f_locals or not any(
            key in frame.f_locals["payload"] for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict"))
        return original_validate(*args, **kwargs)
    monkeypatch.setattr(entry, "read_checkpoint_snapshot", record_snapshot)
    monkeypatch.setattr(entry, "validate_retrieval", validate_after_release)
    train(dataclasses.replace(config, output_dir=tmp_path / "destination", epochs=3,
                              resume_from=config.output_dir / "last.pt"))


def test_pending_snapshots_cleared_when_materialization_fails(tmp_path, monkeypatch):
    import spectraexpert.train_structure_aware as entry
    from spectraexpert.training.checkpoints import CheckpointSnapshot
    pending = [(CheckpointSnapshot(b"content"), tmp_path / "last.pt")]
    def fail(*args, **kwargs):
        raise OSError("injected materialization failure")
    monkeypatch.setattr(entry, "materialize_checkpoint", fail)
    with pytest.raises(OSError) as caught:
        entry.materialize_pending_checkpoints(pending)
    assert pending == []
    traceback = caught.value.__traceback__
    while traceback:
        if traceback.tb_frame.f_code.co_name == "materialize_pending_checkpoints":
            assert traceback.tb_frame.f_locals["snapshot"] is None
            assert traceback.tb_frame.f_locals["destination"] is None
        traceback = traceback.tb_next


def _ddp_worker(rank, store, config, queue, fail_batch=False, resume=False):
    torch.set_num_threads(1)
    try:
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank,
                                world_size=2, timeout=timedelta(seconds=15))
        if fail_batch and rank == 1:
            from spectraexpert.data.structure_dataset import StructureAwareSpectralDataset
            def broken_item(self, index):
                raise ValueError("injected invalid training row")
            StructureAwareSpectralDataset.__getitem__ = broken_item
        result = train(config)
        if resume:
            partial = dataclasses.replace(config, output_dir=config.output_dir.parent / "resumed", epochs=1)
            train(partial)
            train(dataclasses.replace(partial, epochs=2, resume_from=partial.output_dir / "last.pt"))
        queue.put((rank, "ok", result["global_steps"], result["graph_gradient_norm"]))
    except Exception as error:
        queue.put((rank, "error", str(error)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("mode,fail_batch,resume", [("spectral_only_v1", False, False),
    ("spectrum_graph_v2", False, True), ("spectrum_graph_v2", True, False)])
def test_ddp_equal_short_batches_and_rank_failure(tiny_dataset_dir, tmp_path, mode, fail_batch, resume):
    config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "ddp",
                             training_mode=mode, ddp=True, batch_size=3, epochs=2)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_ddp_worker,
        args=(rank, str(tmp_path / "store"), config, queue, fail_batch, resume)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(25)
        assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
        results = sorted(queue.get(timeout=2) for _ in range(2))
        if fail_batch:
            assert all(result[1] == "error" for result in results)
            assert results[0][2] == results[1][2]
            assert "rank 1 ValueError: injected invalid training row" in results[0][2]
            assert "timeout" not in results[0][2].lower()
            assert not (config.output_dir / "last.pt").exists()
        else:
            assert all(result[1] == "ok" for result in results), results
            assert all(result[2] == 4 for result in results)
            payload = torch.load(config.output_dir / "last.pt", weights_only=False)
            assert len(payload["rng_states"]) == 2
            assert payload["epoch"] == 1
            if mode == "spectrum_graph_v2":
                assert all(result[3] > 0 for result in results)
            if resume:
                resumed = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)
                assert all(torch.equal(value, resumed["model_state_dict"][key])
                           for key, value in payload["model_state_dict"].items())
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(2)
        queue.close()
        queue.join_thread()


def _ddp_rng_failure_worker(rank, store, config, queue):
    torch.set_num_threads(1)
    try:
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank,
                                world_size=2, timeout=timedelta(seconds=15))
        # CPU gloo 下固定 CUDA 不可用；来源只有 rank 1 含 fake CUDA RNG，精确触发单 rank 拒绝。
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        train(config)
        queue.put((rank, "ok"))
    except Exception as error:
        queue.put((rank, "error", str(error)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("existing_destination", [False, True])
def test_ddp_rank_one_rng_failure_prevents_all_output_writes(tiny_dataset_dir, tmp_path, existing_destination):
    source_config = TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path / "source")
    train(source_config)
    source = source_config.output_dir / "last.pt"
    payload = torch.load(source, weights_only=False)
    payload["train_config"]["world_size"] = 2
    payload["rng_states"].append(copy.deepcopy(payload["rng_states"][0]))
    payload["rng_states"][1]["cuda"] = [torch.tensor([42], dtype=torch.uint8)]
    torch.save(payload, source)
    destination = tmp_path / "destination"
    if existing_destination:
        destination.mkdir()
        # 相同字节的 symlink 若被物化，read_bytes 不变但链接类型改变，也必须检出。
        (destination / "last.pt").symlink_to(source)
        (destination / "best.pt").symlink_to(source)
    before = {p.name: (p.is_symlink(), p.read_bytes()) for p in destination.iterdir()} if destination.exists() else None
    config = dataclasses.replace(source_config, output_dir=destination, resume_from=source, ddp=True)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_ddp_rng_failure_worker,
        args=(rank, str(tmp_path / "rng-store"), config, queue)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(25)
        assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
        results = sorted(queue.get(timeout=2) for _ in range(2))
        assert all(result[1] == "error" for result in results), results
        assert results[0][2] == results[1][2]
        assert "rank 1 ValueError: rng_states.cuda" in results[0][2]
        assert "timeout" not in results[0][2].lower()
        after = {p.name: (p.is_symlink(), p.read_bytes()) for p in destination.iterdir()} if destination.exists() else None
        assert after == before
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(2)
        queue.close()
        queue.join_thread()

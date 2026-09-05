"""功能概述：用临时三件套 Parquet 运行单 CPU 的两种训练模式和恢复流程。
主要职责：fixture 固定分子、光谱和 split；验证 train-only 统计量、graph 梯度更新、
v1 独立候选光谱模式、相同 seed 的连续/恢复训练、来源哈希和空训练保护。
代码库关系：覆盖 Task 4–6 到 train_structure_aware 的真实集成；不读取项目大型数据目录。
输入输出与副作用：写入 tmp_path 内微型数据、日志、best/last checkpoint。
"""
import dataclasses
import hashlib
import json
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

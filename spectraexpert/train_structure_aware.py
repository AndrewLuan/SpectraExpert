"""功能概述：提供共享 split 的 spectral_only_v1 和 spectrum_graph_v2 训练/消融入口。
主要职责：TrainConfig 声明实验参数；train 校验 manifest/source、仅从 train fit normalizer，
运行增强目标和排除同 identity 的验证 top1_tanimoto early stopping，并写 best/last 原子
checkpoint 与 JSONL；包含自身候选的结果以 inclusive_ 前缀单独记录。
恢复来源只读取一次不可变字节快照，load 与来源 SHA-256 共用此快照；已有 last 的当前
内容必须与快照相同，通过后原子物化快照为普通文件，避免 symlink 或来源替换造成混装。
bad_epochs 为零且分数一致才表示恢复状态本身为 best，此时已有 best 也必须匹配同一快照；
平分后的 last 不替代历史 best，所有目标冲突均在写入前拒绝。
resume best、已有 best 与 sibling historical best 在物化前均复用 validate_best_payload，
拒绝缺失选择指标、非有限值或声明 best 分数与实际 validation 不一致的文件。
setup validation 阶段只验证各 rank 来源/模型/元数据/RNG 并准备快照，所有 rank 通过相同
_phase 错误同步后才进入 setup materialization，由 rank 0 创建目录或原子物化 best/last；
单 rank 验证失败时全部退出且不写文件，不增加可能与失败路径 collective 错序的单独 barrier。
_ValidationDataset 提供无增强原始验证光谱；_phase 同步 rank 异常；_run 执行实际 epoch 循环；
_prepare_data 校验来源和 split 身份隔离，_sha256 分块计算文件 hash，_validate_config
拒绝非法训练规模；_seed_worker 固定 worker 随机性，_validation_collate 保持图和标签对齐，
main 将 CLI 参数转换为配置。v1 只有光谱模型和动态 NT-Xent，v2 消费 Task 5 完整结构目标。
代码库关系：复用 Task 4 模型、Task 5 gather/objective、Task 6 dataset/collate；直接消费五维
element_presence。候选分别为 spectrum/graph，无需预先建立检索 index；train/val 不接触 test。
输入输出与副作用：读取显式 data_dir 三件套及现有 split manifest，写 output_dir；seed 与
每 epoch sampler 固定，DDP 使用等长 DistributedSampler 并 set_epoch。单 CPU 可 smoke，
多 GPU 由 torchrun --ddp 启动并使用 NCCL；创建的 process group 在成功或异常后销毁。
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from spectraexpert.data.molecular_graph import collate_graphs
from spectraexpert.data.structure_dataset import (StructureAwareSpectralDataset,
    canonical_smiles_sha256, canonicalize_smiles, structure_aware_collate)
from spectraexpert.data.structure_features import DescriptorNormalizer, StructureFeatureConfig
from spectraexpert.training.checkpoints import (MODES, capture_rng, load_checkpoint,
    materialize_checkpoint, read_checkpoint_snapshot, rebuild_model, restore_rng_state, save_checkpoint,
    validate_best_payload)
from spectraexpert.training.validation import validate_retrieval
from spectraexpert.utils.distributed import gather_identity, gather_with_local_grad
from spectraexpert.utils.structure_losses import StructureAwareObjective, dynamic_nt_xent


@dataclasses.dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    split_manifest: Path | None = None
    training_mode: str = "spectrum_graph_v2"
    epochs: int = 100
    patience: int = 10
    batch_size: int = 64
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    projection_dim: int = 128
    graph_hidden_dim: int = 128
    patch_size: int = 16
    stride: int = 8
    lambda_aug: float = 1.
    lambda_struct: float = 1.
    lambda_aux: float = 1.
    lambda_fingerprint: float = 0.
    temperature: float = .1
    seed: int = 0
    ddp: bool = False
    num_workers: int = 0
    resume_from: Path | None = None
    max_samples: int | None = None
    learning_rate: float = 1e-3

    def __post_init__(self):
        self.data_dir, self.output_dir = Path(self.data_dir), Path(self.output_dir)
        self.split_manifest = Path(self.split_manifest) if self.split_manifest else self.data_dir / "split.json"
        if self.resume_from is not None:
            self.resume_from = Path(self.resume_from)

    @classmethod
    def tiny(cls, **kwargs):
        values = dict(epochs=1, batch_size=4, embed_dim=16, depth=1, num_heads=4,
                      projection_dim=8, graph_hidden_dim=16, patch_size=8, stride=4)
        return cls(**(values | kwargs))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _phase(name):
    error = None
    try:
        yield
    except Exception as caught:
        error = caught
    if dist.is_initialized():
        detail = None if error is None else f"rank {dist.get_rank()} {type(error).__name__}: {error}"
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, detail)
        if any(failures):
            raise RuntimeError(f"distributed training {name}: " + "; ".join(x for x in failures if x)) from error
    elif error is not None:
        raise error


def _seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class _ValidationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset
        graph = data.graphs[index]
        spectrum = data.spectra[index].copy()
        if data.normalize:
            spectrum = data._normalize_spectrum(spectrum)
        return dict(spectrum=torch.from_numpy(spectrum), graph=graph,
            fingerprint=graph.fingerprint, identity_id=torch.tensor(data.identity_ids[index]),
            heavy_atoms=graph.descriptors[5])


def _validation_collate(items):
    return {key: collate_graphs([item[key] for item in items]) if key == "graph"
            else torch.stack([item[key] for item in items]) for key in items[0]}


def _prepare_data(config):
    manifest_bytes = config.split_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != "scaffold_split_v1":
        raise ValueError("split manifest schema_version mismatch")
    files = {name: _sha256(config.data_dir / name) for name in
             ("spectra.parquet", "smiles.parquet", "wavenumber.parquet")}
    if "source_sha256" in manifest and manifest["source_sha256"] != files:
        raise ValueError("split manifest source_sha256 mismatch")
    smiles = pd.read_parquet(config.data_dir / "smiles.parquet")
    if "mol_id" not in smiles or smiles["mol_id"].duplicated().any():
        raise ValueError("split manifest source requires unique mol_id")
    column = "smiles_canonical" if "smiles_canonical" in smiles else "smiles"
    canonical = [canonicalize_smiles(value) for value in smiles[column]]
    all_ids = smiles["mol_id"].tolist()
    source_hash = canonical_smiles_sha256(all_ids, canonical)
    if manifest.get("source_smiles_sha256") != source_hash:
        raise ValueError("split manifest source_smiles_sha256 mismatch")
    assignments = []
    identity_sets = []
    mapping = dict(zip(all_ids, canonical))
    for split in ("train", "validation", "test"):
        values = manifest.get(split)
        if not isinstance(values, list) or any(type(value) is not int for value in values):
            raise ValueError(f"split manifest {split}: expected integer IDs")
        if split != "test" and not values:
            raise ValueError(f"split manifest {split} is empty")
        if set(values) - set(all_ids):
            raise ValueError(f"split manifest {split}: unknown mol_id")
        assignments.extend(values)
        identity_sets.append({mapping[value] for value in values})
    if len(assignments) != len(set(assignments)) or set(assignments) != set(all_ids):
        raise ValueError("split manifest mol_id assignments must partition source exactly")
    if any(identity_sets[i] & identity_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("split manifest canonical identity leakage")
    if len(identity_sets[1]) < 2:
        raise ValueError("validation query has no eligible candidates after identity exclusion")
    train_ids = manifest["train"][:config.max_samples] if config.max_samples is not None else manifest["train"]
    train_data = StructureAwareSpectralDataset.from_directory(config.data_dir, mol_ids=train_ids, normalize=True)
    val_data = StructureAwareSpectralDataset.from_directory(config.data_dir, mol_ids=manifest["validation"], normalize=True)
    stats = DescriptorNormalizer.fit(torch.stack([graph.descriptors for graph in train_data.graphs]))
    train_data.descriptor_normalizer = stats
    val_data.descriptor_normalizer = stats
    metadata = dict(split_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                    source_sha256=files, source_smiles_sha256=source_hash)
    return train_data, val_data, stats, metadata


def _validate_config(config):
    if config.training_mode not in MODES:
        raise ValueError("training_mode must be spectral_only_v1 or spectrum_graph_v2")
    for key in ("epochs", "patience", "batch_size", "embed_dim", "depth", "num_heads",
                "projection_dim", "graph_hidden_dim", "patch_size", "stride"):
        if getattr(config, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if config.max_samples is not None and config.max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")


def train(config: TrainConfig):
    _validate_config(config)
    created_group = False
    if config.ddp and not dist.is_initialized():
        if "RANK" not in os.environ:
            raise ValueError("--ddp requires torchrun RANK/WORLD_SIZE environment")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(seconds=120))
        created_group = True
    try:
        if dist.is_initialized() and not config.ddp:
            raise ValueError("existing process group requires --ddp")
        return _run(config)
    finally:
        if created_group:
            dist.destroy_process_group()


def _run(config):
    rank, world = (dist.get_rank(), dist.get_world_size()) if config.ddp else (0, 1)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    with _phase("setup validation"):
        materializations = []
        objective = StructureAwareObjective(temperature=config.temperature, lambda_aug=config.lambda_aug,
            lambda_struct=config.lambda_struct, lambda_aux=config.lambda_aux, lambda_fingerprint=config.lambda_fingerprint)
        training, validation_data, stats, metadata = _prepare_data(config)
        architecture = {key: getattr(config, key) for key in
                        ("patch_size", "stride", "embed_dim", "depth", "num_heads", "projection_dim")}
        architecture["signal_length"] = len(training.wavenumber_axis)
        if config.training_mode == "spectrum_graph_v2":
            architecture.update(graph_hidden_dim=config.graph_hidden_dim, descriptor_dim=15,
                                predict_fingerprint=config.lambda_fingerprint > 0)
        model = rebuild_model(config.training_mode, architecture).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        weights = {key: getattr(config, key) for key in
                   ("lambda_aug", "lambda_struct", "lambda_aux", "lambda_fingerprint")}
        if config.training_mode == "spectral_only_v1":
            weights = dict(lambda_aug=1., lambda_struct=0., lambda_aux=0., lambda_fingerprint=0.)
        # 恢复必须维持计算路径；总 epochs 和输出路径可变，其余训练超参必须匹配。
        resume_config = {key: getattr(config, key) for key in
            ("batch_size", "num_workers", "max_samples", "temperature", "learning_rate", "patience")}
        resume_config["world_size"] = world
        resume_config["normalize"] = True
        expected = metadata | dict(loss_weights=weights, seed=config.seed, train_config=resume_config)
        start_epoch, global_steps, best, bad_epochs = 0, 0, float("-inf"), 0
        if config.resume_from:
            resume_snapshot = read_checkpoint_snapshot(config.resume_from)
            model, payload = load_checkpoint(resume_snapshot, model=model, optimizer=optimizer, device=device,
                expected_mode=config.training_mode, expected_model_config=architecture,
                expected_metadata=expected, restore_rng=True, rank=rank)
            saved = payload["descriptor_stats"]
            if not torch.equal(stats.mean, saved["mean"]) or not torch.equal(stats.std, saved["std"]):
                raise ValueError("descriptor_stats: train statistics mismatch")
            stats = DescriptorNormalizer(saved["mean"], saved["std"])
            training.descriptor_normalizer = validation_data.descriptor_normalizer = stats
            start_epoch, global_steps = payload["epoch"] + 1, payload["global_steps"]
            best, bad_epochs = payload["best_score"], payload["bad_epochs"]
            resume_is_best = bad_epochs == 0
            if resume_is_best:
                validate_best_payload(payload)
            else:
                validation = payload["validation"]
                score = validation.get(payload["selection_metric"]) if isinstance(validation, dict) else None
                if (type(score) not in (int, float) or type(best) not in (int, float)
                        or not np.isfinite(score) or not np.isfinite(best) or score > best):
                    raise ValueError("resume selection metadata inconsistent with best_score/bad_epochs")
            if rank == 0:
                best_destination = config.output_dir / "best.pt"
                best_options = dict(expected_mode=config.training_mode, expected_model_config=architecture,
                                    expected_metadata=expected | {"best_score": best, "bad_epochs": 0})
                best_snapshot = None
                if best_destination.exists():
                    best_snapshot = read_checkpoint_snapshot(best_destination)
                    _, best_payload = load_checkpoint(best_snapshot, **best_options)
                    validate_best_payload(best_payload)
                last_destination = config.output_dir / "last.pt"
                if (last_destination.exists()
                        and _sha256(last_destination) != resume_snapshot.sha256):
                    raise ValueError("resume destination last.pt conflict: differs from resume source")
                if resume_is_best:
                    if best_snapshot is not None and best_snapshot.sha256 != resume_snapshot.sha256:
                        raise ValueError("resume destination best.pt conflict: differs from resume best snapshot")
                    # 只记录已验证快照；所有 rank 完成验证且同步成功后才允许物化。
                    materializations.append((resume_snapshot, best_destination))
                elif best_snapshot is None:
                    best_source = config.resume_from.parent / "best.pt"
                    if best_source.is_file():
                        historical_snapshot = read_checkpoint_snapshot(best_source)
                        _, best_payload = load_checkpoint(historical_snapshot, **best_options)
                        validate_best_payload(best_payload)
                        materializations.append((historical_snapshot, best_destination))
                    else:
                        raise ValueError("resume requires historical best.pt alongside last checkpoint")
                materializations.append((resume_snapshot, last_destination))
            restore_rng_state(payload["rng_states"][rank])
            del resume_snapshot
        sampler = DistributedSampler(training, num_replicas=world, rank=rank, shuffle=True,
                                     seed=config.seed, drop_last=True) if config.ddp else None
        if sampler is not None and len(sampler) == 0:
            raise ValueError("training sampler is empty; increase train samples or reduce world size")
        val_loader = DataLoader(_ValidationDataset(validation_data), batch_size=config.batch_size,
            shuffle=False, collate_fn=_validation_collate, generator=torch.Generator().manual_seed(config.seed))
    with _phase("setup materialization"):
        if rank == 0:
            for snapshot, destination in materializations:
                materialize_checkpoint(snapshot, destination)
            config.output_dir.mkdir(parents=True, exist_ok=True)
        materializations.clear()
    wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=config.training_mode == "spectrum_graph_v2", broadcast_buffers=False) if config.ddp else model
    graph_gradient_norm = 0.
    history = []
    for epoch in range(start_epoch, config.epochs):
        if bad_epochs >= config.patience:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader = DataLoader(training, batch_size=config.batch_size, sampler=sampler,
            shuffle=sampler is None, collate_fn=structure_aware_collate, num_workers=config.num_workers,
            worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(config.seed + epoch))
        with _phase("epoch loader"):
            if len(loader) == 0:
                raise ValueError("training dataloader is empty")
        model.train()
        totals = dict(total=0., cross=0., aug=0., struct=0., descriptor=0., presence=0., fingerprint=0.)
        iterator = iter(loader)
        for _ in range(len(loader)):
            with _phase("batch input"):
                batch = next(iterator)
                batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with _phase("forward"):
                if config.training_mode == "spectrum_graph_v2":
                    output = wrapped(batch["spectrum_view1"], batch["spectrum_view2"], batch["graph"])
                else:
                    embeddings = wrapped(torch.cat([batch["spectrum_view1"], batch["spectrum_view2"]]))
                    first, second = embeddings.chunk(2)
            if config.training_mode == "spectrum_graph_v2":
                losses = objective(output, batch["fingerprint"], batch["descriptor_targets"],
                                   batch["element_presence"], batch["identity_id"])
            else:
                aug = dynamic_nt_xent(gather_with_local_grad(first), gather_with_local_grad(second),
                                      config.temperature, gather_identity(batch["identity_id"]))
                aug = aug.detach() + world * (aug - aug.detach())
                losses = {key: aug.new_zeros(()) for key in totals}
                losses.update(total=aug, aug=aug)
            with _phase("finite loss"):
                if not all(torch.isfinite(value).item() for value in losses.values()):
                    raise FloatingPointError("non-finite training loss")
            losses["total"].backward()
            with _phase("gradients"):
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError(f"non-finite gradient: {name}")
                        if name.startswith("graph_encoder."):
                            graph_gradient_norm = max(graph_gradient_norm, parameter.grad.norm().item())
                optimizer.step()
            for key, value in losses.items():
                totals[key] += value.detach().item()
            global_steps += 1
        loss_vector = torch.tensor([totals[key] / len(loader) for key in totals], device=device)
        if config.ddp:
            dist.all_reduce(loss_vector)
            loss_vector /= world
        totals = dict(zip(totals, loss_vector.cpu().tolist()))
        metrics = None
        with _phase("validation"):
            if rank == 0:
                metrics = validate_retrieval(model, val_loader, device,
                    exclude_identity=True,
                    candidate_modality="graph" if config.training_mode == "spectrum_graph_v2" else "spectrum")
                inclusive = validate_retrieval(model, val_loader, device,
                    candidate_modality="graph" if config.training_mode == "spectrum_graph_v2" else "spectrum")
                metrics.update({"inclusive_" + key: value for key, value in inclusive.items()})
        if config.ddp:
            objects = [metrics]
            dist.broadcast_object_list(objects, src=0)
            metrics = objects[0]
        improved = metrics["top1_tanimoto"] > best
        if improved:
            best, bad_epochs = metrics["top1_tanimoto"], 0
        else:
            bad_epochs += 1
        rng_states = [capture_rng()]
        if config.ddp:
            rng_states = [None] * world
            dist.all_gather_object(rng_states, capture_rng())
        record = dict(epoch=epoch, global_steps=global_steps, training_mode=config.training_mode,
            losses=totals, validation=metrics, feature_config=StructureFeatureConfig().to_metadata(),
            gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
        with _phase("checkpoint"):
            if rank == 0:
                arguments = dict(epoch=epoch, global_steps=global_steps, validation=metrics,
                    descriptor_stats=stats, training_mode=config.training_mode, seed=config.seed,
                    loss_weights=weights, best_score=best, bad_epochs=bad_epochs, rng_states=rng_states,
                    train_config=resume_config, **metadata)
                if improved:
                    save_checkpoint(config.output_dir / "best.pt", model, optimizer, **arguments)
                save_checkpoint(config.output_dir / "last.pt", model, optimizer, **arguments)
                with (config.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
        history.append(record)
    return dict(global_steps=global_steps, best_score=best, history=history,
                graph_gradient_norm=graph_gradient_norm)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train explicit spectral-only or spectrum/graph retrieval")
    parser.add_argument("--training-mode", choices=MODES, required=True)
    for key in ("data_dir", "output_dir", "split_manifest"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    for field in dataclasses.fields(TrainConfig):
        if field.name in {"training_mode", "data_dir", "output_dir", "split_manifest"}:
            continue
        flag = "--" + field.name.replace("_", "-")
        if field.name == "ddp":
            parser.add_argument(flag, action="store_true")
        else:
            kind = Path if field.name == "resume_from" else int if field.name == "max_samples" else type(field.default)
            parser.add_argument(flag, type=kind, default=field.default)
    train(TrainConfig(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()

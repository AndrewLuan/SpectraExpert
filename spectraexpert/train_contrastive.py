#!/usr/bin/env python3
"""
功能概述：训练基于 SimCLR 的 ``SpectrumTransformer``，是 SpectraExpert 检索阶段生成
checkpoint 的命令行入口。
主要职责：``parse_arguments`` 定义训练与 DDP 参数；``setup_seed`` 和 DDP 辅助函数保证可复现
执行；``Logger``/学习率调度器记录过程；``main`` 组织数据集、模型、优化器和 checkpoint 保存。
代码库关系：依赖 ``spectraexpert.data.spectral_dataset``、模型与 NT-Xent 损失；输出的
``model_state_dict`` 和 ``args`` 由 ``spectraexpert.inference.SpectralEncoder`` 恢复。
输入输出与副作用：读取 Parquet 数据、写 TensorBoard/CSV/checkpoint；只有命令行执行 main 时启动
训练，导入模块本身不读取数据或启动分布式进程。

Usage:
    # Single GPU
    python train_contrastive.py \
        --data_dir /path/to/data/qm9s \
        --batch_size 128 \
        --epochs 80 \
        --name spectral_simclr_qm9s

    # Multi-GPU (8 cards)
    torchrun --nproc_per_node=8 train_contrastive.py \
        --data_dir /path/to/data/qm9s \
        --batch_size 128 \
        --epochs 80 \
        --name spectral_simclr_qm9s \
        --ddp

    # Multi-GPU with specific GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ...
"""

import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Add parent to path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from spectraexpert.models.spectrum_encoder import SpectrumTransformer, SpectrumSimCLR
from spectraexpert.utils.ntxent_loss import NTXentLoss

warnings.filterwarnings("ignore")

# Fix random seeds for reproducibility (matching MatExpert convention)
_SEED = 0


def setup_seed(seed: int = _SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Spectrum Encoder with SimCLR contrastive learning"
    )

    # =========================================================
    # Model hyperparameters
    # =========================================================
    parser.add_argument("--signal_length", type=int, default=3501,
                        help="Length of input spectrum (pixels)")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch size for 1D patch embedding")
    parser.add_argument("--stride", type=int, default=8,
                        help="Patch stride (overlap when < patch_size)")
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="Transformer embedding dimension")
    parser.add_argument("--depth", type=int, default=6,
                        help="Number of transformer encoder layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--projection_dim", type=int, default=256,
                        help="Output embedding dimension")

    # =========================================================
    # Training hyperparameters
    # =========================================================
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (AdamW)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="NT-Xent temperature")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Per-device batch size (per GPU)")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4,
                        help="Gradient accumulation steps before optimizer step")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="AdamW weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear warmup epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")

    # =========================================================
    # DDP / Distributed
    # =========================================================
    parser.add_argument("--ddp", action="store_true", default=False,
                        help="Enable DDP (use with torchrun)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank (set by torchrun); ignored if --ddp not set")
    parser.add_argument("--find_unused_params", action="store_true", default=False,
                        help="DDP find_unused_parameters (rarely needed)")

    # =========================================================
    # Data
    # =========================================================
    parser.add_argument("--data_dir", type=str,
                        default="/home/pleiades/projects/SpectrExpert/data/qm9s",
                        help="Directory containing QM9S Parquet files")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size (debugging)")

    # =========================================================
    # I/O
    # =========================================================
    parser.add_argument("--name", type=str, default="spectral_simclr",
                        help="Run name (saved model dir & log prefix)")
    parser.add_argument("--saved_dir", type=str, default=None,
                        help="Checkpoint output dir (default: data/checkpoints)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log output dir (default: data/logs)")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root (default: auto-detect from script location)")
    parser.add_argument("--tensorboard_dir", type=str, default=None,
                        help="TensorBoard log dir (default: {log_dir}/tb_{name})")
    parser.add_argument("--log_step_interval", type=int, default=50,
                        help="Log loss/grad_norm to TensorBoard every N steps")
    parser.add_argument("--log_embedding_interval", type=int, default=5,
                        help="Log weight histograms every N epochs (0=off)")
    parser.add_argument("--save_ckpt_epochs", type=int, default=5,
                        help="Save checkpoint every N epochs (0=only best & final)")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Only run evaluation (NOT implemented yet)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Auto-detect project root
    if args.project_root is None:
        args.project_root = str(Path(__file__).resolve().parent.parent.parent)

    # Default saved / log dirs
    if args.saved_dir is None:
        args.saved_dir = os.path.join(args.project_root, "data", "checkpoints")
    if args.log_dir is None:
        args.log_dir = os.path.join(args.project_root, "data", "logs")

    return args


# -----------------------------------------------------------------------
# Helper: barrier-aware logging (rank 0 only)
# -----------------------------------------------------------------------

class Logger:
    """Logs to console + file; only prints on rank 0 in DDP mode."""

    def __init__(self, log_path: str, is_master: bool = True):
        self.is_master = is_master
        if is_master:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, "w", buffering=1)
        else:
            self.log_file = None

    def log(self, msg: str):
        if not self.is_master:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        self.log_file.write(line + "\n")

    def close(self):
        if self.log_file:
            self.log_file.close()


# -----------------------------------------------------------------------
# Cosine LR schedule with linear warmup (same as MatExpert)
# -----------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs,
                                     steps_per_epoch):
    """Cosine LR schedule with linear warmup."""

    def lr_lambda(current_step):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch

        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -----------------------------------------------------------------------
# DDP helpers
# -----------------------------------------------------------------------

def is_ddp_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_ddp_enabled() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_ddp_enabled() else 1


def is_master() -> bool:
    return get_rank() == 0


def ddp_barrier():
    if is_ddp_enabled():
        dist.barrier()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = parse_arguments()

    # ---------------------------------------------------------
    # DDP initialisation (via torchrun)
    # ---------------------------------------------------------
    if args.ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        if local_rank < 0:
            raise RuntimeError(
                "DDP requested but LOCAL_RANK not set. "
                "Use: torchrun --nproc_per_node=N train_contrastive.py ... --ddp"
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        if not is_master():
            # Suppress non-master stdout
            sys.stdout = open(os.devnull, "w")
    else:
        local_rank = 0
        world_size = 1
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_master_rank = is_master()

    # Set seed (different per rank to avoid identical augmentations)
    setup_seed(_SEED + rank)

    # ---------------------------------------------------------
    # Create directories & logger
    # ---------------------------------------------------------
    os.makedirs(args.saved_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = Logger(
        log_path=os.path.join(args.log_dir, f"train_{args.name}.log"),
        is_master=is_master_rank,
    )

    # TensorBoard (master only)
    writer = None
    tb_dir = None
    if is_master_rank:
        tb_dir = args.tensorboard_dir or os.path.join(args.log_dir, f"tb_{args.name}")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        logger.log(f"  TensorBoard log dir: {tb_dir}")

    logger.log("=" * 60)
    logger.log(f"Training Spectrum SimCLR — {args.name}")
    logger.log(f"  Model: SpectrumTransformer(depth={args.depth}, "
               f"embed_dim={args.embed_dim}, heads={args.num_heads}, "
               f"patch={args.patch_size}, stride={args.stride})")
    logger.log(f"  Training: lr={args.lr}, batch_per_gpu={args.batch_size}, "
               f"total_batch={args.batch_size * world_size}, "
               f"accum={args.accumulate_grad_batches}, "
               f"epochs={args.epochs}, temperature={args.temperature}")
    logger.log(f"  DDP: {'ON' if is_ddp_enabled() else 'OFF'} "
               f"({world_size} GPU{'s' if world_size > 1 else ''})")
    logger.log("=" * 60)

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    logger.log("Loading data...")

    # Create dataset first (needed by DistributedSampler)
    from spectraexpert.data.spectral_dataset import SpectralDataset

    data_dir = Path(args.data_dir)
    dataset = SpectralDataset(
        spectra_path=str(data_dir / "spectra.parquet"),
        smiles_path=str(data_dir / "smiles.parquet"),
        wavenumber_path=str(data_dir / "wavenumber.parquet"),
        normalize=True,
        max_samples=args.max_samples,
    )

    # Sampler: DDP DistributedSampler or None
    sampler = None
    shuffle = True
    if is_ddp_enabled():
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=_SEED,
        )
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # NT-Xent loss requires fixed batch_size
    )

    steps_per_epoch = len(dataloader)
    logger.log(f"  Dataset: {len(dataloader.dataset)} samples, "
               f"{steps_per_epoch} steps/epoch/GPU")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    logger.log(f"  Device: {device}")

    encoder = SpectrumTransformer(
        signal_length=args.signal_length,
        patch_size=args.patch_size,
        stride=args.stride,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        projection_dim=args.projection_dim,
    )
    model = SpectrumSimCLR(encoder).to(device)

    # Wrap in DDP
    if is_ddp_enabled():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.find_unused_params,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ---------------------------------------------------------
    # Optimiser & Loss
    # ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )
    nt_xent_criterion = NTXentLoss(
        batch_size=args.batch_size,
        temperature=args.temperature,
    ).to(device)

    # ---------------------------------------------------------
    # Resume from checkpoint
    # ---------------------------------------------------------
    start_epoch = 0
    best_loss = float("inf")
    num_bad_epochs = 0
    global_step = 0

    if args.resume_from:
        logger.log(f"Resuming from {args.resume_from}")
        map_loc = {"cuda:0": f"cuda:{local_rank}"} if is_ddp_enabled() else device
        checkpoint = torch.load(args.resume_from, map_location=map_loc)
        if is_ddp_enabled():
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        num_bad_epochs = checkpoint.get("num_bad_epochs", 0)
        global_step = checkpoint.get("global_step", 0)
        logger.log(f"  Resumed at epoch {start_epoch}, global_step={global_step}, "
                   f"best_loss={best_loss:.6f}")
        # Reset scheduler state if resuming (re-build with correct steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
        )

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    logger.log("\nStarting training...")
    header = (
        f"{'Epoch':>6} | {'Loss':>10} | {'LR':>12} | {'Bad':>4}"
        f" | {'Time':>8} | {'GPU Mem':>9}"
    )
    logger.log(header)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()

        # DDP: set epoch for DistributedSampler (ensures different shuffle per epoch)
        if is_ddp_enabled():
            dataloader.sampler.set_epoch(epoch)

        loss_sum = 0.0
        cnt = 0
        optimizer.zero_grad()
        peak_mem = 0

        for step, batch in enumerate(dataloader):
            anchors, positives, _ = batch
            anchors = anchors.to(device, non_blocking=True)
            positives = positives.to(device, non_blocking=True)

            # Forward
            z_i, z_j = model(anchors, positives)
            loss = nt_xent_criterion(z_i, z_j)

            # Backward with gradient accumulation (MatExpert pattern)
            loss = loss / args.accumulate_grad_batches
            loss.backward()

            if (step + 1) % args.accumulate_grad_batches == 0 or \
                    step == steps_per_epoch - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                cnt += 1

            loss_sum += loss.item() * args.accumulate_grad_batches

            # Track peak GPU memory
            if torch.cuda.is_available():
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device) / 1e9)

            # --- TensorBoard: per-step (master only) ---
            if is_master_rank and writer is not None \
                    and (global_step + 1) % args.log_step_interval == 0:
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/lr",
                                  optimizer.param_groups[0]["lr"], global_step)

                # Gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar("train/grad_norm", total_norm, global_step)

            global_step += 1

        # End-of-epoch stats
        loss_avg = loss_sum / steps_per_epoch
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # All-reduce loss across GPUs for logging
        if is_ddp_enabled():
            loss_tensor = torch.tensor([loss_avg], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            loss_avg = loss_tensor.item()

        mem_str = f"{peak_mem:.1f}/{torch.cuda.get_device_properties(device).total_memory/1e9:.1f}G" \
            if torch.cuda.is_available() else "N/A"

        logger.log(
            f"{epoch + 1:>6} | {loss_avg:>10.6f} | {current_lr:>12.2e} | "
            f"{num_bad_epochs:>4} | {epoch_time:>8.1f}s | {mem_str:>9}"
        )

        # --- TensorBoard: epoch-level (master only) ---
        if is_master_rank and writer is not None:
            writer.add_scalar("train/loss_epoch", loss_avg, epoch)
            writer.add_scalar("train/lr_epoch", current_lr, epoch)
            writer.add_scalar("train/epoch_time", epoch_time, epoch)

            # Weight & grad histograms
            if args.log_embedding_interval > 0 \
                    and (epoch + 1) % args.log_embedding_interval == 0:
                with torch.no_grad():
                    sample_anchors, sample_positives, _ = next(iter(dataloader))
                    sample_anchors = sample_anchors.to(device)
                    z_sample, _ = model(sample_anchors, sample_anchors)
                    writer.add_histogram("embeddings/anchor", z_sample, epoch)

                model_to_log = model.module if is_ddp_enabled() else model
                for name, param in model_to_log.named_parameters():
                    if param.requires_grad and param.dim() > 0:
                        safe_name = name.replace(".", "/")
                        writer.add_histogram(
                            f"weights/{safe_name}",
                            param.data.cpu().float().numpy(),
                            epoch,
                        )
                        if param.grad is not None:
                            writer.add_histogram(
                                f"grads/{safe_name}",
                                param.grad.cpu().float().numpy(),
                                epoch,
                            )

        # Save loss to CSV (master only)
        if is_master_rank:
            csv_path = os.path.join(args.log_dir, f"loss_{args.name}.csv")
            with open(csv_path, "a") as f:
                f.write(f"{epoch + 1},{loss_avg:.6f},{current_lr:.2e}\n")

        # ---------------------------------------------------------
        # Checkpoint — best loss
        # ---------------------------------------------------------
        if loss_avg < best_loss:
            best_loss = loss_avg
            num_bad_epochs = 0
            if is_master_rank:
                ckpt_path = os.path.join(args.saved_dir, f"{args.name}_best.pth")
                state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": (
                        model.module.state_dict()
                        if is_ddp_enabled()
                        else model.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "num_bad_epochs": num_bad_epochs,
                    "args": vars(args),
                }
                torch.save(state, ckpt_path)
                logger.log(f"  ✓ Saved best model → {ckpt_path}")
        else:
            num_bad_epochs += 1

        # Periodic checkpoint (every N epochs)
        if is_master_rank and args.save_ckpt_epochs > 0 \
                and (epoch + 1) % args.save_ckpt_epochs == 0:
            ckpt_path = os.path.join(
                args.saved_dir, f"{args.name}_epoch{epoch + 1}.pth"
            )
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": (
                    model.module.state_dict()
                    if is_ddp_enabled()
                    else model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "num_bad_epochs": num_bad_epochs,
                "args": vars(args),
            }
            torch.save(state, ckpt_path)
            logger.log(f"  → Saved periodic checkpoint → {ckpt_path}")

        # Early stopping (MatExpert pattern)
        if num_bad_epochs >= args.patience:
            logger.log(f"  Early stopping triggered at epoch {epoch + 1} "
                       f"(no improvement for {num_bad_epochs} epochs)")
            break

    # ---------------------------------------------------------
    # Save final model (master only)
    # ---------------------------------------------------------
    if is_master_rank:
        final_path = os.path.join(args.saved_dir, f"{args.name}_final.pth")
        state = {
            "epoch": args.epochs - 1,
            "global_step": global_step,
            "model_state_dict": (
                model.module.state_dict()
                if is_ddp_enabled()
                else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "num_bad_epochs": num_bad_epochs,
            "args": vars(args),
        }
        torch.save(state, final_path)
        logger.log(f"\nSaved final model → {final_path}")
        logger.log(f"Best loss: {best_loss:.6f}")

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    logger.close()
    if writer is not None:
        writer.close()

    if is_ddp_enabled():
        ddp_barrier()
        dist.destroy_process_group()

    if is_master_rank:
        print(f"\n✓ Training complete. Log: {os.path.join(args.log_dir, 'train_' + args.name + '.log')}")
        if tb_dir:
            print(f"✓ TensorBoard: tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()

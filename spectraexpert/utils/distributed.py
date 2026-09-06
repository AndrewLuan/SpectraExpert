"""
功能概述：为结构感知对比目标提供 rank-major 的 DDP tensor 聚合，并明确限制各 rank batch 等长。
主要职责：``gather_with_local_grad`` 聚合 embedding 且把本 rank slice 替换为原计算图 tensor；
``gather_identity`` 聚合 canonical identity、fingerprint 和监督 target，并始终切断这些标签的梯度。
代码库关系：由 ``spectraexpert.utils.structure_losses.StructureAwareObjective`` 调用；Task 7 的训练
sampler 必须保证各 rank local batch size 相同，本模块不填充或截断样本，也不生成第二套 identity。
输入输出与副作用：单进程 embedding 原样返回；已初始化 DDP 时执行同步 all_gather，按 rank 顺序
沿第 0 维拼接。各 rank batch 不等会在数据 gather 前共同抛出含完整 size 列表的 RuntimeError。
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _distributed_world() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        return 1, 0
    return dist.get_world_size(), dist.get_rank()


def _validate_tensor(tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("distributed gather expects a torch.Tensor")
    if tensor.ndim == 0:
        raise ValueError("distributed gather expects a batch dimension")


def _equal_batch_sizes(tensor: torch.Tensor, world_size: int) -> list[int]:
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    gathered_sizes = [torch.empty_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    if len(set(sizes)) != 1:
        raise RuntimeError(
            "DDP all_gather requires equal local batch sizes; "
            f"got per-rank batch sizes {sizes}"
        )
    return sizes


def gather_with_local_grad(tensor: torch.Tensor) -> torch.Tensor:
    """聚合全 rank tensor，并只为本 rank 对应 slice 保留 autograd 计算图。"""

    _validate_tensor(tensor)
    world_size, rank = _distributed_world()
    if world_size == 1:
        return tensor
    _equal_batch_sizes(tensor, world_size)
    local = tensor.contiguous()
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local.detach())
    gathered[rank] = local
    return torch.cat(gathered, dim=0)


def gather_identity(values: torch.Tensor) -> torch.Tensor:
    """按 rank 顺序聚合不需要梯度的 identity、fingerprint 或监督 target。"""

    _validate_tensor(values)
    detached = values.detach()
    world_size, _ = _distributed_world()
    if world_size == 1:
        return detached
    _equal_batch_sizes(detached, world_size)
    local = detached.contiguous()
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return torch.cat(gathered, dim=0)


__all__ = ["gather_identity", "gather_with_local_grad"]

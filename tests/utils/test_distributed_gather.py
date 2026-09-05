"""
功能概述：以两进程 CPU gloo 验证结构感知 loss 的 DDP 全局样本顺序与本 rank 梯度保留契约。
主要职责：覆盖 ``gather_with_local_grad``、``gather_identity`` 的 rank-major 拼接、无梯度标签、
全局 cross-modal negatives、不等 local batch，以及单 rank validation/非有限 loss 的一致失败；
显式检查 identity offset 后的排列和 DDP 参数梯度等价性。
代码库关系：直接消费 ``spectraexpert.utils.distributed`` 与 ``StructureAwareObjective``；worker 是
可由 spawn 导入的模块级测试辅助函数，不进入生产类，也不依赖 GPU、训练 sampler 或远端服务。
输入输出与副作用：每个测试仅临时创建本机 gloo FileStore 和两个子进程；所有 join 均有硬超时，
finally 会 terminate 残留进程、关闭队列并销毁进程组，避免失败测试遗留后台 worker。
"""

from __future__ import annotations

import queue
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from spectraexpert.models import MultimodalOutput
from spectraexpert.utils.distributed import gather_identity, gather_with_local_grad
from spectraexpert.utils.structure_losses import (
    StructureAwareObjective,
    multi_positive_cross_entropy,
)


_WORLD_SIZE = 2
_PROCESS_TIMEOUT_SECONDS = 15.0
_COLLECTIVE_TIMEOUT_SECONDS = 8.0


def _init_process_group(rank: int, store_path: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_COLLECTIVE_TIMEOUT_SECONDS),
    )


def _gather_worker(rank: int, store_path: str, result_queue) -> None:
    try:
        _init_process_group(rank, store_path)
        local = torch.tensor(
            [[rank * 4.0 + 1.0, rank * 4.0 + 2.0],
             [rank * 4.0 + 3.0, rank * 4.0 + 4.0]],
            requires_grad=True,
        )
        gathered = gather_with_local_grad(local)
        gathered.square().sum().backward()
        local_ids = torch.tensor([rank * 2, rank * 2 + 1])
        global_ids = gather_identity(local_ids)
        result_queue.put(
            (
                "ok",
                rank,
                gathered.detach().tolist(),
                local.grad.tolist(),
                global_ids.tolist(),
                global_ids.requires_grad,
            )
        )
    except BaseException as error:
        result_queue.put(("error", rank, repr(error)))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _global_objective_worker(rank: int, store_path: str, result_queue) -> None:
    try:
        global_inputs = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        )
        expected_weight = torch.eye(3, requires_grad=True)
        expected_encoded = F.linear(
            torch.cat((global_inputs, global_inputs + 0.1, global_inputs + 0.2)),
            expected_weight,
        )
        expected_spectrum1, expected_spectrum2, expected_graph = (
            F.normalize(chunk, dim=1) for chunk in expected_encoded.chunk(3)
        )
        expected_ids = torch.arange(4)
        expected = 0.25 * sum(
            (
                multi_positive_cross_entropy(
                    expected_spectrum1 @ expected_graph.T / 0.5,
                    expected_ids,
                    expected_ids,
                ),
                multi_positive_cross_entropy(
                    expected_graph @ expected_spectrum1.T / 0.5,
                    expected_ids,
                    expected_ids,
                ),
                multi_positive_cross_entropy(
                    expected_spectrum2 @ expected_graph.T / 0.5,
                    expected_ids,
                    expected_ids,
                ),
                multi_positive_cross_entropy(
                    expected_graph @ expected_spectrum2.T / 0.5,
                    expected_ids,
                    expected_ids,
                ),
            )
        )
        expected.backward()

        _init_process_group(rank, store_path)
        model = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(3))
        model = DistributedDataParallel(model)
        local_inputs = global_inputs[rank * 2:rank * 2 + 2]
        encoded = model(
            torch.cat((local_inputs, local_inputs + 0.1, local_inputs + 0.2))
        )
        spectrum1, spectrum2, graph = (
            F.normalize(chunk, dim=1) for chunk in encoded.chunk(3)
        )
        output = MultimodalOutput(
            spectrum_view1_embedding=spectrum1,
            spectrum_view2_embedding=spectrum2,
            graph_embedding=graph,
            spectrum_view1_features=torch.zeros(2, 1),
            spectrum_view2_features=torch.zeros(2, 1),
            graph_features=torch.zeros(2, 1),
            descriptor_prediction=torch.zeros(2, 1, requires_grad=True),
            element_presence_logits=torch.zeros(2, 1, requires_grad=True),
            fingerprint_logits=None,
        )
        local_ids = torch.tensor([rank * 2, rank * 2 + 1])
        fingerprints = torch.eye(4, dtype=torch.bool)[rank * 2:rank * 2 + 2]
        objective = StructureAwareObjective(
            temperature=0.5,
            lambda_aug=0.0,
            lambda_struct=0.0,
            lambda_aux=0.0,
        )
        losses = objective(
            output,
            fingerprints,
            torch.zeros(2, 1),
            torch.zeros(2, 1, dtype=torch.bool),
            local_ids,
        )

        losses["total"].backward()
        gradient_error = (
            model.module.weight.grad - expected_weight.grad
        ).abs().max().item()
        result_queue.put(
            ("ok", rank, float(losses["cross"].detach()), float(expected.detach()),
             gradient_error)
        )
    except BaseException as error:
        result_queue.put(("error", rank, repr(error)))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _uneven_worker(rank: int, store_path: str, result_queue) -> None:
    try:
        _init_process_group(rank, store_path)
        gather_with_local_grad(torch.ones(2 + rank, 3, requires_grad=True))
    except RuntimeError as error:
        result_queue.put(("ok", rank, str(error)))
    except BaseException as error:
        result_queue.put(("error", rank, repr(error)))
        raise
    else:
        result_queue.put(("error", rank, "uneven gather unexpectedly succeeded"))
        raise AssertionError("uneven gather unexpectedly succeeded")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _validation_failure_worker(rank: int, store_path: str, result_queue) -> None:
    try:
        _init_process_group(rank, store_path)
        embeddings = F.normalize(torch.eye(2), dim=1).requires_grad_()
        output = MultimodalOutput(
            spectrum_view1_embedding=embeddings,
            spectrum_view2_embedding=embeddings.clone(),
            graph_embedding=embeddings.clone(),
            spectrum_view1_features=torch.zeros(2, 1),
            spectrum_view2_features=torch.zeros(2, 1),
            graph_features=torch.zeros(2, 1),
            descriptor_prediction=torch.zeros(2, 1, requires_grad=True),
            element_presence_logits=torch.zeros(2, 1, requires_grad=True),
            fingerprint_logits=None,
        )
        descriptor_targets = torch.zeros(2, 2 if rank == 0 else 1)
        StructureAwareObjective()(
            output,
            torch.eye(2, dtype=torch.bool),
            descriptor_targets,
            torch.zeros(2, 1, dtype=torch.bool),
            torch.arange(rank * 2, rank * 2 + 2),
        )
    except Exception as error:
        result_queue.put(("ok", rank, type(error).__name__, str(error)))
    else:
        result_queue.put(("error", rank, "validation unexpectedly succeeded"))
        raise AssertionError("validation unexpectedly succeeded")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _nonfinite_failure_worker(rank: int, store_path: str, result_queue) -> None:
    try:
        _init_process_group(rank, store_path)
        model = torch.nn.Linear(2, 2, bias=False)
        model = DistributedDataParallel(model)
        encoded = model(torch.eye(2))
        embeddings = F.normalize(encoded, dim=1)
        output = MultimodalOutput(
            spectrum_view1_embedding=embeddings,
            spectrum_view2_embedding=embeddings,
            graph_embedding=embeddings,
            spectrum_view1_features=encoded,
            spectrum_view2_features=encoded,
            graph_features=encoded,
            descriptor_prediction=encoded[:, :1],
            element_presence_logits=encoded[:, :1],
            fingerprint_logits=None,
        )
        descriptor_targets = torch.zeros(2, 1)
        if rank == 0:
            descriptor_targets[0, 0] = torch.nan
        losses = StructureAwareObjective()(
            output,
            torch.eye(2, dtype=torch.bool),
            descriptor_targets,
            torch.zeros(2, 1, dtype=torch.bool),
            torch.arange(rank * 2, rank * 2 + 2),
        )
        losses["total"].backward()
    except Exception as error:
        result_queue.put(("ok", rank, type(error).__name__, str(error)))
    else:
        result_queue.put(("error", rank, "non-finite objective unexpectedly succeeded"))
        raise AssertionError("non-finite objective unexpectedly succeeded")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_two_processes(worker):
    context = mp.get_context("spawn")
    existing_child_pids = {process.pid for process in mp.active_children()}
    result_queue = context.Queue()
    processes = []
    results = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        store_path = str(Path(temporary_directory) / "gloo-store")
        try:
            processes = [
                context.Process(target=worker, args=(rank, store_path, result_queue))
                for rank in range(_WORLD_SIZE)
            ]
            for process in processes:
                process.start()
            deadline = time.monotonic() + _PROCESS_TIMEOUT_SECONDS
            for process in processes:
                process.join(max(0.0, deadline - time.monotonic()))
            alive = [process.pid for process in processes if process.is_alive()]
            if alive:
                raise AssertionError(f"DDP workers exceeded timeout: {alive}")
            bad_exits = [process.exitcode for process in processes if process.exitcode != 0]
            if bad_exits:
                raise AssertionError(f"DDP workers exited unsuccessfully: {bad_exits}")
            for _ in range(_WORLD_SIZE):
                try:
                    results.append(result_queue.get(timeout=2.0))
                except queue.Empty as error:
                    raise AssertionError("DDP worker did not report a result") from error
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2.0)
            result_queue.close()
            result_queue.join_thread()
    assert {process.pid for process in mp.active_children()} <= existing_child_pids
    assert all(result[0] == "ok" for result in results), results
    return sorted(results, key=lambda result: result[1])


def test_two_process_gather_preserves_rank_order_local_gradient_and_detaches_identity():
    results = _run_two_processes(_gather_worker)
    expected_values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

    for _, rank, gathered, gradient, global_ids, identity_requires_grad in results:
        assert gathered == expected_values
        assert gradient == [
            [2.0 * value for value in row]
            for row in expected_values[rank * 2:rank * 2 + 2]
        ]
        assert global_ids == [0, 1, 2, 3]
        assert identity_requires_grad is False


def test_objective_uses_global_negatives_and_identity_offsets_on_each_rank():
    results = _run_two_processes(_global_objective_worker)

    for _, _, actual, expected, gradient_error in results:
        assert actual == pytest.approx(expected, rel=1e-6)
        assert gradient_error < 1e-6


def test_uneven_batches_fail_on_every_rank_with_all_rank_sizes():
    results = _run_two_processes(_uneven_worker)

    for _, _, message in results:
        assert "batch" in message.lower()
        assert "[2, 3]" in message


@pytest.mark.parametrize(
    "worker, expected_context",
    [
        (_validation_failure_worker, "ValueError"),
        (_nonfinite_failure_worker, "FloatingPointError"),
    ],
)
def test_single_rank_objective_failure_is_synchronized_before_collective_or_backward(
    worker, expected_context
):
    results = _run_two_processes(worker)
    error_types = {result[2] for result in results}
    messages = {result[3] for result in results}

    assert error_types == {"DistributedObjectiveError"}
    assert len(messages) == 1
    message = messages.pop()
    assert "rank 0" in message
    assert expected_context in message
    assert "timeout" not in message.lower()

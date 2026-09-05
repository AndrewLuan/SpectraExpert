"""
功能概述：为结构指标的 spawn 子进程测试提供可 picklable 的顶层 worker 故障注入函数。
主要职责：``completed_values_worker`` 驱动已知的递减 GED 上界；``raising_values_worker`` 模拟
generator 在首次 yield 后失败；``blocking_worker`` 与 ``ignore_sigterm_worker`` 分别模拟长时间
预处理和无法响应 SIGTERM 的子进程。
代码库关系：仅由 ``tests/eval/test_structure_metrics.py`` 通过
``spectraexpert.eval.metrics._GED_WORKER_TARGET`` 私有 seam 传给 multiprocessing spawn；它们调用
metrics 的内部 worker 计算函数，不属于生产公开 API。
输入输出与副作用：所有 worker 接收可序列化 SMILES/version 和 Pipe connection；正常 helper
回传 GED 或 ``None``，阻塞 helper 只用于验证父进程 terminate/kill/join 清理，不读写数据文件。
"""

from __future__ import annotations

import signal
import time

import spectraexpert.eval.metrics as metrics


def _send(connection: object, value: float | None) -> None:
    try:
        connection.send(value)  # type: ignore[attr-defined]
    finally:
        connection.close()  # type: ignore[attr-defined]


def _fixed_values_search(*args: object, **kwargs: object):
    return iter([5.0, 3.0, 2.0])


def _raising_values_search(*args: object, **kwargs: object):
    yield 5.0
    raise RuntimeError("search failed")


def completed_values_worker(
    connection: object,
    smiles_a: str,
    smiles_b: str,
    include_h: bool,
    version: str,
    normalized: bool,
) -> None:
    _send(
        connection,
        metrics._worker_metric(  # type: ignore[attr-defined]
            smiles_a,
            smiles_b,
            include_h,
            version,
            normalized,
            search_factory=_fixed_values_search,
        ),
    )


def raising_values_worker(
    connection: object,
    smiles_a: str,
    smiles_b: str,
    include_h: bool,
    version: str,
    normalized: bool,
) -> None:
    _send(
        connection,
        metrics._worker_metric(  # type: ignore[attr-defined]
            smiles_a,
            smiles_b,
            include_h,
            version,
            normalized,
            search_factory=_raising_values_search,
        ),
    )


def blocking_worker(
    connection: object,
    smiles_a: str,
    smiles_b: str,
    include_h: bool,
    version: str,
    normalized: bool,
) -> None:
    time.sleep(1.0)
    _send(connection, None)


def ignore_sigterm_worker(
    connection: object,
    smiles_a: str,
    smiles_b: str,
    include_h: bool,
    version: str,
    normalized: bool,
) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.01)

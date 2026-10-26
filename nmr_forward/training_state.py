# 功能概述：在完整 epoch 边界保存/恢复可继续训练的状态，不负责最佳模型选择或测试评估。
# 主要职责：save_training_state 原子替换指定 last.pt；restore_training_state 先严格核对
# 调用者提供的 JSON 训练契约，再恢复模型、优化器、调度器和所有随机数状态，返回已完成
# epoch 与 best。_contract_text 将配置规范化；_rng_state 收集独立 DataLoader generator。
# 代码库关系：供 nmr_forward/train.py 集成；test_training_state.py 用 CPU 训练验证等价性。
# 输入输出与副作用：contract 必须包含调用者认为影响训练的全部设置及数据缓存身份；
# 不从临时文件恢复。写入使用同目录唯一临时文件、fsync 和 os.replace，失败保留旧 checkpoint。
# 仅加载本项目可信 checkpoint（torch.load 使用 pickle）；恢复要求相同 CUDA 设备数量。
# RNG 在其他状态恢复后才应用；保证 epoch 边界、无跨 epoch worker 私有状态时的重放，
# 不承诺任意 GPU 算子确定性或 epoch 中途恢复。scheduler 可为 None，但保存恢复须一致。
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


_FORMAT_VERSION = 1


def _contract_text(contract):
    """严格限制为 JSON 数据，保留 bool/int/float 类型及全部键，拒绝 NaN。"""
    if not isinstance(contract, dict):
        raise TypeError('training contract must be a dict')
    def check(value):
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError('contract keys must be strings')
            for item in value.values():
                check(item)
        elif type(value) is list:
            for item in value:
                check(item)
        elif value is not None and type(value) not in (str, int, float, bool):
            raise TypeError('contract values must be JSON types')
    check(contract)
    return json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _rng_state(loader_generator):
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'loader_generator': loader_generator.get_state(),
    }


def save_training_state(path, *, model, optimizer, scheduler, epoch, best,
                        contract, loader_generator):
    """同步保存已完成 epoch；调用者应先完成 scheduler.step() 再保存。"""
    contract_text = _contract_text(contract)
    if type(epoch) is not int or epoch < 0:
        raise ValueError('epoch must be a nonnegative completed epoch number')
    state = {
        'format_version': _FORMAT_VERSION,
        'contract_text': contract_text,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': None if scheduler is None else scheduler.state_dict(),
        'epoch': epoch, 'best': best,
        'rng': _rng_state(loader_generator),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp',
                                dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def restore_training_state(path, *, model, optimizer, scheduler,
                           expected_contract, loader_generator):
    """从指定完整 checkpoint 恢复；契约不符时模型和 RNG 均保持原样。"""
    contract_text = _contract_text(expected_contract)
    state = torch.load(Path(path), map_location='cpu', weights_only=False)
    if state.get('format_version') != _FORMAT_VERSION:
        raise ValueError('unsupported training checkpoint version')
    if state.get('contract_text') != contract_text:
        raise ValueError('training contract mismatch; checkpoint not restored')
    required = {'model', 'optimizer', 'scheduler', 'epoch', 'best', 'rng'}
    if not required.issubset(state):
        raise ValueError('incomplete training checkpoint')
    if (state['scheduler'] is None) != (scheduler is None):
        raise ValueError('scheduler presence mismatch')
    rng = state['rng']
    if not {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'loader_generator'}.issubset(rng):
        raise ValueError('incomplete checkpoint RNG state')
    current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(rng['torch_cuda']) != current_cuda_count:
        raise ValueError('CUDA device count differs from checkpoint')
    if type(state['epoch']) is not int or state['epoch'] < 0:
        raise ValueError('invalid completed epoch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu'])
    if rng['torch_cuda']:
        torch.cuda.set_rng_state_all(rng['torch_cuda'])
    loader_generator.set_state(rng['loader_generator'])
    return state['epoch'], state['best']

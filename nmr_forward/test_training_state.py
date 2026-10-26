# 功能概述：用同步 CPU 小模型验证 epoch 边界保存恢复的训练等价性和失败边界。
# 主要职责：TrainingStateTests 比较连续训练与中断恢复的精确参数、Adam 状态和调度器；
# 检查不同训练契约不会修改模型/RNG，孤立 tmp 不可恢复，写入失败不会覆盖旧断点。
# 代码库关系：只依赖 training_state.py、PyTorch、NumPy 和 unittest；不使用实验数据。
# 输入输出与副作用：使用临时目录保存可信 checkpoint，每轮包含随机批序、Dropout、
# Python/NumPy 噪声以实际检验全部 RNG；没有 sleep、网络、服务器或异步任务。
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from .training_state import save_training_state, restore_training_state
except ImportError:
    from training_state import save_training_state, restore_training_state


CONTRACT = {'cache_sha256': 'synthetic-v1', 'width': 5, 'depth': 1, 'seed': 7,
            'epochs': 4, 'lr': 0.01, 'batch_size': 4, 'aux_weight': 0.0,
            'tie_methyl': False}


def make_training(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.25),
                                torch.nn.Linear(5, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    generator = torch.Generator().manual_seed(seed + 1)
    return model, optimizer, scheduler, generator


def train_epochs(parts, count):
    model, optimizer, scheduler, generator = parts
    x = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 36
    loader = DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=4,
                        shuffle=True, generator=generator)
    for _ in range(count):
        for inputs, targets in loader:
            noise = random.random() * 0.01 + float(np.random.normal()) * 0.01
            optimizer.zero_grad()
            loss = (model(inputs) - targets - noise).square().mean()
            loss.backward()
            optimizer.step()
        scheduler.step()


def kwargs(parts):
    model, optimizer, scheduler, generator = parts
    return dict(model=model, optimizer=optimizer, scheduler=scheduler,
                loader_generator=generator)


class TrainingStateTests(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_resumed_training_is_bitwise_identical(self):
        continuous = make_training()
        train_epochs(continuous, 4)
        expected_draws = (random.random(), np.random.random(), torch.rand(3))
        interrupted = make_training()
        train_epochs(interrupted, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            save_training_state(path, **kwargs(interrupted), epoch=2, best=0.25,
                                contract=CONTRACT)
            resumed = make_training(seed=99)
            epoch, best = restore_training_state(path, **kwargs(resumed),
                                                 expected_contract=CONTRACT)
            self.assertEqual((epoch, best), (2, 0.25))
            for original, restored in zip(interrupted[:3], resumed[:3]):
                self.assert_nested_equal(original.state_dict(), restored.state_dict())
            train_epochs(resumed, 4 - epoch)
            for original, restored in zip(continuous[:3], resumed[:3]):
                self.assert_nested_equal(original.state_dict(), restored.state_dict())
            actual_draws = (random.random(), np.random.random(), torch.rand(3))
            self.assert_nested_equal(expected_draws, actual_draws)
            self.assertTrue(torch.equal(continuous[3].get_state(), resumed[3].get_state()))
            self.assertFalse(list(Path(directory).glob('*.tmp')))

    def test_contract_mismatch_does_not_modify_model_or_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            original = make_training()
            save_training_state(path, **kwargs(original), epoch=0, best=1., contract=CONTRACT)
            for changed in ({**CONTRACT, 'epochs': 5}, {**CONTRACT, 'tie_methyl': 0},
                            {key: value for key, value in CONTRACT.items() if key != 'seed'}):
                restored = make_training(99)
                before = copy.deepcopy(restored[0].state_dict())
                rng_before = torch.get_rng_state().clone()
                with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                    restore_training_state(path, **kwargs(restored), expected_contract=changed)
                self.assert_nested_equal(before, restored[0].state_dict())
                self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_orphan_temporary_file_is_not_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            (Path(directory) / 'last.pt.orphan.tmp').write_bytes(b'partial')
            with self.assertRaises(FileNotFoundError):
                restore_training_state(path, **kwargs(make_training()), expected_contract=CONTRACT)

    def test_failed_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            parts = make_training()
            save_training_state(path, **kwargs(parts), epoch=0, best=1., contract=CONTRACT)
            before = path.read_bytes()
            with patch.object(torch, 'save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    save_training_state(path, **kwargs(parts), epoch=1, best=.5, contract=CONTRACT)
            self.assertEqual(before, path.read_bytes())
            self.assertFalse(list(Path(directory).glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()

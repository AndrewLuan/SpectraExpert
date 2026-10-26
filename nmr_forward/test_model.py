# 功能概述：检验 nmr_forward.model 的张量契约、积分质量、padding 隔离和可微性。
# 主要职责：ModelTests 构造小型人工化学键图，验证原子重排等变性与梯度；
# test_density 检查有限网格上每个氢等质量、空分子为零以及无效位点零梯度。
# 辅助头测试检查峰组多重性/J的shape、正值与梯度，验证关闭辅助头时原checkpoint兼容。
# 甲基消融测试验证仅C上3/4H绑定、CH2及NH3保留、无效槽不参与平均及有效梯度。
# 代码库关系：依赖同目录 model.py 和 PyTorch；python -m unittest
# nmr_forward.test_model 可从 SpectrExpert 目录执行。此测试不验证实测谱预测精度。
# 输入输出与副作用：测试仅在 CPU 产生小张量，无文件和网络访问；每个测试设置随机种子。
import unittest

import torch
from torch.nn import functional as nn_functional

try:
    from .model import NMRForwardModel, gaussian_density
except ImportError:  # 同时支持 unittest 模块入口和直接执行本文件。
    from model import NMRForwardModel, gaussian_density


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = NMRForwardModel(hidden_dim=24, depth=2)
        self.batch = {
            "x": torch.randn(2, 4, 18),
            "adj": torch.zeros(2, 4, 4, 5),
            "mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
            "hcount": torch.tensor([[3, 2, 0, 4], [4, 1, 0, 0]]),
            "cond": torch.randn(2, 4),
        }
        self.batch["adj"][:, 0, 1, 0] = 1
        self.batch["adj"][:, 1, 0, 0] = 1

    def test_shape_and_mask(self):
        result = self.model(self.batch)
        self.assertEqual(result["shifts"].shape, (2, 4, 4))
        self.assertEqual(result["proton_mask"].sum(dim=(1, 2)).tolist(), [5, 5])
        self.assertTrue(torch.isfinite(result["shifts"]).all())

    def test_permutation_and_padding(self):
        expected = self.model(self.batch)
        order = torch.tensor([2, 0, 3, 1])
        permuted = {k: v.clone() for k, v in self.batch.items()}
        for key in ("x", "mask", "hcount"):
            permuted[key] = permuted[key][:, order]
        permuted["adj"] = permuted["adj"][:, order][:, :, order]
        actual = self.model(permuted)
        torch.testing.assert_close(actual["shifts"], expected["shifts"][:, order])
        torch.testing.assert_close(actual["proton_mask"], expected["proton_mask"][:, order])
        changed = {k: v.clone() for k, v in self.batch.items()}
        changed["x"][~changed["mask"]] = 10000
        changed["adj"][:, 3, :, :] = 10000
        changed["adj"][:, :, 3, :] = 10000
        output = self.model(changed)
        valid = self.batch["mask"]
        torch.testing.assert_close(output["shifts"][valid], expected["shifts"][valid])

    def test_density(self):
        shifts = torch.tensor([[[1., 3., 6., 9.]], [[2., 4., 5., 8.]]], requires_grad=True)
        mask = torch.tensor([[[True, True, False, False]], [[False] * 4]])
        grid = torch.linspace(-2, 16, 512)
        density = gaussian_density(shifts, mask, grid)
        torch.testing.assert_close(density.sum(-1), torch.tensor([1., 0.]))
        torch.testing.assert_close(density[0, grid < 2].sum(), torch.tensor(0.5))
        (density * grid.square()).sum().backward()
        self.assertTrue(torch.isfinite(shifts.grad).all())
        self.assertGreater(shifts.grad[mask].abs().sum().item(), 0)
        self.assertEqual(shifts.grad[~mask].abs().sum().item(), 0)

    def test_end_to_end_gradient(self):
        result = self.model(self.batch)
        grid = torch.linspace(-2, 16, 512)
        density = gaussian_density(result["shifts"], result["proton_mask"], grid)
        (density * grid.square()).sum().backward()
        gradient = self.model.encoder[0].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_optional_auxiliary_heads(self):
        model = NMRForwardModel(hidden_dim=24, depth=2, aux_classes=9)
        result = model(self.batch)
        self.assertEqual(result["mult_logits"].shape, (2, 4, 4, 9))
        self.assertEqual(result["couplings"].shape, (2, 4, 4, 3))
        self.assertTrue((result["couplings"] > 0).all())
        self.assertAlmostEqual(result["couplings"].mean().item(), 7., delta=0.2)
        mask = result["proton_mask"]
        logits = result["mult_logits"][mask]
        labels = torch.arange(len(logits)) % 9
        loss = (nn_functional.cross_entropy(logits, labels)
                + (result["couplings"][mask] - 5.).square().mean())
        loss.backward()
        for parameters in (model.mult_head[-1].weight, model.coupling_head[-1].weight,
                           model.encoder[0].weight):
            self.assertTrue(torch.isfinite(parameters.grad).all())
            self.assertGreater(parameters.grad.abs().sum().item(), 0)

    def test_auxiliary_disabled_preserves_baseline(self):
        model = NMRForwardModel(hidden_dim=24, depth=2, aux_classes=0)
        model.load_state_dict(self.model.state_dict(), strict=True)
        self.assertEqual(set(model(self.batch)), {"shifts", "proton_mask"})
        torch.testing.assert_close(model(self.batch)["shifts"], self.model(self.batch)["shifts"])

    def test_optional_methyl_tying(self):
        batch = {key: value.clone() for key, value in self.batch.items()}
        batch["x"][..., 0] = 0.06
        batch["hcount"] = torch.tensor([[3, 2, 3, 4], [4, 3, 0, 0]])
        batch["x"][0, 2, 0] = 0.07  # NH3即使3H也不能按甲基绑定。
        baseline = NMRForwardModel(hidden_dim=24, depth=2, aux_classes=9)
        tied = NMRForwardModel(hidden_dim=24, depth=2, aux_classes=9, tie_methyl=True)
        tied.load_state_dict(baseline.state_dict(), strict=True)
        original, output = baseline(batch), tied(batch)
        torch.testing.assert_close(original["proton_mask"], output["proton_mask"])
        for key in ("shifts", "mult_logits", "couplings"):
            for row, atom, count in [(0, 0, 3), (1, 0, 4), (1, 1, 3)]:
                expected = original[key][row, atom, :count].mean(0)
                torch.testing.assert_close(output[key][row, atom, :count], expected.expand_as(output[key][row, atom, :count]))
            # CH2、NH3及padding节点原样保留。
            for row, atom in [(0, 1), (0, 2), (0, 3)]:
                torch.testing.assert_close(output[key][row, atom], original[key][row, atom])
            torch.testing.assert_close(output[key][0, 0, 3], original[key][0, 0, 3])
        self.assertNotEqual(output["shifts"][0, 1, 0].item(), output["shifts"][0, 1, 1].item())
        # 单一有效甲基槽的梯度平均传向原来的三个槽，第四无效槽不参与。
        output["shifts"][0, 0, 0].backward()
        torch.testing.assert_close(tied.head[-1].bias.grad, torch.tensor([1/3, 1/3, 1/3, 0.]))
        self.assertGreater(tied.encoder[0].weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()

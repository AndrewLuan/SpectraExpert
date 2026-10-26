# 功能概述：验证受限实验可见性消融的不可隐藏C-H约束、监督资格、梯度和基础模型兼容。
# 主要职责：VisibilityTests构造乙醇样式图与人工积分，不依赖实测数据；对照重构前基础
# forward计算路径，确认encode/decode提取没有改变默认输出或参数名。
# 代码库关系：测试model.py与visibility.py，内存中创建小型CPU张量，无文件/网络访问。
# 边界：合成测试不证明真实实验质量、不验证交换速率，只确认模型与损失实现契约。
import unittest

import torch

try:
    from .model import NMRForwardModel
    from .visibility import VisibilityNMRModel, visibility_loss
except ImportError:
    from model import NMRForwardModel
    from visibility import VisibilityNMRModel, visibility_loss


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.batch = {"x": torch.zeros(1, 4, 18), "adj": torch.zeros(1, 4, 4, 5),
                      "mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
                      "hcount": torch.tensor([[3, 2, 1, 4]]), "cond": torch.zeros(1, 4),
                      "target_w": torch.tensor([[3., 2., 1.]])}
        self.batch["x"][0, :, 0] = torch.tensor([0.06, 0.06, 0.08, 0.08])
        self.batch["adj"][0, 0, 1, 0] = self.batch["adj"][0, 1, 0, 0] = 1
        self.batch["adj"][0, 1, 2, 0] = self.batch["adj"][0, 2, 1, 0] = 1

    def test_default_refactor_matches_original_path(self):
        model = NMRForwardModel(hidden_dim=24, depth=2)
        batch = self.batch
        x, mask = batch["x"], batch["mask"].to(torch.float32)
        adj = batch["adj"] * mask[:, :, None, None] * mask[:, None, :, None]
        h = model.encoder(x) * mask.unsqueeze(-1)
        for layer in model.layers:
            h = layer(h, adj, mask)
        pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        context = torch.cat((h, pooled[:, None, :].expand(-1, 4, -1),
                             batch["cond"][:, None, :].expand(-1, 4, -1)), -1)
        expected = model.head(context)
        output = model(batch)
        torch.testing.assert_close(output["shifts"], expected, rtol=0, atol=0)
        self.assertEqual(set(output), {"shifts", "proton_mask"})
        # 新模型只增加visibility_head参数，旧模型可严格加载基础参数子集。
        extended = VisibilityNMRModel(hidden_dim=24, depth=2)
        base_state = {k: v for k, v in extended.state_dict().items() if not k.startswith("visibility_head.")}
        model.load_state_dict(base_state, strict=True)
        torch.testing.assert_close(model(batch)["shifts"], extended(batch)["shifts"], rtol=0, atol=0)

    def test_cannot_hide_carbon_and_padding_stays_zero(self):
        model = VisibilityNMRModel(hidden_dim=24, depth=2)
        output = model(self.batch)
        self.assertAlmostEqual(output["proton_weights"][output["exchangeable_mask"]].mean().item(), 0.95, delta=0.005)
        with torch.no_grad():
            model.visibility_head[-1].weight.zero_()
            model.visibility_head[-1].bias.fill_(-100.)
        output = model(self.batch)
        mass, mask = output["proton_weights"], output["proton_mask"]
        self.assertEqual(mass[mask & ~output["exchangeable_mask"]].tolist(), [1.] * 5)
        self.assertEqual(mass[~mask].sum().item(), 0.)
        self.assertAlmostEqual(mass[output["exchangeable_mask"]].item(), 0.02, places=6)

    def test_integer_eligibility_and_gradient(self):
        model = VisibilityNMRModel(hidden_dim=24, depth=2)
        batch = {k: v.expand(6, *v.shape[1:]).clone() for k, v in self.batch.items()}
        batch["target_w"] = torch.tensor([[3., 2., 1.], [3., 2., 0.], [2., 2., 0.],
                                          [3., 3., 1.], [2.5, 2.5, 0.], [float("nan"), 1., 0.]])
        output = model(batch)
        losses = visibility_loss(output, batch)
        counts = losses["counts"]
        self.assertEqual(counts["records"], 6)
        self.assertEqual(counts["eligible_count_records"], 2)
        self.assertEqual(counts["complete_integral_records"], 1)
        self.assertEqual(counts["present_anchor_slots"], 1)
        self.assertEqual(counts["skipped_outside_structural_range"], 2)
        self.assertEqual(counts["skipped_noninteger_integrals"], 1)
        self.assertEqual(counts["skipped_invalid_integrals"], 1)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        for parameter in (model.visibility_head[-1].weight, model.encoder[0].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_no_exchangeable_h_has_zero_visibility_loss(self):
        self.batch["x"][0, 2, 0] = 0.06
        model = VisibilityNMRModel(hidden_dim=24, depth=2)
        result = visibility_loss(model(self.batch), self.batch)
        self.assertEqual(result["loss"].item(), 0.)
        result["loss"].backward()
        self.assertEqual(model.visibility_head[-1].weight.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()

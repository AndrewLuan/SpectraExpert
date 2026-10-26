# 功能概述：用小型合成积分峰表验证 prepare、train 与模型的端到端数据契约。
# 主要职责：测试 canonical 分组、CIP与E/Z特征、氢槽掩码、分位数积分匹配、padding
# 梯度隔离、精确W1与SciPy对照，以及乙醇/丙烷两样本 CPU 过拟合；不读取服务器实测数据。
# 代码库关系：依赖同目录 prepare.py、train.py、model.py、RDKit 和 PyTorch。
# train.py 当前采用脚本式导入，因此本入口显式加入自身目录以支持模块/脚本执行。
# 副作用与边界：缓存升级测试只在自动回收的临时目录写小型文件，其余测试用内存张量；
# 临时减少 CPU 线程并在结束时恢复，不访问网络或研究数据目录。
# 合成过拟合仅证明实现可优化，不是泛化性能或实验物理精度证据。
import pathlib
import sys
import unittest
import tempfile

import numpy as np
import torch
from rdkit import Chem
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import NMRForwardModel
from prepare import convert, features
from train import collate, pred_quantiles, exact_w1
from upgrade_cache import upgrade, file_sha256


def sample(smiles, centers, weights):
    record = dict(nucleus="1H", representation="integral_peak_spectrum", smiles=smiles,
                  value={"x": centers, "y": weights}, frequency=400, solvent="CDCl3")
    item, error = convert(record)
    if error:
        raise AssertionError(error)
    return item


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_same_connectivity_stays_in_split(self):
        equivalent = [sample(s, [1, 3], [3, 3]) for s in ["CCO", "OCC", "C(C)O"]]
        self.assertEqual(len({r["group"] for r in equivalent}), 1)
        self.assertEqual(len({r["split"] for r in equivalent}), 1)
        enantiomers = [sample(s, [1], [5]) for s in ["C[C@H](O)F", "C[C@@H](O)F"]]
        self.assertEqual(enantiomers[0]["group"], enantiomers[1]["group"])
        self.assertEqual(enantiomers[0]["split"], enantiomers[1]["split"])

    def test_integral_quantiles_and_mask(self):
        target = sample("CCC", [0.9, 1.4], [6, 2])
        out = {
            "shifts": torch.tensor([[[0.9, 0.9, 0.9, -100.], [1.4, 1.4, 500., 500.], [0.9, 0.9, 0.9, 500.]]], requires_grad=True),
            "proton_mask": torch.tensor([[[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]]], dtype=torch.bool),
        }
        prediction = pred_quantiles(out)
        torch.testing.assert_close(prediction[0], torch.from_numpy(target["target"]))
        prediction.sum().backward()
        self.assertEqual(out["shifts"].grad[~out["proton_mask"]].abs().sum().item(), 0)
        self.assertTrue((out["shifts"].grad[out["proton_mask"]] > 0).all())

    def test_variable_size_collation(self):
        rows = [sample("C", [0.2], [4]), sample("CCO", [1.2, 3.6, 2.], [3, 2, 1])]
        batch = collate(rows)
        self.assertEqual(batch["x"].shape, (2, 3, 18))
        self.assertEqual(batch["hcount"].sum(1).tolist(), [4, 6])
        model = NMRForwardModel(hidden_dim=24, depth=2)
        prediction = pred_quantiles(model(batch))
        self.assertEqual(prediction.shape, (2, 128))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_stereochemistry_survives_features(self):
        e = sample("CC/C=C/C", [1], [10])
        z = sample("CC/C=C\\C", [1], [10])
        self.assertFalse(np.array_equal(e["x"], z["x"]))
        self.assertEqual(e["x"][:, 16].sum(), 2)
        self.assertEqual(z["x"][:, 17].sum(), 2)
        # CIP 标签应跟随同一个物理原子，不受邻居的枚举顺序改变。
        mol = Chem.MolFromSmiles("C[C@H](O)F")
        order = [2, 0, 3, 1]
        original = features(mol)[0]
        reordered = features(Chem.RenumberAtoms(mol, order))[0]
        np.testing.assert_array_equal(original[order], reordered)
        self.assertEqual(original[:, 9:11].sum(), 1)

    def test_exact_w1_matches_scipy_and_gradients(self):
        shifts = torch.tensor([[[0.2, 1.3, 9., -500.]], [[2.2, 5.1, 8.3, 800.]]], requires_grad=True)
        mask = torch.tensor([[[1, 1, 0, 0]], [[1, 1, 1, 0]]], dtype=torch.bool)
        tx = torch.tensor([[0., 2., 10000.], [1., 3., 7.]])
        tw = torch.tensor([[3., 1., 0.], [0.1, 2., 8.]])
        out = {"shifts": shifts, "proton_mask": mask}
        batch = {"target_x": tx, "target_w": tw}
        result = exact_w1(out, batch)
        expected = [wasserstein_distance(shifts[i][mask[i]].detach().numpy(), tx[i].numpy(),
                                        v_weights=tw[i].numpy()) for i in range(2)]
        np.testing.assert_allclose(result.detach().numpy(), expected, atol=1e-6)
        result.sum().backward()
        self.assertTrue(torch.isfinite(shifts.grad).all())
        self.assertEqual(shifts.grad[~mask].abs().sum().item(), 0)
        # 远离支持点重合处，解析梯度应匹配中心有限差分。
        for index in [(0, 0, 0), (0, 0, 1), (1, 0, 2)]:
            plus, minus = shifts.detach().clone(), shifts.detach().clone()
            plus[index] += 0.001
            minus[index] -= 0.001
            fd = (exact_w1({"shifts": plus, "proton_mask": mask}, batch).sum()
                  - exact_w1({"shifts": minus, "proton_mask": mask}, batch).sum()) / 0.002
            self.assertAlmostEqual(fd.item(), shifts.grad[index].item(), delta=3e-4)

    def test_exact_w1_keeps_tiny_integral(self):
        out = {"shifts": torch.zeros(1, 1, 4), "proton_mask": torch.tensor([[[1, 0, 0, 0]]], dtype=torch.bool)}
        batch = {"target_x": torch.tensor([[0., 10.]]), "target_w": torch.tensor([[0.9999, 0.0001]])}
        # 128个中点分位数会遗漏此峰；精确积分距离仍是10*0.0001。
        self.assertAlmostEqual(exact_w1(out, batch).item(), 0.001, places=7)

    def test_exact_w1_soft_mass_matches_scipy_and_has_mass_gradient(self):
        mass = torch.tensor([[[1., .4, 99., 99.]]], requires_grad=True)
        out = {'shifts': torch.tensor([[[1., 4., 100., -100.]]]),
               'proton_mask': torch.tensor([[[1, 1, 0, 0]]], dtype=torch.bool),
               'proton_weights': mass}
        batch = {'target_x': torch.tensor([[0., 5.]]), 'target_w': torch.tensor([[1., 1.]])}
        result = exact_w1(out, batch)
        expected = wasserstein_distance([1., 4.], [0., 5.], u_weights=[1., .4], v_weights=[1., 1.])
        self.assertAlmostEqual(result.item(), expected, places=6)
        result.sum().backward()
        self.assertTrue(torch.isfinite(mass.grad).all())
        self.assertGreater(abs(mass.grad[0, 0, 1].item()), .01)
        self.assertEqual(mass.grad[0, 0, 2:].abs().sum().item(), 0.)

    def test_cache_upgrade_preserves_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            old = sample("CC/C=C/C", [1., 2.], [6., 4.])
            old["x"] = old["x"][:, :16].copy()
            torch.save([old], root / "cache.pt")
            original_sha = file_sha256(root / "cache.pt")
            audit = upgrade(root / "cache.pt", root / "upgraded")
            new = torch.load(root / "upgraded" / "cache.pt", weights_only=False)[0]
            self.assertEqual(new["x"].shape[-1], 18)
            self.assertEqual(new["x"][:, 16].sum(), 2)
            np.testing.assert_array_equal(old["target"], new["target"])
            np.testing.assert_array_equal(old["hcount"], new["hcount"])
            self.assertEqual(old["split"], new["split"])
            self.assertEqual(audit["source_cache_sha256"], original_sha)
            self.assertEqual(file_sha256(root / "cache.pt"), original_sha)
            with self.assertRaises(FileExistsError):
                upgrade(root / "cache.pt", root / "upgraded")

    def test_small_synthetic_fit(self):
        torch.manual_seed(31)
        rows = [sample("CCO", [1.2, 3.6, 2.], [3, 2, 1]), sample("CCC", [0.9, 1.4], [6, 2])]
        batch = collate(rows)
        model = NMRForwardModel(hidden_dim=32, depth=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        losses = []
        for _ in range(180):
            optimizer.zero_grad()
            loss = exact_w1(model(batch), batch).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        final = np.mean(losses[-10:])
        self.assertLess(final, 0.12)
        self.assertLess(final, losses[0] * 0.06)


if __name__ == "__main__":
    unittest.main()

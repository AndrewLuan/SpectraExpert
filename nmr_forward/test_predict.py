# 功能概述：验证独立推理只依赖二维结构、条件及checkpoint，输出可复现且包含边界说明。
# 主要职责：PredictTests构造小型CPU模型checkpoint，测试基础/辅助输出、canonical
# SMILES等价性、缺元信息警告、无target输入、旧16维模型兼容及异常输入无半成品。
# visibility checkpoint验证N/O/S报告质量、C-H固定及总积分；旧checkpoint不增加新积分字段。
# 代码库关系：调用predict.py与model.py；仅在自动清理的临时目录写合成checkpoint和JSON，
# 不访问研究数据或服务器。随机checkpoint仅用于接口测试，不能证明预测精度。
import json
import pathlib
import tempfile
import unittest
import subprocess
import sys

import torch

try:
    from .model import NMRForwardModel
    from .visibility import VisibilityNMRModel
    from .multiplets import MULTIPLICITY_CLASSES
    from .predict import build_input, predict, predict_to_file
except ImportError:
    from model import NMRForwardModel
    from visibility import VisibilityNMRModel
    from multiplets import MULTIPLICITY_CLASSES
    from predict import build_input, predict, predict_to_file


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(8)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)
        self.directory.cleanup()

    def checkpoint(self, auxiliary=False, feature_dim=18, tie_methyl=False, visibility=False):
        count = len(MULTIPLICITY_CLASSES) if auxiliary else 0
        model_class = VisibilityNMRModel if visibility else NMRForwardModel
        model = model_class(feature_dim=feature_dim, hidden_dim=24, depth=2, aux_classes=count, tie_methyl=tie_methyl)
        path = self.root / "model.pt"
        torch.save({"model": model.state_dict(), "feature_dim": feature_dim,
                    "args": {"width": 24, "depth": 2, "aux_weight": 0.1 if auxiliary else 0, "tie_methyl": tie_methyl, "visibility": visibility},
                    "epoch": 3}, path)
        return path

    def test_structure_only_prediction(self):
        batch, _, _, _ = build_input("CCO", "CDCl3", 400)
        self.assertEqual(set(batch), {"x", "adj", "mask", "hcount", "cond"})
        checkpoint = self.checkpoint()
        a = predict_to_file(checkpoint, "CCO", self.root / "prediction.json", "CDCl3", 400)
        b = predict(checkpoint, "OCC", "CDCl3", 400)
        self.assertNotIn("predicted_reported_total_h", a)
        self.assertNotIn("visibility", a["checkpoint"])
        self.assertTrue(all("predicted_reported_integration_h" not in site for site in a["proton_sites"]))
        self.assertEqual(a["structure_protons"], 6)
        self.assertEqual(len(a["proton_sites"]), 6)
        self.assertEqual(a["proton_sites"], b["proton_sites"])
        self.assertEqual(a["atoms"], b["atoms"])
        self.assertEqual(a["warnings"], [])
        self.assertEqual(len(a["checkpoint"]["sha256"]), 64)
        self.assertNotIn("peak_group_j_candidates_hz", a["proton_sites"][0])
        self.assertEqual(json.loads((self.root / "prediction.json").read_text()), a)

    def test_auxiliary_and_missing_metadata(self):
        result = predict(self.checkpoint(auxiliary=True), "CCC")
        self.assertIn("missing_frequency_encoded_as_zero", result["warnings"])
        self.assertIn("missing_solvent_encoded_as_zero", result["warnings"])
        for site in result["proton_sites"]:
            self.assertAlmostEqual(sum(site["multiplicity_probabilities"].values()), 1., places=6)
            self.assertEqual(len(site["peak_group_j_candidates_hz"]), 3)
            self.assertTrue(all(j > 0 for j in site["peak_group_j_candidates_hz"]))

    def test_visibility_checkpoint_reports_constrained_integrations(self):
        checkpoint = self.checkpoint(auxiliary=True, tie_methyl=True, visibility=True)
        for smiles in ("CCO", "CCN", "CCS", "CCC"):
            with self.subTest(smiles=smiles):
                result = predict(checkpoint, smiles, "CDCl3", 400)
                elements = {atom["canonical_atom_index"]: atom["element"] for atom in result["atoms"]}
                for site in result["proton_sites"]:
                    self.assertEqual(site["structural_integration_h"], 1)
                    mass = site["predicted_reported_integration_h"]
                    if elements[site["canonical_atom_index"]] in ("N", "O", "S"):
                        self.assertGreater(mass, 0.02)
                        self.assertLess(mass, 1.)
                    else:
                        self.assertEqual(mass, 1.)
                    self.assertIn("multiplicity_probabilities", site)
                self.assertEqual(result["predicted_reported_total_h"],
                                 sum(site["predicted_reported_integration_h"] for site in result["proton_sites"]))
                self.assertTrue(result["checkpoint"]["visibility"])
                self.assertIn("predicted_reported_integration_is_latent_reported_mass_not_validated_exchange_probability",
                              result["model_boundaries"])
        saved = predict_to_file(checkpoint, "CCO", self.root / "visibility.json", "CDCl3", 400)
        self.assertEqual(json.loads((self.root / "visibility.json").read_text()), saved)

    def test_invalid_smiles_writes_nothing(self):
        path = self.root / "new_directory" / "result.json"
        with self.assertRaisesRegex(ValueError, "SMILES"):
            predict_to_file(self.checkpoint(), "not_a_smiles", path)
        self.assertFalse(path.exists())
        self.assertFalse(path.parent.exists())

    def test_legacy_features_and_no_overwrite(self):
        path = self.root / "result.json"
        checkpoint = self.checkpoint(feature_dim=16)
        result = predict_to_file(checkpoint, "C[C@H](O)F", path)
        self.assertIn("legacy_16_features_use_CW_CCW_and_do_not_encode_E_Z", result["warnings"])
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            predict_to_file(checkpoint, "CCC", path)
        self.assertEqual(path.read_bytes(), before)

    def test_cli_entrypoint(self):
        checkpoint = self.checkpoint()
        out = self.root / "cli.json"
        completed = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).with_name("predict.py")),
             "--checkpoint", str(checkpoint), "--smiles", "CCO", "--solvent", "CDCl3",
             "--frequency", "400", "--out", str(out)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        self.assertEqual(json.loads(completed.stdout)["structure_protons"], 6)
        self.assertEqual(json.loads(out.read_text())["input"]["frequency_mhz"], 400.)

    def test_methyl_checkpoint_configuration(self):
        result = predict(self.checkpoint(auxiliary=True, tie_methyl=True), "CCO")
        methyl = next(atom for atom in result["atoms"] if atom["attached_hydrogens"] == 3)
        sites = [site for site in result["proton_sites"] if site["canonical_atom_index"] == methyl["canonical_atom_index"]]
        for key in ("shift_ppm", "multiplicity_probabilities", "peak_group_j_candidates_hz"):
            self.assertEqual(sites[0][key], sites[1][key])
            self.assertEqual(sites[1][key], sites[2][key])
        self.assertTrue(result["checkpoint"]["tie_methyl"])
        self.assertIn("tied_CH3_CH4_slots_assume_fast_methyl_rotation_in_solution_not_universal_equivalence", result["model_boundaries"])


if __name__ == "__main__":
    unittest.main()

# 功能概述：仅从SMILES与可选实验条件加载训练checkpoint并生成1H位移位点预测。
# 主要职责：build_input与prepare.features保持二维特征一致；predict加载模型并输出
# 每个原子/无序氢槽的位移、可选多重性概率及峰组内J候选；predict_to_file原子写JSON；
# main提供 --checkpoint --smiles --solvent --frequency --out CLI，可选 --device。
# 代码库关系：依赖model.NMRForwardModel、prepare.features、multiplets类别顺序；
# 不导入训练数据，不构造target，不使用峰表或坐标。读取的是可信本地PyTorch checkpoint。
# 输入输出与副作用：输出checkpoint SHA256、canonical二维结构、条件、质子数和边界；
# 无效SMILES/模型错误发生时不写结果，拒绝覆写已有out。16维旧模型显式重建旧CW/CCW
# 特征并告警；18维模型使用CIP与E/Z。J为最多三个峰组内部候选，不是核对耦合矩阵。
# 氢槽并无经验证的原子归属，固定单位积分是结构先验，不是预测的实验积分；不生成谱线。
# visibility=True 时实例化 visibility.VisibilityNMRModel，新增各位点预测报告积分及总和；
# 仅N/O/S候选允许报告质量减少，C-H固定1；不是经验证交换概率，不改变结构积分字段。
# 旧checkpoint不新增预测积分字段，维持原输出契约。
# checkpoint的tie_methyl默认False；启用时报告快速甲基转动近似，不能外推到所有体系。
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import tempfile

import torch
from rdkit import Chem

try:
    from .model import NMRForwardModel
    from .visibility import VisibilityNMRModel
    from .prepare import features
    from .multiplets import MULTIPLICITY_CLASSES
except ImportError:
    from model import NMRForwardModel
    from visibility import VisibilityNMRModel
    from prepare import features
    from multiplets import MULTIPLICITY_CLASSES


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_input(smiles: str, solvent: str | None, frequency: float | None, feature_dim: int = 18):
    """创建仅含x/adj/mask/hcount/cond的单分子batch，返回canonical分子及警告。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Invalid or empty SMILES")
    mol = Chem.RemoveHs(mol)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(canonical)
    x, adj, hcount = features(mol)
    if hcount.sum() == 0 or hcount.max() > 4:
        raise ValueError("Structure has no supported attached protons or more than four H per atom")
    if feature_dim not in (16, 18):
        raise ValueError(f"Unsupported checkpoint feature dimension: {feature_dim}")
    warnings = []
    if feature_dim == 16:
        x = x[:, :16].copy()
        for atom in mol.GetAtoms():
            tag = atom.GetChiralTag()
            x[atom.GetIdx(), 9] = tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW
            x[atom.GetIdx(), 10] = tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        warnings.append("legacy_16_features_use_CW_CCW_and_do_not_encode_E_Z")
    if frequency is None:
        warnings.append("missing_frequency_encoded_as_zero")
        freq = 0.
    else:
        freq = float(frequency)
        if not math.isfinite(freq) or freq <= 0:
            raise ValueError("Frequency must be a finite positive MHz value")
    solvent_text = str(solvent or "").strip().lower()
    if not solvent_text:
        warnings.append("missing_solvent_encoded_as_zero")
    flags = [float("water" in solvent_text or "d2o" in solvent_text),
             float("cdcl3" in solvent_text), float("dmso" in solvent_text)]
    if solvent_text and not any(flags):
        warnings.append("solvent_not_distinguished_by_current_condition_features")
    if "." in canonical:
        warnings.append("multicomponent_structure")
    if mol.GetNumAtoms() > 96:
        warnings.append("atom_count_exceeds_training_preparation_limit_96")
    if any(a.GetAtomicNum() not in [6, 7, 8, 9, 15, 16, 17, 35, 53] for a in mol.GetAtoms()):
        warnings.append("uncommon_element")
    batch = {
        "x": torch.from_numpy(x).unsqueeze(0),
        "adj": torch.from_numpy(adj).unsqueeze(0),
        "mask": torch.ones(1, mol.GetNumAtoms(), dtype=torch.bool),
        "hcount": torch.from_numpy(hcount).unsqueeze(0),
        "cond": torch.tensor([[freq / 600., *flags]], dtype=torch.float32),
    }
    return batch, mol, canonical, warnings


@torch.no_grad()
def predict(checkpoint, smiles, solvent=None, frequency=None, device="cpu"):
    """返回可JSON序列化的预测；checkpoint必须来自可信训练流程。"""
    path = pathlib.Path(checkpoint).resolve()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    feature_dim = int(ck.get("feature_dim", ck["model"]["encoder.0.weight"].shape[1]))
    classes = tuple(ck.get("multiplicity_classes", MULTIPLICITY_CLASSES))
    aux_classes = int(ck.get("aux_classes", args.get("aux_classes", len(classes) if args.get("aux_weight", 0) > 0 else 0)))
    if aux_classes and aux_classes != len(classes):
        raise ValueError("Checkpoint auxiliary class count does not match class labels")
    batch, mol, canonical, warnings = build_input(smiles, solvent, frequency, feature_dim)
    visibility = bool(args.get("visibility", False))
    model_class = VisibilityNMRModel if visibility else NMRForwardModel
    model = model_class(feature_dim=feature_dim, hidden_dim=int(args["width"]),
                            depth=int(args["depth"]), aux_classes=aux_classes,
                            tie_methyl=bool(args.get("tie_methyl", False))).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    output = model({k: v.to(device) for k, v in batch.items()})
    proton_sites, atoms = [], []
    probabilities = output["mult_logits"].softmax(-1) if aux_classes else None
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        count = int(batch["hcount"][0, index])
        site_ids = []
        for slot in range(count):
            site_id = f"atom_{index}_slot_{slot}"
            site_ids.append(site_id)
            site = {"site_id": site_id, "canonical_atom_index": index, "hydrogen_slot": slot,
                    "shift_ppm": float(output["shifts"][0, index, slot]), "structural_integration_h": 1}
            if visibility:
                site["predicted_reported_integration_h"] = float(output["proton_weights"][0, index, slot])
            if aux_classes:
                probs = probabilities[0, index, slot].cpu().tolist()
                site["multiplicity_probabilities"] = dict(zip(classes, probs))
                site["multiplicity_argmax"] = classes[max(range(len(probs)), key=probs.__getitem__)]
                site["peak_group_j_candidates_hz"] = output["couplings"][0, index, slot].cpu().tolist()
            proton_sites.append(site)
        atoms.append({"canonical_atom_index": index, "element": atom.GetSymbol(),
                      "attached_hydrogens": count, "proton_site_ids": site_ids})
    limits = [
        "integral_peak_table_prediction_not_measured_continuous_spectrum",
        "hydrogen_slots_are_unordered_latent_predictions_not_validated_atom_assignments",
        "unit_hydrogen_integration_is_a_structural_prior_not_experimental_area_prediction",
        "no_linewidth_exchange_strong_coupling_or_conformer_ensemble_simulation",
        "only_frequency_and_coarse_solvent_indicators_are_conditioned_on",
    ]
    if aux_classes:
        limits.extend(["J_candidates_are_peak_group_internal_parameters_not_a_nuclear_pair_matrix",
                       "three_J_candidates_do_not_imply_three_observed_couplings_unused_entries_are_not_validated"])
    if args.get("tie_methyl", False):
        limits.append("tied_CH3_CH4_slots_assume_fast_methyl_rotation_in_solution_not_universal_equivalence")
    if visibility:
        limits.extend(["predicted_reported_integration_is_latent_reported_mass_not_validated_exchange_probability",
                       "only_N_O_S_attached_H_candidates_can_have_reduced_reported_mass_C_H_remains_one"])
    result = {
        "schema_version": 1,
        "checkpoint": {"path": str(path), "sha256": file_sha256(path), "epoch": ck.get("epoch"),
                       "feature_dim": feature_dim, "aux_classes": aux_classes,
                       "tie_methyl": bool(args.get("tie_methyl", False))},
        "input": {"smiles": smiles, "canonical_isomeric_smiles": canonical,
                  "representation": "2d_molecular_graph", "solvent": solvent, "frequency_mhz": frequency,
                  "encoded_conditions": batch["cond"][0].tolist()},
        "nucleus": "1H", "structure_protons": len(proton_sites), "atoms": atoms,
        "proton_sites": proton_sites, "warnings": warnings, "model_boundaries": limits,
    }
    if visibility:
        result["checkpoint"]["visibility"] = True
        result["predicted_reported_total_h"] = sum(site["predicted_reported_integration_h"] for site in proton_sites)
    return result


def predict_to_file(checkpoint, smiles, out, solvent=None, frequency=None, device="cpu"):
    """计算成功且JSON有限数检查通过后才创建临时文件，拒绝覆写已有结果。"""
    out = pathlib.Path(out)
    if out.exists():
        raise FileExistsError(out)
    result = predict(checkpoint, smiles, solvent, frequency, device)
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent,
                                         prefix=out.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            stream.write(encoded)
        # 硬链接原子提交且目标若已存在则失败，避免并发执行覆写结果。
        os.link(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--solvent")
    parser.add_argument("--frequency", type=float, help="1H resonance frequency in MHz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = predict_to_file(args.checkpoint, args.smiles, args.out, args.solvent, args.frequency, args.device)
    print(json.dumps({"out": str(pathlib.Path(args.out).resolve()),
                      "structure_protons": result["structure_protons"], "warnings": result["warnings"]}))


if __name__ == "__main__":
    main()

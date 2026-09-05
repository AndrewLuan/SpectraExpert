"""
功能概述：定义与 GED、Morgan Tanimoto 和辅助描述符监督共同对齐的唯一结构特征 schema。
主要职责：``StructureFeatureConfig`` 固定 schema 版本与 Morgan 参数；``BOND_TYPE_TO_ID``、
``atom_features`` 和 ``bond_features`` 提供图编码；``morgan_fingerprint`` 生成布尔位向量；
``molecular_descriptors`` 按 ``DESCRIPTOR_NAMES`` 生成 15 维标签；``DescriptorNormalizer`` 只用
调用方传入的训练 tensor 拟合逐列统计量并安全处理近零方差列。
代码库关系：由 ``spectraexpert.data.molecular_graph`` 直接复用，后续 GED、特征缓存、图编码器、
checkpoint 与索引 metadata 也必须消费本文件的映射和配置，不能维护平行键类型表。
输入输出与副作用：输入已由 RDKit 解析并清理的 ``Chem.Mol``、Atom 或 Bond，输出 CPU tensor、
整数特征 tuple 或纯 metadata 字典；不读取数据集、不缓存全局状态，非法空分子会明确抛错。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors


BOND_TYPE_TO_ID = {
    "OTHER": 0,
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
}

NODE_FEATURE_NAMES = (
    "atomic_number",
    "degree",
    "total_valence",
    "formal_charge_bucket",
    "total_h_count",
    "aromatic",
    "ring",
)

EDGE_FEATURE_NAMES = ("bond_type", "conjugated", "ring")

DESCRIPTOR_NAMES = (
    "C",
    "N",
    "O",
    "F",
    "H",
    "heavy_atoms",
    "molecular_weight",
    "ring_count",
    "aromatic_ring_count",
    "aromatic_atoms",
    "single_bonds",
    "double_bonds",
    "triple_bonds",
    "aromatic_bonds",
    "formal_charge",
)


@dataclass(frozen=True)
class StructureFeatureConfig:
    """可序列化的结构 schema 与 Morgan fingerprint 参数。"""

    schema_version: str = "metric_aligned_graph_v1"
    morgan_radius: int = 2
    morgan_bits: int = 2048
    use_bond_types: bool = True
    use_chirality: bool = False

    def to_metadata(self) -> dict[str, object]:
        """返回可写入缓存、checkpoint 和索引的完整兼容性 metadata。"""

        return dataclasses.asdict(self)


@dataclass(frozen=True)
class DescriptorNormalizer:
    """保存训练 split 的逐列描述符均值与总体标准差。"""

    mean: torch.FloatTensor
    std: torch.FloatTensor

    @classmethod
    def fit(cls, values: torch.Tensor) -> "DescriptorNormalizer":
        """从二维训练 tensor 拟合统计量，近零方差列的除数固定为 1。"""

        if values.ndim != 2:
            raise ValueError("descriptor training values must be two-dimensional")
        mean = values.float().mean(dim=0)
        std = values.float().std(dim=0, unbiased=False)
        return cls(
            mean=mean,
            std=torch.where(std < 1e-8, torch.ones_like(std), std),
        )

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """用已保存的训练统计量标准化任意 split 的描述符。"""

        return (values.float() - self.mean) / self.std


def _require_molecule(molecule: Chem.Mol) -> None:
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("molecule must contain at least one atom")


def _heavy_atom_molecule(molecule: Chem.Mol) -> Chem.Mol:
    _require_molecule(molecule)
    heavy_molecule = Chem.RemoveAllHs(molecule)
    if heavy_molecule.GetNumAtoms() == 0:
        raise ValueError("molecule must contain at least one heavy atom")
    return heavy_molecule


def _clip(value: int, lower: int, upper: int) -> int:
    return max(lower, min(int(value), upper))


def atom_features(atom: Chem.Atom) -> tuple[int, int, int, int, int, int, int]:
    """编码 GED 元素身份与 radius-2 局部原子环境的七个离散字段。"""

    atomic_number = int(atom.GetAtomicNum())
    if not 1 <= atomic_number <= 118:
        atomic_number = 0
    formal_charge = _clip(atom.GetFormalCharge(), -3, 3) + 3
    return (
        atomic_number,
        _clip(atom.GetDegree(), 0, 8),
        _clip(atom.GetTotalValence(), 0, 12),
        formal_charge,
        _clip(atom.GetTotalNumHs(includeNeighbors=True), 0, 8),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    )


def bond_features(bond: Chem.Bond) -> tuple[int, int, int]:
    """以 RDKit 离散枚举编码键类型，绝不把芳香键 1.5 浮点截断成单键。"""

    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        category = "SINGLE"
    elif bond_type == Chem.BondType.DOUBLE:
        category = "DOUBLE"
    elif bond_type == Chem.BondType.TRIPLE:
        category = "TRIPLE"
    elif bond_type == Chem.BondType.AROMATIC:
        category = "AROMATIC"
    else:
        category = "OTHER"
    return (
        BOND_TYPE_TO_ID[category],
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    )


def morgan_fingerprint(
    molecule: Chem.Mol,
    config: StructureFeatureConfig = StructureFeatureConfig(),
) -> torch.BoolTensor:
    """按显式 radius、位数、bond type 与 chirality 参数生成 Morgan 布尔位向量。"""

    molecule = _heavy_atom_molecule(molecule)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.morgan_radius,
        fpSize=config.morgan_bits,
        includeChirality=config.use_chirality,
        useBondTypes=config.use_bond_types,
    )
    bit_vector = generator.GetFingerprint(molecule)
    fingerprint = torch.zeros(config.morgan_bits, dtype=torch.bool)
    on_bits = list(bit_vector.GetOnBits())
    if on_bits:
        fingerprint[on_bits] = True
    return fingerprint


def molecular_descriptors(molecule: Chem.Mol) -> torch.FloatTensor:
    """按 ``DESCRIPTOR_NAMES`` 返回组成、规模、环、键和电荷的 15 维 float32 tensor。"""

    molecule = _heavy_atom_molecule(molecule)
    element_counts = {6: 0, 7: 0, 8: 0, 9: 0}
    for atom in molecule.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if atomic_number in element_counts:
            element_counts[atomic_number] += 1

    hydrogen_count = sum(
        atom.GetTotalNumHs(includeNeighbors=True) for atom in molecule.GetAtoms()
    )
    bond_counts = [0] * len(BOND_TYPE_TO_ID)
    for bond in molecule.GetBonds():
        bond_counts[bond_features(bond)[0]] += 1

    values = (
        element_counts[6],
        element_counts[7],
        element_counts[8],
        element_counts[9],
        hydrogen_count,
        molecule.GetNumHeavyAtoms(),
        Descriptors.MolWt(molecule),
        rdMolDescriptors.CalcNumRings(molecule),
        rdMolDescriptors.CalcNumAromaticRings(molecule),
        sum(atom.GetIsAromatic() for atom in molecule.GetAtoms()),
        bond_counts[BOND_TYPE_TO_ID["SINGLE"]],
        bond_counts[BOND_TYPE_TO_ID["DOUBLE"]],
        bond_counts[BOND_TYPE_TO_ID["TRIPLE"]],
        bond_counts[BOND_TYPE_TO_ID["AROMATIC"]],
        sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
    )
    return torch.tensor(values, dtype=torch.float32)


__all__ = [
    "BOND_TYPE_TO_ID",
    "DESCRIPTOR_NAMES",
    "DescriptorNormalizer",
    "EDGE_FEATURE_NAMES",
    "NODE_FEATURE_NAMES",
    "StructureFeatureConfig",
    "atom_features",
    "bond_features",
    "molecular_descriptors",
    "morgan_fingerprint",
]

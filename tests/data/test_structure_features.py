"""
功能概述：锁定结构特征的单一版本化 schema，防止训练、缓存、GED 与检索阶段使用不同编码。
主要职责：验证固定 Morgan 参数与位向量类型、15 维描述符顺序，以及只由训练 tensor 拟合的
``DescriptorNormalizer`` 均值和总体标准差；常量断言保护芳香键不会退化为单键类别。
代码库关系：直接消费 ``spectraexpert.data.structure_features``，后续图化、结构指标、缓存和
checkpoint metadata 都依赖这里验证的公开契约；测试使用真实 RDKit 分子而不使用 mock。
输入输出与副作用：仅在内存中构造小分子和 tensor，不读取数据集、不写文件；schema、数值或
归一化行为偏离约定时由 pytest assertion 报错。
"""

import pytest
import torch
from rdkit import Chem

from spectraexpert.data.structure_features import (
    BOND_TYPE_TO_ID,
    DESCRIPTOR_NAMES,
    DescriptorNormalizer,
    StructureFeatureConfig,
    molecular_descriptors,
    morgan_fingerprint,
)


def test_structure_feature_config_exposes_metric_aligned_metadata():
    config = StructureFeatureConfig()

    assert config.to_metadata() == {
        "schema_version": "metric_aligned_graph_v1",
        "morgan_radius": 2,
        "morgan_bits": 2048,
        "use_bond_types": True,
        "use_chirality": False,
    }
    assert BOND_TYPE_TO_ID == {
        "OTHER": 0,
        "SINGLE": 1,
        "DOUBLE": 2,
        "TRIPLE": 3,
        "AROMATIC": 4,
    }


def test_morgan_fingerprint_has_fixed_boolean_schema():
    molecule = Chem.MolFromSmiles("CCO")

    fingerprint = morgan_fingerprint(molecule)

    assert fingerprint.shape == (2048,)
    assert fingerprint.dtype == torch.bool
    assert fingerprint.any()


def test_public_structure_features_ignore_explicit_hydrogen_nodes():
    implicit_hydrogen = Chem.MolFromSmiles("CO")
    explicit_hydrogen = Chem.AddHs(implicit_hydrogen)

    assert torch.equal(
        morgan_fingerprint(explicit_hydrogen),
        morgan_fingerprint(implicit_hydrogen),
    )
    torch.testing.assert_close(
        molecular_descriptors(explicit_hydrogen),
        molecular_descriptors(implicit_hydrogen),
    )


def test_molecular_descriptors_follow_fixed_order_and_include_implicit_hydrogen():
    molecule = Chem.MolFromSmiles("CO")

    descriptors = molecular_descriptors(molecule)

    assert DESCRIPTOR_NAMES == (
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
    assert descriptors.shape == (15,)
    assert descriptors.dtype == torch.float32
    torch.testing.assert_close(
        descriptors,
        torch.tensor(
            [1, 0, 1, 0, 4, 2, 32.042, 0, 0, 0, 1, 0, 0, 0, 0],
            dtype=torch.float32,
        ),
        rtol=0,
        atol=1e-3,
    )


def test_descriptor_normalizer_fits_only_supplied_training_values():
    train_descriptors = torch.tensor(
        [[1.0, 10.0, 5.0], [3.0, 10.0, 9.0]], dtype=torch.float64
    )

    normalizer = DescriptorNormalizer.fit(train_descriptors)
    transformed = normalizer.transform(train_descriptors)

    torch.testing.assert_close(normalizer.mean, torch.tensor([2.0, 10.0, 7.0]))
    torch.testing.assert_close(normalizer.std, torch.tensor([1.0, 1.0, 2.0]))
    torch.testing.assert_close(transformed.mean(dim=0), torch.zeros(3))
    assert transformed.dtype == torch.float32


@pytest.mark.parametrize("values", [torch.ones(3), torch.ones(2, 2, 2)])
def test_descriptor_normalizer_rejects_non_matrix_training_values(values):
    with pytest.raises(ValueError, match="two-dimensional"):
        DescriptorNormalizer.fit(values)

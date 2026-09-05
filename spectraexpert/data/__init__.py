"""
功能概述：集中导出 SpectraExpert 的光谱数据接口与唯一的版本化分子结构 schema。
主要职责：公开 ``SpectralDataset``/``create_dataloader``、结构配置与归一化器、单图/批图
dataclass，以及 SMILES 图化、拼批、Morgan fingerprint 和分子描述符函数。
代码库关系：供训练、评测、缓存和推理模块从 ``spectraexpert.data`` 使用稳定入口；具体光谱逻辑
位于 ``spectral_dataset``，结构编码只由 ``structure_features`` 和 ``molecular_graph`` 定义。
输入输出与副作用：本模块仅重导出符号，不读取 Parquet、不图化分子也不创建 tensor；实际错误处理
与副作用遵循各实现模块的契约。
"""

from spectraexpert.data.molecular_graph import (
    GraphBatch,
    MoleculeGraph,
    collate_graphs,
    featurize_smiles,
)
from spectraexpert.data.spectral_dataset import SpectralDataset, create_dataloader
from spectraexpert.data.structure_features import (
    BOND_TYPE_TO_ID,
    DESCRIPTOR_NAMES,
    DescriptorNormalizer,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    StructureFeatureConfig,
    molecular_descriptors,
    morgan_fingerprint,
)

__all__ = [
    "BOND_TYPE_TO_ID",
    "DESCRIPTOR_NAMES",
    "DescriptorNormalizer",
    "EDGE_FEATURE_NAMES",
    "GraphBatch",
    "MoleculeGraph",
    "NODE_FEATURE_NAMES",
    "SpectralDataset",
    "StructureFeatureConfig",
    "collate_graphs",
    "create_dataloader",
    "featurize_smiles",
    "molecular_descriptors",
    "morgan_fingerprint",
]

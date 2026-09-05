"""
功能概述：验证 canonical SMILES 到二维重原子图以及无 padding 图批次的端到端公开契约。
主要职责：覆盖七列节点/三列边特征、离散芳香/单双键类别、双向边、非法输入、无键分子、
跨图 edge offset、graph membership、批次 metadata 和设备迁移。
代码库关系：直接消费 ``spectraexpert.data.molecular_graph``，为后续 GraphEncoder、结构数据集和
离线特征缓存提供回归保护；测试依赖 ``structure_features`` 中唯一的键类型映射。
输入输出与副作用：只图化少量内存 SMILES 并拼接 tensor，不访问大型 QM9S 数据或持久化文件；
解析失败或 tensor 契约不符时以 ``ValueError`` 或 pytest assertion 明确失败。
"""

import pytest
import torch

from spectraexpert.data.molecular_graph import (
    GraphBatch,
    MoleculeGraph,
    collate_graphs,
    featurize_smiles,
)
from spectraexpert.data.structure_features import BOND_TYPE_TO_ID


def test_benzene_uses_distinct_aromatic_bond_category():
    graph = featurize_smiles("c1ccccc1", mol_id=7)

    assert isinstance(graph, MoleculeGraph)
    assert graph.node_features.shape == (6, 7)
    assert graph.edge_index.shape == (2, 12)
    assert graph.edge_features.shape == (12, 3)
    assert set(graph.edge_features[:, 0].tolist()) == {
        BOND_TYPE_TO_ID["AROMATIC"]
    }
    assert graph.fingerprint.shape == (2048,)
    assert graph.metadata["morgan_radius"] == 2
    assert graph.metadata["use_chirality"] is False


@pytest.mark.parametrize(
    ("smiles", "expected_bond_type"),
    [
        ("C1CCCCC1", "SINGLE"),
        ("C=C", "DOUBLE"),
        ("C#N", "TRIPLE"),
    ],
)
def test_bond_orders_use_discrete_categories(smiles, expected_bond_type):
    graph = featurize_smiles(smiles, mol_id=1)

    assert set(graph.edge_features[:, 0].tolist()) == {
        BOND_TYPE_TO_ID[expected_bond_type]
    }


def test_node_features_use_atomic_number_and_fixed_local_environment_columns():
    graph = featurize_smiles("OCC", mol_id=3)

    assert graph.canonical_smiles == "CCO"
    assert graph.node_features.tolist() == [
        [6, 1, 4, 3, 3, 0, 0],
        [6, 2, 4, 3, 2, 0, 0],
        [8, 1, 2, 3, 1, 0, 0],
    ]


@pytest.mark.parametrize("smiles", ["not-a-smiles", ""])
def test_invalid_or_empty_smiles_raise_value_error(smiles):
    with pytest.raises(ValueError, match="SMILES"):
        featurize_smiles(smiles, mol_id=9)


def test_single_atom_graph_has_empty_edge_tensors():
    graph = featurize_smiles("O", mol_id=2)

    assert graph.node_features.shape == (1, 7)
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_features.shape == (0, 3)


def test_collate_offsets_edges_and_preserves_graph_membership():
    batch = collate_graphs(
        [
            featurize_smiles("CC", mol_id=1),
            featurize_smiles("O", mol_id=2),
        ]
    )

    assert isinstance(batch, GraphBatch)
    assert batch.node_features.shape[0] == 3
    assert batch.graph_index.tolist() == [0, 0, 1]
    assert batch.edge_index.tolist() == [[0, 1], [1, 0]]
    assert batch.edge_index.max().item() < 3
    assert batch.edge_features.shape == (2, 3)
    assert batch.fingerprints.shape == (2, 2048)
    assert batch.descriptors.shape == (2, 15)
    assert batch.mol_ids.tolist() == [1, 2]
    assert batch.canonical_smiles == ("CC", "O")


def test_collate_supports_a_batch_containing_only_bondless_graphs():
    batch = collate_graphs(
        [featurize_smiles("O", mol_id=2), featurize_smiles("N", mol_id=4)]
    )

    assert batch.edge_index.shape == (2, 0)
    assert batch.edge_features.shape == (0, 3)
    assert batch.graph_index.tolist() == [0, 1]


def test_graph_batch_to_moves_every_tensor_and_preserves_text_metadata():
    batch = collate_graphs([featurize_smiles("CO", mol_id=5)])

    moved = batch.to("cpu")

    for field_name in (
        "node_features",
        "edge_index",
        "edge_features",
        "graph_index",
        "fingerprints",
        "descriptors",
        "mol_ids",
    ):
        assert getattr(moved, field_name).device.type == "cpu"
    assert moved.canonical_smiles == batch.canonical_smiles


def test_collate_rejects_an_empty_graph_sequence():
    with pytest.raises(ValueError, match="at least one"):
        collate_graphs([])

"""
功能概述：验证纯 PyTorch 两跳图编码器的尺度、置换不变性、边依赖和异常输入边界。
主要职责：使用真实 GraphBatch 检查 0/1/2-hop 梯度、mean/sum 规模信息和无边图，阻止跨图污染。
代码库关系：直接消费 molecular_graph 的唯一特征 schema 与 models.graph_encoder；不读取数据集。
输入输出与副作用：构造微型分子与内存 tensor；仅运行前向/反向，不修改磁盘或启动实验。
"""

from dataclasses import replace

import pytest
import torch

from spectraexpert.data.molecular_graph import collate_graphs, featurize_smiles


def _model(**kwargs):
    from spectraexpert.models.graph_encoder import GraphEncoder

    torch.manual_seed(7)
    return GraphEncoder(hidden_dim=32, projection_dim=16, **kwargs)


def test_graph_encoder_exposes_three_scales_and_unit_embedding():
    batch = collate_graphs([featurize_smiles("CCO", 1), featurize_smiles("c1ccccc1", 2)])
    out = _model(num_layers=2)(batch)
    assert len(out.node_states) == 3
    assert [h.shape for h in out.node_states] == [(9, 32)] * 3
    assert out.backbone.shape == (2, 32)
    assert out.embedding.shape == (2, 16)
    torch.testing.assert_close(out.embedding.norm(dim=1), torch.ones(2))
    for h in out.node_states:
        h.retain_grad()
    out.embedding[:, 0].sum().backward()
    assert all(h.grad is not None and h.grad.abs().sum() > 0 for h in out.node_states)


def test_batch_and_atom_permutations_preserve_per_molecule_embeddings():
    graphs = [featurize_smiles("CCO", 1), featurize_smiles("N", 2)]
    model = _model().eval().double()
    batch = collate_graphs(graphs)
    output = model(batch)
    individual = torch.cat([model(collate_graphs([g])).embedding for g in graphs])
    torch.testing.assert_close(output.embedding, individual)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    permuted = replace(batch, node_features=batch.node_features[permutation],
                       graph_index=batch.graph_index[permutation], edge_index=inverse[batch.edge_index])
    torch.testing.assert_close(model(permuted).embedding, output.embedding)
    assert output.backbone.dtype == torch.float64


def test_bond_features_change_messages_without_changing_zero_hop_states():
    batch = collate_graphs([featurize_smiles("CC", 1)])
    changed = batch.edge_features.clone()
    changed[:, 0] = 2
    model = _model().eval()
    before, after = model(batch), model(replace(batch, edge_features=changed))
    torch.testing.assert_close(before.node_states[0], after.node_states[0])
    assert not torch.allclose(before.node_states[1], after.node_states[1])
    assert not torch.allclose(before.embedding, after.embedding)
    before.embedding[:, 0].sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if "edge_embeddings" in name)


def test_no_edge_graphs_are_finite_and_sum_readout_retains_atom_count():
    batch = collate_graphs([featurize_smiles("C", 1), featurize_smiles("C.C", 2)])
    output = _model().eval()(batch)
    assert all(torch.isfinite(h).all() for h in output.node_states)
    torch.testing.assert_close(output.embedding.norm(dim=1), torch.ones(2))
    for h in output.node_states:
        torch.testing.assert_close(h[0], h[1])
        torch.testing.assert_close(h[1], h[2])
    assert not torch.allclose(output.backbone[0], output.backbone[1])


@pytest.mark.parametrize("num_layers", [0, 1, 3])
def test_rejects_message_passing_depth_other_than_two(num_layers):
    with pytest.raises(ValueError, match="two|2"):
        _model(num_layers=num_layers)


@pytest.mark.parametrize("case", ["empty_batch", "empty_graph", "missing_id", "out_of_range", "cross_graph"])
def test_rejects_empty_or_inconsistent_graph_membership(case):
    batch = collate_graphs([featurize_smiles("CC", 1), featurize_smiles("N", 2)])
    if case == "empty_batch":
        batch = replace(batch, mol_ids=torch.empty(0, dtype=torch.long))
    elif case == "empty_graph":
        batch = replace(batch, node_features=batch.node_features[:0], graph_index=batch.graph_index[:0],
                        edge_index=batch.edge_index[:, :0], edge_features=batch.edge_features[:0])
    elif case == "missing_id":
        batch = replace(batch, graph_index=torch.tensor([0, 0, 0]))
    elif case == "out_of_range":
        batch = replace(batch, graph_index=torch.tensor([0, 0, 2]))
    else:
        batch = replace(batch, edge_index=torch.tensor([[0, 2], [2, 0]]))
    with pytest.raises(ValueError):
        _model()(batch)


def test_mean_and_sum_readouts_all_receive_gradient():
    batch = collate_graphs([featurize_smiles("CO", 1)])
    model = _model().eval()
    captured = []
    handle = model.readout.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    output = model(batch)
    handle.remove()
    features = captured[0]
    features.retain_grad()
    # 每一尺度按 mean、sum 顺序拼接；两个节点时 sum 必为 mean 的两倍。
    assert features.shape == (1, 192)
    for scale in range(3):
        mean, total = features[:, scale * 64:(scale + 1) * 64].chunk(2, dim=1)
        torch.testing.assert_close(total, mean * 2)
    output.embedding[:, 0].sum().backward()
    assert all(chunk.abs().sum() > 0 for chunk in features.grad.chunk(6, dim=1))

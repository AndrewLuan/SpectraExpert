"""
功能概述：验证双编码器训练输出、光谱独立查询和可序列化模型恢复契约。
主要职责：锁定 legacy Transformer forward 数值和参数键，检查辅助头只依赖两个光谱视图。
代码库关系：使用 SpectrumTransformer、GraphEncoder 与真实 molecular_graph fixture；覆盖模型包公开 API。
输入输出与副作用：使用小 tensor 进行前向/反向和内存 JSON roundtrip，无外部数据、网络或训练产物。
"""

from dataclasses import replace
import inspect
import json

import pytest
import torch
import torch.nn.functional as F

from spectraexpert.data.molecular_graph import collate_graphs, featurize_smiles
from spectraexpert.models.spectrum_encoder import SpectrumTransformer


def _model(predict_fingerprint=False):
    from spectraexpert.models import GraphEncoder, StructureAwareRetriever

    return StructureAwareRetriever(
        spectrum_encoder=SpectrumTransformer(signal_length=65, patch_size=8, stride=4,
                                             embed_dim=32, depth=1, num_heads=4, projection_dim=16),
        graph_encoder=GraphEncoder(hidden_dim=24, projection_dim=16),
        descriptor_dim=15, predict_fingerprint=predict_fingerprint,
    )


def test_spectrum_backbone_split_preserves_legacy_forward_and_state_keys():
    model = SpectrumTransformer(signal_length=65, patch_size=8, stride=4, embed_dim=32,
                                depth=1, num_heads=4, projection_dim=16).eval()
    x = torch.randn(2, 65)
    # 固定修改前单层 checkpoint 的公开参数键，防止恢复时 silently missing/unexpected keys。
    legacy_keys = {
        "pos_embed", "patch_embed.proj.weight", "patch_embed.proj.bias",
        "encoder.layers.0.self_attn.in_proj_weight", "encoder.layers.0.self_attn.in_proj_bias",
        "encoder.layers.0.self_attn.out_proj.weight", "encoder.layers.0.self_attn.out_proj.bias",
        "encoder.layers.0.linear1.weight", "encoder.layers.0.linear1.bias",
        "encoder.layers.0.linear2.weight", "encoder.layers.0.linear2.bias",
        "encoder.layers.0.norm1.weight", "encoder.layers.0.norm1.bias",
        "encoder.layers.0.norm2.weight", "encoder.layers.0.norm2.bias",
        "norm.weight", "norm.bias", "projection.0.weight", "projection.2.weight",
    }
    legacy_pooled = model.norm(model.encoder(model.pos_drop(model.patch_embed(x) + model.pos_embed))).mean(1)
    expected = F.normalize(model.projection(legacy_pooled), dim=1)
    torch.testing.assert_close(model.encode_backbone(x), legacy_pooled)
    torch.testing.assert_close(model.project(model.encode_backbone(x)), expected)
    torch.testing.assert_close(model(x), expected)
    assert (model.embed_dim, model.projection_dim) == (32, 16)
    assert set(model.state_dict()) == legacy_keys


@pytest.mark.parametrize("predict_fingerprint", [False, True])
def test_multimodal_outputs_and_auxiliary_gradient_depend_only_on_spectra(predict_fingerprint):
    model = _model(predict_fingerprint).eval()
    query1, query2 = torch.randn(2, 65), torch.randn(2, 65)
    graphs = collate_graphs([featurize_smiles("CCO", 1), featurize_smiles("N", 2)])
    out = model(query1, query2, graphs)
    assert out.spectrum_view1_features.shape == out.spectrum_view2_features.shape == (2, 32)
    assert out.graph_features.shape == (2, 24)
    for embedding in (out.spectrum_view1_embedding, out.spectrum_view2_embedding, out.graph_embedding):
        assert embedding.shape == (2, 16)
        torch.testing.assert_close(embedding.norm(dim=1), torch.ones(2))
    assert out.descriptor_prediction.shape == (2, 15)
    assert out.element_presence_logits.shape == (2, 5)
    assert (out.fingerprint_logits is not None) == predict_fingerprint
    if predict_fingerprint:
        assert out.fingerprint_logits.shape == (2, 2048)
    altered = replace(graphs, descriptors=torch.full_like(graphs.descriptors, float("nan")),
                      fingerprints=~graphs.fingerprints, canonical_smiles=("unavailable", "unavailable"))
    changed = model(query1, query2, altered)
    torch.testing.assert_close(out.descriptor_prediction, changed.descriptor_prediction)
    torch.testing.assert_close(out.graph_embedding, changed.graph_embedding)
    swapped = model(query2, query1, graphs)
    torch.testing.assert_close(out.descriptor_prediction, swapped.descriptor_prediction)
    loss = out.descriptor_prediction.square().sum() + out.element_presence_logits.square().sum()
    if out.fingerprint_logits is not None:
        loss = loss + out.fingerprint_logits.square().mean()
    loss.backward()
    assert model.spectrum_encoder.patch_embed.proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.graph_encoder.parameters())


def test_query_encoding_and_auxiliary_prediction_need_spectrum_only():
    model = _model(True).eval()
    query = torch.randn(2, 65)
    assert list(inspect.signature(model.encode_spectrum).parameters) == ["spectrum"]
    embedding = model.encode_spectrum(query)
    torch.testing.assert_close(embedding, model.spectrum_encoder(query))
    descriptor, fingerprint = model.predict_auxiliary(query)
    assert descriptor.shape == (2, 15)
    assert fingerprint.shape == (2, 2048)
    assert ((fingerprint >= 0) & (fingerprint <= 1)).all()
    assert _model(False).predict_auxiliary(query)[1] is None
    graphs = collate_graphs([featurize_smiles("CC", 1)])
    torch.testing.assert_close(model.encode_graph(graphs), model.graph_encoder(graphs).embedding)


def test_model_config_reconstructs_architecture_and_exact_eval_outputs():
    from spectraexpert.models import StructureAwareRetriever

    model = _model(True).eval()
    config = model.to_config()
    assert set(config) == {"signal_length", "patch_size", "stride", "embed_dim", "depth", "num_heads",
                           "projection_dim", "graph_hidden_dim", "descriptor_dim", "predict_fingerprint"}
    restored = StructureAwareRetriever.from_config(json.loads(json.dumps(config))).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    query = torch.randn(2, 65)
    graphs = collate_graphs([featurize_smiles("C", 1), featurize_smiles("CCO", 2)])
    before, after = model(query, query, graphs), restored(query, query, graphs)
    for field in before.__dataclass_fields__:
        torch.testing.assert_close(getattr(before, field), getattr(after, field))
    config["embed_dim"] = 900
    assert model.to_config()["embed_dim"] == 32


def test_rejects_mismatched_projection_and_training_batch_sizes():
    from spectraexpert.models import GraphEncoder, StructureAwareRetriever

    good = _model()
    with pytest.raises(ValueError, match="projection"):
        StructureAwareRetriever(good.spectrum_encoder, GraphEncoder(32, 12))
    graphs = collate_graphs([featurize_smiles("C", 1)])
    with pytest.raises(ValueError, match="batch|shape"):
        good(torch.randn(2, 65), torch.randn(2, 65), graphs)
    with pytest.raises(ValueError, match="batch|shape"):
        good(torch.randn(1, 65), torch.randn(2, 65), graphs)


@pytest.mark.parametrize("override", [{"mlp_ratio": 2.0}, {"dropout": 0.0}, {"graph_dropout": 0.0}])
def test_fixed_checkpoint_schema_rejects_nondefault_unserialized_parameters(override):
    from spectraexpert.models import GraphEncoder, StructureAwareRetriever

    config = dict(override)
    graph_dropout = config.pop("graph_dropout", 0.1)
    spectrum = SpectrumTransformer(signal_length=65, patch_size=8, stride=4, embed_dim=32,
                                   depth=1, num_heads=4, projection_dim=16, **config)
    graph = GraphEncoder(32, 16, dropout=graph_dropout)
    with pytest.raises(ValueError, match="checkpoint schema"):
        StructureAwareRetriever(spectrum, graph)

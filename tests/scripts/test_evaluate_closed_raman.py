"""功能概述：测试闭库检索的全排名指标与非连续 ID 加载契约。
主要职责：验证 MRR 不截断、Top-K 边界，打乱谱行序仍按冻结 rows/mol_id 连接；
并检查错误哈希被拒绝。仅在 tmp_path 写微型 Parquet，不读取真实数据。
"""
import json
import numpy as np
import pandas as pd
import pytest
from scripts.evaluate_closed_raman import summarize, load_role, digest
from spectraexpert.inference import SpectralEncoder


def test_full_rank_metrics():
    result = summarize([dict(group_rank=i, latency_ms=1., canonical_smiles_hit_top1=False) for i in (1, 5, 10, 100)])
    assert result['top1'] == .25
    assert result['top5'] == .5
    assert result['top10'] == .75
    assert result['mrr'] == pytest.approx((1 + .2 + .1 + .01) / 4)


def test_noncontinuous_ids_and_source_hash(tmp_path):
    folder = tmp_path / 'experimental_test'
    folder.mkdir()
    rows = [dict(mol_id=19, group_hash='b'), dict(mol_id=3, group_hash='a')]
    pd.DataFrame(rows).to_parquet(folder / 'smiles.parquet')
    (folder / 'rows.json').write_text(json.dumps(rows))
    axis = np.arange(500, 4001)
    pd.DataFrame({'wavenumber': axis}).to_parquet(folder / 'wavenumber.parquet')
    a = np.linspace(.1, 1, 3501)
    matrix = pd.DataFrame([a, a[::-1]], columns=[f'w_{w}' for w in axis])
    matrix.insert(0, 'mol_id', [3, 19])
    matrix.to_parquet(folder / 'spectra.parquet')
    sample = {'outputs_sha256': {'experimental_test': {p.name: digest(p) for p in folder.iterdir()}}}
    metadata, spectra = load_role(tmp_path, 'experimental_test', sample)
    assert metadata['mol_id'].tolist() == [19, 3]
    order, _ = SpectralEncoder.retrieve(a, spectra, topk=2)
    assert metadata.iloc[order[0]]['mol_id'] == 3
    sample['outputs_sha256']['experimental_test']['rows.json'] = 'wrong'
    with pytest.raises(ValueError, match='hash mismatch'):
        load_role(tmp_path, 'experimental_test', sample)

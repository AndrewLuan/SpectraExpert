"""功能概述：验证理论 Raman 转换与正式训练入口的兼容性。
主要职责：fixture 构造不同 split 的 gzip 谱；测试坐标插值、坏谱过滤、重复 identity、
跨集泄漏拒绝及 v2 单 epoch checkpoint。依赖真实 prepare/train，不模拟核心计算。
输入输出与副作用：只在 pytest tmp_path 写小型合成谱和 checkpoint，不访问生产数据或 GPU。
"""
import gzip
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.prepare_specgpt_theory import prepare
from spectraexpert.train_structure_aware import TrainConfig, train


def source(tmp_path, *, leakage=False):
    outputs = {}
    for split, smiles in [('train', ['CCO', 'CCO', 'CCN', 'CCC']),
                          ('valid', ['CO', 'CN']), ('test', ['CC', 'C'])]:
        path = tmp_path / f'{split}.jsonl.gz'
        outputs[split] = str(path)
        with gzip.open(path, 'wt') as handle:
            for i, s in enumerate(smiles):
                if leakage and split == 'valid' and i == 0:
                    s = 'CCO'
                handle.write(json.dumps(dict(smiles=s, group_hash=s, modality='raman',
                    spectrum=(np.linspace(0, 1, 3000) ** (i + 1)).tolist())) + '\n')
            handle.write(json.dumps(dict(smiles={'train': 'O', 'valid': 'N', 'test': 'F'}[split], group_hash='zero-' + split,
                modality='raman', spectrum=[0.] * 3000)) + '\n')
    manifest = tmp_path / 'source.json'
    manifest.write_text(json.dumps(dict(processing_version='continuous_v1', domain='theory',
        config=dict(grid_size=3000, modalities={'Raman': dict(axis_min=0, axis_max=4000, axis_unit='cm-1')}),
        files=[dict(domain='theory', modality='Raman', outputs=outputs)])))
    return manifest


def test_conversion_and_v2_training(tmp_path):
    torch.set_num_threads(1)
    output = tmp_path / 'prepared'
    report = prepare(source(tmp_path), output, batch_size=2)
    assert report['counts'] == {'read': 11, 'accepted': 8, 'constant_spectrum': 3}
    spectra = pd.read_parquet(output / 'spectra.parquet')
    assert spectra.shape == (8, 3502)
    assert spectra.iloc[0]['w_500'] == pytest.approx(.125)
    assert spectra.iloc[0]['w_4000'] == 1.
    manifest = json.loads((output / 'split.json').read_text())
    assert manifest['train'] == [0, 1, 2, 3]
    assert manifest['validation'] == [4, 5]
    # 大 patch 让 CPU 冒烟只用少量 token，仍消费实际 3501 点谱。
    train(TrainConfig.tiny(data_dir=output, output_dir=tmp_path / 'run', patch_size=128, stride=128))
    assert (tmp_path / 'run' / 'best.pt').is_file()


def test_leakage_is_atomic(tmp_path):
    output = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='leakage'):
        prepare(source(tmp_path, leakage=True), output)
    assert not output.exists()
    assert not list(tmp_path.glob('prepared.staging-*'))


def test_rejects_experimental_manifest(tmp_path):
    manifest = source(tmp_path)
    record = json.loads(manifest.read_text())
    record['files'][0]['domain'] = 'exp'
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='only theory Raman'):
        prepare(manifest, tmp_path / 'prepared')


@pytest.mark.parametrize('bad_spectrum', [[1., 2.], [float('nan')] * 3000])
def test_invalid_spectra_are_counted(tmp_path, bad_spectrum):
    manifest = source(tmp_path)
    path = tmp_path / 'train.jsonl.gz'
    with gzip.open(path, 'rt') as handle:
        records = [json.loads(line) for line in handle]
    records[0]['spectrum'] = bad_spectrum
    with gzip.open(path, 'wt') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')
    output = tmp_path / 'prepared'
    report = prepare(manifest, output)
    assert report['counts']['invalid_spectrum'] == 1
    assert report['counts']['accepted'] == 7
    with pytest.raises(FileExistsError):
        prepare(manifest, output)

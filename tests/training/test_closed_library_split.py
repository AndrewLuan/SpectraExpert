"""功能概述：验证冻结闭库数据在正式训练入口的 provenance/分区/保护组契约。
主要职责：closed_data 构造独立微型理论数据与样本清单；测试真实单轮训练及每个绑定字段
被破坏时在创建训练输出前失败。复用已有 tiny_dataset_dir，不访问实验光谱或远端。
输入输出与副作用：仅使用 pytest tmp_path；源数据哈希按各测试场景显式重建。
"""
import json

import pandas as pd
import pytest

from tests.training.test_train_smoke import tiny_dataset_dir
from scripts.freeze_raman_training_split import build_split, digest
from spectraexpert.train_structure_aware import TrainConfig, _prepare_data, train


@pytest.fixture
def closed_data(tiny_dataset_dir):
    directory = tiny_dataset_dir
    smiles = pd.read_parquet(directory / 'smiles.parquet')
    smiles['group_hash'] = [f'group-{i}' for i in smiles['mol_id']]
    smiles['source_path'] = '/source/theory/raman/train.jsonl.gz'
    smiles.to_parquet(directory / 'smiles.parquet')
    rows = smiles.to_dict('records')
    (directory / 'rows.json').write_text(json.dumps(rows))
    hashes = {n: digest(directory / n) for n in ('spectra.parquet', 'smiles.parquet', 'wavenumber.parquet')}
    sample = dict(schema_version='raman_closed_library_sample_v1', seed=0,
                  candidate_count=12, test_count=1, candidate_groups=smiles['group_hash'].tolist(),
                  test_groups=['group-0'], outputs_sha256={'theory_train_candidate': hashes | {'rows.json': digest(directory / 'rows.json')}})
    sample_path = directory.parent / 'sample_manifest.json'
    sample_path.write_text(json.dumps(sample))
    split = build_split(rows, sample['test_groups'], count=2)
    split.update(source_sha256=hashes, source_rows_sha256=digest(directory / 'rows.json'),
                 sample_manifest_sha256=digest(sample_path))
    (directory / 'split.json').write_text(json.dumps(split))
    return directory


def test_closed_library_training(closed_data, tmp_path):
    config = TrainConfig.tiny(data_dir=closed_data, output_dir=tmp_path / 'run')
    training, validation, _, metadata = _prepare_data(config)
    assert len(training) == 10 and len(validation) == 2
    assert 0 in training.mol_ids and 0 not in validation.mol_ids
    assert len(metadata['source_smiles_sha256']) == 64
    result = train(config)
    assert result['graph_gradient_norm'] > 0
    assert (config.output_dir / 'best.pt').is_file()


@pytest.mark.parametrize('field', ['source_sha256', 'source_rows_sha256', 'sample_manifest_sha256',
    'protected_test_groups', 'train_count', 'validation_groups', 'train'])
def test_rejects_broken_contract(closed_data, tmp_path, field):
    path = closed_data / 'split.json'
    split = json.loads(path.read_text())
    split[field] = [] if isinstance(split[field], list) else 'corrupt'
    path.write_text(json.dumps(split))
    output = tmp_path / 'run'
    with pytest.raises(ValueError):
        train(TrainConfig.tiny(data_dir=closed_data, output_dir=output))
    assert not output.exists()


def test_protected_identity_cannot_move_to_validation(closed_data):
    path = closed_data / 'split.json'
    split = json.loads(path.read_text())
    replacement = split['validation'][0]
    split['train'][split['train'].index(0)] = replacement
    split['validation'][0] = 0
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match='protected groups'):
        _prepare_data(TrainConfig.tiny(data_dir=closed_data, output_dir=closed_data / 'run'))

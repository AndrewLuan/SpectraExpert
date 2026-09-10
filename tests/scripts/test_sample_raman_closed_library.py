"""功能概述：用微型合成索引验证闭库采样的角色隔离、包含关系与可复现性。
主要职责：test_sample_contract 建六个合成 gzip 分片及 pilot 缓存，执行完整采样，检查
200/N 同类的强制包含契约、理论来源、实验来源、固定行数、固定随机顺序与覆盖保护。
代码库关系：通过 scripts 目录加载独立 sample_raman_closed_library；不访问服务器或真实数据。
输入输出与副作用：全部临时 SQLite、JSONL、Parquet 限于 pytest tmp_path；不训练模型。
"""
import gzip
import importlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest


def test_sample_contract(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    mod = importlib.import_module('sample_raman_closed_library')
    groups = [f'{i:032x}' for i in range(1,5)]
    index = tmp_path/'index.sqlite'
    with sqlite3.connect(index) as con:
        con.execute('CREATE TABLE molecule_groups(group_hash BLOB,exp_raman_count INT,theory_raman_count INT)')
        con.executemany('INSERT INTO molecule_groups VALUES(?,?,?)', [(bytes.fromhex(g),int(i<3),1) for i,g in enumerate(groups)])
    vector = np.linspace(0,1,3000)
    pilot = tmp_path/'pilot'
    pilot.mkdir()
    records = [dict(group_hash=g,smiles='CC',source='SDBS',condition='',coverage=[0,3000],split='train',line=i+1) for i,g in enumerate(groups[:3])]
    (pilot/'records.json').write_text(json.dumps({'query':records,'candidate':records}))
    (pilot/'protocol.json').write_text(json.dumps({'overlap':mod.load_overlap(index)}))
    np.savez(pilot/'overlap_arrays.npz',query=np.stack([vector]*3),candidate=np.stack([vector]*3))
    root = tmp_path/'continuous'
    source = root/'combined'/'theory'/'raman'
    source.mkdir(parents=True)
    for split in ('train','valid','test'):
        with gzip.open(source/f'{split}.jsonl.gz','wt') as f:
            if split=='train':
                for g in groups:
                    f.write(json.dumps(dict(group_hash=g,smiles='CC',source='theory',condition='',coverage=[0,3000],spectrum=vector.tolist()))+'\n')
    for name in ('a','b'):
        mod.sample(root,index,pilot,tmp_path/name,3,2,0)
    first=json.loads((tmp_path/'a'/'sample_manifest.json').read_text())
    second=json.loads((tmp_path/'b'/'sample_manifest.json').read_text())
    assert len(first['test_groups'])==2 and len(first['candidate_groups'])==3
    assert set(first['test_groups']) <= set(first['candidate_groups'])
    assert first['test_groups']==second['test_groups']
    assert first['candidate_groups']==second['candidate_groups']
    train=pq.read_table(tmp_path/'a'/'theory_train_candidate'/'smiles.parquet').to_pydict()
    test=pq.read_table(tmp_path/'a'/'experimental_test'/'smiles.parquet').to_pydict()
    assert set(train['source'])=={'theory'} and set(test['source'])=={'SDBS'}
    assert set(test['mol_id']) <= set(train['mol_id'])
    assert pq.ParquetFile(tmp_path/'a'/'theory_train_candidate'/'spectra.parquet').metadata.num_rows==3
    assert len(pq.read_table(tmp_path/'a'/'experimental_test'/'wavenumber.parquet'))==3501
    assert not mod.usable(np.ones(3000),[0,3000])
    assert not mod.usable(vector,[2000,3000])
    with pytest.raises(FileExistsError):
        mod.sample(root,index,pilot,tmp_path/'a',3,2,0)

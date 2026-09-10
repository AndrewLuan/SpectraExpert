"""功能概述：验证精确闭库修复保持槽位、完整结构语义和固定选择规则。
主要职责：row 创建合成来源谱；主测试覆盖同位素替换、不可补齐目标替换和确定性；
失败测试覆盖候选重组不足、低质量谱以及缓存与实际来源不一致。
代码库关系：调用 repair_closed_raman_exact 的选择器与来源校验，不依赖真实数据和 API。
输入输出与副作用：源谱只写 pytest 临时目录，不修改项目数据、不训练。
"""
import gzip
import json

import numpy as np
import pytest

from scripts.repair_closed_raman_exact import canonical, read_records, select_changes, verify_sources


def row(group, smiles, mid=0, domain='theory'):
    return dict(group_hash=group, smiles=smiles, mol_id=mid, source=domain,
                source_path=f'/combined/{domain}/raman/train.jsonl.gz', source_split='train',
                source_line=1, condition='', coverage=[0,3000], spectrum=np.linspace(0,1,3000,dtype=np.float32))


def test_exact_repair_replacement_and_repeatability():
    candidates = [row('a','[13CH3]N',3), row('b','CC(O)C(=O)O',8), row('c','CCC',9)]
    queries = [row('a','CN',3,'exp'),row('b','C[C@H](O)C(=O)O',8,'exp')]
    cache = dict(candidate=[row('a','CN'),row('d','CCO'),row('c','CCC')],
                 query=[row('d','CCO',domain='exp'),row('c','CCC',domain='exp')])
    first, q, changes = select_changes(candidates,queries,cache)
    assert [r['mol_id'] for r in first] == [3,8,9]
    assert [r['mol_id'] for r in q] == [3,8]
    assert first[0]['smiles'] == 'CN' and q[0] is queries[0]
    assert first[1]['group_hash'] == q[1]['group_hash'] == 'd'
    assert changes == select_changes(candidates,queries,cache)[2]
    assert candidates[0]['smiles'] == '[13CH3]N'
    assert canonical('N[13CH3]') != canonical('CN')
    assert canonical('C[C@H](O)C(=O)O') != canonical('C[C@@H](O)C(=O)O')


def test_rejects_bad_quality_and_occupied_groups():
    bad = row('d','CCO'); bad['spectrum'][:] = 0
    with pytest.raises(ValueError,match='insufficient'):
        select_changes([row('a','[13CH3]N')],[row('a','CN',domain='exp')],
                       dict(candidate=[bad,row('a','[13CH3]N')],query=[row('d','CCO',domain='exp')]))


def test_source_verification_rejects_mislabeled_spectrum(tmp_path):
    item = row('a','CN'); path=tmp_path/'source.gz'; item['source_path']=str(path)
    actual={k:v for k,v in item.items() if k!='spectrum'} | {'spectrum':item['spectrum'].tolist()}
    with gzip.open(path,'wt') as f: f.write(json.dumps(actual)+'\n')
    assert verify_sources([item])[0]['checked_lines']==[1]
    item['smiles']='[13CH3]N'
    with pytest.raises(ValueError,match='metadata mismatch'):verify_sources([item])


def test_preserve_original_precision_before_interpolation(tmp_path):
    path=tmp_path/'old.gz'
    values=[0.123456,0.654321]
    with gzip.open(path,'wt') as f:f.write(json.dumps({'spectrum':values})+'\n')
    loaded=read_records(path)[0]['spectrum']
    assert loaded.dtype==np.float64
    assert np.array_equal(loaded,np.array(values))
    assert not np.array_equal(loaded,loaded.astype(np.float32))

"""功能概述：验证修正版 baseline 只留下八条新请求，并拒绝复用发生变化的旧请求。
主要职责：微型 200 槽位夹具模拟数据加载与峰表，实际写读请求/响应；验证准备与续跑幂等性。
代码库关系：调用 refresh_closed_llm_baseline.prepare；替换数据读取与摘要校验，不访问 API。
输入输出与副作用：只使用 pytest 临时目录；断言旧预测保留及请求漂移失败，不执行 GED。
"""
import copy
import json

import numpy as np
import pytest

from scripts import refresh_closed_llm_baseline as mod


def test_prepare_reuses_only_unchanged_queries(tmp_path,monkeypatch):
    root,parent,previous,output=[tmp_path/n for n in ('new','parent','old','out')]
    for p in (root,parent,previous):p.mkdir()
    for name in ('requests','responses'):(previous/name).mkdir()
    save=mod.base.save
    monkeypatch.setattr(mod.base,'digest',lambda p:'hash')
    save(root/'sample_manifest.json',dict(parent_manifest_sha256='hash',repair_report_sha256='hash'))
    save(root/'repair_report.json',dict(replaced_query_indices=list(range(8))))
    source_names=('spectraexpert/eval/evaluate.py','spectraexpert/eval/metrics.py',
                  'spectraexpert/pipeline/prompts/stage3_direct_identification.py')
    save(previous/'config.json',dict(manifest_sha256='hash',model=mod.base.MODEL,
         temperature=.2,max_tokens=4096,endpoint=mod.base.ENDPOINT,source_sha256=dict.fromkeys(source_names,'hash')))
    old=[dict(mol_id=i,ground_truth_smiles='CC',group_hash='g') for i in range(200)]
    new=copy.deepcopy(old)
    for r in new[:8]:r['ground_truth_smiles']='CN';r['group_hash']='h'
    old_s=np.ones((200,1),dtype=np.float32);new_s=old_s.copy();new_s[:8]=2
    monkeypatch.setattr(mod.base,'load_queries',lambda p,*a:(new,new_s,{'CC':'g','CN':'h'}) if p==root else (old,old_s,{}))
    monkeypatch.setattr(mod.base,'_PeakDetector',lambda c:None)
    monkeypatch.setattr(mod.base,'make_messages',lambda s,d:([{'content':str(s.tolist())}],[]))
    rows=[]
    for i in range(200):
        request=dict(model=mod.base.MODEL,temperature=.2,max_tokens=4096,messages=[{'content':str(old_s[i].tolist())}])
        save(previous/'requests'/f'{i:04d}.json',request)
        save(previous/'responses'/f'{i:04d}.json',dict(content='PREDICTED_SMILES: CC',query_index=i))
        rows.append(dict(old[i],query_index=i,strict_smiles='CC',legacy_smiles='CC',
                         strict={'group_match':True},legacy={'group_match':True}))
    save(previous/'results.json',dict(queries=rows))
    result=mod.prepare(root,parent,previous,output)
    assert len(result[3])==192 and result[4]==list(range(8))
    assert len(list((output/'responses').glob('*.json')))==192
    assert not (output/'responses/0000.json').exists()
    assert len(list((output/'requests').glob('*.json')))==200
    mod.prepare(root,parent,previous,output)
    cached=output/'responses/0100.json'
    original=cached.read_text();cached.unlink()
    with pytest.raises(ValueError,match='reused response'):mod.prepare(root,parent,previous,output)
    cached.write_text(original)
    save(previous/'requests/0100.json',{})
    with pytest.raises(ValueError,match='request differs'):mod.prepare(root,parent,previous,output)


def test_summary_keeps_null_ged_out_of_denominator():
    metric=dict(valid=True,exact_match=False,group_match=False,tanimoto=.2,legacy_ged_v1=3.,chemical_ged_v2=None)
    rows=[dict(api_error=None,strict=metric,legacy=metric)]
    s=mod.summarize(rows)
    assert s['strict']['valid_rate']==1
    assert s['strict']['chemical_ged_v2']==dict(n=0,mean=None,median=None)

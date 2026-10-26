"""功能概述：为消息消融增加经原表SHA与结构分区验证的苯中甲酰胺支持。
add_support只接受h_acyl_inventory导出的train/val，验证与恢复元数据同一原表；
全部test结构及重复冻结记录拒绝进入训练。积分总数超过结构H数视为不可能记录，
保存拒绝原因；少报积分保留为部分观测，不推定采集窗口或补未报告峰。
新图由原GINE权重编码，已有物理计数只复用冻结cache中的结构特征，未认证结构
保留physical_features的计数缺失掩码。整数积分恰好匹配总H仅启用原软分布监督。
返回新增train、独立val和证据，更新进程内graphs/metadata/labels；不写原prepared。
extra_weights保持原抽样分布的总质量，新增部分按连接结构均权；所有模型臂共用
同一fraction和seed。h_message_train只用原全val选epoch，新增val单列诊断。
"""
import collections
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch
from .data import graph,physical_features,identity,quantiles
from .h_conditions import local_environment
from .h_metadata import raw_conditions
from .h_acyl import acyl_environment,ACYL_NAMES
from .h_labels import make_key
from .observation import coverage
from .emission_model import make_model
from .prepare import sha


def extra_weights(original,rows,fraction):
    if not rows or not 0<fraction<1:raise ValueError('invalid_extra_support_fraction')
    original=original.to(dtype=torch.float64)
    counts=collections.Counter(r['group'] for r in rows);w=torch.tensor([1/counts[r['group']] for r in rows],dtype=torch.float64)
    return torch.cat([(1-fraction)*original/original.sum(),fraction*w/w.sum()])


def add_support(cache,metadata,labels,folder,manifest,owner,device):
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text());expected=manifest['raw_recovery_sha256']
    if (folder/'EXIT_CODE').read_text().strip()!='0' or list(summary['source_sha256'].values())!=[expected]:raise ValueError('extra_support_source_mismatch')
    existing={make_key(r) for r in cache['rows']};physics={r['smiles']:r['physics'] for r in cache['rows']};rows={'train':[],'val':[]};rejected=[];missing=[];required=set()
    for split in rows:
        for item in map(json.loads,(folder/('eligible_'+split+'.jsonl')).read_text().splitlines()):
            raw=item['record'];s,group,actual=identity(item['canonical_smiles'])
            if actual!=split or group!=item['group'] or identity(raw['smiles'])[1]!=group or actual=='test':raise ValueError('extra_support_split_mismatch')
            if not acyl_environment(s)[:,ACYL_NAMES.index('formyl')].any():raise ValueError('extra_support_not_formamide')
            f=np.asarray(physics[s],np.float32).copy() if s in physics else physical_features(s);h=round(float(f[0])*40);xs=raw['value']['x'];ys=np.asarray(raw['value']['y'],float)
            if not np.isfinite(ys).all() or (ys<=0).any() or ys.sum()>h+.05:
                rejected.append(dict(record_id=raw['record_id'],reason='reported_integration_exceeds_structural_H_or_invalid',structural_H=h,reported_H=float(ys.sum())));continue
            meta,_=raw_conditions(raw);row=dict(domain='experimental',source=raw['source'],record_id=str(raw['record_id']),nucleus='1H',smiles=s,split=split,group=group,physics=f,cond=meta['old_cond'],q=quantiles(xs,ys),observation=coverage(raw),measure='integral_weighted')
            key=make_key(row)
            if key in existing:rejected.append(dict(record_id=raw['record_id'],reason='already_frozen'));continue
            try:g=cache['graphs'][s].clone() if s in cache['graphs'] else graph(s)
            except ValueError as exc:rejected.append(dict(record_id=raw['record_id'],reason=str(exc)));continue
            if not hasattr(g,'local_environment'):g.local_environment=torch.tensor(local_environment(s))
            cache['graphs'][s]=g;required.add(s);metadata[key]=meta;labels[key]=dict(groups={},integral_consistent=bool(np.allclose(ys,ys.round(),atol=.001) and abs(ys.sum()-h)<.001));rows[split].append(row)
            if s not in physics:missing.append(s)
    if not rows['train'] or not rows['val'] or {r['group'] for r in rows['train']} & {r['group'] for r in rows['val']}:raise ValueError('empty_or_overlapping_extra_support')
    saved=torch.load(owner,map_location='cpu',weights_only=False);encoder=make_model(saved['args']).to(device).eval();encoder.load_state_dict(saved['state_dict'])
    with torch.no_grad():
        for s in sorted(required):
            g=cache['graphs'][s]
            if not hasattr(g,'frozen_embedding'):g.frozen_embedding=encoder.structure_embedding(Batch.from_data_list([g]).to(device)).cpu().clone()
    del encoder;torch.cuda.empty_cache()
    audit=dict(raw_sha256=expected,input_files_sha256={split:sha(folder/('eligible_'+split+'.jsonl')) for split in rows},accepted={split:dict(records=len(rs),connectivities=len({r['group'] for r in rs})) for split,rs in rows.items()},rejected=rejected,missing_certified_physics=sorted(set(missing)),validation_use='diagnostics only; original full macro val selects epoch')
    return rows['train'],rows['val'],audit

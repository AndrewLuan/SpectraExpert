"""功能概述：在固定训练结构子集上分解single/dual候选池的归一化与组成效应。
selection_manifest按预注册hash选64个非internal结构及每结构一个base query，只取已存在
observed/frequency_crop/peak_dropout；缺失变体写skip，不补数据。natural_row先剔除训练
追加正例并白名单化模型输入，评分后target_metrics才读取标签。raw_trace/instrument保留
实际ResearchRanker的33维median/固定scale/clamp与各role三维O median/clamp路径；
run_probes只替换中心，保留候选、raw特征、coarse、尺度、alpha/gamma及全部权重。
aggregate_differences/aggregate_common_diagnostics按共同候选数加权逐维差异并保留全局max，
中心另报query均值及固定scale归一化；pair_changes统计固定epsilon翻转/tie；query_report输出特征/中心/
归一化/logit/delta/cost差、自然及共同子集RR、竞争计数和精确RR代数分解。
main只读旧轮train缓存和冻结R0O1F0权重，核producer EXIT、身份/输入/源码SHA、角色、
有限模型及始末状态，独占写selection/query/summary/真实EXIT。无训练、调参、test读取。
"""
import argparse
import collections
import datetime
import os
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCHEMA = 'pool_normalization_counterfactual_20261023'
SEEDS = (193, 211)
GROUPS = 64
VARIANTS = ('observed', 'frequency_crop', 'peak_dropout')
EPS = 1e-7
RTOL = 1e-6
ATOL = 1e-7
CANDIDATE_SHA = '72b72140c09ae14ae8d0aa4edad624a151093c9df246b2052de8a3a4ea616bad'
KEYS = ('query_order', 'source', 'record_id', 'smiles', 'group')
CANDIDATE_FIELDS = ('features','coarse_scores','predicted_ppm','tags','H','K','per_quantile_sigma')
PROBES = ('S_native','D_native','D_use_S_Xcenter','D_use_S_Ocenter',
          'D_use_both_Scenters','S_use_both_Dcenters')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def value_sha(value):
    """保留tensor位模式、dtype/shape及嵌套标量，用于始末不变核验。"""
    digest = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            digest.update(('tensor:'+str(t.dtype)+':'+str(tuple(t.shape))).encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            digest.update(b'dict')
            for k in sorted(v, key=lambda x:str(x)):
                visit(k); visit(v[k])
        elif isinstance(v, (list, tuple)):
            digest.update(('sequence:'+str(len(v))).encode())
            for x in v: visit(x)
        else:
            digest.update((type(v).__name__+':'+repr(v)).encode())
        digest.update(b'\0')
    visit(value)
    return digest.hexdigest()


def identity(row):
    return tuple(row[k] for k in KEYS)+(row['variant'],)


def base_identity(row):
    return tuple(row[k] for k in KEYS)


def internal_group(group):
    return int(hashlib.sha256(('ranker_internal_317'+str(group)).encode()).hexdigest()[:12],16)%5 == 0


def identity_hash(value, prefix):
    return hashlib.sha256((prefix+json.dumps(value,separators=(',',':'),ensure_ascii=False)).encode()).hexdigest()


def index_rows(data):
    rows = data['rows']; indexed = {identity(r):r for r in rows}
    require(len(indexed)==len(rows), 'duplicate_query_identity')
    return indexed


def selection_manifest(indexes, count=GROUPS):
    """第一缓存作固定参考；其他seed/池不得通过交集筛选静默丢弃结构。"""
    reference = indexes[(193,'single')]
    groups = collections.defaultdict(set)
    for row in reference.values():
        if row['variant']=='observed' and not internal_group(row['group']):
            groups[row['group']].add(base_identity(row))
    require(len(groups)>=count, 'insufficient_train_groups')
    chosen = sorted(groups,key=lambda g:(hashlib.sha256(('pool_norm_diagnostic_20261023'+g).encode()).hexdigest(),g))[:count]
    entries, skips = [], []
    for group in chosen:
        base = min(groups[group],key=lambda k:(identity_hash(k,'pool_norm_base_query_20261023'),k))
        for variant in VARIANTS:
            key = base+(variant,)
            present = [key in ix for ix in indexes.values()]
            if not any(present):
                skips.append(dict(group=group,base_query=list(base),variant=variant,reason='variant_absent_in_all_seed_pool_caches'))
                continue
            require(all(present), 'cross_seed_pool_missing_variant')
            rows = [ix[key] for ix in indexes.values()]
            first = rows[0]
            for row in rows[1:]:
                require(identity(row)==key, 'query_identity_mismatch')
                for field in ('kind','mask_content_sha256','observation_input','query_observation'):
                    require(value_sha(row[field])==value_sha(first[field]), 'query_observation_mismatch:'+field)
            entries.append(dict(group=group,base_query=list(base),variant=variant,key=list(key),
                                mask_content_sha256=first['mask_content_sha256']))
    return dict(schema=SCHEMA,selection_scope='train_only_no_internal',group_count=count,
        group_hash_prefix='pool_norm_diagnostic_20261023',base_query_hash_prefix='pool_norm_base_query_20261023',
        groups=chosen,entries=entries,skips=skips,variants=list(VARIANTS),tuning=False)


def natural_row(row):
    """此返回值无身份或真实target；row_delta所需标签位固定为无目标，仅用于全512中心。"""
    n = row['evaluation_pool_size']
    require(type(n) is int and n==512, 'natural_pool_must_be_512')
    indices = row['candidate_indices']
    require(len(indices) in (n,n+1) and len(indices[:n].unique())==n, 'invalid_candidate_pool')
    require(indices.dtype in (torch.int32,torch.int64) and bool(((indices[:n]>=0)&(indices[:n]<114982)).all()), 'invalid_candidate_indices')
    result = dict(evaluation_pool_size=n,target_position=-1,target_in_coarse_pool=False,
                  kind=row['kind'],
                  observation_input={k:row['observation_input'][k] for k in
                      ('semantics','kind','xs','weights','reported_count','windows','partial_peaks') if k in row['observation_input']},
                  query_observation={k:row['query_observation'][k] for k in ('windows','partial_peaks') if k in row['query_observation']})
    for field in CANDIDATE_FIELDS:
        if field not in row: continue
        require(len(row[field])==len(indices), 'candidate_tensor_length:'+field)
        result[field] = row[field][:n]
    require(result['features'].shape==(n,33) and result['coarse_scores'].shape==(n,), 'feature_dimensions')
    require(torch.isfinite(result['features']).all() and torch.isfinite(result['coarse_scores']).all(), 'nonfinite_raw_X')
    require(type(result['kind']) is int and 0<=result['kind']<4, 'invalid_kind')
    return result


def raw_trace(model, row, scale, device):
    x = row['features'].to(device)
    require(scale.shape==(33,) and torch.isfinite(scale).all() and bool((scale>0).all()), 'invalid_fixed_scale')
    sigma = row.get('per_quantile_sigma')
    o = model.observation_model(row['predicted_ppm'].to(device),row['tags'].to(device).long(),
        row['observation_input'],H=row['H'].to(device),K=row['K'].to(device),
        per_quantile_sigma=None if sigma is None else sigma.to(device))
    require(o.shape==(512,3) and torch.isfinite(o).all(), 'nonfinite_raw_O')
    return dict(x=x,o=o,xcenter=x.median(0).values,ocenter=o.median(0).values,scale=scale)


def instrument(model, raw, kind, xcenter=None, ocenter=None):
    """没有label/candidate ID参数；scale始终取该role保存值。"""
    xc = raw['xcenter'] if xcenter is None else xcenter
    oc = raw['ocenter'] if ocenter is None else ocenter
    zx = (raw['x']-xc)/raw['scale']
    zo = raw['o']-oc
    x, o = zx.clamp(-20,20), zo.clamp(-20,20)
    logits = model.network(torch.cat((x,o),1))[:,kind]
    delta = 5*torch.tanh(logits/5)
    require(torch.isfinite(logits).all() and torch.isfinite(delta).all(), 'nonfinite_instrumentation')
    return dict(zx_raw=zx,zx=x,zo_raw=zo,zo=o,logits=logits,delta=delta,
                xcenter=xc,ocenter=oc)


def model_snapshot(models):
    return value_sha([(m.state_dict(),scale,saved) for m,scale,saved in models])


@torch.inference_mode()
def run_probes(models, single, dual, device):
    require(single['kind']==dual['kind'], 'scoring_kind_mismatch')
    before = value_sha((single,dual)); state_before = model_snapshot(models)
    rows = dict(S=single,D=dual)
    raw = {pool:[raw_trace(m,row,scale,device) for m,scale,_ in models] for pool,row in rows.items()}
    obs = single['query_observation']
    gamma = .75 if obs.get('windows') or obs.get('partial_peaks') else .25
    plans = dict(S_native=('S','S','S'),D_native=('D','D','D'),
        D_use_S_Xcenter=('D','S','D'),D_use_S_Ocenter=('D','D','S'),
        D_use_both_Scenters=('D','S','S'),S_use_both_Dcenters=('S','D','D'))
    outputs = {}; native_errors=[]
    for name,(pool,xpool,opool) in plans.items():
        traces=[]
        for role,(model,scale,saved) in enumerate(models):
            trace=instrument(model,raw[pool][role],rows[pool]['kind'],
                raw[xpool][role]['xcenter'],raw[opool][role]['ocenter'])
            traces.append(trace)
            if name in ('S_native','D_native'):
                native = model.row_delta(rows[pool],scale,device)
                require(torch.allclose(native,trace['delta'],rtol=RTOL,atol=ATOL),
                        'native_instrumentation_mismatch:'+name+':'+str(role))
                native_errors.append(float((native-trace['delta']).abs().max()))
        correction=[saved['alpha']*trace['delta']*scale[0] for trace,(_,scale,saved) in zip(traces,models)]
        cost=rows[pool]['coarse_scores'].to(device)+(1-gamma)*correction[0]+gamma*correction[1]
        require(torch.isfinite(cost).all(), 'nonfinite_final_cost')
        outputs[name]=dict(pool=pool,traces=traces,cost=cost)
    require(before==value_sha((single,dual)), 'raw_scoring_input_changed')
    require(state_before==model_snapshot(models), 'model_scale_or_saved_state_changed')
    return outputs,raw,dict(gamma=gamma,native_delta_max_abs_errors=native_errors,
        instrumentation_rtol=RTOL,instrumentation_atol=ATOL,raw_unchanged=True,model_scale_state_unchanged=True)


def diff_stats(a,b):
    a,b=a.detach().cpu().double(),b.detach().cpu().double()
    if a.ndim==1:a=a[:,None]
    if b.ndim==1:b=b[:,None]
    require(a.shape==b.shape, 'difference_shape_mismatch')
    if not len(a):
        return dict(count=0,mean_signed=None,mean_abs=None,max_abs=None,exact_equal=True)
    diff=b-a
    return dict(count=len(a),mean_signed=diff.mean(0).tolist(),mean_abs=diff.abs().mean(0).tolist(),
        max_abs=diff.abs().max(0).values.tolist(),exact_equal=bool(torch.equal(a,b)))


def pair_changes(left,right):
    left,right=np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)
    require(left.ndim==1 and left.shape==right.shape and np.isfinite(left).all() and np.isfinite(right).all(), 'invalid_pair_scores')
    i,j=np.triu_indices(len(left),1)
    a,b=left[i]-left[j],right[i]-right[j]
    sa,sb=np.where(a>EPS,1,np.where(a < -EPS,-1,0)),np.where(b>EPS,1,np.where(b < -EPS,-1,0))
    pairs=len(a); both_strict=int(((sa!=0)&(sb!=0)).sum())
    reverse=int((sa*sb==-1).sum())
    to_tie=int(((sa!=0)&(sb==0)).sum());from_tie=int(((sa==0)&(sb!=0)).sum())
    return dict(candidates=len(left),pairs=pairs,both_strict_pairs=both_strict,strict_reversals=reverse,
        strict_reversal_rate=reverse/pairs if pairs else None,
        strict_reversal_rate_among_both_strict=reverse/both_strict if both_strict else None,
        strict_to_tie=to_tie,tie_to_strict=from_tie,tie_transitions=to_tie+from_tie,
        tie_transition_rate=(to_tie+from_tie)/pairs if pairs else None,
        tie_both=int(((sa==0)&(sb==0)).sum()),epsilon=EPS)


def target_id_after_scoring(row):
    position=row['target_position']
    require(type(position) is int and 0<=position<len(row['candidate_indices']), 'training_target_missing')
    return int(row['candidate_indices'][position])


def rank_in_pool(indices,cost,target,outside):
    positions=(indices==target).nonzero().reshape(-1)
    if len(positions):
        require(len(positions)==1,'duplicate_target')
        rank=int((cost<=cost[int(positions[0])]+EPS).sum())
        return dict(in_pool=True,rank=rank,RR=1/rank)
    require(type(outside) is int and 512<outside<=114982,'invalid_outside_pool_rank')
    return dict(in_pool=False,rank=outside,RR=1/outside)


def rr_decomposition(single,dual,both):
    result=dict(total=dual-single,normalization=dual-both,composition=both-single)
    require(abs(result['total']-result['normalization']-result['composition'])<=1e-12, 'RR_decomposition_identity')
    return result


def competition(indices,cost,target,common):
    pos=(indices==target).nonzero().reshape(-1)
    require(len(pos)==1,'competition_requires_common_target')
    threshold=cost[int(pos[0])]
    result={}
    for label,ids in (('common',[i for i in common if i!=target]),('unique_to_this_pool',[int(i) for i in indices.tolist() if int(i) not in common])):
        lookup={int(v):n for n,v in enumerate(indices.tolist())}
        values=cost[torch.as_tensor([lookup[i] for i in ids],device=cost.device,dtype=torch.long)]
        strict=int((values<threshold-EPS).sum());ties=int(((values-threshold).abs()<=EPS).sum())
        result[label]=dict(candidates=len(ids),strict_outranking=strict,tied_with_target=ties,
                           conservative_outranking=strict+ties)
    return result


def trace_comparison(left,right,li,ri):
    result={}
    for role in range(2):
        a,b=left['traces'][role],right['traces'][role]
        report={field:diff_stats(a[field][li],b[field][ri])
                for field in ('zx_raw','zx','zo_raw','zo','logits','delta')}
        for field in ('zx','zo'):
            report[field+'_clamp_counts_left']=((a[field+'_raw'][li].abs()>20).sum(0)).cpu().tolist()
            report[field+'_clamp_counts_right']=((b[field+'_raw'][ri].abs()>20).sum(0)).cpu().tolist()
        result['no_aug' if role==0 else 'aug']=report
    result['final_cost']=diff_stats(left['cost'][li],right['cost'][ri])
    return result


def query_report(seed,key,srow,drow,outputs,raw,checks):
    """本函数是评分后才读取target的位置；标签只用于结果度量。"""
    target=target_id_after_scoring(srow)
    require(target==target_id_after_scoring(drow),'target_identity_mismatch')
    indices={'S':srow['candidate_indices'][:512].cpu(),'D':drow['candidate_indices'][:512].cpu()}
    lookup={p:{int(i):n for n,i in enumerate(v.tolist())} for p,v in indices.items()}
    common=sorted(lookup['S'].keys()&lookup['D'].keys())
    positions={p:torch.as_tensor([lookup[p][i] for i in common],device=outputs['S_native']['cost'].device,dtype=torch.long) for p in ('S','D')}
    ranks={}
    for name,output in outputs.items():
        p=output['pool'];row=srow if p=='S' else drow
        ranks[name]=rank_in_pool(indices[p].to(output['cost'].device),output['cost'],target,row['outside_pool_rank'])
        if target in common:
            vals=output['cost'][positions[p]]
            common_rank=int((vals<=vals[common.index(target)]+EPS).sum())
            ranks[name].update(common_only_rank=common_rank,common_only_RR=1/common_rank)
        else:
            ranks[name].update(common_only_rank=None,common_only_RR=None)
    present=(ranks['S_native']['in_pool'],ranks['D_native']['in_pool'])
    category={(True,True):'common',(False,True):'new',(True,False):'lost',(False,False):'neither'}[present]
    comparisons=[('S_native','D_native'),('D_native','D_use_S_Xcenter'),('D_native','D_use_S_Ocenter'),
        ('D_native','D_use_both_Scenters'),('S_native','S_use_both_Dcenters'),('S_native','D_use_both_Scenters')]
    flips,changes={},{}
    for a,b in comparisons:
        left,right=outputs[a],outputs[b];li,ri=positions[left['pool']],positions[right['pool']]
        name=a+'__to__'+b
        flips[name]=dict(common=pair_changes(left['cost'][li].cpu().tolist(),right['cost'][ri].cpu().tolist()))
        changes[name]=dict(common=trace_comparison(left,right,li,ri))
        if left['pool']==right['pool']:
            allpos=torch.arange(512,device=left['cost'].device)
            flips[name]['natural512']=pair_changes(left['cost'].cpu().tolist(),right['cost'].cpu().tolist())
            changes[name]['natural512']=trace_comparison(left,right,allpos,allpos)
    rawdiff={}
    centers={}
    for role in range(2):
        a,b=raw['S'][role],raw['D'][role];label='no_aug' if role==0 else 'aug'
        rawdiff[label]=dict(X33=diff_stats(a['x'][positions['S']],b['x'][positions['D']]),
                            O3=diff_stats(a['o'][positions['S']],b['o'][positions['D']]))
        centers[label]=dict(S_Xcenter=a['xcenter'].cpu().tolist(),D_Xcenter=b['xcenter'].cpu().tolist(),
            D_minus_S_Xcenter=(b['xcenter']-a['xcenter']).cpu().tolist(),
            S_Ocenter=a['ocenter'].cpu().tolist(),D_Ocenter=b['ocenter'].cpu().tolist(),
            D_minus_S_Ocenter=(b['ocenter']-a['ocenter']).cpu().tolist(),
            fixed_scale=a['scale'].cpu().tolist(),
            native_X_clamp_counts_S=(outputs['S_native']['traces'][role]['zx_raw'].abs()>20).sum(0).cpu().tolist(),
            native_X_clamp_counts_D=(outputs['D_native']['traces'][role]['zx_raw'].abs()>20).sum(0).cpu().tolist())
    competitors={}
    if category=='common':
        for name,output in outputs.items():
            p=output['pool']
            competitors[name]=competition(indices[p].to(output['cost'].device),output['cost'],target,set(common))
    decomposition=rr_decomposition(ranks['S_native']['RR'],ranks['D_native']['RR'],ranks['D_use_both_Scenters']['RR'])
    common_decomposition=None
    if category=='common':
        common_decomposition=rr_decomposition(ranks['S_native']['common_only_RR'],
            ranks['D_native']['common_only_RR'],ranks['D_use_both_Scenters']['common_only_RR'])
    return dict(seed=seed,key=list(key),query_order=key[0],source=key[1],record_id=key[2],
        smiles=key[3],group=key[4],variant=key[5],mask_content_sha256=srow['mask_content_sha256'],
        target_category=category,common_candidates=len(common),added_candidates=512-len(common),
        removed_candidates=512-len(common),ranks=ranks,RR_decomposition=decomposition,
        common_only_RR_decomposition=common_decomposition,raw_common_differences=rawdiff,
        centers=centers,score_changes=changes,pair_changes=flips,common_target_competitors=competitors,
        both_S_centers_common_cost_residual=changes['S_native__to__D_use_both_Scenters']['common']['final_cost'],
        checks=checks)


def aggregate_pairs(parts):
    fields=('pairs','both_strict_pairs','strict_reversals','strict_to_tie','tie_to_strict','tie_transitions','tie_both')
    totals={f:sum(p[f] for p in parts) for f in fields}
    def avg(field):
        values=[p[field] for p in parts if p[field] is not None]
        return sum(values)/len(values) if values else None
    return dict(queries=len(parts),counts=totals,
        query_mean_strict_reversal_rate=avg('strict_reversal_rate'),
        pair_weighted_strict_reversal_rate=totals['strict_reversals']/totals['pairs'] if totals['pairs'] else None,
        query_mean_tie_transition_rate=avg('tie_transition_rate'),
        pair_weighted_tie_transition_rate=totals['tie_transitions']/totals['pairs'] if totals['pairs'] else None,
        query_mean_reversal_rate_among_both_strict=avg('strict_reversal_rate_among_both_strict'),
        pair_weighted_reversal_rate_among_both_strict=totals['strict_reversals']/totals['both_strict_pairs'] if totals['both_strict_pairs'] else None)


def aggregate_differences(stats):
    """按共同候选数聚合逐query差异；逐维max保持全样本最坏差异。"""
    valid=[s for s in stats if s['count']>0]
    if not valid:
        return dict(queries_with_common_candidates=0,common_candidate_observations=0,
                    candidate_weighted_mean_abs=None,candidate_weighted_mean_signed=None,max_abs=None)
    weights=np.asarray([s['count'] for s in valid],dtype=np.float64)
    means=np.asarray([s['mean_abs'] for s in valid],dtype=np.float64)
    signed=np.asarray([s['mean_signed'] for s in valid],dtype=np.float64)
    maxima=np.asarray([s['max_abs'] for s in valid],dtype=np.float64)
    require(means.ndim==2 and means.shape==signed.shape==maxima.shape
            and np.isfinite(means).all() and np.isfinite(signed).all() and np.isfinite(maxima).all(),
            'invalid_aggregate_difference_dimensions')
    return dict(queries_with_common_candidates=len(valid),common_candidate_observations=int(weights.sum()),
        candidate_weighted_mean_abs=((means*weights[:,None]).sum(0)/weights.sum()).tolist(),
        candidate_weighted_mean_signed=((signed*weights[:,None]).sum(0)/weights.sum()).tolist(),
        max_abs=maxima.max(0).tolist())


def aggregate_center_differences(part,role,field,normalize=False):
    values=[];weights=[]
    for row in part:
        if row['common_candidates']==0:continue
        center=row['centers'][role]
        delta=np.asarray(center[field],dtype=np.float64)
        if normalize:
            scale=np.asarray(center['fixed_scale'],dtype=np.float64)
            require(scale.shape==delta.shape and np.isfinite(scale).all() and (scale>0).all(),
                    'invalid_aggregate_fixed_scale')
            delta=delta/scale
        require(np.isfinite(delta).all(),'nonfinite_aggregate_center')
        values.append(np.abs(delta));weights.append(row['common_candidates'])
    if not values:
        return dict(queries_with_common_candidates=0,common_candidate_observations=0,
                    common_candidate_weighted_mean_abs=None,query_mean_abs=None,max_abs=None)
    array=np.stack(values);weights=np.asarray(weights,dtype=np.float64)
    return dict(queries_with_common_candidates=len(values),common_candidate_observations=int(weights.sum()),
        common_candidate_weighted_mean_abs=((array*weights[:,None]).sum(0)/weights.sum()).tolist(),
        query_mean_abs=array.mean(0).tolist(),max_abs=array.max(0).tolist())


def aggregate_common_diagnostics(part):
    raw={role:{field:aggregate_differences([r['raw_common_differences'][role][field] for r in part])
               for field in ('X33','O3')} for role in ('no_aug','aug')}
    centers={role:dict(X33=aggregate_center_differences(part,role,'D_minus_S_Xcenter'),
        X33_div_fixed_scale=aggregate_center_differences(part,role,'D_minus_S_Xcenter',True),
        O3=aggregate_center_differences(part,role,'D_minus_S_Ocenter')) for role in ('no_aug','aug')}
    scores={}
    for comparison in part[0]['score_changes']:
        result={}
        for role in ('no_aug','aug'):
            result[role]={field:aggregate_differences([r['score_changes'][comparison]['common'][role][field] for r in part])
                          for field in ('zx_raw','zx','zo_raw','zo','logits','delta')}
            for field in ('zx_clamp_counts_left','zx_clamp_counts_right','zo_clamp_counts_left','zo_clamp_counts_right'):
                values=[r['score_changes'][comparison]['common'][role][field] for r in part if r['common_candidates']>0]
                result[role][field+'_sum']=np.asarray(values,dtype=np.int64).sum(0).tolist() if values else None
        result['final_cost']=aggregate_differences([r['score_changes'][comparison]['common']['final_cost'] for r in part])
        scores[comparison]=result
    return dict(common_feature_differences=raw,center_differences=centers,common_score_differences=scores,
                aggregation='Per-feature candidate-weighted mean absolute/signed differences, per-feature global max absolute; centers also show query mean. Zero common-candidate queries skipped.')


def aggregate(rows):
    def report(part):
        if not part:return dict(queries=0)
        return dict(queries=len(part),
            mean_RR={p:sum(r['ranks'][p]['RR'] for r in part)/len(part) for p in PROBES},
            **aggregate_common_diagnostics(part),
            mean_RR_decomposition={k:sum(r['RR_decomposition'][k] for r in part)/len(part)
                                   for k in ('total','normalization','composition')},
            pair_changes={name:{scope:aggregate_pairs([r['pair_changes'][name][scope] for r in part])
                for scope in part[0]['pair_changes'][name]} for name in part[0]['pair_changes']},
            max_native_delta_instrumentation_error=max(max(r['checks']['native_delta_max_abs_errors']) for r in part),
            all_raw_common_X33_exact=all(r['raw_common_differences']['no_aug']['X33']['exact_equal'] for r in part),
            all_raw_common_O3_exact_by_role={role:all(r['raw_common_differences'][role]['O3']['exact_equal'] for r in part) for role in ('no_aug','aug')})
    return dict(overall=report(rows),
        by_seed={str(s):report([r for r in rows if r['seed']==s]) for s in SEEDS},
        by_variant={v:report([r for r in rows if r['variant']==v]) for v in VARIANTS},
        by_source={s:report([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})},
        by_target_category={c:report([r for r in rows if r['target_category']==c]) for c in ('common','new','lost','neither')},
        by_seed_variant={str(s):{v:report([r for r in rows if r['seed']==s and r['variant']==v]) for v in VARIANTS} for s in SEEDS})


def run(args,out):
    source=Path(args.source).resolve();root=Path(args.root).resolve()
    old=root.parent/'nmr_global_retrieval_20261022'
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path)
        if key not in pins:pins[key]=sha(path)
        require(expected is None or pins[key]==expected,'input_pin_mismatch:'+key)
        return pins[key]
    pin(__file__);pin(args.contract)
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for p in sorted((source/'nmr_transfer').glob('*.py')):pin(p)
    sys.path.insert(0,str(source))
    from nmr_transfer.h_research_ranker import read_feature_cache
    from nmr_transfer.h_pool_ranker import load_checkpoint,check_science,validate_pair,validate_caches
    from nmr_transfer.h_global_features import FEATURE_NAMES
    from nmr_transfer.h_research_observation import O_FEATURE_NAMES
    caches,indexes,models_by_seed={}, {}, {}
    for seed in SEEDS:
        for pool in ('single','dual'):
            path=old/f'features/inherited_s{seed}/{pool}/train/features.pt'
            for name in ('summary.json','EXIT_CODE'):pin(path.parent/name)
            digest=pin(path)
            data,_=read_feature_cache(path,known_sha=digest)
            manifest=data['manifest']
            require(manifest['cohort']=='train' and manifest['recall']==pool,'train_cache_only')
            require(manifest['candidate_order_sha256']==CANDIDATE_SHA
                    and manifest['candidate_count']==114982 and manifest['pool_size']==512,'candidate_universe_changed')
            caches[(seed,pool)]=data;indexes[(seed,pool)]=index_rows(data)
        validate_caches(caches[(seed,'single')],caches[(seed,'dual')])
        a,b=caches[(seed,'single')]['manifest'],caches[(seed,'dual')]['manifest']
        for field in ('schema','query_sha256','candidate_order_sha256','checkpoint_sha256','source_inputs_sha256','identity_content_sha256','variant_mask_content_sha256'):
            require(a[field]==b[field],'within_seed_pool_manifest_mismatch:'+field)
        models=[]
        for aug in (False,True):
            role='aug' if aug else 'no_aug'
            path=old/f'rankers/factor_R0O1F0/s{seed}/{role}/best.pt'
            for name in ('summary.json','EXIT_CODE'):pin(path.parent/name)
            require((path.parent/'EXIT_CODE').read_text().strip()=='0','checkpoint_producer_not_exit0')
            digest=pin(path);model,scale,saved=load_checkpoint(path,torch.device(args.device))
            producer=json.loads((path.parent/'summary.json').read_text())
            require(producer['best']['epoch']==saved['epoch'] and producer['best']['alpha']==saved['alpha'],
                    'checkpoint_producer_best_mismatch')
            require(producer['training_cache_sha256']==saved['train_sha256'], 'checkpoint_producer_training_cache_mismatch')
            for field in ('best_checkpoint_sha256','checkpoint_sha256'):
                if field in producer:require(producer[field]==digest,'checkpoint_producer_byte_SHA_mismatch')
            require(saved['schema']=='nmr_research_ranker_20261022' and saved['observation'] is True,'frozen_R0O1F0_only')
            require(saved['args']['seed']==seed and saved['args']['augment'] is aug,'checkpoint_role_mismatch')
            require(saved['cache_protocol']['recall']=='single' and saved['train_sha256']==pins[str(old/f'features/inherited_s{seed}/single/train/features.pt')],'checkpoint_training_identity')
            require(scale.shape==(33,) and torch.isfinite(scale).all() and bool((scale>0).all()),'nonfinite_scale')
            require(math.isfinite(saved['alpha']) and all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),'nonfinite_model')
            for p,h in saved['ranker_source_sha256'].items():pin(p,h)
            for parameter in model.parameters():parameter.requires_grad_(False)
            model.eval()
            for pool in ('single','dual'):check_science(saved,caches[(seed,pool)])
            models.append((model,scale,saved))
        validate_pair(models,caches[(seed,'single')],seed)
        models_by_seed[seed]=models
    reference=caches[(193,'single')]['manifest']
    for data in caches.values():
        for field in ('query_sha256','candidate_order_sha256'):
            require(data['manifest'][field]==reference[field],'cross_seed_manifest_identity:'+field)
    selection=selection_manifest(indexes)
    (out/'protocol.json').write_text(json.dumps(dict(schema=SCHEMA,args=vars(args),argv=sys.argv,pid=os.getpid(),
        started_utc=started_utc,input_sha256=dict(pins),groups=GROUPS,training=False,search=False),indent=2)+'\n')
    selection.update(input_sha256=dict(pins),fixed_seeds=list(SEEDS),feature_names=FEATURE_NAMES,
                     observation_feature_names=list(O_FEATURE_NAMES),epsilon=EPS)
    (out/'selection.json').write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n')
    selection_sha=pin(out/'selection.json')
    initial_model_hash={str(s):model_snapshot(models_by_seed[s]) for s in SEEDS}
    reports=[];started=time.monotonic()
    with (out/'query.jsonl').open('x') as stream:
        for seed in SEEDS:
            models=models_by_seed[seed]
            for entry in selection['entries']:
                key=tuple(entry['key']);sr=indexes[(seed,'single')][key];dr=indexes[(seed,'dual')][key]
                before=value_sha((sr,dr))
                ns,nd=natural_row(sr),natural_row(dr)
                outputs,raw,checks=run_probes(models,ns,nd,torch.device(args.device))
                report=query_report(seed,key,sr,dr,outputs,raw,checks)
                require(before==value_sha((sr,dr)),'original_cached_row_mutated')
                report['checks']['original_rows_unchanged']=True
                reports.append(report)
                stream.write(json.dumps(report,allow_nan=False)+'\n');stream.flush()
            print(json.dumps(dict(seed=seed,completed_queries=len(selection['entries']),elapsed_seconds=time.monotonic()-started)),flush=True)
    final_model_hash={str(s):model_snapshot(models_by_seed[s]) for s in SEEDS}
    require(initial_model_hash==final_model_hash,'final_frozen_model_or_scale_changed')
    after={p:sha(p) for p in pins};require(pins==after,'source_or_input_changed_during_diagnostic')
    result=dict(schema=SCHEMA,status='PASS',scope='fixed_train_subset_only_no_test_no_training_no_search',
        seeds=list(SEEDS),groups=GROUPS,queries=len(reports),selection_sha256=selection_sha,
        argv=sys.argv,pid=os.getpid(),started_utc=started_utc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selection_path=str(out/'selection.json'),epsilon=EPS,probes=list(PROBES),summary=aggregate(reports),
        input_sha256=pins,inputs_unchanged=True,initial_model_state_sha256=initial_model_hash,
        final_model_state_sha256=final_model_hash,source_files_unchanged=True,elapsed_seconds=time.monotonic()-started,
        feature_names=FEATURE_NAMES,observation_feature_names=list(O_FEATURE_NAMES),external_api_calls=0,
        interpretation=dict(norm='D native RR minus D with both S centers RR; fixed D candidates/raw/features/weights/scale',
            composition='D with both S centers RR minus S native RR; includes candidate composition and measured common raw/FP residuals',
            raw_common='Exact equality is measured, never assumed; raw X33/O3 and both-S-centers final-cost residual reported per query',
            ranks='Conservative epsilon1e-7 counts; out-of-pool target uses cached outside_pool_rank; appended training positives are never scored',
            pair_weighting='query mean rates and total events divided by total eligible pairs both reported',
            missing_variants='Only existing fixed-base-query variants used; absent in every seed/pool is explicit skip, inconsistent missing fails'))
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--contract',required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4)
        result=run(args,out)
        (out/'EXIT_CODE').write_text('0\n')
        print(json.dumps(dict(status=result['status'],groups=result['groups'],queries=result['queries'])),flush=True)
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(status='FAIL',error=repr(exc)))+'\n')
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__=='__main__':
    main()

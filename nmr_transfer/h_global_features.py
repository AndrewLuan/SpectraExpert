"""功能概述：提取不依赖真结构的候选谱匹配特征，供训练集难负样本校准。

observed_support 从实际查询读位置/积分，HMDB 仅用等权峰簇，峰高不当积分；缺失窗
不补全。candidate_features 仅输入候选结构核数、预测谱/类别和查询，返回 KDE、
稳健距离、公共位移、报告数量、支持重叠等特征。反向重叠是待训练校准的证据，
不直接声明未知观测为完整谱或把物理环境数等同观测簇数。窗口外支持从特征中隔离。
公共偏移在候选各自的似然中积分，不由正确结构求 oracle；污染成分对所有候选共享。
所有输出为有限 float32 张量，不读取 ID/SMILES/母节点真归属，不访问文件或网络。
调用方负责 train/val 边界、候选短名单、特征缓存和冻结后的独立核验。
"""
import math

import numpy as np
import torch

from .h_observation_finalize import visible_lines
from .h_observed import cluster_lines, count_nll
from .h_observation_study import uniform_peaks


FEATURE_NAMES = [
    'coarse_score', 'kde_008', 'kde_020', 'kde_040', 'kde_080',
    'contamination_003', 'contamination_010', 'shift_010', 'shift_020',
    'observed_nearest_mean', 'observed_nearest_sq', 'observed_nearest_max',
    'predicted_nearest_mean', 'predicted_nearest_capped',
    'overlap_010', 'overlap_020', 'overlap_040',
    'center_difference', 'spread_difference', 'quantile_difference',
    'log_candidate_H', 'log_report_ratio', 'report_count_nll',
    'log_supports_003_ratio', 'log_supports_008_ratio', 'log_supports_016_ratio',
    'exchangeable_fraction', 'aromatic_fraction', 'aldehyde_fraction',
    'predicted_span', 'observed_span', 'log_observed_positions',
    'empty_window_support',
]


def observed_support(query, raw):
    if query['source']=='HMDB':
        xs,_=cluster_lines(visible_lines(query,raw),.08)
        weights=np.ones(len(xs),np.float32); kind=0; absolute_count=False
    elif query.get('observed_xs'):
        xs=np.asarray(query['observed_xs'],np.float32)
        weights=np.asarray(query['observed_weights'],np.float32)
        kind=1 if query.get('measure')=='integral_weighted' else 2
        absolute_count=bool(np.allclose(weights,weights.round(),atol=.001))
    else:
        xs,weights=np.unique(np.asarray(query['q'],np.float32),return_counts=True)
        weights=weights.astype(np.float32); kind=3; absolute_count=False
    if len(xs)==0 or weights.sum()<=0:raise ValueError('global_empty_query_support')
    unique,inverse=np.unique(xs,return_inverse=True)
    weights=np.bincount(inverse,weights=weights,minlength=len(unique)).astype(np.float32)
    xs=unique
    return dict(xs=xs,weights=weights,kind=kind,absolute_count=absolute_count,
                reported=float(weights.sum()), positions=len(xs))


def _weighted_quantiles(xs, weights, points=128):
    order=torch.argsort(xs); xs=xs[order]; weights=weights[order]
    cdf=weights.cumsum(0)/weights.sum()
    q=(torch.arange(points,device=xs.device,dtype=xs.dtype)+.5)/points
    return xs[torch.searchsorted(cdf.contiguous(),q).clamp_max(len(xs)-1)]


@torch.inference_mode()
def candidate_features(predictions,tags,counts,coarse,query,prior,raw):
    device=predictions.device; support=observed_support(query,raw)
    x=torch.as_tensor(support['xs'],device=device)
    w=torch.as_tensor(support['weights'],device=device); w=w/w.sum()
    p=predictions*10
    if support['kind']==0:
        distinct=uniform_peaks(predictions)*10
        nearest=(distinct[:,:,None]-p[:,None,:]).abs().argmin(2)
        tags=tags.gather(1,nearest); p=distinct
    active=torch.ones_like(p,dtype=torch.bool)
    windows=query['observation']['windows']
    if windows:
        active.zero_()
        for a,b in windows:active|=(p>=a)&(p<=b)
    mass=active.float(); total=mass.sum(1,keepdim=True).clamp_min(1)
    has_support=active.any(1)
    logmass=torch.where(active,-total.log().expand_as(p),torch.full_like(p,-torch.inf))
    residual=x[None,:,None]-p[:,None,:]
    r2=residual.square(); normalizer=math.sqrt(2*math.pi)
    def point_logp(sigma,shift=0.):
        square=r2 if shift==0 else (residual+shift).square()
        return torch.logsumexp(logmass[:,None,:]-.5*square/sigma**2,2)-math.log(sigma*normalizer)
    terms={}
    for sigma in (.08,.2,.4,.8):terms[sigma]=point_logp(sigma)
    mean=lambda value:(value*w[None,:]).sum(1)
    kde={s:-mean(v) for s,v in terms.items()}
    robust={e:-mean(torch.logaddexp(terms[.2]+math.log1p(-e),torch.full_like(terms[.2],math.log(e/40)))) for e in (.03,.1)}
    shifts=torch.tensor([-.4,-.2,-.1,0.,.1,.2,.4],device=device)
    shifted=torch.stack([kde[.2] if float(s)==0 else -mean(point_logp(.2,float(s))) for s in shifts])
    integrated={}
    for sigma in (.1,.2):
        lp=-.5*(shifts/sigma).square(); lp=lp-torch.logsumexp(lp,0)
        effective=max(1,support['positions'])
        integrated[sigma]=-torch.logsumexp(-effective*shifted+lp[:,None],0)/effective
    d=residual.abs().masked_fill(~active[:,None,:],float('inf'))
    observed=d.min(2).values.clamp_max(40.)
    reverse=residual.abs().min(1).values
    reverse_mean=lambda value:(value*mass).sum(1)/total[:,0]
    ps=(p*mass).sum(1)/total[:,0]; os=(x*w).sum()
    pv=(((p-ps[:,None]).square()*mass).sum(1)/total[:,0]).sqrt()
    ov=((x-os).square()*w).sum().sqrt()
    points=p.shape[1]
    observed_q=_weighted_quantiles(x,w,points)
    if windows:
        ordered,_=p.masked_fill(~active,1e6).sort(1)
        target=((torch.arange(points,device=device)+.5)[None,:]/points*total).long().clamp_max(points-1)
        predicted_q=ordered.gather(1,target)
    else:predicted_q=p
    report=support['reported']
    visible_counts=counts*(total[:,0]/p.shape[1]) if windows else counts
    ratios=torch.log(visible_counts.clamp_min(.5)/max(.5,report)) if support['absolute_count'] else torch.zeros_like(counts)
    count=torch.zeros_like(counts)
    entry=prior.get(query['source'])
    if entry is not None and support['absolute_count'] and not windows:
        count=count_nll(counts,int(round(report)),entry)
    retained=p.masked_fill(~active,-torch.inf)
    previous=torch.cat([torch.full_like(p[:,:1],-torch.inf),retained[:,:-1].cummax(1).values],1)
    support_ratios=[]
    for tolerance in (.03,.08,.16):
        starts=(((p-previous)>tolerance)&active).sum(1)
        support_ratios.append(torch.log(starts.clamp_min(.5)/support['positions']))
    span=p.masked_fill(~active,-1e6).max(1).values-p.masked_fill(~active,1e6).min(1).values
    span=torch.where(has_support,span,torch.full_like(span,40.))
    reversed_mean=torch.where(has_support,reverse_mean(reverse),torch.full_like(counts,40.))
    reversed_capped=torch.where(has_support,reverse_mean(reverse.clamp_max(.5)),torch.full_like(counts,.5))
    cols=[coarse,kde[.08],kde[.2],kde[.4],kde[.8],robust[.03],robust[.1],integrated[.1],integrated[.2],
        mean(observed),mean(observed.square()),observed.max(1).values,
        reversed_mean,reversed_capped,
        *[reverse_mean((reverse<t).float()) for t in (.1,.2,.4)],
        (ps-os).abs(),(pv-ov).abs(),(predicted_q-observed_q[None,:]).abs().mean(1),
        counts.clamp_min(.5).log(),ratios,count,*support_ratios,
        reverse_mean(((tags>=5)&(tags<=7)).float()),reverse_mean((tags==3).float()),reverse_mean((tags==4).float()),
        span,torch.full_like(counts,float(x.max()-x.min())),torch.full_like(counts,math.log(support['positions'])),
        (~has_support).float()]
    result=torch.stack(cols,1)
    if result.shape[1]!=len(FEATURE_NAMES):raise ValueError('global_feature_schema_mismatch')
    result=torch.nan_to_num(result,nan=40.,posinf=40.,neginf=-40.).clamp(-40.,40.)
    result[:,0]=torch.nan_to_num(coarse,nan=1e6,posinf=1e6,neginf=-1e6)
    return result.float(),support

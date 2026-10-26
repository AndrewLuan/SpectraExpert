"""功能概述：对无积分HMDB峰表按“可分辨预测环境”做部分匹配和报告计数建模。
environment_points将冻结分位预测按最大0.08ppm簇宽合并，每个簇一单位；重复的
核质量分位点不增加环境权重。结果是模型的可分辨位置数，不宣称为精确化学环境数。
fit_count_mixture只用train已报告簇数与模型簇数拟合beta-binomial/Poisson混合，
normal/dropout分开，峰高不作H积分。environment_features允许未报告预测簇被跳过，
公共误差偏移在预声明宽网格上边缘化；计数项防止无代价偏爱超多环境的候选。
水相比较全H/非交换H/两者潜变量混合；非交换支持为空时回退到全H，不能让实际
观测与空谱比较。显式频窗继续原评分，不能从首末峰反推实测区间。
soft_partial_match对全部有序无放回子集求和，再除以组合数，避免只选最佳子集使
环境多的候选获得额外优势；观测数超过预测簇数时用有放回位置密度回退，明确
保留报告计数代价。它是有序匹配近似，不宣称为完整多重峰生成模型。
全部候选只使用自身图预测，查询只给已报告位置/状态。无数据写入或外部调用；
workflow负责训练/验证边界、源码和逐查询证据。
"""
import math
import numpy as np
import torch
from scipy.special import logsumexp
from .h_observed import cluster_lines,count_components,count_nll,DROPOUT_RATES,marginal_shift
from .h_score import partial_transport

SHIFTS=[-.8,-.6,-.4,-.2,0.,.2,.4,.6,.8]
GRID=[dict(kind='baseline')]+[dict(kind='environment',visibility=v,shift_sigma=s,beta=b) for v in ['fixed','full','mixed'] for s in [0.,.2,.5] for b in [0.,.25,1.,4.]]
SOFT_GRID=[GRID[0]]+[dict(cfg,matching='soft') for cfg in GRID[1:]]


@torch.inference_mode()
def environment_points(predictions,width=.08):
    p=predictions.sort(1).values*10;rows,points=p.shape;valid=torch.isfinite(p)&(p<1e5);unique=valid.clone();unique[:,1:]&=(p[:,1:]-p[:,:-1])>1e-5
    group=torch.zeros(rows,points,dtype=torch.long,device=p.device);current=torch.zeros(rows,dtype=torch.long,device=p.device);start=p[:,0]
    for j in range(1,points):
        split=unique[:,j]&((p[:,j]-start)>width+1e-8);current=current+split.long();start=torch.where(split,p[:,j],start);group[:,j]=current
    positions=torch.arange(points,device=p.device)[None,:].expand(rows,-1);begin=torch.full_like(group,points).scatter_reduce_(1,group,positions,reduce='amin',include_self=True);end=torch.full_like(group,-1).scatter_reduce_(1,group,positions,reduce='amax',include_self=True)
    prefix=torch.cat([torch.zeros(rows,1,device=p.device),p.masked_fill(~unique,0.).cumsum(1)],1);count_prefix=torch.cat([torch.zeros(rows,1,device=p.device),unique.float().cumsum(1)],1);aa=begin.clamp(0,points);bb=(end+1).clamp(0,points);sums=prefix.gather(1,bb)-prefix.gather(1,aa);sizes=count_prefix.gather(1,bb)-count_prefix.gather(1,aa);mask=sizes>0;values=sums/sizes.clamp_min(1);n=mask.sum(1);maximum=max(1,int(n.max()));return values[:,:maximum],mask[:,:maximum]


def visible_fallback(full,fixed):
    empty=~(torch.isfinite(fixed)&(fixed<1e4)).any(1);return torch.where(empty[:,None],full,fixed),empty


def fit_count_mixture(pairs):
    if not pairs:raise ValueError('empty_training_reporting_pairs')
    n,k=np.asarray(pairs,float).T
    if np.any(n<=0) or np.any(k<=0) or not np.allclose(n,n.round()) or not np.allclose(k,k.round()):raise ValueError('invalid_environment_counts')
    ll=count_components(n,k,DROPOUT_RATES,concentration=2.);prior=np.ones(len(DROPOUT_RATES))/len(DROPOUT_RATES)
    for _ in range(40):
        logits=ll+np.log(prior);post=np.exp(logits-logsumexp(logits,axis=1,keepdims=True));prior=(post.sum(0)+1)/(len(n)+len(prior))
    return dict(rates=DROPOUT_RATES.tolist(),weights=prior.tolist(),concentration=2.,training_records=len(n),median_reported_fraction=float(np.median(k/n)),measure='reported peak clusters per candidate predicted-resonance cluster, never H integral')


@torch.inference_mode()
def soft_partial_match(atoms,mask,observed,scale=.2):
    observed=observed.sort().values;k=len(observed);n=mask.sum(1).float();previous=torch.zeros(len(atoms),atoms.shape[1]+1,device=atoms.device,dtype=atoms.dtype);fallback=torch.zeros(len(atoms),device=atoms.device,dtype=atoms.dtype)
    for x in observed:
        emission=(-(atoms-x).abs()/scale).masked_fill(~mask,-torch.inf);match=previous[:,:-1]+emission;current=torch.logcumsumexp(match,1);previous=torch.cat([torch.full((len(atoms),1),-torch.inf,device=atoms.device,dtype=atoms.dtype),current],1);fallback+=-torch.logsumexp(emission,1)+n.clamp_min(1).log()
    comb=torch.lgamma(n+1)-math.lgamma(k+1)-torch.lgamma((n-k).clamp_min(0)+1);conditional=(-previous[:,-1]+comb)/k
    return torch.where(n>=k,conditional,fallback/k)


@torch.inference_mode()
def environment_features(full,fixed,query,xs,priors,water,matching='hard'):
    observed,_=cluster_lines(xs,.08);target=torch.tensor(observed,dtype=torch.float32,device=full[0].device);k=len(observed);family='water' if water else 'nonwater';process='dropout' if query['observation'].get('partial_peaks') else 'normal';costs={};counts={}
    for kind,points in [('full',full),('fixed',fixed)]:
        if kind=='fixed' and not water:costs[kind]=costs['full'];counts[kind]=counts['full'];continue
        atoms,mask=points;costs[kind]=torch.stack([soft_partial_match(atoms,mask,target+s) if matching=='soft' else partial_transport(atoms,mask,target+s)/.2 for s in SHIFTS]);counts[kind]=count_nll(mask.sum(1).float(),k,priors[process][family][kind])
    return dict(costs=costs,counts=counts,k=k,water=water)


def environment_score(features,config):
    values={kind:marginal_shift(costs,SHIFTS,config['shift_sigma'],features['k'])+config['beta']*features['counts'][kind] for kind,costs in features['costs'].items()}
    if config['visibility']!='mixed':return values[config['visibility']]
    if not features['water']:return values['full']
    k=features['k'];return -torch.logaddexp(-k*values['fixed']+math.log(.9),-k*values['full']+math.log(.1))/k

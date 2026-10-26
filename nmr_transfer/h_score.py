"""功能概述：¹H缺测感知的单位核质量部分匹配，保留实验积分而允许任意预测H未报告。
spectral_observation/attach_observation只提取已观测位置与质量，不将真实结构/完整性
标志交给评分器。fit_count_prior仅用train的报告H数/结构H数拟合二项遗漏混合，
报告率为1时恰为完整报告；超过1的成分单独用Poisson多报过程，不混同于核遗漏。
unit_emissions从候选的128分位数重建单位H质量；partial_transport用有序注入匹配DP
让每个观测H匹配一个独立预测H，预测H可免费跳过，避免平均峰密度偏爱小分子。
count_score用训练拟合的报告率先验约束无代价丢弃，明确建模少报/多报；已知窗口
只计算窗内预测核数。无积分的HMDB峰表、非整数积分和超长记录回退到原似然。
score_features输出原似然、部分匹配和计数项，select_score由固定val网格选择β；
query_config按输入观测格式和明确的缺测状态选择评分：原子稀疏标注不是积分，
观测路由对它继续用似然；积分谱的正常/已知频段/明确缺峰参数分别由val选择。
不使用真实候选索引、测试位点归属或test拟合偏移。理论/实验缺峰都不补零。
"""
import math
import numpy as np
import torch
from scipy.special import gammaln,logsumexp
from .data import identity
from .likelihood import log_score

RATES=np.array([.1,.25,.5,.75,1.,1.25,1.5,2.])
CONFIGS=[dict(kind='likelihood',beta=0.)]+[dict(kind='partial_transport',beta=b) for b in [0.,.25,1.,4.]]


def spectral_observation(meta):
    if 'xs' in meta and 'weights' in meta:
        return np.asarray(meta['xs'],float),np.asarray(meta['weights'],float)
    if meta.get('groups'):
        xs=np.array([v for vs in meta['groups'].values() for v in vs],float)
        return xs,np.ones_like(xs)
    return None


def attach_observation(query,meta):
    result=dict(query);observed=spectral_observation(meta)
    if observed is not None:result.update(observed_xs=observed[0].tolist(),observed_weights=observed[1].tolist())
    return result


def fit_count_prior(annotations):
    groups={}
    for (domain,source,record,smi),meta in annotations.items():
        if domain!='experimental' or identity(smi)[2]!='train':continue
        observed=spectral_observation(meta)
        if observed is None:continue
        _,weights=observed;n=float(meta['structural_H']);k=float(weights.sum())
        if n<=0 or k<=0 or not np.allclose(weights,weights.round(),atol=.001):continue
        groups.setdefault(source,[]).append((n,k))
    report={}
    for source,pairs in groups.items():
        n,k=np.array(pairs).T;mu=n[:,None]*RATES
        ll=k[:,None]*np.log(mu)-mu-gammaln(k[:,None]+1)
        coefficient=gammaln(n+1)-gammaln(k+1)-gammaln(np.maximum(n-k,0)+1)
        for j,rate in enumerate(RATES):
            if rate<1:ll[:,j]=np.where(k<=n,coefficient+k*np.log(rate)+(n-k)*np.log1p(-rate),-1e6)
            elif rate==1:ll[:,j]=np.where(k==n,0.,-1e6)
        prior=np.ones(len(RATES))/len(RATES)
        for _ in range(20):
            logits=ll+np.log(prior);posterior=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
            prior=(posterior.sum(0)+1)/(len(n)+len(RATES))
        report[source]=dict(model='binomial_dropout_plus_Poisson_overreport',rates=RATES.tolist(),weights=prior.tolist(),training_records=len(n),
                            median_reported_fraction=float(np.median(k/n)))
    return report


def unit_emissions(predictions,counts):
    n=counts.round().long().clamp_min(0);maximum=max(1,int(n.max()))
    position=(torch.arange(maximum,device=n.device)[None,:]+.5)/n[:,None].clamp_min(1)
    indices=(position*predictions.shape[1]).long().clamp(0,predictions.shape[1]-1)
    return predictions.gather(1,indices)*10,torch.arange(maximum,device=n.device)[None,:]<n[:,None]


def partial_transport(atoms,mask,observed):
    observed=torch.sort(observed).values;k=len(observed);n=mask.sum(1)
    previous=torch.zeros(len(atoms),atoms.shape[1]+1,device=atoms.device)
    for x in observed:
        match=(previous[:,:-1]+(atoms-x).abs()).masked_fill(~mask,float('inf'))
        current=torch.cummin(match,dim=1).values
        previous=torch.cat([torch.full((len(atoms),1),float('inf'),device=atoms.device),current],1)
    distance=previous[:,-1]/k
    # 记录多报时显式付出观测污染代价，不伪造额外物理H；总分母保持原查询质量。
    q=((torch.arange(k,device=atoms.device)+.5)/k)[None,:]
    idx=(q*n[:,None]).long().clamp_max(atoms.shape[1]-1)
    fallback=(atoms.gather(1,idx)-observed[None,:]).abs().mean(1)+.5*(k-n).clamp_min(0)/k
    return torch.where(n>=k,distance,fallback)


def count_score(atoms,mask,query,k,prior):
    n=mask.sum(1).float();windows=query['observation']['windows']
    if windows:
        inside=torch.zeros_like(mask)
        for a,b in windows:inside|=(atoms>=a)&(atoms<=b)
        n=(inside&mask).sum(1).float()
    entry=prior.get(query['source'])
    if entry is None:entry=dict(rates=RATES,weights=np.ones(len(RATES))/len(RATES))
    rates=torch.tensor(entry['rates'],device=atoms.device);weights=torch.tensor(entry['weights'],device=atoms.device)
    mu=(n[:,None]*rates).clamp_min(1e-8)
    ll=k*mu.log()-mu-math.lgamma(k+1)
    coefficient=torch.lgamma(n+1)-math.lgamma(k+1)-torch.lgamma((n-k).clamp_min(0)+1)
    for j,rate in enumerate(entry['rates']):
        if rate<1:ll[:,j]=torch.where(n>=k,coefficient+k*math.log(rate)+(n-k)*math.log1p(-rate),torch.full_like(n,-1e6))
        elif rate==1:ll[:,j]=torch.where(n==k,torch.zeros_like(n),torch.full_like(n,-1e6))
    return -torch.logsumexp(ll+weights.log(),1)/k


@torch.inference_mode()
def score_features(predictions,kernel,counts,query,prior):
    base=log_score(kernel,query,'1H');xs=query.get('observed_xs');ws=query.get('observed_weights')
    if xs is None or ws is None:return dict(likelihood=base,transport=base,count=torch.zeros_like(base),eligible=False)
    ws=np.asarray(ws);k=int(round(float(ws.sum())))
    if not 1<=k<=96 or not np.allclose(ws,ws.round(),atol=.001):
        return dict(likelihood=base,transport=base,count=torch.zeros_like(base),eligible=False)
    observed=torch.tensor(np.repeat(np.asarray(xs),ws.round().astype(int)),dtype=torch.float32,device=predictions.device)
    atoms,mask=unit_emissions(predictions,counts)
    distance=partial_transport(atoms,mask,observed)/.2
    return dict(likelihood=base,transport=distance,count=count_score(atoms,mask,query,k,prior),eligible=True)


def select_score(features,config):
    if config['kind']=='likelihood':return features['likelihood']
    return features['transport']+config['beta']*features['count']


def query_config(config,query):
    if config['kind']!='observation_router':return config
    if query.get('measure')!='integral_weighted':return dict(kind='likelihood',beta=0.)
    if query['observation'].get('partial_peaks'):return config['peak_dropout']
    if query['observation']['windows']:return config['frequency_crop']
    return config['full_or_unknown']

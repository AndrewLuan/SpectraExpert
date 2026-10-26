"""功能概述：在固定SpecGNN预测上建模¹H峰表口径、整峰遗漏和未知参考偏移。
cluster_lines只用观测位置，按最大簇宽合并多重峰细线；峰高不转换为H积分。
fit_dropout_prior对train积分表施加与evaluate.missing_queries完全相同的冻结随机删峰，
按候选H数拟合少报/多报混合；保存正常报告过程经整峰删除后的经验分布。
count_nll支持二项/Poisson及可选beta-binomial过散布，核数仅来自候选结构。
shifted_transport对所有候选使用同一有限偏移网格与零均值先验，不读取真实候选索引；
line_score对簇中心使用预测密度，边缘化同一参考偏移。参考偏移仅为观测潜变量，
不能解释为已验证的仪器校正；候选相关非均匀残差仍由原模型表达。
依赖h_score的有序部分匹配和原likelihood；由h_observation_study负责仅val选参及
同库同查询复评。此模块无文件写入、无外部API，也不修改原始缓存或历史结果。
"""
import math
import numpy as np
import torch
from scipy.special import gammaln,logsumexp,betaln
from .data import identity
from .evaluate import missing_queries
from .h_score import unit_emissions,partial_transport
from .likelihood import log_score

DROPOUT_RATES=np.array([.1,.25,.5,.625,.75,.875,1.,1.25,1.5,2.])


def cluster_lines(xs,width):
    """按最大簇宽分组，防止单链接把拥挤的整个谱区链成一个峰；不赋积分。"""
    x=np.unique(np.asarray(xs,dtype=float))
    if not len(x) or not np.isfinite(x).all():raise ValueError('invalid_line_positions')
    groups=[];start=0
    for i in range(1,len(x)):
        if x[i]-x[start]>width+1e-8:groups.append(x[start:i]);start=i
    groups.append(x[start:])
    return np.array([g.mean() for g in groups]),[g.tolist() for g in groups]


def count_components(n,k,rates,concentration=None):
    """CPU拟合的概率质量；concentration有限时明确建模删峰质量相关性。"""
    n=np.asarray(n)[:,None];k=np.asarray(k)[:,None];rates=np.asarray(rates)[None,:]
    mu=n*rates;ll=k*np.log(np.maximum(mu,1e-12))-mu-gammaln(k+1)
    coefficient=gammaln(n+1)-gammaln(k+1)-gammaln(np.maximum(n-k,0)+1)
    for j,rate in enumerate(rates[0]):
        if rate<1:
            if concentration is None:v=coefficient[:,0]+k[:,0]*np.log(rate)+(n[:,0]-k[:,0])*np.log1p(-rate)
            else:v=coefficient[:,0]+betaln(k[:,0]+rate*concentration,n[:,0]-k[:,0]+(1-rate)*concentration)-betaln(rate*concentration,(1-rate)*concentration)
            ll[:,j]=np.where(k[:,0]<=n[:,0],v,-1e6)
        elif rate==1:ll[:,j]=np.where(k[:,0]==n[:,0],0.,-1e6)
    return ll


def fit_dropout_prior(annotations,concentration=None):
    samples={}
    for (domain,source,record,smi),meta in annotations.items():
        if domain!='experimental' or 'xs' not in meta or identity(smi)[2]!='train':continue
        xs=np.asarray(meta['xs']);w=np.asarray(meta['weights']);n=float(meta['structural_H'])
        if n<=0 or not np.allclose(w,w.round(),atol=.001):continue
        # 复用原压力变体的128分位数和哈希抽样，包括至少4分位数的原筛选条件。
        from .data import quantiles
        q=dict(record_id=record,smiles=smi,q=quantiles(xs,w),observed_xs=xs.tolist(),observed_weights=w.tolist(),observation=dict(kind='unknown',windows=[]))
        for variant,r in missing_queries(q,'1H'):
            if variant=='peak_dropout':samples.setdefault(source,[]).append((n,sum(r['observed_weights'])))
    report={}
    for source,pairs in samples.items():
        n,k=np.array(pairs).T;ll=count_components(n,k,DROPOUT_RATES,concentration)
        prior=np.ones(len(DROPOUT_RATES))/len(DROPOUT_RATES)
        for _ in range(40):
            logits=ll+np.log(prior);posterior=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
            prior=(posterior.sum(0)+1)/(len(n)+len(DROPOUT_RATES))
        report[source]=dict(rates=DROPOUT_RATES.tolist(),weights=prior.tolist(),concentration=concentration,
            training_records=len(n),median_reported_fraction=float(np.median(k/n)),model='train_only_whole_peak_dropout_mixture')
    return report


def count_nll(counts,k,entry):
    n=counts.round();mu=(n[:,None]*torch.tensor(entry['rates'],device=n.device)).clamp_min(1e-8)
    ll=k*mu.log()-mu-math.lgamma(k+1);c=entry.get('concentration')
    coeff=torch.lgamma(n+1)-math.lgamma(k+1)-torch.lgamma((n-k).clamp_min(0)+1)
    for j,rate in enumerate(entry['rates']):
        if rate<1:
            if c is None:v=coeff+k*math.log(rate)+(n-k)*math.log1p(-rate)
            else:
                a=rate*c;b=(1-rate)*c
                v=coeff+torch.lgamma(k+torch.full_like(n,a))+torch.lgamma((n-k).clamp_min(0)+b)-torch.lgamma(n+c)-math.lgamma(a)-math.lgamma(b)+math.lgamma(c)
            ll[:,j]=torch.where(n>=k,v,torch.full_like(n,-1e6))
        elif rate==1:ll[:,j]=torch.where(n==k,torch.zeros_like(n),torch.full_like(n,-1e6))
    return -torch.logsumexp(ll+torch.tensor(entry['weights'],device=n.device).log(),1)/k


@torch.inference_mode()
def shifted_transport(predictions,counts,query,shifts):
    ws=np.asarray(query['observed_weights']);k=int(round(float(ws.sum())))
    if not 1<=k<=96 or not np.allclose(ws,ws.round(),atol=.001):return None
    xs=np.repeat(np.asarray(query['observed_xs']),ws.round().astype(int))
    observed=torch.tensor(xs,dtype=torch.float32,device=predictions.device)
    atoms,mask=unit_emissions(predictions,counts)
    return torch.stack([partial_transport(atoms,mask,observed+shift)/.2 for shift in shifts])


def marginal_shift(costs,shifts,sigma,effective=1):
    if sigma==0:return costs[list(shifts).index(0.)]
    shifts=torch.tensor(shifts,device=costs.device);prior=-.5*(shifts/sigma).square();prior=prior-torch.logsumexp(prior,0)
    return -torch.logsumexp(-effective*costs+prior[:,None],0)/effective


@torch.inference_mode()
def line_score(kernel,query,xs,width,shifts,sigma):
    centers,_=cluster_lines(xs,width);rr=dict(query,q=centers)
    costs=torch.stack([log_score(kernel,dict(rr,q=centers+s),'1H') for s in shifts])
    return marginal_shift(costs,shifts,sigma,effective=len(centers))

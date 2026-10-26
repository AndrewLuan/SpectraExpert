"""功能概述：用潜在报告状态建模无积分谱表中未报告的羰基CH峰。
candidate_families只读结构，将纯甲酰胺CH、纯醛CH和混合/无此类候选分开；
counterfactual_quantiles只移除同次发射tag=aldehyde_CH的分位成分，再归一化其余
密度。混合类型或移除后空谱回退原预测，不能生成无支持的虚假高分。
mixture_score对整张谱的“报告/未报告”两状态做log-sum-exp；不会逐峰挑最优状态，
不会读取真实候选身份，也不按首末观测点推定窗口。显式频窗由调用方沿用旧评分。
fit_probability用训练谱两状态似然、连接结构均权和Beta收缩估计报告概率；空支持
回退父层先验。它是报告过程统计量，不能解释为物理消失概率或已测采集频窗。
供h_formyl_workflow使用；不拟合位移、没有DMF专用数值、无文件或网络副作用。
"""
import math
import numpy as np
import torch
from .h_acyl import carbonyl_h_sites

GRID=[dict(kind='baseline')]+[dict(kind='formyl_reporting',strength=s) for s in [.5,1.]]


def candidate_families(smiles):
    result=[]
    for s in smiles:
        tags=set(carbonyl_h_sites(s).values());result.append(next(iter(tags)) if len(tags)==1 else 'other')
    return result


def counterfactual_quantiles(pred,tags,families):
    allowed=torch.tensor([f!='other' for f in families],device=pred.device)
    keep=(tags!=4)|(~allowed[:,None]);empty=~keep.any(1);keep[empty]=True
    # 使用原分位质量重新采样；位置不变，不设任何羰基或甲基位移常数。
    cdf=keep.cumsum(1)/keep.sum(1,keepdim=True);u=((torch.arange(pred.shape[1],device=pred.device)+.5)/pred.shape[1])[None,:].expand_as(pred).contiguous()
    indices=torch.searchsorted(cdf.contiguous(),u).clamp_max(pred.shape[1]-1)
    altered=pred.gather(1,indices);active=allowed&(~empty)&(tags==4).any(1)
    return torch.where(active[:,None],altered,pred),active


def mixture_score(full,missing,probability,effective,strength=1.):
    if strength==0:return full
    original_p=torch.as_tensor(probability,dtype=full.dtype,device=full.device)
    p=original_p.clamp(1e-6,1-1e-6)
    mixed=-torch.logaddexp(-effective*full+p.log(),-effective*missing+torch.log1p(-p))/effective
    return torch.where((full==missing)|(original_p>=1),full,full+strength*(mixed-full))


def fit_probability(samples,prior=.9,strength=20.):
    if not samples:return dict(probability=prior,records=0,connectivities=0,identified=False)
    # 每条样本full_loglike/missing_loglike已含整谱观测数；同结构重复文献均分权重。
    groups={s['group'] for s in samples};counts={g:sum(s['group']==g for s in samples) for g in groups}
    w=np.array([1/counts[s['group']] for s in samples]);a=np.array([s['full_loglike'] for s in samples]);b=np.array([s['missing_loglike'] for s in samples]);p=prior
    for _ in range(100):
        logits=np.clip(a-b+math.log(p)-math.log1p(-p),-700,700);z=1/(1+np.exp(-logits));p=float((np.sum(w*z)+strength*prior)/(w.sum()+strength))
    return dict(probability=p,records=len(samples),connectivities=len(groups),identified=True,prior=prior,prior_strength_connectivities=strength)

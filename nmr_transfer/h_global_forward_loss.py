"""功能概述：为整体检索的 GNN 微调提供可微观测似然与教师保持损失。

observed_nll 在归一化 ¹H 位移上计算 Gaussian KDE，显式窗口按预测质量条件化；
未知覆盖只对预声明全窗/半窗过程边缘化，绝不从已报首末峰推测频窗。窗口低质量
候选付出有限惩罚而不获免费增益。没有完整谱假设或逐未报峰的硬负标签。
pairwise_rank_loss 只比较同一个训练查询的真结构与纯训练结构负例；teacher_anchor
保持完整物理分布，避免缺报损失拖动未观察部分。两项不读取分子名、ID或测试数据。
调用方负责训练/验证分割、观测口径、条件批次和负样本等价屏蔽；所有计算可反传。
"""
import math

import torch
from torch.nn import functional as F


def observed_nll(predicted,observed,windows,window_mask,sigma=.02,complete=None):
    if predicted.ndim!=2 or observed.shape!=predicted.shape:
        raise ValueError('ranking_quantile_shape_mismatch')
    residual=(observed[:,:,None]-predicted[:,None,:])/sigma
    nll=(-torch.logsumexp(-.5*residual.square(),2)+math.log(predicted.shape[1])+math.log(sigma*math.sqrt(2*math.pi))).mean(1)
    def mass(bounds):
        lo=(bounds[:,0,None]-predicted)/(sigma*math.sqrt(2))
        hi=(bounds[:,1,None]-predicted)/(sigma*math.sqrt(2))
        return (.5*(hi.erf()-lo.erf())).mean(1).clamp_min(1e-12)
    effective=1+(torch.diff(observed,dim=1).abs()>1e-7).sum(1)
    terms=[-effective*nll+math.log(.7)]
    for a,b in ((-.2,.6),(.6,1.6)):
        bounds=predicted.new_tensor([a,b])[None,:].expand(len(predicted),-1)
        amount=mass(bounds)
        compatible=(observed.min(1).values>=a)&(observed.max(1).values<=b)&(amount>=1e-8)
        conditional=nll+amount.log()
        value=-effective*conditional+math.log(.15)
        terms.append(value.masked_fill(~compatible,-torch.inf))
    unknown=-torch.logsumexp(torch.stack(terms),0)/effective
    if complete is not None:unknown=torch.where(complete.reshape(-1),nll,unknown)
    order=windows[:,:,0].masked_fill(~window_mask,float('inf')).argsort(1)
    ordered=windows.gather(1,order[:,:,None].expand(-1,-1,2)); valid=window_mask.gather(1,order)
    explicit=torch.zeros_like(nll); current=torch.zeros_like(ordered[:,0,:]); current_valid=torch.zeros_like(valid[:,0])
    for j in range(windows.shape[1]):
        item=ordered[:,j,:]; active=valid[:,j]
        extend=active&current_valid&(item[:,0]<=current[:,1])
        explicit+=mass(current)*(current_valid&~extend)
        current=torch.stack([torch.where(extend,current[:,0],item[:,0]),
                             torch.where(extend,torch.maximum(current[:,1],item[:,1]),item[:,1])],1)
        current_valid=active
    explicit+=mass(current)*current_valid
    conditional=torch.where(explicit>=1e-8,nll+explicit.clamp_min(1e-12).log(),nll+20.)
    return torch.where(window_mask.any(1),conditional,unknown)


def pairwise_rank_loss(costs,margin=.1,temperature=.5):
    if costs.ndim!=2 or costs.shape[1]<2:raise ValueError('ranking_positive_and_negative_required')
    return F.softplus((costs[:,:1]-costs[:,1:]+margin)/temperature).mean()


def teacher_anchor(predicted,teacher):
    if predicted.shape!=teacher.shape:raise ValueError('teacher_anchor_shape_mismatch')
    return (predicted-teacher.detach()).abs().mean()

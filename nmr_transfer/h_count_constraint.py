"""功能概述：用结构认证环境数约束¹H发射，隔离峰数与实验观测完整性。
optimal_partition在模型预测的一维位置上做质量加权最优K段划分，不读取任何目标谱。
project_details以每段加权中心替换原成分值，保留质量、母节点、交换标签及原顺序，
同步重算分位点；count_penalty返回图均衡的投影平方位移，用于软约束训练。
认证数来自batch.physics第5列×30；0表示未知并恒等回退。只有H计数独立有效时
才使用，不能用C认证标记补出H计数。认证K是环境容量上界；预测已有少于K个
不同位置时保持原值，不添加人为微小偏移伪造峰。物理简并允许多个环境同位移。
最优划分用CPU float64确定离散分支，再在原device用torch加权中心保留梯度。
依赖emission_model.weighted_quantiles；由NuclearSpecGNN的显式count_decode开关
及h_count_train调用。默认模型和旧checkpoint输出不变；无持久缓存和外部副作用。
"""
import numpy as np
import torch


def optimal_partition(values,weights,k):
    values=np.asarray(values,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    if values.ndim!=1 or values.shape!=weights.shape or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights<=0).any():raise ValueError('invalid_count_components')
    if isinstance(k,bool) or int(k)!=k or k<1:raise ValueError('invalid_certified_count')
    n=len(values)
    if not n:return np.empty(0,np.int64)
    if len(np.unique(values))<=k:return np.arange(n,dtype=np.int64)
    order=np.argsort(values,kind='stable');x=values[order];w=weights[order]
    # 平移到局部原点减少大ppm中心下的二阶矩消减误差。
    x=x-x[0];pw=np.r_[0,w.cumsum()];px=np.r_[0,(w*x).cumsum()];pxx=np.r_[0,(w*x*x).cumsum()]
    mass=pw[None,:]-pw[:,None];total=px[None,:]-px[:,None];sq=pxx[None,:]-pxx[:,None]
    cost=np.maximum(0,sq-total**2/np.maximum(mass,1e-300));cost[np.tril_indices(n+1)]=np.inf
    previous=np.full(n+1,np.inf);previous[0]=0;backs=[]
    for step in range(1,k+1):
        choices=previous[:,None]+cost;back=choices.argmin(0);current=choices[back,np.arange(n+1)];current[:step]=np.inf;backs.append(back);previous=current
    labels=np.empty(n,np.int64);end=n
    for j in range(k-1,-1,-1):
        begin=int(backs[j][end]);labels[order[begin:end]]=j;end=begin
    if end:raise RuntimeError('count_partition_backtrace')
    return labels


def project_details(details,batch):
    from .emission_model import weighted_quantiles
    values=details['values'];weights=details['weights'];groups=details['groups'];counts=(batch.physics.reshape(-1,8)[:,5]*30).detach().cpu().numpy()
    xs=values.detach().cpu().double().numpy();ws=weights.detach().cpu().double().numpy();gs=groups.detach().cpu().numpy();cluster=np.arange(len(xs),dtype=np.int64);next_id=len(xs);changed=np.zeros(len(xs),dtype=bool)
    for graph_id,k in enumerate(counts):
        if k<=0:continue
        if abs(k-round(float(k)))>1e-3:raise ValueError('fractional_certified_count')
        indices=np.flatnonzero(gs==graph_id)
        if not len(indices):continue
        if k>ws[indices].sum()+1e-3:raise ValueError('count_exceeds_nuclear_mass')
        if len(np.unique(xs[indices]))<=int(round(float(k))):continue
        changed[indices]=True
        labels=optimal_partition(xs[indices],ws[indices],int(round(float(k))));cluster[indices]=labels+next_id;next_id+=int(labels.max())+1
    if not changed.any():return details
    ids=torch.as_tensor(cluster,device=values.device);mass=torch.zeros(next_id,device=values.device,dtype=values.dtype).index_add(0,ids,weights);sums=torch.zeros_like(mass).index_add(0,ids,values*weights);projected=(sums/mass.clamp_min(torch.finfo(values.dtype).tiny))[ids]
    projected=torch.where(torch.as_tensor(changed,device=values.device),projected,values)
    return dict(details,values=projected,quantiles=weighted_quantiles(projected,weights,groups,batch.num_graphs))


def count_penalty(details,batch):
    projected=project_details(details,batch);g=details['groups'];w=details['weights'];error=(details['values']-projected['values']).square()*w
    sums=torch.zeros(batch.num_graphs,device=w.device,dtype=w.dtype).index_add(0,g,error);mass=torch.zeros_like(sums).index_add(0,g,w);known=batch.physics.reshape(-1,8)[:,5]>0
    per_graph=sums/mass.clamp_min(1)
    return per_graph[known].mean() if known.any() else details['values'].sum()*0

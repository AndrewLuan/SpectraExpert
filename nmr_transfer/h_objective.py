"""功能概述：用已有¹H原子归属约束正确母节点，避免全分子任意峰即可匹配的退化。
site_loss对母节点内完整H标注用质量分位数误差，部分标注只做观测到预测的匹配，
不把未标注H置零；母节点重排不改变损失。shape_loss仅对完整逐H标注或积分计数
相符的训练记录增加分布约束，后者是软辅助目标，不声明测量完整。
objective组合原观测损失、位点误差、分布辅助和固定0.25ppm核宽度的谱似然目标；
这些标签仅在训练/验证使用，检索评分从不读取真结构的位点或完整性标签。
依赖emission_model的details输出和train.Spectra的侧表字段；返回损失及可单独审计的分量。
site_loss允许proxy提供site_weights，在无序理论轨道合并后保留原母节点的总权重；
未给权重的旧目标不变。该权重不是积分或真实H质量，且不用于预测或检索。
母节点重映射可能产生交错owner；site_loss先稳定排序成分，再调用要求有序batch
索引的分位数打包，防止合法轨道被误判为空谱。位置、核质量和目标均不变。
"""
import math
import torch
from .emission_model import weighted_quantiles
from .observation import observation_loss


def site_loss(details,batch):
    values=details['values'];owners=details['owners'];nodes=len(batch.x)
    order=owners.argsort(stable=True)
    pred=weighted_quantiles(values[order],details['weights'][order],owners[order],nodes,points=6)
    mask=batch.site_mask;count=mask.sum(1);valid=count>0
    target=batch.site_targets.masked_fill(~mask,float('inf')).sort(1).values
    indices=(count[:,None]*((torch.arange(6,device=values.device)+.5)/6)[None,:]).long().clamp_max(5)
    target=target.gather(1,indices);target=torch.where(valid[:,None],target,torch.zeros_like(target))
    complete=count==batch.site_expected
    full=(pred-target).abs().mean(1)
    partial=(pred[:,:,None]-target[:,None,:]).abs().min(1).values.mean(1)
    loss=torch.where(complete,full,partial);loss=torch.where(valid,loss,torch.zeros_like(loss))
    weights=getattr(batch,'site_weights',torch.ones_like(loss))
    sums=torch.zeros(batch.num_graphs,device=values.device).index_add_(0,batch.batch,loss*weights)
    counts=torch.zeros_like(sums).index_add_(0,batch.batch,valid.float()*weights)
    return sums/counts.clamp_min(1),counts>0


def objective(details,batch,kind='site'):
    pred=details['quantiles'];presence=observation_loss(pred,batch).mean()
    zero=pred.sum()*0
    if kind=='presence':return dict(loss=presence,presence=presence,site=zero,shape=zero,nll=zero)
    sites,eligible=site_loss(details,batch)
    site=sites[eligible].mean() if eligible.any() else zero
    shape_mask=batch.shape_supervision.reshape(-1)>0
    shape=(pred[shape_mask]-batch.supervised_q[shape_mask]).abs().mean() if shape_mask.any() else zero
    # 对未知缺峰仍仅计算已观测信号的归一化密度，不逐个惩罚未观测分量。
    residual=(batch.q[:,:,None]-pred[:,None,:])/.025
    nll=(-torch.logsumexp(-.5*residual.square(),dim=-1)+math.log(pred.shape[1])).mean()
    return dict(loss=presence+.75*site+.5*shape+.005*nll,presence=presence,site=site,shape=shape,nll=nll)

"""功能概述：按候选H母节点局部环境构建异方差混合谱密度。
node_classes只读候选图，区分脂肪CHn、芳香/醛CH及NH/OH/SH；quantile_classes
用与weighted_quantiles相同CDF选择每个预测分位点的环境类别；reference模式按
冻结分位点到显式发射支持的最近距离取类别，避免CDF边界微扰改动旧预测均值。
class_density给每个分位成分归一化高斯密度，再按质量相加；大宽度不能无代价
提高得分。profile保留原likelihood的网格和窗口积分契约。
parent_grid_nll用训练已归属H在正确母节点的混合似然评价预声明宽度网格，部分
标注只用已观测值。select_widths用固定50个父节点强度的log宽度先验向0.2ppm
收缩，空类别回退0.2；之后仅在检索val选择统一收缩比例，不按测试误差调宽度。
无原始数据改写、外部调用或文件副作用；workflow负责训练/验证边界与持久化。
"""
import math
import torch
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from .h_conditions import LOCAL_NAMES
from .likelihood import GRID_SIZE,BOUNDS

CLASSES=['aliphatic_CH3','aliphatic_CH2','aliphatic_CH','aromatic_CH','aldehyde_CH','NH','OH','SH','other']
WIDTH_GRID=[.05,.075,.1,.15,.2,.25,.3,.4,.5,.7,1.,1.5,2.]


def node_classes(batch):
    env=batch.local_environment;z=batch.x[:,:54].argmax(1)+1;result=torch.full((len(z),),8,dtype=torch.long,device=z.device)
    for flag,tag in [('CH3',0),('CH2',1),('CH',2),('aromatic',3),('aldehyde',4)]:
        result[(env[:,LOCAL_NAMES.index(flag)]>0)&(z==6)]=tag
    for atomic,tag in [(7,5),(8,6),(16,7)]:result[z==atomic]=tag
    return result


def quantile_classes(details,batch,points=128,reference=None):
    values=details['values'];weights=details['weights'];groups=details['groups'];classes=node_classes(batch)[details['owners']]
    if not len(values):return torch.full((batch.num_graphs,points),8,dtype=torch.uint8,device=values.device),torch.full((batch.num_graphs,points),1e6,device=values.device)
    packed,_=to_dense_batch(values,groups,fill_value=1e6,batch_size=batch.num_graphs);mass,_=to_dense_batch(weights,groups,fill_value=0.,batch_size=batch.num_graphs);tags,_=to_dense_batch(classes,groups,fill_value=8,batch_size=batch.num_graphs)
    if reference is not None:
        residual=(reference[:,:,None]-packed[:,None,:]).abs().masked_fill(mass[:,None,:]<=0,float('inf'));indices=residual.argmin(2)
        return tags.gather(1,indices).to(torch.uint8),packed.gather(1,indices)
    ordered,order=packed.sort(1);mass=mass.gather(1,order);tags=tags.gather(1,order);cdf=mass.cumsum(1)/mass.sum(1,keepdim=True).clamp_min(1)
    q=((torch.arange(points,device=values.device)+.5)/points)[None,:].expand(batch.num_graphs,-1).contiguous();indices=torch.searchsorted(cdf.contiguous(),q).clamp_max(ordered.shape[1]-1)
    return tags.gather(1,indices).to(torch.uint8),ordered.gather(1,indices)


@torch.inference_mode()
def class_density(predictions,tags,widths):
    low,high=BOUNDS['1H'];step=(high-low)/(GRID_SIZE-1);p=predictions*10;position=(p-low)/step;left=position.floor().long();fraction=position-left;profile=torch.zeros(len(p),GRID_SIZE,device=p.device)
    for tag,sigma in enumerate(widths):
        active=tags==tag
        if not active.any():continue
        histogram=torch.zeros_like(profile)
        for offset,weight in [(0,1-fraction),(1,fraction)]:
            idx=left+offset;valid=(idx>=0)&(idx<GRID_SIZE)&active;histogram.scatter_add_(1,idx.clamp(0,GRID_SIZE-1),weight*valid/p.shape[1])
        radius=int(math.ceil(4*sigma/step));x=torch.arange(-radius,radius+1,device=p.device)*step;kernel=(torch.exp(-.5*(x/sigma).square())/(math.sqrt(2*math.pi)*sigma)).reshape(1,1,-1)
        for start in range(0,len(p),1024):profile[start:start+1024]+=F.conv1d(histogram[start:start+1024,None,:],kernel,padding=radius).squeeze(1)
    return dict(logp=profile.clamp_min(1e-12).log(),cdf=torch.cat([torch.zeros(len(p),1,device=p.device),profile.cumsum(1)*step],1),low=low,high=high,step=step,widths=list(widths))


@torch.inference_mode()
def parent_grid_nll(details,batch):
    n=len(batch.x);pred,_=to_dense_batch(details['values'],details['owners'],fill_value=0.,batch_size=n);mass,_=to_dense_batch(details['weights'],details['owners'],fill_value=0.,batch_size=n);valid=(batch.site_mask.sum(1)>0)&(mass.sum(1)>0);tags=node_classes(batch)[valid];target=batch.site_targets[valid]*10;observed=batch.site_mask[valid];pred=pred[valid]*10;mass=mass[valid];logmass=torch.where(mass>0,mass.clamp_min(1e-30).log(),-torch.inf)-mass.sum(1,keepdim=True).log();totals=torch.zeros(len(CLASSES),len(WIDTH_GRID),device=pred.device);counts=torch.zeros(len(CLASSES),device=pred.device)
    for j,sigma in enumerate(WIDTH_GRID):
        residual=(target[:,:,None]-pred[:,None,:])/sigma;cost=-torch.logsumexp(logmass[:,None,:]-.5*residual.square(),2)+math.log(sigma*math.sqrt(2*math.pi));cost=(cost.masked_fill(~observed,0.).sum(1)/observed.sum(1));totals[:,j].index_add_(0,tags,cost)
    counts.index_add_(0,tags,torch.ones(len(tags),device=pred.device));return totals,counts


def select_widths(totals,counts,prior_strength=50.):
    grid=torch.tensor(WIDTH_GRID,dtype=totals.dtype,device=totals.device);penalty=prior_strength*(grid.log()-math.log(.2)).square();choice=(totals+penalty[None,:]).argmin(1);return grid[choice].cpu().tolist()

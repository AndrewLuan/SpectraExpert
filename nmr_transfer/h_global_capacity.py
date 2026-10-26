"""功能概述：用有限容量正观测匹配检验 HMDB 峰簇被同一宽核重复解释的问题。

cluster_predictions 按与观测相同的最大簇宽合并预测位置，避免单链接链并整段。
capacity_excess 用一维有序动态规划，每个预测峰簇最多解释一个观测峰簇；预测
未报告部分可免费跳过，观测污染可付固定有限代价，因此没有完整谱或缺峰负标签。
返回相对独立最近邻匹配的额外代价，只度量重复使用容量，不重复惩罚位移本身。
宽度/权重在train/val预声明网格比较；分子、ID、真结构、归属与外部服务均不输入。
显式频窗只保留窗内预测支持，未知覆盖保持未知。单位均为ppm。
"""
import torch


@torch.inference_mode()
def cluster_predictions(predictions,width):
    """输入每行有序位置，输出有序簇均值与有效标志；重复分位点不增加容量。"""
    if predictions.ndim!=2 or predictions.shape[1]<1 or width<=0:
        raise ValueError('capacity_invalid_prediction_shape_or_width')
    if not torch.isfinite(predictions).all() or (predictions.diff(dim=1)<-1e-6).any():
        raise ValueError('capacity_requires_finite_sorted_positions')
    size=predictions.shape;ids=torch.zeros(size,device=predictions.device,dtype=torch.long)
    first=predictions[:,0];current=ids[:,0]
    for j in range(1,size[1]):
        new=predictions[:,j]-first>width+1e-8
        current=current+new;ids[:,j]=current;first=torch.where(new,predictions[:,j],first)
    totals=torch.zeros_like(predictions).scatter_add_(1,ids,predictions)
    counts=torch.zeros_like(predictions).scatter_add_(1,ids,torch.ones_like(predictions))
    return totals/counts.clamp_min(1),counts>0


@torch.inference_mode()
def capacity_excess(centers,valid,observed,windows=(),sigma=.2,outlier=.5):
    """最优单次容量匹配减去独立最近匹配；窗外/未报预测支持没有代价。"""
    observed=torch.as_tensor(observed,device=centers.device,dtype=centers.dtype).sort().values
    if observed.ndim!=1 or not len(observed) or not torch.isfinite(observed).all():
        raise ValueError('capacity_invalid_observation')
    active=valid.clone()
    if windows:
        inside=torch.zeros_like(active)
        for lo,hi in windows:inside|=(centers>=lo)&(centers<=hi)
        active&=inside
    missing=(outlier/sigma)**2;k=len(observed)
    cost=((centers[:,:,None]-observed[None,None,:])/sigma).square().clamp_max(missing)
    cost=cost.masked_fill(~active[:,:,None],float('inf'))
    independent=cost.min(1).values.clamp_max(missing).sum(1)
    steps=torch.arange(k+1,device=centers.device,dtype=centers.dtype)*missing
    dp=steps[None,:].expand(len(centers),-1).clone()
    for j in range(centers.shape[1]):
        candidate=torch.minimum(dp[:,1:],dp[:,:-1]+cost[:,j,:])
        # 可跳过任意观测污染；累积最小值等价于逐观测递推，避免K次Python循环。
        joined=torch.cat([dp[:,:1],candidate],1)
        dp=(joined-steps[None,:]).cummin(1).values+steps[None,:]
    return ((dp[:,-1]-independent)/k).clamp_min(0)

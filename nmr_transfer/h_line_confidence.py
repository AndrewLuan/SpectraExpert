"""功能概述：把HMDB峰高仅用于细线置信度，明确不转换为H积分。
line_table将同位置重复行取最大峰高，按原0.08ppm最大簇宽聚类；相对峰高由簇内
最大值决定，避免重复行提高权重。未给峰高时全部保留并使用原等峰评分。
confidence_nll使用归一化谱密度与均匀污染的混合概率；低峰高允许更高污染概率，
不会免费删除不匹配位置。未知覆盖仍边缘化原全窗/半窗假设，公共参考偏移使用
冻结零均值先验。阈值过滤为独立消融，至少保留最高细线，不改变查询身份。
GRID预声明原等峰基线及置信度/阈值对照，只由固定val选择；低峰高不能解释成
已证实杂质或卫星峰。输入不含真结构计数/原子归属，所有候选共用观测处理。
依赖h_observed与likelihood的旧评分契约；无文件写入、外部调用或原始表改写。
"""
import math
import numpy as np
import torch
from .h_observed import cluster_lines,marginal_shift,line_score
from .likelihood import GRID_SIZE,HALVES,interval_mass

SHIFTS=[-.2,-.1,0.,.1,.2]
GRID=[dict(kind='baseline',eta=0.,gamma=1.,threshold=0.)]+[dict(kind='confidence',eta=e,gamma=g,threshold=0.) for e in [.25,.5,.8] for g in [.5,1.]]+[dict(kind='threshold',eta=0.,gamma=1.,threshold=t) for t in [.01,.03,.05]]


def line_table(raw,width=.08,threshold=0.):
    xs=np.asarray(raw['value']['x'],float);ys=np.asarray(raw['value'].get('y',[]),float)
    if ys.ndim!=1 or len(ys)!=len(xs) or not np.isfinite(ys).all() or np.max(ys)<=0:ys=np.ones(len(xs))
    unique=np.unique(xs);heights=np.array([max(0,float(ys[xs==x].max())) for x in unique]);relative=heights/max(float(heights.max()),1e-12);keep=relative>=threshold;unique=unique[keep];relative=relative[keep]
    centers,groups=cluster_lines(unique,width);confidence=np.array([relative[np.isin(unique,group)].max() for group in groups]);return centers,confidence


def sample_logp(kernel,xs):
    u=torch.tensor((xs-kernel['low'])/kernel['step'],device=kernel['logp'].device).clamp(0,GRID_SIZE-1);i=u.floor().long();j=(i+1).clamp_max(GRID_SIZE-1);f=u-i;return kernel['logp'][:,i]*(1-f)+kernel['logp'][:,j]*f


@torch.inference_mode()
def confidence_nll(kernel,query,centers,relative,eta,gamma):
    values=sample_logp(kernel,centers);contamination=torch.tensor(eta*(1-np.asarray(relative)**gamma),device=values.device,dtype=values.dtype)
    def mixed(logdensity,span):
        log_signal=torch.log1p(-contamination);log_noise=torch.where(contamination>0,contamination.log()-math.log(span),-torch.inf);return torch.logaddexp(logdensity+log_signal,log_noise).mean(1)
    n=max(1,len(centers));state=query['observation'];windows=state['windows']
    if windows:
        mass=interval_mass(kernel,windows);width=sum(b-a for a,b in windows);conditional=(values-mass[:,None].log()).masked_fill(mass[:,None]<1e-8,-1e6);return -mixed(conditional,width)
    base=mixed(values,kernel['high']-kernel['low'])
    if state['kind']=='complete_atoms':return -base
    terms=[n*base+math.log(.7)]
    for window in HALVES['1H']:
        if centers.min()>=window[0] and centers.max()<=window[1]:
            mass=interval_mass(kernel,[window]);conditional=(values-mass[:,None].log()).masked_fill(mass[:,None]<1e-8,-1e6);terms.append(n*mixed(conditional,window[1]-window[0])+math.log(.15))
    return -torch.logsumexp(torch.stack(terms),0)/n


@torch.inference_mode()
def score(kernel,query,raw,config,sigma):
    if config['kind']=='baseline':return line_score(kernel,query,raw['value']['x'],.08,SHIFTS,sigma)
    centers,relative=line_table(raw,.08,config['threshold']);costs=[]
    for shift in SHIFTS:costs.append(confidence_nll(kernel,query,centers+shift,relative,config['eta'],config['gamma']))
    return marginal_shift(torch.stack(costs),SHIFTS,sigma,max(1,len(centers)))

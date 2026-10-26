"""功能概述：保留原HMDB密度匹配，仅增加预测环境的报告计数与宽偏移对照。
density_features复用冻结的全H/非交换H密度和旧0.08ppm观测簇；峰高不作积分。
compare保留旧参考偏移，或在同一零均值宽网格上边缘化公共误差，不对真候选
单独拟合。计数先验来自train模型环境数/已报告簇数，缺峰变体用独立train增强先验。
replace_empty_support只对候选自身非交换支持完全为空的行回退全H，其他行逐位
不变；频窗查询由workflow继续旧路由。所有候选使用相同观测和先验规则。
GRID含原基线与24个预声明对照，只由固定val选择；密度/报告过程是近似观测模型，
不把预测簇数称为精确结构环境数。无文件写入、训练或外部调用。
"""
import torch
from .h_env_report import SHIFTS
from .h_observation_finalize import query_score,visible_lines,blend
from .h_observed import line_score,marginal_shift,cluster_lines,count_nll
from .likelihood import log_score

GRID=[dict(kind='baseline')]+[dict(kind='density_reporting',visibility_fallback=f,shift=s,beta=b) for f in [False,True] for s in ['legacy',.2,.5] for b in [0.,.25,1.,4.]]


def replace_empty_support(kernel,full,empty):
    return dict(kernel,logp=torch.where(empty[:,None],full['logp'],kernel['logp']),cdf=torch.where(empty[:,None],full['cdf'],kernel['cdf']))


@torch.inference_mode()
def density_features(context,q,priors,inputs):
    frozen,prior,config,oldpriors,raw=inputs;water=context.water;cfg=frozen['water_HMDB'] if water and frozen['water_HMDB'] is not None else frozen['HMDB'];xs=visible_lines(q,raw);centers,_=cluster_lines(xs,.08);k=len(centers)
    if context.density_kernels is None:
        kernel=context.given[cfg['kernel']]
        if 'visibility' in cfg:kernel=blend(kernel,context.given['fixed_'+cfg['kernel']],cfg['visibility'])
        context.density_kernels={False:kernel,True:replace_empty_support(kernel,context.given[cfg['kernel']],context.empty_fixed) if water else kernel}
    scores={}
    for fallback,kernel in context.density_kernels.items():
        if fallback and not water:
            for mode in ['legacy',.2,.5]:scores[fallback,mode]=scores[False,mode]
            continue
        scores[fallback,'legacy']=query_score(q,context.given,context.counts,prior,config,frozen,oldpriors,raw) if not fallback else line_score(kernel,q,xs,.08,[-.2,-.1,0.,.1,.2],cfg['shift_sigma'])
        costs=torch.stack([log_score(kernel,dict(q,q=centers+s),'1H') for s in SHIFTS])
        for sigma in [.2,.5]:scores[fallback,sigma]=marginal_shift(costs,SHIFTS,sigma,k)
    family='water' if water else 'nonwater';process='dropout' if q['observation'].get('partial_peaks') else 'normal';kind='fixed' if water else 'full';points=context.fixed_points if water else context.full_points;count=count_nll(points[1].sum(1).float(),k,priors[process][family][kind]);return dict(scores=scores,count=count)


def density_score(features,config):return features['scores'][config['visibility_fallback'],config['shift']]+config['beta']*features['count']

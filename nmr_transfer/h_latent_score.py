"""功能概述：对稀疏逐原子谱表边缘化未知溶剂/公共参考偏移，并建模报告原子数。
这些谱表不是积分；count_nll仅使用train拟合的“报告计数|候选H数”过程，查询只给
实际报告的位移/计数，不给真实结构总H数、母节点、完整性或测试拟合偏移。
weighted_log_score按重复分位点的精确频数计算原KDE公式，减少重复gather；给定
条件/零偏移分支仍调用原log_score，使baseline路径逐位保持原评分。
prototype_keys使用明确训练支持的Water/CDCl3/DMSO假设，未知pH保持掩码；latent
表示对缺失条件积分，不将任何一种假设填作已知实验条件。条件已知时只用原条件。
fit_reporting_priors为明确整峰删除变体从train原子报告表模拟相同删峰，避免复用
完整报告计数先验。explicit窗口查询沿用冻结评分，不将窗外H当作漏报。
features生成四种条件先验×三种偏移宽度与计数项；CONFIGS由固定val选择，test不选。
本文件无文件写入或外部API；workflow负责保存冻结配置和同库评估。
bank_key 同时兼容只记录 condition_key_mode 的新 bank 与记录 condition_mode 的旧 bank，
缺失两者时明确报错；不能在存在新字段时提前求值旧字段造成 KeyError。
"""
import math
import numpy as np
import torch
from .data import identity
from .h_conditions import encode_conditions,SOLVENTS
from .h_condinfer import condition_key
from .h_diagnose import cond_key
from .h_labels import make_key
from .h_score import spectral_observation
from .h_observed import fit_dropout_prior,count_nll,marginal_shift
from .likelihood import log_score,interval_mass,GRID_SIZE,HALVES

SHIFTS=[-.2,-.1,0.,.1,.2]
CONDITIONS=['given','train_mixture','uniform_prototypes','train_prototypes']
CONFIGS=[dict(condition=c,shift_sigma=s,beta=b) for c in CONDITIONS for s in [0.,.1,.2] for b in [0.,.25,1.,4.]]
PROTOTYPES=['Water','CDCl3','DMSO']


def bank_key(bank,row,metadata):
    if len(next(iter(bank['predictions'])))==4:return cond_key(row)
    mode=bank.get('condition_key_mode') or bank.get('condition_mode')
    if mode is None:raise ValueError('conditional_bank_mode_missing')
    return condition_key(row,metadata,mode)


def prototype_keys(bank):
    result={}
    for name in PROTOTYPES:
        meta=encode_conditions(dict(solvent=name));r=dict(source='latent_prototype',domain='experimental',record_id=name,smiles='',cond=meta['old_cond'])
        result[name]=bank_key(bank,r,{make_key(r):meta})
    return result


def fit_reporting_priors(annotations,metadata):
    atomic={}
    for key,meta in annotations.items():
        if key[0]!='experimental' or not key[1].startswith('nmrshiftdb2'):continue
        observed=spectral_observation(meta)
        if observed is not None:atomic[key]=dict(meta,xs=observed[0],weights=observed[1])
    drop=fit_dropout_prior(atomic,concentration=2.)
    sums=np.zeros(3);seen={}
    for (domain,source,rid,smi),meta in metadata.items():
        if domain!='experimental':continue
        if smi not in seen:seen[smi]=identity(smi)[2]
        if seen[smi]!='train':continue
        v=np.asarray(meta['vector']);sums+=v[[SOLVENTS.index(x) for x in ['water','chloroform','dmso']]]
    prior=(sums+1)/(sums.sum()+3)
    return dict(dropout=drop,solvent_prior=prior.tolist(),solvent_training_mass=sums.tolist(),scope='training partitions only; known solvent metadata frequency; unknown/other solvent excluded from prototype prior')


def eligible(query):
    return query.get('measure')=='observed_assigned_atom_equal_partial_possible' and not query['observation']['windows'] and len(query.get('observed_xs',[]))>0


def unknown_solvent(query):
    if '_condition_known' in query:return not query['_condition_known']
    return not np.any(np.asarray(query['cond'])[1:]>0)


def weighted_log_score(kernel,query):
    q,w=np.unique(np.asarray(query['q']),return_counts=True);device=kernel['logp'].device
    u=torch.tensor((q-kernel['low'])/kernel['step'],device=device).clamp(0,GRID_SIZE-1);i=u.floor().long();j=(i+1).clamp_max(GRID_SIZE-1);f=u-i
    values=kernel['logp'][:,i]*(1-f)+kernel['logp'][:,j]*f
    weights=torch.tensor(w/w.sum(),dtype=values.dtype,device=device);base=(values*weights).sum(1);state=query['observation']
    if state['windows']:
        mass=interval_mass(kernel,state['windows']);return -(base-mass.log()).masked_fill(mass<1e-8,-1e6)
    if state['kind']=='complete_atoms':return -base
    effective=max(1,len(q));terms=[effective*base+math.log(.7)]
    for window in HALVES['1H']:
        if q.min()>=window[0] and q.max()<=window[1]:
            mass=interval_mass(kernel,[window]);conditional=(base-mass.log()).masked_fill(mass<1e-8,-1e6);terms.append(effective*conditional+math.log(.15))
    return -torch.logsumexp(torch.stack(terms),0)/effective


@torch.inference_mode()
def features(query,kernels,counts,normal_prior,reporting):
    effective=max(1,len(np.unique(query['q'])));states=['given']+PROTOTYPES if unknown_solvent(query) else ['given'];costs=[]
    for name in states:
        kernel=kernels[name];values=[]
        for shift in SHIFTS:
            if name=='given' and shift==0:score=log_score(kernel,query,'1H')
            else:score=weighted_log_score(kernel,dict(query,q=np.asarray(query['q'])+shift))
            values.append(score)
        costs.append(torch.stack(values))
    base=torch.stack(costs);nll={};p=torch.tensor(reporting['solvent_prior'],device=counts.device)
    for sigma in [0.,.1,.2]:
        per_condition=torch.stack([marginal_shift(c,SHIFTS,sigma,effective) for c in base]);nll['given',sigma]=per_condition[0]
        if len(states)==1:
            for name in CONDITIONS[1:]:nll[name,sigma]=per_condition[0]
        else:
            priors={'train_mixture':torch.cat([p.new_tensor([.5]),p*.5]),'uniform_prototypes':torch.cat([p.new_tensor([0.]),torch.ones_like(p)/3]),'train_prototypes':torch.cat([p.new_tensor([0.]),p])}
            for name,prior in priors.items():
                logprior=torch.where(prior>0,prior.log(),torch.full_like(prior,-float('inf')))
                nll[name,sigma]=-torch.logsumexp(-effective*per_condition+logprior[:,None],0)/effective
    ws=np.asarray(query['observed_weights']);k=int(round(float(ws.sum())));entry=(reporting['dropout'] if query['observation'].get('partial_peaks') else normal_prior).get(query['source'])
    count=torch.zeros_like(counts)
    if entry is not None and k>=1 and np.allclose(ws,ws.round(),atol=.001):count=count_nll(counts,k,entry)
    return dict(nll=nll,count=count)


def score(features,config):
    return features['nll'][config['condition'],config['shift_sigma']]+config['beta']*features['count']

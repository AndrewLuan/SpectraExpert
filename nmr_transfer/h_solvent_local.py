"""功能概述：用低容量局部残差检验稀少苯/NMe2酰胺响应的跨结构泛化。
SolventLocalSpecGNN冻结DeepConditionSpecGNN全部既有参数，只训练54个线性头参数。
输入是24维局部描述符及冻结预测中心/分裂logit；不使用全图池化作为新增残差输入。
dimethyl_amide_mask按二维连接识别同一酰胺N上的两个甲基；残差只在明确苯溶剂
和这些甲基上启用，其他溶剂/位置严格为零。中心调整最多1ppm，分裂logit调整
最多2；核质量、旧H槽及对称双峰发射不变，不按DMF身份设置数值。
由emission_model.make_model恢复；训练/专用验证仅使用结构hash train/val记录。
invariant_outside_benzene标记供h_condinfer在逐张量核对冻结基座后复用未变化bank。
"""
import torch
from torch import nn
from .h_deep_context import DeepConditionSpecGNN
from .h_conditions import LOCAL_DIM, LOCAL_NAMES, CONDITION_DIM, SOLVENTS


def dimethyl_amide_mask(batch):
    env=batch.local_environment
    methyl=env[:,LOCAL_NAMES.index('amide_N_methyl')]>0
    nitrogen=env[:,LOCAL_NAMES.index('amide_N')]>0
    src,tgt=batch.edge_index;linked=methyl[src]&nitrogen[tgt]
    counts=torch.zeros(len(env),device=env.device,dtype=torch.long)
    counts.index_add_(0,tgt[linked],torch.ones_like(tgt[linked]))
    result=torch.zeros(len(env),device=env.device,dtype=torch.bool)
    result[src[linked&(counts[tgt]>=2)]]=True
    return result


class SolventLocalSpecGNN(DeepConditionSpecGNN):
    invariant_outside_benzene=True

    def __init__(self,nucleus='1H',width=256,layers=3,condition_mode='local',response_mode='solvent',context_depth=6):
        super().__init__(nucleus,width,layers,condition_mode,response_mode,context_depth)
        for p in self.parameters():p.requires_grad_(False)
        self.solvent_local_head=nn.Linear(LOCAL_DIM+2,2)
        nn.init.zeros_(self.solvent_local_head.weight);nn.init.zeros_(self.solvent_local_head.bias)

    def adjust_conditions(self,shifts,split_logits,context,batch):
        shifts,split_logits=super().adjust_conditions(shifts,split_logits,context,batch)
        c=batch.condition_vector.reshape(-1,CONDITION_DIM)[batch.batch]
        gate=c[:,SOLVENTS.index('benzene')]*dimethyl_amide_mask(batch)
        features=torch.cat([batch.local_environment.float(),shifts[:,:3].mean(1,keepdim=True).detach(),split_logits.detach()],1)
        delta=self.solvent_local_head(features).tanh()*gate[:,None]
        return shifts+.1*delta[:,:1],split_logits+2*delta[:,1:]

"""功能概述：¹H条件残差消融模型，冻结同一个SpecGNN骨干与原始发射头。
ConditionSpecGNN所有消融均有相同残差MLP、初始化和训练参数量；control保留旧4维
条件，solvent新增溶剂特征，ph再增加水相pH，local再增加局部环境。缺少条件不填值。
adjust_conditions对6个H槽及酰胺双峰分裂logit作有界残差调整；甲基质量守恒与
原模型的对称分裂处理保持不变。所有候选共享查询条件，不使用候选实验标签。
training_details只对存在非交换H的明确水相HMDB训练目标抑制交换H；全为交换H
的分子仍保留观测到的发射，避免用空谱占位构造假损失。逐原子
标注/积分谱仍使用全部预测，不根据真实结构猜查询是否缺峰。
依赖emission_model的恒等扩展点、h_conditions维度及h_objective；无文件副作用。
"""
import torch
from torch import nn
from .emission_model import NuclearSpecGNN,weighted_quantiles
from .model import MLP
from .h_conditions import CONDITION_DIM,SOLVENT_DIM,LOCAL_DIM

MODES=['control','solvent','ph','local']


class ConditionSpecGNN(NuclearSpecGNN):
    def __init__(self,nucleus='1H',width=256,layers=3,condition_mode='local'):
        super().__init__(nucleus,width,layers)
        if nucleus!='1H' or condition_mode not in MODES:raise ValueError('invalid_condition_model')
        self.condition_mode=condition_mode
        for param in self.parameters():param.requires_grad_(False)
        self.condition_head=MLP([2*width+12+CONDITION_DIM+LOCAL_DIM,128,7])
        nn.init.zeros_(self.condition_head[-1].weight);nn.init.zeros_(self.condition_head[-1].bias)

    def adjust_conditions(self,shifts,split_logits,context,batch):
        extra=batch.condition_vector.reshape(-1,CONDITION_DIM).clone()
        if self.condition_mode=='control':extra.zero_()
        elif self.condition_mode=='solvent':extra[:,SOLVENT_DIM:]=0
        env=batch.local_environment.float() if self.condition_mode=='local' else torch.zeros(len(context),LOCAL_DIM,device=context.device)
        change=self.condition_head(torch.cat([context,extra[batch.batch],env],1))
        return shifts+.2*change[:,:6].tanh(),split_logits+2*change[:,6:].tanh()


def training_details(details,batch):
    if not batch.line_water.any():return details
    fixed_mass=torch.zeros(batch.num_graphs,device=details['values'].device).index_add_(0,details['groups'],details['weights']*(~details['exchangeable']).float())
    suppress=batch.line_water.reshape(-1)&(fixed_mass>0)
    keep=~(details['exchangeable']&suppress[details['groups']])
    q=weighted_quantiles(details['values'][keep],details['weights'][keep],details['groups'][keep],batch.num_graphs)
    return dict(details,quantiles=q)

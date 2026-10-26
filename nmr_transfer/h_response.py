"""功能概述：在完整冻结条件模型上增加锚定的水相pH/溶剂响应。
AnchoredSpecGNN.adjust_conditions先复用旧条件残差，再增加仅在条件明确时启用
的响应。pH曲线以7为零点；未知pH严格为零，不将7填作观测值。曲线转折参数仅
是可学习响应形状，不宣称为已测pKa。局部环境分支描述酸/胺邻近的结构响应。
溶剂分支使用已知溶剂描述符相对水的差，未知溶剂严格为零；所有模式参数量相同，
只有输入/分支掩码不同。原发射头、GINE、已有条件头全部冻结，核质量契约不变。
由emission_model.make_model按anchored_atom和response_mode恢复；训练入口负责
warm-start完整旧state。无数据读取、谱标签依赖、外部调用或持久化副作用。
"""
import torch
from torch import nn
from .h_condmodel import ConditionSpecGNN
from .h_conditions import CONDITION_DIM,SOLVENT_DIM,LOCAL_DIM,encode_conditions
from .model import MLP

RESPONSE_MODES=['ph','local','pairs','solvent']


class AnchoredSpecGNN(ConditionSpecGNN):
    def __init__(self,nucleus='1H',width=256,layers=3,condition_mode='local',response_mode='local'):
        super().__init__(nucleus,width,layers,condition_mode)
        if response_mode not in RESPONSE_MODES:raise ValueError('invalid_response_mode')
        self.response_mode=response_mode
        for p in self.parameters():p.requires_grad_(False)
        self.response_head=MLP([2*width+12+LOCAL_DIM,64,9+42])
        nn.init.zeros_(self.response_head[-1].weight);nn.init.zeros_(self.response_head[-1].bias)
        # 转折和斜率从平滑中性响应开始；幅度为零，新增模型逐位复现初始模型。
        self.register_buffer('water_descriptors',torch.tensor(encode_conditions(dict(solvent='Water'))['vector'][SOLVENT_DIM-7:SOLVENT_DIM-1]),persistent=False)

    def adjust_conditions(self,shifts,split_logits,context,batch):
        shifts,split_logits=super().adjust_conditions(shifts,split_logits,context,batch)
        env=batch.local_environment.float() if self.response_mode!='ph' else torch.zeros(len(context),LOCAL_DIM,device=context.device)
        response=self.response_head(torch.cat([context,env],1));c=batch.condition_vector.reshape(-1,CONDITION_DIM)[batch.batch]
        ph=7+7*c[:,SOLVENT_DIM+1];turn=2+10*response[:,7].sigmoid();slope=.5+3.5*response[:,8].sigmoid()
        curve=((ph-turn)/slope).sigmoid()-((7-turn)/slope).sigmoid();curve=curve*c[:,SOLVENT_DIM]
        change=response[:,:7].tanh()*curve[:,None]
        if self.response_mode=='solvent':
            # six physical descriptors occupy the six slots before solvent-known.
            solvent=(c[:,SOLVENT_DIM-7:SOLVENT_DIM-1]-self.water_descriptors)*c[:,SOLVENT_DIM-1:SOLVENT_DIM]
            interaction=response[:,9:].reshape(-1,7,6).tanh();change=change+.25*(interaction*solvent[:,None,:]).sum(2)
        return shifts+.15*change[:,:6],split_logits+1.5*change[:,6:]

"""功能概述：在已训练的六层骨干上增加溶剂条件化的两步GINE消息。
SolventMessageSpecGNN冻结全部原参数；三个同布局消融late/structure/solvent
分别只用新发射残差、增加无溶剂消息、再把查询溶剂加入节点和边调制。
酰基环境来自h_acyl的纯结构描述符，明确区分甲酰和其他酰基；不检查DMF身份。
message_condition对未知溶剂严格归零，pH仍由已有显式水相分支处理。
unconditioned_embedding可跨条件缓存原深层表示；structure_embedding拒绝旧
final_context_embedding，防止带条件的最终表示被错误共享。h_condinfer必须只缓存
unconditioned_context_embedding，再逐查询条件重算消息。所有新增输出零初始化，
暖启动精确保留旧谱；仍由原发射器保证甲基分裂和H质量守恒。无文件副作用。
"""
import torch
from torch import nn
from .h_deep_context import DeepConditionSpecGNN
from .h_conditions import SOLVENT_DIM,CONDITION_DIM
from .h_acyl import ACYL_DIM
from .model import MLP,Block

MESSAGE_MODES=['late','structure','solvent']


class SolventMessageSpecGNN(DeepConditionSpecGNN):
    conditioned_messages=True

    def __init__(self,nucleus='1H',width=256,layers=3,condition_mode='local',response_mode='solvent',context_depth=6,message_mode='solvent'):
        super().__init__(nucleus,width,layers,condition_mode,response_mode,context_depth)
        if message_mode not in MESSAGE_MODES:raise ValueError('invalid_message_mode')
        self.message_mode=message_mode
        for p in self.parameters():p.requires_grad_(False)
        self.solvent_message_bonds=MLP([222+2*ACYL_DIM+SOLVENT_DIM,width,width],bias=False,last_act=True)
        self.solvent_message_node=MLP([ACYL_DIM+SOLVENT_DIM,width,2*width])
        self.solvent_message_blocks=nn.ModuleList(Block(width) for _ in range(2))
        self.solvent_message_gates=nn.Parameter(torch.zeros(2))
        self.solvent_message_head=MLP([2*width+12+ACYL_DIM+SOLVENT_DIM,128,7])
        nn.init.zeros_(self.solvent_message_head[-1].weight);nn.init.zeros_(self.solvent_message_head[-1].bias)

    def message_condition(self,batch):
        c=batch.condition_vector.reshape(-1,CONDITION_DIM)[:,:SOLVENT_DIM]
        return c*c[:,-1:] if self.message_mode=='solvent' else torch.zeros_like(c)

    def unconditioned_embedding(self,batch):
        if hasattr(batch,'final_context_embedding'):raise ValueError('conditional_context_must_not_reuse_final_cache')
        if hasattr(batch,'unconditioned_context_embedding'):return batch.unconditioned_context_embedding.float()
        return super().structure_embedding(batch)

    def structure_embedding(self,batch):
        h=self.unconditioned_embedding(batch)
        if self.message_mode=='late':return h
        c=self.message_condition(batch)[batch.batch];env=batch.acyl_environment.float();x=batch.x.float();i,j=batch.edge_index
        bond=self.solvent_message_bonds(torch.cat([x[i],batch.edge_attr.float(),x[j],env[i],env[j],c[i]],1))
        scale,shift=self.solvent_message_node(torch.cat([env,c],1)).chunk(2,1)
        for gate,block in zip(self.solvent_message_gates,self.solvent_message_blocks):
            z=h*(1+.1*scale.tanh())+.1*shift.tanh();h=h+gate.tanh()*(block(z,batch.edge_index,bond)-z)
        return h

    def adjust_conditions(self,shifts,split_logits,context,batch):
        shifts,split_logits=super().adjust_conditions(shifts,split_logits,context,batch)
        # 所有消融的新晚期头均看相同条件；只有消息层的输入掩码不同。
        c=batch.condition_vector.reshape(-1,CONDITION_DIM)[:,:SOLVENT_DIM];c=c*c[:,-1:]
        d=self.solvent_message_head(torch.cat([context,batch.acyl_environment.float(),c[batch.batch]],1)).tanh()
        return shifts+.1*d[:,:6],split_logits+d[:,6:]

"""功能概述：在原三层冻结GINE表示上增加零门控消息传递，检验更远局部上下文。
DeepConditionSpecGNN保留已有条件/锚定响应、六H槽和核质量约束；额外六个Block
始终具有相同参数布局，context_depth只启用0/1/3/6层。每层残差由初始为零的
tanh门控乘上，因此所有深度都精确复现同一初始均值，再逐步学习邻域信息。
旧GINE和发射头冻结，条件/响应头与新块可训练；不是3D或构象平均模型。
structure_embedding优先复用同权重的final_context_embedding，供候选多条件推断；
普通训练仍从prepared中经权重核对的原GINE缓存出发，不能把新块应用两次。
verify_cached_backbone拒绝与缓存所属权重不同的旧GINE。由make_model恢复，
h_deeptrain训练、h_condinfer按最终上下文一次缓存；无文件读取或外部副作用。
"""
import torch
from torch import nn
from .h_response import AnchoredSpecGNN
from .model import MLP,Block

DEPTHS=[0,1,3,6]
GINE_PREFIXES=('init_atom_emb.','bond_emb_list.','atom_bond_block_list.')


def verify_cached_backbone(state,reference):
    keys={k for k in state if k.startswith(GINE_PREFIXES)};expected={k for k in reference if k.startswith(GINE_PREFIXES)}
    if keys!=expected or any(not torch.equal(state[k],reference[k]) for k in keys):raise ValueError('prepared_GINE_embedding_owner_mismatch')


class DeepConditionSpecGNN(AnchoredSpecGNN):
    def __init__(self,nucleus='1H',width=256,layers=3,condition_mode='local',response_mode='solvent',context_depth=3):
        super().__init__(nucleus,width,layers,condition_mode,response_mode)
        if context_depth not in DEPTHS:raise ValueError('unsupported_context_depth')
        self.context_depth=context_depth
        for name,p in self.named_parameters():p.requires_grad_(name.startswith(('condition_head.','response_head.')))
        self.context_bonds=nn.ModuleList(MLP([222,width,width],bias=False,last_act=True) for _ in range(6));self.context_blocks=nn.ModuleList(Block(width) for _ in range(6));self.context_gates=nn.Parameter(torch.zeros(6))

    def structure_embedding(self,batch):
        if hasattr(batch,'final_context_embedding'):return batch.final_context_embedding.float()
        h=super().structure_embedding(batch)
        if not self.context_depth:return h
        x=batch.x.float();src,tgt=batch.edge_index;pair=torch.cat([x[src],batch.edge_attr.float(),x[tgt]],1)
        for i in range(self.context_depth):
            updated=self.context_blocks[i](h,batch.edge_index,self.context_bonds[i](pair));h=h+self.context_gates[i].tanh()*(updated-h)
        return h

"""功能概述：用户提供的显式氢GINE SpecGNN在NMR上的谱分布解码与域适配。
主要职责：MLP/Block保留端点原始特征构边、mean聚合、双LayerNorm和残差；SpecGNN
保留99/24输入、256宽三层骨干及每原子输出后mean读出。输出改为128个排序分位数，
避免凭空指定NMR线宽，也允许同一二维表示下表达多个非等价环境的谱位置分布。
global_head用池化图表示及结构物理特征补充分布解码，避免强制每个同表示原子只出一峰。
可选adapter是按结构环境和溶剂条件学习的零初始化残差；仅实验适配时启用。
关系：data.py提供99/24特征和8维物理特征，train.py执行预训练、微调及校正重放。
图只含结构，不读位移；physical的已知分组计数来自既有确定性算法并保留缺失标记。
不恢复J、多重性或采样谱线，不把分位数个数解释为环境数。无网络或文件副作用。
"""
import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_mean_pool
from .data import Q


class MLP(nn.Sequential):
    def __init__(self, dims, bias=True, last_act=False, dropout=0.):
        layers = []
        for i,(n,m) in enumerate(zip(dims[:-1],dims[1:])):
            linear = nn.Linear(n,m,bias=bias)
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            if last_act or i < len(dims)-2:
                layers += [nn.SiLU(), nn.Dropout(dropout)]
        super().__init__(*layers)


class Block(nn.Module):
    def __init__(self, width, dropout=0.):
        super().__init__()
        self.conv = GINEConv(MLP([width,width,width]), edge_dim=width, aggr='mean')
        self.norm = nn.LayerNorm(width)
        self.graph_norm = nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edges, bond):
        return h + self.drop(nn.functional.silu(self.graph_norm(self.norm(self.conv(h,edges,bond)))))


class SpecGNN(nn.Module):
    def __init__(self, width=256, layers=3, adapter=False):
        super().__init__()
        self.init_atom_emb = MLP([99,width,width], bias=False, last_act=True)
        self.bond_emb_list = nn.ModuleList(MLP([24+99*2,width,width],bias=False,last_act=True) for _ in range(layers))
        self.atom_bond_block_list = nn.ModuleList(Block(width) for _ in range(layers))
        self.output_mlp = MLP([width,2*width,Q])
        self.global_head = MLP([width+8, width, Q])
        if adapter:
            self.adapter = MLP([width+8+4,width,Q])
            nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
        else:
            self.adapter = None

    def forward(self, batch):
        x = batch.x.float(); edge_attr = batch.edge_attr.float()
        h = self.init_atom_emb(x)
        i,j = batch.edge_index
        pair = torch.cat([x[i],edge_attr,x[j]],dim=1)
        for embed,block in zip(self.bond_emb_list,self.atom_bond_block_list):
            h = block(h,batch.edge_index,embed(pair))
        pooled = global_mean_pool(h,batch.batch)
        physics = batch.physics.reshape(-1,8)
        q = global_mean_pool(self.output_mlp(h),batch.batch) + self.global_head(torch.cat([pooled,physics],dim=1))
        if self.adapter is not None:
            q = q + self.adapter(torch.cat([pooled,physics,batch.cond.reshape(-1,4)],dim=1))
        return q.sort(dim=1).values

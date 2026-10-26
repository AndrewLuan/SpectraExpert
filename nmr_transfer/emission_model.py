"""功能概述：保留SpecGNN GINE骨干，将原子/环境位移先按核质量叠加，再生成分位数。
主要职责：NuclearSpecGNN.components输出无原子归属承诺的谱混合成分；forward按质量
CDF取128个离散分位数，避免逐原子分位数平均造成Wasserstein重心式谱形平滑。
13C每个碳贡献质量1；1H重原子最多6个无序H槽，每H质量1；碳上3/4H共享均值。
酰胺N上两个甲基采用对称双成分，单个甲基质量守恒，整体可表达两个非等价甲基峰；
成分不是cis/trans原子指派。shift_head读取节点/全局/物理结构/溶剂上下文。
make_model根据checkpoint的model参数建立原分布模型或本模型，供补充评测使用。
components(metadata=True)及forward(details=True)额外返回母节点与交换H掩码，供
¹H位点监督/失败诊断使用；默认接口及已有权重参数完全相同。weighted_quantiles可复用
核质量CDF，并支持按图或按母节点聚合；空核组保留越界数值占位。
无目标核时components返回空；forward的越界占位只用于数值失败计数，不代表物理峰。
structure_embedding支持复用冻结GINE节点表示；adjust_conditions默认恒等，供条件残差
子类调整位移/酰胺分裂logit。默认模型参数、发射与旧权重契约不变。make_model可
按condition_atom建立冻结骨干的条件消融模型；anchored_atom额外增加已知条件门控
响应，原条件模型参数仍冻结；deep_condition_atom增加零门控上下文块；
solvent_local_atom只增加明确苯/NMe2酰胺上的54参数局部残差，其他条件恒等；
solvent_message_atom在冻结深骨干上引入酰基环境和查询溶剂调制的消息层。
certified_group_atom通过h_equivalence_model为已认证结构生成固定物理环境槽，
未知结构保留原发射；registry路径和SHA随checkpoint记录。
preserved_group_atom使用h_preserved_group_model从原谱初始化组中心，只做必要
质量合并；零残差头允许各组独立修正，避免额外强制等间距或重置原酰胺分裂。
可选domain_conditioning将已知理论/实验域加入同容量残差头，默认None保持旧权重布局。
joint_reporting仅为preserved模型附加共享观测概率小头；完整物理成分接口不变。
显式count_decode启用h_count_constraint按结构认证H环境容量投影成分，
同步重算分位点并保留同次母节点/质量标签；默认关闭，未知计数恒等回退。
use_frozen_embedding默认True；端到端消融显式False
时重算GINE，不能误用旧骨干的结构缓存。amide_gap_scale默认1，允许只用
训练标注拟合酰胺甲基间距收缩，不改变中心、核质量或其他位移。
依赖原model.py的GINE参数命名与data.py图契约；不读取标签、坐标或源记录编号。
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from .model import SpecGNN,MLP


class NuclearSpecGNN(nn.Module):
    def __init__(self,nucleus,width=256,layers=3):
        super().__init__();self.nucleus=nucleus;self.adapter=True;self.amide_gap_scale=1.;self.use_frozen_embedding=True
        base=SpecGNN(width,layers)
        self.init_atom_emb=base.init_atom_emb;self.bond_emb_list=base.bond_emb_list
        self.atom_bond_block_list=base.atom_bond_block_list
        self.slots=6 if nucleus=='1H' else 1
        self.shift_head=MLP([2*width+12,width,self.slots])
        nn.init.normal_(self.shift_head[-1].weight,std=.005)
        nn.init.constant_(self.shift_head[-1].bias,.3 if nucleus=='1H' else .5)
        self.split_head=MLP([2*width+12,width,1])
        nn.init.zeros_(self.split_head[-1].weight);nn.init.constant_(self.split_head[-1].bias,-4.6)

    def structure_embedding(self,batch):
        if self.use_frozen_embedding and hasattr(batch,'frozen_embedding'):return batch.frozen_embedding.float()
        x=batch.x.float();src,tgt=batch.edge_index;h=self.init_atom_emb(x)
        pair=torch.cat([x[src],batch.edge_attr.float(),x[tgt]],1)
        for embed,block in zip(self.bond_emb_list,self.atom_bond_block_list):h=block(h,batch.edge_index,embed(pair))
        return h

    def adjust_conditions(self,shifts,split_logits,context,batch):
        return shifts,split_logits

    def components(self,batch,metadata=False):
        x=batch.x.float();src,tgt=batch.edge_index;h=self.structure_embedding(batch)
        pooled=global_mean_pool(h,batch.batch)
        context=torch.cat([h,pooled[batch.batch],batch.physics.reshape(-1,8)[batch.batch],batch.cond.reshape(-1,4)[batch.batch]],1)
        shifts,split_logits=self.adjust_conditions(self.shift_head(context),self.split_head(context),context,batch);z=x[:,:54].argmax(1)+1
        counts=torch.zeros(len(z),dtype=torch.long,device=z.device)
        hydrogen_edge=(z[src]==1)&(z[tgt]!=1)
        counts.index_add_(0,tgt[hydrogen_edge],torch.ones_like(tgt[hydrogen_edge]))
        attached=torch.zeros(len(z),dtype=torch.bool,device=z.device);attached[src[hydrogen_edge]]=True
        carbonyl=torch.zeros(len(z),dtype=torch.bool,device=z.device)
        carbonyl[src[(z[src]==6)&(z[tgt]==8)&(batch.edge_attr[:,2]>0)]]=True
        amide=torch.zeros_like(carbonyl);amide[src[(z[src]==7)&carbonyl[tgt]]]=True
        methyl=(z==6)&(counts==3);n_methyl=torch.zeros_like(counts)
        mc=methyl[src]&amide[tgt];n_methyl.index_add_(0,tgt[mc],torch.ones_like(tgt[mc]))
        split=torch.zeros_like(carbonyl);split[src[mc&(n_methyl[tgt]>=2)]]=True
        if self.nucleus=='1H':
            valid=(torch.arange(self.slots,device=z.device)[None,:]<counts[:,None])&(z[:,None]!=1)
            # H2等同核分子没有重原子母节点，两个核各自贡献一个成分，不能丢作空谱。
            valid |= ((z==1)&~attached)[:,None]&(torch.arange(self.slots,device=z.device)[None,:]==0)
            tied=(z==6)&((counts==3)|(counts==4))
            center=(shifts*valid).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)
            shifts=torch.where(tied[:,None],center,shifts)
        else:valid=(z==6)[:,None]
        delta=F.softplus(split_logits)*.5*self.amide_gap_scale
        offsets=torch.tensor([-1.,1.],device=z.device)
        values=shifts[:,:,None]+split[:,None,None]*delta[:,:,None]*offsets
        weights=torch.zeros_like(values);weights[:,:,0]=valid
        weights=torch.where(split[:,None,None],valid[:,:,None].float().expand_as(weights)*.5,weights)
        groups=batch.batch[:,None,None].expand_as(weights)
        active=weights>0
        result=(values[active],weights[active],groups[active])
        if metadata:
            owners=torch.arange(len(z),device=z.device)[:,None,None].expand_as(weights)
            exchange=((z==7)|(z==8)|(z==16))[:,None,None].expand_as(weights)
            return (*result,owners[active],exchange[active])
        return result

    def forward(self,batch,details=False):
        values,weights,groups,owners,exchange=self.components(batch,metadata=True)
        output=weighted_quantiles(values,weights,groups,batch.num_graphs)
        if not len(values):output=output+self.shift_head[-1].weight.sum()*0
        result=dict(quantiles=output,values=values,weights=weights,groups=groups,owners=owners,exchangeable=exchange)
        if getattr(self,'count_decode',False):
            from .h_count_constraint import project_details
            result=project_details(result,batch)
        return result if details else result['quantiles']


def weighted_quantiles(values,weights,groups,batches,points=128):
    if not len(values):return torch.full((batches,points),1e6,device=values.device)
    packed,_=to_dense_batch(values,groups,fill_value=1e6,batch_size=batches)
    mass,_=to_dense_batch(weights,groups,fill_value=0.,batch_size=batches)
    ordered,order=packed.sort(1);mass=mass.gather(1,order)
    cdf=mass.cumsum(1)/mass.sum(1,keepdim=True).clamp_min(1)
    q=((torch.arange(points,device=values.device)+.5)/points)[None,:].expand(batches,-1).contiguous()
    indices=torch.searchsorted(cdf.contiguous(),q).clamp_max(ordered.shape[1]-1)
    return ordered.gather(1,indices)


def make_model(config):
    if config.get('model') in ('certified_group_atom','preserved_group_atom'):
        from .h_equivalence_model import CertifiedGroupSpecGNN
        from .h_preserved_group_model import PreservedGroupSpecGNN
        cls=PreservedGroupSpecGNN if config.get('model')=='preserved_group_atom' else CertifiedGroupSpecGNN
        kwargs=dict(domain_conditioning=config.get('domain_conditioning'),joint_reporting=config.get('joint_reporting',False)) if config.get('model')=='preserved_group_atom' else {}
        model=cls(config['nucleus'],config['width'],config['layers'],config['condition_mode'],config['response_mode'],config['context_depth'],config['message_mode'],config['equivalence_registry_path'],config['equivalence_registry_sha256'],config['equivalence_mode'],**kwargs)
    elif config.get('model')=='solvent_message_atom':
        from .h_solvent_message import SolventMessageSpecGNN
        model=SolventMessageSpecGNN(config['nucleus'],config['width'],config['layers'],config['condition_mode'],config['response_mode'],config['context_depth'],config['message_mode'])
    elif config.get('model')=='solvent_local_atom':
        from .h_solvent_local import SolventLocalSpecGNN
        model=SolventLocalSpecGNN(config['nucleus'],config['width'],config['layers'],config['condition_mode'],config['response_mode'],config['context_depth'])
    elif config.get('model')=='deep_condition_atom':
        from .h_deep_context import DeepConditionSpecGNN
        model=DeepConditionSpecGNN(config['nucleus'],config['width'],config['layers'],config['condition_mode'],config['response_mode'],config['context_depth'])
    elif config.get('model')=='anchored_atom':
        from .h_response import AnchoredSpecGNN
        model=AnchoredSpecGNN(config['nucleus'],config['width'],config['layers'],config['condition_mode'],config['response_mode'])
    elif config.get('model')=='condition_atom':
        from .h_condmodel import ConditionSpecGNN
        model=ConditionSpecGNN(config['nucleus'],config['width'],config['layers'],config['condition_mode'])
    elif config.get('model')=='atom':model=NuclearSpecGNN(config['nucleus'],config['width'],config['layers'])
    else:model=SpecGNN(config['width'],config['layers'],adapter=config['mode']=='domain')
    if isinstance(model,NuclearSpecGNN):
        if config.get('count_decode',False) and config['nucleus']!='1H':raise ValueError('H_count_decoder_requires_proton_model')
        model.count_decode=bool(config.get('count_decode',False));model.amide_gap_scale=float(config.get('amide_gap_scale',1.));model.use_frozen_embedding=bool(config.get('use_frozen_embedding',True))
    return model

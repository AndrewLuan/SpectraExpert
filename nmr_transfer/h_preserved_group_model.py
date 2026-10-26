"""功能概述：用原发射分布初始化固定物理组，分离计数约束与额外峰形限制。
orbit_layout由认证H槽及二维原子对称秩构造组轨道；不读取谱值或标签。orbit_barycentres
按每母节点分配给轨道的H质量，将原成分分布切为同质量区间并取区间均值；局部槽
控制保留原离散谱，物理组仅作需要的质量合并。双精度积分及相对原点避免相等位移
出现浮点假分裂。映射稳定排序后才调用要求有序索引的to_dense_batch。
PreservedGroupSpecGNN复用CertifiedGroupSpecGNN的registry SHA/布局检查和溶剂消息
骨干，替换为零初始化逐组残差；输入基准中心、轨道内偏移和槽坐标，允许不同间距。
forward固定组数与核质量，保留母节点/交换标签，未知结构沿用原发射；同次重算
分位点供原训练与检索使用。只缓存结构轨道，不缓存带条件表示，无外部服务。
domain_conditioning可选shared/observed两种等容量对照；后者只向组残差头追加已知
理论/实验域标记，前者追加零。modality_features检查标记契约；缺失不静默当实验域。
域字段由Spectra按记录类型产生，候选批次统一experimental，不读取候选自己的谱。
joint_reporting可附加h_joint_conditions.ReportingHead；它单独建模观测报告概率，
不改变forward的完整物理发射。默认False保持已有checkpoint布局完全一致。
"""
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from rdkit import Chem
from .h_equivalence_model import CertifiedGroupSpecGNN,MAX_GROUPS
from .h_solvent_message import SolventMessageSpecGNN
from .h_conditions import CONDITION_DIM,LOCAL_DIM
from .h_acyl import ACYL_DIM
from .emission_model import weighted_quantiles
from .model import MLP


def orbit_layout(smiles,ids):
    ranks=list(Chem.CanonicalRankAtoms(Chem.MolFromSmiles(smiles),breakTies=False,includeChirality=True));signatures={};mapping=np.full(MAX_GROUPS,-1,np.int64);ordinal=np.zeros(MAX_GROUPS,np.int64);sizes=np.zeros(MAX_GROUPS,np.int64)
    for group in np.unique(ids[ids>=0]):
        parents,counts=np.unique(np.where(ids==group)[0],return_counts=True);signature=(int(counts.sum()),tuple(sorted((ranks[int(parent)],int(count)) for parent,count in zip(parents,counts))));signatures.setdefault(signature,[]).append(int(group))
    for orbit,numbers in enumerate(signatures.values()):
        for j,group in enumerate(numbers):mapping[group]=orbit;ordinal[group]=j;sizes[group]=len(numbers)
    return mapping,ordinal,sizes


def orbit_barycentres(details,parents,group_ids,group_orbits,ordinals,sizes,total_groups):
    """返回活动组号及其初始中心；所有原成分按实际父节点质量分配到结构轨道。"""
    values=details['values'];weights=details['weights'];owners=details['owners'];n=int(max(int(owners.max())+1,int(parents.max())+1));order=owners.argsort(stable=True);packed,_=to_dense_batch(values[order],owners[order],fill_value=0.,batch_size=n);mass,_=to_dense_batch(weights[order],owners[order],fill_value=0.,batch_size=n);parent_mass=mass.sum(1)
    edge=parents*total_groups+group_orbits[group_ids];unique,count=torch.unique(edge,sorted=True,return_counts=True);parent=unique//total_groups;orbit=unique%total_groups;fraction=count.to(mass.dtype)/parent_mass[parent].clamp_min(1);v=packed[parent];w=mass[parent]*fraction[:,None];active=w>0;v=v[active];w=w[active];o=orbit[:,None].expand_as(active)[active];unique_orbits,compact=torch.unique(o,sorted=True,return_inverse=True);order=compact.argsort(stable=True);v,_=to_dense_batch(v[order].double(),compact[order],fill_value=0.,batch_size=len(unique_orbits));w,_=to_dense_batch(w[order].double(),compact[order],fill_value=0.,batch_size=len(unique_orbits));valid=w>0;v,sorting=v.masked_fill(~valid,float('inf')).sort(1);w=w.gather(1,sorting);v=v.masked_fill(w<=0,0.);anchor=v[:,0];shifted=v-anchor[:,None];cdf=w.cumsum(1)/w.sum(1,keepdim=True);moment=(shifted*w).cumsum(1)/w.sum(1,keepdim=True)
    active_groups=group_ids.unique(sorted=True);rows=torch.searchsorted(unique_orbits,group_orbits[active_groups]);m=sizes[active_groups].double();j=ordinals[active_groups].double();bounds=torch.stack([j/m,(j+1)/m],1);selected_cdf=cdf[rows].contiguous();indices=torch.searchsorted(selected_cdf,bounds.contiguous()).clamp_max(cdf.shape[1]-1);before_cdf=torch.cat([torch.zeros_like(cdf[:,:1]),cdf[:,:-1]],1)[rows];before_moment=torch.cat([torch.zeros_like(moment[:,:1]),moment[:,:-1]],1)[rows];integrals=before_moment.gather(1,indices)+(bounds-before_cdf.gather(1,indices))*shifted[rows].gather(1,indices);centres=anchor[rows]+m*(integrals[:,1]-integrals[:,0]);return active_groups,centres.to(values.dtype)


class PreservedGroupSpecGNN(CertifiedGroupSpecGNN):
    def __init__(self,*args,domain_conditioning=None,joint_reporting=False,**kwargs):
        if domain_conditioning not in (None,'shared','observed'):raise ValueError('invalid_domain_conditioning')
        super().__init__(*args,**kwargs);width=args[1] if len(args)>1 else kwargs['width'];self.domain_conditioning=domain_conditioning;self.equivalence_head=MLP([2*width+12+CONDITION_DIM+LOCAL_DIM+ACYL_DIM+5+int(domain_conditioning is not None),128,1]);nn.init.zeros_(self.equivalence_head[-1].weight);nn.init.zeros_(self.equivalence_head[-1].bias);self._orbits=lru_cache(maxsize=240000)(lambda s:orbit_layout(s,self._layout(s)[0]))
        if joint_reporting:
            from .h_joint_conditions import ReportingHead
            self.reporting_head=ReportingHead()

    def modality_features(self,features,batch,which):
        if self.domain_conditioning is None:return features
        if not hasattr(batch,'modality_is_theory') or batch.modality_is_theory.numel()!=batch.num_graphs:raise ValueError('known_query_modality_required')
        modality=batch.modality_is_theory.reshape(-1).to(features.dtype)
        if not torch.all((modality==0)|(modality==1)):raise ValueError('invalid_query_modality')
        if self.domain_conditioning=='shared':modality=torch.zeros_like(modality)
        return torch.cat([features,modality[which,None]],1)

    def attach_orbits(self,batch):
        if hasattr(batch,'preserved_group_orbits'):return
        info=[self._orbits(s) for s in batch.structure_smiles];device=batch.x.device;batch.preserved_group_orbits=torch.as_tensor(np.stack([a[0] for a in info]),device=device);batch.preserved_group_ordinal=torch.as_tensor(np.stack([a[1] for a in info]),device=device);batch.preserved_group_sizes=torch.as_tensor(np.stack([a[2] for a in info]),device=device)

    def forward(self,batch,details=False):
        original=SolventMessageSpecGNN.forward(self,batch,details=True);self.attach_layout(batch);known=batch.equivalence_known
        if not known.any():return original if details else original['quantiles']
        self.attach_orbits(batch);h=self.structure_embedding(batch);pooled=global_mean_pool(h,batch.batch);ids=batch.equivalence_ids;active=ids>=0;parents=torch.arange(len(batch.x),device=h.device)[:,None].expand_as(ids)[active];gid=ids[active]+batch.batch[parents]*MAX_GROUPS;graphs=batch.batch[parents];total=batch.num_graphs*MAX_GROUPS;which=torch.arange(total,device=h.device)//MAX_GROUPS;orbits=batch.preserved_group_orbits.reshape(-1)+which*MAX_GROUPS;ordinal=batch.preserved_group_ordinal.reshape(-1);sizes=batch.preserved_group_sizes.reshape(-1)
        active_groups,base=orbit_barycentres(original,parents,gid,orbits,ordinal,sizes,total);centres=torch.zeros(total,device=h.device,dtype=base.dtype).index_copy(0,active_groups,base)
        def pool(x):
            sums=torch.zeros((total,x.shape[1]),device=x.device,dtype=x.dtype).index_add(0,gid,x[parents]);mass=torch.zeros(total,device=x.device,dtype=x.dtype).index_add(0,gid,torch.ones(len(gid),device=x.device,dtype=x.dtype));return sums/mass[:,None].clamp_min(1),mass
        grouped,mass=pool(h);local,_=pool(batch.local_environment.float());acyl,_=pool(batch.acyl_environment.float());valid_orbit=orbits[active_groups];sums=torch.zeros(total,device=h.device).index_add(0,valid_orbit,base);count=torch.zeros(total,device=h.device).index_add(0,valid_orbit,torch.ones_like(base));orbit_mean=sums[orbits.clamp_min(0)]/count[orbits.clamp_min(0)].clamp_min(1);slot=2*ordinal.float()/(sizes-1).clamp_min(1)-1;slot=torch.where(sizes>1,slot,torch.zeros_like(slot));features=torch.cat([grouped,pooled[which],batch.physics.reshape(-1,8)[which],batch.cond.reshape(-1,4)[which],batch.condition_vector.reshape(-1,CONDITION_DIM)[which],local,acyl,mass[:,None]/12,sizes[:,None]/6,centres[:,None],(centres-orbit_mean)[:,None],slot[:,None]],1);features=self.modality_features(features,batch,which);centres=centres+.2*self.equivalence_head(features)[:,0].tanh();emitted=centres[gid];keep=~known[original['groups']];z=batch.x[:,:54].argmax(1)+1;exchange=(z[parents]==7)|(z[parents]==8)|(z[parents]==16);additions=dict(values=emitted,weights=torch.ones_like(emitted),groups=graphs,owners=parents,exchangeable=exchange);result={key:torch.cat([original[key][keep],new]) for key,new in additions.items()};order=result['groups'].argsort(stable=True)
        for key in result:result[key]=result[key][order]
        expected=torch.zeros(batch.num_graphs,device=h.device).index_add(0,original['groups'],original['weights']);actual=torch.zeros_like(expected).index_add(0,result['groups'],result['weights'])
        if not torch.equal(expected,actual):raise ValueError('preserved_group_H_mass_changed')
        result['quantiles']=weighted_quantiles(result['values'],result['weights'],result['groups'],batch.num_graphs);return result if details else result['quantiles']

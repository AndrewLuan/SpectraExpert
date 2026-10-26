"""功能概述：将冻结的物理环境分区变成固定数量、固定核质量的¹H潜在发射槽。
equivalence_layout从canonical二维图和无标签registry生成每H槽的环境编号；
physical模式使用认证分区，local模式使用等容量头的局部H槽对照（甲基内部仍等价）。
相同二维环境的多个物理组以共享正间距的对称潜在槽区分，间距由结构和查询条件
学习，不按分子身份或实测位移赋值，也不声称潜在组一定具有可分辨的不同ppm。
CertifiedGroupSpecGNN继承溶剂条件消息骨干；未知/冲突结构精确保留原发射。已知
结构按组聚合节点特征与原预测母节点中心，等组共享中心，按原H质量展开成分。
保留母节点/交换标签，分位点由同次成分重算，供原位点损失和误差宽度使用。
结构布局只缓存SMILES/registry，不缓存条件化特征；batch需Spectra附加的
structure_smiles。registry文件SHA在构造时验证，原缓存、标签、谱与外部服务不改。
"""
import collections
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from rdkit import Chem
from .h_solvent_message import SolventMessageSpecGNN
from .h_conditions import CONDITION_DIM,LOCAL_DIM
from .h_acyl import ACYL_DIM
from .model import MLP
from .emission_model import weighted_quantiles
from .prepare import sha

MAX_GROUPS=192


def equivalence_layout(smiles,entry,mode='physical'):
    if mode not in ('physical','local'):raise ValueError('invalid_equivalence_mode')
    mol=Chem.MolFromSmiles(smiles);full=Chem.AddHs(mol);n=full.GetNumAtoms();ids=np.full((n,6),-1,np.int16);offsets=np.zeros(MAX_GROUPS,np.float32);duplicates=np.zeros(MAX_GROUPS,np.float32)
    if not entry or not entry.get('groups'):return ids,offsets,duplicates,False
    if Chem.MolToSmiles(mol,isomericSmiles=True)!=smiles:raise ValueError('equivalence_requires_canonical_smiles')
    expected={a.GetIdx():sum(x.GetAtomicNum()==1 for x in a.GetNeighbors()) for a in full.GetAtoms() if a.GetAtomicNum()!=1};groups=entry['groups']
    totals=collections.Counter()
    for g in groups:totals.update(g['parents'])
    if any(totals[i]!=c for i,c in expected.items()) or set(totals)-set(expected):raise ValueError('equivalence_H_mass_mismatch')
    if mode=='local':
        groups=[]
        for i,c in expected.items():
            if not c:continue
            if full.GetAtomWithIdx(i).GetAtomicNum()==6 and c>=3:groups.append(dict(parents={i:c},mass=c))
            else:groups.extend(dict(parents={i:1},mass=1) for _ in range(c))
    if len(groups)>MAX_GROUPS:raise ValueError('equivalence_group_limit')
    ranks=list(Chem.CanonicalRankAtoms(mol,breakTies=False,includeChirality=True));signatures=collections.defaultdict(list);used=collections.Counter()
    for number,g in enumerate(groups):
        signature=(g['mass'],tuple(sorted((ranks[int(i)],c) for i,c in g['parents'].items())));signatures[signature].append(number)
        for parent,count in g['parents'].items():
            parent=int(parent);begin=used[parent];end=begin+count
            if end>6:raise ValueError('equivalence_parent_slot_limit')
            ids[parent,begin:end]=number;used[parent]=end
    for numbers in signatures.values():
        m=len(numbers);offsets[numbers]=np.linspace(-1,1,m) if m>1 else 0;duplicates[numbers]=m
    return ids,offsets,duplicates,True


class CertifiedGroupSpecGNN(SolventMessageSpecGNN):
    def __init__(self,nucleus,width,layers,condition_mode,response_mode,context_depth,message_mode,registry_path,registry_sha256,equivalence_mode='physical'):
        super().__init__(nucleus,width,layers,condition_mode,response_mode,context_depth,message_mode)
        if nucleus!='1H':raise ValueError('equivalence_decoder_requires_protons')
        if sha(registry_path)!=registry_sha256:raise ValueError('equivalence_registry_SHA_mismatch')
        self.equivalence_registry=torch.load(registry_path,map_location='cpu',weights_only=False);self.equivalence_mode=equivalence_mode
        self.equivalence_head=MLP([2*width+12+CONDITION_DIM+LOCAL_DIM+ACYL_DIM+2,128,2]);nn.init.zeros_(self.equivalence_head[-1].weight);nn.init.zeros_(self.equivalence_head[-1].bias)
        with torch.no_grad():self.equivalence_head[-1].bias[1]=-4.6
        self._layout=lru_cache(maxsize=240000)(lambda s:equivalence_layout(s,self.equivalence_registry.get(s),self.equivalence_mode))

    def attach_layout(self,batch):
        if hasattr(batch,'equivalence_ids'):return
        names=batch.structure_smiles;ptr=batch.ptr.detach().cpu().tolist();arrays=[];offsets=[];duplicates=[];known=[]
        for j,s in enumerate(names):
            ids,o,d,k=self._layout(s)
            if len(ids)!=ptr[j+1]-ptr[j]:raise ValueError('equivalence_graph_node_contract')
            arrays.append(ids);offsets.append(o);duplicates.append(d);known.append(k)
        device=batch.x.device;batch.equivalence_ids=torch.as_tensor(np.concatenate(arrays),device=device,dtype=torch.long);batch.equivalence_offsets=torch.as_tensor(np.stack(offsets),device=device);batch.equivalence_duplicates=torch.as_tensor(np.stack(duplicates),device=device);batch.equivalence_known=torch.tensor(known,device=device)

    def forward(self,batch,details=False):
        original=super().forward(batch,details=True);self.attach_layout(batch);known=batch.equivalence_known
        if not known.any():return original if details else original['quantiles']
        h=self.structure_embedding(batch);pooled=global_mean_pool(h,batch.batch);ids=batch.equivalence_ids;active=ids>=0;parents=torch.arange(len(batch.x),device=h.device)[:,None].expand_as(ids)[active];gid=ids[active]+batch.batch[parents]*MAX_GROUPS;graphs=batch.batch[parents];total=batch.num_graphs*MAX_GROUPS
        def pool(x):
            sums=torch.zeros((total,x.shape[1]),device=x.device,dtype=x.dtype).index_add(0,gid,x[parents]);mass=torch.zeros(total,device=x.device,dtype=x.dtype).index_add(0,gid,torch.ones(len(gid),device=x.device,dtype=x.dtype));return sums/mass[:,None].clamp_min(1),mass
        grouped,mass=pool(h);local,_=pool(batch.local_environment.float());acyl,_=pool(batch.acyl_environment.float());which=torch.arange(total,device=h.device)//MAX_GROUPS
        features=torch.cat([grouped,pooled[which],batch.physics.reshape(-1,8)[which],batch.cond.reshape(-1,4)[which],batch.condition_vector.reshape(-1,CONDITION_DIM)[which],local,acyl,mass[:,None]/12,batch.equivalence_duplicates.reshape(-1,1)/6],1);change=self.equivalence_head(features)
        d=original;parent_mass=torch.zeros(len(h),device=h.device).index_add(0,d['owners'],d['weights']);parent_sum=torch.zeros_like(parent_mass).index_add(0,d['owners'],d['values']*d['weights']);parent_center=parent_sum/parent_mass.clamp_min(1)
        centers,_=pool(parent_center[:,None]);center=centers[:,0]+.2*change[:,0].tanh();gap=.5*F.softplus(change[:,1]);center=center+gap*batch.equivalence_offsets.reshape(-1)
        emitted=center[gid];keep=~known[d['groups']];z=batch.x[:,:54].argmax(1)+1;exchange=(z[parents]==7)|(z[parents]==8)|(z[parents]==16)
        result={}
        additions=dict(values=emitted,weights=torch.ones_like(emitted),groups=graphs,owners=parents,exchangeable=exchange)
        for key,new in additions.items():result[key]=torch.cat([d[key][keep],new])
        order=result['groups'].argsort(stable=True)
        for key in result:result[key]=result[key][order]
        expected=torch.zeros(batch.num_graphs,device=h.device).index_add(0,d['groups'],d['weights']);actual=torch.zeros_like(expected).index_add(0,result['groups'],result['weights'])
        if not torch.equal(expected,actual):raise ValueError('equivalence_emission_H_mass_changed')
        result['quantiles']=weighted_quantiles(result['values'],result['weights'],result['groups'],batch.num_graphs)
        return result if details else result['quantiles']

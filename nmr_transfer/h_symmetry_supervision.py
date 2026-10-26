"""功能概述：为二维不可区分的理论母节点提供无序轨道监督，消除任意原子编号的定向约束。
parent_orbits只由含立体标记的二维canonical图计算重原子自同构轨道，不使用位移。
apply_theory_orbits仅合并每个母节点H标签均完整的多母节点轨道；将其原始H标签
合成六点质量分位目标并映射到共同proxy owner，供grouped_objective的六点site_loss。
group_expected=6表示该proxy分位分布完整，不替换真实site_expected核质量。
group_site_weight在anchor保留被合并母节点数量，其他合并节点置零，以保持原位点
目标在图内平均的总权重；不把轨道合并顺带变成化学类别重新加权。
未报告任一母节点的轨道保持原逐母节点/部分匹配，不复制缺失位移。整谱q、位点
原标签、真实核数和物理分区不变。由ReplaySpectra显式theory_orbits开关调用；
默认所有旧流程不变，实验NMe2监督沿用原逻辑。它不修改预测发射或赋cis/trans标签。
"""
import collections
from functools import lru_cache
import torch
from rdkit import Chem
from .emission_model import weighted_quantiles


@lru_cache(maxsize=240000)
def parent_orbits(smiles):
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles));ranks=Chem.CanonicalRankAtoms(mol,breakTies=False,includeChirality=True);groups=collections.defaultdict(list)
    for atom,rank in zip(mol.GetAtoms(),ranks):
        if atom.GetAtomicNum()!=1 and any(a.GetAtomicNum()==1 for a in atom.GetNeighbors()):groups[int(rank)].append(atom.GetIdx())
    return tuple(tuple(v) for v in groups.values() if len(v)>1)


def apply_theory_orbits(g,smiles):
    for parents in parent_orbits(smiles):
        ids=torch.tensor(parents,dtype=torch.long,device=g.site_mask.device)
        if not bool((g.site_mask[ids].sum(1)==g.site_expected[ids]).all()):continue
        values=g.site_targets[ids][g.site_mask[ids]]
        if not len(values):continue
        target=weighted_quantiles(values,torch.ones_like(values),torch.zeros(len(values),dtype=torch.long,device=values.device),1,points=6)[0];anchor=parents[0]
        g.group_targets[ids]=0;g.group_mask[ids]=False;g.group_expected[ids]=0;g.group_targets[anchor]=target;g.group_mask[anchor]=True;g.group_expected[anchor]=6;g.group_owner[ids]=anchor;g.group_site_weight[ids]=0;g.group_site_weight[anchor]=len(parents)
    return g

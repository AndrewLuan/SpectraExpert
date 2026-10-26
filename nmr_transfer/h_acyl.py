"""功能概述：用二维结构区分甲酰胺与其他酰基环境，不读取任何实验位移。
acyl_environment返回显式H图逐节点的酰基类别与N取代特征；特征同时附着在
羰基C、酰胺N和N的其他直接重原子邻居，使N-甲基能识别另一侧的酰基。
多酰胺节点允许多热编码；甲酰、芳基、烯基、炔基、烷基和杂原子酰基分别编码。
carbonyl_h_sites供报告过程区分甲酰胺CH与普通醛CH；返回canonical显式H母节点
索引，不把化学类别解释成实测峰完整性。由条件消息模型、支持清点和观测模型
共用；lru缓存只保存不可变结构特征，调用方不得就地修改返回数组。
"""
from functools import lru_cache
import numpy as np
from rdkit import Chem

ACYL_NAMES=['amide','formyl','alkyl','aryl','vinyl','alkynyl','hetero_acyl','amide_NH','amide_N_tertiary']
ACYL_DIM=len(ACYL_NAMES)


@lru_cache(maxsize=160000)
def acyl_environment(smiles):
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles));out=np.zeros((mol.GetNumAtoms(),ACYL_DIM),np.float32)
    for c in mol.GetAtoms():
        if c.GetAtomicNum()!=6 or not any(b.GetBondType()==Chem.BondType.DOUBLE and b.GetOtherAtom(c).GetAtomicNum()==8 for b in c.GetBonds()):continue
        ns=[a for a in c.GetNeighbors() if a.GetAtomicNum()==7]
        for n in ns:
            v=np.zeros(ACYL_DIM,np.float32);v[0]=1
            for a in c.GetNeighbors():
                if a.GetIdx()==n.GetIdx() or (a.GetAtomicNum()==8 and mol.GetBondBetweenAtoms(c.GetIdx(),a.GetIdx()).GetBondType()==Chem.BondType.DOUBLE):continue
                if a.GetAtomicNum()==1:tag='formyl'
                elif a.GetAtomicNum()!=6:tag='hetero_acyl'
                elif a.GetIsAromatic():tag='aryl'
                elif a.GetHybridization()==Chem.HybridizationType.SP:tag='alkynyl'
                elif a.GetHybridization()==Chem.HybridizationType.SP2:tag='vinyl'
                else:tag='alkyl'
                v[ACYL_NAMES.index(tag)]=1
            v[ACYL_NAMES.index('amide_NH' if any(a.GetAtomicNum()==1 for a in n.GetNeighbors()) else 'amide_N_tertiary')]=1
            ids={c.GetIdx(),n.GetIdx()}|{a.GetIdx() for a in n.GetNeighbors() if a.GetAtomicNum()!=1}
            for i in ids:out[i]=np.maximum(out[i],v)
    out.flags.writeable=False
    return out


def carbonyl_h_sites(smiles):
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles));out={}
    for c in mol.GetAtoms():
        if c.GetAtomicNum()!=6 or not any(a.GetAtomicNum()==1 for a in c.GetNeighbors()):continue
        if not any(b.GetBondType()==Chem.BondType.DOUBLE and b.GetOtherAtom(c).GetAtomicNum()==8 for b in c.GetBonds()):continue
        out[c.GetIdx()]='formamide' if any(a.GetAtomicNum()==7 for a in c.GetNeighbors()) else 'aldehyde'
    return out

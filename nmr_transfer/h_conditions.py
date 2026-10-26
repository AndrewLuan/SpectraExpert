"""功能概述：把¹H记录条件与候选局部环境转为不含谱标签的显式特征。
solvent_features识别常见溶剂/混合物，并使用RDKit分子计数描述符共享稀有溶剂信息；
未知、未识别与已知溶剂保持区分。encode_conditions只在明确水相且pH可解析时
启用pH基函数，非水相记录中的pH原值仅供审计，不默认成水相pH。
名称匹配支持MeOD、(CD3)2SO、CD3COCD3等明确别名和Unicode下标；
不把CDC13、CDCI3等含义不明拼写自动修成氯仿，也不从实验位移推测溶剂。
condition_from_old只恢复旧4维条件中已有的溶剂信息，不能恢复被丢弃的苯等名称。
local_environment从候选二维图提取CHn、酰胺甲基、邻杂原子及到羧基/非酰胺N的距离；
这些是结构描述符，不断言实测质子化状态或固定pKa。ConditionsSpectra在旧Spectra
上添加条件/局部特征，并为HMDB训练使用冻结0.08ppm簇中心与水相可见性标记。
检索时condition_vector必须来自查询记录；不能按每个候选自己的实验记录查条件。
ConditionsSpectra另附h_acyl纯结构特征，供溶剂消息模型区分甲酰与其他酰基；
不改变旧24维local_environment或既有checkpoint接口。
依赖data/h_labels及既有观测处理，未知条件保持掩码；无外部API和文件写入。
"""
import re
from functools import lru_cache
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Lipinski
from .data import condition
from .h_labels import make_key
from .h_observed import cluster_lines
from .data import quantiles
from .train import Spectra
from .h_acyl import acyl_environment

SOLVENTS=['unknown','water','chloroform','dmso','benzene','acetone','methanol','acetonitrile','dichloromethane','ethanol','carbon_tetrachloride','cyclohexane','neat','other']
SOLVENT_SMILES={'water':'O','chloroform':'ClC(Cl)Cl','dmso':'CS(C)=O','benzene':'c1ccccc1','acetone':'CC(C)=O','methanol':'CO','acetonitrile':'CC#N','dichloromethane':'ClCCl','ethanol':'CCO','carbon_tetrachloride':'ClC(Cl)(Cl)Cl','cyclohexane':'C1CCCCC1'}
SOLVENT_DIM=len(SOLVENTS)+7
PH_DIM=6
CONDITION_DIM=SOLVENT_DIM+PH_DIM
LOCAL_NAMES=['sp3_C','aromatic','sp2_C','carbonyl_C','N_H','O_H','S_H','CH3','CH2','CH','amide_N_methyl','neighbor_O','neighbor_N','neighbor_S','neighbor_carbonyl','neighbor_aromatic','formal_charge','carboxyl_proximity','nonamide_N_proximity','phenol_proximity','ring','amide_N','aldehyde','hetero_aromatic_neighbor']
LOCAL_DIM=len(LOCAL_NAMES)


@lru_cache(maxsize=128)
def solvent_features(value):
    text=str(value or '').strip().lower().translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉','0123456789'));tags=[]
    tests={'water':['water','d2o','h2o'],'chloroform':['cdcl3','chcl3','chloroform'],'dmso':['dmso','(cd3)2so','cd3socd3','dimethylsulfoxide','dimethyl sulfoxide'],'benzene':['benzene','c6d6','c6h6'],'acetone':['acetone','cd3cocd3','(cd3)2co'],'methanol':['methanol','cd3od','meod','meoh','ch3oh','me-d3-od'],'acetonitrile':['acetonitrile','cd3cn','mecn'],'dichloromethane':['cd2cl2','dichloromethane'],'ethanol':['ethanol','c2d5od','etoh','etod'],'carbon_tetrachloride':['ccl4','carbon tetrachloride'],'cyclohexane':['cyclohexane'],'neat':['neat']}
    for name,tokens in tests.items():
        if name=='ethanol' and 'methanol' in text:continue
        if any(t in text for t in tokens):tags.append(name)
    v=np.zeros(SOLVENT_DIM,np.float32)
    if not text:v[0]=1;return v,[]
    if not tags:v[SOLVENTS.index('other')]=1;v[-1]=1;return v,['other']
    for name in tags:
        v[SOLVENTS.index(name)]=1/len(tags)
        if name=='neat':continue
        mol=Chem.MolFromSmiles(SOLVENT_SMILES[name]);atoms=list(mol.GetAtoms());n=max(1,len(atoms))
        desc=np.array([n/8,sum(a.GetIsAromatic() for a in atoms)/n,sum(a.GetAtomicNum() not in [1,6] for a in atoms)/n,Lipinski.NumHDonors(mol)/2,Lipinski.NumHAcceptors(mol)/2,sum(a.GetFormalCharge() for a in atoms)/3],np.float32)
        v[len(SOLVENTS):len(SOLVENTS)+6]+=desc/len(tags)
    v[-1]=1;return v,tags


def parse_ph(value):
    text=str(value or '').strip()
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',text):return None
    p=float(text);return p if 0<=p<=14 else None


def encode_conditions(record):
    c=record.get('experimental_conditions') or record.get('conditions') or {};raw=record.get('solvent') or c.get('solvent') or ''
    solvent,tags=solvent_features(raw);ph=parse_ph(c.get('pH',c.get('ph')));p=np.zeros(PH_DIM,np.float32)
    if 'water' in tags and ph is not None:p=np.array([1,(ph-7)/7,np.tanh((ph-4)/2),np.tanh((ph-7)/2),np.tanh((ph-10)/2),0],np.float32)
    return dict(vector=np.concatenate([solvent,p]),old_cond=np.array(condition(record),np.float32),solvent=str(raw),family='+'.join(tags) or 'unknown',raw_pH=c.get('pH',c.get('ph')),pH=ph if p[0] else None,pH_used=bool(p[0]),metadata_known=bool(raw))


def condition_from_old(cond):
    c=np.asarray(cond);names=[s for i,s in [(1,'Water'),(2,'CDCl3'),(3,'DMSO')] if c[i]>0]
    result=encode_conditions(dict(solvent='+'.join(names)));result['old_cond']=c.copy();result['recovered_from_old_mask']=True;return result


def local_environment(smiles):
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles));n=mol.GetNumAtoms();out=np.zeros((n,LOCAL_DIM),np.float32)
    carbonyl={a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==6 and any(b.GetBondType()==Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetAtomicNum()==8 for b in a.GetBonds())}
    amides={a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==7 and any(x.GetIdx() in carbonyl for x in a.GetNeighbors())}
    carboxyl={a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() in carbonyl and any(b.GetBondType()==Chem.BondType.SINGLE and b.GetOtherAtom(a).GetAtomicNum()==8 for b in a.GetBonds())}
    nitrogens={a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==7 and a.GetIdx() not in amides}
    phenols={a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==8 and any(x.GetIsAromatic() for x in a.GetNeighbors())}
    distance=Chem.GetDistanceMatrix(mol)
    for a in mol.GetAtoms():
        i=a.GetIdx();z=a.GetAtomicNum();adj=list(a.GetNeighbors());h=sum(x.GetAtomicNum()==1 for x in adj);zs={x.GetAtomicNum() for x in adj};carbon=z==6
        near=lambda ids:1/(1+min(distance[i,j] for j in ids)) if ids else 0
        out[i]=[carbon and a.GetHybridization()==Chem.HybridizationType.SP3,a.GetIsAromatic(),carbon and a.GetHybridization()==Chem.HybridizationType.SP2,i in carbonyl,z==7 and h>0,z==8 and h>0,z==16 and h>0,carbon and h==3,carbon and h==2,carbon and h==1,carbon and h==3 and any(x.GetIdx() in amides for x in adj),8 in zs,7 in zs,16 in zs,any(x.GetIdx() in carbonyl for x in adj),any(x.GetIsAromatic() for x in adj),a.GetFormalCharge()/3,near(carboxyl),near(nitrogens),near(phenols),a.IsInRing(),i in amides,i in carbonyl and h>0,any(x.GetIsAromatic() and x.GetAtomicNum() not in [1,6] for x in adj)]
    return out


class ConditionsSpectra(Spectra):
    def __init__(self,rows,graphs,nucleus,metadata,**kwargs):
        super().__init__(rows,graphs,nucleus,**kwargs);self.metadata=metadata
    def __getitem__(self,i):
        r=self.rows[i];g=super().__getitem__(i);meta=self.metadata.get(make_key(r))
        g.acyl_environment=torch.tensor(acyl_environment(r['smiles']).copy())
        if meta is None:meta=condition_from_old(r['cond'])
        v=r.get('condition_vector',meta['vector']);g.condition_vector=torch.tensor(np.asarray(v,np.float32)).reshape(1,-1)
        g.line_water=torch.tensor([r['source']=='HMDB' and g.condition_vector[0,SOLVENTS.index('water')]>0])
        if r['source']=='HMDB' and 'line_centers' in meta:
            g.q=torch.tensor(quantiles(meta['line_centers'])/10).reshape(1,-1);g.supervised_q=g.q.clone()
        return g

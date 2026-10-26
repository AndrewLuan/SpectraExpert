"""功能概述：对实验甲基内部归属冲突建立可审计的位点监督可靠性对照。
conflicts只检查同一普通CH3母节点至少两个已给标签的跨度；不根据位移重排原子。
apply_quality只在train实验记录上可选关闭整记录的母节点监督，保留原q、观测掩码、
完整H数量证据和整谱分布辅助。未知/缺失标签不填零，理论单构象位移不使用此规则。
该规则标记归属不一致，不宣称数据库记录或原结构必然错误；原始模式用于配对消融。
返回每来源标记计数和明细，不改数据文件、候选或test。由h_count_curriculum调用。
"""
import collections
import numpy as np
from rdkit import Chem
from .h_labels import make_key

MODES=('original','methyl_record_mask')


def conflicts(smiles,annotation,tolerance=.05):
    if not annotation.get('groups'):return []
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles));result=[]
    for parent,values in annotation.get('groups',{}).items():
        atom=mol.GetAtomWithIdx(int(parent));hs=[h for h in atom.GetNeighbors() if h.GetAtomicNum()==1]
        if atom.GetAtomicNum()!=6 or len(hs)!=3 or any(h.GetIsotope() not in (0,1) for h in hs) or len(values)<2:continue
        xs=np.asarray(values,float)
        if not np.isfinite(xs).all():raise ValueError('nonfinite_experimental_site_labels')
        span=float(xs.max()-xs.min())
        if span>tolerance:result.append(dict(parent=int(parent),reported_H=len(xs),span_ppm=span))
    return result


def apply_quality(labels,rows,mode,tolerance=.05):
    if mode not in MODES:raise ValueError('invalid_site_quality_mode')
    flagged=[];counts=collections.Counter()
    for r in rows:
        if r['domain']!='experimental' or r['split']!='train':raise ValueError('quality_requires_experimental_train')
        key=make_key(r);old=labels.get(key,{})
        issues=conflicts(r['smiles'],old,tolerance)
        if not issues:continue
        counts[r['source']]+=1;flagged.append(dict(source=r['source'],record_id=r['record_id'],smiles=r['smiles'],conflicting_methyl_sites=issues))
        if mode=='methyl_record_mask':labels[key]=dict(old,groups={},site_assignment_quality='methyl_label_conflict',trusted_site_H=0,quality_threshold_ppm=tolerance)
    return dict(mode=mode,tolerance_ppm=tolerance,flagged_records=len(flagged),by_source=dict(counts),records=flagged,scope='training site supervision only; source spectra and completeness evidence retained')

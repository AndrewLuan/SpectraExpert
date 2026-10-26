"""功能概述：把¹H标签的坐标顺序与重原子逐位补H顺序作为两种显式假设审计。
label_parents默认复用h_labels.source_groups的坐标契约；heavy_parent_order则仅读取
源重原子顺序与结构H数，将源H标签行映到逐母节点补H次序，不用位移决定映射。
该模式是待原始数据验证的标签顺序假设，不修坐标/理论原子顺序，不声称原子级真值。
annotation保留原观测掩码、位移多重集和完整H标志，仅改变训练/诊断母节点侧表。
main在train/val比较两种解释与独立理论母节点中心的偏差及甲基标签一致性；
test标签不进入本审计。不修改原谱/候选/原标签，无数值偏移拟合或外部服务。
"""
import argparse,collections,json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from .data import identity
from .h_labels import source_groups,make_key
from .prepare import environment,sha

MODES=('coordinate_order','heavy_parent_order')


def label_parents(r,mode='coordinate_order'):
    smi,original=source_groups(r)
    if mode=='coordinate_order':return smi,original
    if mode!='heavy_parent_order':raise ValueError('invalid_H_label_order')
    raw=Chem.AddHs(Chem.MolFromSmiles(r['smiles']));canonical=Chem.AddHs(Chem.MolFromSmiles(smi));mapping=canonical.GetSubstructMatch(raw,useChirality=True);order=r['atom_order']['source_to_canonical'];tokens=r['atom_order']['atomic_numbers'];expanded=[]
    if [raw.GetAtomWithIdx(i).GetAtomicNum() for i in order]!=tokens:raise ValueError('label_order_source_elements')
    for node in order:
        a=raw.GetAtomWithIdx(node)
        if a.GetAtomicNum()==1:continue
        hs=[x for x in a.GetNeighbors() if x.GetAtomicNum()==1 and x.GetIsotope() in (0,1)]
        if hs:expanded.extend([(mapping[node],environment(hs[0]))]*len(hs))
    indices=sorted(original)
    if indices!=list(range(len(order)-len(indices),len(order))):raise ValueError('H_label_parent_order_requires_H_suffix')
    if len(indices)!=len(expanded):raise ValueError('label_order_H_count')
    return smi,dict(zip(indices,expanded))


def annotation(r,mode):
    smi,parents=label_parents(r,mode);sites=collections.defaultdict(list)
    for i,(v,m) in enumerate(zip(r['atom_targets'],r['atom_target_mask'])):
        if m:
            if i not in parents or v is None:raise ValueError('invalid_H_label_mask')
            sites[parents[i][0]].append(float(v))
    return smi,dict(groups=dict(sites),observed_H=sum(map(len,sites.values())),structural_H=len(parents),full_atom_labels=sum(map(len,sites.values()))==len(parents),label_order_hypothesis=mode)


def main():
    p=argparse.ArgumentParser()
    for k in ['source','theory','out']:p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);theory={str(r['record_id']):r for r in map(json.loads,open(a.theory))};records=[];errors=collections.Counter();annotations={m:{} for m in MODES}
    try:
        for r in map(json.loads,open(a.source)):
            if r.get('nucleus')!='1H':continue
            try:
                smi,group,split=identity(r['smiles'])
                if split=='test':continue
                key=('experimental',r['source_ref'],str(r['record_id']),smi);mol=Chem.AddHs(Chem.MolFromSmiles(smi));methyl={atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum()==6 and sum(h.GetAtomicNum()==1 for h in atom.GetNeighbors())==3};order=source_groups(r)[1];reference=collections.defaultdict(list)
                if str(r['record_id']) in theory:
                    tr=theory[str(r['record_id'])]
                    if identity(tr['smiles'])[0]!=smi:raise ValueError('theory_structure_mismatch')
                    for i,v in zip(tr['value']['atom_index'],tr['value']['x']):reference[order[i-1][0]].append(float(v))
                trial={}
                for mode in MODES:
                    _,ann=annotation(r,mode);annotations[mode][key]=ann;xs=[v for values in ann['groups'].values() for v in values];residual=[abs(v-np.mean(reference[p])) for p,values in ann['groups'].items() if p in reference for v in values];spans=[max(values)-min(values) for p,values in ann['groups'].items() if p in methyl and len(values)>=2];trial[mode]=dict(observed_H=len(xs),spectrum_multiset=sorted(xs),theory_parent_center_MAE=float(np.mean(residual)) if residual else None,methyl_spans=spans)
                if trial[MODES[0]]['spectrum_multiset']!=trial[MODES[1]]['spectrum_multiset']:raise ValueError('label_reordering_changed_spectrum')
                for t in trial.values():t.pop('spectrum_multiset')
                records.append(dict(source=r['source_ref'],split=split,record_id=r['record_id'],smiles=smi,group=group,trials=trial))
            except (ValueError,KeyError,IndexError) as exc:errors[str(exc)]+=1
        summary={}
        for source,split in sorted({(r['source'],r['split']) for r in records}):
            subset=[r for r in records if r['source']==source and r['split']==split];v={}
            for mode in MODES:
                residual=[r['trials'][mode]['theory_parent_center_MAE'] for r in subset if r['trials'][mode]['theory_parent_center_MAE'] is not None];spans=[x for r in subset for x in r['trials'][mode]['methyl_spans']];v[mode]=dict(records=len(subset),paired_theory_records=len(residual),mean_theory_parent_MAE=float(np.mean(residual)) if residual else None,median_theory_parent_MAE=float(np.median(residual)) if residual else None,methyl_groups=len(spans),methyl_span_over_005=int(sum(x>.05 for x in spans)),methyl_span_over_02=int(sum(x>.2 for x in spans)))
            summary[source+'|'+split]=v
        for mode in MODES:torch.save(annotations[mode],out/(mode+'.pt'))
        with (out/'records.jsonl').open('x') as f:
            for r in records:f.write(json.dumps(r)+'\n')
        report=dict(summary=summary,errors=dict(errors),test_labels_used=False,interpretation='heavy_parent_order is a structural indexing hypothesis, not source-verified reassignment',source_sha256={str(f):sha(f) for f in [a.source,a.theory]},artifact_sha256={f.name:sha(f) for f in out.glob('*.pt')});(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'EXIT_CODE').write_text('0\n');print(json.dumps(report),flush=True)
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

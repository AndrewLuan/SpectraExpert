"""功能概述：为¹H失败诊断和位点监督建立独立标签侧表，不改原始数据缓存。
source_groups将源原子索引经atom_order和二维图同构映射到canonical图的H母节点；
make_key区分域/来源/记录/结构。main仅读取既有NMRShiftDB标注、重算谱和冻结积分缓存，
保存每母节点已标注位移、全H标注证据、原始积分及数据一致性标志，不引入坐标输入。
重算标签使用上一轮仅train拟合的全局/环境校准。对称映射保留源质量标志，映射不符拒绝。
完整逐H标注仅用于训练监督与诊断；检索不能借用真实结构来推断查询完整性或缺失原子。
main输出新annotations.pt/manifest.json，拒绝覆盖；依赖data.identity及prepare.environment。
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from .data import identity
from .prepare import environment,sha


def make_key(r):
    return (r['domain'],r['source'],str(r['record_id']),r['smiles'])


def source_groups(record):
    raw=Chem.AddHs(Chem.MolFromSmiles(record['smiles']))
    smi,_,_=identity(record['smiles']);canonical=Chem.AddHs(Chem.MolFromSmiles(smi))
    # 显式H也参与同构，防止芳香N的普通子结构匹配将[nH]母节点映到无H的n。
    mapping=canonical.GetSubstructMatch(raw,useChirality=True)
    if len(mapping)!=raw.GetNumAtoms():raise ValueError('no_explicit_H_graph_mapping')
    heavy_ids=[a.GetIdx() for a in raw.GetAtoms() if a.GetAtomicNum()!=1]
    parents={i:mapping[i] for i in heavy_ids};order=record['atom_order']['source_to_canonical']
    numbers=record['atom_order']['atomic_numbers']
    if [raw.GetAtomWithIdx(i).GetAtomicNum() for i in order]!=numbers:raise ValueError('source_order_mismatch')
    result={}
    for source,node in enumerate(order):
        atom=raw.GetAtomWithIdx(node)
        if atom.GetAtomicNum()!=1:continue
        if atom.GetDegree()!=1:raise ValueError('nonterminal_H')
        parent=atom.GetNeighbors()[0].GetIdx()
        if parent not in parents:raise ValueError('homonuclear_H_assignment')
        result[source]=(parents[parent],environment(atom))
    return smi,result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    path=Path('/data/SpecGPT/dataset/NMR_exprimental/atom_assigned/data.jsonl')
    labels={};mapping={};counts=collections.Counter();errors=collections.Counter()
    for line in path.open():
        r=json.loads(line)
        if r['nucleus']!='1H':continue
        try:
            smi,order=source_groups(r);sites=collections.defaultdict(list)
            for i,(v,m) in enumerate(zip(r['atom_targets'],r['atom_target_mask'])):
                if m:
                    if i not in order:raise ValueError('label_on_non_H')
                    sites[order[i][0]].append(float(v))
            if not sites:continue
            meta=dict(groups=dict(sites),observed_H=sum(map(len,sites.values())),structural_H=len(order),
                      full_atom_labels=sum(map(len,sites.values()))==len(order),
                      source_mapping_quality=r.get('quality',{}),scope='atom annotation completeness, not acquisition-window evidence')
            key=('experimental',r['source_ref'],str(r['record_id']),smi);labels[key]=meta
            mapping[str(r['record_id'])]=(smi,order)
            counts[r['source_ref']+'|'+('full' if meta['full_atom_labels'] else 'partial')]+=1
        except (ValueError,KeyError,IndexError) as exc:errors[str(exc)]+=1
    fit=json.loads((base/'prepared_coverage/domain_shift.json').read_text())['fits']['recalculated|1H']
    theoretical=Path('/data/SpecGPT/benchmarks/theoretical_1H_NMR.jsonl')
    for line in theoretical.open():
        r=json.loads(line);entry=mapping.get(str(r['record_id']))
        if not entry:continue
        smi,order=entry
        if identity(r['smiles'])[0]!=smi:errors['theory_structure_mismatch']+=1;continue
        sites=collections.defaultdict(list)
        try:
            for i,v in zip(r['value']['atom_index'],r['value']['x']):
                parent,env=order[i-1];tag=fit['environment_residuals'].get(env,{})
                residual=tag.get('median_ppm',0.) if tag.get('atoms',0)>=100 else 0.
                sites[parent].append(fit['affine_scale']*float(v)+fit['affine_offset_ppm']+residual)
            labels['theory','recalculated',str(r['record_id']),smi]=dict(groups=dict(sites),observed_H=sum(map(len,sites.values())),structural_H=len(order),full_atom_labels=sum(map(len,sites.values()))==len(order),calibration='previous train-only atom-aligned fit')
            counts['recalculated|mapped']+=1
        except KeyError:errors['theory_source_index_missing']+=1
    integral_path=Path('/home/user/IRfingerprints/SpectrExpert/outputs/nmr_forward_20260922/prepared_v2_mod48/cache.pt')
    for r in torch.load(integral_path,map_location='cpu',weights_only=False,mmap=True):
        try:smi,_,_=identity(r['smiles'])
        except ValueError:continue
        weights=np.asarray(r['weights']);expected=int(r['protons']);total=float(weights.sum())
        # 计数相符仅提供积分一致性，不据此宣称测量完整，也不用于候选过滤。
        integral_consistent=bool(np.allclose(weights,weights.round(),atol=.001) and abs(total-expected)<.001)
        labels['experimental',str(r['source']),str(r['record_id']),smi]=dict(groups={},
            observed_integral=total,structural_H=expected,integral_consistent=integral_consistent,
            full_atom_labels=False,xs=r['xs'],weights=r['weights'],flags=r.get('flags',[]))
        counts[str(r['source'])+'|'+('integral_consistent' if integral_consistent else 'integral_other')]+=1
    torch.save(labels,out/'annotations.pt')
    manifest=dict(records=len(labels),counts=dict(counts),errors=dict(errors),source_sha256={str(x):sha(x) for x in [path,theoretical,integral_path]},
                  labels_sha256=sha(out/'annotations.pt'),fit='previous train partition only; no fitting performed here',
                  retrieval_boundary='true-structure atom annotations are diagnostic/training only; never passed to candidate scorer')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()

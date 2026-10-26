"""功能概述：审计确定性物理环境数与现有¹H谱监督口径，生成无标签结构分组侧表。
build_registry只读已冻结structure/prediction，拒绝同canonical结构跨记录计数或母节点
分区冲突；parent_groups以RDKit序列化顺序映射重原子，不猜测CH2槽的原子归属。
main按来源/分区报告已有计数覆盖；理论位移差异只检查train/val，不读test谱数值。
输出registry.pt、audit.json和退出码至独占目录；原谱、候选、缓存均只读。
侧表保留完整与部分母节点分组及质量，供h_count_constraint训练使用；
它声明模型假设下的化学环境，不把静态DFT小数去重数当溶液谱峰真值。
依赖prepare.identity、RDKit及现有prepared缓存；不调用外部服务。
"""
import argparse,ast,collections,json,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from .data import identity
from .prepare import sha


def parent_groups(s,groups):
    raw=Chem.MolFromSmiles(s['source_smiles']);full=Chem.AddHs(raw);order=s['source_to_canonical']
    if sorted(order)!=list(range(full.GetNumAtoms())) or [full.GetAtomWithIdx(i).GetAtomicNum() for i in order]!=s['tokens']:raise ValueError('source_atom_contract')
    if any(a.GetAtomicNum()==1 for a in raw.GetAtoms()):raise ValueError('explicit_H_unsupported')
    canonical=Chem.MolToSmiles(raw,isomericSmiles=True);out_order=ast.literal_eval(raw.GetProp('_smilesAtomOutputOrder'));parents={old:new for new,old in enumerate(out_order)}
    expected=[i for i,j in enumerate(order) if full.GetAtomWithIdx(j).GetAtomicNum()==1 and full.GetAtomWithIdx(j).GetIsotope() in (0,1)]
    flat=[i for g in groups for i in g]
    if sorted(flat)!=expected or len(flat)!=len(set(flat)):raise ValueError('incomplete_H_partition')
    mol=Chem.MolFromSmiles(canonical);result=[]
    for g in groups:
        counts=collections.Counter()
        for i in g:
            atom=full.GetAtomWithIdx(order[i])
            if atom.GetDegree()!=1:raise ValueError('nonterminal_H')
            counts[parents[atom.GetNeighbors()[0].GetIdx()]]+=1
        result.append(dict(parents=dict(counts),mass=len(g),complete=all(mol.GetAtomWithIdx(i).GetTotalNumHs()==n for i,n in counts.items())))
    result.sort(key=lambda g:tuple(sorted(g['parents'].items())))
    return canonical,result


def build_registry(root):
    structures={r['record_id']:r for r in map(json.loads,open(root/'structures.jsonl'))};registry={};conflicts=set();errors=collections.Counter();records={}
    for r in map(json.loads,open(root/'predictions.jsonl')):
        if r['record_id'] not in structures:continue
        s=structures[r['record_id']]
        try:
            canonical=identity(s['source_smiles'])[0];counts={n:v['count'] for n,v in r['nuclei'].items() if v.get('status')=='predicted'}
            groups=None
            if '1H' in counts:
                mapped,groups=parent_groups(s,r['nuclei']['1H']['groups'])
                if mapped!=canonical:raise ValueError('canonical_contract')
            entry=dict(counts=counts,groups=groups)
            if canonical in registry and registry[canonical]!=entry:conflicts.add(canonical)
            registry[canonical]=entry;records[str(r['record_id'])]=(canonical,r['nuclei'].get('1H',{}).get('groups'))
        except (ValueError,KeyError,IndexError) as exc:errors[str(exc)]+=1
    for s in conflicts:registry.pop(s,None)
    return registry,records,dict(structures=len(structures),conflicting_structures=len(conflicts),mapping_errors=dict(errors))


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--registry',required=True);p.add_argument('--bank',required=True);p.add_argument('--theory',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(4)
    try:
        registry,records,report=build_registry(Path(a.registry));torch.save(registry,out/'registry.pt');print('registry',len(registry),report,flush=True)
        cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);coverage=collections.defaultdict(collections.Counter);allowed={}
        for r in cache['rows']:
            if r['nucleus']!='1H':continue
            key='|'.join([r['domain'],r['source'],r['split']]);v=coverage[key];v['records']+=1;entry=registry.get(r['smiles'],{});k=entry.get('counts',{}).get('1H',0)
            if k:
                v['certified_records']+=1;v['complete_parent_partition_records']+=int(all(g['complete'] for g in entry['groups']));v['encoded_count_matches']+=int(abs(float(r['physics'][5])*30-k)<1e-3)
            if r['domain']=='theory' and r['source']=='recalculated' and r['split'] in ('train','val'):allowed[str(r['record_id'])]=r['split']
        report['coverage']={k:dict(v) for k,v in coverage.items()};del cache
        spans=collections.defaultdict(list);count_equal=collections.Counter();n=collections.Counter()
        for r in map(json.loads,open(a.theory)):
            rid=str(r['record_id'])
            if rid not in allowed or rid not in records:continue
            canonical,groups=records[rid]
            if not groups:continue
            values={int(i)-1:float(x) for i,x in zip(r['value']['atom_index'],r['value']['x'])};split=allowed[rid];n[split]+=1
            count_equal[split]+=int(len(set(values.values()))==len(groups))
            for g in groups:
                xs=[values[i] for i in g if i in values]
                if len(xs)==len(g) and len(xs)>1:spans[split].append(max(xs)-min(xs))
        report['train_val_theory_only']={s:dict(records=n[s],exact_decimal_count_equal=count_equal[s],within_environment_span_ppm_quantiles=np.quantile(spans[s],[0,.25,.5,.75,.9,.95,1]).tolist() if spans[s] else []) for s in n}
        bank=torch.load(a.bank,map_location='cpu',weights_only=False,mmap=True)
        report['frozen_library']={key:dict(records=len(bank[key]),certified_records=sum(bool(registry.get(s,{}).get('counts',{}).get('1H')) for s in (bank['smiles'] if key=='smiles' else [r['smiles'] for r in bank[key]]))) for key in ['smiles','val','test','paired']}
        report['source_sha256']={str(f):sha(f) for f in [Path(a.registry)/'structures.jsonl',Path(a.registry)/'predictions.jsonl',Path(a.theory)]};report['seconds']=time.time()-start;report['registry_structures']=len(registry);report['test_spectral_values_read']=False
        (out/'audit.json').write_text(json.dumps(report,indent=2));(out/'EXIT_CODE').write_text('0\n');print(json.dumps(report),flush=True)
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

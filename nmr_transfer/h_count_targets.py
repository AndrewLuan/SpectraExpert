"""功能概述：从冻结原子顺序和物理分组构建显式区分口径的理论监督侧表。
source_parents核对源原子置换并映射到canonical重原子；target_variants分别保留原始
静态位移、既有train-only校准位移、校准后物理组均值，不修改原始文件。--source QM9-NMR按prepare相同31.7608参考将gas屏蔽转换
为位移，仅使用掩码已给出的H；--feature-registry保持原prepared计数输入口径，
新增结构分组独立进入环境解码器，防止验证阶段偷补训练中缺失的计数特征。
只有所有普通H均有标签且物理组完整覆盖才生成，缺失标签整条拒绝；组均值按核
质量守恒，并保留每母节点无序H标签。物理平均目标是可检验假设，不冒充实验真值。
main先按结构哈希选择train/val，再访问位移；test数值不进入侧表。独立理论val
来自相同冻结结构分区；原实验val仍独占选模，理论val只做预测诊断。
输出variants.pt含annotations/rows、manifest.json与退出码，目录独占；依赖已有
h_count_audit分组核验、prepare环境校准和h_labels.make_key；无网络或模型拟合。
"""
import argparse,ast,collections,json,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from .data import identity,quantiles,physical_features
from .h_count_audit import parent_groups,build_registry
from .h_labels import make_key
from .prepare import environment,sha

MODES=('raw','calibrated','physical')


def source_parents(s):
    raw=Chem.MolFromSmiles(s['source_smiles']);full=Chem.AddHs(raw);order=s['source_to_canonical']
    if sorted(order)!=list(range(full.GetNumAtoms())) or [full.GetAtomWithIdx(i).GetAtomicNum() for i in order]!=s['tokens']:raise ValueError('source_atom_contract')
    canonical=Chem.MolToSmiles(raw,isomericSmiles=True);parents={old:new for new,old in enumerate(ast.literal_eval(raw.GetProp('_smilesAtomOutputOrder')))};result={}
    for i,j in enumerate(order):
        a=full.GetAtomWithIdx(j)
        if a.GetAtomicNum()==1 and a.GetIsotope() in (0,1):
            if a.GetDegree()!=1 or a.GetNeighbors()[0].GetIdx() not in parents:raise ValueError('nonterminal_H_or_unmapped_parent')
            result[i]=(parents[a.GetNeighbors()[0].GetIdx()],environment(a))
    return canonical,result


def target_variants(s,p,r,fit):
    groups=p['nuclei'].get('1H',{})
    if groups.get('status')!='predicted':raise ValueError('uncertified_H_groups')
    canonical,layout=parent_groups(s,groups['groups']);smi,parents=source_parents(s)
    if smi!=canonical or identity(r['smiles'])[0]!=canonical:raise ValueError('theory_structure_mismatch')
    raw={int(i)-1:float(x) for i,x in zip(r['value']['atom_index'],r['value']['x'])}
    if len(raw)!=len(r['value']['atom_index']) or set(raw)!=set(parents) or not np.isfinite(list(raw.values())).all():raise ValueError('incomplete_or_invalid_theory_H')
    calibrated={}
    for i,x in raw.items():
        tag=fit.get('environment_residuals',{}).get(parents[i][1],{});residual=tag.get('median_ppm',0.) if tag.get('atoms',0)>=100 else 0.
        calibrated[i]=fit['affine_scale']*x+fit['affine_offset_ppm']+residual
    physical=dict(calibrated)
    for group in groups['groups']:
        center=float(np.mean([calibrated[i] for i in group]))
        for i in group:physical[i]=center
    result={}
    for mode,values in [('raw',raw),('calibrated',calibrated),('physical',physical)]:
        sites=collections.defaultdict(list)
        for i,v in values.items():sites[parents[i][0]].append(v)
        result[mode]=dict(q=quantiles(list(values.values())),annotation=dict(groups=dict(sites),observed_H=len(values),structural_H=len(parents),full_atom_labels=True,calibration=mode))
    if abs(sum(physical.values())-sum(calibrated.values()))>1e-7:raise ValueError('group_projection_mass_moment_changed')
    return canonical,result


def main():
    p=argparse.ArgumentParser()
    for k in ['registry','theory','fit','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--source',choices=['recalculated','QM9-NMR'],default='recalculated');p.add_argument('--feature-registry')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);root=Path(a.registry);start=time.time();torch.set_num_threads(4)
    try:
        registry,_,registry_audit=build_registry(root);structures={r['record_id']:r for r in map(json.loads,open(root/'structures.jsonl'))};predictions={r['record_id']:r for r in map(json.loads,open(root/'predictions.jsonl'))};fit=json.loads(Path(a.fit).read_text())['fits'][a.source+'|1H'];feature_registry=torch.load(a.feature_registry,map_location='cpu',weights_only=False) if a.feature_registry else None;annotations={};val=[];counts=collections.Counter();errors=collections.Counter()
        for r in map(json.loads,open(a.theory)):
            try:
                if a.source=='QM9-NMR' and r.get('nucleus')!='1H':continue
                smi,group,split=identity(r['smiles'])
                if split=='test':continue
                if smi not in registry:errors['uncertified_or_conflicting_structure']+=1;continue
                rid=r['record_id']
                if rid not in structures or rid not in predictions:continue
                if a.source=='QM9-NMR':
                    if r.get('quantity_kind')!='isotropic_shielding' or r.get('conditions',{}).get('phase')!='gas':raise ValueError('QM9_quantity_or_phase')
                    r=dict(smiles=r['smiles'],record_id=rid,value=dict(atom_index=[i+1 for i,m in enumerate(r['atom_target_mask']) if m],x=[31.7608-float(x) for x,m in zip(r['atom_targets'],r['atom_target_mask']) if m]))
                s,variants=target_variants(structures[rid],predictions[rid],r,fit)
                if s!=smi:raise ValueError('canonical_mapping_mismatch')
                row=dict(smiles=s,group=group,split=split,nucleus='1H',source=a.source,domain='theory',record_id=str(rid),cond=np.zeros(4,np.float32),q=variants['raw']['q'],measure='atom_equal',observation=dict(kind='complete_atoms',windows=[],evidence='full theoretical atom coverage; no experimental acquisition claim'),physics=physical_features(s,feature_registry.get(s,{}).get('counts',{}) if feature_registry is not None else {n:x['count'] for n,x in predictions[rid]['nuclei'].items() if x.get('status')=='predicted'}))
                annotations[make_key(row)]=variants;counts[split]+=1
                if split=='val':val.append(row)
            except (ValueError,KeyError,IndexError) as exc:errors[str(exc)]+=1
        torch.save(dict(variants=annotations,theory_validation=val),out/'variants.pt');manifest=dict(source=a.source,feature_registry_sha256=sha(a.feature_registry) if a.feature_registry else None,registry_audit=registry_audit,records=dict(counts),errors=dict(errors),test_targets_included=False,fit='existing train-only affine/environment fit',selection='theory validation diagnostic only; original experimental validation selects epochs',source_sha256={str(f):sha(f) for f in [root/'structures.jsonl',root/'predictions.jsonl',Path(a.theory),Path(a.fit)]},artifact_sha256=sha(out/'variants.pt'),seconds=time.time()-start);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'EXIT_CODE').write_text('0\n');print(json.dumps(manifest),flush=True)
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

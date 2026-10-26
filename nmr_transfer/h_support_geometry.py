"""功能概述：为新增train/val支持分子生成结构限定的多构象物理分组侧表。
worker只接收canonical SMILES、结构分区和标识，不接收谱/积分；按固定三个seed
ETKDG→MMFF/UFF收敛优化，再调用冻结物理求解器。三个构象均成功且canonical
母节点分区完全一致才接入未知结构；已有分组始终保留。它是模型假设下有限构象
稳定性证据，不声明独立物理准确率。原计数特征、候选库、test及源谱均不改。
main核对原支持文件的结构分区，写新registry、逐构象状态、布局、SHA和退出码；
目录独占。依赖h_count_audit的母节点契约与既有v5，CPU worker各自有45秒物理限时。
"""
import argparse,collections,concurrent.futures,json,multiprocessing,sys
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem,rdBase
from rdkit.Chem import AllChem
from .data import identity
from .h_count_audit import parent_groups
from .prepare import sha

SEEDS=(173,193,211)


def worker(item):
    from benchmark import worker as solve
    s=item['smiles'];trials=[];layouts=[]
    for seed in SEEDS:
        try:
            mol=Chem.AddHs(Chem.MolFromSmiles(s));params=AllChem.ETKDGv3();params.randomSeed=seed;params.numThreads=1;params.maxIterations=1000
            if AllChem.EmbedMolecule(mol,params)!=0:raise ValueError('embedding_failed')
            if AllChem.MMFFHasAllMoleculeParams(mol):engine='MMFF';status=AllChem.MMFFOptimizeMolecule(mol,maxIters=1000)
            elif AllChem.UFFHasAllMoleculeParams(mol):engine='UFF';status=AllChem.UFFOptimizeMolecule(mol,maxIters=1000)
            else:raise ValueError('force_field_parameters_missing')
            if status:raise ValueError('geometry_optimization_not_converged')
            structure=dict(record_id=item['group']+'_'+str(seed),source_smiles=s,coordinates=mol.GetConformer().GetPositions().tolist(),source_to_canonical=list(range(mol.GetNumAtoms())),tokens=[a.GetAtomicNum() for a in mol.GetAtoms()],nuclei=['1H']);prediction=solve(structure);n=prediction['nuclei']['1H'];trial=dict(seed=seed,engine=engine,structure=structure,prediction=prediction)
            if n.get('status')=='predicted':
                canonical,groups=parent_groups(structure,n['groups'])
                if canonical!=s:raise ValueError('canonical_layout_mismatch')
                layouts.append(dict(counts={'1H':n['count']},groups=groups))
            trials.append(trial)
        except Exception as exc:trials.append(dict(seed=seed,error=repr(exc)))
    stable=len(layouts)==len(SEEDS) and all(x==layouts[0] for x in layouts)
    return dict(**item,conformer_partition_stable=stable,trials=trials,layout=layouts[0] if stable else None)


def main():
    p=argparse.ArgumentParser()
    for k in ['support','registry','solver','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(Path(a.solver).parent));import benchmark;torch.set_num_threads(1)
    try:
        registry=torch.load(a.registry,map_location='cpu',weights_only=False);support=Path(a.support);items={};files=[]
        for split in ['train','val']:
            f=support/('eligible_'+split+'.jsonl');files.append(f)
            for r in map(json.loads,f.read_text().splitlines()):
                smi,group,actual=identity(r['canonical_smiles'])
                if actual!=split or r['group']!=group:raise ValueError('support_geometry_partition_mismatch')
                if registry.get(smi,{}).get('groups'):continue
                items[smi]=dict(smiles=smi,group=group,split=split)
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:results=list(pool.map(worker,[items[s] for s in sorted(items)]))
        approved=[]
        for r in results:
            if r['conformer_partition_stable']:
                old=registry.get(r['smiles'],{});counts=dict(old.get('counts',{}),**r['layout']['counts']);registry[r['smiles']]=dict(counts=counts,groups=r['layout']['groups']);approved.append(r['smiles'])
        torch.save(registry,out/'registry.pt');(out/'geometry_records.json').write_text(json.dumps(results,indent=2));report=dict(attempted=len(results),approved=len(approved),by_split={split:dict(attempted=sum(r['split']==split for r in results),approved=sum(r['split']==split and r['conformer_partition_stable'] for r in results)) for split in ['train','val']},approved_smiles=approved,seeds=SEEDS,rdkit_version=rdBase.rdkitVersion,spectral_values_used=False,existing_registry_preserved=True,source_sha256={str(f):sha(f) for f in files+[Path(a.registry),Path(a.solver),Path(__file__)]},registry_sha256=sha(out/'registry.pt'));(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'EXIT_CODE').write_text('0\n');print(json.dumps(report),flush=True)
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

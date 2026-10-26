"""功能概述：把通过三个生成构象一致性检查的候选结构分区写入独立扩展registry。
main只读取冻结候选SMILES及基准registry；完整保留原H分组，未知结构在已审计的
重原子数<=45范围生成三构象。所有候选采用同一规则，含test结构但绝不访问任何
谱、实验条件或真实查询身份。有限构象一致性是模型证据，不等同外部物理准确率。
worker复用h_support_geometry的固定seed、收敛优化和物理调用期限；稳定且H质量
已校验的分区才接入。新项记录生成几何来源，旧计数输入、候选次序、标签不改。
独占目录保存逐构象审计、进度、registry/SHA/EXIT_CODE；既有28臂不使用该侧表。
冻结身份不能在当前RDKit契约下逐字复现时，保留候选并回退未知；不重命名候选。
"""
import argparse,collections,concurrent.futures,json,multiprocessing,sys,time
from pathlib import Path
import torch
from rdkit import Chem
from .data import identity
from .h_support_geometry import worker,SEEDS
from .prepare import sha


def main():
    p=argparse.ArgumentParser()
    for name in ['smiles','registry','solver','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=192);p.add_argument('--max-heavy',type=int,default=45);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(1)
    try:
        registry=torch.load(a.registry,map_location='cpu',weights_only=False);names=json.loads(Path(a.smiles).read_text());items=[];excluded=collections.Counter()
        if len(names)!=len(set(names)):raise ValueError('candidate_smiles_not_unique')
        for s in names:
            if registry.get(s,{}).get('groups'):excluded['existing_H_partition_preserved']+=1;continue
            if '.' in s:excluded['multicomponent']+=1;continue
            canonical,group,split=identity(s);mol=Chem.MolFromSmiles(s)
            if canonical!=s:excluded['noncanonical_frozen_identity']+=1;continue
            if mol.GetNumHeavyAtoms()>a.max_heavy:excluded['above_audited_size']+=1;continue
            if not any(atom.GetTotalNumHs()>0 for atom in mol.GetAtoms()):excluded['no_attached_H']+=1;continue
            items.append(dict(smiles=s,group=group,split=split))
        sys.path.insert(0,str(Path(a.solver).parent));import benchmark;approved=collections.Counter();processed=0
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
            with (out/'records.jsonl').open('x') as f:
                for r in pool.map(worker,items,chunksize=1):
                    processed+=1;f.write(json.dumps(r)+'\n')
                    if r['conformer_partition_stable']:
                        old=registry.get(r['smiles'],{});registry[r['smiles']]=dict(counts=dict(old.get('counts',{}),**r['layout']['counts']),groups=r['layout']['groups'],partition_evidence='generated_geometry_three_seed',conformer_seeds=SEEDS);approved[r['split']]+=1
                    if processed%100==0:f.flush();(out/'status.json').write_text(json.dumps(dict(stage='geometry_partitions',processed=processed,total=len(items),approved=sum(approved.values()),seconds=time.time()-start)))
        torch.save(registry,out/'registry.pt');report=dict(attempted=len(items),approved=sum(approved.values()),approved_by_split=dict(approved),excluded=dict(excluded),registry_entries=len(registry),source_sha256={str(f):sha(f) for f in [Path(a.smiles),Path(a.registry),Path(a.solver),Path(__file__),Path(__file__).parent/'h_support_geometry.py']},registry_sha256=sha(out/'registry.pt'),max_heavy=a.max_heavy,spectral_targets_used=False,frozen_physics_features_changed=False,original_H_partitions_preserved=True,seconds=time.time()-start,scope='generated finite-conformer stability; separate from original-coordinate certificates');(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'EXIT_CODE').write_text('0\n');print(json.dumps(report),flush=True)
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

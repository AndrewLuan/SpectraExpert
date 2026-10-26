"""功能概述：检查生成构象的物理分区是否稳定，并与原坐标分区作结构限定复现对照。
select只访问冻结候选SMILES、已知registry及二维重原子数，按固定hash在train/val
的已知/未知和三个大小层各取固定数量；不访问位移、峰数标签或测试记录。
main复用h_support_geometry.worker的三个固定seed、收敛优化及冻结物理求解器。
已知组比较K和canonical母节点分区；未知组仅记录有限构象稳定性，二者均不是
外部物理准确率。保存每个输入结构、原分区、逐构象坐标/状态和来源SHA；不合并
或替换任何训练registry，不改变冻结28臂、候选、物理特征或实验观测。
所有输出独占且有EXIT_CODE。CPU进程池由--workers限制；每个物理调用原有45秒
截止，几何优化与三个seed沿用先前支持审计契约。无外部服务。
"""
import argparse,collections,concurrent.futures,hashlib,json,multiprocessing,sys
from pathlib import Path
import torch
from rdkit import Chem
from .data import identity
from .h_support_geometry import worker
from .prepare import sha


def select(smiles,registry,per_bucket):
    buckets=collections.defaultdict(list)
    for s in smiles:
        if '.' in s:continue
        canonical,group,split=identity(s)
        if canonical!=s or split=='test':continue
        mol=Chem.MolFromSmiles(s);heavy=mol.GetNumHeavyAtoms()
        if heavy>45 or not any(a.GetTotalNumHs()>0 for a in mol.GetAtoms()):continue
        size='small' if heavy<=15 else 'medium' if heavy<=30 else 'large';known=bool(registry.get(s,{}).get('groups'));buckets[known,size].append(dict(smiles=s,group=group,split=split,known=known,size=size,heavy_atoms=heavy))
    items=[]
    for key,rows in sorted(buckets.items()):items.extend(sorted(rows,key=lambda r:hashlib.sha256(('geometry_partition_pilot193|'+r['smiles']).encode()).hexdigest())[:per_bucket])
    return items,{str(k):len(v) for k,v in buckets.items()}


def main():
    p=argparse.ArgumentParser()
    for name in ['bank','registry','solver','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=64);p.add_argument('--per-bucket',type=int,default=80);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    try:
        registry=torch.load(a.registry,map_location='cpu',weights_only=False);bank=torch.load(a.bank,map_location='cpu',weights_only=False,mmap=True);smiles=bank['smiles'];items,available=select(smiles,registry,a.per_bucket);(out/'candidate_smiles.json').write_text(json.dumps(smiles));del bank;sys.path.insert(0,str(Path(a.solver).parent));import benchmark
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
            records=[]
            with (out/'records.jsonl').open('x') as f:
                for r in pool.map(worker,items):
                    original=registry.get(r['smiles'],{});r['original_layout']=original
                    if r['known'] and r['conformer_partition_stable']:r['count_agrees']=r['layout']['counts']['1H']==original['counts']['1H'];r['partition_agrees']=r['layout']['groups']==original['groups']
                    records.append(r);f.write(json.dumps(r)+'\n');f.flush()
        summary={}
        for known in [False,True]:
            rs=[r for r in records if r['known']==known];stable=[r for r in rs if r['conformer_partition_stable']];summary['known' if known else 'unknown']=dict(attempted=len(rs),stable=len(stable),count_agrees=sum(r.get('count_agrees',False) for r in stable) if known else None,partition_agrees=sum(r.get('partition_agrees',False) for r in stable) if known else None,by_size={size:dict(attempted=sum(r['size']==size for r in rs),stable=sum(r['size']==size for r in stable)) for size in ['small','medium','large']})
        report=dict(results=summary,available=available,source_sha256={str(f):sha(f) for f in [Path(a.registry),Path(a.solver),Path(__file__),Path(__file__).parent/'h_support_geometry.py']},candidate_smiles_sha256=sha(out/'candidate_smiles.json'),spectral_targets_used=False,test_structures_in_pilot=False,registry_modified=False,scope='finite-conformer consistency and source-coordinate reproduction; not experimental peak-count accuracy');(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'EXIT_CODE').write_text('0\n');print(json.dumps(report),flush=True)
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

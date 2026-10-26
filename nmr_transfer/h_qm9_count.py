"""功能概述：将QM9原有三维结构对齐到显式H图，独立运行冻结确定性物理分组。
structure_only仅用SMILES、元素和源坐标；DetermineBonds恢复图后要求与源SMILES
完整显式H同构（含已有手性），反向置换并核对元素/边。任何错配拒绝，不用位移选映射。
worker调用原benchmark.worker的45秒物理求解期限。main在结构train/val上生成
不含谱标签的structures/predictions JSONL，先做小样本对齐pilot；test结构不参与。
输出记录、错误、源/求解器SHA及退出码，目录独占，不能覆盖原registry和源数据。
依赖RDKit、原v5/benchmark.py及prepare.identity；无外部服务、坐标优化或标签拟合。
"""
import argparse,collections,concurrent.futures,json,multiprocessing,sys,time
from pathlib import Path
import numpy as np
from rdkit import Chem,RDLogger,rdBase
from rdkit.Chem import rdDetermineBonds
from .data import identity
from .prepare import sha


def structure_only(r):
    raw=Chem.MolFromSmiles(r['smiles']);full=Chem.AddHs(raw);atoms=r['atoms'];xyz=np.asarray(r['coordinates'],float);n=full.GetNumAtoms()
    if xyz.shape!=(n,3) or len(atoms)!=n or not np.isfinite(xyz).all():raise ValueError('QM9_coordinate_shape')
    block=str(n)+'\nQM9 source geometry\n'+'\n'.join(str(a)+' '+' '.join(format(float(x),'.17f') for x in p) for a,p in zip(atoms,xyz))+'\n';source=Chem.MolFromXYZBlock(block)
    if source is None:raise ValueError('QM9_XYZ_parse')
    rdDetermineBonds.DetermineBonds(source,charge=Chem.GetFormalCharge(raw),allowChargedFragments=True,embedChiral=True,maxIterations=10000)
    mapping=source.GetSubstructMatch(full,useChirality=True)
    if len(mapping)!=n or len(set(mapping))!=n:raise ValueError('QM9_coordinate_SMILES_graph_mismatch')
    order=np.argsort(mapping).tolist();tokens=[a.GetAtomicNum() for a in source.GetAtoms()]
    if tokens!=[full.GetAtomWithIdx(j).GetAtomicNum() for j in order]:raise ValueError('QM9_element_alignment')
    source_edges={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in source.GetBonds()};model_edges={tuple(sorted((mapping[b.GetBeginAtomIdx()],mapping[b.GetEndAtomIdx()]))) for b in full.GetBonds()}
    if source_edges!=model_edges:raise ValueError('QM9_connectivity_alignment')
    return dict(record_id=str(r['record_id']),source_smiles=r['smiles'],coordinates=xyz.tolist(),source_to_canonical=order,tokens=tokens,nuclei=['1H'])


def worker(r):
    try:
        structure=structure_only(r)
        from benchmark import worker as physical_worker
        return structure,physical_worker(structure),None
    except Exception as exc:return None,None,dict(record_id=r['record_id'],reason=type(exc).__name__+':'+str(exc))


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--solver',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=1000);p.add_argument('--workers',type=int,default=8);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time();sys.path.insert(0,str(Path(a.solver).parent));RDLogger.DisableLog('rdApp.*');items=[];split_counts=collections.Counter();errors=collections.Counter();statuses=collections.Counter()
    try:
        import benchmark
        import igraph
        for line in open(a.input):
            r=json.loads(line)
            if r.get('nucleus')!='1H':continue
            smi,group,split=identity(r['smiles'])
            if split=='test':continue
            items.append({k:r[k] for k in ['smiles','record_id','atoms','coordinates']});split_counts[split]+=1
            if a.limit and len(items)>=a.limit:break
        with (out/'structures.jsonl').open('x') as sf,(out/'predictions.jsonl').open('x') as pf,(out/'rejected.jsonl').open('x') as ef:
            with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
                for i,(structure,pred,error) in enumerate(pool.map(worker,items,chunksize=8),1):
                    if error:errors[error['reason']]+=1;ef.write(json.dumps(error)+'\n')
                    else:sf.write(json.dumps(structure)+'\n');pf.write(json.dumps(pred)+'\n');statuses[pred['nuclei']['1H']['status']]+=1
                    if i%1000==0:sf.flush();pf.flush();ef.flush();(out/'status.json').write_text(json.dumps(dict(processed=i,total=len(items),seconds=time.time()-start)))
        report=dict(igraph_version=igraph.__version__,rdkit_version=rdBase.rdkitVersion,records=len(items),split_counts=dict(split_counts),alignment_rejected=dict(errors),prediction_status=dict(statuses),test_structures_included=False,spectral_fields_passed_to_workers=False,source_sha256={str(f):sha(f) for f in [Path(a.input),Path(a.solver),Path(__file__)]},seconds=time.time()-start);(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'EXIT_CODE').write_text('0\n');print(json.dumps(report),flush=True)
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

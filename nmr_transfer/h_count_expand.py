"""功能概述：合并既有和QM9的结构认证分组，保持冻结候选和物理输入特征。
merge_registries按H分区合并，不把仅缺13C分组误判为冲突；两源H分组不一致时
整结构回退未知，不任选某个构象。main保存扩展registry和QM9独立registry，另从
原cache/候选bank恢复计数输入侧表，供h_count_targets保持原已知/缺失口径。
所有新产物写独占目录，原registry/cache/bank只读。summary记录覆盖、冲突、SHA；
不读谱位移作决策，不改变候选/测试次序，不调用外部服务。
"""
import argparse,json,time
from pathlib import Path
import torch
from .h_count_audit import build_registry
from .prepare import sha


def merge_registries(original,extra):
    merged=dict(original);conflicts=[]
    for s,new in extra.items():
        if not new.get('groups'):continue
        old=merged.get(s)
        if old and old.get('groups'):
            if old['counts'].get('1H')!=new['counts'].get('1H') or old['groups']!=new['groups']:
                conflicts.append(s);merged.pop(s);continue
        counts=dict(old.get('counts',{}) if old else {});counts['1H']=new['counts']['1H'];merged[s]=dict(counts=counts,groups=new['groups'])
    return merged,sorted(conflicts)


def feature_counts(physics):
    p=physics.tolist() if hasattr(physics,'tolist') else physics
    return dict(counts={n:int(round(float(p[i])*30)) for n,i in [('1H',5),('13C',6)] if float(p[i])>0},groups=None)


def main():
    p=argparse.ArgumentParser()
    for k in ['original','extra','prepared','bank','out']:p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(4)
    try:
        original=torch.load(a.original,map_location='cpu',weights_only=False);extra,_,audit=build_registry(Path(a.extra));merged,conflicts=merge_registries(original,extra);torch.save(merged,out/'registry.pt');torch.save(extra,out/'QM9_registry.pt');cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);features={}
        for r in cache['rows']:
            entry=feature_counts(r['physics'])
            if r['smiles'] in features and features[r['smiles']]!=entry:raise ValueError('frozen_physics_count_inconsistent')
            features[r['smiles']]=entry
        del cache;bank=torch.load(a.bank,map_location='cpu',weights_only=False,mmap=True)
        for s,f in zip(bank['smiles'],bank['physics']):
            entry=feature_counts(f)
            if s in features and features[s]!=entry:raise ValueError('candidate_train_physics_count_inconsistent')
            features[s]=entry
        torch.save(features,out/'feature_counts.pt');report=dict(original=len(original),extra=len(extra),merged=len(merged),cross_source_H_conflicts=len(conflicts),conflicting_smiles=conflicts,extra_registry_audit=audit,frozen_feature_structures=len(features),source_sha256={str(f):sha(f) for f in [Path(a.original),Path(a.extra)/'structures.jsonl',Path(a.extra)/'predictions.jsonl',Path(a.prepared)/'manifest.json']},artifact_sha256={f.name:sha(f) for f in out.glob('*.pt')},candidate_coverage={key:sum(bool(merged.get(s,{}).get('groups')) for s in (bank['smiles'] if key=='smiles' else [r['smiles'] for r in bank[key]])) for key in ['smiles','val','test','paired']},seconds=time.time()-start);(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'EXIT_CODE').write_text('0\n');print(json.dumps({k:v for k,v in report.items() if k!='conflicting_smiles'}),flush=True)
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

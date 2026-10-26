"""功能概述：只读定位授权旧4完整validation与新同4格的逐字段语义差异。
仅读取这8份已闭包预测；不读取旧partial、test、tensor，不修改比较规则或科学结果。
每个字段汇总不同项数、受影响query数和绝对误差，保留原字节SHA；不据此重选。
"""
from pathlib import Path
import json,hashlib,collections,math
HERE=Path(__file__).resolve().parent
ROOT='/home/user/IRfingerprints/SpectrExpert/outputs/nmr_cap10_controlled_20261024T163633Z'
IGNORE={'pid','ppid','parent_pid','started_utc','finished_utc','elapsed_seconds','timing'}
FIELDS=('query_order','variant','source','record_id','smiles','group')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    out=HERE/'validation_semantic_difference_review.json';assert not out.exists()
    map_paths=[HERE/'attempt_history/validation_resource_not_ready/metadata_view.json',HERE/'collected/validation_resource_ready/metadata_view.json']
    files={};hashes={};pins={};results=[]
    for p in map_paths:
        raw=p.read_bytes();pins[str(p)]=sha(raw);m=json.loads(raw);assert m['remote_before_sha256']==m['remote_after_sha256']
        files.update(m['files']);hashes.update(m['sha256'])
    def read(path):
        p=Path(files[path]).resolve();assert p.name=='predictions.jsonl' and p.is_file()
        raw=p.read_bytes();assert sha(raw)==hashes[path];pins[str(p)]=sha(raw)
        rows=[json.loads(line) for line in raw.splitlines() if line];assert len(rows)==904
        return {tuple(r['query_identity'][k] for k in FIELDS):r for r in rows}
    for seed in (193,211):
        for pool in ('single','dual'):
            jid=f'validation_C5_s{seed}_{pool}_original_val'
            a=read(ROOT+'/validation_native_cost/'+jid+'/predictions.jsonl')
            b=read(ROOT+'/validation_resource_ready/'+jid+'/predictions.jsonl');assert set(a)==set(b)
            diffs={};affected=set();rank_changed=0
            def compare(x,y,path,identity):
                if x==y:return
                if isinstance(x,dict) and isinstance(y,dict):
                    for k in set(x)|set(y):
                        if k in IGNORE:continue
                        compare(x.get(k,'<MISSING>'),y.get(k,'<MISSING>'),path+'.'+k,identity)
                    return
                if isinstance(x,list) and isinstance(y,list) and len(x)==len(y):
                    for aa,bb in zip(x,y):compare(aa,bb,path+'[]',identity)
                    return
                d=diffs.setdefault(path,dict(different_values=0,queries=set(),numeric_values=0,max_abs=0.,example=[]))
                d['different_values']+=1;d['queries'].add(identity);affected.add(identity)
                if type(x) in (float,int) and type(y) in (float,int) and math.isfinite(x) and math.isfinite(y):
                    d['numeric_values']+=1;d['max_abs']=max(d['max_abs'],abs(x-y))
                if len(d['example'])<2:d['example'].append(dict(old=x if not isinstance(x,(list,dict)) else type(x).__name__,new=y if not isinstance(y,(list,dict)) else type(y).__name__))
            for k in a:
                compare(a[k],b[k],'row',k);rank_changed+=a[k]['rank']!=b[k]['rank']
            for d in diffs.values():d['affected_queries']=len(d.pop('queries'))
            results.append(dict(id=jid,rows=904,identity_exact=True,affected_queries=len(affected),rank_changed_queries=rank_changed,fields=diffs))
    for p,h in pins.items():assert sha(Path(p).read_bytes())==h
    result=dict(scope='field_diagnostic_for_authorized_old4_complete_validation_only_not_scientific_acceptance',status='DIFFERENCE_INSPECTION',
      cells=results,source_input_sha256=pins,inputs_unchanged=True,scientific_changes=False,selection_performed=False,test_read=False)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(path=str(out),sha256=sha(out.read_bytes()),cells=[{k:v for k,v in c.items() if k!='fields'}|{'different_fields':list(c['fields'])} for c in results])))
if __name__=='__main__':main()

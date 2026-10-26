"""功能概述：只读比较旧机制12完整行与新完整结果的同身份行，仅定位语义字段差异。
范围由95b恢复策略明确授权；不读取旧test/旧partial验证，不重选，不改原结果或容差。
"""
from pathlib import Path
import json,hashlib,math
HERE=Path(__file__).resolve().parent
ROOT='/home/user/IRfingerprints/SpectrExpert/outputs/nmr_cap10_controlled_20261024T163633Z'
FIELDS=('query_order','variant','source','record_id','smiles','group')
IGNORE={'pid','ppid','parent_pid','started_utc','finished_utc','elapsed_seconds','timing'}
def sha(x):return hashlib.sha256(x).hexdigest()
def main():
    out=HERE/'mechanism12_semantic_difference_review.json';assert not out.exists();files={};hashes={};pins={};result=[]
    for p in [HERE/'attempt_history/mechanism_identity/manifest.json',HERE/'collected/mechanism_fixed_rows/metadata_view.json']:
        raw=p.read_bytes();pins[str(p)]=sha(raw);m=json.loads(raw);assert m['remote_before_sha256']==m['remote_after_sha256'];files.update(m['files']);hashes.update(m['sha256'])
    def read(key,n):
        p=Path(files[key]).resolve();assert p.name=='predictions.jsonl';raw=p.read_bytes();assert sha(raw)==hashes[key];pins[str(p)]=sha(raw)
        rows=[json.loads(line) for line in raw.splitlines() if line];assert len(rows)==n
        return {tuple(row['query_identity'][k] for k in FIELDS):row for row in rows}
    for cap in (5,10):
        for seed in (193,211):
            jid=f'mechanism_C{cap}_s{seed}_dual_train'
            a=read(ROOT+'/mechanism_resource_ready/'+jid+'/predictions.jsonl',3);b=read(ROOT+'/mechanism_fixed_rows/'+jid+'/predictions.jsonl',200)
            assert a.keys()<=b.keys();diffs={}
            def compare(x,y,path):
                if x==y:return
                if isinstance(x,dict) and isinstance(y,dict):
                    for k in set(x)|set(y):
                        if k not in IGNORE:compare(x.get(k,'<MISSING>'),y.get(k,'<MISSING>'),path+'.'+k)
                elif isinstance(x,list) and isinstance(y,list) and len(x)==len(y):
                    for aa,bb in zip(x,y):compare(aa,bb,path+'[]')
                else:
                    d=diffs.setdefault(path,dict(count=0,max_abs=0.));d['count']+=1
                    if type(x) in (int,float) and type(y) in (int,float) and math.isfinite(x) and math.isfinite(y):d['max_abs']=max(d['max_abs'],abs(x-y))
            for key in a:compare(a[key],b[key],'row')
            result.append(dict(id=jid,compared_complete_lines=3,rank_changed_queries=sum(a[k]['rank']!=b[k]['rank'] for k in a),fields=diffs))
    for p,h in pins.items():assert sha(Path(p).read_bytes())==h
    out.write_text(json.dumps(dict(status='DIFFERENCE_INSPECTION',scope='authorized12mechanism_lines_semantic_consistency_only',cells=result,input_sha256=pins,inputs_unchanged=True,test_read=False,selection_performed=False),indent=2)+'\n')
    print(json.dumps(dict(path=str(out),sha256=sha(out.read_bytes()),cells=result)))
if __name__=='__main__':main()

"""功能概述：只读保全机制identity失败原始证据，禁止自动重试或解析预测/指标。

两次有界SSH只读取登记的4任务失败收据、路径/进程元数据并流式SHA核全部213科学输入、
25源码及完整原始输出。rsync仅复制既有字节；prediction只哈希，不JSON解析、不读数值。
原run、缺失after、所有partial保持原状。输出新本地attempt目录，保存每次真实SSH/rsync
退出、stdout/stderr、始末SHA、原文件大小及映射；不修改远端、不启动任何模型任务。
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT='/home/user/IRfingerprints/SpectrExpert/outputs/nmr_cap10_controlled_20261024T163633Z'
QUEUE_SHA='6f8c76a26501d749ad7a72c3f7985ed017112004b3e8650d5c58940b2a1e0711'
BINDING_SHA='2f56b0470a15b9746851aa4779c422ad4a8816af70f532450e18d75a9b3d556d'
REMOTE=r'''
import hashlib,json,os,pathlib,datetime
r=pathlib.Path(ROOT);qp=r/'mechanism_resource_ready_queue.json'
def safe(p):
 p=pathlib.Path(p);assert p.is_absolute() and '..' not in p.parts and not p.is_symlink(),str(p)
 x=p.resolve(strict=True);assert x==p,str(p)
 assert not any(v=='test' or v.startswith('sealed') for v in p.parts),str(p)
 assert p.name not in ('cohorts.pt','test.pt','sealed_test_report.md'),str(p)
 return p
def sha(p):
 h=hashlib.sha256()
 with safe(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(safe(p).read_bytes())
assert sha(qp)==QUEUE_SHA
q=read(qp);assert q['root']==str(r) and q['phase']=='mechanism' and q['run_name']=='mechanism_resource_ready'
assert len(q['jobs'])==4 and len(q['source_sha256'])==25 and len(q['input_sha256'])==213
assert sha(q['bindings'])==BINDING_SHA and read(q['bindings'])['input_sha256']==q['input_sha256']
run=r/'runs/mechanism_resource_ready';done=read(run/'completion.json');outer=read(run/'launch_receipt.json');start=read(run/'driver_start.json');status=read(run/'status.json')
assert done['status']=='FAILED' and type(done['external_exit']) is int and done['external_exit']==1
assert all(type(outer[k]) is int and outer[k]==1 for k in ('internal_exit','external_exit'))
assert safe(run/'EXIT_CODE').read_bytes()==safe(run/'EXTERNAL_EXIT_CODE').read_bytes()==b'1\n'
assert done['queue_sha256']==outer['queue_sha256']==start['queue_sha256']==QUEUE_SHA
assert done['pid']==outer['child_pid']==start['pid'] and start['parent_pid']==outer['wrapper_pid']
before=read(run/'input_fingerprints.before.json');assert before==q['input_sha256']
input_hash={p:sha(p) for p in q['input_sha256']};assert input_hash==q['input_sha256']
source_hash={p:sha(p) for p in q['source_sha256']};assert source_hash==q['source_sha256']
paths=set()
def include(p,required=True):
 p=pathlib.Path(p)
 if not p.exists():assert not required,str(p);return
 if p.is_dir():
  for x in p.rglob('*'):
   if x.is_file():include(x)
 else:paths.add(str(safe(p)))
include(run);include(qp);include(q['bindings']);include(r/'protocol.json')
for job in q['jobs']:
 include(job['job_out'],False);include(job['access_log'],False)
for p in q['review_paths']:include(p)
for p in q['source_sha256']:include(p)
for p in q['input_sha256']:
 if pathlib.Path(p).suffix.lower() not in ('.pt','.pth','.ckpt','.npz','.npy'):include(p)
for suffix in ('.dispatch.json','.launch_claim.json','.operator_claim.json','.launcher.stdout.log'):
 include(r/(q['run_name']+suffix),False)
for ref in (q['operator_implementation_review'],q['analysis_tools']['implementation_review'],q['recovery_from']['failure_review'],q['recovery_from']['strategy']):include(ref['path'])
artifacts={p:sha(p) for p in sorted(paths)}
observations={}
for pid in {done['pid'],outer['wrapper_pid']}|{j['pid'] for j in done['jobs']}:
 try:os.kill(pid,0);alive=True
 except ProcessLookupError:alive=False
 except PermissionError:alive=True
 entry={'PID_exists':alive}
 if alive:
  proc=pathlib.Path('/proc')/str(pid)
  try:entry['cmdline']=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except FileNotFoundError:entry['exited_during_observation']=True
 observations[str(pid)]=entry
print(json.dumps(dict(schema='cap10_mechanism_identity_failure_inventory_20261025',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 root=str(r),queue_sha256=QUEUE_SHA,bindings_sha256=BINDING_SHA,input_sha256=input_hash,source_sha256=source_hash,
 artifacts=artifacts,artifact_bytes={p:safe(p).stat().st_size for p in artifacts},
 completion=done,launch=outer,status=status,original_after_exists=(run/'input_fingerprints.after.json').exists(),
 original_operator_before_exists=(run/'operator_fingerprints.before.json').exists(),
 original_operator_after_exists=(run/'operator_fingerprints.after.json').exists(),PID_observations=observations,
 prediction_values_parsed=False,metrics_values_interpreted=False,model_calls=0,training=False,test_read=False),sort_keys=True))
'''

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    own=Path(__file__).resolve();own_sha=sha(own);started=datetime.now(timezone.utc).isoformat();attempts=[]
    try:
        code='ROOT='+repr(ROOT)+'\nQUEUE_SHA='+repr(QUEUE_SHA)+'\nBINDING_SHA='+repr(BINDING_SHA)+'\n'+REMOTE
        remote_command=shlex.join(['/home/user/.venvs/spectrexpert/bin/python','-B','-c',code])
        argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','JJ',remote_command]
        def inventory(name):
            t=datetime.now(timezone.utc).isoformat();result=subprocess.run(argv,capture_output=True,text=True,timeout=420)
            (out/(name+'.stdout.json')).write_text(result.stdout);(out/(name+'.stderr.log')).write_text(result.stderr)
            attempts.append(dict(kind=name,argv=argv,started_utc=t,finished_utc=datetime.now(timezone.utc).isoformat(),actual_exit=result.returncode))
            assert type(result.returncode) is int and result.returncode==0,(name,result.returncode,result.stderr)
            return json.loads(result.stdout)
        before=inventory('remote_inventory_before')
        files=out/'files_from.txt';files.write_text(''.join(p.lstrip('/')+'\n' for p in before['artifacts']))
        payload=out/'copied';payload.mkdir()
        rsync=['rsync','-azR','--files-from='+str(files),'-e','ssh -o BatchMode=yes -o ConnectTimeout=15','JJ:/',str(payload)+'/']
        result=subprocess.run(rsync,capture_output=True,text=True,timeout=420)
        (out/'rsync.stdout.log').write_text(result.stdout);(out/'rsync.stderr.log').write_text(result.stderr)
        attempts.append(dict(kind='rsync',argv=rsync,actual_exit=result.returncode));assert result.returncode==0,result.stderr
        local={p:str(payload/p.lstrip('/')) for p in before['artifacts']}
        copied={p:sha(f) for p,f in local.items()};assert copied==before['artifacts']
        after=inventory('remote_inventory_after')
        for key in ('artifacts','artifact_bytes','input_sha256','source_sha256','completion','launch','status','original_after_exists'):
            assert before[key]==after[key],('changed_during_collection',key)
        current=dict(before['input_sha256'])
        for p,h in {**before['source_sha256'],**before['artifacts']}.items():
            assert p not in current or current[p]==h,('conflicting_pin',p);current[p]=h
        manifest=dict(schema='cap10_mechanism_identity_failure_preservation_20261025',status='PRESERVED_REQUIRES_INDEPENDENT_BOUNDARY_REVIEW',
          started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),remote_root=ROOT,local_payload=str(payload),
          files=local,sha256=copied,artifact_bytes=before['artifact_bytes'],
          hash_only_inputs={p:h for p,h in current.items() if p not in local},remote_before_sha256=current,remote_after_sha256=current,
          input_sha256=before['input_sha256'],source_sha256=before['source_sha256'],
          remote_inventory_before_exit=0,remote_inventory_after_exit=0,rsync_exit=0,
          inputs_and_sources_unchanged=True,all_copied_bytes_SHA_verified=True,original_after_exists=before['original_after_exists'],
          completion=before['completion'],launch=before['launch'],PID_observations=after['PID_observations'],
          collection_executions=attempts,prediction_values_parsed=False,metrics_values_interpreted=False,
          model_calls=0,training=False,test_read=False,automatic_retry=False,collector_sha256=own_sha)
        assert sha(own)==own_sha,'collector_source_changed'
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'EXIT_CODE').write_text('0\n')
        print(json.dumps(dict(status='PRESERVED',files=len(local),input_pins=213,source_pins=25,manifest=str(out/'manifest.json'),manifest_sha256=sha(out/'manifest.json'))))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),executions=attempts),indent=2)+'\n');(out/'EXIT_CODE').write_text('1\n');raise

if __name__=='__main__':main()

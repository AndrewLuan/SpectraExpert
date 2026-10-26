#!/usr/bin/env python3
"""功能概述：收集cap10受控研究一个已完成stage的原字节证据，不加载张量、不计算指标。
主要职责：remote_inventory通过SSH核实严格双层exit0、全部job完成及launcher退出，
再列出run、job输出、访问日志、queue、review与本阶段锁定源码的SHA；main以rsync
只复制该清单并复核本地及远端SHA。任何未完成、符号链接、越界路径或SHA变化均拒绝。
代码库关系：消费supervise_controlled_queue.py/launch_controlled_stage.py的收据；
由root经run_screened.py显式调用。科学输入的大型cache不复制，旧失败run不覆盖。
输入输出与副作用：--queue为JJ的本阶段queue；--out为不存在的本地收集目录。
JJ只读返回metadata inventory，在本地新建原字节证据、manifest和EXIT_CODE；无模型调用、
无外部LLM API、无自动重试、无远端科学输出修改。test阶段必须已有queue冻结锁定。
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,pathlib,shlex,subprocess,sys

REMOTE = r"""
import datetime,hashlib,json,os,pathlib,sys
qp=pathlib.Path(sys.argv[1]).resolve();q=json.loads(qp.read_text())
root=pathlib.Path(q['root']).resolve();run_name=q.get('run_name',q['phase'])
run=root/'runs'/run_name
assert q['phase'] in ('train','validation','mechanism','test')
if q['phase']=='test': assert q['selection_locked_before_test'] is True
def read(p): return json.loads(p.read_text())
def zero(v): assert type(v) is int and v==0
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
c=read(run/'completion.json');l=read(run/'launch_receipt.json')
assert c['status']=='COMPLETE' and c['inputs_unchanged'] is True
zero(c['external_exit']);zero(l['internal_exit']);zero(l['external_exit'])
assert (run/'EXIT_CODE').read_text().strip()=='0'
assert (run/'EXTERNAL_EXIT_CODE').read_text().strip()=='0'
qh=sha(qp);assert c['queue_sha256']==qh and l['queue_sha256']==qh
assert len(c['jobs'])==len(q['jobs'])
for j in c['jobs']:zero(j['internal_exit']);zero(j['external_exit'])
proc=pathlib.Path('/proc')/str(l['wrapper_pid'])/'stat'
if proc.exists():
 assert proc.read_text().rsplit(')',1)[1].strip().split()[0]=='Z','launcher_still_live'
before=read(run/'input_fingerprints.before.json');after=read(run/'input_fingerprints.after.json')
assert before==after==q['input_sha256']
paths=set()
def include(p):
 p=pathlib.Path(p)
 assert p.exists(),str(p)
 if p.is_dir():
  for f in p.rglob('*'):
   if f.is_file():include(f)
 else:
  assert not p.is_symlink()
  f=p.resolve();assert f.is_relative_to(root),str(f)
  paths.add(f)
include(run)
for j in q['jobs']:include(j['job_out']);include(j['access_log'])
include(qp);include(q['bindings'])
for p in q['review_paths']:include(p)
for p in q['source_sha256']:
 if pathlib.Path(p).resolve().is_relative_to(root):include(p)
for n in [run_name+'.dispatch.json',run_name+'.launch_claim.json',run_name+'.operator_claim.json',run_name+'.launcher.stdout.log']:
 p=root/n
 if p.exists():include(p)
if q['phase']=='train':
 for n in ['actual_train_recovery_deployment_preflight.json','JJ_recovery_synthetic_receipt.json','JJ_recovery_synthetic.log']:
  include(root/n)
 include(root/'audits/training')
 for n in ['audit_cap_training.py','cap_training_audit_implementation_review.json','cap_training_audit_author_receipt.json','JJ_training_audit_implementation_synthetic_receipt.json','JJ_training_audit_implementation_synthetic.log','actual_training_audit_execution_receipt.json','actual_training_audit_ssh_receipt.json','actual_training_audit_dispatch.json','actual_training_audit.log','zero_budget_failure_independent_review.json']:
  include(root/n)
assert all(not any(x in f.relative_to(root).parts for x in ['sealed','attempt_history']) for f in paths)
artifacts={str(p.relative_to(root)):sha(p) for p in sorted(paths)}
expected_inputs=dict(q['input_sha256'])
if q['phase']=='train':
 for p,h in read(root/'audits/training/summary.json')['input_sha256'].items():
  assert p not in expected_inputs or expected_inputs[p]==h
  expected_inputs[p]=h
actual_inputs={p:sha(pathlib.Path(p)) for p in expected_inputs}
assert actual_inputs==expected_inputs,'actual_collection_input_SHA_changed'
print(json.dumps({'schema':'cap10_stage_collection_inventory_20261025','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'run_name':run_name,'phase':q['phase'],'queue_sha256':qh,'artifacts':artifacts,'full_input_sha256':actual_inputs,'all_real_exits_zero':True,'payload_values_interpreted':False},sort_keys=True))
"""

def remote_inventory(queue: str) -> dict:
    argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','JJ',
          shlex.join(['/home/user/.venvs/spectrexpert/bin/python','-B','-c',REMOTE,queue])]
    p=subprocess.run(argv,check=True,capture_output=True,text=True)
    return json.loads(p.stdout)

def sha(p: pathlib.Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=pathlib.Path(a.out).resolve()
    assert not out.exists(),'collection_destination_already_exists'
    inv=remote_inventory(a.queue)
    out.mkdir(parents=True);payload=out/'remote';payload.mkdir()
    (out/'inventory.before.json').write_text(json.dumps(inv,indent=2)+'\n')
    names=out/'payload.files.txt';names.write_text(''.join(k+'\n' for k in inv['artifacts']))
    argv=['rsync','-az','--files-from='+str(names),'-e','ssh -o BatchMode=yes -o ConnectTimeout=10',
          'JJ:'+inv['root']+'/',str(payload)+'/']
    x=subprocess.run(argv,check=False)
    assert x.returncode==0,'rsync_failed'
    actual={k:sha(payload/k) for k in inv['artifacts']}
    assert actual==inv['artifacts'],'local_payload_sha_mismatch'
    after=remote_inventory(a.queue)
    assert after['artifacts']==inv['artifacts'] and after['full_input_sha256']==inv['full_input_sha256'],'remote_changed_while_collecting'
    (out/'inventory.after.json').write_text(json.dumps(after,indent=2)+'\n')
    manifest={**inv,'local_payload':str(payload),'local_sha256':actual,'rsync_exit':x.returncode,
              'source_sha256':sha(pathlib.Path(__file__)),'remote_stable_during_collection':True}
    files={inv['root']+'/'+k:str(payload/k) for k in actual}
    hashes={inv['root']+'/'+k:h for k,h in actual.items()}
    runrel='runs/'+inv['run_name']
    oldbefore=json.loads((payload/runrel/'input_fingerprints.before.json').read_text())
    oldafter=json.loads((payload/runrel/'input_fingerprints.after.json').read_text())
    c_before=dict(hashes);c_after=dict(hashes)
    assert all(inv['full_input_sha256'].get(p)==h for p,h in oldbefore.items())
    assert all(after['full_input_sha256'].get(p)==h for p,h in oldafter.items())
    for target,values in ((c_before,inv['full_input_sha256']),(c_after,after['full_input_sha256'])):
        for path,digest in values.items():
            assert path not in target or target[path]==digest
            target[path]=digest
    assert c_before==c_after
    manifest.update(files=files,sha256=hashes,hash_only_inputs={p:h for p,h in inv['full_input_sha256'].items() if p not in files},
        remote_before_sha256=c_before,remote_after_sha256=c_after,
        remote_inventory_before_exit=0,remote_inventory_after_exit=0)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'EXIT_CODE').write_text('0\n')
    print(json.dumps({'status':'PASS','phase':inv['phase'],'run_name':inv['run_name'],
                      'artifacts':len(actual),'manifest':str(out/'manifest.json')}))
    return 0

if __name__=='__main__':
    raise SystemExit(main())

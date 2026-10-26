#!/usr/bin/env python3
"""功能概述：收集native cost恢复的显式准入metadata，不加载模型或解析科学payload。
main消费root填写的request、既有245项inventory、失败与诊断保全件，补齐新binding/q/source
和独审/服务器测试收据；所有复制放在新准入目录内，生成before/after实际SHA及copy map。
仅对明确登记的train/validation文件做SHA，拒绝sealed/test缓存与预测文件；.pt只hash不复制。
输入输出：--request填实pathSHA；--out不得存在。失败保留已有收集，不派发任何科学任务。
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,pathlib,shlex,shutil,subprocess
P=pathlib.Path
MR='/home/user/IRfingerprints/SpectrExpert/outputs/nmr_cap10_controlled_20261024T163633Z'
RP='/home/user/.venvs/spectrexpert/bin/python'
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(P(p).read_text())
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def refs(x):
 if isinstance(x,dict):
  if set(('path','sha256'))<=set(x):yield x
  else:
   for y in x.values():yield from refs(y)
 elif isinstance(x,list):
  for y in x:yield from refs(y)
def allowed(p):
 z=P(p);assert z.is_absolute() and '..' not in z.parts and str(z).startswith('/home/user/IRfingerprints/SpectrExpert/')
 assert not any(s=='sealed' or s.startswith('sealed-') or s=='test' for s in z.parts)
 assert 'prediction' not in z.name.lower() and z.name!='cohorts.pt'
def rsync(names,out):
 assert names
 f=out/'fetch.files.txt';f.write_text(''.join(p[len(MR)+1:]+'\n' for p in names))
 for p in names:assert p.startswith(MR+'/')
 x=subprocess.run(['rsync','-az','--files-from='+str(f),'-e','ssh -o BatchMode=yes -o ConnectTimeout=10','JJ:'+MR+'/',str(out/'fetched')+'/'])
 assert x.returncode==0
def inventory(expected,request_path,request_sha):
 code="""import pathlib,json,hashlib,sys,datetime
p=pathlib.Path(sys.argv[1]);assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]
e=json.loads(p.read_text())['expected_sha256']
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
before={p:sha(p) for p in e};assert before==e
after={p:sha(p) for p in e};assert after==e
print(json.dumps(dict(actual_exit=0,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs_unchanged=True,test_read=False,model_calls=0,tensor_deserialization=False,remote_before_sha256=before,remote_after_sha256=after)))
"""
 a=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','JJ',shlex.join([RP,'-B','-c',code,request_path,request_sha])]
 x=subprocess.run(a,capture_output=True,text=True);assert x.returncode==0,x.stderr
 v=json.loads(x.stdout);assert v['remote_before_sha256']==v['remote_after_sha256']==expected
 return v,x.returncode
def main():
 ap=argparse.ArgumentParser()
 for k in ('request','out'):ap.add_argument('--'+k,required=True)
 a=ap.parse_args();m=P(__file__).resolve().parent;req=read(a.request);out=P(a.out).resolve();assert not out.exists();out.mkdir(parents=True);(out/'fetched').mkdir()
 assert req['schema']=='cap10_native_cost_evaluation_admission_request_20261025' and req['root']==MR
 base=read(m/'pre_admission/artifact_map_with_aliases.json');assert base['remote_before_sha256']==base['remote_after_sha256']
 expected=dict(base['remote_before_sha256']);copies=dict(base['files'])
 def pin(p,h,local=None):
  allowed(p);assert len(h)==64
  assert p not in expected or expected[p]==h,(p,'conflicting_expected_SHA')
  expected[p]=h
  if local:copies[p]=str(local)
 failed=read(m/'attempt_history/validation_finite_cost/manifest.json')
 for rel in ('runs/validation/completion.json','runs/validation/launch_receipt.json'):
  pin(MR+'/'+rel,failed['artifacts'][rel],P(failed['local_payload'])/rel)
 d=m/'diagnostic_cost_evidence'
 for rel in ('diagnostics/validation_cost_contract_row0/summary.json','actual_validation_cost_diagnostic_execution_receipt.json','access_logs/diagnostics/validation_cost_contract_row0.json'):
  pin(MR+'/'+rel,sha(d/rel),d/rel)
 for v in refs(req):pin(v['path'],v['sha256'])
 fresh=[p for p in expected if p not in copies and p.startswith(MR+'/') and P(p).suffix in ('.json','.log','.py','.txt') and p not in base['hash_only_inputs']]
 rsync(fresh,out)
 for p in fresh:
  f=out/'fetched'/p[len(MR)+1:];assert sha(f)==expected[p];copies[p]=str(f)
 def doc(p):return read(copies[p])
 binding=doc(req['bindings']['path'])
 for p,h in {**binding['input_sha256'],**binding['source_sha256']}.items():pin(p,h)
 for v in req['queues'].values():
  queue=doc(v['path'])
  for p,h in queue['source_sha256'].items():pin(p,h)
 jj=doc(req['evaluation_JJ_execution']['path'])
 assert jj['source_sha256']==jj['source_end_sha256'] and type(jj['actual_exit'])is int and jj['actual_exit']==0
 assert jj['test_source_and_review_unchanged'] is True
 for p,h in {**jj['source_sha256'],**jj['test_source_and_review_sha256']}.items():pin(p,h)
 fresh=[p for p in expected if p not in copies and p.startswith(MR+'/') and P(p).suffix in ('.json','.log','.py','.txt') and p not in base['hash_only_inputs']]
 if fresh:
  rsync(fresh,out)
  for p in fresh:
   f=out/'fetched'/p[len(MR)+1:];assert sha(f)==expected[p];copies[p]=str(f)
 for p in expected:allowed(p)
 inventory_request=out/'inventory.request.json';dump(inventory_request,dict(expected_sha256=expected))
 remote_request=MR+'/native_cost_admission_inventory_request.json'
 x=subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=10',str(inventory_request),'JJ:'+remote_request]);assert x.returncode==0
 inv,ssh_exit=inventory(expected,remote_request,sha(inventory_request));dump(out/'actual_inventory.before.json',inv)
 payload=out/'payload';files={};hashes={}
 for p,src in copies.items():
  assert sha(src)==expected[p]
  if P(p).suffix=='.pt':continue
  dest=payload/p.lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);assert sha(dest)==expected[p]
  files[p]=str(dest);hashes[p]=expected[p]
 after,ssh_exit2=inventory(expected,remote_request,sha(inventory_request));dump(out/'actual_inventory.after.json',after)
 assert inv['remote_before_sha256']==after['remote_after_sha256']
 v=dict(schema='cap10_evaluation_admission_artifact_map_20261025',files=files,sha256=hashes,
  hash_only_inputs={p:h for p,h in expected.items() if p not in files},
  remote_before_sha256=inv['remote_before_sha256'],remote_after_sha256=after['remote_after_sha256'],
  actual_inventory=dict(path=str(out/'actual_inventory.after.json'),sha256=sha(out/'actual_inventory.after.json'),ssh_actual_exit=ssh_exit2),
  current_source_input_unchanged=True,test_read=False,source_sha256={str(P(__file__).resolve()):sha(__file__)},
  provenance=dict(original_artifact_map=str(m/'pre_admission/artifact_map_with_aliases.json'),failed_manifest=str(m/'attempt_history/validation_finite_cost/manifest.json'),request=str(P(a.request).resolve())),
  actual_inventory_first_ssh_exit=ssh_exit)
 dump(out/'artifact_map.json',v)
 print(json.dumps(dict(status='COLLECTED_FOR_INDEPENDENT_ADMISSION',copied=len(files),hash_only=len(v['hash_only_inputs']),total=len(expected),source_inputs_unchanged=True,artifact_map=str(out/'artifact_map.json'))))
if __name__=='__main__':main()

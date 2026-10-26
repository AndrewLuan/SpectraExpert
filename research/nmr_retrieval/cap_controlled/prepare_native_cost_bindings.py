#!/usr/bin/env python3
"""功能概述：为已保全的 validation cost 契约失败创建独立恢复 binding 与固定16+4队列。
主要职责：读取原冻结 registry、实际失败收据、独审和服务器回归收据，逐字节核原输入、
原源码和修复源码；只替换评价源码位置、绑定文件及输出位置，完整保留8fit、缓存和选择规则。
输入输出：--root 指本轮目录；三个 review 参数是显式文件。不加载张量、不评价、不重训。
新 metadata 使用 native_cost 名称，原失败文件与原源均不覆盖；由独立准入后唯一 operator 执行。
"""
from __future__ import annotations
import argparse,copy,datetime,hashlib,json,pathlib
OLD_BINDING_SHA='61ec3d6066b1207bef3ea3b760f8b5ded715c89d834150332193f3991decee6c'
EVAL=('cap_evaluation_common.py','evaluate_cap_ranker.py','summarize_cap_validation.py','analyze_cap_mechanism.py')
OLD_Q={'validation':'42b15d451e7ca0d35d19ab442e12828fc014ab015a0f23b869e628217c4c6457','mechanism':'b68f3e60d69975c74f51e39ea8f12a25c3881d1a05fef2eb657afcdb88dbe7a6'}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
 with p.open('x') as f:f.write(json.dumps(v,indent=2)+'\n')
def zero(v):assert type(v)is int and v==0
def main():
 ap=argparse.ArgumentParser()
 for n in ('root','implementation-review','jj-tests-receipt','failure-review'):ap.add_argument('--'+n,required=True)
 a=ap.parse_args();r=pathlib.Path(a.root).resolve();old=r/'validation_bindings.json'
 assert sha(old)==OLD_BINDING_SHA
 outnames=['native_cost_validation_bindings.json','validation_native_cost_queue.json','mechanism_native_cost_queue.json','native_cost_metadata_preparation_receipt.json']
 assert all(not(r/n).exists() for n in outnames)
 b=read(old);p=read(r/'protocol.json')
 assert p['test_allowed']is False and p['promotion']is False
 run=r/'runs/validation';failed=read(run/'completion.json');outer=read(run/'launch_receipt.json')
 assert failed['status']=='FAILED' and type(outer['internal_exit'])is int and outer['internal_exit']==1
 assert type(outer['external_exit'])is int and outer['external_exit']==1
 assert outer['queue_sha256']==OLD_Q['validation']
 reviews=[pathlib.Path(a.implementation_review).resolve(),pathlib.Path(a.failure_review).resolve()]
 for rp in reviews:
  v=read(rp);assert v['status']=='PASS' and v['independent']is True
 ir=read(reviews[0]);jrpath=pathlib.Path(a.jj_tests_receipt).resolve();jr=read(jrpath)
 zero(jr['actual_exit']);assert jr['source_sha256']==jr['source_end_sha256']
 original_input=copy.deepcopy(b['input_sha256'])
 for path,h in original_input.items():assert sha(path)==h,('original_input_changed',path)
 for path,h in b['source_sha256'].items():assert sha(path)==h,('original_source_changed',path)
 sources=dict(b['source_sha256']);inputs=dict(original_input);replacements={}
 for name in EVAL:
  oldpath=str(r/name);newpath=r/'native_cost_evaluation'/name;h=sha(newpath)
  assert ir['reviewed_sha256'].get(str(newpath))==h
  assert jr['source_sha256'].get(str(newpath))==h
  assert oldpath in sources
  sources.pop(oldpath);sources[str(newpath)]=h;inputs[str(newpath)]=h
  replacements[oldpath]=str(newpath)
 for rp in reviews+[jrpath]:inputs[str(rp)]=sha(rp)
 b.update(source_sha256=sources,input_sha256=inputs,evaluation_implementation=ref(reviews[0]),
  evaluation_JJ_tests=ref(jrpath),failure_recovery_review=ref(reviews[1]),prior_validation_bindings=ref(old),
  evaluation_source_root=str(r/'native_cost_evaluation'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  input_scope='unchanged audited fits and caches; independently reviewed native cost contract recovery')
 bp=r/'native_cost_validation_bindings.json';write(bp,b)
 for phase in ('validation','mechanism'):
  oldq=r/(phase+'_queue.json');assert sha(oldq)==OLD_Q[phase];q=read(oldq)
  name=phase+'_native_cost';q['run_name']=name;q['bindings']=str(bp)
  q['input_sha256']=inputs;q['source_sha256']=dict(sources)
  for f in ('protocol.json','isolation_entry.py','supervise_controlled_queue.py','launch_controlled_stage.py'):
   q['source_sha256'][str(r/f)]=sha(r/f)
  q['source_sha256'][str(bp)]=sha(bp)
  q['review_paths']=[str(r/(name+'_admission_independent_review.json'))]
  q['recovery_from']=dict(queue=ref(oldq),failed_validation_run=str(run),failure_review=ref(reviews[1]),automatic_retry=False)
  for j in q['jobs']:
   original_out=j['job_out'];original_access=j['access_log']
   target=str(r/name/j['id']);access=str(r/'access_logs'/name/(j['id']+'.json'))
   mapping={**replacements,str(old):str(bp),original_out:target,original_access:access}
   j['command']=[mapping.get(x,x) for x in j['command']]
   j['argv']=[mapping.get(x,x) for x in j['argv']]
   j['job_out']=target;j['access_log']=access
   assert not pathlib.Path(target).exists() and not pathlib.Path(access).exists()
  assert len(q['jobs'])==(16 if phase=='validation' else 4)
  assert not(r/'runs'/name).exists()
  write(r/(name+'_queue.json'),q)
 assert all(sha(path)==h for path,h in original_input.items())
 receipt=dict(status='PREPARED_REQUIRES_INDEPENDENT_ADMISSION',test_reads=0,model_calls=0,training=False,
  scientific_configuration_changed=False,automatic_retry=False,source_sha256={str(pathlib.Path(__file__).resolve()):sha(__file__)},
  prior_bindings=ref(old),files={n:sha(r/n) for n in outnames[:3]},input_count=len(inputs))
 write(r/outnames[3],receipt);print(json.dumps(receipt))
if __name__=='__main__':main()

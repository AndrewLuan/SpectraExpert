#!/usr/bin/env python3
"""功能概述：为GPU资源有界就绪恢复创建独立16val/4机制队列，科学binding与leaf完全沿用。
main核原binding/两queue、已保全失败run、独立operator与纯reader实施review和锁定恢复策略。
只新增operator_paths/resource_wait、分析源、独立运行/输出/访问位置；不加载模型、缓存或指标。
--root等参数仅显式metadata路径；不覆盖旧队列/失败/已成功4cell。真实完整SHA与准入在后续独审，
本工具本身不授权执行，不自动重试。213科学输入、15leaf源码与原训练8fit均保持原值。
"""
from __future__ import annotations
import argparse,copy,datetime,hashlib,json,pathlib
OLD_BINDING='2f56b0470a15b9746851aa4779c422ad4a8816af70f532450e18d75a9b3d556d'
OLD_Q={'validation':'637383cb6e719c97150747d5e53df53a2fe21c7b3a15b832765964ea50e0a3dc','mechanism':'f0cd9f05bd409198526de2eff19322b07ccd2cf8d69d367f7d6112f452407571'}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
 with pathlib.Path(p).open('x') as f:f.write(json.dumps(v,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser()
 for n in ('root','operator-review','reader-review','failure-review','strategy'):ap.add_argument('--'+n,required=True)
 a=ap.parse_args();r=pathlib.Path(a.root).resolve();bp=r/'native_cost_validation_bindings.json'
 assert sha(bp)==OLD_BINDING
 b=read(bp);assert len(b['input_sha256'])==213 and len(b['source_sha256'])==15
 refs=[pathlib.Path(x).resolve() for x in [a.operator_review,a.reader_review,a.failure_review]]
 for p in refs:
  v=read(p);assert v['status']=='PASS' and v['independent']is True
 op,reader=read(refs[0]),read(refs[1])
 local_root=pathlib.Path(read(r/'protocol.json')['local_root'])
 def reviewed(doc,path,h):
  aliases=(str(path),str(local_root/path.relative_to(r)))
  values=[doc['reviewed_sha256'][x] for x in aliases if x in doc['reviewed_sha256']]
  return bool(values) and all(x==h for x in values)
 strategy=pathlib.Path(a.strategy).resolve();s=read(strategy)
 assert s['status']=='LOCKED_INFRASTRUCTURE_RECOVERY_STRATEGY_NOT_EXECUTION_ADMISSION'
 assert s['recovery']=='all16_fresh_in_one_new_run_after_operator_and_actual_queue_independent_PASS'
 assert s['rerun_choice_based_on_metrics']is False and s['new_training']is False and s['test_read']is False
 failed=r/'runs/validation_native_cost';c=read(failed/'completion.json');outer=read(failed/'launch_receipt.json')
 assert c['status']=='FAILED' and 'gpu_not_idle' in c['error']
 assert type(outer['internal_exit'])is int and outer['internal_exit']==1 and type(outer['external_exit'])is int and outer['external_exit']==1
 assert len(c['jobs'])==8 and sum(type(j['internal_exit'])is int and j['internal_exit']==0 and type(j['external_exit'])is int and j['external_exit']==0 for j in c['jobs'])==4
 operators={}
 for role,name in [('supervisor','supervise_resource_ready_queue.py'),('launcher','launch_resource_ready_stage.py')]:
  p=r/name;h=sha(p);assert reviewed(op,p,h);operators[role]=ref(p)
 reader_sources={}
 for name in ('cap_evaluation_common.py','summarize_cap_validation.py','analyze_cap_mechanism.py'):
  p=r/'resource_ready_analysis'/name;h=sha(p);assert reviewed(reader,p,h);reader_sources[str(p)]=h
 outnames=[phase+'_resource_ready_queue.json' for phase in ('validation','mechanism')]
 assert all(not(r/n).exists() for n in outnames+['resource_ready_metadata_preparation_receipt.json'])
 for phase in ('validation','mechanism'):
  oldq=r/(phase+'_native_cost_queue.json');assert sha(oldq)==OLD_Q[phase];q=read(oldq)
  assert q['bindings']==str(bp) and q['input_sha256']==b['input_sha256']
  name=phase+'_resource_ready';assert not(r/'runs'/name).exists()
  q['run_name']=name;q['operator_paths']=operators;q['resource_wait']={'maximum_seconds':120,'poll_seconds':.2}
  q['source_sha256'].update({x['path']:x['sha256'] for x in operators.values()});q['source_sha256'].update(reader_sources)
  assert len(q['source_sha256'])==25
  q['review_paths']=[str(r/(name+'_admission_independent_review.json'))]
  q['analysis_tools']={'source_root':str(r/'resource_ready_analysis'),'implementation_review':ref(refs[1])}
  q['operator_implementation_review']=ref(refs[0])
  q['recovery_from']={'prior_queue':ref(oldq),'failed_validation_run':str(failed),'failure_review':ref(refs[2]),'strategy':ref(strategy),'automatic_retry':False,'prior_completed_cells_used_for_selection':False}
  for j in q['jobs']:
   out=str(r/name/j['id']);access=str(r/'access_logs'/name/(j['id']+'.json'))
   mapping={j['job_out']:out,j['access_log']:access}
   j['argv']=[mapping.get(x,x) for x in j['argv']];j['command']=[mapping.get(x,x) for x in j['command']]
   j['job_out']=out;j['access_log']=access
   assert not pathlib.Path(out).exists() and not pathlib.Path(access).exists()
  assert q['bindings']==str(bp) and q['input_sha256']==b['input_sha256'] and q['max_parallel']==8 and q['maximum_seconds']==10800
  assert len(q['jobs'])==(16 if phase=='validation' else 4)
  write(r/(name+'_queue.json'),q)
 assert sha(bp)==OLD_BINDING
 receipt=dict(status='PREPARED_REQUIRES_INDEPENDENT_ADMISSION',scientific_inputs_hashed=False,scientific_inputs_changed=False,
  original_scientific_binding=ref(bp),scientific_input_count=213,active_qsource_count=25,training=False,test_read=False,
  model_calls=0,automatic_retry=False,files={n:sha(r/n) for n in outnames},source_sha256={str(pathlib.Path(__file__).resolve()):sha(__file__)})
 write(r/'resource_ready_metadata_preparation_receipt.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()

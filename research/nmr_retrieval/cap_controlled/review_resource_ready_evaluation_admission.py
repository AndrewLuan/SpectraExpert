"""功能概述：只读独审资源就绪恢复的16验证/4机制具体队列，不改历史准入工具。

本工具先调用已锁native checker，保留8fit/200审计/206原输入/213当前输入和
首次dtype失败、单行诊断的完整核验；再核第二失败、全16重评策略、两新operator、
三纯reader及真实JJ套件。新队列仅换独立run/out/access及执行元数据，科学leaf不变。
只读已复制JSON/源码/log与远端实际始末SHA库存，不反序列化张量、不读预测或指标，
不启动进程、SSH或科学任务。独审PASS仅授权登记的单阶段具体队列，不替代执行收据。
输入request显式登记全部路径/SHA；输出必须全新，失败仅留failure与EXIT1。
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

BASE_SHA='fafdea31c98ce5237ed806953c17033e50336b470c6ba6dfb4108ce2b32668b7'
BINDING_SHA='2f56b0470a15b9746851aa4779c422ad4a8816af70f532450e18d75a9b3d556d'
NATIVE_QUEUES={'validation':'637383cb6e719c97150747d5e53df53a2fe21c7b3a15b832765964ea50e0a3dc',
 'mechanism':'f0cd9f05bd409198526de2eff19322b07ccd2cf8d69d367f7d6112f452407571'}
OPERATORS={'supervisor':'supervise_resource_ready_queue.py','launcher':'launch_resource_ready_stage.py'}
READERS=('cap_evaluation_common.py','summarize_cap_validation.py','analyze_cap_mechanism.py')

def need(ok,label):
    if not ok:raise ValueError(label)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def source_path(p):
    p=Path(p).absolute();need('..' not in p.parts and not p.is_symlink(),'source_path_canonical')
    r=p.resolve(strict=True)
    need(p==r and r.suffix=='.py','source_path_exact_py');return r

BASE_PATH=source_path(Path(__file__).parent/'review_native_cost_evaluation_admission.py')
need(sha(BASE_PATH.read_bytes())==BASE_SHA,'locked_native_checker_SHA')
spec=importlib.util.spec_from_file_location('locked_native_admission',BASE_PATH)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def reviewed_alias(doc,required,root,protocol,label):
    """只接受精确canonical或protocol.local_root相同相对路径；双键冲突拒绝。"""
    need(doc.get('status')=='PASS' and doc.get('independent') is True,label+'_independent_PASS')
    local=Path(protocol['local_root']);pins=doc.get('reviewed_sha256',{})
    for p,h in required.items():
        path=Path(p);aliases=[str(path)]
        if path.is_relative_to(root):aliases.append(str(local/path.relative_to(root)))
        values=[pins[k] for k in aliases if k in pins]
        need(values and all(v==h for v in values),label+'_current_SHA:'+p)

def verify_JJ(bundle,request,prefix,sources,expected):
    ref=request[prefix+'_JJ_execution'];jj=bundle.ref(ref);base.zero(jj['actual_exit'])
    logref=request[prefix+'_JJ_log'];log=bundle.raw(logref['path'],logref['sha256']).decode()
    count=request['expected_'+prefix+'_JJ_test_count']
    need(type(count) is int and count==expected,prefix+'_exact_suite_count')
    need(re.findall(r'^Ran ([0-9]+) tests? in [^\n]+$',log,re.M)==[str(count)]
      and re.search(r'^OK$',log,re.M) is not None and re.search(r'^FAILED\b',log,re.M) is None,prefix+'_raw_JJ_suite_PASS')
    need(jj['source_sha256']==jj['source_end_sha256'] and
      all(jj['source_sha256'].get(p)==h for p,h in sources.items()),prefix+'_JJ_actual_sources')
    for before,after in (('all_deployed_sha256','all_deployed_end_sha256'),
                         ('immutable_scientific_sha256','immutable_scientific_end_sha256')):
        if before in jj:need(jj[before]==jj[after],prefix+'_JJ_preserved_sources')
    for key in ('actual_ssh_exit','ssh_actual_exit'):
        if key in jj:base.zero(jj[key])
    if 'log_sha256' in jj:need(jj['log_sha256']==logref['sha256'],prefix+'_JJ_log_binding')
    for key in ('actual_test_data_read','test_read'):
        if key in jj:need(jj[key] is False,prefix+'_JJ_no_test')
    return jj

def verify_strategy(bundle,request):
    s=bundle.ref(request['recovery_strategy'])
    expected=dict(schema='cap10_resource_readiness_recovery_strategy_20261025',
      status='LOCKED_INFRASTRUCTURE_RECOVERY_STRATEGY_NOT_EXECUTION_ADMISSION',prior_run='validation_native_cost',
      recovery='all16_fresh_in_one_new_run_after_operator_and_actual_queue_independent_PASS',
      completed_cells=4,cancelled_cells=4,unlaunched_cells=8,logical_validation_matrix_cells=16,
      prior_completed_cells_retained=True,prior_completed_cells_used_for_selection=False,
      prior_partial_predictions_used_for_selection=False,controller_opened_prior_validation_metrics=False,
      rerun_choice_based_on_metrics=False,compare_all_four_prior_completed_cells_against_new_semantic_predictions=True,
      additional_completed_validation_evaluations_expected=4,training_draws_unchanged=4194304,
      optimizer_steps_unchanged=131072,new_training=False,test_read=False,test_reselection_allowed=False,
      promotion=False,official_system_changed=False,old_training_operator_source_bytes_unchanged=True,
      actual_recovery_launch_allowed=False)
    need(all(type(s.get(k)) is type(v) and s[k]==v for k,v in expected.items()),'locked_result_independent_all16_fresh_strategy')
    need(s['resources']==dict(maximum_wait_seconds=120,poll_seconds=.2,stage_deadline_seconds=10800,
      memory_threshold_mib_exclusive=1024,utilization_threshold_percent_exclusive=5,
      foreign_processes='reject and never signal',owned_completed_PID_telemetry='bounded observation with lineage checks'),
      'strict_resource_strategy')
    return s

def verify_second_failure(bundle,root,request):
    f=bundle.ref(request['resource_failure_review'])
    need(f.get('status')=='PASS' and f.get('independent') is True and
      f.get('scope')=='failed_validation_resource_execution_exposure_and_preservation_not_recovery_admission','second_failure_review_scope')
    expected=dict(queue_sha256=NATIVE_QUEUES['validation'],wrapper_pid=1692993,driver_pid=1692998,
      copied_artifacts=82,actual_input_pins_preserved=213,completed_validation_cells=4,
      completed_validation_records=3616,cancelled_partial_tune_cells=4,unlaunched_C10_cells=8,
      training_draws_added=0,training_optimizer_steps_added=0,training_total_draws_remains=4194304,
      training_total_steps_remains=131072,metrics_values_interpreted=False,actual_test_payload_read=False,
      measured_utilization_memory_apps_at_failed_guard_available=False,selection_performed=False,
      automatic_retry=False,scientific_recovery_admission=False)
    need(all(type(f.get(k)) is type(v) and f[k]==v for k,v in expected.items()),'second_failure_actual_exposure_accounting')
    run=root/'runs/validation_native_cost';done=bundle.json(run/'completion.json');outer=bundle.json(run/'launch_receipt.json')
    need(done['status']=='FAILED' and done['error']=="AssertionError(('gpu_not_idle', 0))",'actual_failed_resource_guard_only')
    need(done['pid']==outer['child_pid']==1692998 and outer['wrapper_pid']==1692993,'second_failed_PID')
    need(done['queue_sha256']==outer['queue_sha256']==NATIVE_QUEUES['validation'],'second_failed_queue_SHA')
    for v in (done['external_exit'],outer['external_exit'],outer['internal_exit']):need(type(v) is int and v==1,'second_failure_strict_double1')
    jobs=done['jobs'];need(len(jobs)==8 and len({j['id'] for j in jobs})==8,'second_failed_eight_launched')
    successes={j['id'] for j in jobs if type(j['external_exit']) is int and j['external_exit']==0 and type(j['internal_exit']) is int and j['internal_exit']==0}
    wanted={f'validation_C5_s{s}_{p}_original_val' for s in (193,211) for p in ('single','dual')}
    need(successes==wanted and successes=={j['id'] for j in f['completed_cells']},'four_completed_C5_original_val_only')
    for j in jobs:
        if j['id'] not in successes:
            need(j['id'].endswith('_tune') and j['id'].startswith('validation_C5_') and
              j['internal_exit'] is None and type(j['external_exit']) is int and j['external_exit']==-15,'four_owned_partial_cancellation')
    for rel in ('validation_native_cost_queue.json','runs/validation_native_cost/completion.json','runs/validation_native_cost/launch_receipt.json'):
        p=root/rel;base.reviewed_preserved_alias(f,p,bundle.hashes[str(p)],'/attempt_history/validation_resource_not_ready/remote/'+rel)
    return f

def verify_new_queue(bundle,root,binding,protocol,request,phase,operators,readers):
    ref=request['queues'][phase];path=root/(phase+'_resource_ready_queue.json');name=phase+'_resource_ready'
    need(ref==dict(path=str(path),sha256=bundle.hashes[str(path)]),'explicit_resource_queue_ref')
    need(request['runs'][phase]==str(root/'runs'/name) and request['review_paths'][phase]==str(root/(name+'_admission_independent_review.json')),'explicit_resource_run_review')
    q=bundle.ref(ref);oldpath=root/(phase+'_native_cost_queue.json');old=bundle.json(oldpath,NATIVE_QUEUES[phase])
    need(q['bindings']==str(root/'native_cost_validation_bindings.json') and q['input_sha256']==binding['input_sha256'] and len(q['input_sha256'])==213,'unchanged_213_scientific_inputs')
    need(q['operator_paths']=={role:dict(path=str(root/file),sha256=operators[str(root/file)]) for role,file in OPERATORS.items()},'exact_new_operator_paths')
    need(q['resource_wait']==dict(maximum_seconds=120,poll_seconds=.2) and type(q['resource_wait']['maximum_seconds']) is int and type(q['resource_wait']['poll_seconds']) is float,'exact_120_point2_resource_policy')
    need(q['source_sha256']=={**old['source_sha256'],**operators,**readers} and len(q['source_sha256'])==25,'exact25_source_union')
    need(q['run_name']==name and q['review_paths']==[request['review_paths'][phase]],'resource_name_and_review_path')
    need(q['analysis_tools']==dict(source_root=str(root/'resource_ready_analysis'),implementation_review=request['reader_implementation_review']), 'exact_resource_reader_binding')
    need(q['operator_implementation_review']==request['operator_implementation_review'],'exact_operator_review_binding')
    need(q['recovery_from']==dict(prior_queue=dict(path=str(oldpath),sha256=NATIVE_QUEUES[phase]),
      failed_validation_run=str(root/'runs/validation_native_cost'),failure_review=request['resource_failure_review'],
      strategy=request['recovery_strategy'],automatic_retry=False,prior_completed_cells_used_for_selection=False),'transparent_second_recovery_binding')
    need(len(q['jobs'])==len(old['jobs'])==(16 if phase=='validation' else 4),'same_registered_matrix_size')
    for oldjob,new in zip(old['jobs'],q['jobs']):
        out=str(root/name/oldjob['id']);access=str(root/'access_logs'/name/(oldjob['id']+'.json'))
        replacements={oldjob['job_out']:out,oldjob['access_log']:access}
        expected=dict(oldjob,job_out=out,access_log=access,
          command=[replacements.get(x,x) for x in oldjob['command']],argv=[replacements.get(x,x) for x in oldjob['argv']])
        need(new==expected,'only_output_access_changed_no_scientific_change')
    changed={'run_name','source_sha256','review_paths','jobs','operator_paths','resource_wait','analysis_tools','operator_implementation_review','recovery_from'}
    need({k:v for k,v in q.items() if k not in changed}=={k:v for k,v in old.items() if k not in changed},'no_other_queue_changes')
    for p,h in q['source_sha256'].items():bundle.raw(p,h);bundle.pin(p,h)
    for p,h in q['input_sha256'].items():bundle.pin(p,h)
    return q

def verify(bundle,request):
    need(request['schema']=='cap10_resource_ready_evaluation_admission_request_20261025','resource_request_schema')
    root=Path(request['root']);protocol=bundle.json(root/'protocol.json')
    bp=str(root/'native_cost_validation_bindings.json');need(request['bindings']==dict(path=bp,sha256=BINDING_SHA),'unchanged_native_binding_SHA')
    binding=bundle.ref(request['bindings'])
    need(request['evaluation_source_root']==str(root/'native_cost_evaluation') and request['analysis_source_root']==str(root/'resource_ready_analysis'),'explicit_scientific_and_reader_roots')
    # 原native核验仍完整执行，只把request的登记阶段重定向到保留原队列，结果不作为此次准入输出。
    prior=copy.deepcopy(request);prior['schema']='cap10_native_cost_evaluation_admission_request_20261025'
    for phase in ('validation','mechanism'):
        p=str(root/(phase+'_native_cost_queue.json'));prior['queues'][phase]=dict(path=p,sha256=NATIVE_QUEUES[phase])
        prior['runs'][phase]=str(root/'runs'/(phase+'_native_cost'));prior['review_paths'][phase]=str(root/(phase+'_native_cost_admission_independent_review.json'))
    base.verify(bundle,prior)
    verify_second_failure(bundle,root,request);verify_strategy(bundle,request)
    operators={str(root/name):bundle.hashes[str(root/name)] for name in OPERATORS.values()}
    readers={str(root/'resource_ready_analysis'/name):bundle.hashes[str(root/'resource_ready_analysis'/name)] for name in READERS}
    impl=bundle.ref(request['operator_implementation_review']);need(impl.get('scope')=='cap10_resource_ready_operator_and_launcher_implementation','operator_implementation_scope')
    reviewed_alias(impl,operators,root,protocol,'operator_implementation')
    impl=bundle.ref(request['reader_implementation_review']);need(impl.get('scope')=='cap10_resource_ready_validation_and_mechanism_readers_implementation','reader_implementation_scope')
    reviewed_alias(impl,readers,root,protocol,'reader_implementation')
    verify_JJ(bundle,request,'operator',operators,43);verify_JJ(bundle,request,'reader',readers,43)
    outputs={}
    for phase in ('validation','mechanism'):
        q=verify_new_queue(bundle,root,binding,protocol,request,phase,operators,readers);p=request['queues'][phase]['path']
        outputs[phase]=dict(schema='cap10_resource_ready_evaluation_admission_independent_review_20261025',
          status='PASS',independent=True,scope='exact_'+phase+'_resource_ready_recovery_after_two_preserved_failures',
          admission_phase=phase,reviewed_sha256={**q['source_sha256'],p:bundle.hashes[p]},jobs=len(q['jobs']),
          training_run=binding['training_run'],verified_current_input_sha256=q['input_sha256'],operator_paths=q['operator_paths'],
          resource_wait=q['resource_wait'],analysis_tools=q['analysis_tools'],training_results_review=request['training_results_review'],
          evaluation_implementation_review=request['evaluation_implementation_review'],operator_implementation_review=request['operator_implementation_review'],
          reader_implementation_review=request['reader_implementation_review'],operator_JJ_execution=request['operator_JJ_execution'],
          reader_JJ_execution=request['reader_JJ_execution'],resource_failure_review=request['resource_failure_review'],
          recovery_strategy=request['recovery_strategy'],all16_fresh=True,prior_four_successes_used_for_selection=False,
          compare_all_four_prior_successful_semantic_predictions_required=True,original213_scientific_inputs_preserved=True,
          prior_training200_pins_preserved=True,test_reads=0,new_training_jobs=0,selection_performed=False,
          automatic_retry=False,promotion=False,independent_acceptance_status='unavailable')
    return outputs

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();own=base.safe(__file__);own_sha=sha(own.read_bytes());request_path=base.safe(args.request)
    raw=request_path.read_bytes();request_sha=sha(raw);out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        request=json.loads(raw);bundle=base.Bundle(request['artifact_map']);outputs=verify(bundle,request);bundle.finish()
        need(sha(source_path(BASE_PATH).read_bytes())==BASE_SHA and sha(base.safe(own).read_bytes())==own_sha
          and sha(base.safe(request_path).read_bytes())==request_sha,'checker_or_request_drift')
        for phase,doc in outputs.items():
            doc.update(utc=datetime.now(timezone.utc).isoformat(),source_sha256={str(own):own_sha,str(BASE_PATH):BASE_SHA},
              source_end_sha256={str(own):own_sha,str(BASE_PATH):BASE_SHA},artifact_map_sha256=bundle.map_sha,
              input_sha256={**bundle.reads,str(request_path):request_sha},inputs_unchanged=True)
            (out/(phase+'_resource_ready_admission_independent_review.json')).write_text(json.dumps(doc,indent=2)+'\n')
        (out/'EXIT_CODE').write_text('0\n');print(json.dumps(dict(status='PASS',validation_jobs=16,mechanism_jobs=4)))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc)),indent=2)+'\n');(out/'EXIT_CODE').write_text('1\n');raise

if __name__=='__main__':main()

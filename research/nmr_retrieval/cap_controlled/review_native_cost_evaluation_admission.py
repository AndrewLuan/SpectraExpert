"""功能概述：只读独审native成本契约修复后的16val/4机制恢复准入，原9031checker不变。
严格核原206输入/旧200训练审计闭包不变，原失败74份保全与单行dtype诊断；新4源码只能
迁移至native_cost_evaluation，新binding/run/queue/access/output与旧失败全部隔离。
本工具只用标准库读取显式复制的JSON/源码，并比较远端实际始末inventory的SHA。
不打开张量、预测、test或旧报告，不启动进程/SSH/模型，不新增选择。训练tensor与RNG
事实由已完成独立CPU audit和独立actual结果review提供，本文不声称第二次tensor审计。
request显式登记artifact_map和四份review/执行证明；实际材料未齐时禁止生成PASS。
每phase核精确矩阵、命令、三tensor白名单、成功training_run、8final64/alpha1与中心，
并将该queue全部source+queue字节SHA写入operator要求的reviewed_sha256。
输出目录必须不存在；失败保存failure.json/EXIT_CODE1，不生成任一phase的PASS。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re

EVAL=('cap_evaluation_common.py','evaluate_cap_ranker.py','summarize_cap_validation.py','analyze_cap_mechanism.py')
OLD_BINDING_SHA='61ec3d6066b1207bef3ea3b760f8b5ded715c89d834150332193f3991decee6c'
OLD_QUEUES={'validation':'42b15d451e7ca0d35d19ab442e12828fc014ab015a0f23b869e628217c4c6457',
 'mechanism':'b68f3e60d69975c74f51e39ea8f12a25c3881d1a05fef2eb657afcdb88dbe7a6'}
CHECKS={'same_initial_tensors','frozen_observation','frozen_centers','frozen_scale',
        'matched_exposure','matched_pool_exposure','matched_negative_exposure',
        'strict_negative_qualification','finite','full_budget','real_driver_exit','source_input_unchanged'}
MATRIX=set(itertools.product((5,10),(193,211),('no_aug','aug')))


def need(ok,name):
    if not ok:raise ValueError(name)


def digest(raw):return hashlib.sha256(raw).hexdigest()
def hexsha(x):return type(x) is str and re.fullmatch('[0-9a-f]{64}',x) is not None
def zero(x):need(type(x) is int and x==0,'strict_actual_exit0')


def safe(path):
    raw=Path(path).absolute();need('..' not in raw.parts,'parent_traversal')
    resolved=raw.resolve(strict=True)
    for p in (raw,resolved):
        need(not any(v=='test' or v.startswith(('sealed','test-')) for v in p.parts),'test_input_forbidden')
        need(p.suffix in ('.json','.py','.log','.txt') or p.name in ('EXIT_CODE','EXTERNAL_EXIT_CODE'),'metadata_or_source_only')
        need('prediction' not in p.name.lower() and 'report' not in p.name.lower(),'prediction_or_report_forbidden')
    need(not raw.is_symlink(),'symlink_forbidden');return resolved


class Bundle:
    def __init__(self,path):
        self.map_path=safe(path);raw=self.map_path.read_bytes();self.map_sha=digest(raw);self.map=json.loads(raw)
        self.files=self.map['files'];self.hashes=self.map['sha256'];self.remote=self.map['remote_before_sha256']
        need(type(self.files) is dict and self.files and set(self.files)==set(self.hashes),'copy_map_exact')
        need(self.remote==self.map['remote_after_sha256'] and self.remote and all(hexsha(h) for h in self.remote.values()),'current_inventory_begin_end')
        need(set(self.remote)==set(self.files)|set(self.map['hash_only_inputs']),'current_inventory_complete_union')
        need(all(self.remote.get(p)==h for p,h in {**self.hashes,**self.map['hash_only_inputs']}.items()),'copied_and_hash_only_exact_current')
        self.reads={};self.local={};self.side_inputs={};self.external_aliases={}
        for name in ('base_manifest','supplemental_map'):
            if name not in self.map:continue
            ref=self.map[name];side=safe(ref['path']);data=side.read_bytes();need(digest(data)==ref['sha256'],'existing_copy_manifest_SHA')
            self.side_inputs[str(side)]=ref['sha256'];doc=json.loads(data)
            for p,local in doc['files'].items():
                if p in self.files:
                    need(local==self.files[p] and doc['sha256'][p]==self.hashes[p],'existing_copy_alias_exact')
                    self.external_aliases[p]=local
        need('actual_inventory' in self.map,'native_actual_inventory_required')
        if 'actual_inventory' in self.map:
            ref=self.map['actual_inventory'];zero(ref['ssh_actual_exit']);side=safe(ref['path']);data=side.read_bytes()
            need(digest(data)==ref['sha256'],'actual_inventory_receipt_SHA');self.side_inputs[str(side)]=ref['sha256'];actual=json.loads(data)
            zero(actual['actual_exit'])
            need(actual['inputs_unchanged'] is True and actual['test_read'] is False and type(actual['model_calls']) is int and actual['model_calls']==0,'actual_inventory_scope')
            need(actual['remote_before_sha256']==actual['remote_after_sha256']==self.remote,'actual_inventory_exact_current_maps')

    def raw(self,path,expected=None):
        path=str(path);need(path in self.files,'required_copy:'+path)
        local=safe(self.files[path]);need(local.is_relative_to(self.map_path.parent) or self.external_aliases.get(path)==str(local),'copied_file_containment_or_exact_existing_alias')
        raw=local.read_bytes();h=digest(raw)
        need(h==self.hashes[path] and (expected is None or h==expected),'actual_copied_SHA:'+path)
        need(path not in self.reads or self.reads[path]==h,'read_drift')
        self.reads[path]=h;self.local[path]=str(local);return raw

    def json(self,path,expected=None):return json.loads(self.raw(path,expected))
    def ref(self,ref):return self.json(ref['path'],ref['sha256'])
    def pin(self,path,sha):need(hexsha(sha) and self.remote.get(str(path))==sha,'current_inventory_pin:'+str(path))
    def exit0(self,path):need(self.raw(path)==b'0\n','actual_EXIT0_file')
    def finish(self):
        need(digest(safe(self.map_path).read_bytes())==self.map_sha,'map_drift')
        for p,h in self.side_inputs.items():need(digest(safe(p).read_bytes())==h,'existing_manifest_or_inventory_drift')
        for p,h in self.reads.items():need(digest(safe(self.local[p]).read_bytes())==h,'consumed_file_drift')


def reviewed(doc,required,label):
    need(doc.get('status')=='PASS' and doc.get('independent') is True,label+'_independent_PASS')
    need(all(doc.get('reviewed_sha256',{}).get(p)==h for p,h in required.items()),label+'_complete_SHA')

def reviewed_preserved_alias(doc,path,expected,suffix):
    """既有独审使用local archive路径；只承认精确logical suffix与相同实际SHA。"""
    keys=[p for p in doc['reviewed_sha256'] if p==str(path) or p.endswith(suffix)]
    need(bool(keys) and all(doc['reviewed_sha256'][p]==expected for p in keys),'review_covers_actual_preserved_metadata:'+str(path))


def verify_training(bundle,root,binding,protocol,request):
    run=root/'runs/train_after_import_fix'
    need(binding['training_run']==str(run),'successful_training_run_only')
    auditref=binding['training_audit'];audit=bundle.ref(auditref);bundle.exit0(Path(auditref['path']).parent/'EXIT_CODE')
    implref=auditref['implementation_review'];impl=bundle.ref(implref)
    need(impl.get('scope')=='cap10_training_audit_implementation','training_audit_review_scope')
    reviewed(impl,audit['source_sha256'],'training_audit_implementation')
    need(audit['schema']=='cap10_training_audit_20261024' and audit['status']=='PASS' and audit['independent'] is True
         and audit['inputs_unchanged'] is True and audit['training_run']==str(run),'actual_CPU_audit_PASS')
    need(set(audit['checks'])==CHECKS and all(v is True for v in audit['checks'].values()),'all12_audit_checks')
    need(audit['input_sha256']==audit['input_end_sha256'],'audit_actual_begin_end')
    for p,h in {**audit['input_sha256'],**audit['source_sha256']}.items():bundle.pin(p,h)
    tqref=binding['training_queue'];tq=bundle.ref(tqref);tbref=binding['training_bindings'];tb=bundle.ref(tbref)
    need(set(binding['caches'])=={'193','211'},'exact_cache_seeds')
    for seed in ('193','211'):
        need(set(binding['caches'][seed])=={'train','original_val','tune'},'train_val_cache_cohorts_only')
        for cohort in ('train','original_val','tune'):
            need(set(binding['caches'][seed][cohort])=={'single','dual'},'exact_cache_pools')
            for pool in ('single','dual'):
                cache=binding['caches'][seed][cohort][pool];old=tb['caches'][seed][cohort][pool]
                need(all(cache[k]==old[k] for k in ('path','sha256','manifest')),'inherited_seed_cache_identity')
                producer=cache['producer']
                need(producer['path']==old['producer'] and producer['exit_path']==str(Path(old['producer']).parent/'EXIT_CODE'),'exact_native_cache_producer_refs')
                need(binding['input_sha256'].get(producer['path'])==producer['sha256'] and producer['exit_path'] in binding['input_sha256'],'producer_metadata_fully_pinned')
    need(tq['phase']=='train' and tq['run_name']=='train_after_import_fix' and tq['root']==str(root),'actual_successful_queue')
    need(audit['queue_sha256']==tqref['sha256'] and audit['bindings_sha256']==tbref['sha256']
         and audit['protocol_sha256']==bundle.hashes[str(root/'protocol.json')],'audit_bound_exact_contract')
    result=bundle.ref(request['training_results_review'])
    need(result.get('scope')=='actual_eight_training_results_and_reviewed_CPU_audit_provenance','actual_training_result_scope')
    reviewed(result,{auditref['path']:auditref['sha256'],tqref['path']:tqref['sha256'],tbref['path']:tbref['sha256'],
                     str(root/'protocol.json'):bundle.hashes[str(root/'protocol.json')]},'actual_training_results')
    need(result.get('models')==8 and result.get('total_history_rows')==512 and result.get('total_draws')==4194304
         and result.get('total_optimizer_steps')==131072 and result.get('test_data_opened') is False,'actual_training_review_facts')
    need(result['execution']['training_run']==str(run),'actual_training_review_run')
    done=bundle.json(run/'completion.json');outer=bundle.json(run/'launch_receipt.json');start=bundle.json(run/'driver_start.json')
    for x in (done['external_exit'],outer['internal_exit'],outer['external_exit']):zero(x)
    bundle.exit0(run/'EXIT_CODE');bundle.exit0(run/'EXTERNAL_EXIT_CODE')
    need(done['status']=='COMPLETE' and done['inputs_unchanged'] is True and len(done['jobs'])==8,'actual_eight_COMPLETE')
    need(done['queue_sha256']==outer['queue_sha256']==start['queue_sha256']==tqref['sha256'],'actual_training_queue_SHA')
    need(type(start['pid']) is int and start['pid']>0 and start['pid']==done['pid']==outer['child_pid']
         and type(outer['wrapper_pid']) is int and outer['wrapper_pid']>0 and start['parent_pid']==outer['wrapper_pid'],'actual_training_PID')
    for job in done['jobs']:zero(job['internal_exit']);zero(job['external_exit'])
    rows={(m['cap'],m['seed'],m['role']):m for m in audit['models']}
    need(len(rows)==len(audit['models'])==8 and set(rows)==MATRIX,'exact_audit_eight_roles')
    for (cap,seed,role),m in rows.items():
        ref=binding['roles'][str(cap)][str(seed)][role];folder=root/f'rankers/C{cap}/s{seed}/{role}'
        need(all(type(m[k]) is int and m[k]==v for k,v in [('epochs',64),('draws',524288),('optimizer_steps',16384)])
             and type(m['alpha']) in (int,float) and m['alpha']==1,'fixed_final64_alpha1_budget')
        need(ref['checkpoint']==m['checkpoint']==str(folder/'final.pt') and ref['checkpoint_sha256']==m['checkpoint_sha256'],'audit_final_checkpoint_ref')
        need(ref['best']=={'epoch':64,'alpha':1.,'selection':'fixed_final_no_selection'},'no_best_or_alpha_selection')
        parent=tb['roles'][str(seed)][role]
        for key,name in [('centers','centers.json'),('center_rows','center_rows.jsonl')]:
            need(ref[key]==str(folder/name) and ref[key+'_sha256']==m[key+'_sha256']==parent[key+'_sha256'],'inherited_36centers_and_rows')
        need(ref['center33']==parent['center33'] and ref['O_center3']==parent['O_center3'] and len(ref['center33'])==33 and len(ref['O_center3'])==3,'actual_fixed_center_values')
        for key in ('checkpoint','centers','center_rows','producer_summary'):
            bundle.pin(ref[key],ref[key+'_sha256']);need(binding['input_sha256'].get(ref[key])==ref[key+'_sha256'],'registered_model_metadata')
    return result,audit


def verify_queue(bundle,root,binding,protocol,path,phase):
    q=bundle.json(path);count=16 if phase=='validation' else 4
    name=phase+'_native_cost';binding_path=root/'native_cost_validation_bindings.json'
    need(q['schema']=='cap10_controlled_queue_20261024' and q['phase']==phase and q['run_name']==name and q['root']==str(root),'phase_queue_root')
    need(q['bindings']==str(binding_path) and q['input_sha256']==binding['input_sha256'],'queue_exact_registry')
    need(q['maximum_seconds']==10800 and type(q['max_parallel']) is int and 1<=q['max_parallel']<=8,'queue_resource_budget')
    need(q['automatic_retry'] is False and q['promotion'] is False and type(q['external_api_calls']) is int and q['external_api_calls']==0,'no_retry_promotion_API')
    need(q['review_paths']==[str(root/(name+'_admission_independent_review.json'))],'exact_phase_admission_path')
    expected_sources=dict(binding['source_sha256'])
    for filename in ('protocol.json','native_cost_validation_bindings.json','isolation_entry.py','supervise_controlled_queue.py','launch_controlled_stage.py'):
        expected_sources[str(root/filename)]=bundle.hashes[str(root/filename)]
    need(q['source_sha256']==expected_sources,'complete_exact_queue_sources')
    for p,h in q['source_sha256'].items():bundle.raw(p,h)
    for p,h in {**q['source_sha256'],**q['input_sha256']}.items():bundle.pin(p,h)
    combinations=set(itertools.product((5,10),(193,211),('single','dual'),('original_val','tune'))) if phase=='validation' else set(itertools.product((5,10),(193,211),('dual',),('train',)))
    need(len(q['jobs'])==count and len({j['id'] for j in q['jobs']})==count,'complete_unique_phase_jobs')
    seen=set()
    for j in q['jobs']:
        command=j['command'];need(len(command)%2==1 and command[0]==str(root/'native_cost_evaluation/evaluate_cap_ranker.py'),'canonical_evaluator')
        flags=command[1:];need(all(type(v) is str and v.startswith('--') for v in flags[::2]),'CLI_flag_pairs')
        args=dict(zip(flags[::2],flags[1::2]));need(len(args)==len(flags)//2,'no_duplicate_CLI')
        cap=int(args['--cap']);seed=int(args['--seed']);pool=args['--pool'];cohort=args['--cohort'];key=(cap,seed,pool,cohort)
        need(key in combinations and key not in seen,'exact_registered_phase_cell');seen.add(key)
        jid=f'{phase}_C{cap}_s{seed}_{pool}_{cohort}';out=str(root/name/jid);access=str(root/'access_logs'/name/(jid+'.json'))
        cache=binding['caches'][str(seed)][cohort][pool]
        expected={'--source':protocol['source_checkout'],'--protocol':str(root/'protocol.json'),'--bindings':str(binding_path),
          '--cache':cache['path'],'--cap':str(cap),'--seed':str(seed),'--pool':pool,'--cohort':cohort,'--device':'cuda:0','--threads':'4','--out':out}
        if phase=='mechanism':expected['--query-manifest']=str(root/'fixed_train_pairs.json')
        need(args==expected,'exact_registered_CLI')
        reads=[binding['roles'][str(cap)][str(seed)][r]['checkpoint'] for r in ('no_aug','aug')]+[cache['path']]
        need(j['id']==jid and j['job_out']==out and j['access_log']==access and j['cwd']==protocol['source_checkout'],'exact_output_scope')
        need(type(j['gpu']) is int and 0<=j['gpu']<8 and j['allowed_reads']==reads and len(set(reads))==3,'three_exact_tensor_reads')
        need(all(Path(p).suffix=='.pt' for p in reads),'three_tensor_extensions')
        argv=[q['python'],'-B',str(root/'isolation_entry.py'),'--bindings',q['bindings'],'--phase',phase,'--access-log',access,'--output-root',out]
        for p in reads:argv+=['--allow-read',p]
        need(j['argv']==argv+['--']+command,'exact_guarded_process_argv')
        need(cache['manifest']['cohort']==cohort and cache['manifest']['recall']==pool,'cache_condition_identity')
        for p in reads:need(p in q['input_sha256'],'three_tensors_fully_pinned')
    need(seen==combinations,'complete_registered_matrix')
    oldpath=root/(phase+'_queue.json');old=bundle.json(oldpath,OLD_QUEUES[phase])
    need(q['recovery_from']==dict(queue=dict(path=str(oldpath),sha256=OLD_QUEUES[phase]),
      failed_validation_run=str(root/'runs/validation'),failure_review=binding['failure_recovery_review'],automatic_retry=False),'exact_transparent_recovery')
    need(len(old['jobs'])==count,'original_registered_matrix')
    for oldjob,newjob in zip(old['jobs'],q['jobs']):
        mapping={str(root/n):str(root/'native_cost_evaluation'/n) for n in EVAL}
        mapping.update({str(root/'validation_bindings.json'):str(binding_path),oldjob['job_out']:newjob['job_out'],oldjob['access_log']:newjob['access_log']})
        expected_job=dict(oldjob,command=[mapping.get(x,x) for x in oldjob['command']],
          argv=[mapping.get(x,x) for x in oldjob['argv']],job_out=newjob['job_out'],access_log=newjob['access_log'])
        need(newjob==expected_job,'no_scientific_job_argument_or_role_change')
    changed={'run_name','bindings','input_sha256','source_sha256','review_paths','recovery_from','jobs'}
    need({k:v for k,v in q.items() if k not in changed}=={k:v for k,v in old.items() if k not in changed},'no_other_queue_changes')
    return q


def verify_recovery(bundle,root,binding,request):
    oldpath=root/'validation_bindings.json';old=bundle.json(oldpath,OLD_BINDING_SHA)
    need(binding['prior_validation_bindings']==dict(path=str(oldpath),sha256=OLD_BINDING_SHA),'exact_original_binding')
    need(binding['evaluation_source_root']==str(root/'native_cost_evaluation'),'isolated_native_source_root')
    allowed_changes={'source_sha256','input_sha256','evaluation_implementation','evaluation_JJ_tests',
      'failure_recovery_review','prior_validation_bindings','evaluation_source_root','created_utc','input_scope'}
    need({k:v for k,v in binding.items() if k not in allowed_changes}=={k:v for k,v in old.items() if k not in allowed_changes},'all_training_cache_role_metadata_unchanged')
    need(len(old['input_sha256'])==206 and all(binding['input_sha256'].get(p)==h for p,h in old['input_sha256'].items()),'preserve_all_original206_inputs')
    replacements={str(root/n):str(root/'native_cost_evaluation'/n) for n in EVAL}
    sources=dict(old['source_sha256'])
    new_inputs={}
    for p,new in replacements.items():
        need(p in sources and new in bundle.hashes,'registered_old_new_eval_source')
        sources.pop(p);sources[new]=bundle.hashes[new];new_inputs[new]=bundle.hashes[new]
    need(binding['source_sha256']==sources,'exact_four_source_path_replacement_only')
    refs=(binding['evaluation_implementation'],binding['evaluation_JJ_tests'],binding['failure_recovery_review'])
    need(binding['evaluation_implementation']==request['evaluation_implementation_review'] and binding['evaluation_JJ_tests']==request['evaluation_JJ_execution'],'exact_implementation_JJ_binding')
    for ref in refs:new_inputs[ref['path']]=ref['sha256']
    need(binding['input_sha256']=={**old['input_sha256'],**new_inputs},'only_declared_new_sources_and_reviews_added')
    failure=bundle.ref(binding['failure_recovery_review'])
    need(failure['status']=='PASS' and failure['independent'] is True and failure['scope']=='failed_validation_preservation_and_exposure_accounting_not_recovery_admission','actual_failure_review')
    need(failure['queue_sha256']==OLD_QUEUES['validation'] and failure['copied_artifacts']==74 and failure['actual_remote_input_pins']==206 and failure['prior_training_audit_inputs_preserved']==200,'failure_original_closure')
    need(failure['launched_C5_jobs']==8 and failure['pending_C10_jobs']==8 and failure['original_val_internal_failures']==4 and failure['tune_peer_cleanup']==4 and failure['completed_prediction_rows']==0 and failure['usable_validation_metric_summaries']==0,'transparent_failed_exposure_counts')
    need(failure['model_loading_and_validation_reading_occurred'] is True and failure['no_zero_validation_exposure_claim'] is True and failure['automatic_retry'] is False and failure['recovery_admission'] is False,'no_zero_exposure_or_retry_claim')
    oldrun=root/'runs/validation';done=bundle.json(oldrun/'completion.json');launch=bundle.json(oldrun/'launch_receipt.json')
    need(done['status']=='FAILED' and done['queue_sha256']==launch['queue_sha256']==OLD_QUEUES['validation'],'old_failure_stays_failed')
    need(done['pid']==launch['child_pid']==failure['driver_pid'] and launch['wrapper_pid']==failure['wrapper_pid'],'old_failed_PID_binding')
    for k in ('internal_exit','external_exit'):need(type(launch[k]) is int and launch[k]==1,'old_failure_strict1')
    for rel in ('validation_bindings.json','validation_queue.json','mechanism_queue.json','runs/validation/completion.json','runs/validation/launch_receipt.json'):
        path=root/rel
        reviewed_preserved_alias(failure,path,bundle.hashes[str(path)],'/attempt_history/validation_finite_cost/remote/'+rel)
    probe=bundle.ref(request['cost_diagnostic_result_review'])
    need(probe['status']=='PASS' and probe['independent'] is True and probe['scope']=='actual_single_validation_row_native_cost_contract_not_full_validation_or_recovery_admission','actual_probe_review')
    need(probe['generalization_scope']=='C5_seed193_single_original_val_row0_only' and probe['no_NaN_or_shape_failure_in_this_row'] is True and probe['validation_recovery_admission'] is False,'diagnosis_limited_to_dtype_contract')
    for k in ('actual_inner_exit','actual_external_exit','actual_ssh_exit'):zero(probe[k])
    for rel in ('diagnostics/validation_cost_contract_row0/summary.json','actual_validation_cost_diagnostic_execution_receipt.json','access_logs/diagnostics/validation_cost_contract_row0.json'):
        path=root/rel
        reviewed_preserved_alias(probe,path,bundle.hashes[str(path)],'/diagnostic_cost_evidence/'+rel)
    diagnostic=bundle.json(root/'diagnostics/validation_cost_contract_row0/summary.json')
    need((diagnostic['cap'],diagnostic['seed'],diagnostic['pool'],diagnostic['cohort'],diagnostic['row_index'])==(5,193,'single','original_val',0),'actual_probe_fixed_row')
    d=diagnostic['diagnostic']
    need(d['legacy_assertions']==dict(shape512=True,dtype_float32=False,all_finite=True) and
      d['native_cost']['shape']==[512] and d['native_cost']['dtype']=='torch.float64' and d['native_cost']['finite'] is True and d['native_cost']['nonfinite_count']==0,'actual_probe_only_dtype_failure')
    execution=bundle.json(root/'actual_validation_cost_diagnostic_execution_receipt.json')
    for k in ('actual_inner_exit','actual_external_exit','actual_ssh_exit'):zero(execution[k])
    need(execution['summary_sha256']==bundle.hashes[str(root/'diagnostics/validation_cost_contract_row0/summary.json')],'actual_probe_execution_summary_SHA')
    return failure,probe

def verify(bundle,request):
    root=Path(request['root']);binding=bundle.json(root/'native_cost_validation_bindings.json');protocol=bundle.json(root/'protocol.json')
    need(request['schema']=='cap10_native_cost_evaluation_admission_request_20261025','request_schema')
    bp=str(root/'native_cost_validation_bindings.json')
    need(request['bindings']==dict(path=bp,sha256=bundle.hashes[bp]) and request['evaluation_source_root']==str(root/'native_cost_evaluation'),'explicit_request_new_binding_source_root')
    for phase in ('validation','mechanism'):
        qp=str(root/(phase+'_native_cost_queue.json'))
        need(request['queues'][phase]==dict(path=qp,sha256=bundle.hashes[qp]) and
          request['runs'][phase]==str(root/'runs'/(phase+'_native_cost')) and
          request['review_paths'][phase]==str(root/(phase+'_native_cost_admission_independent_review.json')),'explicit_request_new_queue_run_review')
    need(protocol['remote_root']==str(root) and protocol['caps']==[5,10] and protocol['seeds']==[193,211]
         and protocol['test_allowed'] is False and protocol['promotion'] is False,'registered_protocol')
    need(protocol['final_epoch']==64 and protocol['alpha']==1. and type(protocol['alpha']) is not bool and protocol['epoch_offset']==0
         and protocol['pool_size']==512 and protocol['candidate_library_size']==114982,'fixed_native_contract')
    failure,probe=verify_recovery(bundle,root,binding,request)
    training,audit=verify_training(bundle,root,binding,protocol,request)
    implementation=bundle.ref(request['evaluation_implementation_review'])
    need(implementation.get('scope')=='cap10_validation_and_mechanism_implementation','evaluation_review_scope')
    actual_sources={str(root/'native_cost_evaluation'/n):bundle.hashes[str(root/'native_cost_evaluation'/n)] for n in EVAL}
    reviewed(implementation,actual_sources,'evaluation_implementation')
    jj=bundle.ref(request['evaluation_JJ_execution'])
    zero(jj['actual_exit'])
    logref=request['evaluation_JJ_log'];log=bundle.raw(logref['path'],logref['sha256']).decode()
    counts=re.findall(r'^Ran ([0-9]+) tests? in [^\n]+$',log,re.MULTILINE)
    count=request['expected_JJ_test_count'];need(type(count) is int and count>=67,'expected_full_suite_test_count')
    need(counts==[str(count)] and re.search(r'^OK$',log,re.MULTILINE) is not None and not re.search(r'^FAILED\b',log,re.MULTILINE),'JJ_full_suite_actual_execution')
    need(all(jj['source_sha256'].get(p)==h for p,h in actual_sources.items()) and jj['source_end_sha256']==jj['source_sha256'],'JJ_same_actual_native_sources')
    for p,h in binding['source_sha256'].items():
        bundle.raw(p,h)
        need(training['reviewed_sha256'].get(p,implementation['reviewed_sha256'].get(p))==h,'every_source_previously_independently_reviewed')
    for p,h in binding['input_sha256'].items():bundle.pin(p,h)
    manifest=bundle.ref(binding['fixed_train_pairs'])
    need(protocol['mechanism']['pairs']==400 and protocol['mechanism']['original_obstructed_pairs']==87 and protocol['mechanism']['saturation_threshold']==.95,'fixed400_87_point95_contract')
    need(binding['fixed_train_pairs']['path']==str(root/'fixed_train_pairs.json') and binding['fixed_train_pairs']['sha256']==protocol['mechanism']['fixed_train_pairs']['sha256'],'original_fixed_manifest_identity')
    need(manifest['schema']=='cap10_fixed_train_pairs_20261024' and manifest['count']==400 and manifest['obstructed_count']==87 and manifest['selection_input'] is False,'actual_fixed_manifest_contract')
    need(binding['input_sha256'].get(binding['fixed_train_pairs']['path'])==binding['fixed_train_pairs']['sha256'],'fixed400_input_pin')
    outputs={}
    for phase in ('validation','mechanism'):
        path=str(root/(phase+'_native_cost_queue.json'));q=verify_queue(bundle,root,binding,protocol,path,phase)
        required={**q['source_sha256'],path:bundle.hashes[path]}
        outputs[phase]=dict(schema='cap10_evaluation_admission_independent_review_20261025',status='PASS',independent=True,
          reviewer='evidence_audit independent of native bindings/queue and evaluation source authors',scope='exact_'+phase+'_native_cost_recovery_after_actual_eightfit_audit',admission_phase=phase,
          reviewed_sha256=required,jobs=len(q['jobs']),training_run=binding['training_run'],test_reads=0,new_training_jobs=0,
          verified_current_input_sha256=q['input_sha256'],training_results_review=request['training_results_review'],
          evaluation_implementation_review=request['evaluation_implementation_review'],evaluation_JJ_execution=request['evaluation_JJ_execution'],
          evaluation_JJ_log=request['evaluation_JJ_log'],JJ_test_count_from_actual_log=count,
          failed_attempt_review=binding['failure_recovery_review'],cost_diagnostic_result_review=request['cost_diagnostic_result_review'],
          original_input_pins_preserved=206,prior_training_audit_input_pins_preserved=200,original_sources_unchanged=True,
          scientific_training_configuration_changed=False,automatic_retry=False,
          input_verification='Actual copied metadata/source bytes plus existing remote before/after inventory SHA; no tensor deserialization or second tensor hash.',
          selection_performed=False,promotion=False,independent_acceptance_status='unavailable')
    return outputs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    own=safe(__file__);own_sha=digest(own.read_bytes());requestpath=safe(args.request);raw=requestpath.read_bytes();request_sha=digest(raw)
    try:
        request=json.loads(raw);bundle=Bundle(request['artifact_map']);outputs=verify(bundle,request);bundle.finish()
        need(digest(safe(own).read_bytes())==own_sha and digest(safe(requestpath).read_bytes())==request_sha,'checker_or_request_drift')
        for phase,doc in outputs.items():
            doc.update(utc=datetime.now(timezone.utc).isoformat(),source_sha256={str(own):own_sha},source_end_sha256={str(own):own_sha},
              artifact_map_sha256=bundle.map_sha,input_sha256={**bundle.reads,str(requestpath):request_sha},inputs_unchanged=True)
            (out/(phase+'_native_cost_admission_independent_review.json')).write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n')
        (out/'EXIT_CODE').write_text('0\n');print(json.dumps({'status':'PASS','validation_jobs':16,'mechanism_jobs':4}))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps({'error':repr(exc)},indent=2)+'\n');(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

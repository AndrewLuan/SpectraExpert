# 中文说明：原6a9结果reader原字节保全。本新入口仅把顶层reranking_seconds排除于跨run科学语义全等；原raw计时与成本证据不修改。
"""功能概述：独立核验本轮已完成validation/mechanism复制结果，不调用作者评估模块。
先核collection原字节/remote始末SHA与严格driver双退出，再独立重算16cell自然512排名、
648表/两seed信号/全部预选guard和4cell固定400/87机制。共享候选逐cost/trace核不变。
只接受新轮train/validation产物与已完成训练的声明SHA；禁止test/旧sealed路径和tensor解包。
缺证据即失败，不启动评分、不恢复失败任务、不修改任何源、权重、协议或历史收据。
"""
import argparse,csv,hashlib,itertools,json,math,re
from pathlib import Path
from datetime import datetime,timezone
import cap_validation_result_math_independent as calc

def need(ok,label):
    if not ok:raise ValueError(label)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def zero(v):need(type(v) is int and v==0,'strict_int_EXIT0')
def stamp(t):
    v=datetime.fromisoformat(t.replace('Z','+00:00'));need(v.utcoffset().total_seconds()==0,'UTC');return v
def safe(path):
    p=Path(path).absolute();need('..' not in p.parts,'parent_traversal')
    r=p.resolve(strict=True)
    for q in (p,r):
        text=str(q).lower()
        need('/test/' not in text and '/sealed' not in text,'test_or_sealed_forbidden')
        need(q.suffix.lower() not in ('.pt','.pth','.ckpt','.npz','.npy'),'tensor_open_forbidden')
        need(q.name not in ('cohorts.pt','sealed_test_report.md'),'heldout_or_old_report')
    need(not p.is_symlink(),'symlink_input');return r

class Evidence:
    def __init__(self,manifests):
        self.paths={};self.hashes={};self.remote={};self.localpins={};self.manifests=[]
        for path in manifests:
            p=safe(path);raw=p.read_bytes();self.localpins[str(p)]=digest(raw);m=json.loads(raw)
            if 'source_manifest' in m:
                ref=m['source_manifest'];original=safe(ref['path']);original_raw=original.read_bytes()
                need(digest(original_raw)==ref['sha256'],'metadata_view_original_manifest_SHA')
                source=json.loads(original_raw);self.localpins[str(original)]=ref['sha256']
                if source.get('schema')=='cap10_validation_resource_failure_preservation_20261025':
                    need(source['source_inputs_unchanged'] is True and source['artifacts']==source['local_sha256'],'original_failed_preservation_exact_SHA')
                    artifacts={str(Path(source['remote_root'])/k):h for k,h in source['artifacts'].items()}
                    union=dict(source['input_sha256'])
                    for key,h in artifacts.items():
                        need(key not in union or union[key]==h,'original_failed_input_artifact_conflict');union[key]=h
                    source={**source,'files':{str(Path(source['remote_root'])/k):str((Path(source['local_payload'])/k).resolve()) for k in source['artifacts']},
                      'sha256':artifacts,'remote_before_sha256':union,'remote_after_sha256':union}
                need(m['remote_before_sha256']==source['remote_before_sha256'] and
                  m['remote_after_sha256']==source['remote_after_sha256'],'metadata_view_preserves_full_actual_union')
                need(all(source['files'].get(k)==v and source['sha256'][k]==m['sha256'][k] for k,v in m['files'].items()),'metadata_view_copy_subset_exact')
                for key in ('remote_inventory_before_exit','remote_inventory_after_exit','rsync_exit'):
                    need(type(source[key]) is int and source[key]==0 and m[key]==source[key],'metadata_view_actual_exits_preserved')
            for key in ('remote_inventory_before_exit','remote_inventory_after_exit','rsync_exit'):zero(m[key])
            need(m['remote_before_sha256']==m['remote_after_sha256'] and m['remote_before_sha256'],'remote_begin_end')
            need(set(m['files'])==set(m['sha256']),'complete_copy_map')
            need(set(m['remote_before_sha256'])==set(m['files'])|set(m['hash_only_inputs']),'complete_collector_union')
            for remote,h in m['remote_before_sha256'].items():
                need(remote not in self.remote or self.remote[remote]==h,'remote_union_conflict')
                self.remote[remote]=h
            for remote,local in m['files'].items():
                q=safe(local);need(q.is_relative_to(p.parent),'copied_containment')
                need(remote not in self.hashes or self.hashes[remote]==m['sha256'][remote],'copy_union_conflict')
                need(m['remote_before_sha256'][remote]==m['sha256'][remote],'copy_remote_SHA')
                self.paths[remote]=str(q);self.hashes[remote]=m['sha256'][remote]
            for remote,h in m['hash_only_inputs'].items():need(self.remote[remote]==h,'hash_only_actual_SHA')
            self.manifests.append((p,m))
        self.finish()
    def raw(self,path):
        key=str(path);need(key in self.paths,'missing_copied_metadata:'+key)
        p=safe(self.paths[key]);raw=p.read_bytes();need(digest(raw)==self.hashes[key],'consumed_bytes_SHA:'+key)
        return raw
    def json(self,path):return json.loads(self.raw(path))
    def pin(self,path,h):need(self.remote.get(str(path))==h,'declared_input_not_actual_hash:'+str(path))
    def exit0(self,path):need(self.raw(path)==b'0\n','actual_file_EXIT0')
    def finish(self):
        for p,h in self.localpins.items():need(digest(safe(p).read_bytes())==h,'manifest_changed')
        for remote,p in self.paths.items():need(digest(safe(p).read_bytes())==self.hashes[remote],'copied_bytes_changed:'+remote)

def compare(actual,expected,label):
    if isinstance(expected,dict):
        need(isinstance(actual,dict),'dict:'+label)
        for k,v in expected.items():need(k in actual,'missing:'+label+'.'+k);compare(actual[k],v,label+'.'+k)
    elif isinstance(expected,list):
        need(type(actual) is list and len(actual)==len(expected),'list:'+label)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,label+'.'+str(i))
    elif type(expected) is float:
        need(type(actual) in (int,float) and math.isfinite(actual) and abs(actual-expected)<=1e-12,'float:'+label)
    else:need(type(actual) is type(expected) and actual==expected,'exact:'+label)

def semantic_prediction(obj,top_level=True):
    if isinstance(obj,dict):
        return {k:semantic_prediction(v,False) for k,v in obj.items()
          if k not in ('pid','ppid','parent_pid','started_utc','finished_utc','elapsed_seconds','timing')
          and not (top_level and k=='reranking_seconds')}
    if isinstance(obj,list):return [semantic_prediction(v,False) for v in obj]
    return obj

def actual_stage(e,root,queue_path,run_path,phase,count):
    q=e.json(queue_path);run=Path(run_path)
    need(q['phase']==phase and run==root/'runs'/q['run_name'] and len(q['jobs'])==count,'phase_run_matrix')
    for p,h in {**q['input_sha256'],**q['source_sha256']}.items():e.pin(p,h)
    start=e.json(run/'driver_start.json');done=e.json(run/'completion.json');launch=e.json(run/'launch_receipt.json');status=e.json(run/'status.json')
    claim=e.json(root/(q['run_name']+'.operator_claim.json'));wrapper=e.json(root/(q['run_name']+'.launch_claim.json'))
    qsha=e.hashes[str(queue_path)]
    refs=q.get('operator_paths')
    if refs is not None:
        need(type(refs) is dict and set(refs)=={'supervisor','launcher'},'exact_new_operator_roles')
        for role,name in [('supervisor','supervise_resource_ready_queue.py'),('launcher','launch_resource_ready_stage.py')]:
            ref=refs[role];need(ref==dict(path=str(root/name),sha256=q['source_sha256'].get(str(root/name))),'exact_new_operator_ref')
            e.pin(ref['path'],ref['sha256'])
        need(q['resource_wait']==dict(maximum_seconds=120,poll_seconds=.2),'exact_resource_wait')
        need(all(o.get('operator_paths')==refs for o in (start,done,launch)),'new_operator_envelopes')
        need(start.get('resource_wait')==done.get('resource_wait')==q['resource_wait'],'new_resource_wait_envelopes')
    driver=refs['supervisor']['path'] if refs is not None else str(root/'supervise_controlled_queue.py')
    argv=[q['python'],'-B',driver,'--queue',str(queue_path),'--out',str(run),'--execute']
    need(start['argv']==argv[2:] and launch['argv']==argv,'driver_argv')
    need(all(o['queue_sha256']==qsha for o in (start,done,launch,claim,wrapper)),'driver_queue_SHA')
    need(start['pid']==done['pid']==launch['child_pid']==claim['pid'] and type(start['pid']) is int and start['pid']>0,'driver_PID')
    need(start['parent_pid']==launch['wrapper_pid']==wrapper['wrapper_pid'],'wrapper_PID')
    for o in (start,done,launch):need(o['phase']==phase and o['run_name']==q['run_name'],'actual_phase_run')
    zero(launch['internal_exit']);zero(launch['external_exit']);zero(done['external_exit'])
    e.exit0(run/'EXIT_CODE');e.exit0(run/'EXTERNAL_EXIT_CODE')
    need(done['status']=='COMPLETE' and done['inputs_unchanged'] is True and done['error'] is None,'COMPLETE_source_unchanged')
    need(status['phase']=='COMPLETE' and type(status['completed']) is int and status['completed']==count and status['running']==[] and status['pending']==0 and status['error'] is None,'all_jobs_completed')
    need(stamp(launch['started_utc'])<=stamp(start['started_utc'])<=stamp(done['finished_utc'])<=stamp(launch['finished_utc']),'same_host_wrapper_UTC')
    before=e.json(run/'input_fingerprints.before.json');after=e.json(run/'input_fingerprints.after.json')
    need(before==after==q['input_sha256']==start['input_sha256'] and start['source_sha256']==q['source_sha256'],'actual_complete_input_begin_end')
    small=e.json(run/'operator_fingerprints.before.json')
    need(small==e.json(run/'operator_fingerprints.after.json') and set(small)==set(q['source_sha256'])|{str(queue_path)}|set(q['review_paths']),'actual_small_closure_begin_end')
    for p,h in small.items():e.pin(p,h)
    jobs={r['id']:r for r in done['jobs']};need(len(jobs)==len(done['jobs'])==count and set(jobs)=={r['id'] for r in q['jobs']},'exact_job_matrix')
    if refs is not None:
        need(done.get('cancellation_signals')==[],'successful_operator_not_cancelled')
        events=done['resource_events'];gpu_jobs=[j for j in q['jobs'] if j['gpu'] is not None]
        need(e.json(run/'resource_readiness.json')['events']==events and len(events)==len(gpu_jobs),'one_idle_ready_event_per_GPU_dispatch')
        for event,job in zip(events,gpu_jobs):
            sample=event['snapshot']
            need(event['gpu']==job['gpu'] and type(event['probes']) is int and event['probes']>0
              and type(event['waited_seconds']) in (int,float) and 0<=event['waited_seconds']<=120,'registered_GPU_ready_event')
            need(type(sample['memory_mib']) is int and 0<=sample['memory_mib']<1024
              and type(sample['utilization_percent']) is int and 0<=sample['utilization_percent']<5
              and sample['compute_pids']==[],'strict_three_GPU_idle_conditions')
            need(stamp(start['started_utc'])<=stamp(event['started_utc'])<=stamp(event['ready_utc'])<=stamp(jobs[job['id']]['started_utc']),'resource_ready_before_owned_dispatch')
        samples=[json.loads(line) for line in e.raw(run/'resource_samples.jsonl').splitlines() if line]
        need(len(samples)>=len(events),'all_GPU_resource_observations_retained')
        completed={rec['pid']:rec for rec in jobs.values()}
        for observation in samples:
            sample=observation['snapshot'];retired=observation['completed_owned_pids']
            need(observation['owned_parent_pid']==start['pid'] and set(retired)<=completed.keys(),'resource_sample_owned_lineage')
            need(all(stamp(completed[pid]['finished_utc'])<=stamp(observation['utc']) for pid in retired),'reaped_before_retired_telemetry')
            need(all(type(pid) is int and pid in retired for pid in sample['compute_pids']),'no_foreign_GPU_process_dispatched')
            need(type(sample['memory_mib']) is int and sample['memory_mib']>=0
              and type(sample['utilization_percent']) is int and 0<=sample['utilization_percent']<=100
              and type(observation['resource_wait_seconds']) in (int,float) and 0<=observation['resource_wait_seconds']<=120,
              'bounded_raw_resource_observation')
    for job in q['jobs']:
        r=e.json(run/'jobs'/(job['id']+'.receipt.json'));started=e.json(run/'jobs'/(job['id']+'.started.json'));access=e.json(job['access_log'])
        need(r==jobs[job['id']] and all(r.get(k)==v for k,v in started.items()),'receipt_completion_start')
        zero(r['internal_exit']);zero(r['external_exit']);zero(access['internal_exit'])
        need(type(r['pid']) is int and r['pid']>0 and r['pid']==access['pid'] and r['parent_pid']==access['parent_pid']==start['pid'],'actual_child_PID')
        need(all(r[k]==job[k] for k in ('argv','cwd','gpu','job_out','access_log')),'exact_job_argv')
        need(r['queue_sha256']==qsha and r['requested_inputs_sha256']=={p:before[p] for p in job['allowed_reads']},'actual_job_input_SHA')
        need(access['phase']==phase and access['argv']==job['command'] and access['allowed_reads']==sorted(job['allowed_reads']) and access['error'] is None and access['rejected_paths']==[],'registered_scientific_IO_only')
        need(stamp(start['started_utc'])<=stamp(r['started_utc'])<=stamp(access['started_utc'])<=stamp(access['finished_utc'])<=stamp(r['finished_utc'])<=stamp(done['finished_utc']),'same_host_child_UTC')
        for p in job['allowed_reads']:need(access['recorded_reads'].get(p,0)>0,'registered_tensor_actually_opened')
        need(all(p in job['allowed_reads'] or Path(p).is_relative_to(Path(job['job_out'])) for p in access['recorded_reads']),'unregistered_input_read')
        e.exit0(Path(job['job_out'])/'EXIT_CODE')
    q['_independent_actual_receipts']=jobs
    return q,dict(phase=phase,run=str(run),wrapper_pid=launch['wrapper_pid'],driver_pid=start['pid'],jobs=count,strict_double_exit0=True,
      operator_paths=refs,resource_wait=q.get('resource_wait'))

def compare_prior_successes(e,root,args,new_jobs):
    """全部4原成功cell与新同cell逐语义字段精确相等，旧partial不参与任何选型。"""
    failure=e.json(args.resource_failure_review);strategy=e.json(args.recovery_strategy)
    need(failure['status']=='PASS' and failure['independent'] is True and failure['completed_validation_cells']==4,
      'independently_preserved_four_successes')
    need(strategy['recovery']=='all16_fresh_in_one_new_run_after_operator_and_actual_queue_independent_PASS'
      and strategy['prior_completed_cells_used_for_selection'] is False and strategy['prior_partial_predictions_used_for_selection'] is False
      and strategy['compare_all_four_prior_completed_cells_against_new_semantic_predictions'] is True,'locked_all16_semantic_comparison')
    prior={r['id']:r for r in failure['completed_cells']}
    wanted={f'validation_C5_s{s}_{p}_original_val' for s in (193,211) for p in ('single','dual')}
    need(set(prior)==wanted,'exact_four_prior_successful_cells')
    fresh={j['id']:j for j in new_jobs};counts={}
    for jid in sorted(wanted):
        path=root/'validation_native_cost'/jid/'predictions.jsonl';raw=e.raw(path)
        need(digest(raw)==prior[jid]['prediction_sha256'],'preserved_prior_prediction_SHA')
        old=[json.loads(line) for line in raw.splitlines() if line]
        new=[json.loads(line) for line in e.raw(Path(fresh[jid]['job_out'])/'predictions.jsonl').splitlines() if line]
        a={calc.identity(row):semantic_prediction(row) for row in old};b={calc.identity(row):semantic_prediction(row) for row in new}
        need(len(a)==len(old)==len(b)==len(new)==904 and a==b,'all_prior_completed_semantic_fields_exact:'+jid)
        counts[jid]=904
    return dict(prior_completed_cells=4,rows_compared=3616,per_cell_rows=counts,all_semantic_fields_exact=True,
      excluded_fields=['pid','ppid','parent_pid','started_utc','finished_utc','elapsed_seconds','timing'],
      excluded_top_level_measurement_fields=['reranking_seconds'],raw_timing_values_preserved=True,
      prior_results_used_for_selection=False,prior_partial_predictions_used=False)

def compare_prior_mechanism_lines(e,root,args,new_jobs):
    """根锁定窄授权：仅机制失败中12完整行作评分一致性对照，不用于选型或扩大分析。"""
    f=e.json(args.mechanism_failure_review);s=e.json(args.mechanism_recovery_strategy)
    need(f['status']=='PASS' and f['independent'] is True and f['completed_cells']==0 and f['complete_output_lines_lower_bound']==12,'preserved12_mechanism_lines')
    need(s['schema']=='cap10_fixed_mechanism_identity_recovery_strategy_20261025'
      and s['recovery']=='all4_fresh_in_one_new_run_after_mechanism_only_adapter_and_actual_queue_independent_PASS'
      and s['prior_partial_predictions_used_for_selection'] is False and s['validation_rerun'] is False
      and s['compare_all_prior_complete_mechanism_prediction_lines_to_new_semantic_rows'] is True,'locked12_line_semantic_only_scope')
    need(e.hashes[str(args.mechanism_failure_review)]==s['failure_review']['sha256']
      and s['failure_review']['path']==str(args.mechanism_failure_review),'mechanism_strategy_binds_actual_failure_review')
    records={r['id']:r for r in f['cells']};fresh={j['id']:j for j in new_jobs}
    need(set(records)==set(fresh) and len(records)==4,'same_four_mechanism_cells')
    counts={}
    for jid,record in records.items():
        path=root/'mechanism_resource_ready'/jid/'predictions.jsonl';raw=e.raw(path)
        need(digest(raw)==record['prediction_sha256'] and record['complete_output_lines']==3
          and record['ends_in_newline'] is True and raw.endswith(b'\n'),'original3_complete_mechanism_lines_SHA')
        old=[json.loads(line) for line in raw.splitlines() if line];need(len(old)==3,'exact_original3_complete_lines')
        new=[json.loads(line) for line in e.raw(Path(fresh[jid]['job_out'])/'predictions.jsonl').splitlines() if line]
        a={calc.identity(row):semantic_prediction(row) for row in old};b={calc.identity(row):semantic_prediction(row) for row in new}
        need(len(a)==3 and len(b)==len(new)==200 and all(k in b and b[k]==v for k,v in a.items()),'prior12_mechanism_semantics_exact:'+jid)
        counts[jid]=3
    return dict(cells=4,complete_lines_compared=12,per_cell_lines=counts,all_semantic_fields_exact=True,
      selection_input=False,scope='scorer_consistency_only_no_new_scientific_analysis')

def read_evaluation(e,job,bindings_path,protocol_path,actual,binding):
    out=Path(job['job_out']);s=e.json(out/'summary.json');p=e.json(out/'protocol.json');a=p['args']
    need(s['schema']=='cap_ranker_evaluation_20261024' and s['status']=='PASS','evaluation_schema_PASS')
    need(a['out']==str(out) and a['bindings']==str(bindings_path) and a['protocol']==str(protocol_path),'evaluation_exact_paths')
    for key in ('cap','seed','pool','cohort'):need(a[key]==s[key],'evaluation_role')
    need(s['inputs_unchanged'] is True and s['model_state_before']==s['model_state_after'],'frozen_model_state')
    for key in ('training','test_read','selection_performed','promotion'):need(s[key] is False,'eval_scope_'+key)
    need(s['pid']==p['pid'] and stamp(s['started_utc'])<=stamp(s['finished_utc']),'eval_PID_time')
    need(s['pid']==actual['pid'] and s['ppid']==actual['parent_pid'] and p['ppid']==actual['parent_pid'] and p['argv']==job['command'],'producer_actual_process_argv')
    need(stamp(actual['started_utc'])<=stamp(s['started_utc'])<=stamp(s['finished_utc'])<=stamp(actual['finished_utc']),'producer_actual_receipt_time')
    cache=binding['caches'][str(s['seed'])][s['cohort']][s['pool']]
    need(a['cache']==cache['path'] and s['cache_sha256']==cache['sha256'],'actual_registered_cache')
    expected_weights={binding['roles'][str(s['cap'])][str(s['seed'])][r]['checkpoint']:binding['roles'][str(s['cap'])][str(s['seed'])][r]['checkpoint_sha256'] for r in ('no_aug','aug')}
    need(s['checkpoints']==expected_weights and s['source_sha256']==binding['source_sha256'],'actual_registered_weights_and_sources')
    for path,h in s['input_sha256'].items():e.pin(path,h)
    need(all(s['input_sha256'].get(path)==h for path,h in p['input_sha256'].items()),'eval_initial_pins_preserved')
    for path,h in s['source_sha256'].items():need(s['input_sha256'].get(path)==h,'actual_eval_source_input')
    raw=e.raw(out/'predictions.jsonl');need(digest(raw)==s['predictions_sha256'],'prediction_bytes_declared')
    rows=[json.loads(line) for line in raw.splitlines() if line]
    need(len(rows)==s['records'] and len({calc.identity(r) for r in rows})==len(rows),'unique_prediction_count')
    identity_hashes=sorted(digest(json.dumps(calc.identity(r),ensure_ascii=False,separators=(',',':')).encode()) for r in rows)
    need(s['identity_sha256']==digest('\n'.join(identity_hashes).encode()),'actual_identity_SHA')
    if s['cohort']=='train':
        need(s.get('mechanism_evaluator')==binding['mechanism_evaluator'] and job['command'][0]==binding['mechanism_evaluator']['path'],'actual_fixed_mechanism_evaluator')
        scan=s.get('mechanism_identity_scan');manifest=s['cache_manifest']
        need(type(scan) is dict and all(type(scan.get(k)) is int for k in ('cache_rows_scanned','selected_rows','unselected_rows_skipped')),'typed_full_mechanism_identity_scan')
        need(scan==dict(schema='cap10_fixed_train_identity_scan_20261025',cache_rows_scanned=manifest['records'],
          registered_variant_counts=manifest['variants'],selected_rows=200,unselected_rows_skipped=manifest['records']-200,
          selected_identity_sha256=s['identity_sha256']) and len(rows)==200,'full_raw_train_scan_selected200')
    for r in rows:
        need(r['final_cost_dtype'] in ('torch.float32','torch.float64'),'declared_native_cost_dtype')
        need(r['coarse_dtype'] in ('torch.float32','torch.float64') and r['correction_dtypes']==['torch.float32','torch.float32'],'native_coarse_corrections_dtype')
        need(r['final_cost_dtype']==r['coarse_dtype'],'native_cost_promotion')
        calc.audit_row(r,r['final_cost_dtype'])
        if s['cohort']!='train':need(r['appended_training_positive'] is False,'natural_no_positive_injection')
    expected={v:calc.metrics([r for r in rows if r['query_identity']['variant']==v]) for v in calc.VARIANTS}
    compare(s['metrics'],expected,'evaluation_metrics')
    return (s['cap'],s['seed'],s['pool'],s['cohort']),rows,s

def verify(e,args):
    root=Path(args.root);protocol=e.json(args.protocol);bindings=e.json(args.bindings)
    mechanism_bindings_path=getattr(args,'mechanism_bindings',None) or args.bindings
    mechanism_bindings=e.json(mechanism_bindings_path)
    for key in ('roles','caches','training_run','training_queue','training_bindings','training_audit','fixed_train_pairs'):
        need(mechanism_bindings[key]==bindings[key],'mechanism_same_actual_training_cache_and_fixed_pairs:'+key)
    need(protocol['schema']=='cap10_controlled_protocol_20261024' and protocol['only_eligible_new_config']=='cap10_dual' and protocol['promotion'] is False,'registered_protocol')
    need(bindings['training_run'].endswith('/runs/train_after_import_fix'),'successful_training_run_only')
    valq,valstage=actual_stage(e,root,args.validation_queue,args.validation_run,'validation',16)
    meq,mestage=actual_stage(e,root,args.mechanism_queue,args.mechanism_run,'mechanism',4)
    need(valq['bindings']==str(args.bindings) and meq['bindings']==str(mechanism_bindings_path),'separate_explicit_val_mechanism_registries')
    mechanism_evaluator=getattr(args,'mechanism_evaluator',None)
    if mechanism_evaluator is not None:
        need(all(j['command'][0]==str(mechanism_evaluator) for j in meq['jobs']),'explicit_registered_mechanism_evaluator')
    matrix={};summary_index={}
    # Read one cap/seed/cohort single-dual pair at a time, then discard large arrays.
    invariance={}
    jobs=valq['jobs']
    def arg(job,k):return job['command'][job['command'].index('--'+k)+1]
    grouped={}
    for job in jobs:grouped.setdefault((int(arg(job,'cap')),int(arg(job,'seed')),arg(job,'cohort')),{})[arg(job,'pool')]=job
    need(set(grouped)==set(itertools.product((5,10),calc.SEEDS,('original_val','tune'))),'all_validation_cells')
    for (cap,seed,cohort),pair in grouped.items():
        need(set(pair)=={'single','dual'},'both_pools')
        ka,a,sa=read_evaluation(e,pair['single'],args.bindings,args.protocol,valq['_independent_actual_receipts'][pair['single']['id']],bindings)
        kb,b,sb=read_evaluation(e,pair['dual'],args.bindings,args.protocol,valq['_independent_actual_receipts'][pair['dual']['id']],bindings)
        need(ka==(cap,seed,'single',cohort) and kb==(cap,seed,'dual',cohort),'actual_validation_matrix_role')
        need(sa['checkpoints']==sb['checkpoints'],'same_weights_across_pools')
        invariance[f'C{cap}/s{seed}/{cohort}']=calc.common_invariance(a,b)
        for key,rr,s in ((ka,a,sa),(kb,b,sb)):
            for r in rr:
                r['_independent_candidate_order_SHA']=digest(json.dumps(r['natural_candidate_indices'],separators=(',',':')).encode())
                r['_independent_raw_coarse_SHA']=digest(json.dumps(r['natural_candidate_coarse_costs'],separators=(',',':')).encode())
                for field in ('natural_candidate_indices','natural_candidate_costs','natural_candidate_coarse_costs','natural_candidate_trace_sha256'):r.pop(field)
            matrix[key]=rr;summary_index[key]=s
    for seed,pool,cohort in itertools.product(calc.SEEDS,('single','dual'),('original_val','tune')):
        a={calc.identity(r):r for r in matrix[(5,seed,pool,cohort)]};b={calc.identity(r):r for r in matrix[(10,seed,pool,cohort)]}
        need(a.keys()==b.keys(),'crosscap_exact_query_identity')
        for key in a:
            for field in ('target_in_pool','target_candidate_index','cached_outside_pool_rank','gamma','mask_content_sha256','_independent_candidate_order_SHA','_independent_raw_coarse_SHA'):
                need(a[key][field]==b[key][field],'crosscap_exact_condition:'+field)
    signals={};parts={};expected_table={}
    metric_keys=tuple(calc.metrics([]))
    for cap,pool in itertools.product((5,10),('single','dual')):
        config=f'cap{cap}_{pool}';merged={seed:matrix[(cap,seed,pool,'original_val')]+matrix[(cap,seed,pool,'tune')] for seed in calc.SEEDS}
        signals[config],parts[config]=calc.validation_signals(merged)
        for cohort,variant,source in itertools.product(('original_val','tune','merged_validation'),calc.VARIANTS,('all',)+calc.SOURCES):
            per={seed:[r for r in (merged[seed] if cohort=='merged_validation' else matrix[(cap,seed,pool,cohort)]) if r['query_identity']['variant']==variant and (source=='all' or r['query_identity']['source']==source)] for seed in calc.SEEDS}
            for seed in calc.SEEDS:expected_table[(config,str(seed),cohort,variant,source)]=calc.metrics(per[seed])
            expected_table[(config,'two_seed_mean',cohort,variant,source)]=calc.equal_seed_metrics(per)
    result=e.json(Path(args.validation_summary)/'summary.json');e.exit0(Path(args.validation_summary)/'EXIT_CODE')
    compare(result['signals'],signals,'signals');compare(result['per_seed_signals'],parts,'per_seed_signals')
    choose=calc.selection(signals['cap10_dual'],signals['cap5_dual']);compare(result['selection'],choose,'selection')
    need(result['preselected']==choose['preselected'] and result['promotion'] is False,'single_eligible_or_fixed_fallback')
    table=list(csv.DictReader(e.raw(Path(args.validation_summary)/'metrics.tsv').decode().splitlines(),delimiter='\t'))
    need(len(table)==len(expected_table)==648,'complete_648_table')
    seen=set()
    for row in table:
        key=tuple(row[k] for k in ('config','seed','cohort','variant','source'));need(key in expected_table and key not in seen,'unique_648_keys');seen.add(key)
        for metric in metric_keys:
            text=row[metric];value=None if text=='' else (int(text) if metric=='queries' else float(text))
            compare(value,expected_table[key][metric],'TSV.'+str(key)+'.'+metric)
    need(result['table_sha256']==e.hashes[str(Path(args.validation_summary)/'metrics.tsv')],'table_SHA')
    origin=e.json(mechanism_bindings['fixed_train_pairs']['path']);per_cap={5:[],10:[]}
    for job in meq['jobs']:
        key,rows,s=read_evaluation(e,job,mechanism_bindings_path,args.protocol,meq['_independent_actual_receipts'][job['id']],mechanism_bindings)
        cap,seed,pool,cohort=key;need(pool=='dual' and cohort=='train' and len(rows)==200,'fixed_train_mechanism_role')
        per_cap[cap].extend(dict(r,seed=seed) for r in rows)
    pair_stats=calc.fixed_pair_summary(origin['records'],per_cap)
    mechanism=e.json(Path(args.mechanism_summary)/'summary.json');e.exit0(Path(args.mechanism_summary)/'EXIT_CODE')
    compare(mechanism['fixed_train_pair_analysis'],pair_stats,'fixed400_87')
    compare(mechanism['common_candidate_invariance'],invariance,'common_invariance')
    need(mechanism['selection_input'] is False and mechanism['training'] is False and mechanism['test_read'] is False and mechanism['promotion'] is False,'mechanism_nonselection_scope')
    for obj in (result,mechanism):
        need(obj['inputs_unchanged'] is True,'summary_inputs_unchanged')
        for p,h in obj['input_sha256'].items():e.pin(p,h)
    selection=e.json(args.validation_selection)
    need(selection.get('preselected')==choose['preselected'] and selection.get('promotion') is False,'rule_applied_selection_matches_independent_math')
    if 'selection' in selection:compare(selection['selection'],choose,'rule_applied_selection')
    if 'input_sha256' in selection:
        for p,h in selection['input_sha256'].items():e.pin(p,h)
    need(result.get('validation_run')==str(args.validation_run),'summary_actual_successful_validation_run')
    prior_comparison=compare_prior_successes(e,root,args,valq['jobs'])
    prior_mechanism_comparison=compare_prior_mechanism_lines(e,root,args,meq['jobs'])
    return dict(stages=[valstage,mestage],validation_run=str(args.validation_run),rows648_verified=True,fixed_train_pairs400=True,original87=True,
      native_dtype_rank_recomputed=True,all_guard_rules_verified=True,common_candidate_invariance=True,
      signals=signals,preselected=choose['preselected'],promotion=False,independent_acceptance='unavailable',
      prior12_mechanism_semantic_comparison=prior_mechanism_comparison,
      prior_validation_exposure=dict(first_dtype_failed_cells=4,second_attempt_completed_cells=4,second_attempt_partial_cells=4,
        second_attempt_unlaunched_cells=8,training_draws_added=0,test_reads=0),prior_four_semantic_comparison=prior_comparison)

def main():
    source_pins={str(Path(__file__).resolve()):digest(Path(__file__).read_bytes()),
      str(Path(calc.__file__).resolve()):digest(Path(calc.__file__).read_bytes())}
    p=argparse.ArgumentParser();p.add_argument('--manifest',action='append',required=True)
    p.add_argument('--mechanism-bindings');p.add_argument('--mechanism-evaluator')
    for key in ('root','validation-queue','validation-run','mechanism-queue','mechanism-run','bindings','protocol','validation-summary','mechanism-summary','validation-selection','resource-failure-review','recovery-strategy','mechanism-failure-review','mechanism-recovery-strategy','out'):p.add_argument('--'+key,required=True)
    args=p.parse_args();out=Path(args.out).absolute();need(not out.exists(),'unique_output');out.mkdir(parents=True)
    try:
        e=Evidence(args.manifest);result=verify(e,args);e.finish()
        need(all(digest(safe(path).read_bytes())==h for path,h in source_pins.items()),'checker_source_changed')
        result.update(schema='cap10_validation_mechanism_results_independent_review_20261025',status='PASS',independent=True,
          scope='actual_train_validation_results_and_fixed400_mechanism_not_test',
          reviewer='evidence_audit independent of trainer and evaluation/summary/mechanism authors',
          reviewed_sha256={**e.remote,**e.localpins,**source_pins},checker_sha256=source_pins[str(Path(__file__).resolve())],
          math_checker_sha256=source_pins[str(Path(calc.__file__).resolve())],utc=datetime.now(timezone.utc).isoformat())
        (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');(out/'EXIT_CODE').write_text('0\n')
        print(json.dumps({'status':'PASS','review':str(out/'review.json'),'sha256':digest((out/'review.json').read_bytes())}))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc)),ensure_ascii=False,indent=2)+'\n');(out/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()

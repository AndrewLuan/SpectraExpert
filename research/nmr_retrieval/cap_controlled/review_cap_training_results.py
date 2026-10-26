"""功能概述：只读复核已完成八份 C5/C10 训练及实际 CPU audit 的完整收集证据。

本脚本独立于 trainer/cap 模块，只有标准库，不导入 producer 或审计器，不反序列化张量。
Bundle 核复制原字节和远端始末 SHA；driver 核实际成功 run 的 PID/argv/双退出；
models 核八格、64 条 history、完整预算、负例资格记录及两 cap 相同的三类抽样流。
initial/final 的完整 tensor dtype/shape/逐值与 train 缓存 RNG 重放，接收已由非作者
审查的 CPU audit 的实际结果；本脚本不把这种证据接收表述为第二次张量重算。

main 仅在 root 已收集完成的 manifest 上运行，显式核首轮 import 失败的零预算审查，
不混入新成功 run。不读取旧 test、未完成 checkpoint，不执行 SSH/评分/训练/选型。
输出独立 status PASS 文档，promotion=false/独立验收 unavailable，保留全部来源 SHA。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA='cap10_actual_training_results_review_20261024'
AUDIT_SCHEMA='cap10_training_audit_20261024'
MODEL_SCHEMA='nmr_cap_continuation_20261024'
MASK='external_heldout_OR_ranker_internal_OR_same_query_group; target_loss_only'
CHECKS={'same_initial_tensors','frozen_observation','frozen_centers','frozen_scale',
        'matched_exposure','matched_pool_exposure','matched_negative_exposure',
        'strict_negative_qualification','finite','full_budget','real_driver_exit','source_input_unchanged'}
BUDGET=dict(epochs=64,samples=8192,batch_size=32,lr=.001,weight_decay=.001,anchor=.1,
            temperature=.5,clip_grad_norm=2.,draws_per_fit=524288,optimizer_steps=16384)
MATRIX={(cap,seed,role) for cap in (5,10) for seed in (193,211) for role in ('no_aug','aug')}


def require(ok,name):
    if not ok:raise ValueError(name)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def valid_sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def zero(value):require(type(value) is int and value==0,'strict_int_exit0')
def integer(value,wanted):return type(value) is int and value==wanted


def utc(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.utcoffset() is not None and result.utcoffset().total_seconds()==0,'actual_UTC')
    return result


def safe(path,allow_training_tensor=False):
    raw=Path(path).absolute();require('..' not in raw.parts,'forbidden_parent_traversal')
    resolved=raw.resolve(strict=True)
    for p in (raw,resolved):
        text=str(p).lower()
        require('/test/' not in text and '/sealed' not in text and
                (allow_training_tensor or p.suffix.lower() not in ('.pt','.pth','.ckpt','.npz','.npy')) and p.name.lower() not in
                ('predictions.jsonl','test.json','test.jsonl','sealed_test_report.md'),'forbidden_review_input')
    require(not raw.is_symlink(),'review_symlink_input')
    return resolved


class Bundle:
    def __init__(self,path,root,supplemental_map=None):
        self.map_path=safe(path);raw=self.map_path.read_bytes();self.map_sha=sha(raw)
        self.map=json.loads(raw);self.files=dict(self.map['files']);self.hashes=dict(self.map['sha256'])
        self.base_copied_files=len(self.files);self.supplement_copied_files=0
        self.supplement_path=None;self.supplement_sha=None
        self.hash_only=dict(self.map['hash_only_inputs'])
        require(isinstance(self.files,dict) and self.files and set(self.files)==set(self.hashes),'copied_map_complete')
        before=self.map['remote_before_sha256'];after=self.map['remote_after_sha256']
        require(before==after and before and all(valid_sha(v) for v in before.values()),'collector_remote_begin_end')
        self.remote=before
        require(all(before.get(p)==h for p,h in self.hashes.items()),'copied_matches_remote_SHA')
        require(all(before.get(p)==h for p,h in self.map['hash_only_inputs'].items()),'hash_only_matches_remote_SHA')
        require(set(before)==set(self.files)|set(self.map['hash_only_inputs']),'collector_complete_union')
        # collector 可以复制这16个已完成训练文件；只核原字节，不解包或读其 tensor 值。
        self.tensor_paths={str(Path(root)/f'rankers/C{cap}/s{seed}/{role}/{name}.pt')
                           for cap,seed,role in MATRIX for name in ('initial','final')}
        self.locals={p:str(safe(v,p in self.tensor_paths)) for p,v in self.files.items()}
        require(all(Path(p).is_relative_to(self.map_path.parent) for p in self.locals.values()),'copied_local_containment')
        require(all(Path(p).is_absolute() and '..' not in Path(p).parts for p in self.files),'canonical_remote_map')
        self.consumed={}
        if supplemental_map is not None:
            self.supplement_path=safe(supplemental_map);raw=self.supplement_path.read_bytes();self.supplement_sha=sha(raw)
            extra=json.loads(raw)
            require(extra['schema']=='cap10_training_review_metadata_supplement_20261024' and
                    extra['base_manifest_sha256']==self.map_sha,'supplement_exact_base_manifest')
            require(set(extra['files'])==set(extra['sha256']) and bool(extra['files']),'supplement_complete_files')
            for remote,local in extra['files'].items():
                require(remote not in self.files and self.hash_only.get(remote)==extra['sha256'][remote]==before.get(remote),
                        'supplement_existing_actual_remote_SHA_only')
                local=safe(local)
                require(local.suffix=='.json' and local.is_relative_to(self.supplement_path.parent),'supplement_JSON_containment')
                self.files[remote]=str(local);self.locals[remote]=str(local);self.hashes[remote]=extra['sha256'][remote]
                self.hash_only.pop(remote);self.supplement_copied_files+=1
        self.check_copy()

    def check_copy(self):
        for remote,local in self.locals.items():
            require(str(safe(local,remote in self.tensor_paths))==local,'copied_path_redirected')
        require(all(sha(Path(local).read_bytes())==self.hashes[p] for p,local in self.locals.items()),'all_copied_bytes_SHA')

    def raw(self,path):
        path=str(path);require(path in self.locals,'required_copied_artifact:'+path)
        local=safe(self.locals[path]);raw=local.read_bytes()
        require(sha(raw)==self.hashes[path],'consumed_bytes_SHA')
        self.consumed[path]=self.hashes[path]
        return raw

    def json(self,path):return json.loads(self.raw(path))
    def lines(self,path):return [json.loads(v) for v in self.raw(path).splitlines() if v]
    def exit0(self,path):require(self.raw(path)==b'0\n','actual_EXIT0_file')
    def pin(self,path,wanted):require(valid_sha(wanted) and self.remote.get(str(path))==wanted,'remote_pinned_artifact:'+str(path))

    def finish(self):
        self.check_copy()
        require(sha(safe(self.map_path).read_bytes())==self.map_sha,'artifact_map_changed')
        if self.supplement_path is not None:
            require(sha(safe(self.supplement_path).read_bytes())==self.supplement_sha,'supplement_map_changed')


def actual_driver(bundle,root,queue_path,q):
    run=root/'runs'/q['run_name']
    start=bundle.json(run/'driver_start.json');done=bundle.json(run/'completion.json')
    launch=bundle.json(run/'launch_receipt.json');status=bundle.json(run/'status.json')
    claim=bundle.json(root/(q['run_name']+'.operator_claim.json'))
    wrapper_claim=bundle.json(root/(q['run_name']+'.launch_claim.json'))
    qsha=bundle.hashes[str(queue_path)]
    expected=[q['python'],'-B',str(root/'supervise_controlled_queue.py'),'--queue',str(queue_path),'--out',str(run),'--execute']
    require(start['argv']==expected[2:] and launch['argv']==expected,'actual_driver_argv')
    require(all(x['queue_sha256']==qsha for x in (start,done,launch,claim,wrapper_claim)),'actual_driver_queue_SHA')
    require(type(start['pid']) is int and start['pid']>0 and
            start['pid']==done['pid']==launch['child_pid']==claim['pid'],'actual_driver_PID')
    require(type(start['parent_pid']) is int and start['parent_pid']>0 and
            start['parent_pid']==launch['wrapper_pid']==wrapper_claim['wrapper_pid'],'actual_wrapper_PID')
    require(all(x['phase']=='train' and x['run_name']==q['run_name'] for x in (start,done,launch)),'actual_run_name')
    zero(launch['internal_exit']);zero(launch['external_exit']);zero(done['external_exit'])
    bundle.exit0(run/'EXIT_CODE');bundle.exit0(run/'EXTERNAL_EXIT_CODE')
    require(done['status']=='COMPLETE' and done['inputs_unchanged'] is True and done['error'] is None,'driver_complete')
    require(status['phase']=='COMPLETE' and integer(status['completed'],8) and status['running']==[] and
            integer(status['pending'],0) and status['error'] is None,'all_eight_completed_status')
    require(utc(launch['started_utc'])<=utc(start['started_utc'])<=utc(done['finished_utc'])<=utc(launch['finished_utc']),
            'actual_driver_UTC_envelope')
    require((utc(done['finished_utc'])-utc(start['started_utc'])).total_seconds()<=q['maximum_seconds']+30,'driver_time_budget')
    before=bundle.json(run/'input_fingerprints.before.json');after=bundle.json(run/'input_fingerprints.after.json')
    require(before==after==q['input_sha256']==start['input_sha256'] and start['source_sha256']==q['source_sha256'],
            'actual_driver_76_begin_end')
    small_before=bundle.json(run/'operator_fingerprints.before.json');small_after=bundle.json(run/'operator_fingerprints.after.json')
    require(small_before==small_after and set(small_before)==set(q['source_sha256'])|{str(queue_path)}|set(q['review_paths']),
            'actual_operator_complete_closure')
    for path,h in {**before,**small_before}.items():bundle.pin(path,h)
    jobs={row['id']:row for row in done['jobs']}
    require(len(jobs)==len(done['jobs'])==8 and set(jobs)=={row['id'] for row in q['jobs']},'complete_eight_receipts')
    for job in q['jobs']:
        r=bundle.json(run/'jobs'/(job['id']+'.receipt.json'))
        started=bundle.json(run/'jobs'/(job['id']+'.started.json'))
        access=bundle.json(job['access_log'])
        require(r==jobs[job['id']] and all(r.get(k)==v for k,v in started.items()),'job_receipt_start_completion')
        zero(r['internal_exit']);zero(r['external_exit']);zero(access['internal_exit'])
        require(type(r['pid']) is int and r['pid']>0 and r['pid']==access['pid'] and
                type(r['parent_pid']) is int and r['parent_pid']==access['parent_pid']==start['pid'],'job_actual_PID_parent')
        require(all(r[k]==job[k] for k in ('argv','cwd','gpu','job_out','access_log')),'job_exact_argv_conditions')
        require(r['queue_sha256']==qsha and r['requested_inputs_sha256']=={p:before[p] for p in job['allowed_reads']},'job_actual_input_SHA')
        require(access['phase']=='train' and access['argv']==job['command'] and
                access['allowed_reads']==sorted(job['allowed_reads']) and access['rejected_paths']==[] and access['error'] is None,
                'job_actual_isolation')
        require(utc(start['started_utc'])<=utc(r['started_utc'])<=utc(access['started_utc'])<=utc(access['finished_utc'])
                <=utc(r['finished_utc'])<=utc(done['finished_utc']),'job_actual_UTC_envelope')
        for p in job['allowed_reads']:require(access['recorded_reads'].get(p,0)>0,'three_bound_tensors_actually_opened')
        for p in access['recorded_reads']:
            require(p in job['allowed_reads'] or Path(p).is_relative_to(Path(job['job_out'])),'unregistered_scientific_read')
        bundle.exit0(Path(job['job_out'])/'EXIT_CODE')
    return dict(training_run=str(run),wrapper_pid=launch['wrapper_pid'],driver_pid=start['pid'],
                started_utc=start['started_utc'],finished_utc=done['finished_utc'],jobs=8)


def model_metadata(bundle,directory,role_record,row,job,bindings,heldout):
    summary=bundle.json(directory/'summary.json');history=bundle.lines(directory/'history.jsonl')
    qualification=bundle.json(directory/'negative_qualification.json')
    initial=bundle.json(directory/'initial_tensor_sha256.json');producer=bundle.json(directory/'protocol.json')
    command=job['command'];require(len(command)%2==1,'fit_command_layout')
    cli={command[i][2:].replace('-','_'):command[i+1] for i in range(1,len(command),2)}
    require(len(cli)==(len(command)-1)//2,'fit_command_duplicate_flags')
    for key in ('cap','seed'):cli[key]=int(cli[key])
    require(producer['args']==cli and cli['out']==str(directory),'producer_exact_actual_CLI')
    require(summary['args']==dict(cli,augment=row['role']=='aug',training_pool='mixed',loss='listwise',
                **{k:v for k,v in BUDGET.items() if k not in cli}),'complete_saved_args')
    require(summary['schema']==MODEL_SCHEMA and type(summary['delta_cap']) is int and summary['delta_cap']==row['cap'] and
            type(summary['seed']) is int and summary['seed']==row['seed'] and summary['role']==row['role'],'actual_model_role_schema')
    require(summary['budget']==BUDGET and integer(summary['epoch'],64) and summary['alpha']==1 and type(summary['alpha']) is not bool and
            integer(summary['draws'],524288) and integer(summary['optimizer_steps'],16384),'actual_complete_model_budget')
    require(summary['negative_mask_policy']==MASK and summary['optimizer_reset'] is True and
            integer(summary['continuation_epoch_offset'],0) and summary['selection']=='fixed_final_epoch64_alpha1_no_selection',
            'actual_new_training_contract')
    require(all(summary[k] is True for k in ('centers_frozen','scale_frozen','observation_frozen','initial_tensors_inherited_exact')),
            'actual_producer_frozen_flags')
    for key in ('centers_sha256','center_rows_sha256','parent_checkpoint_sha256'):
        wanted=role_record[{'parent_checkpoint_sha256':'checkpoint_sha256'}.get(key,key)]
        require(row[key]==wanted,'model_parent_or_center_SHA')
    require(summary['parent_checkpoint']==role_record['checkpoint'] and
            summary['parent_checkpoint_sha256']==row['parent_checkpoint_sha256'],'saved_parent_identity')
    caches=bindings['caches'][str(row['seed'])]['train']
    require(summary['train_sha256']==caches['single']['sha256'] and summary['dual_train_sha256']==caches['dual']['sha256'],
            'actual_original_training_cache_pair')
    require(initial==row['initial_tensor_sha256']==summary['initial_tensor_sha256'] and
            '__scale__' in initial and 'fixed_x_center' in initial and 'fixed_o_center' in initial and
            all(valid_sha(h) for h in initial.values()),'complete_initial_tensor_metadata')
    frozen=canonical({k:v for k,v in initial.items() if not k.startswith('network.')})
    require(summary['frozen_tensor_sha256']==frozen,'actual_frozen_tensor_digest')
    bundle.pin(directory/'initial.pt',summary['initial_checkpoint_sha256'])
    bundle.pin(directory/'final.pt',row['checkpoint_sha256'])
    require(summary['final_checkpoint_sha256']==row['checkpoint_sha256'] and row['checkpoint']==str(directory/'final.pt'),
            'actual_final_checkpoint_identity')
    for path,h in summary['source_inputs_start_sha256'].items():bundle.pin(path,h)
    require(summary['source_inputs_start_sha256']==summary['source_inputs_end_sha256']==summary['input_sha256']==producer['input_sha256'],
            'actual_model_input_complete_begin_end')
    require(summary['epoch_records']==history and len(history)==64,'actual64_history')
    for epoch,item in enumerate(history,1):
        require(integer(item['epoch'],epoch) and integer(item['draws'],8192) and item['pool_draws']=={'0':4096,'1':4096} and
                all(type(v) is int for v in item['pool_draws'].values()) and integer(item['optimizer_steps'],256) and
                integer(item['cumulative_draws'],8192*epoch) and integer(item['cumulative_optimizer_steps'],256*epoch),
                'actual_each_epoch_full_budget_5050')
        require(type(item['mean_training_loss']) in (int,float) and math.isfinite(item['mean_training_loss']) and
                item['frozen_tensor_sha256']==frozen,'finite_history_and_frozen_state')
        for name in ('exposure_sha256','pool_exposure_sha256','negative_exposure_sha256'):
            require(valid_sha(item[name]) and valid_sha(item['cumulative_'+name]),'all_epoch_exposure_hashes')
    for name in ('exposure_sha256','pool_exposure_sha256','negative_exposure_sha256'):
        require(summary[name]==row[name]==history[-1]['cumulative_'+name],'actual_final_exposure_hash')
    records=qualification['records']
    require(qualification['policy']==MASK and qualification['sha256']==row['strict_negative_qualification_sha256']==
            summary['strict_negative_qualification_sha256']==canonical(records),'actual_negative_qualification_digest')
    require(integer(summary['training_rows'],len(records)//2) and len(records)==2*summary['training_rows'],
            'complete_two_pools_negative_records')
    by_key={}
    for record in records:
        key=tuple(record['query_key'])
        require(len(key)==6 and type(record['pool']) is int and record['pool'] in (0,1) and
                type(record['negatives']) is int and 1<=record['negatives']<=512 and
                valid_sha(record['ordered_eligible_candidate_sha256']) and valid_sha(record['excluded_mask_sha256']),
                'strict_negative_record_shape')
        require((key,record['pool']) not in by_key,'duplicate_negative_record');by_key[key,record['pool']]=record
        require(int(hashlib.sha256(('ranker_internal_317'+key[5]).encode()).hexdigest()[:12],16)%5!=0,
                'fit_queries_are_noninternal')
        require(key[5] not in heldout,'fit_queries_exclude_historical_heldout')
        if row['role']=='no_aug':require(key[1]=='observed','no_aug_only_observed')
    require(all((key,1-pool) in by_key for key,pool in by_key),'both_natural_pool_qualifications')
    require(integer(summary['training_groups'],len({key[5] for key,pool in by_key})),'actual_fit_group_count')
    require(qualification['qualification']==role_record['qualification'],'authority_candidate_group_qualification')
    for name,key in (('centers.json','centers_sha256'),('center_rows.jsonl','center_rows_sha256')):
        bundle.pin(directory/name,row[key]);bundle.pin(role_record['centers' if name=='centers.json' else 'center_rows'],row[key])
    center=bundle.json(directory/'centers.json')
    require(center['row_audit_sha256']==row['center_rows_sha256'] and
            center['center33']==role_record['center33'] and center['O_center3']==role_record['O_center3'] and
            len(center['center33'])==33 and len(center['O_center3'])==3 and
            all(type(x) in (int,float) and math.isfinite(x) for x in center['center33']+center['O_center3']),
            'actual36_center_JSON_inheritance')
    return history


def verify(bundle,root,queue_path,bindings_path,protocol_path,audit_path,implementation_path,failure_path):
    q=bundle.json(queue_path);bindings=bundle.json(bindings_path);protocol=bundle.json(protocol_path)
    audit=bundle.json(audit_path/'summary.json');bundle.exit0(audit_path/'EXIT_CODE')
    implementation=bundle.json(implementation_path)
    require(implementation['status']=='PASS' and implementation['independent'] is True and
            implementation['scope']=='cap10_training_audit_implementation','CPU_audit_implementation_independent_PASS')
    auditor=str(root/'audit_cap_training.py')
    require(implementation['reviewed_sha256'].get(auditor)==bundle.remote.get(auditor),'CPU_audit_exact_implemented_source')
    require(audit['schema']==AUDIT_SCHEMA and audit['status']=='PASS' and audit['independent'] is True and
            audit['inputs_unchanged'] is True and set(audit['checks'])==CHECKS and all(v is True for v in audit['checks'].values()),
            'CPU_audit_actual_full_checks_PASS')
    require(q['root']==str(root) and q['phase']=='train' and q['run_name']=='train_after_import_fix' and
            q['bindings']==str(bindings_path) and q['input_sha256']==bindings['input_sha256'],'successful_training_queue_binding')
    for path in q['review_paths']:
        admission=bundle.json(path)
        require(admission['status']=='PASS' and admission['independent'] is True and admission['admission_phase']=='train',
                'actual_training_admission_independent_PASS')
        for p,h in {**q['source_sha256'],str(queue_path):bundle.hashes[str(queue_path)]}.items():
            require(admission['reviewed_sha256'].get(p)==h,'actual_training_admission_complete_SHA_binding')
    require(audit['queue_sha256']==bundle.hashes[str(queue_path)] and audit['bindings_sha256']==bundle.hashes[str(bindings_path)] and
            audit['protocol_sha256']==bundle.hashes[str(protocol_path)] and audit['training_run']==str(root/'runs'/q['run_name']),
            'actual_audit_exact_successful_run')
    require(audit['source_sha256']==dict(q['source_sha256'],**{auditor:bundle.remote[auditor]}) and
            audit['input_sha256']==audit['input_end_sha256'],'CPU_audit_complete_source_input_closure')
    audit_execution=bundle.json(root/'actual_training_audit_execution_receipt.json')
    for key in ('actual_inner_exit','actual_external_exit','actual_ssh_exit'):zero(audit_execution[key])
    argv=audit_execution['remote_argv'];require(isinstance(argv,list) and argv.count(auditor)==1,'actual_CPU_audit_argv')
    pos=argv.index(auditor)
    require(argv[:pos] in ([q['python']],[q['python'],'-B']),'actual_CPU_audit_python')
    flags=argv[pos+1:]
    require(len(flags)%2==0 and all(flags[i].startswith('--') for i in range(0,len(flags),2)), 'actual_CPU_audit_flag_layout')
    cli={flags[i][2:].replace('-','_'):flags[i+1] for i in range(0,len(flags),2)}
    require(len(cli)==len(flags)//2 and cli==dict(root=str(root),run=str(root/'runs'/q['run_name']),
        queue=str(queue_path),bindings=str(bindings_path),protocol=str(protocol_path),source=bindings['source_checkout'],
        implementation_review=str(implementation_path),out=str(audit_path)),'actual_CPU_audit_exact_args')
    require(audit_execution['source_sha256'].get(auditor)==bundle.remote[auditor],'actual_CPU_audit_executed_source')
    for path,h in audit_execution['source_sha256'].items():bundle.pin(path,h)
    for path,h in {**audit['input_sha256'],**audit['source_sha256']}.items():bundle.pin(path,h)
    budget=protocol['budget']
    require(all(budget.get(k)==v and type(budget[k]) is not bool for k,v in BUDGET.items()) and
            all(integer(budget.get(k),v) for k,v in dict(fits=8,samples_per_epoch=8192,single_pool_draws_per_epoch=4096,
                dual_pool_draws_per_epoch=4096,total_draws=4194304,total_optimizer_steps=131072,new_forward_bank_extractions=0).items()) and
            budget['maximum_stage_seconds']==q['maximum_seconds'] and budget['loss']=='listwise' and budget['training_pool']=='mixed' and
            budget['checkpoint']=='final epoch64 only' and budget['alpha']==1 and type(budget['alpha']) is not bool and
            budget['early_stopping'] is False and budget['cap_grid_allowed'] is False and
            protocol['caps']==[5,10] and protocol['seeds']==[193,211] and
            protocol['final_epoch']==64 and protocol['alpha']==1 and protocol['epoch_offset']==0,'actual_preregistered_budget_matrix')
    require(audit['test_data_opened'] is False and integer(audit['new_model_calls'],0) and integer(audit['new_training_jobs'],0),
            'CPU_audit_no_scientific_execution')
    execution=actual_driver(bundle,root,queue_path,q)
    require(audit['execution']=={k:execution[k] for k in ('wrapper_pid','driver_pid','jobs','started_utc','finished_utc')}
            | {'queue_sha256':bundle.hashes[str(queue_path)]},'CPU_audit_driver_metadata_exact')
    models=audit['models'];by_key={(r['cap'],r['seed'],r['role']):r for r in models}
    require(len(by_key)==len(models)==8 and set(by_key)==MATRIX,'actual_eight_model_matrix')
    histories={}
    contract=bundle.json(bindings['cohort_contract']);heldout=set(contract['training_excluded_groups'])
    require(len(heldout)==bindings['heldout_groups'] and
            sha('\n'.join(sorted(heldout)).encode())==bindings['heldout_groups_sha256'],'authority_heldout_groups')
    for key,row in by_key.items():
        cap,seed,role=key
        require(all(integer(row[k],v) for k,v in (('cap',cap),('seed',seed),('epochs',64),('draws',524288),('optimizer_steps',16384)))
                and row['alpha']==1 and type(row['alpha']) is not bool,'audit_model_exact_budget')
        jobs=[job for job in q['jobs'] if job['id']==f'fit_C{cap}_s{seed}_{role}']
        require(len(jobs)==1,'actual_model_job_identity')
        directory=root/f'rankers/C{cap}/s{seed}/{role}'
        histories[key]=model_metadata(bundle,directory,bindings['roles'][str(seed)][role],row,jobs[0],bindings,heldout)
    for seed in (193,211):
        for role in ('no_aug','aug'):
            left,right=by_key[5,seed,role],by_key[10,seed,role]
            for name in ('initial_tensor_sha256','strict_negative_qualification_sha256','exposure_sha256',
                         'pool_exposure_sha256','negative_exposure_sha256','parent_checkpoint_sha256','centers_sha256','center_rows_sha256'):
                require(left[name]==right[name],'actual_C5_C10_same_'+name)
            for a,b in zip(histories[5,seed,role],histories[10,seed,role]):
                for name in ('exposure_sha256','pool_exposure_sha256','negative_exposure_sha256'):
                    require(a[name]==b[name] and a['cumulative_'+name]==b['cumulative_'+name],
                            'actual_all64_paired_'+name)
    failure=bundle.json(failure_path)
    require(failure['status']=='PASS' and failure['independent'] is True and failure['training_completed'] is False and
            integer(failure['draws'],0) and integer(failure['optimizer_steps'],0) and integer(failure['history_rows'],0) and
            integer(failure['input_pins_unchanged'],len(q['input_sha256'])) and
            failure['archive_manifest_sha256']==q['recovery']['prior_archive_manifest_sha256'] and
            q['recovery']['prior_run']==str(root/'runs/train') and integer(q['recovery']['prior_draws'],0) and
            integer(q['recovery']['prior_optimizer_steps'],0) and q['recovery']['scientific_configuration_changed'] is False and
            q['recovery']['automatic_retry'] is False,'first_import_failure_zero_budget_preserved')
    return dict(execution=execution,models=8,total_history_rows=512,total_draws=4194304,total_optimizer_steps=131072,
                negative_policy=MASK,prior_failed_attempt_zero_budget=True,
                tensor_verification='Accepted actual CPU audit from independently reviewed pinned implementation; no second tensor read.',
                cache_RNG_verification='Accepted actual independent CPU cache replay; all64 raw stream hashes separately compared across caps.')


def main(argv=None):
    parser=argparse.ArgumentParser()
    for name in ('artifact-map','root','queue','bindings','protocol','audit','implementation-review','failure-review',
                 'audit-execution-sha256','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--supplemental-map')
    args=parser.parse_args(argv);out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    bundle=None
    try:
        own=safe(__file__);own_sha=sha(own.read_bytes())
        bundle=Bundle(args.artifact_map,args.root,args.supplemental_map)
        bundle.pin(Path(args.root)/'actual_training_audit_execution_receipt.json',args.audit_execution_sha256)
        evidence=verify(bundle,Path(args.root),Path(args.queue),Path(args.bindings),Path(args.protocol),Path(args.audit),
                        Path(args.implementation_review),Path(args.failure_review))
        bundle.finish()
        require(sha(safe(own).read_bytes())==own_sha,'reviewer_source_changed')
        result=dict(schema=SCHEMA,status='PASS',independent=True,scope='actual_eight_training_results_and_reviewed_CPU_audit_provenance',
            utc=datetime.now(timezone.utc).isoformat(),checks={k:True for k in
                ('real_driver_double0','complete8_models','complete64_history','matched_cap_streams','strict_negative_metadata',
                 'frozen_center_scale_O_evidence','full_budget','actual_CPU_audit_and_source','collector_SHA_closure','prior_failure_preserved')},
            **evidence,artifact_map_sha256=bundle.map_sha,
            reviewed_sha256={**bundle.remote,str(bundle.map_path):bundle.map_sha,
                             **({str(bundle.supplement_path):bundle.supplement_sha} if bundle.supplement_path else {})},
            copied_sha256=bundle.hashes,hash_only_inputs=bundle.hash_only,
            base_copied_files=bundle.base_copied_files,supplement_copied_files=bundle.supplement_copied_files,
            supplemental_map_sha256=bundle.supplement_sha,
            source_sha256={str(own):own_sha},source_end_sha256={str(own):own_sha},
            test_data_opened=False,selection_performed=False,promotion=False,independent_acceptance_status='unavailable')
        (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        (out/'EXIT_CODE').write_text('0\n');print(json.dumps(dict(status='PASS',models=8,history_rows=512)))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),consumed=bundle.consumed if bundle else {}),indent=2)+'\n')
        (out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

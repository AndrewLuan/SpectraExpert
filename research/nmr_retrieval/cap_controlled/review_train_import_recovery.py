"""功能概述：独立核对首轮 import 失败的零预算证据和受控恢复队列。

只读本轮保全的原始日志、PID/退出收据、SHA 库存和源码；不访问 tensor/test。
保存原始 SSH 归档命令快照，验证实际输出不存在断言与 76 输入失败后重哈希。
恢复准入只允许 devnull 字符设备窄修、run/claim/access 证据隔离，不改变训练。
本工具不启动子任务、不连接服务器、不覆盖首轮审查或失败产物。
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

M = Path(__file__).resolve().parent
R = M / 'attempt_history/train_import_devnull/remote'
MR = '/home/user/IRfingerprints/SpectrExpert/outputs/nmr_cap10_controlled_20261024T163633Z'
seen = {}
checks = []


def blob(path):
    path = Path(path).resolve(strict=True)
    assert path.is_relative_to(M) and path.suffix not in ('.pt','.pth','.ckpt')
    raw = path.read_bytes()
    seen[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def load(path):
    return json.loads(blob(path))


def require(ok, label):
    assert ok, label
    checks.append(label)


def output(name, value):
    path = M / name
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(name, hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    archive = load(R / 'archive_sha256.json')
    require(len(archive)==55 and len([p for p in R.rglob('*') if p.is_file()])==56, '55_preserved_payloads_plus_manifest')
    for path,digest in archive.items():
        require(hashlib.sha256(blob(R / path)).hexdigest()==digest, 'preserved_bytes:' + path)
    old = load(R / 'original/train_queue.json')
    complete = load(R / 'runs/train/completion.json')
    start = load(R / 'runs/train/driver_start.json')
    launch = load(R / 'runs/train/launch_receipt.json')
    proof = load(R / 'zero_budget_failure_proof.json')
    before = load(R / 'runs/train/input_fingerprints.before.json')
    after = load(R / 'input_fingerprints.after_failure.json')
    qsha = hashlib.sha256(blob(R / 'original/train_queue.json')).hexdigest()
    require(complete['status']=='FAILED' and type(complete['external_exit']) is int and complete['external_exit']==1, 'first_driver_failed_not_success')
    require(launch['internal_exit']==launch['external_exit']==1 and type(launch['internal_exit']) is type(launch['external_exit']) is int, 'first_wrapper_true_double1')
    require(blob(R/'runs/train/EXIT_CODE').strip()==blob(R/'runs/train/EXTERNAL_EXIT_CODE').strip()==b'1', 'failed_exit_files')
    require(complete['pid']==start['pid']==launch['child_pid']==1686408 and start['parent_pid']==launch['wrapper_pid']==1686403, 'first_real_PID_chain')
    require(complete['queue_sha256']==start['queue_sha256']==launch['queue_sha256']==proof['queue_sha256']==qsha, 'first_queue_bound')
    require(before==after==old['input_sha256'] and len(before)==76, 'all76_rehashed_unchanged_after_failure')
    require(len(complete['jobs'])==8 and len({x['id'] for x in complete['jobs']})==8, 'eight_launched_not_eight_trained')
    double_fail, terminated = [], []
    for job in old['jobs']:
        record=load(R/'runs/train/jobs'/(job['id']+'.receipt.json'))
        begun=load(R/'runs/train/jobs'/(job['id']+'.started.json'))
        require(record in complete['jobs'] and all(record[k]==v for k,v in begun.items()), 'first_job_receipt_complete_start')
        require(record['argv']==job['argv'] and record['parent_pid']==start['pid'] and record['queue_sha256']==qsha, 'first_job_identity')
        log=blob(R/'runs/train/jobs'/(job['id']+'.log')).decode()
        if record['internal_exit']==1:
            require(type(record['internal_exit']) is int and record['external_exit']==1, 'failed_child_double1')
            access=load(R/'access_logs'/(job['id']+'.json'))
            require(access['pid']==record['pid'] and access['parent_pid']==start['pid'] and access['internal_exit']==1 and access['argv']==job['command'], 'failed_access_PID_argv')
            require(access['rejected_paths']==['/dev/null'] and 'dill' in log and 'import torch' in log and 'write outside task output: /dev/null' in log, 'import_failure_actual_stack')
            double_fail.append(job['id'])
        else:
            require(record['internal_exit'] is None and record['external_exit']==-15 and record['terminated_after_peer_failure'] is True and log=='', 'owned_peer_terminated_before_output')
            terminated.append(job['id'])
    require(len(double_fail)==2 and len(terminated)==6, 'two_import_failures_six_owned_peer_terminations')
    commands=[json.loads(line) for line in (M/'commands.jsonl').read_text().splitlines()]
    relevant=[r for r in commands if r.get('command',[None])[0]=='ssh' and 'zero_budget_failure_proof' in json.dumps(r)]
    require(len(relevant)==1 and type(relevant[0]['exit_code']) is int and relevant[0]['exit_code']==0, 'actual_archive_SSH0_record')
    remote_argv=shlex.split(relevant[0]['command'][-1])
    code=remote_argv[remote_argv.index('-c')+1]
    require("assert all(not Path(j['job_out']).exists() for j in q['jobs'])" in code and "after[path]==expected" in code, 'executed_remote_absence_and_hash_assertions')
    original_source=blob(R/'original/train_cap_ranker.py').decode()
    tree=ast.parse(original_source)
    main_function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    imports=[n.lineno for n in tree.body if isinstance(n,ast.Import) and any(x.name=='torch' for x in n.names)]
    require(imports and imports[0]<main_function.lineno and 'out.mkdir(parents=True, exist_ok=False)' in original_source and original_source.index('out.mkdir(parents=True, exist_ok=False)') < original_source.index('context = prepare_training(args, runtime, bindings, record, caches, pins, torch.device(args.device))'), 'trainer_import_before_mkdir_before_any_budget')
    require(proof['all_planned_outputs_absent'] is True and proof['optimizer_steps']==proof['training_draws']==proof['history_rows']==0 and proof['automatic_retry_performed'] is False, 'zero_budget_inference_from_no_output_and_source_order')
    q=load(M/'train_after_import_fix_queue.json')
    require({k for k in set(q)|set(old) if q.get(k)!=old.get(k)}=={'source_sha256','recovery','review_paths','run_name','jobs'}, 'only_recovery_evidence_and_operator_metadata_changed')
    require(q['phase']=='train' and q['run_name']=='train_after_import_fix' and q['input_sha256']==old['input_sha256'] and q['bindings']==old['bindings'], 'same76_training_inputs_new_immutable_run')
    require(q['recovery']['prior_run']==MR+'/runs/train' and q['recovery']['prior_archive_manifest_sha256']==hashlib.sha256(blob(R/'archive_sha256.json')).hexdigest() and q['recovery']['prior_draws']==q['recovery']['prior_optimizer_steps']==0 and q['recovery']['scientific_configuration_changed'] is False and q['recovery']['automatic_retry'] is False, 'recovery_explicitly_binds_zero_budget_failed_attempt')
    require(q['review_paths']==[MR+'/train_recovery_admission_independent_review.json'], 'new_specific_recovery_review')
    for prior,current in zip(old['jobs'],q['jobs']):
        require(prior['id']==current['id'] and prior['command']==current['command'] and prior['allowed_reads']==current['allowed_reads'] and prior['job_out']==current['job_out'] and prior['cwd']==current['cwd'] and prior['gpu']==current['gpu'], 'same_scientific_fit_command_role_output_budget')
        expected=dict(prior)
        expected['access_log']=MR+'/access_logs/train_after_import_fix/'+prior['id']+'.json'
        expected['argv']=list(prior['argv'])
        expected['argv'][expected['argv'].index('--access-log')+1]=expected['access_log']
        require(expected==current, 'only_outer_access_log_is_attempt_scoped')
    changed={k for k in q['source_sha256'] if q['source_sha256'][k]!=old['source_sha256'].get(k)}
    operators={'isolation_entry.py','supervise_controlled_queue.py','launch_controlled_stage.py'}
    require(changed=={MR+'/'+x for x in operators}, 'only_three_operator_sources_changed')
    require(blob(M/'cap_ranker.py')==blob(R/'original/cap_ranker.py') and blob(M/'train_cap_ranker.py')==blob(R/'original/train_cap_ranker.py') and blob(M/'protocol.json')==blob(R/'original/protocol.json') and blob(M/'execution_bindings.json')==blob(R/'original/execution_bindings.json'), 'model_trainer_scientific_protocol_and_binding_unchanged')
    repair=blob(M/'isolation_entry.py').decode()
    require('path == str(Path(os.devnull).resolve()) and stat.S_ISCHR(Path(path).stat().st_mode)' in repair and 'permitted_devnull_modes=devices' in repair, 'real_character_device_only_with_distinct_receipt')
    tests=load(M/'operator_recovery_synthetic_receipt.json')
    require(type(tests['actual_exit']) is int and tests['actual_exit']==0 and tests['tests']==26 and tests['inputs_unchanged'] is True and tests['source_before_sha256']==tests['source_after_sha256'], '26_independent_recovery_synthetics_actual0')
    for path,value in tests['source_after_sha256'].items():
        require(hashlib.sha256(blob(path)).hexdigest()==value, 'recovery_tested_final_source')
    prior_review=load(R/'original/train_admission_independent_review.json')
    require(prior_review['status']=='PASS' and prior_review['queue_sha256']==qsha, 'original_admission_preserved_not_rewritten')
    reviewed=dict(prior_review['reviewed_sha256'])
    for path,value in q['source_sha256'].items():
        if path.startswith(MR+'/'):
            require(hashlib.sha256(blob(M/Path(path).relative_to(MR))).hexdigest()==value,'current_repair_source_SHA')
        else:
            require(old['source_sha256'][path]==value,'prior_science_source_SHA_unchanged')
        reviewed[path]=value
    reviewed[MR+'/train_after_import_fix_queue.json']=hashlib.sha256(blob(M/'train_after_import_fix_queue.json')).hexdigest()
    for path,value in list(seen.items()):
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value,'review_input_end_SHA')
    now=datetime.now(timezone.utc).isoformat()
    failure=dict(schema='cap10_first_train_import_failure_independent_review_20261025',status='PASS',independent=True,utc=now,
        scope='first_failed_attempt_zero_scientific_budget_only',training_completed=False,jobs_launched=8,
        import_failed_children=2,owned_peers_terminated=6,draws=0,optimizer_steps=0,history_rows=0,
        basis='Actual SSH0 archive command asserted all8 planned outputs absent, preserved trainer has top-level torch import before main and output mkdir before any data preparation/AdamW/draw; both observed failures have exact torch/dill/devnull stacks, peers terminated by owner. No output-removal path in reviewed trainer.',
        queue_sha256=qsha,archive_manifest_sha256=hashlib.sha256(blob(R/'archive_sha256.json')).hexdigest(),
        archive_payloads=55,total_archive_files_including_manifest=56,input_pins_unchanged=76,
        actual_archive_command_snapshot=relevant[0],reviewed_sha256=dict(seen))
    output('zero_budget_failure_independent_review.json',failure)
    reviewed.update(seen)
    output('train_recovery_admission_independent_review.json',dict(
        schema='cap10_train_recovery_admission_independent_review_20261025',status='PASS',independent=True,utc=now,
        admission_phase='train',run_name='train_after_import_fix',scope='exact8_fit_zero_budget_environment_recovery_not_scientific_change',
        queue_sha256=reviewed[MR+'/train_after_import_fix_queue.json'],protocol_sha256=hashlib.sha256(blob(M/'protocol.json')).hexdigest(),
        bindings_sha256=hashlib.sha256(blob(M/'execution_bindings.json')).hexdigest(),source_sha256=q['source_sha256'],input_sha256=q['input_sha256'],
        reviewed_sha256=reviewed,checks=checks,check_count=len(checks),prior_attempt_preserved=True,prior_draws=0,prior_optimizer_steps=0,
        zero_budget_review_sha256=hashlib.sha256((M/'zero_budget_failure_independent_review.json').read_bytes()).hexdigest(),
        synthetic_actual_exit=0,independent_recovery_tests=26,scientific_configuration_changed=False,
        automatic_retry=False,recovery_training_started=False,test_data_opened=False,
        further_validation_or_test_admitted=False,
        dispatch_prerequisites='root source-only sync+exact SHA, JJ fullcold optimizer synthetic0, immutable recovery review deployment, fresh actual8GPU idle, unique manual dispatch; original failed run/access/claims remain.'))


if __name__ == '__main__':
    main()

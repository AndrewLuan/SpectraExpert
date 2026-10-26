"""功能概述：独立核验已复制的validation首失败证据，不重新评分或读取tensor。
核74份原字节、真实driver/8子进程退出、空prediction、保全的206输入与旧训练
审计200输入包含关系。明确区分读取/模型调用已发生与零完成指标，不声称零验证曝光。
该工具只读取指定新轮目录，写自己的审查JSON；不连接远端、不恢复科学任务。
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text())

def need(value, label):
    if not value:
        raise ValueError(label)

def run():
    manifest = ROOT/'attempt_history/validation_finite_cost/manifest.json'
    m = read(manifest)
    payload = Path(m['local_payload']).resolve()
    need(payload == (manifest.parent/'remote').resolve(), 'payload_containment')
    need(m['artifacts'] == m['local_sha256'] and len(m['artifacts']) == 74, '74_original_bytes')
    reviewed = {str(manifest): sha(manifest), str(Path(__file__).resolve()): sha(Path(__file__))}
    for rel, expected in m['artifacts'].items():
        p = (payload/rel).resolve()
        need(p.is_relative_to(payload), 'artifact_containment')
        need('/test/' not in str(p) and '/sealed' not in str(p), 'trainval_only')
        need(sha(p) == expected, 'copied_SHA:'+rel)
        reviewed[str(p)] = expected
    for k in ('remote_inventory_before_exit', 'remote_inventory_after_exit', 'rsync_exit'):
        need(type(m[k]) is int and m[k] == 0, k)
    need(m['source_inputs_unchanged'] is True and m['automatic_retry'] is False, 'preservation')
    q = read(payload/'validation_queue.json')
    need(q['input_sha256'] == m['input_sha256'] and len(m['input_sha256']) == 206, '206_actual_input_inventory')
    audit_path = ROOT/'collected/train/remote/audits/training/summary.json'
    audit = read(audit_path)
    need(len(audit['input_sha256']) == 200, '200_training_audit_inputs')
    need(all(m['input_sha256'].get(p) == h for p,h in audit['input_sha256'].items()), 'training200_unchanged')
    reviewed[str(audit_path)] = sha(audit_path)
    completion = read(payload/'runs/validation/completion.json')
    status = read(payload/'runs/validation/status.json')
    launch = read(payload/'runs/validation/launch_receipt.json')
    need(completion['status'] == 'FAILED' and completion['pid'] == 1690156, 'driver_failed')
    need(launch['wrapper_pid'] == 1690152 and launch['child_pid'] == completion['pid'], 'PID_binding')
    for k in ('internal_exit','external_exit'):
        need(type(launch[k]) is int and launch[k] == 1, 'global_'+k)
    need((payload/'runs/validation/EXIT_CODE').read_text().strip() == '1', 'inner1')
    need((payload/'runs/validation/EXTERNAL_EXIT_CODE').read_text().strip() == '1', 'outer1')
    qsha = sha(payload/'validation_queue.json')
    need(completion['queue_sha256'] == launch['queue_sha256'] == qsha, 'queue_binding')
    need(len(completion['jobs']) == 8 and status['pending'] == 8, '8_launched_8_pending')
    planned = {j['id']:j for j in q['jobs']}
    detail = []
    completed_predictions = 0
    for job in completion['jobs']:
        jid = job['id']; need('_C5_' in jid, 'C5_only_launched')
        receipt = read(payload/('runs/validation/jobs/'+jid+'.receipt.json'))
        need(receipt == job, 'completion_receipt_identical')
        need(job['parent_pid'] == completion['pid'] and job['queue_sha256'] == qsha, 'child_PID_queue')
        need(job['argv'] == planned[jid]['argv'], 'exact_planned_argv')
        need(datetime.fromisoformat(completion['started_utc']) <= datetime.fromisoformat(job['started_utc']) <= datetime.fromisoformat(job['finished_utc']) <= datetime.fromisoformat(completion['finished_utc']), 'UTC_envelope')
        if jid.endswith('original_val'):
            need(type(job['internal_exit']) is int and job['internal_exit'] == 1, 'original_internal1')
            need(job['external_exit'] in (1,-15), 'original_external_failure_or_cleanup')
            out = payload/'validation'/jid
            pred = out/'predictions.jsonl'
            need(pred.stat().st_size == 0, 'no_completed_predictions')
            need(not (out/'summary.json').exists(), 'no_metrics_summary')
            need(read(out/'failure.json')['error'] == "ValueError('finite_cost')", 'finite_cost_failure')
            access = read(payload/('access_logs/validation/'+jid+'.json'))
            need(access['pid'] == job['pid'] and access['parent_pid'] == completion['pid'], 'access_PID')
            need(access['rejected_paths'] == [] and access['internal_exit'] == 1, 'access_failure')
            need(set(access['recorded_reads']) == set(job['requested_inputs_sha256']), 'registered_three_tensors_read')
        else:
            need(jid.endswith('tune') and job['external_exit'] == -15 and job['internal_exit'] is None and job['terminated_after_peer_failure'] is True, 'tune_peer_cleanup')
            need(not (payload/'validation'/jid).exists(), 'no_tune_output_copied')
        detail.append({k:job.get(k) for k in ('id','pid','external_exit','internal_exit','terminated_after_peer_failure')})
    need(all('_C10_' in jid for jid in set(planned)-{j['id'] for j in completion['jobs']}), 'C10_pending_only')
    # This static producer is evaluate-only; don't equate empty outputs with zero model work.
    evaluator = (payload/'evaluate_cap_ranker.py').read_text()
    need('score_natural' in evaluator and 'optimizer.step(' not in evaluator and 'backward(' not in evaluator, 'evaluation_not_training')
    result = dict(schema='cap10_failed_validation_independent_review_20261025',status='PASS',independent=True,
      scope='failed_validation_preservation_and_exposure_accounting_not_recovery_admission',
      queue_sha256=qsha,wrapper_pid=launch['wrapper_pid'],driver_pid=completion['pid'],
      started_utc=completion['started_utc'],finished_utc=completion['finished_utc'],
      copied_artifacts=74,actual_remote_input_pins=206,prior_training_audit_inputs_preserved=200,
      launched_C5_jobs=8,pending_C10_jobs=8,original_val_internal_failures=4,tune_peer_cleanup=4,
      completed_prediction_rows=completed_predictions,usable_validation_metric_summaries=0,
      model_loading_and_validation_reading_occurred=True,model_scoring_may_have_occurred=True,
      training_optimizer_steps_in_this_evaluation_attempt=0,test_payload_read=False,
      no_zero_validation_exposure_claim=True,automatic_retry=False,recovery_admission=False,
      jobs=detail,reviewed_sha256=reviewed,utc=datetime.now().astimezone().isoformat())
    # Recheck every consumed copied artifact and the immutable inventory after review.
    for p,h in reviewed.items(): need(sha(Path(p)) == h, 'review_input_changed:'+p)
    out = ROOT/'validation_failed_attempt_independent_review.json'
    need(not out.exists(), 'unique_review_output')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','review':str(out),'sha256':sha(out),'copied_artifacts':74}))

if __name__ == '__main__':
    run()

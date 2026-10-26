"""功能概述：独立核本轮GPU readiness失败的执行、曝光和保全，不消费验证指标值。
对82原字节文件流式核SHA；只读取完成/退出/producer身份、records和状态字段，预测仅数
完整JSONL换行，不解析成本/排名/指标。区分4完整cell、4取消partial与8未启动C10。
旧gpu_not_idle未保存三项采样，明确不以其他时刻probe反推失败现场util/mem/apps。
不运行模型、SSH、训练、验证、test或选型，只写自己的独立审查与执行证据。
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

ROOT=Path(__file__).resolve().parent
def need(x,label):
    if not x:raise ValueError(label)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def zero(x):need(type(x) is int and x==0,'strict_int0')
def dt(x):return datetime.fromisoformat(x.replace('Z','+00:00'))
def line_info(p):
    count=0;end=b'';size=0
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            count+=block.count(b'\n');size+=len(block);end=block[-1:]
    return dict(complete_output_lines=count,bytes=size,ends_in_newline=not size or end==b'\n')
def main():
    manifest=ROOT/'attempt_history/validation_resource_not_ready/manifest.json';m=read(manifest)
    payload=Path(m['local_payload']).resolve();need(payload==manifest.parent/'remote','exact_payload')
    need(m['schema']=='cap10_validation_resource_failure_preservation_20261025' and len(m['artifacts'])==82 and m['artifacts']==m['local_sha256'],'82_original_artifacts')
    pins={str(manifest):sha(manifest),str(Path(__file__).resolve()):sha(__file__)}
    for rel,h in m['artifacts'].items():
        p=(payload/rel).resolve();need(p.is_relative_to(payload),'artifact_containment')
        need('/test/' not in str(p) and '/sealed' not in str(p),'trainval_only')
        need(sha(p)==h and p.stat().st_size==m['artifact_bytes'][rel],'preserved_byte_SHA_size:'+rel);pins[str(p)]=h
    for key in ('remote_inventory_before_exit','remote_inventory_after_exit','rsync_exit'):zero(m[key])
    need(m['source_inputs_unchanged'] is True and m['automatic_retry'] is False,'collector_scope')
    need(sum(m['artifact_bytes'].values())==682580702,'exact_preserved_byte_count')
    root=Path(m['remote_root']);q=read(payload/'validation_native_cost_queue.json')
    binding=read(payload/'native_cost_validation_bindings.json')
    need(q['phase']=='validation' and q['run_name']=='validation_native_cost' and len(q['jobs'])==16,'registered16_validation')
    need(q['input_sha256']==binding['input_sha256']==m['input_sha256'] and len(m['input_sha256'])==213,'actual213_current_inputs')
    old=read(ROOT/'attempt_history/validation_finite_cost/manifest.json')
    need(all(m['input_sha256'].get(p)==h for p,h in old['input_sha256'].items()) and len(old['input_sha256'])==206,'old206_preserved')
    audit=read(ROOT/'collected/train/remote/audits/training/summary.json')
    need(len(audit['input_sha256'])==200 and all(m['input_sha256'].get(p)==h for p,h in audit['input_sha256'].items()),'training200_preserved')
    run=payload/'runs/validation_native_cost';c=read(run/'completion.json');outer=read(run/'launch_receipt.json');status=read(run/'status.json')
    need(c['status']=='FAILED' and c['error']==m['error']=="AssertionError(('gpu_not_idle', 0))",'exact_observed_error')
    need(c['pid']==outer['child_pid']==1692998 and outer['wrapper_pid']==1692993,'actual_PID_chain')
    for key in ('internal_exit','external_exit'):need(type(outer[key]) is int and outer[key]==1,'strict_global1')
    need((run/'EXIT_CODE').read_bytes()==(run/'EXTERNAL_EXIT_CODE').read_bytes()==b'1\n','actual_double1')
    qsha=sha(payload/'validation_native_cost_queue.json')
    need(c['queue_sha256']==outer['queue_sha256']==qsha,'actual_queue_SHA')
    need(len(c['jobs'])==8 and status['pending']==8,'8_dispatched8_pending')
    planned={j['id']:j for j in q['jobs']};completed=[];partial=[];metadata=[]
    for row in c['jobs']:
        jid=row['id'];job=planned[jid];need('_C5_' in jid,'only_C5_dispatched')
        need(read(run/'jobs'/(jid+'.receipt.json'))==row,'raw_receipt_equals_completion')
        need(row['argv']==job['argv'] and row['parent_pid']==c['pid'] and row['queue_sha256']==qsha,'exact_job_conditions')
        need(row['requested_inputs_sha256']=={p:q['input_sha256'][p] for p in job['allowed_reads']},'exact_current_input_subset')
        need(dt(c['started_utc'])<=dt(row['started_utc'])<=dt(row['finished_utc'])<=dt(c['finished_utc']),'same_JJ_time_envelope')
        out=payload/'validation_native_cost'/jid;pred=out/'predictions.jsonl';lines=line_info(pred)
        info=dict(id=jid,pid=row['pid'],internal_exit=row['internal_exit'],external_exit=row['external_exit'],prediction_sha256=sha(pred),**lines)
        if jid.endswith('original_val'):
            zero(row['internal_exit']);zero(row['external_exit']);need((out/'EXIT_CODE').read_bytes()==b'0\n','completed_leaf_EXIT0')
            s=read(out/'summary.json');p=read(out/'protocol.json')
            need(s['status']=='PASS' and s['records']==904==lines['complete_output_lines'] and lines['ends_in_newline'],'904_completed_records')
            need(s['predictions_sha256']==info['prediction_sha256'] and s['pid']==p['pid']==row['pid'],'completed_producer_binding')
            need(s['model_state_before']==s['model_state_after'] and s['inputs_unchanged'] is True,'completed_state_unchanged')
            need(s['source_sha256']==binding['source_sha256'] and all(q['input_sha256'].get(k)==v for k,v in s['source_sha256'].items()),'same_scientific_sources')
            need(s['test_read'] is False and s['training'] is False and s['selection_performed'] is False and s['promotion'] is False,'scope_no_training_test_selection')
            access=read(payload/Path(job['access_log']).relative_to(root))
            need(access['pid']==row['pid'] and access['parent_pid']==c['pid'] and access['rejected_paths']==[],'completed_isolation_identity');zero(access['internal_exit'])
            completed.append(info)
        else:
            need(jid.endswith('tune') and row['internal_exit'] is None and row['external_exit']==-15 and row['terminated_after_peer_failure'] is True,'cancelled_tune')
            need(not (out/'summary.json').exists(),'partial_not_completed_summary');partial.append(info)
        metadata.append(info)
    need(len(completed)==len(partial)==4,'4_success4_cancel')
    need(all('_C10_' in j for j in set(planned)-{r['id'] for r in c['jobs']}),'all8_C10_unlaunched')
    need(not (run/'resource_samples.jsonl').exists(),'old_guard_has_no_failed_probe_snapshot')
    result=dict(schema='cap10_validation_resource_failure_independent_review_20261025',status='PASS',independent=True,
      scope='failed_validation_resource_execution_exposure_and_preservation_not_recovery_admission',
      error=c['error'],wrapper_pid=outer['wrapper_pid'],driver_pid=c['pid'],queue_sha256=qsha,
      started_utc=c['started_utc'],finished_utc=c['finished_utc'],copied_artifacts=82,copied_bytes=682580702,
      actual_input_pins_preserved=213,prior206_pins_preserved=True,training200_pins_preserved=True,
      completed_validation_cells=4,completed_validation_records=3616,cancelled_partial_tune_cells=4,unlaunched_C10_cells=8,
      completed_cells=completed,cancelled_cells=partial,
      cancelled_complete_output_lines_lower_bound=sum(x['complete_output_lines'] for x in partial),
      partial_scoring_may_exceed_complete_output_lines=True,metrics_values_interpreted=False,
      measured_utilization_memory_apps_at_failed_guard_available=False,
      no_claim_that_any_later_probe_equals_failed_guard_snapshot=True,
      training_draws_added=0,training_optimizer_steps_added=0,training_total_draws_remains=4194304,training_total_steps_remains=131072,
      actual_test_payload_read=False,selection_performed=False,scientific_recovery_admission=False,
      prior_successes_must_remain_preserved=True,automatic_retry=False,reviewed_sha256=pins,
      utc=datetime.now(timezone.utc).isoformat())
    for path,h in pins.items():need(sha(path)==h,'review_end_unchanged')
    out=ROOT/'validation_resource_failure_independent_review.json';need(not out.exists(),'unique_review')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',review=str(out),sha256=sha(out),partial_complete_lines=result['cancelled_complete_output_lines_lower_bound'])))
if __name__=='__main__':main()

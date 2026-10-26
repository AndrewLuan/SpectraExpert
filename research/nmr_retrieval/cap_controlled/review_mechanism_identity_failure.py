"""功能概述：独立审既有机制失败的保全、退出、身份/预算曝光边界，不解析预测或指标。
读取collector实际前后SHA及原始收据，流式复核原文件字节；prediction仅计完整换行与大小。
区分完整成功、进程失败、owned取消和未启动，不把加载/部分推理写成零科学曝光。
原after缺失保持缺失；另存failure后213输入实际SHA不伪装为原driver正常收尾。
只写独立review/receipt；不运行SSH、模型或恢复任务，也不审定未来修复准入。
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

QSHA='6f8c76a26501d749ad7a72c3f7985ed017112004b3e8650d5c58940b2a1e0711'
BSHA='2f56b0470a15b9746851aa4779c422ad4a8816af70f532450e18d75a9b3d556d'
def need(x,label):
    if not x:raise ValueError(label)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def safe(p):
    p=Path(p).absolute();need('..' not in p.parts and not p.is_symlink(),'no_path_redirection')
    r=p.resolve(strict=True);need(p==r,'canonical_existing_path')
    need(not any(v=='test' or v.startswith('sealed') for v in p.parts),'no_test_or_sealed_read');return p
def read(p):return json.loads(safe(p).read_bytes())
def stamp(x):
    t=datetime.fromisoformat(x.replace('Z','+00:00'));need(t.utcoffset().total_seconds()==0,'UTC');return t
def strict(x,value):need(type(x) is int and x==value,'strict_exit:'+str(value))
def line_info(p):
    lines=0;size=0;last=b''
    with safe(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            lines+=block.count(b'\n');size+=len(block);last=block[-1:]
    return dict(complete_output_lines=lines,bytes=size,ends_in_newline=not size or last==b'\n')

def review(manifest):
    mp=safe(manifest);m=read(mp);pins={str(mp):sha(mp),str(Path(__file__).resolve()):sha(__file__)}
    need(m['schema']=='cap10_mechanism_identity_failure_preservation_20261025','failure_preservation_schema')
    need(m['remote_before_sha256']==m['remote_after_sha256'] and m['inputs_and_sources_unchanged'] is True,'actual_remote_begin_end_unchanged')
    need(m['all_copied_bytes_SHA_verified'] is True and set(m['files'])==set(m['sha256']),'copied_file_map')
    need(set(m['remote_before_sha256'])==set(m['files'])|set(m['hash_only_inputs']),'full_current_input_artifact_union')
    for key in ('remote_inventory_before_exit','remote_inventory_after_exit','rsync_exit'):strict(m[key],0)
    for p,h in {**m['sha256'],**m['hash_only_inputs']}.items():need(m['remote_before_sha256'][p]==h,'full_union_actual_SHA')
    for remote,local in m['files'].items():
        path=safe(local);need(path.is_relative_to(mp.parent),'copied_containment')
        need(sha(path)==m['sha256'][remote] and path.stat().st_size==m['artifact_bytes'][remote],'raw_artifact_SHA_size')
        pins[str(path)]=m['sha256'][remote]
    root=Path(m['remote_root'])
    def local(p):
        p=str(p);need(p in m['files'],'required_preserved_file:'+p);return safe(m['files'][p])
    def doc(p):return read(local(p))
    qpath=root/'mechanism_resource_ready_queue.json';q=doc(qpath);need(sha(local(qpath))==QSHA,'exact_failed_queue')
    binding=doc(q['bindings']);need(sha(local(q['bindings']))==BSHA,'same_scientific_binding')
    need(q['phase']=='mechanism' and q['run_name']=='mechanism_resource_ready' and len(q['jobs'])==4,'four_registered_mechanism_cells')
    need(q['input_sha256']==binding['input_sha256']==m['input_sha256'] and len(m['input_sha256'])==213,'full213_actual_inputs_unchanged')
    need(q['source_sha256']==m['source_sha256'] and len(m['source_sha256'])==25,'full25_actual_sources_unchanged')
    need(q['operator_paths']=={role:dict(path=str(root/name),sha256=q['source_sha256'][str(root/name)]) for role,name in (
      ('supervisor','supervise_resource_ready_queue.py'),('launcher','launch_resource_ready_stage.py'))},'new_operator_binding')
    run=root/'runs/mechanism_resource_ready';c=doc(run/'completion.json');outer=doc(run/'launch_receipt.json');start=doc(run/'driver_start.json');status=doc(run/'status.json')
    need(c==m['completion'] and outer==m['launch'] and c['status']=='FAILED' and type(c['error']) is str and c['error'],'actual_failure_snapshot')
    for value in (c['external_exit'],outer['internal_exit'],outer['external_exit']):strict(value,1)
    need(local(run/'EXIT_CODE').read_bytes()==local(run/'EXTERNAL_EXIT_CODE').read_bytes()==b'1\n','actual_global_double1')
    need(c['queue_sha256']==outer['queue_sha256']==start['queue_sha256']==QSHA,'failure_queue_SHA_chain')
    need(c['pid']==start['pid']==outer['child_pid'] and start['parent_pid']==outer['wrapper_pid'],'wrapper_driver_PID_chain')
    expectedargv=[q['python'],'-B',q['operator_paths']['supervisor']['path'],'--queue',str(qpath),'--out',str(run),'--execute']
    need(outer['argv']==expectedargv and start['argv']==expectedargv[2:],'actual_driver_argv')
    need(stamp(outer['started_utc'])<=stamp(start['started_utc'])<=stamp(c['finished_utc'])<=stamp(outer['finished_utc']),'same_JJ_wrapper_time')
    need(doc(run/'input_fingerprints.before.json')==q['input_sha256'] and c['inputs_unchanged'] is None,'failed_driver_after_not_forged')
    need(m['original_after_exists'] is False and str(run/'input_fingerprints.after.json') not in m['files'],'original_after_stays_absent')
    planned={j['id']:j for j in q['jobs']};actual={j['id']:j for j in c['jobs']}
    need(len(actual)==len(c['jobs'])==4 and set(actual)==set(planned) and status['pending']==0,'all_four_dispatched_no_pending')
    cells=[]
    for jid,rec in actual.items():
        job=planned[jid];need(doc(run/'jobs'/(jid+'.receipt.json'))==rec,'completion_equals_job_receipt')
        need(rec['argv']==job['argv'] and rec['cwd']==job['cwd'] and rec['parent_pid']==c['pid'] and rec['queue_sha256']==QSHA,'exact_failed_scientific_conditions')
        need(rec['requested_inputs_sha256']=={p:q['input_sha256'][p] for p in job['allowed_reads']},'exact_three_tensor_subset')
        need(stamp(start['started_utc'])<=stamp(rec['started_utc'])<=stamp(rec['finished_utc'])<=stamp(c['finished_utc']),'same_JJ_child_time')
        ie,ee=rec['internal_exit'],rec['external_exit'];need(ie is None or type(ie) is int,'strict_inner_exit_type');need(type(ee) is int,'strict_outer_exit_type')
        category='complete' if ie==0 and ee==0 else ('owned_cancelled' if ee<0 and rec.get('terminated_after_peer_failure') is True else 'failed')
        out=Path(job['job_out']);pp=str(out/'predictions.jsonl');lines=line_info(local(pp)) if pp in m['files'] else dict(complete_output_lines=0,bytes=0,ends_in_newline=True)
        if str(out/'summary.json') in m['files']:
            s=doc(out/'summary.json');need(category=='complete' and s['status']=='PASS' and s['records']==200 and s['pid']==rec['pid'],'only_successful_summary')
        cells.append(dict(id=jid,pid=rec['pid'],internal_exit=ie,external_exit=ee,category=category,
          predictions_exist=pp in m['files'],prediction_sha256=m['sha256'].get(pp),**lines))
    need(any(x['id']=='mechanism_C5_s211_dual_train' and x['internal_exit']==x['external_exit']==1 for x in cells),'observed_C5_seed211_inner_outer1')
    for p,h in pins.items():need(sha(safe(p))==h,'review_end_file_drift')
    return dict(schema='cap10_mechanism_identity_failure_independent_review_20261025',status='PASS',independent=True,
      reviewer='evidence_audit independent preservation and lifecycle reviewer; root/F own diagnosis and future repair',
      scope='failed_mechanism_execution_preservation_and_exposure_not_recovery_admission',
      queue_sha256=QSHA,bindings_sha256=BSHA,wrapper_pid=outer['wrapper_pid'],driver_pid=c['pid'],
      started_utc=c['started_utc'],finished_utc=c['finished_utc'],error=c['error'],copied_artifacts=len(m['files']),
      actual_input_pins_preserved=213,actual_source_pins_preserved=25,all_four_dispatched=True,cells=cells,
      completed_cells=sum(x['category']=='complete' for x in cells),failed_cells=sum(x['category']=='failed' for x in cells),
      owned_cancelled_cells=sum(x['category']=='owned_cancelled' for x in cells),unlaunched_cells=0,
      complete_output_lines_lower_bound=sum(x['complete_output_lines'] for x in cells),
      partial_scoring_may_exceed_complete_output_lines=True,model_loading_or_cache_reading_may_have_occurred=True,
      metrics_or_prediction_values_interpreted=False,original_driver_after_exists=False,
      failure_after_SHA_is_separate_collection_not_successful_driver_after=True,
      PID_observations=m['PID_observations'],training_draws_added=0,optimizer_steps_added=0,test_payload_reads=0,
      validation_rerun_requested=False,selection_performed=False,automatic_retry=False,recovery_admission=False,
      reviewed_sha256=pins,utc=datetime.now(timezone.utc).isoformat())

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);need(not out.exists(),'unique_review_path')
    result=review(a.manifest);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',review=str(out),sha256=sha(out),completed=result['completed_cells'],failed=result['failed_cells'],cancelled=result['owned_cancelled_cells'])))
if __name__=='__main__':main()

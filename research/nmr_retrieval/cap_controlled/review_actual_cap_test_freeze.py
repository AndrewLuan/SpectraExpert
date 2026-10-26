"""功能概述：非作者独立核本轮实际 frozen selection 与12项test队列。
本模块只复用自己的只读路径/SHA工具与上一轮具体plan审查结果，绝不调用作者freeze。
check 独立构造冻结字段、每格精确CLI和5条允许读取路径、212小元数据源与249最终pins；
只核已准入小字节和远端部署收据。四份test缓存不读取也不stream hash，其实际SHA由
最终审核后operator在首次加载前核对。main只写reviewer自己的新目录，失败保留EXIT1。
"""
from pathlib import Path
import argparse, datetime, json
import review_actual_cap_test_plan as p
M=p.M;S=p.S;R=p.R
def check():
    p.read(__file__);p.read(p.__file__)
    old=p.doc(M/'test_plan_actual_review_attempt2/checks.json')
    for path,h in old['read_sha256'].items():p.read(path,h)
    plan=p.doc(S/'prepare/test_plan.json');b=p.doc(S/'prepare/test_bindings.json')
    oldrequest=p.doc(S/'prepare_request.json');request=p.doc(S/'freeze_request.json')
    review=p.doc(M/'test_plan_independent_review.json','9e6d67c7a8f9bab8280eb3d2f86240ccd4e7834b95f9c8a18f39bf563b91c587')
    p.need(review['status']=='PASS' and review['independent'] is True and all(review['reviewed_sha256'].get(k)==v for k,v in old['required'].items()),'prior_exact_plan_review')
    p.need(request['refs']==oldrequest['refs'] and request['scope']==oldrequest['scope'] and request['stage_contract']==oldrequest['stage_contract'],'same_prepared_contract')
    stage=request['freeze_stage_refs']
    expected_stage={'bindings':{'path':R+'/test_bindings.json','sha256':p.digest(p.read(S/'prepare/test_bindings.json'))},'plan':{'path':R+'/test_plan.json','sha256':p.digest(p.read(S/'prepare/test_plan.json'))},'plan_review':{'path':R+'/test_plan_independent_review.json','sha256':p.digest(p.read(M/'test_plan_independent_review.json'))}}
    p.need(stage==expected_stage,'exact_plan_stage_refs')
    oldmap=p.doc(oldrequest['metadata_map']['path'],oldrequest['metadata_map']['sha256'])
    alias=p.doc(request['metadata_map']['path'],request['metadata_map']['sha256'])
    newlocal={'bindings':S/'prepare/test_bindings.json','plan':S/'prepare/test_plan.json','plan_review':M/'test_plan_independent_review.json'}
    files=dict(oldmap['files']);hashes=dict(oldmap['sha256'])
    for role,ref in stage.items():files[ref['path']]=str(newlocal[role]);hashes[ref['path']]=ref['sha256']
    p.need(alias==dict(files=files,sha256=hashes) and len(files)==67,'exact67_alias_extension')
    for remote,path in files.items():p.read(path,hashes[remote])
    fpath=S/'freeze/frozen_test_selection.json';qpath=S/'freeze/test_queue.json'
    frozen=p.doc(fpath,'4c19b976f4bba9baa747e1fd19550bd4d68bad7a1ef8673ad9ed45894f331354')
    q=p.doc(qpath,'18a01669cc5054150e585c8f15c56b3825bbe36797247a0dbf7305cacc195ced')
    final=p.doc(S/'freeze/required_final_review_sha256.json','8c774b1e0657045269f1ebce485b805e8daf7d1714ba20ebbcbf3444f594e27c')
    refs=request['refs'];inputs=p.merge(plan['evidence_input_sha256'],b['input_sha256'],{stage['plan_review']['path']:stage['plan_review']['sha256']})
    expected_frozen=dict(schema='cap10_sealed_selection_20261024',selection_locked=True,promotion=False,preselected=plan['preselected'],entries=plan['entries'],fixed_test_identity=plan['fixed_test_identity'],input_sha256=inputs,source_sha256=plan['source_sha256'],test_plan=stage['plan'],validation_summary=refs['validation_summary'],validation_selection=refs['validation_lock'],validation_review=refs['validation_review'],validation_rule_applied_selection=refs['validation_selection'],validation_selection_semantics='Final locked selection document; immutable rule-applied selection is separately referenced.',validation_lock=refs['validation_lock'],freeze_review=stage['plan_review'],tool_paths=plan['tool_paths'],frozen_utc=frozen['frozen_utc'])
    p.need(frozen==expected_frozen and len(inputs)==245,'exact_frozen245_and_locked_DAG')
    finalpath=R+'/test_freeze_independent_review.json';fref={'path':R+'/frozen_test_selection.json','sha256':p.digest(p.read(fpath))}
    sources=p.merge(plan['source_sha256'],{k:v for k,v in plan['evidence_input_sha256'].items() if Path(k).suffix not in ('.pt','.pth','.ckpt')},{R+'/protocol.json':refs['protocol']['sha256'],stage['bindings']['path']:stage['bindings']['sha256'],fref['path']:fref['sha256'],stage['plan']['path']:stage['plan']['sha256'],stage['plan_review']['path']:stage['plan_review']['sha256']})
    p.need(finalpath not in sources and finalpath not in b['input_sha256'] and finalpath not in inputs,'no_final_review_SHA_cycle')
    protocol=p.doc(files[refs['protocol']['path']],refs['protocol']['sha256']);python='/home/user/.venvs/spectrexpert/bin/python'
    jobs=[]
    for n,e in enumerate(plan['entries']):
        ca=b['caches'][str(e['seed'])]['test'][e['pool']];producer=ca['producer']
        allowed=[e['cache'],e['checkpoints']['no_aug']['path'],e['checkpoints']['aug']['path'],producer['path'],producer['exit_path']]
        p.need(len(allowed)==len(set(allowed))==5 and len([x for x in allowed if Path(x).suffix=='.pt'])==3 and all(x in b['input_sha256'] for x in allowed),'exact5_allowed_3_tensors')
        command=[b['evaluator']['path'],'--source',protocol['source_checkout'],'--protocol',R+'/protocol.json','--bindings',R+'/test_bindings.json','--frozen-selection',fref['path'],'--freeze-review',finalpath,'--config',e['config'],'--seed',str(e['seed']),'--cache',e['cache'],'--out',e['job_out'],'--device','cuda:0','--threads','4']
        access=R+'/access_logs/test/'+e['job_id']+'.json'
        argv=[python,'-B',R+'/isolation_entry.py','--bindings',R+'/test_bindings.json','--phase','test','--access-log',access,'--output-root',e['job_out']]
        for a in allowed:argv+=['--allow-read',a]
        argv+=['--']+command
        jobs.append(dict(id=e['job_id'],argv=argv,command=command,cwd=protocol['source_checkout'],gpu=n%8,job_out=e['job_out'],access_log=access,allowed_reads=allowed))
    ops={role:{'path':R+'/'+name,'sha256':sources[R+'/'+name]} for role,name in [('supervisor','supervise_resource_ready_queue.py'),('launcher','launch_resource_ready_stage.py')]}
    expected_q=dict(schema='cap10_controlled_queue_20261024',phase='test',run_name='test',root=R,local_root=protocol['local_root'],python=python,bindings=R+'/test_bindings.json',selection_locked_before_test=True,jobs=jobs,input_sha256=b['input_sha256'],source_sha256=sources,review_paths=[finalpath],maximum_seconds=10800,max_parallel=8,operator_paths=ops,resource_wait={'maximum_seconds':120,'poll_seconds':.2},external_api_calls=0,promotion=False,automatic_retry=False)
    p.need(q==expected_q and len(q['jobs'])==12 and len(sources)==212,'exact_12_queue_full_argv_source212')
    required=p.merge(old['required'],inputs,sources,{fref['path']:fref['sha256'],R+'/test_queue.json':p.digest(p.read(qpath))})
    p.need(required==final and len(final)==249,'all249_final_required_reconstructed')
    p.need(set(b['source_sha256']).isdisjoint({plan['tool_paths'][k]['path'] for k in ('statistics','pairmath','matrix')}),'leaf_excludes_downstream_statistics')
    dep=p.doc(S/'deployment/frozen_outputs_receipt.json','c1ac324c5e531293687831bf6f02bff3fee0bfe9bc69a15e2042394676163c2c')
    for key in ('before_probe_exit','rsync_exit','after_probe_exit'):p.zero(dep[key])
    p.need(dep['expected_sha256']==dep['actual_remote_sha256']==p.merge(sources,{R+'/test_queue.json':p.digest(p.read(qpath))}) and len(dep['actual_remote_sha256'])==213,'actual_remote213_small_files')
    p.need(dep['test_tensor_reads']==0 and dep['test_prediction_reads']==0 and dep['scientific_dispatch']==0,'deployment_no_test_tensor_SHA_or_load')
    execution=p.doc(S/'freeze.actual_execution.json')
    p.zero(execution['internal_exit']);p.zero(execution['external_exit'])
    p.need(execution['request_sha256']==p.digest(p.read(S/'freeze_request.json')) and execution['metadata_map_sha256']==request['metadata_map']['sha256'],'actual_freeze_inputs')
    p.need(execution['argv']==[execution['argv'][0],'-B',str(M/'prepare_cap_test_metadata.py'),'--stage','freeze','--request',str(S/'freeze_request.json'),'--out',str(S/'freeze')],'actual_freeze_exact_argv')
    p.need(p.stamp(review['utc'])<=p.stamp(execution['started_utc'])<=p.stamp(frozen['frozen_utc'])<=p.stamp(execution['finished_utc']),'review_before_frozen_UTC')
    p.need(execution['test_tensors_opened'] is False and execution['test_predictions_opened'] is False and execution['scientific_dispatch']==0,'metadata_only_freeze')
    for path,h in execution['outputs_sha256'].items():p.read(path,h)
    receipt=p.doc(S/'freeze/receipt.json','7529fda0bae5405ed9404c9a1f3584ea085c35f403552f00dc0083b927101818')
    p.need(receipt['source_sha256']=={str(M/'prepare_cap_test_metadata.py'):'39c9ec6036e7852798dd7de881cf972aaa799edf90c96c1b01cfc2b9ca10e30e'} and receipt['test_tensor_bytes_read'] is False and receipt['test_predictions_read'] is False and receipt['model_calls']==0,'actual_reviewed_builder_freeze_receipt')
    for path,h in p.READS.items():p.need(p.digest(p.read(path))==h,'final_input_SHA_drift')
    return dict(required=required,read_sha256=dict(p.READS),preselected=frozen['preselected'],declared_test_cache_SHA_deferred=old['declared_test_cache_SHA_deferred'],frozen_ref=fref,queue_ref={'path':R+'/test_queue.json','sha256':p.digest(p.read(qpath))},checks=dict(exact249_required=True,exact212_queue_sources_and213_remote_files=True,exact12_allfresh_jobs=True,exact85_runtime_inputs_and5_allowed3_tensors_per_cell=True,exact_native_cli_and_resource_operators=True,fixed_preselected_locked_DAG_no_cycles=True,leaf_excludes_statistics_source=True,actual_freeze_strict_double0=True,all_small_input_SHA_unchanged=True),actual_test_tensor_reads=0,actual_test_prediction_reads=0,scientific_dispatch=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(exist_ok=False)
    try:
        result=check();(out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');(out/'EXIT_CODE').write_text('0\n')
        print(json.dumps({'status':'PASS','required':len(result['required']),'preselected':result['preselected']}))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps({'error':repr(exc)},indent=2)+'\n');(out/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()

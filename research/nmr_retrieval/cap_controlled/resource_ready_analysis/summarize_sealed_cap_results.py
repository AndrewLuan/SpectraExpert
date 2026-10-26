#!/usr/bin/env python3
"""功能概述：新资源就绪operator收据与原生32/64成本精度下完整12项baseline/cap test执行之后，核实冻结与真实收据，再汇总全部统计。
main先调用sealed_cap_evaluation_common.sealed_gate，确认先验验证选择及独立冻结；
execution核完整12job外层/driver/leaf严格整数0、精确CLI、PID/PPID/UTC、输入源码始末SHA。
read_predictions只在全部metadata通过后读取精确SHA的12份1371预测；validate/analyze_matrix
执行预注册324指标和180配对strata。无模型、训练、择优、API或正式系统变更。
代码库关系：依赖独立新sealed common及纯matrix/pairmath，完全不改已锁val4source。
本入口采用all12 fresh evaluation，不支持部分基准复用；缺完整旧复用资格时协议允许此路径。
输入输出与副作用：--protocol/bindings/frozen-selection/freeze-review为本阶段实际文件，
--queue/run指唯一12job成功run，--analysis-review为独立实施review，--out必须不存在。
只新建analysis.json、metrics.tsv、summary.json、protocol.json和EXIT_CODE；失败写failure。
预测在全部真实执行核验后才读取，模型/feature张量仅比较已核收据中的SHA而不直接打开。三tensor和本cache的两条metadata逐项白名单。
"""
from __future__ import annotations
import argparse,collections,csv,io,json,os,pathlib,sys
_own_dir=pathlib.Path(__file__).resolve().parent
_native_dir=_own_dir.parent/'native_cost_evaluation'
sys.path.insert(0,str(_native_dir))
sys.path.append(str(_own_dir.parent))
import cap_test_evaluation_common as c
if pathlib.Path(c.__file__).resolve()!=_native_dir/'cap_test_evaluation_common.py':
    raise RuntimeError('exact_native_common_import_required')
import sealed_cap_matrix as matrix
import sealed_cap_pairmath as pm
import numpy as np
Path=pathlib.Path

def zero(value,label):
    c.need(type(value) is int and value==0,label+'_strict_int0')

def argv_value(argv,name):
    c.need(argv.count(name)==1,'single_CLI_'+name)
    i=argv.index(name);c.need(i+1<len(argv),'CLI_value_'+name);return argv[i+1]

def load_metadata(pins,path):
    return c.metadata_json(pins,path)

def execution(pins,args,bindings,frozen):
    q=load_metadata(pins,args.queue);run=Path(args.run).resolve()
    c.need(q['phase']=='test' and q['selection_locked_before_test'] is True,'test_queue_frozen')
    c.need(run==Path(q['root'])/'runs'/q.get('run_name','test'),'actual_run_path')
    c.need(q['bindings']==str(Path(args.bindings).resolve()) and q['input_sha256']==bindings['input_sha256'],'actual_registry_queue')
    c.need(len(q['jobs'])==12 and len({j['id'] for j in q['jobs']})==12,'all12_fresh_jobs')
    c.need(all(e['execution']=='evaluate' for e in frozen['entries']),'no_unqualified_baseline_reuse')
    sources=q['source_sha256'];c.need(all(sources.get(p)==h for p,h in bindings['source_sha256'].items()),'all_scientific_sources')
    start=load_metadata(pins,run/'driver_start.json');done=load_metadata(pins,run/'completion.json');outer=load_metadata(pins,run/'launch_receipt.json')
    c.need(done['status']=='COMPLETE' and done['inputs_unchanged'] is True,'actual_COMPLETE')
    zero(done['external_exit'],'driver');zero(outer['internal_exit'],'outer_internal');zero(outer['external_exit'],'outer_external')
    c.need(pins.raw(run/'EXIT_CODE').decode().strip()=='0' and pins.raw(run/'EXTERNAL_EXIT_CODE').decode().strip()=='0','driver_double0_files')
    qsha=pins.values[str(Path(args.queue).resolve())]
    c.need(start['queue_sha256']==done['queue_sha256']==outer['queue_sha256']==qsha,'same_actual_queue')
    c.need(start['source_sha256']==outer['source_sha256']==sources and start['input_sha256']==q['input_sha256'],'actual_source_input_maps')
    c.need(start['pid']==done['pid']==outer['child_pid'] and start['parent_pid']==outer['wrapper_pid'],'driver_outer_PID_chain')
    refs=q.get('operator_paths')
    c.need(type(refs) is dict and set(refs)=={'supervisor','launcher'},'exact_resource_operator_refs')
    wait=q.get('resource_wait')
    c.need(type(wait) is dict and set(wait)=={'maximum_seconds','poll_seconds'} and type(wait['maximum_seconds']) is int and wait['maximum_seconds']==120 and type(wait['poll_seconds']) in (int,float) and wait['poll_seconds']==.2,'exact_resource_wait120_poll02')
    for role,name in [('supervisor','supervise_resource_ready_queue.py'),('launcher','launch_resource_ready_stage.py')]:
        ref=refs[role];expected=str(Path(q['root'])/name)
        c.need(type(ref) is dict and set(ref)=={'path','sha256'} and ref['path']==expected and str(Path(ref['path']).resolve())==expected and c.hexsha(ref['sha256']),'exact_resource_operator_role_path')
        c.need(sources.get(expected)==ref['sha256'],'resource_operator_in_actual_sources')
        pins.add(expected,ref['sha256'])
    c.need(all(v.get('operator_paths')==refs for v in (start,done,outer)),'resource_operator_actual_envelopes')
    c.need(start.get('resource_wait')==done.get('resource_wait')==wait,'resource_wait_actual_envelopes')
    exact=[q['python'],'-B',refs['supervisor']['path'],'--queue',str(Path(args.queue).resolve()),'--out',str(run),'--execute']
    c.need(outer['argv']==exact and start['argv']==exact[2:],'outer_and_start_exact_driver_argv')
    for stem,expected in [('input',q['input_sha256']),('operator',None)]:
        before=load_metadata(pins,run/(stem+'_fingerprints.before.json'));after=load_metadata(pins,run/(stem+'_fingerprints.after.json'))
        c.need(before==after and bool(before),'unchanged_'+stem)
        if expected is not None:c.need(before==expected,'complete_input_fingerprints')
        else:
            expected_small=dict(sources);expected_small[str(Path(args.queue).resolve())]=qsha
            for p in q['review_paths']:expected_small[str(Path(p).resolve())]=pins.add(p)
            c.need(before==expected_small,'complete_operator_fingerprints')
    for p,h in sources.items():pins.add(p,h)
    c.need(len(done['jobs'])==12 and len({x['id'] for x in done['jobs']})==12,'complete12_actual_receipts')
    donejobs={j['id']:j for j in done['jobs']};entries={(e['config'],e['seed']):e for e in frozen['entries']}
    results={};payload_refs=[];seen=set()
    for job in q['jobs']:
        command=job['command'];ref=bindings['evaluator'];evaluator=str(Path(ref['path']).resolve())
        c.need(ref['path']==evaluator and c.hexsha(ref.get('sha256')) and str(Path(command[0]).resolve())==evaluator
               and ref['sha256']==bindings['source_sha256'].get(evaluator)==sources.get(evaluator),'only_exact_registered_native_evaluator')
        config=argv_value(command,'--config');seed=int(argv_value(command,'--seed'));key=(config,seed)
        c.need(key in entries and key not in seen,'exact_cell_once');seen.add(key);entry=entries[key]
        c.need(Path(job['job_out']).resolve()==Path(entry['job_out']).resolve(),'actual_entry_out')
        for name,value in [('protocol',args.protocol),('bindings',args.bindings),('frozen-selection',args.frozen_selection),('freeze-review',args.freeze_review),('cache',entry['cache']),('out',job['job_out'])]:
            c.need(Path(argv_value(command,'--'+name)).resolve()==Path(value).resolve(),'exact_leaf_CLI_'+name)
        expected=[q['python'],'-B',str(Path(q['root'])/'isolation_entry.py'),'--bindings',q['bindings'],'--phase','test','--access-log',job['access_log'],'--output-root',job['job_out']]
        for p in job['allowed_reads']:expected+=['--allow-read',p]
        c.need(job['argv']==expected+['--']+command,'exact_guarded_argv')
        tensors={entry['cache'],*(r['path'] for r in entry['checkpoints'].values())}
        producer=bindings['caches'][str(seed)]['test'][entry['pool']]['producer']
        metadata={producer['path'],producer['exit_path']};reads=tensors|metadata
        c.need(len(tensors)==3 and len(metadata)==2 and all(Path(x).suffix=='.pt' for x in tensors) and set(job['allowed_reads'])==reads and len(job['allowed_reads'])==5,'only_three_tensors_plus_two_exact_cache_metadata')
        rec=load_metadata(pins,run/'jobs'/(job['id']+'.receipt.json'));beg=load_metadata(pins,run/'jobs'/(job['id']+'.started.json'))
        c.need(rec==donejobs[job['id']],'receipt_same_completed_job')
        zero(rec['internal_exit'],'job_internal');zero(rec['external_exit'],'job_external')
        c.need(rec['argv']==job['argv'] and rec['queue_sha256']==qsha and all(rec.get(k)==v for k,v in beg.items()),'job_initial_record_preserved')
        c.need(rec['cwd']==job['cwd']==bindings['source_checkout'] and rec['gpu']==job['gpu'] and rec['job_out']==job['job_out'] and rec['access_log']==job['access_log'],'job_actual_environment_paths')
        c.need(rec['requested_inputs_sha256']=={p:q['input_sha256'][p] for p in job['allowed_reads']},'job_exact_requested_input_pins')
        c.need(rec['pid']==beg['pid'] and rec['parent_pid']==beg['parent_pid']==start['pid'],'job_PID_chain')
        out=Path(job['job_out']);c.need(pins.raw(out/'EXIT_CODE').decode().strip()=='0','leaf_EXIT0')
        s=load_metadata(pins,out/'summary.json');p=load_metadata(pins,out/'protocol.json')
        c.need(s['status']=='PASS' and s['inputs_unchanged'] is True and s['model_state_before']==s['model_state_after'],'leaf_PASS_immutable')
        c.need((s['config'],s['seed'])==key and s['pool']==entry['pool'] and s['cohort']=='test' and s['records']==1371,'leaf_exact1371_cell')
        c.need(all(type(x) is int and x>0 for x in (s['pid'],s['ppid'],rec['pid'],rec['parent_pid'],start['pid'],outer['wrapper_pid'])) and s['pid']==rec['pid'] and s['ppid']==start['pid'],'leaf_actual_PID')
        for name in ('source','protocol','bindings','frozen_selection','freeze_review','config','seed','cache','out','device','threads'):
            cli='--'+name.replace('_','-');actual=argv_value(command,cli)
            c.need(str(p['args'][name])==actual,'producer_protocol_actual_CLI_'+name)
        c.need(c.parse_utc(outer['started_utc'])<=c.parse_utc(start['started_utc'])<=c.parse_utc(rec['started_utc'])<=c.parse_utc(s['started_utc'])<=c.parse_utc(s['finished_utc'])<=c.parse_utc(rec['finished_utc'])<=c.parse_utc(done['finished_utc'])<=c.parse_utc(outer['finished_utc']),'same_host_UTC_containment')
        c.need(s['test_read'] is True and s['training'] is False and s['selection_performed'] is False and s['promotion'] is False and s['external_api_calls']==0,'leaf_scope')
        c.need(s['source_sha256']==bindings['source_sha256'],'leaf_exact_sources')
        c.need(s['evaluator']==bindings['evaluator'],'leaf_exact_evaluator_reference')
        c.need(s['cache_sha256']==entry['cache_sha256'] and s['candidate_order_sha256']==c.ORDER_SHA and s['query_sha256']==frozen['fixed_test_identity']['query_sha256'],'leaf_cache_universe')
        checks={r['path']:r['sha256'] for r in entry['checkpoints'].values()};c.need(s['checkpoints']==checks,'leaf_exact_weights')
        expectedpins={entry['cache']:entry['cache_sha256'],**checks,**bindings['source_sha256']}
        c.need(all(s['input_sha256'].get(k)==v for k,v in expectedpins.items()),'leaf_complete_input_pins')
        access=load_metadata(pins,job['access_log'])
        zero(access['internal_exit'],'guard_internal')
        c.need(access['phase']=='test' and access['error'] is None and access['rejected_paths']==[],'guard_access_PASS')
        c.need(access['pid']==rec['pid'] and access['parent_pid']==start['pid'] and access['argv']==command,'guard_process_command')
        c.need(c.parse_utc(rec['started_utc'])<=c.parse_utc(access['started_utc'])<=c.parse_utc(s['started_utc'])<=c.parse_utc(s['finished_utc'])<=c.parse_utc(access['finished_utc'])<=c.parse_utc(rec['finished_utc']),'access_leaf_UTC_containment')
        c.need(set(access['allowed_reads'])==reads and len(access['allowed_reads'])==5,'guard_exact_allowed_tensors_and_metadata')
        c.need(reads<=set(access['recorded_reads']) and all((p in reads or Path(p).is_relative_to(out)) and type(n) is int and n>0 for p,n in access['recorded_reads'].items()),'guard_exact_recorded_science_reads')
        payload_refs.append((key,out/'predictions.jsonl',s['predictions_sha256']))
        results[str(key)]=s
    c.need(seen==set(entries),'all_registered12_results')
    return q,results,payload_refs

def native_cost_dtype(row):
    """只接受已审fp32校正与原始fp32/fp64粗成本的真实vector提升，不自行cast成本。"""
    allowed={'torch.float32':np.float32,'torch.float64':np.float64}
    coarse=row.get('coarse_dtype');correction=row.get('correction_dtypes');final=row.get('final_cost_dtype')
    c.need(coarse in allowed and correction==['torch.float32','torch.float32'],'registered_native_operand_dtypes')
    c.need(final==coarse,'native_vector_promotion_dtype')
    return allowed[final]


def native_threshold(cost,dtype):
    # tensor + Python scalar先把EPS转为该tensor dtype，再以同dtype执行加法。
    return float(dtype(dtype(cost)+dtype(c.EPS)))


def validate_payload_row(row):
    c.validate_prediction(row)
    ids=row['natural_candidate_indices'];values=row['natural_candidate_costs']
    c.need(type(ids) is list and len(ids)==512 and all(type(i) is int and 0<=i<c.CANDIDATES for i in ids) and len(set(ids))==512,'native512_unique_candidate_IDs')
    c.need(type(values) is list and len(values)==512 and all(c.finite(x) for x in values),'native512_finite_costs')
    dtype=native_cost_dtype(row);costs=np.array(values,dtype=dtype)
    c.need(bool(np.isfinite(costs).all()) and costs.astype(np.float64).tolist()==values,'exact_native_declared_dtype_costs')
    coarse=row.get('natural_candidate_coarse_costs')
    c.need(type(coarse) is list and len(coarse)==512 and all(c.finite(x) for x in coarse),'native512_finite_coarse_costs')
    original=np.array(coarse,dtype=dtype)
    c.need(bool(np.isfinite(original).all()) and original.astype(np.float64).tolist()==coarse,'exact_native_declared_coarse_dtype')
    order=np.lexsort((np.array(ids,dtype=np.int64),costs))
    c.need(row['winner_candidate_index']==ids[int(order[0])],'native_winner_from_512')
    c.need(all(row['top'+str(k)] is (row['rank']<=k) for k in (1,5,10)),'native_TopK_from_rank')
    if row['target_in_pool']:
        target=row['target_candidate_index'];c.need(type(target) is int and target in ids,'native_target_in_512')
        pos=ids.index(target);cost=float(costs[pos]);rank=int((costs<=native_threshold(cost,dtype)).sum())
        c.need(row['rank']==rank and row['target_cost']==cost,'native_rank_from_declared_dtype_costs')
        other=int(order[order!=pos][0]);c.need(row['margin']==cost-float(costs[other]),'native_margin_from_512')
    else:
        c.need(row['target_candidate_index'] is None and row['target_cost'] is None and row['margin'] is None,'native_outside_pool_semantics')
    pm.normalize(row)
    return row

def verify_leaf_values(rows,summary):
    c.need(len(rows)==1371,'full1371_payload')
    for row in rows:validate_payload_row(row)
    c.need(dict(__import__('collections').Counter(r['query_identity']['variant'] for r in rows))==pm.VARIANTS,'full_variant_payload')
    identity=c.ordered_sha(sorted(c.identity_sha(r) for r in rows))
    c.need(summary['identity_sha256']==identity,'leaf_actual_identity_SHA')
    for variant in pm.VARIANTS:
        actual=c.metrics([r for r in rows if r['query_identity']['variant']==variant]);claimed=summary['metrics'][variant]
        c.need(set(actual)==set(claimed),'leaf_complete_metrics')
        for k,v in actual.items():
            a=claimed[k]
            c.need((v is None and a is None) or (type(v) is int and type(a) is int and v==a) or
                   (type(v) in (int,float) and type(a) in (int,float) and c.finite(v) and c.finite(a) and abs(v-a)<2e-12),'leaf_metric_from_actual_predictions_'+variant+'_'+k)
    return True


def run(args,out):
    began=c.utc()
    own=[Path(__file__).resolve(),Path(matrix.__file__).resolve(),Path(pm.__file__).resolve(),Path(c.__file__).resolve()]
    pins,protocol,bindings,frozen,_=c.sealed_gate(args,own,require_cell=False)
    review=load_metadata(pins,args.analysis_review);c.reviewed(review,{str(p):pins.values[str(p)] for p in own},'analysis_implementation')
    c.need('implementation' in review['scope'],'analysis_review_implementation_scope')
    q,results,refs=execution(pins,args,bindings,frozen)
    # 首次读取预测只发生在所有12job元数据和冻结独审通过之后。
    raw={name:{} for name in matrix.CONFIGS}
    for (config,seed),path,digest in refs:
        rows=[json.loads(line) for line in pins.raw(path,digest).decode().splitlines() if line.strip()]
        verify_leaf_values(rows,results[str((config,seed))]);raw[config][seed]=rows
    values=matrix.analyze_matrix(raw,protocol['test_preregistration'])
    pins.finish()
    c.write(out/'analysis.json',values)
    rows=values['metrics'];fields=list(dict.fromkeys(k for row in rows for k in row))
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(rows)
    (out/'metrics.tsv').write_text(stream.getvalue())
    summary=dict(schema='cap10_sealed_results_20261025',status='PASS',preselected=frozen['preselected'],test_reselection_performed=False,
      test_cells=12,new_evaluation_jobs=12,metric_rows=324,comparison_strata=180,bootstrap_seed=317,bootstrap_replicates=5000,
      source_sha256={str(p):pins.values[str(p)] for p in own},input_sha256=pins.values,inputs_unchanged=True,
      cost_semantics='No cost cast. Frozen fp32 corrections plus original coarse fp32/fp64 promote natively; EPS cast before same-dtype addition.',
      native_final_dtype_records=dict(collections.Counter(r['final_cost_dtype'] for cells in raw.values() for rows in cells.values() for r in rows)),
      native_coarse_dtype_records=dict(collections.Counter(r['coarse_dtype'] for cells in raw.values() for rows in cells.values() for r in rows)),
      output_sha256={str(out/n):c.sha(out/n,True) for n in ('analysis.json','metrics.tsv')},
      started_utc=began,finished_utc=c.utc(),pid=os.getpid(),ppid=os.getppid(),training=False,model_calls=0,
      test_read=True,external_api_calls=0,selection_performed=False,promotion=False,official_system_changed=False,
      test_blind_claim=False,independent_acceptance_status='unavailable')
    c.write(out/'summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser()
    for k in ('source','protocol','bindings','frozen-selection','freeze-review','queue','run','analysis-review','out'):p.add_argument('--'+k,required=True)
    args=p.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    c.write(out/'protocol.json',{'args':vars(args),'pid':os.getpid(),'ppid':os.getppid(),'started_utc':c.utc()})
    try:
        s=run(args,out);(out/'EXIT_CODE').write_text('0\n');print(json.dumps({'status':'PASS','metric_rows':s['metric_rows'],'comparison_strata':s['comparison_strata']}))
    except BaseException as exc:
        c.write(out/'failure.json',{'error':repr(exc),'utc':c.utc()});(out/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()

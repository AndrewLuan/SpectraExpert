"""功能概述：本轮完整16项validation的唯一预注册C10dual选择算术与证据汇总。
load_matrix核root operator真实queue/内外退出/实际argv和eval输出；read_evaluation逐query
重算natural rank/fullRR/TopK/metrics。aggregate先合并320+1024同seed行，再等两seed，
source macro每source等权1/5，条件in-pool比值也先每seed后均值。没有缺失source补0。
training_audit核完整8fit最终64/alpha1/冻结center/公平曝光审核；select只允许C10dual，
C5dual是matchedcontrol，新single仅机制；失败保留历史fixedsingle，不枚举替补。
run写648行全cohort/variant/source表、原始cost metadata、SHA及独审待办，不训练、
不读test、不做promotion。root独立结果审核PASS后才能锁定用于后续test freeze。
"""
import argparse
import collections
import csv
import itertools
import json
import math
from pathlib import Path
from cap_evaluation_common import *

SCHEMA='cap_validation_selection_20261024'
METRICS=('MRR','top1','top5','top10','recall512','in_pool_MRR')

def argv_value(argv,flag):
    need(type(argv) is list and argv.count(flag)==1,'exact_argv_flag:'+flag)
    i=argv.index(flag);need(i+1<len(argv),'argv_value:'+flag);return argv[i+1]

def eval_role(job):
    argv=job['command']
    need(bool(argv) and Path(argv[0]).name=='evaluate_cap_ranker.py','registered_evaluator')
    return (int(argv_value(argv,'--cap')),int(argv_value(argv,'--seed')),argv_value(argv,'--pool'),argv_value(argv,'--cohort'))

def load_matrix(pins,queue_path,run_path,bindings_path,protocol_path,kind):
    queue,envelope=driver_receipts(pins,queue_path,run_path)
    need(queue['phase']==kind,'exact_queue_phase')
    if kind=='validation':wanted=set(itertools.product(CAPS,SEEDS,POOLS,('original_val','tune')))
    else:
        need(kind=='mechanism','matrix_kind');wanted={(c,s,'dual','train') for c in CAPS for s in SEEDS}
    need(len(queue['jobs'])==len(wanted),'complete_job_count')
    bindings=pins.json(bindings_path);pins.json(protocol_path);matrix={}
    need(safe(queue['bindings'])==safe(bindings_path),'stage_actual_bindings')
    need(queue['source_sha256'].get(str(safe(bindings_path)))==pins.values[str(safe(bindings_path))] and queue['source_sha256'].get(str(safe(protocol_path)))==pins.values[str(safe(protocol_path))],'stage_actual_meta_SHA')
    need(queue['input_sha256']==bindings['input_sha256'],'stage_exact_input_set')
    for job in queue['jobs']:
        role=eval_role(job);need(role in wanted and role not in matrix,'complete_named_matrix')
        cap,seed,pool,cohort=role
        need(safe(argv_value(job['command'],'--out'))==safe(job['job_out']),'job_actual_out')
        need(safe(argv_value(job['command'],'--bindings'))==safe(bindings_path),'job_actual_bindings')
        need(safe(argv_value(job['command'],'--protocol'))==safe(protocol_path),'job_actual_protocol')
        info=read_evaluation(pins,job['job_out'],cap,seed,pool,cohort,bindings_path,protocol_path,kind=='mechanism')
        summary,protocol,rows=info
        scriptindex=next(i for i,x in enumerate(job['command']) if Path(str(x)).name=='evaluate_cap_ranker.py')
        need(protocol['argv']==job['command'][scriptindex:],'actual_eval_command')
        rec=next(x for x in envelope['completion']['jobs'] if x['id']==job['id'])
        need(summary['pid']==rec['pid'] and summary['ppid']==envelope['start']['pid'],'eval_receipt_PID')
        need(parse_utc(rec['started_utc'])<=parse_utc(summary['started_utc'])<=parse_utc(summary['finished_utc'])<=parse_utc(rec['finished_utc']),'eval_receipt_UTC')
        expectedcache=bindings['caches'][str(seed)][cohort][pool]
        need(summary['cache_sha256']==expectedcache['sha256'],'cache_binding_SHA')
        prodref=expectedcache['producer'];actual_producer=pins.json(prodref['path'],prodref['sha256'])
        need(summary['producer_manifest']==actual_producer,'actual_producer_manifest')
        verify_cache_manifest(summary['cache_manifest'],expectedcache['manifest'],actual_producer,expectedcache['sha256'])
        need(safe(protocol['args']['cache'])==safe(expectedcache['path']),'cache_binding_actual_path')
        need(summary['source_sha256']==bindings['source_sha256'],'complete_eval_source_map')
        rolemodels=bindings['roles'][str(cap)][str(seed)]
        checkpointmap={v['checkpoint']:v['checkpoint_sha256'] for v in rolemodels.values()}
        need(summary['checkpoints']==checkpointmap,'exact_role_checkpoints')
        required={str(safe(bindings_path)):pins.values[str(safe(bindings_path))],
          str(safe(protocol_path)):pins.values[str(safe(protocol_path))],
          str(safe(expectedcache['path'])):expectedcache['sha256'],
          **bindings['source_sha256'],bindings['cap_module']['path']:bindings['cap_module']['sha256']}
        for r in rolemodels.values():
            required.update({r['checkpoint']:r['checkpoint_sha256'],r['centers']:r['centers_sha256'],
              r['center_rows']:r['center_rows_sha256'],r['producer_summary']:r['producer_summary_sha256']})
        need(all(summary['input_sha256'].get(p)==h for p,h in required.items()),'eval_complete_input_closure')
        for path,digest in summary['input_sha256'].items():pins.add(path,digest)
        if kind=='validation':
            need(len(rows)==expectedcache['manifest']['records'],'complete_variant_rows')
            need(sum(r['query_identity']['variant']=='observed' for r in rows)==COHORT_OBSERVED[cohort],'observed_cohort_identity_count')
            need(set(r['query_identity']['variant'] for r in rows)==set(VARIANTS),'three_variants')
        matrix[role]=info
    need(set(matrix)==wanted,'complete_matrix')
    for cohort in (('original_val','tune') if kind=='validation' else ('train',)):
        reference=None
        for cap,seed,pool,c in sorted(matrix):
            if c!=cohort:continue
            keys={identity(r) for r in matrix[(cap,seed,pool,c)][2]}
            if kind=='validation':
                need(reference is None or keys==reference,'crosscap_seed_pool_exact_identities');reference=keys
    if kind=='validation':
        for seed,pool,cohort in itertools.product(SEEDS,POOLS,('original_val','tune')):
            a={identity(r):r for r in matrix[(5,seed,pool,cohort)][2]};b={identity(r):r for r in matrix[(10,seed,pool,cohort)][2]}
            for key in a:
                for field in ('target_in_pool','target_candidate_index','cached_outside_pool_rank','gamma','mask_content_sha256','natural_candidate_identity_sha256','natural_coarse_cost_sha256'):
                    need(a[key][field]==b[key][field],'crosscap_same_cache:'+field)
    return matrix,envelope

def aggregate(matrix):
    signals_all={};seed_signals={};table=[];cost=[]
    for cap,pool in itertools.product(CAPS,POOLS):
        config=f'cap{cap}_{pool}';merged={}
        for seed in SEEDS:
            merged[seed]=matrix[(cap,seed,pool,'original_val')][2]+matrix[(cap,seed,pool,'tune')][2]
            need(len({identity(r) for r in merged[seed]})==len(merged[seed]),'merged_cohort_identity_unique')
        signal,parts=signals(merged);signals_all[config]=signal;seed_signals[config]=parts
        for cohort in ('original_val','tune','merged_validation'):
            for variant,source in itertools.product(VARIANTS,('all',)+SOURCES):
                per={}
                for seed in SEEDS:
                    rr=merged[seed] if cohort=='merged_validation' else matrix[(cap,seed,pool,cohort)][2]
                    rr=[r for r in rr if r['query_identity']['variant']==variant and (source=='all' or r['query_identity']['source']==source)]
                    value=metrics(rr);per[seed]=value
                    table.append(dict(config=config,cap=cap,pool=pool,seed=seed,cohort=cohort,variant=variant,source=source,**value))
                mean={}
                need(per[193]['queries']==per[211]['queries'],'stratum_crossseed_queries')
                mean['queries']=per[193]['queries']
                for key in METRICS:
                    v=[per[s][key] for s in SEEDS]
                    mean[key]=math.fsum(v)/2 if all(x is not None for x in v) else None
                table.append(dict(config=config,cap=cap,pool=pool,seed='two_seed_mean',cohort=cohort,variant=variant,source=source,**mean))
        for seed,cohort in itertools.product(SEEDS,('original_val','tune')):
            s=matrix[(cap,seed,pool,cohort)][0]
            cost.append(dict(config=config,seed=seed,cohort=cohort,records=s['records'],
                reranking_seconds_mean=s['reranking_seconds_mean'],reranking_seconds_p95=s['reranking_seconds_p95'],
                elapsed_seconds=s['elapsed_seconds'],cost_semantics=s['cost_semantics'],
                source_summary=matrix[(cap,seed,pool,cohort)][1]['args']['out']+'/summary.json'))
    need(len(table)==648,'complete648_table')
    return signals_all,seed_signals,table,cost

def training_audit(pins,path,bindings,protocol_path):
    ref=bindings['training_audit'];actualpath=str(safe(path));need(actualpath==str(safe(ref['path'])),'registered_training_audit_path')
    need(bindings['input_sha256'].get(actualpath)==ref['sha256'],'registered_training_audit_SHA')
    audit=pins.json(path,ref['sha256']);need(pins.raw(Path(path).parent/'EXIT_CODE').decode().strip()=='0','training_audit_EXIT0')
    impl=ref['implementation_review'];implpath=str(safe(impl['path']))
    need(bindings['input_sha256'].get(implpath)==impl['sha256'],'registered_audit_implementation_review_SHA')
    review=pins.json(implpath,impl['sha256'])
    need(review['status']=='PASS' and review['independent'] is True and bool(review.get('reviewer')),'independent_audit_implementation_review')
    need(review['scope']==impl['required_scope'] and 'implementation' in review['scope'],'audit_implementation_scope')
    need(bool(audit['source_sha256']) and all(review['reviewed_sha256'].get(p)==h for p,h in audit['source_sha256'].items()),'audit_implementation_complete_source_review')
    need(audit['negative_mask_policy']=='external_heldout_OR_ranker_internal_OR_same_query_group; target_loss_only','audit_strict_mask_policy')
    need(audit['schema']=='cap10_training_audit_20261024' and audit['status']=='PASS' and audit['independent'] is True and audit['inputs_unchanged'] is True,'complete8_training_audit')
    need(set(audit['checks'])=={'same_initial_tensors','frozen_observation','frozen_centers','frozen_scale','matched_exposure','matched_pool_exposure','matched_negative_exposure','strict_negative_qualification','finite','full_budget','real_driver_exit','source_input_unchanged'} and all(v is True for v in audit['checks'].values()),'all_training_checks_PASS')
    need(audit['protocol_sha256']==pins.values[str(safe(protocol_path))],'audit_protocol_binding')
    need(safe(audit['training_run'])==safe(bindings['training_run']),'audit_actual_successful_training_run')
    for name,field in [('training_bindings','bindings_sha256'),('training_queue','queue_sha256')]:
        ref=bindings[name];pins.add(ref['path'],ref['sha256']);need(audit[field]==ref['sha256'],'audit_'+name+'_binding')
    trainq,trainenv=driver_receipts(pins,bindings['training_queue']['path'],bindings['training_run'])
    need(trainq['phase']=='train' and len(trainq['jobs'])==8,'actual_successful8fit_driver')
    run=safe(bindings['training_run'])
    evidence=[run/x for x in ('driver_start.json','completion.json','launch_receipt.json','EXIT_CODE','EXTERNAL_EXIT_CODE','input_fingerprints.before.json','input_fingerprints.after.json','operator_fingerprints.before.json','operator_fingerprints.after.json')]
    evidence += [run/'jobs'/(j['id']+'.receipt.json') for j in trainq['jobs']]+[Path(j['access_log']) for j in trainq['jobs']]
    need(all(audit['input_sha256'].get(str(safe(p)))==pins.values[str(safe(p))] for p in evidence),'audit_actual_run_receipt_closure')
    models=audit['models'];need(len(models)==8,'audit8models')
    got=set()
    for m in models:
        cap,seed,role=m['cap'],m['seed'],m['role'];key=(cap,seed,role)
        need(key not in got,'audit_duplicate_model');got.add(key)
        ref=bindings['roles'][str(cap)][str(seed)][role]
        need(m['checkpoint_sha256']==ref['checkpoint_sha256'] and m['centers_sha256']==ref['centers_sha256'] and m['center_rows_sha256']==ref['center_rows_sha256'] and safe(m['checkpoint'])==safe(ref['checkpoint']),'audited_weights_centers')
        need(m['epochs']==64 and type(m['alpha']) in (int,float) and m['alpha']==1.0 and m['draws']==524288 and m['optimizer_steps']==16384,'audit_full_budget')
    need(got==set(itertools.product(CAPS,SEEDS,('no_aug','aug'))),'audit_complete_roles')
    trained=set()
    for job in trainq['jobs']:
        command=job['command'];need(Path(command[0]).name=='train_cap_ranker.py','registered_training_producer')
        key=(int(argv_value(command,'--cap')),int(argv_value(command,'--seed')),argv_value(command,'--role'))
        need(key in got and key not in trained,'actual8fit_named_roles');trained.add(key)
        ref=bindings['roles'][str(key[0])][str(key[1])][key[2]]
        need(safe(ref['checkpoint'])==safe(Path(job['job_out'])/'final.pt'),'actual_fit_final_path')
    byrole={(m['cap'],m['seed'],m['role']):m for m in models}
    for seed,role in itertools.product(SEEDS,('no_aug','aug')):
        a,b=byrole[(5,seed,role)],byrole[(10,seed,role)]
        for key in ('parent_checkpoint_sha256','strict_negative_qualification_sha256','exposure_sha256','pool_exposure_sha256','negative_exposure_sha256','centers_sha256','center_rows_sha256'):
            need(a[key]==b[key] and type(a[key]) is str and len(a[key])==64,'audit_paired_'+key)
    need(bool(audit['source_sha256']),'audit_source_nonempty')
    for path,digest in audit['source_sha256'].items():
        need(audit['input_sha256'].get(path)==digest,'audit_source_closed');pins.add(path,digest)
    for path,digest in audit['input_sha256'].items():
        if Path(path).suffix=='.pt':pins.defer(path,digest)
        else:pins.add(path,digest)
    return audit

def run(args,out):
    pins=Pins();pins.add(__file__);pins.add(Path(__file__).with_name('cap_evaluation_common.py'));protocol=validate_protocol(pins.json(args.protocol));bindings=pins.json(args.bindings)
    need(bindings['schema']=='cap10_execution_bindings_20261024','bindings_schema')
    need(protocol['validation']['baseline_signals']==BASELINE_SIGNALS,'registered_baseline_protocol')
    baseline=bindings['baseline_reference'];reference=pins.json(baseline['path'],baseline['sha256'])
    need(reference['config']==BASELINE and reference['test_read'] is False,'baseline_reference_scope')
    baseline_signals={k:reference['signals'][k]['value'] for k in BASELINE_SIGNALS}
    need(baseline_signals==BASELINE_SIGNALS,'exact_fixed_single_reference')
    matrix,envelope=load_matrix(pins,args.queue,args.run,args.bindings,args.protocol,'validation')
    audit=training_audit(pins,args.training_audit,bindings,args.protocol)
    allsignals,perseed,table,cost=aggregate(matrix)
    selection=select(allsignals['cap10_dual'],allsignals['cap5_dual'],baseline_signals)
    pins.add(__file__);pins.add(Path(__file__).with_name('cap_evaluation_common.py'))
    with (out/'metrics.tsv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t');w.writeheader();w.writerows(table)
    pins.finish()
    result=dict(schema=SCHEMA,status='PASS',configs=['cap5_single','cap5_dual','cap10_single','cap10_dual'],
      source_sha256={str(Path(__file__).resolve()):pins.values[str(Path(__file__).resolve())]},
      protocol={'path':str(safe(args.protocol)),'sha256':pins.values[str(safe(args.protocol))]},
      bindings={'path':str(safe(args.bindings)),'sha256':pins.values[str(safe(args.bindings))]},
      training_audit={'path':str(safe(args.training_audit)),'sha256':pins.values[str(safe(args.training_audit))]},
      validation_queue={'path':str(safe(args.queue)),'sha256':pins.values[str(safe(args.queue))]},
      validation_run=str(safe(args.run)),baseline_reference=baseline,signals=allsignals,per_seed_signals=perseed,
      selection=selection,preselected=selection['preselected'],table_rows=648,table_sha256=sha(out/'metrics.tsv'),
      cost_records=cost,cost_selection_role='raw timing provenance only; no timing-based alternative selection',
      input_sha256=pins.values,deferred_tensor_sha256=pins.deferred,inputs_unchanged=True,scope='complete_prespecified_validation_only',
      selection_requires_independent_result_review=True,independent_result_review='PENDING',
      test_read=False,training=False,forward_bank_calls=0,external_api_calls=0,
      official_system_changed=False,promotion=False,independent_acceptance_status='unavailable')
    write(out/'summary.json',result);return result

def main():
    ap=argparse.ArgumentParser()
    for k in ('protocol','bindings','training-audit','queue','run','out'):ap.add_argument('--'+k,required=True)
    args=ap.parse_args();out=safe(args.out);out.mkdir(parents=True,exist_ok=False)
    try:run(args,out);(out/'EXIT_CODE').write_text('0\n')
    except BaseException as exc:write(out/'failure.json',{'error':repr(exc),'utc':utc()});(out/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()

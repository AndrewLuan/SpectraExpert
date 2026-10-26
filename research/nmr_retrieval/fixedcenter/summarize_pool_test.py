"""功能概述：核验封存测试真实执行证据，并按预先冻结配置导出消融结果。

verify_run检查builder输出SHA、queue/source、driver内外wait和逐任务双EXIT；read_test
绑定实际producer的argv、checkpoint、cache、query/mask摘要及1000/200/171固定身份。
aggregate报告逐seed、变体、来源及两seed指标均值，不调用验证选型函数，不从测试排名
选择模型；完整冻结清单全部完成后才输出结果。main独占新目录，保存输入始末SHA、
结果表和真实内部退出码。依赖同目录summarize_pool_validation的哈希及身份契约；
只读已生成test产物，不训练、不评估、不修改freeze，独立验收unavailable且不晋升。
"""
import argparse
import csv
import json
import math
from pathlib import Path

import summarize_pool_validation as validation

SCHEMA='pool_sealed_ablation_results_20261023'
VARIANTS={'observed':1000,'frequency_crop':200,'peak_dropout':171}


def need(condition,message):
    if not condition:raise ValueError(message)


def double_exit(record):
    return all(type(record.get(k)) is int and record[k]==0 for k in ('external_exit','internal_exit'))


def fixed_identity(value):
    fields={'query_sha256','identity_content_sha256','variant_mask_content_sha256'}
    return (isinstance(value,dict) and set(value)==fields
            and all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in value.values()))


def verify_run(frozen_path,queue_path,run,pins,verified):
    frozen_path=Path(frozen_path).resolve();queue_path=Path(queue_path).resolve();run=Path(run).resolve()
    frozen=validation.read_json(frozen_path,pins)
    need(frozen['schema']=='nmr_pool_ranker_20261023' and frozen['selection_scope']=='train_val_only'
         and frozen['promotion'] is False and frozen['verify_eligibility']=='unavailable','invalid_frozen_ablation')
    need(frozen['candidate_count']==114982 and frozen['candidate_order_sha256']==validation.CANDIDATE_SHA
         and frozen['test_variants']==VARIANTS,'fixed_test_contract_changed')
    need(frozen.get('preselected') is None or frozen['preselected'] in validation.CONFIGS,'unregistered_preselected')
    need(fixed_identity(frozen.get('fixed_test_identity')),'sealed_fixed_identity_required')
    validation.exit0(frozen_path.parent/'EXIT_CODE',pins)
    receipt=validation.read_json(frozen_path.parent/'build_receipt.json',pins)
    need(receipt['status']=='BUILT_NOT_EXECUTED' and receipt['promotion'] is False,'freeze_builder_incomplete')
    validation.require_sha(receipt['output_sha256'],verified)
    need(receipt['output_sha256'].get(str(frozen_path))==validation.sha(frozen_path)
         and receipt['output_sha256'].get(str(queue_path))==validation.sha(queue_path),'freeze_output_binding')
    validation.require_sha(frozen['reviewed_inputs_sha256'],verified)
    validation.require_sha(frozen['evaluation_inputs_sha256'],verified)
    approval=frozen['freeze_review'];approval_path=str(Path(approval['path']).resolve())
    need(approval_path==approval['path'],'sealed_noncanonical_review_path')
    validation.require_sha({approval_path:approval['sha256']},verified)
    review=validation.read_json(approval_path,pins)
    need(review.get('status')=='PASS' and review.get('independent') is True
         and review.get('scope')=='freeze_plan_and_implementation' and bool(review.get('reviewer')),
         'sealed_independent_review_required')
    need(all(review['reviewed_sha256'].get(p)==v for p,v in frozen['reviewed_inputs_sha256'].items()),
         'sealed_independent_review_binding')
    source=str(Path(__file__).resolve())
    need(frozen['reviewed_inputs_sha256'].get(source)==validation.sha(source),'test_summary_tool_not_frozen')
    queue=validation.read_json(queue_path,pins)
    need(queue['schema']=='nmr_research_jobs_20261022','invalid_sealed_queue')
    validation.require_sha(queue['source_inputs_sha256'],verified)
    validation.exit0(run/'EXIT_CODE',pins);validation.exit0(run/'EXTERNAL_EXIT_CODE',pins)
    launch=validation.read_json(run/'launch_receipt.json',pins)
    need(double_exit(launch) and bool(launch.get('finished_utc')),'sealed_outer_not_complete')
    need(launch['queue_sha256']=={str(queue_path):validation.sha(queue_path)}
         and launch['bundle_sha256']==queue['source_inputs_sha256'],'sealed_launch_binding')
    need(bool(launch['source_sha256']) and all(queue['source_inputs_sha256'].get(p)==v
         for p,v in launch['source_sha256'].items()),'sealed_launch_source_binding')
    validation.require_sha(launch['source_sha256'],verified)
    start=validation.read_json(run/'driver_start.json',pins)
    completion=validation.read_json(run/'completion.json',pins)
    need(completion['success'] is True and completion['inputs_unchanged'] is True,'sealed_driver_incomplete')
    need(start['pid']==launch['child_pid'] and start['argv'][1:]==launch['argv'][3:],'sealed_driver_identity')
    validation.require_sha(start['inputs_sha256'],verified)
    need(all(start['inputs_sha256'].get(p)==v for p,v in {**queue['source_inputs_sha256'],**launch['queue_sha256']}.items()),
         'sealed_start_source_mismatch')
    times=[validation.utc(x) for x in (launch['started_utc'],start['utc'],completion['utc'],launch['finished_utc'])]
    need(times==sorted(times) and times[-1]<=validation.datetime.now(validation.timezone.utc),'sealed_time_order')
    jobs=queue['jobs'];configs=set(frozen['configurations'])
    required=set(validation.CONFIGS)|{'frozen_single'}
    need(configs in (required,required|{'frozen_dual'}),'sealed_configuration_matrix')
    named=frozen['ranker_configurations']
    need(set(named)==configs and all(set(seeds)=={str(s) for s in validation.SEEDS} for seeds in named.values()),
         'sealed_named_ranker_matrix')
    mapped=[named[c][str(s)] for c in sorted(configs) for s in validation.SEEDS]
    need(len(frozen['allowed_ranker_configs'])==len(mapped)
         and sorted(map(validation.content_sha,mapped))==sorted(map(validation.content_sha,frozen['allowed_ranker_configs']))
         and len(set(map(validation.content_sha,mapped)))==len(mapped),'sealed_named_allowlist_mismatch')
    need(len(jobs)==2*len(configs) and len({j['id'] for j in jobs})==len(jobs)
         and len({j['out'] for j in jobs})==len(jobs),'sealed_job_matrix')
    need(set(completion['jobs'])=={j['id'] for j in jobs},'sealed_completion_job_matrix')
    seen=set()
    for job in jobs:
        need(job['kind']=='sealed' and not job.get('deps'),'nonsealed_job_forbidden')
        record=validation.read_json(run/(job['id']+'.receipt.json'),pins)
        need(record==completion['jobs'][job['id']] and record['state']=='complete' and double_exit(record),
             'sealed_job_not_double_exit0')
        need(record['argv']==job['argv'] and record['output']==job['out'] and record['cwd']==job['cwd'],
             'sealed_job_argv_mismatch')
        need(record['requested_inputs_sha256']==job['inputs_sha256'],'sealed_job_input_binding')
        need(all(record['queue_source_sha256'].get(p)==v for p,v in {**queue['source_inputs_sha256'],**launch['queue_sha256']}.items()),
             'sealed_job_source_binding')
        begin=validation.utc(record['started_utc']);end=validation.utc(record['finished_utc'])
        need(times[1]<=begin<=end<=times[2],'sealed_job_time_order')
        validation.require_sha(job['inputs_sha256'],verified);validation.exit0(Path(job['out'])/'EXIT_CODE',pins)
        folder=Path(job['out']);seed=int(folder.parent.name.removeprefix('s'));config=folder.parent.parent.name
        need((config,seed) in {(c,s) for c in configs for s in validation.SEEDS}
             and (config,seed) not in seen and folder.name=='test','sealed_role_matrix')
        need(job['id']==f'sealed_{config}_s{seed}','sealed_role_job_binding')
        seen.add((config,seed))
    return frozen,jobs


def read_test(job,frozen,pins,verified,frozen_path):
    folder=Path(job['out']).resolve();protocol=validation.read_json(folder/'protocol.json',pins);args=protocol['args']
    config=folder.parent.parent.name;seed=int(folder.parent.name.removeprefix('s'));pool=config.rsplit('_',1)[1]
    argv=job['argv'];names=('stage','cache','no_aug_checkpoint','aug_checkpoint','seed','out','frozen_selection')
    need(argv[1:3]==['-m','nmr_transfer.h_pool_ranker'] and len(argv)==3+2*len(names),'sealed_evaluation_command')
    options={}
    for key,value in zip(argv[3::2],argv[4::2]):
        need(key not in options,'sealed_duplicate_option');options[key]=value
    need(set(options)=={'--'+n.replace('_','-') for n in names},'sealed_unknown_option')
    need(all(options['--'+n.replace('_','-')]==str(args[n]) for n in names),'sealed_protocol_argv_mismatch')
    need(args['stage']=='evaluate' and args['seed']==seed and Path(args['out']).resolve()==folder,'sealed_protocol_role')
    need(args['frozen_selection']==str(Path(frozen_path).resolve()),'sealed_frozen_path_mismatch')
    allow=dict(no_aug_checkpoint=args['no_aug_checkpoint'],aug_checkpoint=args['aug_checkpoint'],seed=seed,
               cohort='test',recall=pool,cache_sha256=protocol['inputs_sha256'].get(args['cache']))
    need(allow in frozen['allowed_ranker_configs'] and allow==frozen['ranker_configurations'][config][str(seed)],
         'sealed_ranker_not_allowlisted_for_config')
    validation.require_sha(protocol['inputs_sha256'],verified)
    need(all(protocol['inputs_sha256'].get(p)==h for p,h in job['inputs_sha256'].items()),'sealed_protocol_input_mismatch')
    summary=validation.read_json(folder/'summary.json',pins)
    cache=validation.read_json(Path(args['cache']).parent/'summary.json',pins)
    need(summary['input_sha256']==protocol['inputs_sha256'],'sealed_summary_input_binding')
    need(summary['checkpoints']=={args[n]:protocol['inputs_sha256'][args[n]] for n in ('no_aug_checkpoint','aug_checkpoint')},
         'sealed_summary_checkpoints')
    need(summary['cohort']=='test' and summary['recall']==pool and cache['cohort']=='test' and cache['recall']==pool,
         'sealed_test_role')
    need(cache['candidate_count']==114982 and cache['pool_size']==512 and cache['variants']==VARIANTS
         and cache['base_queries']==1000 and cache['records']==1371 and cache['appended_training_positives']==0,
         'sealed_test_size_changed')
    need(cache['feature_cache_sha256']==protocol['inputs_sha256'][args['cache']],'sealed_cache_sha_binding')
    need(summary['candidate_order_sha256']==cache['candidate_order_sha256']==validation.CANDIDATE_SHA,
         'sealed_candidate_library_changed')
    need(summary['query_sha256']==cache['query_sha256'],'sealed_query_binding')
    need(fixed_identity(frozen.get('fixed_test_identity')),'sealed_fixed_identity_required')
    need(all(cache[k]==v for k,v in frozen['fixed_test_identity'].items()),'sealed_fixed_identity_changed')
    rows=[json.loads(line) for line in validation.read_pinned(folder/'predictions.jsonl',pins).splitlines() if line.strip()]
    validation.validate_rows_identity(rows,cache)
    need(len({validation.identity(r) for r in rows})==1371,'sealed_duplicate_query')
    for row in rows:
        need(type(row['rank']) is int and 1<=row['rank']<=114982,'sealed_invalid_rank')
        need(type(row['target_in_coarse_pool']) is bool and row['evaluation_pool_size']==512
             and (row['rank']<=512)==row['target_in_coarse_pool'],'sealed_natural_pool_membership')
        indices=row['natural_candidate_indices'];costs=row['natural_candidate_costs']
        need(len(indices)==512 and len(set(indices))==512 and all(type(i) is int and 0<=i<114982 for i in indices)
             and len(costs)==512 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in costs),'sealed_invalid_pool')
    normal=[r for r in rows if r['variant']=='observed']
    computed=dict(overall=validation.metrics(normal),by_source={s:validation.metrics([r for r in normal if r['source']==s])
        for s in sorted(validation.SOURCES)},stress={v:validation.metrics([r for r in rows if r['variant']==v]) for v in list(VARIANTS)[1:]})
    need(validation.close_metrics(summary['metrics']['overall'],computed['overall']),'sealed_summary_metric_mismatch')
    for key in ('by_source','stress'):
        need(summary['metrics'][key].keys()==computed[key].keys(),'sealed_summary_strata')
        need(all(validation.close_metrics(summary['metrics'][key][n],m) for n,m in computed[key].items()),'sealed_summary_metric_mismatch')
    cost=summary['reranking_seconds_mean']+cache['matching_seconds_mean']
    need(math.isfinite(cost) and cost>=0,'sealed_invalid_cost')
    return config,seed,rows,cost


def aggregate(data):
    table=[]
    for config,seeds in data.items():
        need(set(seeds)==set(validation.SEEDS),'sealed_missing_seed')
        for variant in VARIANTS:
            for source in ['all']+sorted(validation.SOURCES):
                pair=[]
                for seed in validation.SEEDS:
                    rows=[r for r in seeds[seed] if r['variant']==variant and (source=='all' or r['source']==source)]
                    result=validation.metrics(rows);pair.append(result)
                    table.append(dict(config=config,seed=str(seed),variant=variant,source=source,**result))
                need(pair[0]['queries']==pair[1]['queries'],'sealed_seed_stratum_mismatch')
                mean={k:(sum(p[k] for p in pair)/2 if all(p[k] is not None for p in pair) else None)
                      for k in pair[0]}
                mean['queries']=pair[0]['queries']
                table.append(dict(config=config,seed='two_seed_mean',variant=variant,source=source,**mean))
    return table


def run(args,out):
    pins={};verified={};validation.read_pinned(__file__,pins);validation.read_pinned(validation.__file__,pins)
    frozen,jobs=verify_run(args.frozen,args.queue,args.run,pins,verified);data={};costs={};reference=None
    for job in jobs:
        config,seed,rows,cost=read_test(job,frozen,pins,verified,args.frozen)
        signature={validation.identity(r):r['mask_content_sha256'] for r in rows}
        if reference is None:reference=signature
        need(signature==reference,'sealed_cross_config_seed_identity_changed')
        data.setdefault(config,{})[seed]=rows;costs.setdefault(config,{})[seed]=cost
    table=aggregate(data)
    fields=['config','seed','variant','source','queries','top1','top5','top10','MRR','recall512','in_pool_MRR']
    with (out/'metrics.tsv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(table)
    validation.require_sha(pins,{});validation.require_sha(verified,{})
    result=dict(schema=SCHEMA,scope='sealed_test_ablation_only',preselected=frozen['preselected'],
        selection_performed=False,promotion=False,independent_acceptance_status='unavailable',
        input_sha256=pins,scientific_inputs_sha256=verified,metric_rows=len(table),cost_seconds_by_seed=costs,
        frozen_sha256=validation.sha(args.frozen),external_api_calls=0)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('frozen','queue','run','out'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        result=run(args,out);(out/'EXIT_CODE').write_text('0\n');print(json.dumps(result,default=str))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps({'error':repr(exc)})+'\n');(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

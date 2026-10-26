"""功能概述：汇总候选池研究的完整验证矩阵，只用original_val与tune选型。
verify_run要求固定driver内外真实EXIT0、完整逐任务receipt和输入指纹；read_eval校验
查询身份/观测mask/seed/池及summary数值，不消费test。aggregate生成逐seed、cohort、
variant和source指标，choose按正常MRR与四项守门条件选择唯一预选配置，推理成本用于
1e-12并列。main独占输出summary、metrics.tsv和EXIT_CODE；失败不产生可晋升选择。
新增实验必须有另行预注册和工具审核，本版本只接受最初8配置及冻结双池基准。
bind_evaluation和validate_rows_identity把driver、实际模型路径、完整query/mask绑定到
producer；historical_incumbent只读旧封存验证证明并单列旧主模型，缺失召回证据保持空值。
read_pinned防止同次读入内容漂移，run始末重新核验所有科学输入与工具自身SHA。
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime,timezone
from pathlib import Path

SEEDS=(193,211)
COHORTS=('original_val','tune')
VARIANTS=('observed','frequency_crop','peak_dropout')
FAMILIES=tuple(f'{pool}_{loss}' for pool in ('original','mixed') for loss in ('pairwise16','listwise'))
CONFIGS=tuple(f'{family}_{pool}' for family in FAMILIES for pool in ('single','dual'))
SCHEMA='pool_validation_selection_20261023'
SOURCES={'HMDB','NMRexp','WanJuan','nmrshiftdb2_2018','nmrshiftdb2_2024'}
CANDIDATE_SHA='72b72140c09ae14ae8d0aa4edad624a151093c9df246b2052de8a3a4ea616bad'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_json(path,pins):
    return json.loads(read_pinned(path,pins))


def read_pinned(path,pins):
    path=Path(path).resolve();raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if str(path) in pins and pins[str(path)]!=digest:raise ValueError('input_changed_between_reads:'+str(path))
    pins[str(path)]=digest
    return raw


def exit0(path,pins):
    path=Path(path).resolve()
    if read_pinned(path,pins).strip()!=b'0':raise ValueError('nonzero_or_incomplete_exit:'+str(path))


def require_sha(expected,verified):
    for path,digest in expected.items():
        if path in verified:
            if verified[path]!=digest:raise ValueError('conflicting_input_sha:'+path)
        elif sha(path)!=digest:raise ValueError('input_sha_changed:'+path)
        else:verified[path]=digest


def utc(value):
    try:
        stamp=datetime.fromisoformat(value.replace('Z','+00:00'))
        if stamp.tzinfo is None:raise ValueError('timezone_required')
        return stamp.astimezone(timezone.utc)
    except (AttributeError,TypeError,ValueError) as exc:
        raise ValueError('invalid_completion_utc') from exc


def verify_run(run,phase,pins,verified):
    run=Path(run)
    exit0(run/'EXIT_CODE',pins);exit0(run/'EXTERNAL_EXIT_CODE',pins)
    launch=read_json(run/'launch_receipt.json',pins)
    if any(type(launch.get(k)) is not int or launch[k]!=0 for k in ('external_exit','internal_exit')) or not launch.get('finished_utc'):
        raise ValueError('driver_outer_not_complete')
    launch_start=utc(launch['started_utc']);launch_end=utc(launch['finished_utc'])
    if not launch_start<=launch_end<=datetime.now(timezone.utc):raise ValueError('invalid_driver_time_order')
    require_sha(launch['queue_sha256'],verified)
    if len(launch['queue_sha256'])!=1:raise ValueError('ambiguous_queue')
    queue_path=next(iter(launch['queue_sha256']))
    queue=read_json(queue_path,pins)
    require_sha(queue['source_inputs_sha256'],verified)
    require_sha(launch['source_sha256'],verified)
    if launch['bundle_sha256']!=queue['source_inputs_sha256']:
        raise ValueError('driver_bundle_mismatch')
    completion=read_json(run/'completion.json',pins)
    if completion.get('success') is not True or completion.get('inputs_unchanged') is not True:
        raise ValueError('driver_not_complete')
    driver=read_json(run/'driver_start.json',pins)
    if driver['pid']!=launch['child_pid'] or driver['argv'][1:]!=launch['argv'][3:]:
        raise ValueError('driver_launch_identity_mismatch')
    require_sha(driver['inputs_sha256'],verified)
    if any(driver['inputs_sha256'].get(p)!=v for p,v in {**queue['source_inputs_sha256'],**launch['queue_sha256']}.items()):
        raise ValueError('driver_start_inputs_mismatch')
    driver_start=utc(driver['utc']);driver_end=utc(completion['utc'])
    if not launch_start<=driver_start<=driver_end<=launch_end:raise ValueError('invalid_driver_time_order')
    jobs=queue['jobs']
    expected=48 if phase=='formal' else 12
    if len(jobs)!=expected or len({j['id'] for j in jobs})!=expected:
        raise ValueError('incomplete_or_duplicate_job_matrix')
    counts=collections.Counter(j['kind'] for j in jobs)
    if counts!=({'training':16,'validation':32} if phase=='formal' else {'validation':8,'audit':4}):
        raise ValueError('incorrect_job_kind_matrix')
    if set(completion['jobs'])!={j['id'] for j in jobs}:raise ValueError('completion_job_set')
    outputs=set()
    for job in jobs:
        receipt=read_json(run/(job['id']+'.receipt.json'),pins)
        if receipt!=completion['jobs'][job['id']]:raise ValueError('job_receipt_completion_disagree')
        if (receipt.get('state')!='complete' or any(type(receipt.get(k)) is not int or receipt[k]!=0
                for k in ('external_exit','internal_exit')) or not receipt.get('finished_utc')):
            raise ValueError('job_not_dual_exit0')
        job_start=utc(receipt['started_utc']);job_end=utc(receipt['finished_utc'])
        if not driver_start<=job_start<=job_end<=driver_end:raise ValueError('invalid_job_time_order')
        for dependency in job.get('deps',[]):
            if utc(completion['jobs'][dependency]['finished_utc'])>job_start:
                raise ValueError('job_started_before_dependency_completed')
        if receipt['argv']!=job['argv'] or receipt['output']!=job['out']:
            raise ValueError('job_argv_or_output_mismatch')
        if receipt['cwd']!=job['cwd'] or any(receipt['queue_source_sha256'].get(p)!=v
                for p,v in {**queue['source_inputs_sha256'],**launch['queue_sha256']}.items()):
            raise ValueError('job_source_or_cwd_mismatch')
        if job['out'] in outputs:raise ValueError('duplicate_job_output')
        outputs.add(job['out'])
        if receipt['requested_inputs_sha256']!=job['inputs_sha256']:
            raise ValueError('job_requested_inputs_changed')
        require_sha(job['inputs_sha256'],verified)
        exit0(Path(job['out'])/'EXIT_CODE',pins)
    return {str(Path(j['out']).resolve()):j for j in jobs}


def metrics(rows):
    if not rows:return {'queries':0}
    known=all(type(r.get('target_in_coarse_pool')) is bool for r in rows)
    recalled=[r for r in rows if r.get('target_in_coarse_pool') is True]
    return dict(queries=len(rows),MRR=sum(1/r['rank'] for r in rows)/len(rows),
        **{f'top{k}':sum(r['rank']<=k for r in rows)/len(rows) for k in (1,5,10)},
        recall512=len(recalled)/len(rows) if known else None,
        in_pool_MRR=sum(1/r['rank'] for r in recalled)/len(recalled) if known and recalled else None)


def identity(row):
    return tuple(row[k] for k in ('query_order','variant','source','record_id','smiles','group'))


def content_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_rows_identity(rows,cache):
    """用producer的完整身份与mask摘要核验每条变体，不读取特征缓存中的谱值。"""
    if rows!=sorted(rows,key=lambda r:(r['query_order'],r['variant'])):
        raise ValueError('query_order_changed')
    fields=('query_order','variant','source','record_id','smiles','group')
    identities=[{k:r[k] for k in fields} for r in rows]
    masks=[(tuple(r[k] for k in fields[:-1]),r['mask_content_sha256']) for r in rows]
    if (len(rows)!=cache['records'] or dict(collections.Counter(r['variant'] for r in rows))!=cache['variants']
            or content_sha(identities)!=cache['identity_content_sha256']
            or content_sha(masks)!=cache['variant_mask_content_sha256']):
        raise ValueError('cache_prediction_identity_or_variant_mismatch')
    if {r['source'] for r in rows}!=SOURCES:
        raise ValueError('requires_registered_five_sources')


def bind_evaluation(args,job,expected):
    """要求实际producer协议、driver argv和固定family路径三者一致。"""
    argv=job['argv']
    if argv[1:3]!=['-m','nmr_transfer.h_pool_ranker'] or len(argv[3:])%2:
        raise ValueError('evaluation_command_mismatch')
    options={}
    for key,value in zip(argv[3::2],argv[4::2]):
        if not key.startswith('--') or key in options:raise ValueError('duplicate_or_invalid_evaluation_arg')
        options[key]=value
    names=('stage','seed','out','cache','no_aug_checkpoint','aug_checkpoint')
    if set(options)!={'--'+n.replace('_','-') for n in names}:
        raise ValueError('unregistered_evaluation_arg')
    for name in names:
        value=str(args[name]);command=options['--'+name.replace('_','-')]
        if value!=command or value!=str(expected[name]):
            raise ValueError('evaluation_family_or_argv_mismatch:'+name)


def evaluation_args(root,config,seed,cohort):
    pool=config.rsplit('_',1)[1];family=config.rsplit('_',1)[0]
    old=root.parent/'nmr_global_retrieval_20261022'
    model=old/'rankers/factor_R0O1F0' if family=='frozen' else root/'formal/rankers'/family
    base=root/'diagnostics/frozen' if family=='frozen' else root/'formal'/family
    return dict(stage='evaluate',seed=seed,out=str(base/f's{seed}'/pool/cohort),
        cache=str(old/f'features/inherited_s{seed}'/pool/cohort/'features.pt'),
        no_aug_checkpoint=str(model/f's{seed}/no_aug/best.pt'),
        aug_checkpoint=str(model/f's{seed}/aug/best.pt'))


def close_metrics(actual,expected):
    if actual.keys()!=expected.keys():return False
    for key,value in expected.items():
        if value is None:
            if actual[key] is not None:return False
        elif not math.isclose(actual[key],value,rel_tol=1e-10,abs_tol=1e-12):return False
    return True


def read_eval(folder,seed,pool,cohort,jobs,pins,verified,expected_args=None):
    folder=Path(folder).resolve()
    if cohort not in COHORTS:raise ValueError('test_or_wrong_cohort_or_pool')
    if str(folder) not in jobs or jobs[str(folder)]['kind']!='validation':
        raise ValueError('unregistered_validation_output')
    protocol=read_json(folder/'protocol.json',pins)
    args=protocol['args']
    if (args['stage']!='evaluate' or args['seed']!=seed
            or Path(args['out']).resolve()!=folder):
        raise ValueError('evaluation_role_mismatch')
    if expected_args is None:raise ValueError('expected_evaluation_identity_required')
    bind_evaluation(args,jobs[str(folder)],expected_args)
    for path,digest in jobs[str(folder)]['inputs_sha256'].items():
        if protocol['inputs_sha256'].get(path)!=digest:raise ValueError('job_protocol_input_mismatch')
    require_sha(protocol['inputs_sha256'],verified)
    summary=read_json(folder/'summary.json',pins)
    if summary['cohort']!=cohort or cohort not in COHORTS or summary['recall']!=pool:
        raise ValueError('test_or_wrong_cohort_or_pool')
    if summary['input_sha256']!=protocol['inputs_sha256']:
        raise ValueError('evaluation_input_summary_mismatch')
    for role in ('no_aug_checkpoint','aug_checkpoint'):
        path=str(Path(args[role]).resolve())
        if summary['checkpoints'].get(path)!=protocol['inputs_sha256'].get(path):
            raise ValueError('checkpoint_summary_identity')
    cache_summary=read_json(Path(args['cache']).parent/'summary.json',pins)
    if cache_summary['feature_cache_sha256']!=protocol['inputs_sha256'].get(str(Path(args['cache']).resolve())):
        raise ValueError('cache_producer_sha_mismatch')
    if (cache_summary['cohort']!=cohort or cache_summary['recall']!=pool
            or cache_summary['candidate_count']!=114982 or cache_summary['pool_size']!=512
            or cache_summary['query_sha256']!=summary['query_sha256']
            or cache_summary['candidate_order_sha256']!=summary['candidate_order_sha256']):
        raise ValueError('cache_eval_identity')
    if summary['candidate_order_sha256']!=CANDIDATE_SHA:raise ValueError('candidate_library_changed')
    path=folder/'predictions.jsonl'
    rows=[json.loads(line) for line in read_pinned(path,pins).splitlines() if line.strip()]
    if len({identity(r) for r in rows})!=len(rows):raise ValueError('duplicate_query')
    for row in rows:
        if row['variant'] not in VARIANTS:raise ValueError('unregistered_variant')
        if type(row['rank']) is not int or not 1<=row['rank']<=114982:
            raise ValueError('invalid_rank')
        if (row['evaluation_pool_size']!=512 or type(row['target_in_coarse_pool']) is not bool
                or (row['rank']<=512)!=row['target_in_coarse_pool']):
            raise ValueError('natural_pool_rank_contract')
        indices=row['natural_candidate_indices']; costs=row['natural_candidate_costs']
        if (len(indices)!=512 or len(set(indices))!=512 or len(costs)!=512
                or any(type(i) is not int or not 0<=i<114982 for i in indices)
                or any(not math.isfinite(x) for x in costs)):
            raise ValueError('invalid_natural_pool')
        if not row['mask_content_sha256']:raise ValueError('missing_observation_mask_identity')
    validate_rows_identity(rows,cache_summary)
    normal=[r for r in rows if r['variant']=='observed']
    if len(normal)!= {'original_val':320,'tune':1024}[cohort]:
        raise ValueError('incomplete_normal_cohort')
    expected=dict(overall=metrics(normal),
        by_source={s:metrics([r for r in normal if r['source']==s]) for s in sorted({r['source'] for r in normal})},
        stress={v:metrics([r for r in rows if r['variant']==v]) for v in VARIANTS[1:]})
    for key in ('overall',):
        if not close_metrics(summary['metrics'][key],expected[key]):raise ValueError('summary_metric_mismatch')
    for key in ('by_source','stress'):
        if summary['metrics'][key].keys()!=expected[key].keys():raise ValueError('summary_stratum_mismatch')
        for name,value in expected[key].items():
            if not close_metrics(summary['metrics'][key][name],value):raise ValueError('summary_metric_mismatch')
    cost=summary['reranking_seconds_mean']+cache_summary['matching_seconds_mean']
    if not math.isfinite(cost) or cost<0:raise ValueError('invalid_inference_cost')
    return rows,dict(cost_seconds=cost,candidate_order_sha256=summary['candidate_order_sha256'],
                     query_sha256=summary['query_sha256'],checkpoints=summary['checkpoints'])


def aggregate(data,metadata):
    table=[];signals={}
    for config,seeds in data.items():
        seed_signals=[]
        for seed,cohorts in seeds.items():
            combined=sum((cohorts[c] for c in COHORTS),[])
            # 两cohort不可重复使用同一条原始查询。variant允许复用同一原始结构。
            keys=[(r['source'],r['record_id'],r['smiles'],r['variant']) for r in combined]
            if len(keys)!=len(set(keys)):raise ValueError('cross_cohort_duplicate_query')
            for cohort,rows in list(cohorts.items())+[('combined',combined)]:
                for variant in VARIANTS:
                    subset=[r for r in rows if r['variant']==variant]
                    for source in ['all']+sorted({r['source'] for r in subset}):
                        part=subset if source=='all' else [r for r in subset if r['source']==source]
                        table.append(dict(config=config,seed=seed,cohort=cohort,variant=variant,
                                          source=source,**metrics(part)))
            normal=[r for r in combined if r['variant']=='observed']
            sources=sorted({r['source'] for r in normal})
            if len(sources)!=5:raise ValueError('requires_five_sources')
            stress={v:metrics([r for r in combined if r['variant']==v]) for v in VARIANTS[1:]}
            if any(not m['queries'] for m in stress.values()):raise ValueError('missing_stress_variant')
            normal_metrics=metrics(normal)
            costs=[metadata[config][seed][c]['cost_seconds'] for c in COHORTS]
            cost=(sum(metadata[config][seed][c]['cost_seconds']*len(cohorts[c]) for c in COHORTS)/len(combined)
                  if all(x is not None for x in costs) else None)
            seed_signals.append(dict(MRR=normal_metrics['MRR'],top1=normal_metrics['top1'],
                frequency_crop_MRR=stress['frequency_crop']['MRR'],
                peak_dropout_MRR=stress['peak_dropout']['MRR'],
                macro_MRR=sum(metrics([r for r in normal if r['source']==s])['MRR'] for s in sources)/5,
                cost_seconds=cost))
        if set(seeds)!=set(SEEDS):raise ValueError('incomplete_seed_family')
        signals[config]={k:sum(x[k] for x in seed_signals)/2 if all(x[k] is not None for x in seed_signals) else None
                         for k in seed_signals[0]}
    return table,signals


def historical_incumbent(root,reference,pins,verified):
    """复用旧轮已封存的legacy内容证明，只输出控制，不进入choose。"""
    old=root.parent/'nmr_global_retrieval_20261022'
    source=old/'build_validation_manifest.py';read_pinned(source,pins)
    spec=importlib.util.spec_from_file_location('legacy_validation_builder',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    snapshot=module.Snapshot()
    baseline=snapshot.json(old/'baseline_inputs.json')
    entries,_,candidate_sha,_=module.legacy_entries(baseline,
        old.parent/'nmr_global_retrieval_20261020/freeze_20261020_2000',datetime.now(timezone.utc),snapshot,
        '42581fcdd7fa342569334949eeb29afe374839bacfcae0638f6170720a7dd9e1',
        '7d6f7ee9d321d3d9a2b7e937426be19253fb412251d6ac7993b2c8ffe8b7f19f',
        'f39108b3264e4677067ce7fa631d45d7338b77f2cf4fe751d9be2ac31e9e98eb')
    if candidate_sha!=CANDIDATE_SHA:raise ValueError('historical_candidate_library_changed')
    data={seed:{} for seed in SEEDS};metadata={seed:{} for seed in SEEDS}
    for entry in entries:
        seed=entry['seed'];cohort=entry['cohort'];rows=[]
        for path in entry['predictions']:rows.extend(snapshot.rows(path,cohort))
        if {identity(r) for r in rows}!={identity(r) for r in reference[cohort]}:
            raise ValueError('historical_query_identity_changed')
        for row in rows:
            if type(row['rank']) is not int or not 1<=row['rank']<=114982:raise ValueError('historical_invalid_rank')
        if cohort in data[seed]:raise ValueError('historical_duplicate_seed_cohort')
        data[seed][cohort]=rows;metadata[seed][cohort]={'cost_seconds':None}
    snapshot.finish();require_sha(snapshot.inputs,verified)
    table,signals=aggregate({'incumbent_frozen':data},{'incumbent_frozen':metadata})
    return table,dict(signal=signals['incumbent_frozen'],selection_eligible=False,
                      evidence_sha256=snapshot.inputs,recall_evidence='unavailable_in_legacy_rows')


def choose(signals):
    if set(signals)!=set(CONFIGS)|{'frozen_single','frozen_dual'}:
        raise ValueError('incomplete_or_unregistered_family')
    baseline=signals['frozen_single'];results={};eligible=[]
    for config in CONFIGS:
        value=signals[config]
        gates=dict(normal_MRR=value['MRR']>baseline['MRR'],
            top1=value['top1']>=baseline['top1'],
            crop=value['frequency_crop_MRR']>=baseline['frequency_crop_MRR'],
            dropout=value['peak_dropout_MRR']>=baseline['peak_dropout_MRR'],
            source_macro=value['macro_MRR']>=baseline['macro_MRR'])
        results[config]=dict(signal=value,gates=gates,eligible=all(gates.values()),
                             delta={k:value[k]-baseline[k] for k in baseline})
        if all(gates.values()):eligible.append(config)
    chosen=None
    if eligible:
        best=max(signals[c]['MRR'] for c in eligible)
        tied=[c for c in eligible if best-signals[c]['MRR']<=1e-12]
        chosen=min(tied,key=lambda c:(signals[c]['cost_seconds'],c))
    return dict(preselected=chosen,baseline='frozen_single',baseline_signal=baseline,
                configurations=results,eligible_configurations=eligible,
                promotion=False,selection_scope='train_val_only')


def run(args,out):
    root=Path(args.root).resolve();pins={};verified={}
    read_pinned(__file__,pins)
    formal=verify_run(args.formal_run or root/'runs/formal','formal',pins,verified)
    diagnostic=verify_run(args.diagnostics_run or root/'runs/diagnostics','diagnostics',pins,verified)
    registration=read_json(root/'preregistered_arms.json',pins)
    if ({f['family'] for f in registration['families']}!=set(CONFIGS)
            or registration['seeds']!=list(SEEDS)):
        raise ValueError('registration_matrix_mismatch')
    data={};metadata={}
    expected_outputs=set()
    for config in list(CONFIGS)+['frozen_single','frozen_dual']:
        pool=config.rsplit('_',1)[1]
        family=config.rsplit('_',1)[0]
        base=root/'diagnostics/frozen' if family=='frozen' else root/'formal'/family
        jobs=diagnostic if family=='frozen' else formal
        data[config]={};metadata[config]={}
        for seed in SEEDS:
            data[config][seed]={};metadata[config][seed]={}
            for cohort in COHORTS:
                folder=base/f's{seed}'/pool/cohort
                rows,meta=read_eval(folder,seed,pool,cohort,jobs,pins,verified,
                                    evaluation_args(root,config,seed,cohort))
                data[config][seed][cohort]=rows;metadata[config][seed][cohort]=meta
                expected_outputs.add(str(folder))
    actual={p for p,j in {**formal,**diagnostic}.items() if j['kind']=='validation'}
    if expected_outputs!=actual:raise ValueError('unexpected_or_missing_validation_job')
    reference=data['frozen_single'][SEEDS[0]]
    for config in data:
        for seed in SEEDS:
            for cohort in COHORTS:
                rows=data[config][seed][cohort]
                signature=lambda rr:{identity(r):r['mask_content_sha256'] for r in rr}
                if signature(rows)!=signature(reference[cohort]):
                    raise ValueError('cross_family_or_seed_query_mask_changed')
                if metadata[config][seed][cohort]['query_sha256']!=metadata['frozen_single'][SEEDS[0]][cohort]['query_sha256']:
                    raise ValueError('cross_family_query_sha_changed')
                if metadata[config][seed][cohort]['candidate_order_sha256']!=metadata['frozen_single'][SEEDS[0]][cohort]['candidate_order_sha256']:
                    raise ValueError('candidate_library_changed')
    table,signals=aggregate(data,metadata)
    selection=choose(signals)
    historical_rows,historical=historical_incumbent(root,reference,pins,verified)
    table+=historical_rows
    fields=['config','seed','cohort','variant','source','queries','top1','top5','top10','MRR','recall512','in_pool_MRR']
    with (out/'metrics.tsv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t')
        writer.writeheader();writer.writerows(table)
    require_sha(pins,{})
    require_sha(verified,{})
    result=dict(schema=SCHEMA,selection=selection,signals=signals,metadata=metadata,
        input_sha256=pins,scientific_inputs_sha256=verified,source_sha256={str(Path(__file__).resolve()):sha(__file__)},
        historical_incumbent=historical,driver_evidence=dict(formal_jobs=48,diagnostic_jobs=12,all_dual_exit0=True),
        metric_rows=len(table),external_api_calls=0,independent_acceptance_status='not_performed')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--formal-run');p.add_argument('--diagnostics-run')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    try:
        result=run(args,out)
        (out/'EXIT_CODE').write_text('0\n')
        print(json.dumps(dict(preselected=result['selection']['preselected'],metric_rows=result['metric_rows'],
                              elapsed_seconds=time.time()-started)))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc))))
        (out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

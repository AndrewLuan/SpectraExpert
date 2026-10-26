"""功能概述：只读审计唯一追加固定train中心家族的4fit+8val真实driver产物。
复用已独审旧audit的精确tensor、完整预算与独立曝光重放；独立核36常量证据、
initial/best/last中心冻结和同seed/role原mixed_listwise控制曝光。只读train缓存身份
及产物/producer元数据，不读验证预测或test；不训练、不改科学源。输出独占且始末SHA。
"""
import argparse
import collections
import copy
import datetime
import json
from pathlib import Path
import torch
import audit_pool_training as old
from audit_pool_training import require,sha,read_json,finite,exact,utc,double_exit

SCHEMA='nmr_pool_fixed_center_20261023'
CENTERS={'fixed_x_center','fixed_o_center'}


def audit_model(summary,protocol,inherited,checkpoints,history,expected,exposure,estimate,center_sha,rows_sha,control):
    require(expected['training_pool']=='mixed' and expected['loss']=='listwise','fixed_family_only')
    require(summary['center_estimate_sha256']==center_sha and summary['center_rows_sha256']==rows_sha
            and estimate['row_audit_sha256']==rows_sha,'center_evidence_SHA')
    require(summary['centers_frozen'] is True and summary['scale_frozen'] is True
            and summary['initial_tensors_inherited_exact'] is True,'fixed_declared_frozen')
    require(estimate['seed']==expected['seed'] and estimate['augment'] is expected['augment']
            and estimate['input_sha256']==protocol['inputs_sha256'],'center_role_inputs')
    require(estimate['constant_fp32']==36 and estimate['raw_constant_bytes']==144
            and estimate['selection_scope']=='fit_train_only','center_estimation_scope')
    require(summary['center_query_rows']==estimate['query_rows'] and estimate['query_rows']>0,'center_query_count')
    target={'fixed_x_center':torch.tensor(estimate['center33'],dtype=torch.float32),
            'fixed_o_center':torch.tensor(estimate['O_center3'],dtype=torch.float32)}
    require(target['fixed_x_center'].shape==(33,) and target['fixed_o_center'].shape==(3,)
            and finite(target),'center_capacity_finite')
    normalized=copy.deepcopy(checkpoints)
    for phase,saved in checkpoints.items():
        require(saved['schema']==SCHEMA and saved['center_estimate_sha256']==center_sha,'fixed_checkpoint_schema_center')
        require(saved['state_dict'].keys()==inherited['state_dict'].keys()|CENTERS,'fixed_state_keys')
        require(exact({k:saved['state_dict'][k] for k in CENTERS},target),'fixed_center_changed:'+phase)
        normalized[phase]['schema']=old.SCHEMA
        for k in CENTERS:del normalized[phase]['state_dict'][k]
    result=old.audit_model(summary,protocol,inherited,normalized,history,expected,exposure)
    require(control['schema']==old.SCHEMA and control['epoch']==64
            and all(control['args'].get(k)==v for k,v in old.BUDGET.items())
            and all(control['args'].get(k)==v for k,v in expected.items()),'control_role_budget')
    require(control['exposure_sha256']==result['exposure_sha256']
            and control['pool_exposure_sha256']==result['pool_exposure_sha256'],'control_complete_exposure')
    require(exact(control['scale'],inherited['scale']),'control_scale')
    return dict(result,center_estimate_sha256=center_sha,center_rows_sha256=rows_sha,constant_fp32=36)


def verify_driver(run, queue_path, queue, pin):
    """核验完成的12任务来源闭环；只读receipt/EXIT，不读validation预测。"""
    for filename in ('EXIT_CODE','EXTERNAL_EXIT_CODE'):
        pin(run/filename)
        require((run/filename).read_text().strip() == '0', 'driver_exit_not_zero')
    for filename in ('launch_receipt.json','driver_start.json','completion.json'):
        pin(run/filename)
    launch = read_json(run/'launch_receipt.json')
    driver = read_json(run/'driver_start.json')
    completion = read_json(run/'completion.json')
    double_exit(launch, 'driver')
    start, end = utc(launch['started_utc']), utc(launch['finished_utc'])
    require(start <= end <= datetime.datetime.now(datetime.timezone.utc), 'invalid_driver_time_order')
    require(launch['queue_sha256'] == {str(queue_path): pin(queue_path)}, 'launch_queue_binding')
    require(launch['bundle_sha256'] == queue['source_inputs_sha256'], 'launch_bundle_binding')
    require(bool(launch['source_sha256']), 'missing_launcher_sources')
    for p,digest in launch['source_sha256'].items():
        require(queue['source_inputs_sha256'].get(p) == digest, 'launcher_source_binding')
        pin(p,digest)
    expected_sources = {**queue['source_inputs_sha256'],**launch['queue_sha256']}
    require(driver['inputs_sha256'] == expected_sources, 'driver_start_source_binding')
    require(type(driver['pid']) is int and driver['pid'] > 0 and driver['pid'] == launch['child_pid']
            and type(launch['child_pid']) is int
            and driver['argv'][1:] == launch['argv'][3:], 'driver_launch_identity')
    driver_start, driver_end = utc(driver['utc']), utc(completion['utc'])
    require(start <= driver_start <= driver_end <= end, 'invalid_driver_time_order')
    require(completion.get('success') is True and completion.get('inputs_unchanged') is True, 'driver_incomplete')
    jobs = queue['jobs']
    require(len(jobs) == 12 and len({j['id'] for j in jobs}) == 12
            and collections.Counter(j['kind'] for j in jobs) == {'training':4,'validation':8}, 'fixedcenter_12_job_matrix')
    require(set(completion['jobs']) == {j['id'] for j in jobs}, 'completion_job_set')
    require(len({j['out'] for j in jobs}) == 12, 'duplicate_job_output')
    for job in jobs:
        path = run/(job['id']+'.receipt.json')
        pin(path)
        receipt = read_json(path)
        require(receipt == completion['jobs'][job['id']], 'receipt_completion_binding')
        double_exit(receipt, 'job')
        require(receipt.get('state') == 'complete', 'job_incomplete')
        job_start, job_end = utc(receipt['started_utc']), utc(receipt['finished_utc'])
        require(driver_start <= job_start <= job_end <= driver_end, 'invalid_job_time_order')
        require(receipt['argv'] == job['argv'] and receipt['output'] == job['out']
                and receipt['cwd'] == job['cwd'], 'receipt_job_binding')
        require(receipt['queue_source_sha256'] == expected_sources, 'receipt_source_binding')
        require(receipt['requested_inputs_sha256'] == job['inputs_sha256'], 'receipt_input_binding')
        for dependency in job.get('deps', []):
            require(utc(completion['jobs'][dependency]['finished_utc']) <= job_start, 'dependency_time_order')
        pin(Path(job['out'])/'EXIT_CODE')
        require((Path(job['out'])/'EXIT_CODE').read_text().strip() == '0', 'job_producer_exit')


def verify_evaluations(queue,results,module,pin):
    fits={tuple(r['role']):r for r in results};seen=set();evidence=[]
    for job in queue['jobs']:
        if job['kind']!='validation':continue
        out=Path(job['out']).resolve()
        for name in ('summary.json','protocol.json','EXIT_CODE'):pin(out/name)
        summary,protocol=read_json(out/'summary.json'),read_json(out/'protocol.json')
        args=protocol['args']
        require(args['stage']=='evaluate' and args['out']==job['out'] and module in job['argv'],'eval_stage_module')
        for key in ('stage','source','cache','no_aug_checkpoint','aug_checkpoint','seed','out'):
            flag='--'+key.replace('_','-')
            require(job['argv'][job['argv'].index(flag)+1]==str(args[key]),'eval_argv_protocol:'+key)
        require(protocol['inputs_sha256'].get(module)==pin(module),'eval_module_pin')
        for path,digest in protocol['inputs_sha256'].items():
            require('/test/' not in path and '/sealed/' not in path,'unexpected_test_input')
            pin(path,digest)
        require(summary['input_sha256']==protocol['inputs_sha256'] and finite(summary),'eval_summary_protocol')
        for path,digest in job['inputs_sha256'].items():pin(path,digest)
        cache=Path(args['cache']).resolve()
        require(protocol['inputs_sha256'].get(str(cache))==pin(cache),'eval_cache_required_protocol_pin')
        for name in ('summary.json','EXIT_CODE'):pin(cache.parent/name)
        producer=read_json(cache.parent/'summary.json')
        require((cache.parent/'EXIT_CODE').read_text().strip()=='0'
                and producer['feature_cache_sha256']==pin(cache),'eval_cache_producer')
        role=(args['seed'],summary['recall'],summary['cohort'])
        require(role in {(seed,pool,cohort) for seed in (193,211) for pool in ('single','dual')
                         for cohort in ('original_val','tune')} and role not in seen,'eval_matrix')
        seen.add(role)
        require(cache.parent.name==role[2] and all(summary[k]==producer[k]
                for k in ('cohort','recall','query_sha256','candidate_order_sha256')),'eval_cache_summary_binding')
        expected_paths={str((Path(fits[(args['seed'],aug)]['output'])/'best.pt').resolve()):
                        pin(Path(fits[(args['seed'],aug)]['output'])/'best.pt') for aug in (False,True)}
        require(args['no_aug_checkpoint']==str((Path(fits[(args['seed'],False)]['output'])/'best.pt').resolve())
                and args['aug_checkpoint']==str((Path(fits[(args['seed'],True)]['output'])/'best.pt').resolve()),
                'eval_checkpoint_roles')
        require(summary['checkpoints']==expected_paths
                and all(protocol['inputs_sha256'].get(path)==digest for path,digest in expected_paths.items()),
                'eval_checkpoint_SHA')
        require(summary['center_estimate_sha256']==[fits[(args['seed'],aug)]['center_estimate_sha256']
                for aug in (False,True)],'eval_center_binding')
        evidence.append(dict(job_id=job['id'],role=role,output=str(out),checkpoints=expected_paths))
    require(len(seen)==8,'incomplete_eval_matrix')
    return evidence


def audit(root,queue_path,run,control_root=None):
    root,queue_path,run=map(lambda p:Path(p).resolve(),(root,queue_path,run))
    control_root=Path(control_root).resolve() if control_root else root/'formal/rankers/mixed_listwise'
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path)
        actual=pins.get(key)
        if actual is None:actual=sha(path);pins[key]=actual
        require(expected is None or expected==actual,'input_pin_mismatch:'+key)
        return actual
    pin(__file__);pin(old.__file__);pin(queue_path)
    queue=read_json(queue_path)
    for path,digest in queue['source_inputs_sha256'].items():pin(path,digest)
    module=[p for p in queue['source_inputs_sha256'] if Path(p).name=='h_pool_fixed_center_ranker.py']
    require(len(module)==1,'fixed_module_source_pin_required')
    verify_driver(run,queue_path,queue,pin)
    results=[];roles=set();cache_exposures={}
    for job in queue['jobs']:
        if job['kind']!='training':continue
        out=Path(job['out']).resolve()
        require(out.is_relative_to(root) and not out.is_relative_to(root/'formal'),'fixed_output_scope')
        for name in ('summary.json','protocol.json','history.jsonl','EXIT_CODE','initial.pt','best.pt','last.pt',
                     'centers.json','center_rows.jsonl'):pin(out/name)
        summary,protocol=read_json(out/'summary.json'),read_json(out/'protocol.json')
        args=protocol['args'];role=(args['seed'],args['augment'])
        require(role in {(s,a) for s in (193,211) for a in (False,True)} and role not in roles,'fixed_role_matrix')
        roles.add(role)
        require(args['stage']=='fit' and args['out']==job['out'],'not_fit_output')
        require(module[0] in job['argv'],'fixed_job_module')
        for opt in ('source','train','dual_train','initial_checkpoint','cohort_contract','out','training_pool','loss','seed',
                    'epochs','samples','batch_size','anchor','lr'):
            flag='--'+opt.replace('_','-')
            require(job['argv'][job['argv'].index(flag)+1]==str(args[opt]),'argv_protocol:'+opt)
        require(('--augment' in job['argv']) is args['augment'],'argv_augment')
        for path,digest in protocol['inputs_sha256'].items():
            require('/test/' not in path and '/sealed/' not in path,'unexpected_test_input')
            pin(path,digest)
        require(protocol['inputs_sha256'].get(module[0])==pins[module[0]],'fixed_protocol_module_pin')
        for path,digest in job['inputs_sha256'].items():pin(path,digest)
        train=Path(args['train']).resolve();dual=Path(args['dual_train']).resolve()
        require(train.parent.name=='train' and dual.parent.name=='train','train_cache_only')
        for cache in (train,dual):
            for name in ('summary.json','EXIT_CODE'):pin(cache.parent/name)
            require((cache.parent/'EXIT_CODE').read_text().strip()=='0','train_cache_exit')
            require(read_json(cache.parent/'summary.json')['feature_cache_sha256']==pin(cache),'train_cache_producer_SHA')
        key=(str(train),)+role
        if key not in cache_exposures:
            data=torch.load(train,map_location='cpu',weights_only=False,mmap=True)
            require(data['manifest']['cohort']=='train' and data['manifest']['recall']=='single','wrong_train_cache')
            rows=[{k:r[k] for k in old.KEY_FIELDS} for r in data['rows']]
            require(len({tuple(r[k] for k in old.KEY_FIELDS) for r in rows})==len(rows),'duplicate_train_identity')
            cache_exposures[key]=old.recompute_exposure(rows,*role);del data
        exposure,ng,ni=cache_exposures[key]
        require(summary['training_groups']==ng and summary['internal_groups']==ni,'group_count')
        initial_path=Path(args['initial_checkpoint']).resolve()
        for name in ('summary.json','EXIT_CODE'):pin(initial_path.parent/name)
        require((initial_path.parent/'EXIT_CODE').read_text().strip()=='0','initial_producer_exit')
        inherited=torch.load(initial_path,map_location='cpu',weights_only=False)
        require(len(inherited['state_dict'])==10,'original_ten_tensor_capacity')
        producer=read_json(initial_path.parent/'summary.json')
        require(producer['best']['epoch']==inherited['epoch'] and producer['best']['alpha']==inherited['alpha']
                and producer['training_cache_sha256']==inherited['train_sha256'],'initial_producer_binding')
        checkpoints={k:torch.load(out/(k+'.pt'),map_location='cpu',weights_only=False) for k in ('initial','best','last')}
        control_path=control_root/f's{role[0]}'/('aug' if role[1] else 'no_aug')
        for name in ('last.pt','summary.json','EXIT_CODE'):pin(control_path/name)
        require((control_path/'EXIT_CODE').read_text().strip()=='0','control_exit')
        control=torch.load(control_path/'last.pt',map_location='cpu',weights_only=False)
        control_summary=read_json(control_path/'summary.json')
        require(control_summary['epochs']==64 and control_summary['samples_per_epoch']==8192
                and control_summary['exposure_sha256']==control['exposure_sha256']
                and control_summary['pool_exposure_sha256']==control['pool_exposure_sha256'],'control_summary_binding')
        history=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines() if x]
        expected=dict(training_pool='mixed',loss='listwise',seed=role[0],augment=role[1])
        estimate=read_json(out/'centers.json')
        audit_rows=[json.loads(x) for x in (out/'center_rows.jsonl').read_text().splitlines() if x]
        require(len(audit_rows)==estimate['query_rows'],'center_rows_count')
        result=audit_model(summary,protocol,inherited,checkpoints,history,expected,exposure,estimate,
                           pin(out/'centers.json'),pin(out/'center_rows.jsonl'),control)
        results.append(dict(job_id=job['id'],role=role,output=str(out),**result))
    require(roles=={(s,a) for s in (193,211) for a in (False,True)},'incomplete_fixed_family')
    evaluations=verify_evaluations(queue,results,module[0],pin)
    require({p:sha(p) for p in pins}==pins,'audit_input_changed_during_run')
    return dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        queue_sha256=pin(queue_path),scope='fixedcenter_training_no_test',training_jobs=4,validation_jobs=8,
        checks=dict(same_initial_tensors=True,frozen_observation=True,frozen_centers=True,matched_exposure=True,
                    finite=True,full_budget=True,real_driver_double_exit=True,complete_validation_metadata=True),models=results,evaluations=evaluations,input_sha256=pins,
        inputs_unchanged=True,best_epoch_zero_allowed=True,draws_per_fit=524288,steps_per_fit=16384,
        budget_evidence='last.epoch64 + exact64*8192 args + independent full exposure/pool SHA; history every8 incl0',
        external_api_calls=0)


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','queue','run','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--control-root');args=parser.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        value=audit(args.root,args.queue,args.run,args.control_root)
        (out/'summary.json').write_text(json.dumps(value,indent=2)+'\n');(out/'EXIT_CODE').write_text('0\n')
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(status='FAIL',error=repr(exc)))+'\n')
        (out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

"""功能概述：扩展冻结器统一封存已预注册main8与固定中心2配置及双池控制的24个测试任务。
仅消费完整train/val选择、原48+12任务、追加4fit+8val审计和独立专项review。
复用旧冻结器的真实legacy sealed缓存登记、seed/forward身份、资格不可用和驱动契约。
固定中心具名映射额外绑定中心JSON/rowaudit、模型和新module/adapter的精确SHA。
只读取测试cache的producer summary/EXIT身份；测试tensor与预测从不打开。
旧build_pool_frozen.py和旧科学源保持不变；本工具只构建，绝不执行sealed队列。
"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path

import build_pool_frozen as old
import audit_pool_fixed_center_training as additional_audit
import summarize_pool_validation as validation

SCHEMA=old.SCHEMA
PLAN_SCHEMA='pool_extended_sealed_test_plan_20261023'
JOINT_SCHEMA='pool_joint_validation_20261023'
ADDED_CONFIGS=('fixedcenter_mixed_listwise_single','fixedcenter_mixed_listwise_dual')
CONFIGS=tuple(validation.CONFIGS)+ADDED_CONFIGS
ALL_CONFIGS=CONFIGS+('frozen_single','frozen_dual')
SOURCE_NAMES=old.SOURCE_NAMES+('data.py',)
ADDITIONAL_CHECKS=old.AUDIT_CHECKS+('frozen_centers','real_driver_double_exit','complete_validation_metadata')
sha=old.sha
read_checked=old.read_checked
checked_pins=old.checked_pins
merge=old.merge
exit0=old.exit0


def require(value,message):
    if not value:raise ValueError(message)


def choose(signals):
    """沿用原guard与并列顺序，仅加入已另行预注册的两配置。"""
    require(set(signals)==set(ALL_CONFIGS),'incomplete_or_unregistered_joint_family')
    baseline=signals['frozen_single'];results={};eligible=[]
    for config in CONFIGS:
        value=signals[config]
        gates=dict(normal_MRR=value['MRR']>baseline['MRR'],top1=value['top1']>=baseline['top1'],
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
        configurations=results,eligible_configurations=eligible,promotion=False,selection_scope='train_val_only',main8_preserved=True)


def binding(record,name,path,pins):
    path=str(Path(path).resolve())
    require(record.get(name)==dict(path=path,sha256=pins[path]),'joint_binding:'+name)


def check_result_review(path,summary_path,summary,joint_tool,pins):
    record=read_checked(path,pins)
    require(record.get('status')=='PASS' and record.get('independent') is True
            and bool(record.get('reviewer'))
            and record.get('scope')=='joint_validation_results_and_implementation',
            'independent_joint_validation_review_required')
    required={str(summary_path):pins[str(summary_path)],str(joint_tool):sha(joint_tool)}
    merge(required,summary['input_sha256']);merge(required,summary['source_sha256'])
    require(all(record.get('reviewed_sha256',{}).get(p)==h for p,h in required.items()),
            'joint_review_missing_current_evidence')


def check_additional(root,queue_path,run,audit_path,pins):
    queue_path=Path(queue_path).resolve();run=Path(run).resolve()
    audit=read_checked(audit_path,pins);exit0(Path(audit_path).resolve().parent/'EXIT_CODE',pins)
    queue=read_checked(queue_path,pins)
    require(audit.get('status')=='PASS' and audit.get('scope')=='fixedcenter_training_no_test'
            and audit.get('queue_sha256')==pins[str(queue_path)]
            and audit.get('training_jobs')==4 and audit.get('validation_jobs')==8
            and audit.get('inputs_unchanged') is True
            and audit.get('draws_per_fit')==524288 and audit.get('steps_per_fit')==16384
            and all(audit.get('checks',{}).get(k) is True for k in ADDITIONAL_CHECKS),
            'complete_additional_training_audit_required')
    checked_pins(audit['input_sha256'],pins)
    require(all(audit['input_sha256'].get(str(Path(p).resolve()))==sha(p)
                for p in (additional_audit.__file__,additional_audit.old.__file__)),
            'additional_audit_tool_not_bound')
    def pin(path,expected=None):
        path=str(Path(path).resolve());checked_pins({path:expected or sha(path)},pins);return pins[path]
    additional_audit.verify_driver(run,queue_path,queue,pin)
    models=audit.get('models',[]);evaluations=audit.get('evaluations',[])
    require(len(models)==4 and {tuple(m['role']) for m in models}=={(s,a) for s in validation.SEEDS for a in (False,True)},
            'additional_audit_model_matrix')
    require(len(evaluations)==8 and {tuple(e['role']) for e in evaluations}==
            {(s,p,c) for s in validation.SEEDS for p in ('single','dual') for c in validation.COHORTS},
            'additional_audit_evaluation_matrix')
    by_id={j['id']:j for j in queue['jobs']}
    for item,kind,names in [(m,'training',('summary.json','protocol.json','EXIT_CODE','initial.pt','best.pt','last.pt','centers.json','center_rows.jsonl')) for m in models]+[
            (e,'validation',('summary.json','protocol.json','EXIT_CODE')) for e in evaluations]:
        job=by_id.get(item['job_id']);out=Path(item['output']).resolve()
        require(job is not None and job['kind']==kind and job['out']==str(out)
                and out.is_relative_to(root) and not out.is_relative_to(root/'formal'),'additional_audit_job_binding')
        for name in names:
            path=str(out/name)
            require(path in audit['input_sha256'] and path in pins
                    and audit['input_sha256'][path]==pins[path],'additional_audit_missing_artifact:'+name)
    # 复用新audit的metadata-only完整验证，不重复反序列化训练cache或重放SGD。
    additional_audit.verify_evaluations(queue,models,
        next(p for p in queue['source_inputs_sha256'] if Path(p).name=='h_pool_fixed_center_ranker.py'),pin)
    return audit,queue


def references_for_added(summary,audit,main_references,pins):
    references=dict(main_references)
    evals={tuple(e['role']):e for e in audit['evaluations']}
    for config in ADDED_CONFIGS:
        pool=config.rsplit('_',1)[1]
        for seed in validation.SEEDS:
            expected=main_references[('mixed_listwise_'+pool,seed)]
            for cohort in validation.COHORTS:
                out=Path(evals[(seed,pool,cohort)]['output'])
                protocol_path=out/'protocol.json';protocol=read_checked(protocol_path,pins)
                require(summary['input_sha256'].get(str(protocol_path))==pins[str(protocol_path)],
                        'joint_missing_added_protocol')
                cache=Path(protocol['args']['cache']).resolve()
                legacy=Path(expected['cache']).parents[4]
                expected_cache=legacy/'features'/f'inherited_s{seed}'/pool/cohort/'features.pt'
                require(cache==expected_cache and protocol['args']['seed']==seed,'added_cache_seed_or_cohort')
                producer_path=cache.parent/'summary.json';producer=read_checked(producer_path,pins)
                require(summary['input_sha256'].get(str(producer_path))==pins[str(producer_path)]
                        and producer['checkpoint_sha256']==expected['checkpoint_sha256'],
                        'added_validation_forward_binding')
                require(summary['metadata'][config][str(seed)][cohort]['checkpoints']==
                        evals[(seed,pool,cohort)]['checkpoints'],'joint_added_checkpoint_metadata')
            references[(config,seed)]=expected
    return references


def center_binding(weights,seed,audit,pins):
    models={tuple(m['role']):m for m in audit['models']};artifacts=[];digests=[]
    for role,aug in (('no_aug',False),('aug',True)):
        checkpoint,digest=weights[role];out=Path(checkpoint).parent;model=models[(seed,aug)]
        require(Path(model['output']).resolve()==out and Path(checkpoint).name=='best.pt','added_model_role_path')
        estimate=read_checked(out/'centers.json',pins);rows_path=str(out/'center_rows.jsonl')
        checked_pins({rows_path:model['center_rows_sha256']},pins)
        require(pins[str(out/'centers.json')]==model['center_estimate_sha256']
                and estimate['row_audit_sha256']==model['center_rows_sha256']
                and estimate['seed']==seed and estimate['augment'] is aug
                and estimate['selection_scope']=='fit_train_only','added_center_evidence')
        require(len(estimate['center33'])==33 and len(estimate['O_center3'])==3
                and all(isinstance(v,(float,int)) and math.isfinite(v) for v in estimate['center33']+estimate['O_center3']),
                'added_center_constants')
        artifacts.append(dict(centers=str(out/'centers.json'),centers_sha256=pins[str(out/'centers.json')],
                              row_audit=rows_path,row_audit_sha256=pins[rows_path]))
        digests.append(pins[str(out/'centers.json')])
    return dict(center_artifacts=artifacts,center_estimate_sha256=digests)


def validate_plan(plan,summary,candidate_sha,pins,deferred,references,audit):
    require(plan.get('schema')==PLAN_SCHEMA and plan.get('selection_scope')=='train_val_only'
            and plan.get('preselected')==summary['selection']['preselected']
            and plan.get('purpose')=='sealed_test_ablation_only' and plan.get('promotion') is False,
            'explicit_extended_test_plan_required')
    require(plan.get('python')=='/home/user/.venvs/spectrexpert/bin/python','test_python_must_match_protocol')
    identity=plan.get('fixed_test_identity',{})
    require(set(identity)=={'query_sha256','identity_content_sha256','variant_mask_content_sha256'}
            and all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in identity.values()),
            'fixed_test_identity_required')
    entries=plan.get('evaluations',[]);keys=[(e['config'],e['seed']) for e in entries]
    require(len(keys)==24 and len(set(keys))==24 and set(keys)=={(c,s) for c in ALL_CONFIGS for s in validation.SEEDS},
            'complete_extended24_matrix_required')
    centers={}
    for entry in entries:
        config,seed=entry['config'],entry['seed']
        old.cache_info(entry,identity,candidate_sha,pins,deferred,references[(config,seed)])
        weights=old.weights_for(summary,config,seed)
        for path,digest in weights.values():
            checked_pins({path:digest},pins);exit0(Path(path).parent/'EXIT_CODE',pins)
            read_checked(Path(path).parent/'summary.json',pins)
        if config in ADDED_CONFIGS:
            centers[(config,seed)]=center_binding(weights,seed,audit,pins)
    return entries,centers


def build(args,out):
    root=Path(args.root).resolve();source=Path(args.source).resolve();pins={};deferred={}
    summary_path=Path(args.validation_summary).resolve();summary=read_checked(summary_path,pins)
    require(pins[str(summary_path)]==args.summary_sha256,'joint_summary_sha_mismatch')
    exit0(summary_path.parent/'EXIT_CODE',pins)
    require(summary.get('schema')==JOINT_SCHEMA and summary.get('selection')==choose(summary.get('signals',{})),
            'joint_selection_not_complete_or_reproducible')
    require(set(summary['metadata'])==set(ALL_CONFIGS),'joint_metadata_matrix')
    checked_pins(summary['source_sha256'],pins)
    joint_tool=Path(args.joint_summarizer).resolve()
    require(summary['source_sha256'].get(str(joint_tool))==sha(joint_tool),'joint_summarizer_source_not_bound')
    main_path=Path(args.main_validation_summary).resolve();main=read_checked(main_path,pins)
    exit0(main_path.parent/'EXIT_CODE',pins);binding(summary,'main_validation_summary',main_path,pins)
    require(main.get('schema')==validation.SCHEMA and main.get('selection')==validation.choose(main['signals']),
            'original_validation_selection')
    require(main['selection']['preselected'] is None and main['selection']['eligible_configurations']==[],
            'fixed_original_main_result')
    checked_pins(main['source_sha256'],pins)
    require(main['source_sha256'].get(str(Path(validation.__file__).resolve()))==sha(validation.__file__),
            'original_validation_summarizer_binding')
    for config in tuple(validation.CONFIGS)+('frozen_single','frozen_dual'):
        require(summary['signals'][config]==main['signals'][config]
                and summary['metadata'][config]==main['metadata'][config],'original_validation_changed')
    require(summary.get('historical_incumbent')==main.get('historical_incumbent'),'historical_incumbent_changed')
    run_pins={}
    formal=validation.verify_run(args.formal_run or root/'runs/formal','formal',run_pins,{})
    validation.verify_run(args.diagnostics_run or root/'runs/diagnostics','diagnostics',run_pins,{})
    require(all(main['input_sha256'].get(path)==digest or main['source_sha256'].get(path)==digest
                for path,digest in run_pins.items()),'main_missing_run_evidence')
    merge(pins,run_pins)
    registration=read_checked(args.preregistration,pins)
    require({f['family'] for f in registration['families']}==set(validation.CONFIGS)
            and registration['seeds']==list(validation.SEEDS),'preregistration_matrix')
    additional_registration=read_checked(args.additional_preregistration,pins)
    require(additional_registration.get('schema')=='fixed_train_center_conditional_preregistration_20261023'
            and additional_registration.get('single_additional_family')=='fixedcenter_mixed_listwise'
            and additional_registration.get('configurations')==list(ADDED_CONFIGS)
            and additional_registration.get('seeds')==list(validation.SEEDS)
            and additional_registration.get('roles')==['no_aug','aug']
            and additional_registration.get('fits')==4,'additional_registration_family')
    eligibility=read_checked(args.eligibility,pins)
    require(eligibility.get('status')=='COMPLETE_FIXED_CANDIDATE_ELIGIBILITY_UNAVAILABLE'
            and eligibility.get('qualified_independent_structures')==0 and eligibility.get('promotion_allowed') is False
            and eligibility.get('candidate_count')==114982,'expected_unavailable_eligibility_audit')
    candidate_sha=eligibility['candidate_order_sha256']
    require(all(meta['candidate_order_sha256']==candidate_sha for seeds in summary['metadata'].values()
                for cohorts in seeds.values() for meta in cohorts.values()),'validation_candidate_library_changed')
    audit=read_checked(args.train_audit,pins);exit0(Path(args.train_audit).resolve().parent/'EXIT_CODE',pins)
    launch=read_checked(Path(args.formal_run or root/'runs/formal')/'launch_receipt.json',pins)
    old.check_audit(audit,formal,next(iter(launch['queue_sha256'].values())),pins)
    extra,queue=check_additional(root,args.additional_queue,args.additional_run,args.additional_train_audit,pins)
    for field,path in (('fixed_center_training_audit',args.additional_train_audit),('additional_queue',args.additional_queue)):
        binding(summary,field,path,pins)
    require(summary['additional_run']==str(Path(args.additional_run).resolve()),'joint_additional_run_binding')
    checked_pins(summary['input_sha256'],pins)
    check_result_review(args.validation_review,summary_path,summary,joint_tool,pins)
    legacy_pins={};registry=old.legacy_cache_registry(root,legacy_pins);merge(pins,legacy_pins)
    references=old.cache_references(root,main,pins,registry)
    references=references_for_added(summary,extra,references,pins)
    plan=read_checked(args.test_plan,pins)
    entries,centers=validate_plan(plan,summary,candidate_sha,pins,deferred,references,extra)
    sources={str(source/'nmr_transfer'/name):sha(source/'nmr_transfer'/name) for name in SOURCE_NAMES}
    require(plan.get('source_sha256')==sources,'test_plan_scientific_source_changed');checked_pins(sources,pins)
    tool_paths=[__file__,old.__file__,validation.__file__,additional_audit.__file__,additional_audit.old.__file__,
        joint_tool,args.fixed_center_ranker,args.sealed_adapter,args.test_summarizer]
    tools={str(Path(p).resolve()):sha(p) for p in tool_paths};checked_pins(tools,pins)
    require(plan.get('tool_sha256')==tools,'test_plan_tool_sha_required')
    require(str(Path(args.fixed_center_ranker).resolve()) in queue['source_inputs_sha256']
            and queue['source_inputs_sha256'][str(Path(args.fixed_center_ranker).resolve())]==tools[str(Path(args.fixed_center_ranker).resolve())],
            'additional_training_module_changed')
    required=dict(pins);merge(required,deferred)
    approval=read_checked(args.review,pins);old.review(approval,required)
    now=dt.datetime.now(dt.timezone.utc);end=dt.datetime.fromisoformat(plan['end_utc'].replace('Z','+00:00'))
    require(now<=dt.datetime(2026,10,23,14,35,41,tzinfo=dt.timezone.utc),'freeze_deadline_passed')
    require(end.tzinfo is not None and now<end<=dt.datetime(2026,10,23,16,35,41,tzinfo=dt.timezone.utc),
            'test_deadline_outside_protocol_or_expired')
    evaluation=dict(pins);merge(evaluation,deferred)
    frozen=dict(schema=SCHEMA,frozen_utc=now.isoformat(),selection_scope='train_val_only',
        preselected=summary['selection']['preselected'],baseline='frozen_single',verify_eligibility='unavailable',
        promotion=False,independent_acceptance_status='unavailable',allowed_ranker_configs=[],ranker_configurations={},
        fixed_center_configurations={},candidate_count=114982,candidate_order_sha256=candidate_sha,
        test_variants=old.VARIANTS,fixed_test_identity=plan['fixed_test_identity'],configurations=sorted(ALL_CONFIGS),
        evaluation_inputs_sha256=evaluation,tool_sha256=tools,validation_summary_sha256=args.summary_sha256,
        validation_summary_path=str(summary_path),test_plan_path=str(Path(args.test_plan).resolve()),
        reviewed_inputs_sha256=required,freeze_review=dict(path=str(Path(args.review).resolve()),sha256=pins[str(Path(args.review).resolve())]))
    frozen['sealed_evidence_paths']={key:str(Path(value).resolve()) for key,value in dict(
        main_validation_summary=args.main_validation_summary,training_audit=args.train_audit,
        additional_training_audit=args.additional_train_audit,preregistration=args.preregistration,
        additional_preregistration=args.additional_preregistration,eligibility=args.eligibility,
        builder=__file__,legacy_builder=old.__file__,joint_summarizer=joint_tool,
        validation_summarizer=validation.__file__,training_auditor=additional_audit.__file__,
        legacy_training_auditor=additional_audit.old.__file__,test_summarizer=args.test_summarizer,
        validation_review=args.validation_review).items()}
    frozen_path=out/'frozen_selection.json';jobs=[]
    for entry in sorted(entries,key=lambda e:(e['config'],e['seed'])):
        config,seed=entry['config'],entry['seed'];weights=old.weights_for(summary,config,seed)
        allowed=dict(no_aug_checkpoint=weights['no_aug'][0],aug_checkpoint=weights['aug'][0],seed=seed,
                     cohort='test',recall=config.rsplit('_',1)[1],cache_sha256=entry['cache_sha256'])
        require(allowed not in frozen['allowed_ranker_configs'],'duplicate_effective_ranker_configuration')
        frozen['allowed_ranker_configs'].append(allowed)
        frozen['ranker_configurations'].setdefault(config,{})[str(seed)]=allowed
        output=root/'sealed'/config/f's{seed}'/'test';require(not output.exists(),'sealed_output_already_exists')
        argv=[plan['python'],'-m','nmr_transfer.h_pool_ranker','--stage','evaluate']
        direct=[str(source/'nmr_transfer'/n) for n in old.SOURCE_NAMES[:-1]]
        if config in ADDED_CONFIGS:
            fixed=dict(cache=entry['cache'],cache_sha256=entry['cache_sha256'],cohort='test',recall=allowed['recall'],seed=seed,
                no_aug_checkpoint=weights['no_aug'][0],aug_checkpoint=weights['aug'][0],
                no_aug_sha256=weights['no_aug'][1],aug_sha256=weights['aug'][1],**centers[(config,seed)])
            frozen['fixed_center_configurations'].setdefault(config,{})[str(seed)]=fixed
            argv=[plan['python'],str(Path(args.sealed_adapter).resolve()),'--source',str(source),'--config',config]
            direct += [str(Path(args.sealed_adapter).resolve()),str(Path(args.fixed_center_ranker).resolve()),
                       str(source/'nmr_transfer/data.py'),str(Path(args.review).resolve())]
            for item in fixed['center_artifacts']:direct += [item['centers'],item['row_audit']]
        argv += ['--cache',entry['cache'],'--no-aug-checkpoint',weights['no_aug'][0],
            '--aug-checkpoint',weights['aug'][0],'--seed',str(seed),'--out',str(output),
            '--frozen-selection',str(frozen_path)]
        direct += [entry['cache']]+[str(Path(entry['cache']).parent/name) for name in ('summary.json','EXIT_CODE')]
        for path,digest in weights.values():direct += [path]+[str(Path(path).parent/name) for name in ('summary.json','EXIT_CODE')]
        jobs.append(dict(id=f'sealed_{config}_s{seed}',kind='sealed',gpu=True,argv=argv,cwd=str(source),
            out=str(output),deps=[],inputs_sha256={p:evaluation[p] for p in direct}))
    frozen_path.write_text(json.dumps(frozen,indent=2)+'\n')
    bundle=dict(evaluation);merge(bundle,{str(frozen_path):sha(frozen_path)})
    for job in jobs:merge(job['inputs_sha256'],{str(frozen_path):sha(frozen_path)})
    queue=dict(schema='nmr_research_jobs_20261022',end_utc=plan['end_utc'],source_inputs_sha256=bundle,jobs=jobs)
    queue_path=out/'sealed_queue.json';queue_path.write_text(json.dumps(queue,indent=2)+'\n')
    checked_pins(pins,{})
    receipt=dict(schema='pool_extended_freeze_build_20261023',status='BUILT_NOT_EXECUTED',jobs=24,
        preselected=frozen['preselected'],promotion=False,independent_acceptance_status='unavailable',
        input_sha256=pins,deferred_to_sealed_driver_sha256=deferred,tool_sha256=tools,
        output_sha256={str(p):sha(p) for p in (frozen_path,queue_path)},external_api_calls=0,
        test_cache_values_read=False,test_predictions_read=False)
    (out/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','source','validation-summary','summary-sha256','main-validation-summary','validation-review',
        'joint-summarizer','test-plan','review','train-audit','additional-train-audit','additional-queue','additional-run',
        'eligibility','preregistration','additional-preregistration','fixed-center-ranker','sealed-adapter','test-summarizer','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--formal-run');parser.add_argument('--diagnostics-run')
    args=parser.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        receipt=build(args,out);(out/'EXIT_CODE').write_text('0\n')
        print(json.dumps(dict(status=receipt['status'],jobs=receipt['jobs'],output_sha256=receipt['output_sha256'])))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc)))+'\n')
        (out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

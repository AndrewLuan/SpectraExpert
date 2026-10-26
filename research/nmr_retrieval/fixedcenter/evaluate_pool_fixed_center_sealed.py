"""功能概述：为已专项独审并冻结的fixedcenter两个配置提供sealed test推理适配器。
preflight先核冻结文件、独立review路径/SHA/PASS和named config/seed/recall/cache/权重/
中心artifact完整allowlist，随后才读取test缓存。validate_test核固定1371三身份SHA、
自然512及零训练正例注入；evaluate仅复用冻结模型row_delta和既定gamma/排序公式，
不训练、不选型、不开放verify。main独占输出，保存输入/源码始末SHA及真实EXIT。
本文件为独立D工具，不修改已经launch pin的固定中心模型或任何旧科学源码。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import h_pool_fixed_center_ranker as fixed
from h_pool_fixed_center_ranker import require
from nmr_transfer.h_research_contract import atomic_json,input_fingerprints,rank_shortlist,summarize_predictions
from nmr_transfer.h_research_ranker import result_row

SCHEMA='nmr_pool_ranker_20261023'
CONFIGS={'fixedcenter_mixed_listwise_single','fixedcenter_mixed_listwise_dual'}
VARIANTS={'observed':1000,'frequency_crop':200,'peak_dropout':171}
IDENTITY={'query_sha256','identity_content_sha256','variant_mask_content_sha256'}
EVIDENCE_KEYS={'main_validation_summary','training_audit','additional_training_audit','preregistration',
    'additional_preregistration','eligibility','builder','legacy_builder','joint_summarizer',
    'validation_summarizer','training_auditor','legacy_training_auditor','test_summarizer','validation_review'}
TOOL_KEYS={'builder','legacy_builder','joint_summarizer','validation_summarizer','training_auditor',
           'legacy_training_auditor','test_summarizer'}


def sha(path):return input_fingerprints([path])[str(Path(path).resolve())]
def canonical(path):return str(Path(path).resolve())
def read_json(path):return json.loads(Path(path).read_text())
def source_paths():return [Path(__file__).resolve()]+fixed.source_paths()


def preflight(args):
    """此阶段只读JSON/EXIT/源码和权重metadata；未核review前绝不hash或load test cache。"""
    frozen_path=Path(args.frozen_selection).resolve();frozen=read_json(frozen_path)
    require(frozen['schema']==SCHEMA and frozen['selection_scope']=='train_val_only' and frozen['frozen_utc'],
            'sealed_frozen_schema')
    require(frozen['verify_eligibility']=='unavailable' and frozen['promotion'] is False,'sealed_verify_unavailable')
    approval=frozen['freeze_review'];review_path=Path(approval['path']).resolve()
    require(sha(review_path)==approval['sha256'],'sealed_review_SHA')
    review=read_json(review_path)
    require(review.get('status')=='PASS' and review.get('independent') is True and review.get('reviewer')
            and review.get('scope')=='freeze_plan_and_implementation','sealed_review_PASS_required')
    require(frozen['reviewed_inputs_sha256'] and all(review.get('reviewed_sha256',{}).get(p)==h
            for p,h in frozen['reviewed_inputs_sha256'].items()),'sealed_review_inputs')
    require(args.config in CONFIGS and args.seed in (193,211),'sealed_named_config')
    recall=args.config.rsplit('_',1)[1]
    base_allowed=dict(no_aug_checkpoint=canonical(args.no_aug_checkpoint),aug_checkpoint=canonical(args.aug_checkpoint),
                      seed=args.seed,cohort='test',recall=recall,cache_sha256=None)
    named=frozen['ranker_configurations'][args.config][str(args.seed)]
    base_allowed['cache_sha256']=named['cache_sha256']
    require(named==base_allowed and named in frozen['allowed_ranker_configs'],'sealed_named_allowlist')
    selected=frozen['fixed_center_configurations'][args.config][str(args.seed)]
    require(all(selected[k]==v for k,v in base_allowed.items()) and selected['cache']==canonical(args.cache),
            'sealed_fixed_allowlist')
    pins={str(frozen_path):sha(frozen_path),str(review_path):approval['sha256']}
    pinned=frozen['evaluation_inputs_sha256']
    evidence=frozen.get('sealed_evidence_paths',{})
    require(set(evidence)==EVIDENCE_KEYS,'sealed_mandatory_evidence_paths')
    metadata_paths=[frozen['test_plan_path'],frozen['validation_summary_path']]+list(evidence.values())
    # 固定schema字段导出关键输入，不能由可任意裁剪的required map替代。
    for path in source_paths()+metadata_paths:
        path=canonical(path);actual=sha(path)
        require(frozen['reviewed_inputs_sha256'].get(path)==actual
                and review['reviewed_sha256'].get(path)==actual and pinned.get(path)==actual,
                'sealed_required_review_binding:'+path)
        pins[path]=actual
    plan=read_json(frozen['test_plan_path'])
    require(plan['schema']=='pool_extended_sealed_test_plan_20261023'
            and plan['selection_scope']=='train_val_only' and plan['purpose']=='sealed_test_ablation_only'
            and plan['promotion'] is False and plan['preselected']==frozen['preselected']
            and plan['fixed_test_identity']==frozen['fixed_test_identity'],'sealed_reviewed_plan_identity')
    entries=[e for e in plan['evaluations'] if e['config']==args.config and e['seed']==args.seed]
    require(len(entries)==1 and entries[0]['cache']==canonical(args.cache)
            and entries[0]['cache_sha256']==selected['cache_sha256'],'sealed_reviewed_plan_configuration')
    tools=plan['tool_sha256'];sources=plan['source_sha256']
    require(all(tools.get(canonical(evidence[k]))==pins[canonical(evidence[k])] for k in TOOL_KEYS),
            'sealed_reviewed_plan_tools')
    require(all({**sources,**tools}.get(canonical(path))==pins[canonical(path)] for path in source_paths()),
            'sealed_reviewed_plan_scientific_sources')
    joint=read_json(frozen['validation_summary_path'])
    require(joint['schema']=='pool_joint_validation_20261023','sealed_joint_validation_schema')
    expected_checkpoints={canonical(args.no_aug_checkpoint):selected['no_aug_sha256'],
                          canonical(args.aug_checkpoint):selected['aug_sha256']}
    require(all(joint['metadata'][args.config][str(args.seed)][cohort]['checkpoints']==expected_checkpoints
                for cohort in ('original_val','tune')),'sealed_reviewed_validation_checkpoints')
    require(read_json(evidence['main_validation_summary'])['schema']=='pool_validation_selection_20261023'
            and read_json(evidence['training_audit'])['status']=='PASS'
            and read_json(evidence['additional_training_audit'])['status']=='PASS'
            and read_json(evidence['additional_training_audit'])['scope']=='fixedcenter_training_no_test',
            'sealed_reviewed_producer_evidence')
    def pin(path,expected=None):
        path=canonical(path);actual=sha(path)
        require(expected is None or actual==expected,'sealed_input_SHA')
        require(pinned.get(path)==actual,'sealed_input_not_frozen:'+path)
        require(frozen['reviewed_inputs_sha256'].get(path)==actual
                and review['reviewed_sha256'].get(path)==actual,'sealed_input_not_reviewed:'+path)
        pins[path]=actual;return actual
    # 所有review前置已通过。此处才开始cache byteSHA和producer核验。
    for path in source_paths():pin(path)
    for path in (args.cache,args.no_aug_checkpoint,args.aug_checkpoint):
        digest=pin(path)
        parent=Path(path).resolve().parent
        for name in ('summary.json','EXIT_CODE'):pin(parent/name)
        require((parent/'EXIT_CODE').read_text().strip()=='0','sealed_producer_exit')
    producer=read_json(Path(args.cache).resolve().parent/'summary.json')
    require(producer['cohort']=='test' and producer['recall']==recall
            and producer['feature_cache_sha256']==pins[canonical(args.cache)],'sealed_cache_producer_metadata')
    require(pins[canonical(args.cache)]==selected['cache_sha256'],'sealed_cache_allowlist_SHA')
    require(pins[canonical(args.no_aug_checkpoint)]==selected['no_aug_sha256']
            and pins[canonical(args.aug_checkpoint)]==selected['aug_sha256'],'sealed_checkpoint_allowlist_SHA')
    artifacts=selected['center_artifacts']
    require(len(artifacts)==2,'sealed_center_roles')
    for path,artifact in zip((args.no_aug_checkpoint,args.aug_checkpoint),artifacts):
        parent=Path(path).resolve().parent
        require(artifact['centers']==str(parent/'centers.json')
                and artifact['row_audit']==str(parent/'center_rows.jsonl'),'sealed_center_path_roles')
        pin(artifact['centers'],artifact['centers_sha256']);pin(artifact['row_audit'],artifact['row_audit_sha256'])
    require(selected['center_estimate_sha256']==[a['centers_sha256'] for a in artifacts],'sealed_center_SHA_roles')
    identity=frozen['fixed_test_identity']
    require(set(identity)==IDENTITY and all(isinstance(h,str) and len(h)==64
            and all(c in '0123456789abcdef' for c in h) for h in identity.values()),'sealed_identity_required')
    return frozen,selected,pins


def content_sha(value):
    def convert(item):
        if hasattr(item,'tolist'):return convert(item.tolist())
        if isinstance(item,dict):return {k:convert(v) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [convert(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(convert(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_test(data,frozen,selected):
    manifest=data['manifest'];rows=data['rows']
    require(manifest['cohort']=='test' and manifest['recall']==selected['recall'],'sealed_test_only')
    require(manifest['candidate_count']==114982 and manifest['candidate_order_sha256']==fixed.CANDIDATE_SHA
            and frozen['candidate_order_sha256']==fixed.CANDIDATE_SHA,'sealed_candidate_library')
    require(manifest['pool_size']==512 and manifest['records']==1371 and manifest['base_queries']==1000
            and manifest['appended_training_positives']==0 and len(rows)==1371,'sealed_natural1371')
    require(dict(collections.Counter(r['variant'] for r in rows))==VARIANTS
            and manifest['variants']==VARIANTS,'sealed_variant_counts')
    require(all(manifest[k]==v for k,v in frozen['fixed_test_identity'].items()),'sealed_fixed_identity')
    identity=content_sha([{key:r[key] for key in ('query_order','variant','source','record_id','smiles','group')} for r in rows])
    masks=content_sha([(tuple(r[k] for k in ('query_order','variant','source','record_id','smiles')),
                        r['mask_content_sha256']) for r in rows])
    require(identity==manifest['identity_content_sha256'] and masks==manifest['variant_mask_content_sha256'],
            'sealed_row_identity_or_mask_changed')
    for row in rows:
        fixed.base.validate_row(row,False)
        require(row['evaluation_pool_size']==512 and len(row['candidate_indices'])==512
                and len(row['candidate_indices'].unique())==512,'sealed_no_positive_injection')
    return rows


@torch.inference_mode()
def evaluate(args,out,device,status):
    frozen,selected,pins=preflight(args)
    data,actual=fixed.read_feature_cache(args.cache)
    require(actual==selected['cache_sha256'],'sealed_cache_loaded_SHA')
    rows=validate_test(data,frozen,selected)
    for path,digest in data['manifest']['source_inputs_sha256'].items():
        require(frozen['evaluation_inputs_sha256'].get(path)==digest and sha(path)==digest,'sealed_scientific_input_SHA')
        pins[path]=digest
    models=[fixed.load_checkpoint(p,device) for p in (args.no_aug_checkpoint,args.aug_checkpoint)]
    for role,(model,scale,saved) in zip((False,True),models):
        require(saved['args']['seed']==args.seed and saved['args']['augment'] is role,'sealed_checkpoint_role')
        require(saved['center_estimate_sha256']==selected['center_estimate_sha256'][int(role)],'sealed_loaded_center_SHA')
        fixed.validate_fit_budget(argparse.Namespace(**saved['args']))
        fixed.base.check_science(saved,data)
        require(torch.isfinite(scale).all() and all(torch.isfinite(v).all() for v in model.state_dict().values()),
                'sealed_nonfinite_model')
    require(models[0][2]['train_sha256']==models[1][2]['train_sha256']
            and models[0][2]['dual_train_sha256']==models[1][2]['dual_train_sha256'],'sealed_model_training_pair')
    atomic_json(out/'protocol.json',dict(args=vars(args),inputs_sha256=pins))
    outputs=[];durations=[];started=time.monotonic()
    for i,row in enumerate(rows):
        begin=time.monotonic()
        correction=[saved['alpha']*model.row_delta(row,scale,device)*scale[0] for model,scale,saved in models]
        obs=row['query_observation'];gamma=.75 if obs.get('windows') or obs.get('partial_peaks') else .25
        costs=row['coarse_scores'].to(device)+(1-gamma)*correction[0]+gamma*correction[1]
        rank,top=rank_shortlist(row,costs);durations.append(time.monotonic()-begin)
        outputs.append(dict(result_row(row,rank,top),gamma=gamma,mask_content_sha256=row['mask_content_sha256'],
                            natural_candidate_indices=row['candidate_indices'].tolist(),
                            natural_candidate_costs=costs.cpu().tolist()))
        if (i+1)%100==0:status('evaluating',records=i+1,total=len(rows))
    require(input_fingerprints(pins)==pins,'sealed_input_changed_during_run')
    with (out/'predictions.jsonl').open('x') as stream:
        for row in outputs:stream.write(json.dumps(row)+'\n')
    summary=dict(metrics=summarize_predictions(outputs),cohort='test',config=args.config,seed=args.seed,
                 recall=selected['recall'],query_sha256=data['manifest']['query_sha256'],
                 candidate_order_sha256=data['manifest']['candidate_order_sha256'],
                 input_sha256=pins,checkpoints=input_fingerprints([args.no_aug_checkpoint,args.aug_checkpoint]),
                 center_estimate_sha256=selected['center_estimate_sha256'],
                 elapsed_seconds=time.monotonic()-started,reranking_seconds_mean=float(np.mean(durations)),
                 reranking_seconds_p95=float(np.quantile(durations,.95)),external_api_calls=0)
    atomic_json(out/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','config','cache','no-aug-checkpoint','aug-checkpoint','frozen-selection','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--seed',type=int,choices=(193,211),required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    require(Path(args.source).resolve()==Path(fixed.base.__file__).resolve().parent.parent,'sealed_scientific_source')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    status=lambda stage,**kw:atomic_json(out/'status.json',dict(stage=stage,**kw))
    try:
        torch.set_num_threads(4);evaluate(args,out,torch.device(args.device),status)
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except BaseException as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

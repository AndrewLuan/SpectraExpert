"""功能概述：实际validation锁定后，分两阶段构建固定12格、全部fresh的test元数据。
prepare只产生registry/plan/待独审SHA清单；freeze只接受精确plan独审后产生frozen/queue。
最终专项独审在queue生成之后另行完成，review路径预先出现在CLI但其SHA不进入queue，
从而避免review↔queue循环。operator实际始末独立核该review，叶评分先核冻结再读payload。
本工具仅标准库读取显式映射的源码和JSON/EXIT；不扫描目录、不读张量/预测/旧报告。
旧24计划只消费evaluations/cache_metadata/fixed_test_identity三字段。4缓存SHA来自旧
producer声明与已注册plan，不冒充本轮实际缓存字节重哈希。大输入仅登记为deferred。
缺最终validation lock/独立结果review，或资格不是全部四baseline重评时立即拒绝。
输入refs都是canonical path/SHA，metadata_map提供各ref对应的本地复制路径与SHA。
输出只写--out新目录。字段中的路径始终是scope.remote_root下的最终部署路径。
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

BASELINE='fixedcenter_mixed_listwise_single'
SEEDS=(193,211);POOLS=('single','dual');CONFIGS=('fixed_single','fixed_dual','cap5_single','cap5_dual','cap10_single','cap10_dual')
VARIANTS={'observed':1000,'frequency_crop':200,'peak_dropout':171}
ORDER='72b72140c09ae14ae8d0aa4edad624a151093c9df246b2052de8a3a4ea616bad'
IDENTITY={'query_sha256','identity_content_sha256','variant_mask_content_sha256'}
REFS=('protocol','initial_bindings','validation_bindings','validation_summary','validation_selection','validation_lock',
      'validation_review','legacy_test_plan','baseline_qualification','source_registry','test_implementation_review',
      'statistics_implementation_review','builder_implementation_review')
RUNTIME={'h_pool_ranker.py','h_research_ranker.py','h_research_contract.py','h_research_observation.py','h_global_ranker.py',
         'h_global_features.py','data.py','h_pool_fixed_center_ranker.py','cap_ranker.py','train_cap_ranker.py'}


def need(value,label):
    if not value:raise ValueError(label)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def hexsha(x):return type(x) is str and re.fullmatch('[0-9a-f]{64}',x) is not None
def encoded(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def canonical(path):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts and str(p)==str(path),'canonical_path');return str(p)
def safe(path):
    raw=Path(path).absolute();need('..' not in raw.parts and not raw.is_symlink(),'unsafe_metadata_path');p=raw.resolve(strict=True)
    for v in (raw,p):
        need(v.suffix in ('.json','.py') or v.name in ('EXIT_CODE','EXTERNAL_EXIT_CODE'),'metadata_source_only')
        need('prediction' not in v.name.lower() and 'report' not in v.name.lower(),'prediction_or_report_forbidden')
    return p
def ref_ok(ref):
    need(type(ref) is dict and set(ref)=={'path','sha256'} and hexsha(ref['sha256']),'explicit_path_SHA_ref');canonical(ref['path']);return ref
def merge(*maps):
    result={}
    for mapping in maps:
        need(type(mapping) is dict,'SHA_map_object')
        for path,h in mapping.items():
            canonical(path);need(hexsha(h) and (path not in result or result[path]==h),'conflicting_SHA');result[path]=h
    return result


class Reader:
    def __init__(self,ref):
        ref_ok(ref);self.map_path=safe(ref['path']);raw=self.map_path.read_bytes();need(sha(raw)==ref['sha256'],'metadata_map_SHA')
        self.map_sha=ref['sha256'];self.map=json.loads(raw);self.reads={};self.locals={}
        need(set(self.map['files'])==set(self.map['sha256']),'metadata_map_exact_files')
    def raw(self,path,wanted,allow_registered_test_metadata=False):
        canonical(path);need(hexsha(wanted) and self.map['sha256'].get(path)==wanted,'explicit_copied_ref_SHA')
        local=safe(self.map['files'][path])
        if local.suffix!='.py' and not allow_registered_test_metadata:
            for p in (Path(path),local):need(not any(v=='test' or v=='sealed' or v.startswith(('sealed-','sealed_resume_')) for v in p.parts),'unregistered_test_metadata_forbidden')
        raw=local.read_bytes();need(sha(raw)==wanted,'actual_metadata_or_source_SHA')
        need(path not in self.reads or self.reads[path]==wanted,'metadata_drift');self.reads[path]=wanted;self.locals[path]=str(local);return raw
    def ref(self,ref,allow_registered_test_metadata=False):ref_ok(ref);return json.loads(self.raw(ref['path'],ref['sha256'],allow_registered_test_metadata))
    def finish(self):
        need(sha(safe(self.map_path).read_bytes())==self.map_sha,'metadata_map_changed')
        for path,h in self.reads.items():need(sha(safe(self.locals[path]).read_bytes())==h,'consumed_input_changed')


def independent(review,required,label):
    need(review.get('status')=='PASS' and review.get('independent') is True and bool(review.get('reviewer')) and type(review.get('scope')) is str,label+'_independent_PASS')
    need(all(review.get('reviewed_sha256',{}).get(p)==h for p,h in required.items()),label+'_exact_SHA_coverage')


def validate_lock(reader,refs,protocol):
    # 此闭环必须先于legacy test metadata的首读。
    lock=reader.ref(refs['validation_lock']);summary=reader.ref(refs['validation_summary']);selection=reader.ref(refs['validation_selection']);review=reader.ref(refs['validation_review'])
    need(lock.get('schema')=='cap10_validation_lock_20261024' and lock.get('status')=='LOCKED'
         and lock.get('selection_scope')=='train_validation_only' and lock.get('test_reselection_allowed') is False
         and lock.get('promotion') is False and lock.get('protocol_sha256')==refs['protocol']['sha256'],'complete_actual_validation_lock_required')
    need(summary.get('status')=='PASS' and summary.get('test_read') is False and summary.get('promotion') is False,'validation_result_scope')
    chosen=summary.get('preselected')
    need(chosen in (BASELINE,'cap10_dual') and chosen==lock.get('preselected')==selection.get('preselected') and lock.get('selection_locked') is True,'locked_preselected_unchanged')
    need(type(lock.get('validation_run')) is str and summary.get('validation_run')==lock['validation_run'],'exact_final_successful_validation_run')
    need(all(lock.get(k)==refs[k] for k in ('validation_summary','validation_selection','validation_review')),'lock_binds_prior_result_review_without_cycle')
    need(summary.get('bindings')==refs['validation_bindings'],'actual_reviewed_validation_registry_ref')
    reviewed_inputs=merge(summary['input_sha256'],summary['source_sha256'],{refs[k]['path']:refs[k]['sha256'] for k in ('validation_summary','validation_selection','validation_bindings','initial_bindings')})
    independent(review,reviewed_inputs,'actual_validation_result_review')
    need('validation' in review['scope'],'validation_review_scope')
    values=summary['signals'];candidate=values['cap10_dual'];control=values['cap5_dual'];baseline=protocol['validation']['baseline_signals']
    need(all(type(v[k]) in (int,float) and math.isfinite(v[k]) for v in (candidate,control,baseline) for k in ('MRR','top1','frequency_crop_MRR','peak_dropout_MRR','macro_MRR')),'finite_locked_validation_signals')
    gates={'MRR_gt_fixed_single':candidate['MRR']>baseline['MRR']+1e-12,'MRR_gt_matched_C5dual':candidate['MRR']>control['MRR']+1e-12}
    gates.update({k:candidate[k]>=baseline[k]-1e-12 for k in ('top1','frequency_crop_MRR','peak_dropout_MRR','macro_MRR')})
    need(summary['selection']['gates']==gates and summary['selection']['preselected']==chosen
         and chosen==('cap10_dual' if all(gates.values()) else BASELINE),'existing_locked_guard_decision_verified')
    return chosen,review


def checkpoint_ref(reader,record,inputs,newcap=None):
    result=copy.deepcopy(record);result.update(path=record['checkpoint'],sha256=record['checkpoint_sha256'])
    producer=reader.ref({'path':record['producer_summary'],'sha256':record['producer_summary_sha256']})
    epoch,alpha=(64,1.) if newcap is not None else (record['best']['epoch'],record['best']['alpha'])
    need(type(epoch) is int and epoch>=0 and type(alpha) in (int,float) and 0<=alpha<=1,'registered_role_epoch_alpha')
    if newcap is not None:
        need(producer['epoch']==64 and producer['alpha']==1. and producer['delta_cap']==newcap
             and producer['final_checkpoint_sha256']==record['checkpoint_sha256'],'audited_final64_cap_role')
    else:need(producer['best']['epoch']==epoch and producer['best']['alpha']==alpha,'native_baseline_best_role')
    result.update(epoch=epoch,alpha=alpha)
    required={record['checkpoint']:record['checkpoint_sha256']}
    for key in ('centers','center_rows','producer_summary'):required[record[key]]=record[key+'_sha256']
    exitpath=str(Path(record['checkpoint']).parent/'EXIT_CODE');need(exitpath in inputs,'checkpoint_EXIT_registered')
    need(reader.raw(exitpath,inputs[exitpath])==b'0\n','checkpoint_producer_EXIT0')
    need(all(inputs.get(p)==h for p,h in required.items()),'checkpoint_metadata_input_closure')
    return result


def prepare(reader,request):
    need(request['schema']=='cap10_test_metadata_request_20261025','request_schema');scope=request['scope'];root=Path(canonical(scope['remote_root']))
    need(scope['all_12_fresh'] is True and scope['test_reselection_allowed'] is False and scope['promotion'] is False,'all12fresh_no_reselection_scope')
    need(scope['selection_scope']=='train_validation_only','train_validation_selection_scope')
    refs=request['refs'];need(set(refs)==set(REFS),'exact_core_metadata_refs')
    for ref in refs.values():ref_ok(ref)
    protocol=reader.ref(refs['protocol'])
    need(protocol['schema']=='cap10_controlled_protocol_20261024' and protocol['remote_root']==str(root) and protocol['test_allowed'] is False
         and protocol['promotion'] is False and protocol['pool_size']==512 and protocol['candidate_library_size']==114982,'original_protocol_universe')
    chosen,valreview=validate_lock(reader,refs,protocol)
    initial=reader.ref(refs['initial_bindings']);validation=reader.ref(refs['validation_bindings']);qualification=reader.ref(refs['baseline_qualification'])
    need(validation['training_bindings']==refs['initial_bindings'],'same_audited_initial_four_best_registry')
    need(qualification['schema']=='cap10_baseline_reuse_qualification_20261024' and qualification['status']=='PASS' and qualification['qualified'] is False and qualification['all_four_or_none'] is True
         and qualification['registered_baseline_jobs_count']==4 and qualification['new_evaluation_jobs_count']==12
         and qualification['validation_preselected']==chosen and qualification['test_predictions_opened'] is False and qualification['test_tensors_opened'] is False,'all_four_baselines_fresh_qualification')
    registry=reader.ref(refs['source_registry']);tools=registry['tool_paths'];leaf=registry['leaf_source_sha256'];downstream=registry['downstream_source_sha256'];operators=registry['operator_source_sha256']
    native=root/'native_cost_evaluation';need(scope['evaluation_source_root']==str(native),'exact_native_evaluation_source_root')
    expected_paths={'evaluator':native/'cap_test_evaluate_ranker.py','common':native/'cap_test_evaluation_common.py',
                    'statistics':root/'resource_ready_analysis/summarize_sealed_cap_results.py','pairmath':root/'sealed_cap_pairmath.py','matrix':root/'sealed_cap_matrix.py','cap_module':root/'cap_ranker.py'}
    need(all(tools[k]['path']==str(p) for k,p in expected_paths.items()),'exact_native_tool_role_paths')
    need(len({r['path'] for r in tools.values()})==len(tools),'unique_tool_role_paths')
    need(not(set(leaf)&set(downstream) or set(leaf)&set(operators) or set(downstream)&set(operators)),'disjoint_leaf_downstream_operator_sources')
    need(tools['cap_module']==validation['cap_module'] and tools['fixed_module']==validation['fixed_module'],'audited_actual_runtime_module_refs')
    names={Path(p).name for p in leaf};need(len(leaf)==len(RUNTIME)+2 and names==RUNTIME|{Path(tools[k]['path']).name for k in ('evaluator','common')},'exact_runtime_leaf_source_names')
    need(set(downstream)=={tools[k]['path'] for k in ('statistics','pairmath','matrix')},'exact_downstream_sources')
    need(set(operators)=={str(root/n) for n in ('isolation_entry.py','supervise_resource_ready_queue.py','launch_resource_ready_stage.py')},'exact_operator_sources')
    source=merge(leaf,downstream,operators,{tools['builder']['path']:tools['builder']['sha256']})
    need(tools['builder']['sha256']==sha(Path(__file__).read_bytes()),'actual_builder_source_SHA')
    need(all(source.get(ref['path'])==ref['sha256'] for ref in tools.values()),'all_tool_paths_exact_source_refs')
    proofs=[valreview]+[reader.ref(refs[k]) for k in ('test_implementation_review','statistics_implementation_review','builder_implementation_review')]
    reviewed={}
    for proof in proofs:
        independent(proof,{},'implementation_review');reviewed.update(proof.get('reviewed_sha256',{}))
    need(all(reviewed.get(p)==h for p,h in source.items()),'all_actual_source_independently_reviewed')
    for p,h in source.items():reader.raw(p,h)
    for key in ('evaluator','common','cap_module','fixed_module'):
        ref=tools[key];ref_ok(ref);need(leaf.get(ref['path'])==ref['sha256'],'actual_tool_in_leaf_source')
    inputs=merge(initial['input_sha256'],validation['input_sha256'],source,{r['path']:r['sha256'] for r in refs.values()})
    need(Path(refs['legacy_test_plan']['path']).name=='extended24_test_plan.json','explicit_prior_plan_metadata_only')
    legacy=reader.ref(refs['legacy_test_plan'],True);identity=legacy['fixed_test_identity'];metadata=legacy['cache_metadata'];evaluations=legacy['evaluations']
    need(type(identity) is dict and set(identity)==IDENTITY and all(hexsha(h) for h in identity.values()),'fixed_three_test_identity')
    need(set(metadata)=={f'{seed}/{pool}' for seed,pool in itertools.product(SEEDS,POOLS)},'four_explicit_cache_metadata')
    caches={str(seed):{'test':{}} for seed in SEEDS}
    for seed,pool in itertools.product(SEEDS,POOLS):
        m=metadata[f'{seed}/{pool}'];need(m['seed']==seed and m['recall']==pool and m['candidate_order_sha256']==ORDER,'cache_seed_pool_order')
        parent=Path(m['cache']).parent
        need(parent.name==pool and parent.parent.name=='inherited_s'+str(seed) and Path(m['cache']).name=='features.pt','fixed_registered_cache_seed_path')
        need(m['producer_summary']['path']==str(parent/'summary.json') and m['producer_exit']['path']==str(parent/'EXIT_CODE'),'exact_cache_producer_siblings')
        producer=reader.ref(m['producer_summary'],True);exitref=m['producer_exit'];need(reader.raw(exitref['path'],exitref['sha256'],True)==b'0\n','cache_producer_EXIT0')
        need(producer['cohort']=='test' and producer['recall']==pool and producer['candidate_count']==114982 and producer['pool_size']==512
             and producer['candidate_order_sha256']==ORDER and producer['records']==1371 and producer['base_queries']==1000 and producer['variants']==VARIANTS,'registered_test_cache_manifest')
        need(all(producer[k]==v for k,v in identity.items()) and producer['feature_cache_sha256']==m['cache_sha256'],'producer_three_identity_payload_SHA')
        need(producer['checkpoint_sha256']==m['forward_checkpoint_sha256']==validation['caches'][str(seed)]['original_val'][pool]['manifest']['checkpoint_sha256'],'inherited_seed_forward_checkpoint')
        selected=[e for e in evaluations if e['seed']==seed and e['config']=='fixedcenter_mixed_listwise_'+pool]
        need(len(selected)==1 and (selected[0]['cache'],selected[0]['cache_sha256'])==(m['cache'],m['cache_sha256']),'old_exact_registered_cache_path')
        need(all(selected[0]['checkpoints'][r]=={'path':initial['roles'][str(seed)][r]['checkpoint'],'sha256':initial['roles'][str(seed)][r]['checkpoint_sha256']} for r in ('no_aug','aug')),'same_original_four_best_baseline')
        cache=dict(path=m['cache'],sha256=m['cache_sha256'],manifest=producer,producer=dict(path=m['producer_summary']['path'],sha256=m['producer_summary']['sha256'],exit_path=exitref['path']))
        caches[str(seed)]['test'][pool]=cache
        inputs=merge(inputs,{cache['path']:cache['sha256'],cache['producer']['path']:cache['producer']['sha256'],exitref['path']:exitref['sha256']})
    roles={}
    for family,seed in itertools.product(('fixed','cap5','cap10'),SEEDS):
        pair=initial['roles'][str(seed)] if family=='fixed' else validation['roles'][family[3:]][str(seed)]
        roles[family,seed]={r:checkpoint_ref(reader,pair[r],inputs,None if family=='fixed' else int(family[3:])) for r in ('no_aug','aug')}
    entries=[]
    for config,seed in itertools.product(CONFIGS,SEEDS):
        family,pool=config.rsplit('_',1);cache=caches[str(seed)]['test'][pool];jid=f'test_{config}_s{seed}'
        entries.append(dict(config=config,seed=seed,pool=pool,cache=cache['path'],cache_sha256=cache['sha256'],checkpoints=roles[family,seed],execution='evaluate',job_id=jid,job_out=str(root/'test'/jid)))
    # leaf只登记当前推理实际可用输入；历史审计/下游统计闭包只保留在plan/frozen侧。
    leaf_inputs=merge(leaf,{refs["protocol"]["path"]:refs["protocol"]["sha256"]})
    for entry in entries:
        cache=caches[str(entry["seed"])]["test"][entry["pool"]]
        for path in (entry["cache"],cache["producer"]["path"],cache["producer"]["exit_path"]):leaf_inputs[path]=inputs[path]
        for record in entry["checkpoints"].values():
            for path in (record["path"],record["centers"],record["center_rows"],record["producer_summary"],str(Path(record["path"]).parent/"EXIT_CODE")):leaf_inputs[path]=inputs[path]
    bindings=dict(schema='cap10_test_bindings_20261024',source_checkout=protocol['source_checkout'],entries=entries,caches=caches,
      cap_module=tools['cap_module'],fixed_module=tools['fixed_module'],evaluator=tools['evaluator'],source_sha256=leaf,input_sha256=leaf_inputs)
    plan=dict(schema='cap10_test_plan_20261025',status='PLANNED_REQUIRES_SPECIFIC_REVIEW',entries=entries,preselected=chosen,selection_locked=True,
      promotion=False,selection_performed=False,test_reselection_allowed=False,fixed_test_identity=identity,source_sha256=source,evidence_input_sha256=inputs,
      bindings={'path':str(root/'test_bindings.json'),'sha256':sha(encoded(bindings))},metadata_refs=refs,tool_paths=tools,
      cache_SHA_provenance='Prior registered producer feature_cache_sha256; this tool does not read or hash test tensor bytes.',
      validation_run=reader.ref(refs['validation_lock'])['validation_run'])
    return bindings,plan,protocol


def freeze(reader,request):
    bindings,plan,protocol=prepare(reader,request);refs=request['freeze_stage_refs'];root=Path(request['scope']['remote_root'])
    need(refs['bindings']['path']==str(root/'test_bindings.json') and refs['plan']['path']==str(root/'test_plan.json'),'canonical_prepared_output_paths')
    need(reader.ref(refs['bindings'])==bindings and reader.ref(refs['plan'])==plan,'exact_prepared_plan_and_bindings')
    needed=merge(plan['evidence_input_sha256'],bindings['input_sha256'],plan['source_sha256'],{refs[k]['path']:refs[k]['sha256'] for k in ('bindings','plan')})
    review=reader.ref(refs['plan_review']);independent(review,needed,'specific_test_plan_review');need('test' in review['scope'] and 'implementation' in review['scope'],'specific_plan_implementation_scope')
    inputs=merge(plan['evidence_input_sha256'],bindings['input_sha256'],{refs['plan_review']['path']:refs['plan_review']['sha256']})
    frozen=dict(schema='cap10_sealed_selection_20261024',selection_locked=True,promotion=False,preselected=plan['preselected'],entries=plan['entries'],
      fixed_test_identity=plan['fixed_test_identity'],input_sha256=inputs,source_sha256=plan['source_sha256'],test_plan=refs['plan'],
      validation_summary=request['refs']['validation_summary'],validation_selection=request['refs']['validation_lock'],validation_review=request['refs']['validation_review'],
      validation_rule_applied_selection=request['refs']['validation_selection'],validation_selection_semantics='Final locked selection document; immutable rule-applied selection is separately referenced.',
      validation_lock=request['refs']['validation_lock'],freeze_review=refs['plan_review'],tool_paths=plan['tool_paths'],frozen_utc=datetime.now(timezone.utc).isoformat())
    fref={'path':str(root/'frozen_test_selection.json'),'sha256':sha(encoded(frozen))};final=str(root/'test_freeze_independent_review.json');jobs=[]
    for n,e in enumerate(plan['entries']):
        command=[bindings['evaluator']['path']]
        args=dict(source=protocol['source_checkout'],protocol=str(root/'protocol.json'),bindings=str(root/'test_bindings.json'),frozen_selection=fref['path'],freeze_review=final,config=e['config'],seed=e['seed'],cache=e['cache'],out=e['job_out'],device='cuda:0',threads=4)
        for k,v in args.items():command+=['--'+k.replace('_','-'),str(v)]
        p=bindings['caches'][str(e['seed'])]['test'][e['pool']]['producer'];reads=[e['cache'],*(r['path'] for r in e['checkpoints'].values()),p['path'],p['exit_path']]
        access=str(root/'access_logs/test'/(e['job_id']+'.json'));argv=['/home/user/.venvs/spectrexpert/bin/python','-B',str(root/'isolation_entry.py'),'--bindings',str(root/'test_bindings.json'),'--phase','test','--access-log',access,'--output-root',e['job_out']]
        for path in reads:argv+=['--allow-read',path]
        jobs.append(dict(id=e['job_id'],argv=argv+['--']+command,command=command,cwd=protocol['source_checkout'],gpu=n%8,job_out=e['job_out'],access_log=access,allowed_reads=reads))
    evidence_metadata={p:h for p,h in plan['evidence_input_sha256'].items() if Path(p).suffix not in ('.pt','.pth','.ckpt')}
    sources=merge(plan['source_sha256'],evidence_metadata,{str(root/'protocol.json'):request['refs']['protocol']['sha256'],refs['bindings']['path']:refs['bindings']['sha256'],fref['path']:fref['sha256'],refs['plan']['path']:refs['plan']['sha256'],refs['plan_review']['path']:refs['plan_review']['sha256']})
    need(final not in sources and final not in bindings['input_sha256'],'final_review_outside_preexisting_hash_cycle')
    operator_refs={role:dict(path=str(root/name),sha256=sources[str(root/name)]) for role,name in [('supervisor','supervise_resource_ready_queue.py'),('launcher','launch_resource_ready_stage.py')]}
    queue=dict(schema='cap10_controlled_queue_20261024',phase='test',run_name='test',root=str(root),local_root=protocol['local_root'],python='/home/user/.venvs/spectrexpert/bin/python',
      bindings=str(root/'test_bindings.json'),selection_locked_before_test=True,jobs=jobs,input_sha256=bindings['input_sha256'],source_sha256=sources,review_paths=[final],maximum_seconds=10800,max_parallel=8,operator_paths=operator_refs,resource_wait=dict(maximum_seconds=120,poll_seconds=.2),external_api_calls=0,promotion=False,automatic_retry=False)
    final_required=merge(needed,inputs,sources,{fref['path']:fref['sha256'],str(root/'test_queue.json'):sha(encoded(queue))})
    return frozen,queue,final_required


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','freeze'),required=True);p.add_argument('--request',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False);own=Path(__file__).resolve();startsha=sha(own.read_bytes())
    try:
        requestpath=safe(args.request);raw=requestpath.read_bytes();request=json.loads(raw);reader=Reader(request['metadata_map'])
        if args.stage=='prepare':
            b,plan,_=prepare(reader,request);values={'test_bindings.json':b,'test_plan.json':plan}
            values['required_plan_review_sha256.json']=merge(plan['evidence_input_sha256'],b['input_sha256'],plan['source_sha256'],{plan['bindings']['path']:sha(encoded(b)),str(Path(request['scope']['remote_root'])/'test_plan.json'):sha(encoded(plan))})
        else:
            frozen,queue,required=freeze(reader,request);values={'frozen_test_selection.json':frozen,'test_queue.json':queue,'required_final_review_sha256.json':required}
        reader.finish();need(sha(own.read_bytes())==startsha and sha(safe(requestpath).read_bytes())==sha(raw),'tool_or_request_changed')
        for name,value in values.items():(out/name).write_bytes(encoded(value))
        receipt=dict(status='METADATA_ONLY_REQUIRES_INDEPENDENT_REVIEW',stage=args.stage,source_sha256={str(own):startsha},input_sha256=reader.reads,metadata_map_sha256=reader.map_sha,
          output_sha256={name:sha(encoded(value)) for name,value in values.items()},test_tensor_bytes_read=False,test_predictions_read=False,model_calls=0,scientific_dispatch=0,promotion=False)
        (out/'receipt.json').write_bytes(encoded(receipt));(out/'EXIT_CODE').write_text('0\n');print(json.dumps({'status':receipt['status'],'stage':args.stage}))
    except BaseException as exc:
        (out/'failure.json').write_bytes(encoded({'error':repr(exc)}));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

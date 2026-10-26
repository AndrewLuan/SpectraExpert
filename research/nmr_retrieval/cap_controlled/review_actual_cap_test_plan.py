"""功能概述：非作者独立核本轮实际 cap test plan 的矩阵、来源和准入闭包。
只读取 root 已登记的64个小元数据/source别名与本轮prepare执行收据；不导入或调用
prepare_cap_test_metadata，不读取测试张量或预测。check 独立重建12格角色/缓存、
叶层85与完整246输入集合、validation锁DAG和qualifier全新评估政策。read/finish对
每次读取执行规范路径、实际SHA与末端漂移检查；main仅写本reviewer新目录中的检查结果。
测试cache的4个payload SHA仅来自producer声明，并明确推迟到最终准入后operator核验。
"""
from pathlib import Path
import argparse, copy, datetime, hashlib, itertools, json, math

M=Path(__file__).resolve().parent
S=M/'test_metadata_staging'
R='/home/user/IRfingerprints/SpectrExpert/outputs/nmr_cap10_controlled_20261024T163633Z'
ORDER='72b72140c09ae14ae8d0aa4edad624a151093c9df246b2052de8a3a4ea616bad'
VARIANTS={'observed':1000,'frequency_crop':200,'peak_dropout':171}
CONFIGS=('fixed_single','fixed_dual','cap5_single','cap5_dual','cap10_single','cap10_dual')
ROLES=('no_aug','aug')
READS={}
def need(x,label):
    if not x: raise ValueError(label)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path,expected=None):
    p=Path(path).absolute()
    need('..' not in p.parts and p.resolve(strict=True)==p and not p.is_symlink(),'canonical_no_symlink')
    need(p.suffix.lower() not in ('.pt','.pth','.ckpt','.npy','.npz') and 'prediction' not in p.name.lower(),'no_tensor_prediction_read')
    raw=p.read_bytes();h=digest(raw)
    need(expected is None or h==expected,'actual_SHA:'+str(p))
    need(str(p) not in READS or READS[str(p)]==h,'input_read_drift')
    READS[str(p)]=h
    return raw
def doc(path,expected=None):return json.loads(read(path,expected))
def merge(*maps):
    result={}
    for mapping in maps:
        for p,h in mapping.items():
            need(p not in result or result[p]==h,'SHA_collision:'+p);result[p]=h
    return result
def zero(v):need(type(v) is int and v==0,'strict_integer_EXIT0')
def stamp(t):
    d=datetime.datetime.fromisoformat(t.replace('Z','+00:00'));need(d.utcoffset().total_seconds()==0,'UTC');return d

def check():
    read(__file__)
    request=doc(S/'prepare_request.json')
    aliases=doc(request['metadata_map']['path'],request['metadata_map']['sha256'])
    need(len(aliases['files'])==64 and set(aliases['files'])==set(aliases['sha256']),'exact64_aliases')
    # Byte verification is restricted to these explicit, already admitted small aliases.
    for remote,path in aliases['files'].items():read(path,aliases['sha256'][remote])
    def ref(v):
        need(v['path'] in aliases['files'] and aliases['sha256'][v['path']]==v['sha256'],'registered_ref_SHA')
        d=doc(aliases['files'][v['path']],v['sha256'])
        if Path(v['path']).name=='extended24_test_plan.json':
            return {k:d[k] for k in ('evaluations','cache_metadata','fixed_test_identity')}
        return d
    refs=request['refs'];objects={k:ref(v) for k,v in refs.items()}
    protocol=objects['protocol'];initial=objects['initial_bindings'];validation=objects['validation_bindings']
    summary=objects['validation_summary'];selection=objects['validation_selection'];lock=objects['validation_lock'];review=objects['validation_review']
    need(request['scope']==dict(remote_root=R,all_12_fresh=True,selection_scope='train_validation_only',test_reselection_allowed=False,promotion=False,evaluation_source_root=R+'/native_cost_evaluation'),'exact_scope')
    need(protocol['test_allowed'] is False and protocol['pool_size']==512 and protocol['candidate_library_size']==114982 and protocol['promotion'] is False,'original_protocol')
    need(lock['status']=='LOCKED' and lock['selection_locked'] is True and lock['promotion'] is False and lock['selection_scope']=='train_validation_only' and lock['test_reselection_allowed'] is False,'actual_lock')
    need(review['status']=='PASS' and review['independent'] is True and bool(review['reviewer']),'actual_independent_validation_review')
    need(all(lock[k]==refs[k] for k in ('validation_summary','validation_selection','validation_review')),'lock_three_ref_DAG')
    need(refs['validation_lock']['path'] not in review['reviewed_sha256'],'no_review_lock_cycle')
    need(summary['bindings']==refs['validation_bindings'] and summary['validation_run']==lock['validation_run']==R+'/runs/validation_resource_ready','successful_validation_binding')
    need(lock['protocol_sha256']==refs['protocol']['sha256'] and validation['training_bindings']==refs['initial_bindings'],'protocol_training_link')
    covered=merge(summary['input_sha256'],summary['source_sha256'],{refs[k]['path']:refs[k]['sha256'] for k in ('validation_summary','validation_selection','validation_bindings','initial_bindings')})
    need(all(review['reviewed_sha256'].get(p)==h for p,h in covered.items()),'actual_validation_full_review_coverage')
    baseline=protocol['validation']['baseline_signals'];cand=summary['signals']['cap10_dual'];control=summary['signals']['cap5_dual']
    need(all(type(v[k]) in (int,float) and math.isfinite(v[k]) for v in (baseline,cand,control) for k in ('MRR','top1','frequency_crop_MRR','peak_dropout_MRR','macro_MRR')),'finite_validation_signals')
    gates={'MRR_gt_fixed_single':cand['MRR']>baseline['MRR']+1e-12,'MRR_gt_matched_C5dual':cand['MRR']>control['MRR']+1e-12}
    gates.update({k:cand[k]>=baseline[k]-1e-12 for k in ('top1','frequency_crop_MRR','peak_dropout_MRR','macro_MRR')})
    chosen='cap10_dual' if all(gates.values()) else 'fixedcenter_mixed_listwise_single'
    need(summary['selection']['gates']==gates and summary['selection']['preselected']==summary['preselected']==selection['preselected']==lock['preselected']==chosen,'existing_rule_verified_no_reselection')
    need(selection.get('selection_locked',False) is False,'immutable_rule_document_not_mutated')
    registry=objects['source_registry'];tools=registry['tool_paths'];leaf=registry['leaf_source_sha256'];down=registry['downstream_source_sha256'];ops=registry['operator_source_sha256']
    need(len(leaf)==12 and len(down)==3 and len(ops)==3 and not(set(leaf)&set(down) or set(leaf)&set(ops) or set(down)&set(ops)),'disjoint12_3_3_sources')
    need(set(ops)=={R+'/'+n for n in ('isolation_entry.py','supervise_resource_ready_queue.py','launch_resource_ready_stage.py')},'future_active_operators_exact')
    expected_paths={'evaluator':R+'/native_cost_evaluation/cap_test_evaluate_ranker.py','common':R+'/native_cost_evaluation/cap_test_evaluation_common.py','statistics':R+'/resource_ready_analysis/summarize_sealed_cap_results.py','pairmath':R+'/sealed_cap_pairmath.py','matrix':R+'/sealed_cap_matrix.py','builder':R+'/prepare_cap_test_metadata.py','cap_module':R+'/cap_ranker.py'}
    need(all(tools[k]['path']==p for k,p in expected_paths.items()) and len({v['path'] for v in tools.values()})==8,'exact_native_tool_roles')
    source=merge(leaf,down,ops,{tools['builder']['path']:tools['builder']['sha256']})
    reviewed=merge(*[objects[k]['reviewed_sha256'] for k in ('test_implementation_review','statistics_implementation_review','builder_implementation_review')])
    for p,h in source.items():need(reviewed.get(p)==h or review['reviewed_sha256'].get(p)==h,'source_implementation_review:'+p)
    for k in ('test_implementation_review','statistics_implementation_review','builder_implementation_review'):
        need(objects[k]['status']=='PASS' and objects[k]['independent'] is True and bool(objects[k]['reviewer']),'source_review_PASS')
    plan=doc(S/'prepare/test_plan.json','610659ef13ffab3caa4b182f0ba5a105f91213a7d41db0c65d3d6ec5b5c65030')
    bindings=doc(S/'prepare/test_bindings.json','c335b3232cfb19f17277be054b41b26efcf304ad92414dfc306ce462ddaee1dc')
    required=doc(S/'prepare/required_plan_review_sha256.json','78da1c6ff5a4006c993e6118a978612c7ce732e551d49d580af4f32a17b18dc0')
    need(plan['preselected']==chosen and plan['selection_locked'] is True and plan['promotion'] is False and plan['selection_performed'] is False and plan['test_reselection_allowed'] is False,'frozen_future_selection_policy')
    need(plan['metadata_refs']==refs and plan['source_sha256']==source and plan['tool_paths']==tools,'plan_metadata_source')
    legacy=objects['legacy_test_plan'];identity=legacy['fixed_test_identity']
    need(set(identity)=={'query_sha256','identity_content_sha256','variant_mask_content_sha256'} and plan['fixed_test_identity']==identity,'three_full_identity')
    inputs=merge(initial['input_sha256'],validation['input_sha256'],source,{v['path']:v['sha256'] for v in refs.values()})
    caches={str(seed):{'test':{}} for seed in (193,211)};deferred={}
    for seed,pool in itertools.product((193,211),('single','dual')):
        cm=legacy['cache_metadata'][str(seed)+'/'+pool];producer=ref(cm['producer_summary']);ex=cm['producer_exit']
        need(read(aliases['files'][ex['path']],ex['sha256'])==b'0\n','test_producer_EXIT0')
        need(cm['seed']==seed and cm['recall']==pool and cm['candidate_order_sha256']==producer['candidate_order_sha256']==ORDER,'seed_pool_candidate_order')
        need(all(producer[k]==v for k,v in identity.items()) and producer['feature_cache_sha256']==cm['cache_sha256'],'declared_cache_identity_SHA')
        need(producer['records']==1371 and producer['base_queries']==1000 and producer['variants']==VARIANTS and producer['candidate_count']==114982 and producer['pool_size']==512 and producer['appended_training_positives']==0,'1371_natural512')
        need(producer['checkpoint_sha256']==cm['forward_checkpoint_sha256']==validation['caches'][str(seed)]['original_val'][pool]['manifest']['checkpoint_sha256'],'forward_seed_role')
        parent=Path(cm['cache']).parent
        need(parent.name==pool and parent.parent.name=='inherited_s'+str(seed) and cm['producer_summary']['path']==str(parent/'summary.json') and ex['path']==str(parent/'EXIT_CODE'),'exact_cache_producer_paths')
        prior=[e for e in legacy['evaluations'] if e['config']=='fixedcenter_mixed_listwise_'+pool and e['seed']==seed]
        need(len(prior)==1 and (prior[0]['cache'],prior[0]['cache_sha256'])==(cm['cache'],cm['cache_sha256']),'legacy_registered_same_cache')
        for role in ROLES:need(prior[0]['checkpoints'][role]=={'path':initial['roles'][str(seed)][role]['checkpoint'],'sha256':initial['roles'][str(seed)][role]['checkpoint_sha256']},'legacy_baseline_same_roles')
        cache=dict(path=cm['cache'],sha256=cm['cache_sha256'],manifest=producer,producer=dict(path=cm['producer_summary']['path'],sha256=cm['producer_summary']['sha256'],exit_path=ex['path']))
        caches[str(seed)]['test'][pool]=cache;deferred[cache['path']]=cache['sha256']
        inputs=merge(inputs,{cache['path']:cache['sha256'],cache['producer']['path']:cache['producer']['sha256'],ex['path']:ex['sha256']})
    need(bindings['caches']==caches and bindings['source_sha256']==leaf and bindings['evaluator']==tools['evaluator'],'bindings_actual_cache_source')
    expected_entries=[]
    for config,seed in itertools.product(CONFIGS,(193,211)):
        family,pool=config.rsplit('_',1);oldpair=initial['roles'][str(seed)] if family=='fixed' else validation['roles'][family[3:]][str(seed)]
        pair={}
        for role,original in oldpair.items():
            producer=ref({'path':original['producer_summary'],'sha256':original['producer_summary_sha256']})
            epoch=original['best']['epoch'] if family=='fixed' else 64
            alpha=original['best']['alpha'] if family=='fixed' else 1.
            if family!='fixed':need(producer['epoch']==64 and producer['alpha']==1. and producer['delta_cap']==int(family[3:]) and producer['final_checkpoint_sha256']==original['checkpoint_sha256'],'newfinal64_alpha1_role')
            else:need(producer['best']['epoch']==epoch and producer['best']['alpha']==alpha,'fixed_original_best_role')
            pair[role]=dict(copy.deepcopy(original),path=original['checkpoint'],sha256=original['checkpoint_sha256'],epoch=epoch,alpha=alpha)
            for key in ('checkpoint','centers','center_rows','producer_summary'):need(inputs[original[key]]==original[key+'_sha256'],'role_input_ref')
            ep=str(Path(original['checkpoint']).parent/'EXIT_CODE')
            need(read(aliases['files'][ep],aliases['sha256'][ep])==b'0\n','role_producer_EXIT0')
        ca=caches[str(seed)]['test'][pool];jid='test_'+config+'_s'+str(seed)
        expected_entries.append(dict(config=config,seed=seed,pool=pool,cache=ca['path'],cache_sha256=ca['sha256'],checkpoints=pair,execution='evaluate',job_id=jid,job_out=R+'/test/'+jid))
    need(bindings['entries']==plan['entries']==expected_entries and len({(e['config'],e['seed']) for e in expected_entries})==12,'exact12_allfresh_roles')
    leafinputs=merge(leaf,{refs['protocol']['path']:refs['protocol']['sha256']})
    for entry in expected_entries:
        ca=caches[str(entry['seed'])]['test'][entry['pool']]
        for p in (ca['path'],ca['producer']['path'],ca['producer']['exit_path']):leafinputs[p]=inputs[p]
        for role in entry['checkpoints'].values():
            for p in (role['path'],role['centers'],role['center_rows'],role['producer_summary'],str(Path(role['path']).parent/'EXIT_CODE')):leafinputs[p]=inputs[p]
    need(bindings['input_sha256']==leafinputs and len(leafinputs)==85 and plan['evidence_input_sha256']==inputs and len(inputs)==244,'independent85_leaf_244_evidence_union')
    outputs={R+'/test_bindings.json':digest(read(S/'prepare/test_bindings.json')),R+'/test_plan.json':digest(read(S/'prepare/test_plan.json'))}
    need(required==merge(inputs,leafinputs,source,outputs) and len(required)==246,'all246_required_reconstructed')
    need(plan['bindings']=={'path':R+'/test_bindings.json','sha256':outputs[R+'/test_bindings.json']},'actual_bindings_ref')
    # Qualifier execution and explicit order-proof normalization remain metadata only.
    qreq=doc(S/'baseline_reuse_request.json');admit=doc(S/'baseline_metadata_admission.json');qual=objects['baseline_qualification']
    need(qreq['historical']=={} and admit['historical_evidence']=={} and admit['status']=='ADMITTED' and admit['explicit_root_authorization'] is True,'empty_historical_admitted')
    expectedsha=digest(json.dumps(qreq['expected_cells'],sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    need(expectedsha==admit['expected_cells_sha256']==qual['expected_cells_sha256'],'expected_cells_order_bound')
    need(qreq['order_proof_provenance']==admit['order_proof_provenance'] and qreq['order_proof_provenance']['standalone_producer_query_order_sha256_field_claimed'] is False,'honest_order_digest_alias')
    proof=qreq['order_proof_provenance']['source_mapping'];read(proof['path'],proof['sha256'])
    need(len(qreq['expected_cells'])==4 and {(e['config'],e['seed']) for e in qreq['expected_cells']}=={('fixed_'+pool,seed) for pool,seed in itertools.product(('single','dual'),(193,211))},'qualifier_exact4_cells')
    for cell in qreq['expected_cells']:
        seed=cell['seed'];pool=cell['recall'];ca=caches[str(seed)]['test'][pool]
        expected_roles={}
        for role,original in initial['roles'][str(seed)].items():
            expected_roles[role]={key:{'path':original[field],'sha256':original[field+'_sha256']} for key,field in [('checkpoint','checkpoint'),('centers','centers'),('center_rows','center_rows')]}
            expected_roles[role]['alpha']=original['best']['alpha']
        need(cell['checkpoint_roles']==expected_roles,'qualifier_exact_initial_baseline_roles')
        need(cell['conditions']==dict(cohort='test',query_count=1371,base_queries=1000,variant_counts=VARIANTS,candidate_count=114982,natural_pool_size=512,appended_training_positives=0) and cell['source_sha256']==leaf,'qualifier_exact_conditions_runtime_source')
        need(cell['config']=='fixed_'+pool and cell['cache']=={'path':ca['path'],'sha256':ca['sha256']} and cell['fixed_test_identity']==identity and cell['query_order_sha256']==identity['identity_content_sha256'] and cell['candidate_order_sha256']==ORDER,'expected_cell_identity_order')
    need(qual['status']=='PASS' and qual['qualified'] is False and qual['all_four_or_none'] is True and qual['new_evaluation_jobs_count']==12 and qual['registered_baseline_jobs_count']==4 and qual['registered_new_cap_jobs_count']==8,'all4_baseline_fresh_12total')
    need(qual['input_sha256']==qual['input_end_sha256'] and qual['source_sha256']==qual['source_end_sha256'] and qual['inputs_unchanged'] is True and qual['test_predictions_opened'] is False and qual['test_tensors_opened'] is False,'qualifier_no_test_input_closure')
    for name,stage,sourcepath in [('baseline_qualification','qualifier',M/'qualify_baseline_reuse.py'),('prepare','prepare',M/'prepare_cap_test_metadata.py')]:
        execution=doc(S/(name+'.actual_execution.json'));zero(execution['internal_exit']);zero(execution['external_exit'])
        need(execution['test_tensors_opened'] is False and execution['test_predictions_opened'] is False and execution['scientific_dispatch']==0,'metadata_execution_scope')
        need(stamp(lock['utc'])<=stamp(execution['started_utc'])<=stamp(execution['finished_utc']),'lock_before_actual_metadata')
        need(execution['source_sha256']=={str(sourcepath):digest(read(sourcepath))},'actual_executed_source')
        rp=S/('baseline_reuse_request.json' if stage=='qualifier' else 'prepare_request.json')
        need(execution['request_sha256']==digest(read(rp)) and str(rp) in execution['argv'],'exact_actual_request_argv')
        for p,h in execution['outputs_sha256'].items():read(p,h)
    deployment=doc(S/'deployment/receipt.json','74415258be0c6ee1673531ca22bdbdb78ea24c5fa33ef292b3350343013dbd2a')
    for key in ('probe_before_actual_exit','rsync_actual_exit','probe_after_actual_exit'):zero(deployment[key])
    need(deployment['remote_metadata_source_SHA']==aliases['sha256'] and deployment['formal_RR_source_writes']==0 and deployment['test_tensor_reads']==0 and deployment['test_prediction_reads']==0,'actual_remote64_metadata_closed')
    prepared=doc(S/'deployment/prepared_outputs_receipt.json','79a5d32288dba3b45e5b4eef94864684010b80819f443e86ee696d992cd13271')
    for key in ('before_probe_exit','rsync_exit','after_probe_exit'):zero(prepared[key])
    need(prepared['expected_sha256']==prepared['actual_remote_sha256']==outputs,'actual_remote2_outputs_closed')
    dispositions={}
    for p,h in required.items():
        if p in deferred:dispositions[p]='declared_test_cache_payload_deferred_operator_SHA'
        elif aliases['sha256'].get(p)==h:dispositions[p]='actual_local_and_remote_small_metadata_SHA'
        elif outputs.get(p)==h:dispositions[p]='actual_prepared_local_remote_SHA'
        else:
            need(review['reviewed_sha256'].get(p)==h,'inherited_required_not_actual_reviewed:'+p)
            dispositions[p]='inherited_typed_actual_train_validation_review_c9dd'
    need(len(deferred)==4,'exact4_unopened_test_caches')
    for p,h in READS.items():need(digest(read(p))==h,'end_SHA_drift')
    return dict(required=required,dispositions=dispositions,read_sha256=dict(READS),preselected=chosen,validation_gates=gates,entries=[dict(config=e['config'],seed=e['seed'],pool=e['pool'],execution=e['execution']) for e in expected_entries],declared_test_cache_SHA_deferred=deferred,checks=dict(exact12_allfresh=True,all246_required=True,leaf12_source85_input_and_evidence244=True,validation_lock_DAG=True,qualifier_actual_double0_and_historical_empty=True,explicit1371_order_identity_proof=True,baseline_original_roles_and_new8_final64_alpha1=True,all4_cache_seed_forward_identity=True,canonical_source_metadata_deployment=True,inputs_unchanged=True),actual_test_tensor_reads=0,actual_test_prediction_reads=0,scientific_dispatch=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(exist_ok=False)
    try:
        result=check()
        (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
        (out/'EXIT_CODE').write_text('0\n')
        print(json.dumps({'status':'PASS','required':len(result['required']),'preselected':result['preselected'],'read_files':len(READS)}))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps({'error':repr(exc)},indent=2)+'\n');(out/'EXIT_CODE').write_text('1\n');raise

if __name__=='__main__':main()

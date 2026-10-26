"""功能概述：冻结后单个sealed cell的独立cap/native-fixed评估适配器。
先经cap_test_evaluation_common核实际frozen、最终专项review、12项计划与validation
预选锁，再打开登记test缓存。历史fixed checkpoint保留原alpha/epoch；新cap只收final64
alpha1。自然512先裁剪且标签/身份不进入模型；输出原始fp32成本、trace与完整1371身份。
无forward bank、无训练、无新选型。只写新的job_out，失败保留真实EXIT1。
该文件不修改、不导入已执行validation四源码；共用cap/fixed科学源码仍逐SHA受审。
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cap_test_evaluation_common as _actual_contract
if Path(_actual_contract.__file__).resolve()!=Path(__file__).with_name('cap_test_evaluation_common.py').resolve():
    raise ImportError('native_contract_module_path_mismatch')
from cap_test_evaluation_common import *
SCHEMA='cap10_sealed_evaluation_20261024'
OBS_FIELDS=('semantics','kind','xs','weights','reported_count','windows','partial_peaks')
CANDIDATE_FIELDS=('features','coarse_scores','predicted_ppm','tags','H','K','per_quantile_sigma')


def natural_row(row):
    need(row['evaluation_pool_size']==512 and type(row['evaluation_pool_size']) is int,'natural512')
    ids=row['candidate_indices'];need(len(ids) in (512,513) and ids.dtype in (torch.int32,torch.int64),'candidate_shape')
    need(len(ids.unique())==len(ids) and bool(((ids>=0)&(ids<CANDIDATES)).all()),'candidate_values')
    need(type(row['kind']) is int and 0<=row['kind']<4,'kind')
    out={'kind':row['kind'],'observation_input':{k:row['observation_input'][k] for k in OBS_FIELDS if k in row['observation_input']}}
    for k in CANDIDATE_FIELDS:
        if k not in row:continue
        need(len(row[k])==len(ids),'candidate_length:'+k)
        out[k]=row[k][:512]
        need(bool(torch.isfinite(out[k]).all()),'finite_candidates:'+k)
    need(out['features'].shape==(512,33) and out['coarse_scores'].shape==(512,),'feature_shape')
    return out

def route_gamma(observation):
    return .75 if observation.get('windows') or observation.get('partial_peaks') else .25

def trace_hashes(scored):
    roles=[]
    for trace in scored['traces']:
        arrays=[(k,v.detach().cpu().contiguous().numpy()) for k,v in sorted(trace.items())]
        hashes=[]
        for i in range(512):
            h=hashlib.sha256()
            for k,arr in arrays:
                value=np.asarray(arr[i]);h.update(k.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
            hashes.append(h.hexdigest())
        roles.append(hashes)
    return roles

def outcome(row,scored):
    ids=row['candidate_indices'][:512].cpu().numpy();cost=scored['cost'].cpu().numpy()
    order=np.lexsort((ids,cost));pos=row['target_position'];hit=row['target_in_coarse_pool']
    need(type(hit) is bool and type(pos) is int and ((0<=pos<512)==hit),'target_presence')
    targetcost=None;margin=None;target=None;inrank=None
    if hit:
        targetcost=float(cost[pos]);rank=int((scored['cost']<=scored['cost'][pos]+EPS).sum());inrank=rank;target=int(ids[pos])
        other=int(order[order!=pos][0]);margin=targetcost-float(cost[other])
    else:
        rank=row['outside_pool_rank'];need(type(rank) is int and 512<rank<=CANDIDATES,'native_outside_rank')
    query={k:row[k] for k in IDENTITY}
    out=dict(query_identity=query,coarse_dtype=scored['coarse_dtype'],correction_dtypes=scored['correction_dtypes'],final_cost_dtype=scored['final_cost_dtype'],target_in_pool=hit,rank=rank,RR=1/rank,in_pool_target_rank=inrank,
        cached_outside_pool_rank=None if hit else rank,top1=rank<=1,top5=rank<=5,top10=rank<=10,
        target_candidate_index=target,winner_candidate_index=int(ids[order[0]]),target_cost=targetcost,
        margin=margin,mask_content_sha256=row['mask_content_sha256'],gamma=scored['gamma'],
        natural_candidate_indices=ids.tolist(),natural_candidate_costs=cost.tolist(),
        natural_candidate_coarse_costs=row['coarse_scores'][:512].cpu().tolist(),
        natural_candidate_trace_sha256=trace_hashes(scored),
        maximum_pairwise_correction_margin_bound=2*scored['total_bound'],
        role_absolute_correction_bounds=scored['role_bounds'],appended_training_positive=len(row['candidate_indices'])==513)
    validate_prediction(out);return out

def load_module(path):
    spec=importlib.util.spec_from_file_location('_cap_eval_runtime',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def content_sha(value):
    def convert(item):
        if hasattr(item,'tolist'):return convert(item.tolist())
        if isinstance(item,dict):return {k:convert(v) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [convert(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(convert(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_loaded_test_content(data,frozen):
    """重算旧adapter完全相同的顺序敏感canonical内容，不信任manifest自报SHA。"""
    rows=data['rows'];manifest=data['manifest']
    need(len(rows)==sum(TEST_COUNTS.values()) and manifest['appended_training_positives']==0,'loaded_test_natural_full_count')
    identities=content_sha([{k:r[k] for k in IDENTITY} for r in rows])
    masks=content_sha([(tuple(r[k] for k in IDENTITY[:-1]),r['mask_content_sha256']) for r in rows])
    need(identities==manifest['identity_content_sha256']==frozen['fixed_test_identity']['identity_content_sha256'],'loaded_test_identity_content_SHA')
    need(masks==manifest['variant_mask_content_sha256']==frozen['fixed_test_identity']['variant_mask_content_sha256'],'loaded_test_mask_content_SHA')
    return rows

def model_digest(models):
    h=hashlib.sha256()
    for model,scale,saved in models:
        h.update(json.dumps({k:saved[k] for k in ('schema','alpha','epoch','center_estimate_sha256')},sort_keys=True).encode())
        for k,v in sorted(model.state_dict().items()):
            h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        h.update(scale.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.inference_mode()
def score_natural(models,natural,gamma,device,cap):
    need(set(natural)<=set(CANDIDATE_FIELDS)|{'kind','observation_input'},'score_identity_leak')
    need(gamma in (.25,.75) and len(models)==2,'score_roles')
    traces=[];corrections=[];bounds=[]
    for model,scale,saved in models:
        alpha=saved['alpha'];need(finite(alpha) and 0<=alpha<=1,'registered_alpha')
        need(scale.shape==(33,) and bool(torch.isfinite(scale).all()) and bool((scale>0).all()),'scale')
        captured={}
        def ohook(module,args,output):captured['O']=output.detach()
        def nhook(module,args,output):captured['logits']=output.detach()[:,natural['kind']]
        handles=[model.observation_model.register_forward_hook(ohook),model.network.register_forward_hook(nhook)]
        try:delta=model.row_delta(natural,scale,device)
        finally:
            for handle in handles:handle.remove()
        need(set(captured)=={'O','logits'} and captured['O'].shape==(512,3) and delta.shape==(512,),'hook_shapes')
        x=natural['features'].to(device);o=captured['O']
        trace={'raw_X':x,'raw_O':o,'centered_X':((x-model.fixed_x_center)/scale).clamp(-20,20),
          'centered_O':(o-model.fixed_o_center).clamp(-20,20),'logits':captured['logits'],'delta':delta}
        need(all(bool(torch.isfinite(v).all()) for v in trace.values()),'finite_score')
        need(torch.equal(delta,cap*torch.tanh(captured['logits']/cap)),'registered_cap_formula')
        traces.append(trace)
        # 原fixed与cap native共同使用此精确fp32运算顺序；旧alpha原样保留。
        corrections.append(alpha*delta*scale[0]);bounds.append(alpha*cap*float(scale[0]))
    costs=natural['coarse_scores'].to(device)+(1-gamma)*corrections[0]+gamma*corrections[1]
    need(costs.shape==(512,),'native_cost_shape512')
    coarse_dtype=str(natural['coarse_scores'].dtype);correction_dtypes=[str(v.dtype) for v in corrections]
    need(str(costs.dtype)==promoted_dtype(coarse_dtype,correction_dtypes),'native_cost_dtype_promotion')
    need(bool(torch.isfinite(costs).all()),'native_cost_finite')
    return {'cost':costs,'traces':traces,'gamma':gamma,'role_bounds':bounds,'total_bound':(1-gamma)*bounds[0]+gamma*bounds[1],'coarse_dtype':coarse_dtype,'correction_dtypes':correction_dtypes,'final_cost_dtype':str(costs.dtype)}

def load_models(module,runtime,entry,device,pins,frozen):
    fixed=entry['config'].startswith('fixed_');cap=5 if fixed else int(entry['config'].split('_')[0][3:])
    result=[]
    for role,augment in [('no_aug',False),('aug',True)]:
        ref=entry['checkpoints'][role];path=safe(ref['path'],True);parent=path.parent
        need(safe(ref['producer_summary'],True)==parent/'summary.json' and safe(ref['centers'],True)==parent/'centers.json' and safe(ref['center_rows'],True)==parent/'center_rows.jsonl','actual_role_producer_paths')
        pins.add(path,ref['sha256']);producer=pins.json(ref['producer_summary'],ref['producer_summary_sha256'])
        need(pins.raw(parent/'EXIT_CODE',frozen['input_sha256'][str(parent/'EXIT_CODE')]).decode().strip()=='0','checkpoint_producer_EXIT0')
        estimate=pins.json(ref['centers'],ref['centers_sha256']);pins.add(ref['center_rows'],ref['center_rows_sha256'])
        if fixed:
            model,scale,saved=runtime.fixed.load_checkpoint(path,device)
            need(saved['schema']==runtime.fixed.SCHEMA and saved['observation'] is True,'native_fixed_schema')
            need(producer['best']['epoch']==saved['epoch'] and producer['best']['alpha']==saved['alpha'],'native_fixed_best_producer')
            need(all(saved['args'].get(k)==v for k,v in runtime.fixed.BUDGET.items()),'native_fixed_full_original_budget')
            need(saved['args']['training_pool']=='mixed' and saved['args']['loss']=='listwise','native_fixed_original_loss')
        else:
            model,scale,saved=module.load_checkpoint(path,device,expected_cap=cap,require_final=True,runtime=runtime)
            need(saved['seed']==entry['seed'] and saved['role']==role and saved['epoch']==64 and saved['alpha']==1.,'cap_final64_role')
            for p,h in saved['input_sha256'].items():need(frozen['input_sha256'].get(p)==h,'cap_original_training_inputs_frozen')
        need(saved['args']['seed']==entry['seed'] and saved['args']['augment'] is augment,'checkpoint_seed_role')
        need(saved['epoch']==ref['epoch'] and saved['alpha']==ref['alpha'],'checkpoint_registered_epoch_alpha')
        need(saved['center_estimate_sha256']==ref['centers_sha256'] and estimate['row_audit_sha256']==ref['center_rows_sha256'],'bound_center_metadata')
        need(torch.equal(model.fixed_x_center.cpu(),torch.tensor(estimate['center33'],dtype=torch.float32)) and torch.equal(model.fixed_o_center.cpu(),torch.tensor(estimate['O_center3'],dtype=torch.float32)),'bound_center36_tensors')
        need(scale.shape==(33,) and bool(torch.isfinite(scale).all()) and bool((scale>0).all()),'scale33_finite')
        need(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),'model_tensor_finite')
        need(bool(saved['ranker_source_sha256']),'checkpoint_source_nonempty')
        for p,h in saved['ranker_source_sha256'].items():
            need(frozen['source_sha256'].get(p)==h,'checkpoint_scientific_source_frozen');pins.add(p,h)
        for value in model.parameters():value.requires_grad_(False)
        model.eval();result.append((model,scale,saved))
    for k in ('train_sha256','dual_train_sha256','cache_protocol','source_inputs_sha256'):
        need(result[0][2][k]==result[1][2][k],'matched_role_scientific_lineage:'+k)
    return result,cap

def run(args,out):
    begin=time.monotonic();started=utc()
    pins,protocol,bindings,frozen,entry=sealed_gate(args,[__file__,Path(__file__).with_name('cap_test_evaluation_common.py')])
    need(safe(bindings['evaluator']['path'],True)==safe(__file__,True) and bindings['evaluator']['sha256']==pins.values[str(safe(__file__,True))],'actual_registered_evaluator')
    # 明确准入返回之后才哈希当前cell输入和读取任何test payload。
    for path,digest in bindings['source_sha256'].items():pins.add(path,digest)
    # 全注册闭包由frozen专项审与outer始末核；leaf只打开该cell的cache+2weights。
    pins.add(args.cache,entry['cache_sha256'])
    reg=bindings['caches'][str(args.seed)]['test'][entry['pool']]
    producer=pins.json(reg['producer']['path'],reg['producer']['sha256'])
    need(pins.raw(reg['producer']['exit_path']).decode().strip()=='0','feature_producer_EXIT0')
    need(producer['feature_cache_sha256']==entry['cache_sha256'],'feature_payload_producer_SHA')
    module=load_module(bindings['cap_module']['path'])
    runtime=module.configure_runtime(args.source,bindings['fixed_module']['path'])
    for path in runtime.source_paths():need(frozen['source_sha256'].get(str(safe(path,True)))==pins.values.get(str(safe(path,True))),'runtime_exact_source_closure')
    torch.set_num_threads(args.threads);device=torch.device(args.device)
    data,actual=runtime.read_feature_cache(args.cache,known_sha=entry['cache_sha256'])
    need(actual==entry['cache_sha256'],'actual_cache_SHA')
    manifest=data['manifest'];verify_cache_manifest(manifest,reg['manifest'],producer,actual)
    need(manifest['cohort']=='test' and manifest['recall']==entry['pool'] and manifest['candidate_order_sha256']==ORDER_SHA and ordered_sha(data['smiles'])==ORDER_SHA,'actual_test_role_order')
    need(all(manifest[k]==v for k,v in frozen['fixed_test_identity'].items()),'actual_test_three_identity_SHA')
    need(manifest['records']==sum(TEST_COUNTS.values()) and manifest['base_queries']==TEST_COUNTS['observed'] and manifest['variants']==TEST_COUNTS,'actual_test1371_manifest')
    validate_loaded_test_content(data,frozen)
    models,cap=load_models(module,runtime,entry,device,pins,frozen)
    for model,scale,saved in models:runtime.base.check_science(saved,data)
    before=model_digest(models);rows=[];keys=set();durations=[]
    write(out/'protocol.json',dict(schema=SCHEMA,args=vars(args),argv=sys.argv,pid=os.getpid(),ppid=os.getppid(),started_utc=started,input_sha256=dict(pins.values),scope='exact_frozen_test_cell',test_read=True,training=False,selection_performed=False,promotion=False))
    with (out/'predictions.jsonl').open('x') as stream:
        for row in data['rows']:
            key=identity(row);need(key not in keys,'duplicate_test_identity');keys.add(key)
            runtime.base.validate_row(row,False)
            need(len(row['candidate_indices'])==512,'test_natural512_no_injected_positive')
            start=time.monotonic();natural=natural_row(row);gamma=route_gamma(row['query_observation'])
            prediction=outcome(row,score_natural(models,natural,gamma,device,cap))
            prediction['reranking_seconds']=time.monotonic()-start;durations.append(prediction['reranking_seconds'])
            stream.write(json.dumps(prediction,ensure_ascii=False,allow_nan=False)+'\n');rows.append(prediction)
    need(len(rows)==sum(TEST_COUNTS.values()) and dict(collections.Counter(r['query_identity']['variant'] for r in rows))==TEST_COUNTS,'full1371_predictions')
    after=model_digest(models);need(before==after,'model_state_mutated');pins.finish()
    result=dict(schema=SCHEMA,status='PASS',config=args.config,seed=args.seed,pool=entry['pool'],cohort='test',records=len(rows),
      cap=cap,cache_sha256=actual,query_sha256=manifest['query_sha256'],candidate_order_sha256=ORDER_SHA,
      fixed_test_identity=frozen['fixed_test_identity'],cache_manifest=manifest,producer_manifest=producer,
      identity_sha256=ordered_sha(sorted(identity_sha(r) for r in rows)),metrics={v:metrics([r for r in rows if r['query_identity']['variant']==v]) for v in VARIANTS},
      checkpoints={r['path']:r['sha256'] for r in entry['checkpoints'].values()},center_estimate_sha256=[r['centers_sha256'] for r in entry['checkpoints'].values()],
      model_state_before=before,model_state_after=after,input_sha256=pins.values,source_sha256=bindings['source_sha256'],inputs_unchanged=True,
      evaluator=bindings['evaluator'],frozen_selection={'path':str(safe(args.frozen_selection,True)),'sha256':pins.values[str(safe(args.frozen_selection,True))]},
      freeze_review={'path':str(safe(args.freeze_review,True)),'sha256':pins.values[str(safe(args.freeze_review,True))]},
      preselected=frozen['preselected'],selection_locked=True,predictions_sha256=sha(out/'predictions.jsonl',True),
      reranking_seconds_mean=float(np.mean(durations)),reranking_seconds_p95=float(np.quantile(durations,.95)),elapsed_seconds=time.monotonic()-begin,
      cost_semantics='Original native dtype natural512 costs with declared coarse/correction/final dtypes. Query timing includes audit traces; descriptive instrumented latency.',
      started_utc=started,finished_utc=utc(),pid=os.getpid(),ppid=os.getppid(),test_read=True,training=False,
      forward_bank_calls=0,external_api_calls=0,selection_performed=False,promotion=False,official_system_changed=False,independent_acceptance_status='unavailable')
    write(out/'summary.json',result);return result

def main():
    ap=argparse.ArgumentParser()
    for k in ('source','protocol','bindings','frozen-selection','freeze-review','cache','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--config',choices=CONFIGS,required=True);ap.add_argument('--seed',type=int,choices=SEEDS,required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();need(1<=args.threads<=16,'threads');out=safe(args.out,True);out.mkdir(parents=True,exist_ok=False)
    try:
        result=run(args,out);(out/'EXIT_CODE').write_text('0\n');print(json.dumps({'status':result['status'],'records':result['records']}))
    except BaseException as exc:
        write(out/'failure.json',{'error':repr(exc),'utc':utc()});(out/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()

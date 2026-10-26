"""功能概述：C10/C5固定最终checkpoint的natural512评估入口，只运行ranker评分。
natural_row先裁去训练追加第513正例并删除source/ID/target/mask；score_natural只收公开
观测、kind和候选物理输入，通过同一次row_delta的hook记录O/logit，不调用forward bank。
outcome在评分后读取标签，hit沿native epsilon计rank，miss原样保留cached outside rank。
run核validation_bindings源/缓存/producer/final64 alpha1两role/center闭包，输出原始cost、
trace哈希、完整query指标和始末模型/输入SHA。train仅允许fixed400 manifest中的dual行，
保留原other，不因新cost换比较对象。当前test硬拒；后续test须独立精确freeze适配。
CLI由root唯一operator启动，失败保留EXIT1，新输出目录不可覆盖。依cap_ranker模块与
cap_evaluation_common，不写trainer或旧科学目录。source只作输出分层，不进入评分。
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
from cap_evaluation_common import *

SCHEMA='cap_ranker_evaluation_20261024'
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

def model_digest(models):
    h=hashlib.sha256()
    for model,scale,saved in models:
        h.update(json.dumps({k:saved[k] for k in ('alpha','delta_cap','epoch','seed','role','center_estimate_sha256')},sort_keys=True).encode())
        for k,v in sorted(model.state_dict().items()):
            h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        h.update(scale.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.inference_mode()
def score_natural(models,natural,gamma,device,cap):
    need(set(natural)<=set(CANDIDATE_FIELDS)|{'kind','observation_input'},'score_identity_leak')
    need(gamma in (.25,.75) and len(models)==2,'score_roles')
    traces=[];corrections=[];bounds=[]
    for model,scale,saved in models:
        need(saved['delta_cap']==cap and saved['alpha']==1.0 and saved['epoch']==64,'final_fixed_role')
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
        centered_x=((x-model.fixed_x_center)/scale).clamp(-20,20)
        centered_o=(o-model.fixed_o_center).clamp(-20,20)
        trace={'raw_X':x,'raw_O':o,'centered_X':centered_x,'centered_O':centered_o,'logits':captured['logits'],'delta':delta}
        need(all(bool(torch.isfinite(v).all()) for v in trace.values()),'finite_score')
        expected=cap*torch.tanh(captured['logits']/cap)
        need(torch.equal(delta,expected),'registered_cap_formula')
        traces.append(trace);corrections.append(delta*scale[0]);bounds.append(cap*float(scale[0]))
    costs=natural['coarse_scores'].to(device)+(1-gamma)*corrections[0]+gamma*corrections[1]
    need(costs.shape==(512,) and costs.dtype==torch.float32 and bool(torch.isfinite(costs).all()),'finite_cost')
    return {'cost':costs,'traces':traces,'gamma':gamma,'role_bounds':bounds,
            'total_bound':(1-gamma)*bounds[0]+gamma*bounds[1]}

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
        targetcost=float(cost[pos]);rank=int((cost<=targetcost+EPS).sum());inrank=rank;target=int(ids[pos])
        other=int(order[order!=pos][0]);margin=targetcost-float(cost[other])
    else:
        rank=row['outside_pool_rank'];need(type(rank) is int and 512<rank<=CANDIDATES,'native_outside_rank')
    query={k:row[k] for k in IDENTITY}
    out=dict(query_identity=query,target_in_pool=hit,rank=rank,RR=1/rank,in_pool_target_rank=inrank,
        cached_outside_pool_rank=None if hit else rank,top1=rank<=1,top5=rank<=5,top10=rank<=10,
        target_candidate_index=target,winner_candidate_index=int(ids[order[0]]),target_cost=targetcost,
        margin=margin,mask_content_sha256=row['mask_content_sha256'],gamma=scored['gamma'],
        natural_candidate_indices=ids.tolist(),natural_candidate_costs=cost.tolist(),
        natural_candidate_coarse_costs=row['coarse_scores'][:512].cpu().tolist(),
        natural_candidate_trace_sha256=trace_hashes(scored),
        maximum_pairwise_correction_margin_bound=2*scored['total_bound'],
        role_absolute_correction_bounds=scored['role_bounds'],appended_training_positive=len(row['candidate_indices'])==513)
    validate_prediction(out);return out

def fixed_pair_evidence(row,scored,record,cap):
    need(row['target_in_coarse_pool'] is True,'fixed_train_pair_must_hit_dual')
    ids=row['candidate_indices'][:512].tolist();pos=row['target_position'];other=record['original_other_candidate_index']
    need(other in ids and other!=ids[pos],'fixed_other_natural')
    j=ids.index(other);target=ids[pos];cost=scored['cost'].detach().cpu()
    if 'target_candidate_index' in record:need(record['target_candidate_index']==target,'fixed_target')
    rawgap=float(row['coarse_scores'][pos].double()-row['coarse_scores'][j].double())
    need(abs(rawgap-record['original_coarse_gap'])<2e-5,'fixed_original_coarse_gap')
    traces=[]
    for role,t in zip(('no_aug','aug'),scored['traces']):
        delta=t['delta'].detach().cpu()
        traces.append(dict(role=role,target_delta=float(delta[pos]),original_other_delta=float(delta[j]),
          target_saturation=abs(float(delta[pos]))/cap,original_other_saturation=abs(float(delta[j]))/cap,
          target_saturated95=abs(float(delta[pos]))/cap>=.95,original_other_saturated95=abs(float(delta[j]))/cap>=.95))
    bound=2*scored['total_bound']
    facts=dict(original_other_candidate_index=other,target_candidate_index=target,
       original_coarse_gap=record['original_coarse_gap'],original_pairwise_bound=record['original_pairwise_bound'],
       original_bound_obstructed=record['original_bound_obstructed'],original_rank=record['original_rank'],
       new_pairwise_bound=bound,bound_released=rawgap<=bound,role_scale0=[b/cap for b in scored['role_bounds']],
       actual_fixed_pair_margin=float(cost[pos].double()-cost[j].double()),role_saturation=traces,
       target_coarse_cost=float(row['coarse_scores'][pos]),original_other_coarse_cost=float(row['coarse_scores'][j]),
       target_native_cost=float(cost[pos]),original_other_native_cost=float(cost[j]),
       boundary='Necessary magnitude bound before final fp32 rounding; released does not imply learned correction.')
    precision=fixed_pair_numerics(facts,scored['gamma'])
    need(precision['target_native_replay']==facts['target_native_cost'] and precision['original_other_native_replay']==facts['original_other_native_cost'],'native_fixed_pair_fp32_replay')
    facts.update(precision)
    facts['rounding_boundary']='FP64 real-arithmetic decomposition is explanatory; native evaluated fp32 costs/ranks remain unchanged. Residual is recorded for this same fixed opponent.'
    return facts


def load_module(path):
    spec=importlib.util.spec_from_file_location('_cap_eval_runtime',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def load_models(module,runtime,bindings,cap,seed,device,pins):
    result=[]
    for role,augment in [('no_aug',False),('aug',True)]:
        r=bindings['roles'][str(cap)][str(seed)][role]
        pins.add(r['checkpoint'],r['checkpoint_sha256'])
        need(safe(r['producer_summary'])==safe(Path(r['checkpoint']).parent/'summary.json'),'actual_producer_path')
        need(safe(r['centers'])==safe(Path(r['checkpoint']).parent/'centers.json'),'actual_center_path')
        need(safe(r['center_rows'])==safe(Path(r['checkpoint']).parent/'center_rows.jsonl'),'actual_center_rows_path')
        producer=pins.json(r['producer_summary'],r['producer_summary_sha256'])
        need(pins.raw(Path(r['checkpoint']).parent/'EXIT_CODE').decode().strip()=='0','training_producer_EXIT0')
        pins.add(r['centers'],r['centers_sha256']);pins.add(r['center_rows'],r['center_rows_sha256'])
        model,scale,saved=module.load_checkpoint(r['checkpoint'],device,expected_cap=cap,require_final=True,runtime=runtime)
        need(saved['seed']==seed and saved['role']==role,'checkpoint_seed_role')
        need(saved['epoch']==64 and saved['alpha']==1.0 and saved['delta_cap']==cap,'final64_alpha1_cap')
        need(saved['center_estimate_sha256']==r['centers_sha256'],'bound_center')
        train=bindings['caches'][str(seed)]['train']
        need(saved['train_sha256']==train['single']['sha256'] and saved['dual_train_sha256']==train['dual']['sha256'],'bound_training_caches')
        need(producer['epoch']==64 and producer['alpha']==1.0,'producer_final')
        for v in model.parameters():v.requires_grad_(False)
        model.eval();result.append((model,scale,saved))
    for key in ('train_sha256','dual_train_sha256','cache_protocol','source_inputs_sha256'):
        need(result[0][2][key]==result[1][2][key],'matched_role_science:'+key)
    return result

def run(args,out):
    start=time.monotonic();started=utc()
    need(args.cohort in ('original_val','tune','train'),'test_requires_separate_frozen_adapter')
    need(args.cap in CAPS and args.seed in SEEDS and args.pool in POOLS,'registered_role')
    pins=Pins();pins.add(__file__);pins.add(Path(__file__).with_name('cap_evaluation_common.py'));protocol=validate_protocol(pins.json(args.protocol));bindings=pins.json(args.bindings)
    need(bindings['schema']=='cap10_execution_bindings_20261024','validation_bindings_schema')
    need(safe(args.source)==safe(bindings['source_checkout']),'source_checkout_binding')
    moduleinfo=bindings['cap_module'];pins.add(moduleinfo['path'],moduleinfo['sha256'])
    pins.add(__file__);pins.add(Path(__file__).with_name('cap_evaluation_common.py'))
    need(bool(bindings['source_sha256']),'nonempty_source_closure')
    for path,digest in bindings['source_sha256'].items():pins.add(path,digest)
    cache=bindings['caches'][str(args.seed)][args.cohort][args.pool]
    need(safe(args.cache)==safe(cache['path']),'cache_role_path');pins.add(args.cache,cache['sha256'])
    producer_path=cache['producer']['path'];producer=pins.json(producer_path,cache['producer']['sha256'])
    need(producer['feature_cache_sha256']==cache['sha256'],'feature_producer_sha')
    if 'exit_path' in cache['producer']:need(pins.raw(cache['producer']['exit_path']).decode().strip()=='0','cache_producer_exit0')
    module=load_module(moduleinfo['path'])
    fixed=bindings.get('fixed_module',{'path':str(Path(bindings['evidence_root'])/'h_pool_fixed_center_ranker.py')})
    pins.add(fixed['path'],fixed.get('sha256'))
    runtime=module.configure_runtime(args.source,fixed['path'])
    for path in runtime.source_paths():
        need(str(safe(path)) in pins.values,'runtime_source_not_bound:'+str(path))
    torch.set_num_threads(args.threads);device=torch.device(args.device)
    data,actual=runtime.read_feature_cache(args.cache,known_sha=cache['sha256'])
    need(actual==cache['sha256'],'cache_reader_SHA')
    verify_cache_manifest(data['manifest'],cache['manifest'],producer,actual)
    manifest=data['manifest'];need(manifest['candidate_count']==CANDIDATES and manifest['pool_size']==512 and manifest['candidate_order_sha256']==ORDER_SHA,'cache_universe')
    need(manifest['cohort']==args.cohort and manifest['recall']==args.pool,'cache_role')
    need(ordered_sha(data['smiles'])==ORDER_SHA,'candidate_order')
    models=load_models(module,runtime,bindings,args.cap,args.seed,device,pins)
    for model,scale,saved in models:runtime.base.check_science(saved,data)
    before=model_digest(models);selection=None
    if args.cohort=='train':
        need(args.pool=='dual' and args.query_manifest is not None,'train_only_fixed_dual_manifest')
        info=bindings['fixed_train_pairs'];need(safe(args.query_manifest)==safe(info['path']),'fixed_pairs_path')
        need(info['sha256']==protocol['mechanism']['fixed_train_pairs']['sha256'],'protocol_fixed_pair_SHA')
        selection=pins.json(args.query_manifest,info['sha256'])
        need(selection['schema']=='cap10_fixed_train_pairs_20261024','fixed_pairs_schema')
        need(len(selection['records'])==400,'fixed400')
        selected={identity(z['query_identity']):z for z in selection['records'] if z['seed']==args.seed}
        need(len(selected)==200,'fixed200_per_seed')
    else:need(args.query_manifest is None,'no_validation_filter')
    keys=set();rows=[];durations=[];seen_selected=set()
    write(out/'protocol.json',dict(schema=SCHEMA,args=vars(args),argv=sys.argv,pid=os.getpid(),ppid=os.getppid(),started_utc=started,input_sha256=dict(pins.values),scope='train_validation_ranker_only',training=False,test_read=False))
    with (out/'predictions.jsonl').open('x') as stream:
        for row in data['rows']:
            key=identity(row);need(key not in keys,'duplicate_query');keys.add(key)
            if selection is not None and key not in selected:continue
            runtime.base.validate_row(row,args.cohort=='train')
            begin=time.monotonic();natural=natural_row(row);gamma=route_gamma(row['query_observation'])
            scored=score_natural(models,natural,gamma,device,args.cap);prediction=outcome(row,scored)
            if selection is not None:
                prediction['fixed_train_pair']=fixed_pair_evidence(row,scored,selected[key],args.cap);seen_selected.add(key)
            duration=time.monotonic()-begin;prediction['reranking_seconds']=duration;durations.append(duration)
            stream.write(json.dumps(prediction,ensure_ascii=False,allow_nan=False)+'\n');rows.append(prediction)
    need(selection is None or seen_selected==set(selected),'all_fixed_pairs_scored')
    if selection is None:
        need(len(rows)==manifest['records'] and dict(collections.Counter(r['query_identity']['variant'] for r in rows))==manifest['variants'],'full_cache_evaluation')
        need(sum(r['query_identity']['variant']=='observed' for r in rows)==COHORT_OBSERVED[args.cohort],'full_observed_cohort')
    after=model_digest(models);need(before==after,'model_state_mutated')
    pins.finish()
    result=dict(schema=SCHEMA,status='PASS',cap=args.cap,seed=args.seed,pool=args.pool,cohort=args.cohort,
      scope='fixed_train_pairs_mechanism_only' if selection else 'complete_train_validation_evaluation',
      query_sha256=manifest['query_sha256'],candidate_order_sha256=ORDER_SHA,cache_sha256=cache['sha256'],
      cache_manifest=manifest,producer_manifest=producer,records=len(rows),identity_sha256=ordered_sha(sorted(identity_sha(r) for r in rows)),
      metrics={v:metrics([r for r in rows if r['query_identity']['variant']==v]) for v in VARIANTS},
      started_utc=started,finished_utc=utc(),pid=os.getpid(),ppid=os.getppid(),input_sha256=pins.values,
      source_sha256=bindings['source_sha256'],checkpoints={r['checkpoint']:r['checkpoint_sha256'] for r in bindings['roles'][str(args.cap)][str(args.seed)].values()},
      model_state_before=before,model_state_after=after,inputs_unchanged=True,
      predictions_sha256=sha(out/'predictions.jsonl'),reranking_seconds_mean=float(np.mean(durations)),
      reranking_seconds_p95=float(np.quantile(durations,.95)),elapsed_seconds=time.monotonic()-start,
      cost_semantics='Original fp32 natural512 cost arrays preserved. Per-query durations include score/outcome/trace audit instrumentation and are descriptive, not bare model latency.',
      training=False,test_read=False,forward_bank_calls=0,external_api_calls=0,selection_performed=False,promotion=False)
    write(out/'summary.json',result);return result

def main():
    ap=argparse.ArgumentParser()
    for k in ('source','protocol','bindings','cache','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--cap',type=int,choices=CAPS,required=True);ap.add_argument('--seed',type=int,choices=SEEDS,required=True)
    ap.add_argument('--pool',choices=POOLS,required=True);ap.add_argument('--cohort',choices=('original_val','tune','train'),required=True)
    ap.add_argument('--query-manifest');ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();need(1<=args.threads<=16,'threads')
    out=safe(args.out);out.mkdir(parents=True,exist_ok=False)
    try:
        result=run(args,out);(out/'EXIT_CODE').write_text('0\n');print(json.dumps({'status':result['status'],'records':result['records']}))
    except BaseException as exc:
        write(out/'failure.json',{'error':repr(exc),'utc':utc()});(out/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()

"""功能概述：从本轮完整ranker评估产物核共同候选不变性和固定400/87训练pair机制。
common_invariance逐query/候选ID比较同cap同权重single/dual的原始cost及两role traceSHA，
不跨cap要求分数相同。fixed_pair_analysis仅消费预注册原other，分别报告target/other
abs(delta_C)/C及>=.95，固定pair margin、natural rank、必要上界解除；不能把解除解释为
已学会纠正。distribution保留精确线性分位数；小组空分母返回null、不补0。
run只读root唯一queue、实际回执、bindings及完整评估JSON，始末核SHA；无模型import、
无bankforward、无test、无选型。selection不消费本文件任何统计。与evaluate_cap_ranker的
fixed_train_pair字段和summarize_cap_validation共用cap_evaluation_common契约。
"""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cap_evaluation_common as _actual_contract
if Path(_actual_contract.__file__).resolve()!=Path(__file__).with_name('cap_evaluation_common.py').resolve():
    raise ImportError('native_contract_module_path_mismatch')
from cap_evaluation_common import *

SCHEMA='cap_mechanism_analysis_20261024'

def distribution(values):
    known=[x for x in values if x is not None]
    need(all(finite(x) for x in known),'finite_distribution')
    v=sorted(known)
    def q(p):
        if not v:return None
        z=(len(v)-1)*p;i=int(z);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(z-i)
    return dict(available=len(v),unknown=len(values)-len(v),mean=math.fsum(v)/len(v) if v else None,
      quantiles={k:q(p) for k,p in [('min',0),('p10',.1),('p25',.25),('median',.5),('p75',.75),('p90',.9),('p95',.95),('max',1)]})

def common_invariance(single,dual):
    left={identity(r):r for r in single};right={identity(r):r for r in dual}
    need(len(left)==len(single) and set(left)==set(right),'paired_full_identities')
    candidates=0;mincommon=512;maxcommon=0
    for key,a in left.items():
        b=right[key]
        need(all(a[field]==b[field] for field in ('coarse_dtype','correction_dtypes','final_cost_dtype')),'paired_native_dtypes')
        need(a['gamma']==b['gamma'] and a['mask_content_sha256']==b['mask_content_sha256'],'paired_observation_route')
        lookup=[{cid:i for i,cid in enumerate(r['natural_candidate_indices'])} for r in (a,b)]
        common=set(lookup[0])&set(lookup[1]);need(bool(common),'common_candidates_nonempty')
        candidates+=len(common);mincommon=min(mincommon,len(common));maxcommon=max(maxcommon,len(common))
        for cid in common:
            i,j=lookup[0][cid],lookup[1][cid]
            need(a['natural_candidate_coarse_costs'][i]==b['natural_candidate_coarse_costs'][j],'shared_candidate_coarse_changed')
            need(a['natural_candidate_costs'][i]==b['natural_candidate_costs'][j],'shared_candidate_cost_changed')
            for role in range(2):
                need(a['natural_candidate_trace_sha256'][role][i]==b['natural_candidate_trace_sha256'][role][j],'shared_candidate_trace_changed')
        oa=sorted(common,key=lambda cid:(a['natural_candidate_costs'][lookup[0][cid]],cid))
        ob=sorted(common,key=lambda cid:(b['natural_candidate_costs'][lookup[1][cid]],cid))
        need(oa==ob,'shared_candidate_order_changed')
    return dict(query_pairs=len(left),common_candidate_pairs=candidates,min_common_candidates=mincommon if left else None,
      max_common_candidates=maxcommon if left else None,all_raw_centered_logit_delta_traces_equal=True,
      all_coarse_and_final_cost_equal=True,all_relative_orders_equal=True,
      scope='Same query, candidate and cap-specific role weights; compares all natural512 common candidates.',
      causal_boundary='Invariant shared scores support a candidate-composition explanation within this fixed scorer; they do not identify the cause of different-query conditional means.')

def fixed_pair_analysis(records,per_cap):
    origin={(r['seed'],identity(r['query_identity'])):r for r in records}
    need(len(origin)==len(records)==400,'exact400_origin')
    need(collections.Counter(r['seed'] for r in records)=={193:200,211:200},'fixed200_perseed')
    need(sum(r['original_bound_obstructed'] for r in records)==87,'fixed87_origin')
    aligned={}
    immutable=('original_other_candidate_index','target_candidate_index','original_coarse_gap','original_pairwise_bound','original_bound_obstructed','original_rank')
    for cap in CAPS:
        entries=per_cap[cap];index={(r['seed'],identity(r['query_identity'])):r for r in entries}
        need(len(index)==len(entries) and set(index)==set(origin),'exact_fixed_train_identity')
        for key,row in index.items():
            evidence=row['fixed_train_pair'];old=origin[key]
            need(all(evidence[k]==old[k] for k in immutable),'fixed_original_pair_unchanged')
            need(evidence['bound_released'] is (old['original_coarse_gap']<=evidence['new_pairwise_bound']),'bound_release_arithmetic')
            need(row['target_in_pool'] is True and row['target_candidate_index']==old['target_candidate_index'],'fixed_dual_target')
            need(len(evidence['role_scale0'])==2 and all(finite(x) and x>0 for x in evidence['role_scale0']),'fixed_role_scale0')
            need(row['gamma'] in (.25,.75),'fixed_gamma')
            weights=[1-row['gamma'],row['gamma']]
            need(abs(evidence['new_pairwise_bound']-2*cap*math.fsum(w*x for w,x in zip(weights,evidence['role_scale0'])))<1e-12,'fixed_bound_scale_decomposition')
            need(len(evidence['role_saturation'])==2 and {t['role'] for t in evidence['role_saturation']}=={'no_aug','aug'},'saturation_roles')
            for t in evidence['role_saturation']:
                for target in ('target','original_other'):
                    need(abs(t[target+'_saturation']-abs(t[target+'_delta'])/cap)<1e-15,'saturation_ratio')
                    need(t[target+'_saturated95'] is (t[target+'_saturation']>=.95),'saturation_threshold')
            correction=math.fsum(w*scale*(next(t for t in evidence['role_saturation'] if t['role']==role)['target_delta']-next(t for t in evidence['role_saturation'] if t['role']==role)['original_other_delta']) for w,scale,role in zip(weights,evidence['role_scale0'],('no_aug','aug')))
            precision=fixed_pair_numerics(evidence,row['gamma'])
            need(precision['target_native_replay']==evidence['target_native_cost'] and precision['original_other_native_replay']==evidence['original_other_native_cost'],'fixed_pair_native_fp32_replay')
            need(evidence['target_native_cost']==row['natural_candidate_costs'][row['natural_candidate_indices'].index(old['target_candidate_index'])] and evidence['original_other_native_cost']==row['natural_candidate_costs'][row['natural_candidate_indices'].index(old['original_other_candidate_index'])],'native_replay_matches_evaluated_cost')
            need(evidence['actual_fixed_pair_margin']==precision['native_margin'],'actual_native_pair_margin')
            need(all(evidence[k]==v for k,v in precision.items()),'explicit_pair_precision_evidence')
            need(abs((evidence['target_coarse_cost']-evidence['original_other_coarse_cost'])-old['original_coarse_gap'])<2e-5,'fixed_coarse_pair_identity')
            allowance=8*math.ulp(max(abs(correction),evidence['new_pairwise_bound'],1.))
            need(abs(correction)<=evidence['new_pairwise_bound']+allowance,'real_correction_within_cap_bound')
            need(abs(evidence['actual_fixed_pair_margin']-(row['natural_candidate_costs'][row['natural_candidate_indices'].index(old['target_candidate_index'])]-row['natural_candidate_costs'][row['natural_candidate_indices'].index(old['original_other_candidate_index'])]))<1e-12,'fixed_pair_margin_cost')
        aligned[cap]=index
    for key,old in origin.items():
        a=aligned[5][key]['fixed_train_pair'];b=aligned[10][key]['fixed_train_pair']
        need(abs(a['new_pairwise_bound']-old['original_pairwise_bound'])<1e-12,'C5_preserved_original_bound')
        need(a['target_coarse_cost']==b['target_coarse_cost'] and a['original_other_coarse_cost']==b['original_other_coarse_cost'],'same_fixed_pair_raw_coarse_acrosscap')
        need(a['role_scale0']==b['role_scale0'] and abs(b['new_pairwise_bound']-2*a['new_pairwise_bound'])<1e-12,'paired_C10_bound_and_scale')
    reports={}
    for label in ('all400','original87'):
        keys=[key for key,v in origin.items() if label=='all400' or v['original_bound_obstructed']]
        perseed={}
        for seed in SEEDS:
            sk=[k for k in keys if k[0]==seed];n=len(sk);capstats={}
            for cap in CAPS:
                rr=[aligned[cap][k] for k in sk];fp=[r['fixed_train_pair'] for r in rr]
                saturation={}
                for role in ('no_aug','aug'):
                    items=[next(t for t in x['role_saturation'] if t['role']==role) for x in fp]
                    saturation[role]={which:dict(distribution=distribution([t[which+'_saturation'] for t in items]),
                      saturated95=sum(t[which+'_saturated95'] for t in items),denominator=n,
                      fraction=ratio(sum(t[which+'_saturated95'] for t in items),n)) for which in ('target','original_other')}
                capstats[str(cap)]=dict(queries=n,metrics=metrics(rr),rank=distribution([r['rank'] for r in rr]),
                    rank_minus_original=distribution([r['rank']-origin[k]['original_rank'] for k,r in zip(sk,rr)]),
                    RR_minus_original=distribution([r['RR']-1/origin[k]['original_rank'] for k,r in zip(sk,rr)]),
                    fixed_pair_margin=distribution([x['actual_fixed_pair_margin'] for x in fp]),
                    real_arithmetic_fixed_pair_margin=distribution([x['real_arithmetic_margin'] for x in fp]),
                    native_minus_real_rounding_residual=distribution([x['native_minus_real_margin'] for x in fp]),
                    margin_minus_original=distribution([x['actual_fixed_pair_margin']-origin[k]['original_final_gap'] for k,x in zip(sk,fp)]),
                    bound_released_count=sum(x['bound_released'] for x in fp),bound_release_denominator=n,
                    bound_release_fraction=ratio(sum(x['bound_released'] for x in fp),n),saturation=saturation)
            delta={}
            for metric in ('MRR','top1','top5','top10'):
                a,b=capstats['10']['metrics'][metric],capstats['5']['metrics'][metric]
                delta[metric]=a-b if a is not None and b is not None else None
            perseed[str(seed)]=dict(queries=n,caps=capstats,C10_minus_C5=delta,
                paired_rank_delta=distribution([aligned[10][k]['rank']-aligned[5][k]['rank'] for k in sk]),
                paired_fixed_margin_delta=distribution([aligned[10][k]['fixed_train_pair']['actual_fixed_pair_margin']-aligned[5][k]['fixed_train_pair']['actual_fixed_pair_margin'] for k in sk]))
        equal={}
        for metric in ('MRR','top1','top5','top10'):
            v=[perseed[str(s)]['C10_minus_C5'][metric] for s in SEEDS]
            equal[metric]=math.fsum(v)/2 if all(x is not None for x in v) else None
        reports[label]=dict(query_seed_pairs=len(keys),per_seed=perseed,equal_two_seed_mean_C10_minus_C5=equal,
            total_bound_release_count={str(cap):sum(aligned[cap][k]['fixed_train_pair']['bound_released'] for k in keys) for cap in CAPS},
            interpretation='Fixed sample association only. Counts sum query-seed pairs; ratios are per-seed then equal mean. Bounds are necessary magnitude constraints, not promises of learned rank improvement.')
    return reports

def run(args,out):
    pins=Pins();pins.add(__file__);pins.add(Path(__file__).with_name('cap_evaluation_common.py'));pins.add(Path(__file__).with_name('summarize_cap_validation.py'));protocol=validate_protocol(pins.json(args.protocol));bindings=pins.json(args.bindings)
    need(protocol['mechanism']['saturation_threshold']==.95 and protocol['mechanism']['selection_input'] is False,'registered_mechanism')
    fixed=bindings['fixed_train_pairs'];need(fixed['sha256']==protocol['mechanism']['fixed_train_pairs']['sha256'],'protocol_fixed_pair_SHA');origin=pins.json(fixed['path'],fixed['sha256'])
    need(origin['schema']=='cap10_fixed_train_pairs_20261024','fixed_pair_schema')
    import summarize_cap_validation as _actual_summary
    need(Path(_actual_summary.__file__).resolve()==Path(__file__).with_name('summarize_cap_validation.py').resolve(),'native_summary_module_path_mismatch')
    load_matrix=_actual_summary.load_matrix
    validation,_=load_matrix(pins,args.validation_queue,args.validation_run,args.bindings,args.protocol,'validation')
    mechanism,_=load_matrix(pins,args.mechanism_queue,args.mechanism_run,args.bindings,args.protocol,'mechanism')
    invariance={}
    for cap in CAPS:
        for seed in SEEDS:
            for cohort in ('original_val','tune'):
                a0=validation[(cap,seed,'single',cohort)];b0=validation[(cap,seed,'dual',cohort)]
                a=read_evaluation(pins,a0[1]['args']['out'],cap,seed,'single',cohort,args.bindings,args.protocol,True)
                b=read_evaluation(pins,b0[1]['args']['out'],cap,seed,'dual',cohort,args.bindings,args.protocol,True)
                need(a[0]['checkpoints']==b[0]['checkpoints'],'shared_weights')
                invariance[f'C{cap}/s{seed}/{cohort}']=common_invariance(a[2],b[2])
    pairs={}
    for cap in CAPS:
        pairs[cap]=[]
        for seed in SEEDS:
            rows=mechanism[(cap,seed,'dual','train')][2]
            pairs[cap].extend(dict(r,seed=seed) for r in rows)
    results=fixed_pair_analysis(origin['records'],pairs)
    pins.add(__file__);pins.add(Path(__file__).with_name('cap_evaluation_common.py'))
    pins.finish()
    result=dict(schema=SCHEMA,status='PASS',source_sha256={str(Path(__file__).resolve()):pins.values[str(Path(__file__).resolve())]},
        input_sha256=pins.values,deferred_tensor_sha256=pins.deferred,inputs_unchanged=True,fixed_train_pairs=fixed,
        common_candidate_invariance=invariance,fixed_train_pair_analysis=results,
        selection_input=False,training=False,test_read=False,model_calls=0,forward_bank_calls=0,external_api_calls=0,promotion=False)
    write(out/'summary.json',result);return result

def main():
    ap=argparse.ArgumentParser()
    for k in ('protocol','bindings','validation-queue','validation-run','mechanism-queue','mechanism-run','out'):ap.add_argument('--'+k,required=True)
    args=ap.parse_args();out=safe(args.out);out.mkdir(parents=True,exist_ok=False)
    try:run(args,out);(out/'EXIT_CODE').write_text('0\n')
    except BaseException as exc:write(out/'failure.json',{'error':repr(exc),'utc':utc()});(out/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()

"""功能概述：独立重算C5/C10 train/validation结果的纯算术，不导入评估作者模块。
覆盖自然512原生dtype的epsilon排名、两seed条件比率、五source宏平均、严格预选门槛、
同权重共同候选成本/trace不变性及固定400/87 pair的必要上界与饱和定义。
仅处理调用者已经核验身份与来源的Python记录，不加载模型、tensor或远端文件。
该模块供实际结果独审checker调用，不能启动训练、验证、test或决定新增实验。
"""
import math
import struct

SEEDS = (193,211)
SOURCES = ('HMDB','NMRexp','WanJuan','nmrshiftdb2_2018','nmrshiftdb2_2024')
VARIANTS = ('observed','frequency_crop','peak_dropout')
FIELDS = ('query_order','variant','source','record_id','smiles','group')
BASELINE = 'fixedcenter_mixed_listwise_single'
REFERENCE = {'MRR':.5848916643734858,'top1':.5163690476190477,
 'frequency_crop_MRR':.26751386238989017,'peak_dropout_MRR':.2196237084887636,
 'macro_MRR':.43440429414067594}

def require(ok, label):
    if not ok: raise ValueError(label)

def finite(x):
    return type(x) in (int,float) and math.isfinite(x)

def identity(row):
    return tuple(row.get('query_identity',row)[k] for k in FIELDS)

def cast(x,dtype):
    require(dtype in ('torch.float32','torch.float64'), 'native_dtype')
    return struct.unpack('f',struct.pack('f',x))[0] if dtype == 'torch.float32' else float(x)

def native_pair_cost(coarse,delta0,delta1,scale0,scale1,gamma,coarse_dtype):
    """新cap alpha=1；两校正先各自按fp32乘/加权，再依coarse dtype逐次相加。"""
    require(coarse_dtype in ('torch.float32','torch.float64') and gamma in (.25,.75),'native_pair_dtype_route')
    require(all(finite(v) and cast(v,'torch.float32')==v for v in (delta0,delta1,scale0,scale1)),'saved_fp32_delta_scale')
    first=cast(cast(delta0*scale0,'torch.float32')*(1-gamma),'torch.float32')
    second=cast(cast(delta1*scale1,'torch.float32')*gamma,'torch.float32')
    return cast(cast(cast(coarse,coarse_dtype)+first,coarse_dtype)+second,coarse_dtype)

def audit_row(row,dtype):
    """按实际native成本dtype重算worst-tie epsilon rank，miss保持cache全库排名。"""
    identity(row)
    ids=row['natural_candidate_indices']; costs=row['natural_candidate_costs']
    require(len(ids)==len(set(ids))==len(costs)==512,'natural512')
    require(all(type(i) is int and 0<=i<114982 for i in ids),'candidate_indices')
    require(all(finite(x) and cast(x,dtype)==x for x in costs),'finite_native_costs')
    require(type(row['target_in_pool']) is bool,'membership_bool')
    if row['target_in_pool']:
        require(row['target_candidate_index'] in ids,'natural_target_membership')
        target=costs[ids.index(row['target_candidate_index'])]
        # PyTorch converts the Python EPS scalar to the tensor dtype before add.
        threshold=cast(cast(target,dtype)+cast(1e-7,dtype),dtype)
        rank=sum(c<=threshold for c in costs)
        require(row['in_pool_target_rank']==rank and row['target_cost']==target,'in_pool_cost_rank')
    else:
        rank=row['cached_outside_pool_rank']
        require(type(rank) is int and 512<rank<=114982,'native_miss_fallback')
        require(row['in_pool_target_rank'] is None,'miss_conditional_rank')
    require(type(row['rank']) is int and row['rank']==rank,'rank_recomputed')
    require(finite(row['RR']) and abs(row['RR']-1/rank)<1e-15,'RR_recomputed')
    for k in (1,5,10): require(row['top'+str(k)] is (rank<=k),'topK')
    return row

def metrics(rows):
    n=len(rows)
    if not n:return dict(queries=0,MRR=None,top1=None,top5=None,top10=None,recall512=None,in_pool_MRR=None)
    hits=[r for r in rows if r['target_in_pool']]
    return dict(queries=n,MRR=math.fsum(1/r['rank'] for r in rows)/n,
      top1=sum(r['rank']<=1 for r in rows)/n,top5=sum(r['rank']<=5 for r in rows)/n,
      top10=sum(r['rank']<=10 for r in rows)/n,recall512=len(hits)/n,
      in_pool_MRR=math.fsum(1/r['rank'] for r in hits)/len(hits) if hits else None)

def equal_seed_metrics(per_seed):
    require(set(per_seed)==set(SEEDS),'two_seeds')
    m=[metrics(per_seed[s]) for s in SEEDS]
    require(m[0]['queries']==m[1]['queries'],'matched_seed_denominators')
    return {k:m[0][k] if k=='queries' else (math.fsum(v[k] for v in m)/2 if all(v[k] is not None for v in m) else None) for k in m[0]}

def validation_signals(merged,expected_observed=1344):
    require(set(merged)==set(SEEDS),'two_seed_signals')
    parts={}
    for seed,rows in merged.items():
        require(len({identity(r) for r in rows})==len(rows),'merged_identity_unique')
        observed=[r for r in rows if r['query_identity']['variant']=='observed']
        require(len(observed)==expected_observed,'merged_observed_count')
        source_m=[metrics([r for r in observed if r['query_identity']['source']==s]) for s in SOURCES]
        require(all(x['queries'] for x in source_m),'five_sources')
        stress={v:metrics([r for r in rows if r['query_identity']['variant']==v]) for v in VARIANTS[1:]}
        require(all(x['queries'] for x in stress.values()),'stress_denominators')
        parts[str(seed)]=dict(MRR=metrics(observed)['MRR'],top1=metrics(observed)['top1'],
          frequency_crop_MRR=stress['frequency_crop']['MRR'],peak_dropout_MRR=stress['peak_dropout']['MRR'],
          macro_MRR=math.fsum(x['MRR'] for x in source_m)/5)
    return {k:math.fsum(parts[str(s)][k] for s in SEEDS)/2 for k in REFERENCE},parts

def selection(candidate,control,baseline=REFERENCE):
    require(baseline==REFERENCE,'registered_reference')
    require(all(finite(m[k]) for m in (candidate,control,baseline) for k in REFERENCE),'finite_selection_signals')
    gates={'MRR_gt_fixed_single':candidate['MRR']>baseline['MRR']+1e-12,
      'MRR_gt_matched_C5dual':candidate['MRR']>control['MRR']+1e-12}
    gates.update({k:candidate[k]>=baseline[k]-1e-12 for k in ('top1','frequency_crop_MRR','peak_dropout_MRR','macro_MRR')})
    eligible=all(gates.values())
    return dict(gates=gates,eligible=eligible,preselected='cap10_dual' if eligible else BASELINE)

def common_invariance(single,dual):
    left={identity(r):r for r in single};right={identity(r):r for r in dual}
    require(len(left)==len(single) and len(right)==len(dual) and left.keys()==right.keys(),'paired_full_identity')
    counts=[]
    for key,a in left.items():
        b=right[key]
        require(a['gamma']==b['gamma'] and a['mask_content_sha256']==b['mask_content_sha256'],'same_observation')
        ia={cid:i for i,cid in enumerate(a['natural_candidate_indices'])};ib={cid:i for i,cid in enumerate(b['natural_candidate_indices'])}
        common=ia.keys()&ib.keys(); require(bool(common),'nonempty_common')
        counts.append(len(common))
        for cid in common:
            i,j=ia[cid],ib[cid]
            require(a['natural_candidate_coarse_costs'][i]==b['natural_candidate_coarse_costs'][j],'common_coarse')
            require(a['natural_candidate_costs'][i]==b['natural_candidate_costs'][j],'common_final_cost')
            for role in range(2):require(a['natural_candidate_trace_sha256'][role][i]==b['natural_candidate_trace_sha256'][role][j],'common_trace')
    return dict(query_pairs=len(counts),common_candidate_pairs=sum(counts),min_common_candidates=min(counts) if counts else None,
      max_common_candidates=max(counts) if counts else None)

def fixed_pair_checks(records,per_cap):
    """独立核固定样本与必要上界；不把解除上界阻碍称作纠正已学会。"""
    origin={(r['seed'],identity(r['query_identity'])):r for r in records}
    require(len(origin)==len(records)==400,'fixed400')
    require({s:sum(k[0]==s for k in origin) for s in SEEDS}=={193:200,211:200},'fixed200_each')
    require(sum(r['original_bound_obstructed'] for r in records)==87,'fixed87')
    immutable=('original_other_candidate_index','target_candidate_index','original_coarse_gap','original_pairwise_bound','original_bound_obstructed','original_rank')
    indexed={}
    for cap in (5,10):
        entries=per_cap[cap]; index={(r['seed'],identity(r)):r for r in entries}
        require(len(index)==len(entries)==400 and index.keys()==origin.keys(),'fixed_pair_identity_exact')
        for key,r in index.items():
            old=origin[key];e=r['fixed_train_pair']
            require(all(e[k]==old[k] for k in immutable),'original_fixed_other_unchanged')
            require(r['target_in_pool'] is True and r['target_candidate_index']==old['target_candidate_index'],'fixed_natural_target')
            scales=e['role_scale0'];gamma=r['gamma']
            require(gamma in (.25,.75) and len(scales)==2 and all(finite(x) and x>0 for x in scales),'role_scale_gamma')
            bound=2*cap*((1-gamma)*scales[0]+gamma*scales[1])
            require(abs(e['new_pairwise_bound']-bound)<1e-12,'necessary_bound')
            require(e['bound_released'] is (old['original_coarse_gap']<=bound),'bound_release')
            roles={v['role']:v for v in e['role_saturation']};require(set(roles)=={'no_aug','aug'},'saturation_roles')
            for role,t in roles.items():
                for which in ('target','original_other'):
                    ratio=abs(t[which+'_delta'])/cap
                    require(abs(t[which+'_saturation']-ratio)<1e-15,'saturation_ratio')
                    require(t[which+'_saturated95'] is (ratio>=.95),'saturation95')
                    require(finite(t[which+'_delta']) and abs(t[which+'_delta'])<=cap,'cap_delta_bound')
            ids=r['natural_candidate_indices'];costs=r['natural_candidate_costs']
            margin=costs[ids.index(old['target_candidate_index'])]-costs[ids.index(old['original_other_candidate_index'])]
            require(e['actual_fixed_pair_margin']==margin,'fixed_native_margin')
            require(e['coarse_dtype']==r['coarse_dtype'] and e['correction_dtypes']==r['correction_dtypes']==['torch.float32','torch.float32'] and e['final_cost_dtype']==r['final_cost_dtype']==r['coarse_dtype'],'fixed_pair_declared_native_dtypes')
            replay={}
            for which,cid in (('target',old['target_candidate_index']),('original_other',old['original_other_candidate_index'])):
                coarse=e[which+'_coarse_cost'];require(coarse==r['natural_candidate_coarse_costs'][ids.index(cid)],'fixed_pair_actual_coarse')
                value=native_pair_cost(coarse,roles['no_aug'][which+'_delta'],roles['aug'][which+'_delta'],*scales,gamma,r['coarse_dtype'])
                require(value==e[which+'_native_cost']==e[which+'_native_replay']==costs[ids.index(cid)],'fixed_pair_actual_native_replay')
                replay[which]=value
            correction=math.fsum(w*scale*(roles[role]['target_delta']-roles[role]['original_other_delta']) for w,scale,role in zip((1-gamma,gamma),scales,('no_aug','aug')))
            real=e['target_coarse_cost']-e['original_other_coarse_cost']+correction
            require(e['real_arithmetic_correction']==correction and e['real_arithmetic_margin']==real and e['native_margin']==margin and e['native_minus_real_margin']==margin-real,'fixed_pair_real_and_rounding_residual')
            require(abs(correction)<=bound+8*math.ulp(max(abs(correction),bound,1.)),'necessary_real_correction_bound')
        indexed[cap]=index
    for key,old in origin.items():
        a=indexed[5][key]['fixed_train_pair'];b=indexed[10][key]['fixed_train_pair']
        require(a['role_scale0']==b['role_scale0'],'paired_cap_scales')
        require(abs(a['new_pairwise_bound']-old['original_pairwise_bound'])<1e-12,'C5_old_bound')
        require(abs(b['new_pairwise_bound']-2*a['new_pairwise_bound'])<1e-12,'C10_double_bound')
    return indexed

def distribution(values):
    known=sorted(x for x in values if x is not None)
    require(all(finite(x) for x in known),'finite_distribution')
    def quantile(p):
        if not known:return None
        pos=(len(known)-1)*p;i=int(pos)
        return known[i]+(known[min(i+1,len(known)-1)]-known[i])*(pos-i)
    return dict(available=len(known),unknown=len(values)-len(known),mean=math.fsum(known)/len(known) if known else None,
      quantiles={k:quantile(p) for k,p in (('min',0),('p10',.1),('p25',.25),('median',.5),('p75',.75),('p90',.9),('p95',.95),('max',1))})

def fixed_pair_summary(records,per_cap):
    aligned=fixed_pair_checks(records,per_cap)
    origin={(r['seed'],identity(r['query_identity'])):r for r in records}
    reports={}
    for label in ('all400','original87'):
        keys=[k for k in origin if label=='all400' or origin[k]['original_bound_obstructed']]
        seed_reports={}
        for seed in SEEDS:
            sk=[k for k in keys if k[0]==seed];n=len(sk);caps={}
            for cap in (5,10):
                rr=[aligned[cap][k] for k in sk];fp=[r['fixed_train_pair'] for r in rr]
                sat={}
                for role in ('no_aug','aug'):
                    items=[next(t for t in e['role_saturation'] if t['role']==role) for e in fp]
                    sat[role]={which:dict(distribution=distribution([t[which+'_saturation'] for t in items]),
                      saturated95=sum(t[which+'_saturated95'] for t in items),denominator=n,
                      fraction=sum(t[which+'_saturated95'] for t in items)/n if n else None) for which in ('target','original_other')}
                caps[str(cap)]=dict(queries=n,metrics=metrics(rr),rank=distribution([r['rank'] for r in rr]),
                  rank_minus_original=distribution([r['rank']-origin[k]['original_rank'] for k,r in zip(sk,rr)]),
                  RR_minus_original=distribution([r['RR']-1/origin[k]['original_rank'] for k,r in zip(sk,rr)]),
                  fixed_pair_margin=distribution([e['actual_fixed_pair_margin'] for e in fp]),
                  real_arithmetic_fixed_pair_margin=distribution([e['real_arithmetic_margin'] for e in fp]),
                  native_minus_real_rounding_residual=distribution([e['native_minus_real_margin'] for e in fp]),
                  margin_minus_original=distribution([e['actual_fixed_pair_margin']-origin[k]['original_final_gap'] for k,e in zip(sk,fp)]),
                  bound_released_count=sum(e['bound_released'] for e in fp),bound_release_denominator=n,
                  bound_release_fraction=sum(e['bound_released'] for e in fp)/n if n else None,saturation=sat)
            seed_reports[str(seed)]=dict(queries=n,caps=caps,
              C10_minus_C5={k:caps['10']['metrics'][k]-caps['5']['metrics'][k] if n else None for k in ('MRR','top1','top5','top10')},
              paired_rank_delta=distribution([aligned[10][k]['rank']-aligned[5][k]['rank'] for k in sk]),
              paired_fixed_margin_delta=distribution([aligned[10][k]['fixed_train_pair']['actual_fixed_pair_margin']-aligned[5][k]['fixed_train_pair']['actual_fixed_pair_margin'] for k in sk]))
        reports[label]=dict(query_seed_pairs=len(keys),per_seed=seed_reports,
          equal_two_seed_mean_C10_minus_C5={k:math.fsum(seed_reports[str(s)]['C10_minus_C5'][k] for s in SEEDS)/2 if all(seed_reports[str(s)]['C10_minus_C5'][k] is not None for s in SEEDS) else None for k in ('MRR','top1','top5','top10')},
          total_bound_release_count={str(cap):sum(aligned[cap][k]['fixed_train_pair']['bound_released'] for k in keys) for cap in (5,10)})
    return reports

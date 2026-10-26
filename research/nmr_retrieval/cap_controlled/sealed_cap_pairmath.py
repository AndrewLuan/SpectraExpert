"""功能概述：本轮受控test矩阵的纯配对算术，无文件读取、训练、模型调用或选择。
normalize把新query_identity预测和历史扁平预测统一为六字段身份；index拒绝重复及非法rank。
stratum按canonical connectivity整簇重采样，所有config与seed共用同一簇权重，先逐seed计算
in_pool_MRR条件比值再等权平均；空分母保留null及undefined bootstrap计数。analyze_pair
严格匹配完整身份和mask，不取交集；拆解common/new/lost/neither的full RR贡献与候选变化。
代码库关系：由后续sealed结果gate调用；核心算术来自上一轮已独审analyze_sealed_pairs.py，
其完整源SHA记录在MATH_PROVENANCE_SHA。这里只复用纯函数，不复用旧报告、指标或预测。
输入输出与副作用：接收内存中两seed预测，返回18strata及5000次名义95%区间；
不读文件、不写文件、不运行选择；生产入口必须先完成独立冻结和实际stage收据核验。
"""
import math,json
import numpy as np
SEEDS=(193,211)
VARIANTS={"observed":1000,"frequency_crop":200,"peak_dropout":171}
SOURCES=("HMDB","NMRexp","WanJuan","nmrshiftdb2_2018","nmrshiftdb2_2024")
CATEGORIES=("common","new","lost","neither")
LINEAR=("MRR","top1","top5","top10","recall512","in_pool_RR_contribution","out_of_pool_RR_contribution")
MATH_PROVENANCE_SHA='46f285ab44976c84e2c45ca008839700f3a598864094135d9ed70cbdce298422'
FIELDS=("query_order","variant","source","record_id","smiles","group")

def need(value,message):
    if not value:raise ValueError(message)
def identity(row):return tuple(row[k] for k in FIELDS)
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def index(rows):
    result={}
    for row in rows:
        k=identity(row);need(k not in result,"duplicate_query_identity")
        need(type(row["rank"]) is int and 1<=row["rank"]<=114982,"invalid_rank")
        need(type(row["target_in_coarse_pool"]) is bool,"pool_hit_bool_required")
        need(isinstance(row["group"],str) and bool(row["group"]),"connectivity_group_required")
        result[k]=row
    return result
def outcomes(row):
    rr=1/row["rank"];hit=float(row["target_in_coarse_pool"])
    return np.array([rr,float(row["rank"]<=1),float(row["rank"]<=5),float(row["rank"]<=10),hit,hit*rr,(1-hit)*rr],dtype=np.float64)
def conditional(num,den):
    n,d=np.asarray(num,dtype=float),np.asarray(den,dtype=float)
    return np.divide(n,d,out=np.full(np.broadcast(n,d).shape,np.nan),where=d>0)
def decoded(totals):
    t=np.asarray(totals,dtype=float);n=t[...,30]
    l={k:t[...,i]/n for i,k in enumerate(LINEAR)}
    r={k:t[...,7+i]/n for i,k in enumerate(LINEAR)}
    l["in_pool_MRR"]=(conditional(t[...,14],t[...,15])+conditional(t[...,16],t[...,17]))/2
    r["in_pool_MRR"]=(conditional(t[...,18],t[...,19])+conditional(t[...,20],t[...,21]))/2
    l["in_pool_MRR_pooled"]=conditional(t[...,5],t[...,4])
    r["in_pool_MRR_pooled"]=conditional(t[...,12],t[...,11])
    delta={k:l[k]-r[k] for k in l}
    delta.update({"category_"+c+"_MRR_contribution":t[...,22+i]/n for i,c in enumerate(CATEGORIES)})
    return l,r,delta
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def pool_change(left,right):
    names=("natural_candidate_indices","natural_candidate_costs")
    if not all(all(k in row for k in names) for row in (left,right)):return None
    a,b=left[names[0]],right[names[0]];ca,cb=left[names[1]],right[names[1]]
    need(len(a)==len(set(a))==len(ca)==len(b)==len(set(b))==len(cb)==512,"natural512_pool_required")
    need(all(finite(x) for x in ca+cb),"finite_pool_costs")
    common=set(a)&set(b);va,vb=dict(zip(a,ca)),dict(zip(b,cb))
    ra={v:i for i,v in enumerate(sorted(common,key=lambda v:(va[v],v)))}
    rb={v:i for i,v in enumerate(sorted(common,key=lambda v:(vb[v],v)))}
    displacement=sum(abs(ra[v]-rb[v]) for v in common)/len(common) if common else None
    return dict(common_candidates=len(common),new_candidates=len(set(a)-set(b)),lost_candidates=len(set(b)-set(a)),common_candidate_mean_abs_rank_change=displacement)
def stratum(left,right,keys,replicates=5000,random_seed=317):
    n=len(keys)
    if not n:return dict(status="EMPTY_STRATUM",queries_per_seed=0,connectivity_clusters=0,metrics=None,delta=None,CI=None)
    vectors=[];groups=[];per_seed={};pool=[]
    for seed in SEEDS:
        arr_l=np.array([outcomes(left[seed][k]) for k in keys]);arr_r=np.array([outcomes(right[seed][k]) for k in keys])
        def one(a):
            x={k:a[:,i].mean() for i,k in enumerate(LINEAR)}
            x["in_pool_MRR"]=conditional(a[:,5].sum(),a[:,4].sum())
            return x
        per_seed[seed]=dict(left=one(arr_l),right=one(arr_r))
        per_seed[seed]["delta"]={k:per_seed[seed]["left"][k]-per_seed[seed]["right"][k] for k in per_seed[seed]["left"]}
    for k in keys:
        L=[left[s][k] for s in SEEDS];R=[right[s][k] for s in SEEDS]
        Lout=np.array([outcomes(x) for x in L]);Rout=np.array([outcomes(x) for x in R])
        changes=np.zeros(4);fractions=np.zeros(4)
        for i in range(2):
            category={(True,True):"common",(True,False):"new",(False,True):"lost",(False,False):"neither"}[(L[i]["target_in_coarse_pool"],R[i]["target_in_coarse_pool"])]
            ci=CATEGORIES.index(category);changes[ci]+=(Lout[i,0]-Rout[i,0])/2;fractions[ci]+=.5
            optional=pool_change(L[i],R[i])
            if optional is not None:pool.append(optional)
        vectors.append(np.r_[Lout.mean(0),Rout.mean(0),Lout[0,[5,4]],Lout[1,[5,4]],Rout[0,[5,4]],Rout[1,[5,4]],changes,fractions,1.])
        groups.append(L[0]["group"])
    vectors=np.array(vectors);labels=sorted(set(groups));lookup={g:i for i,g in enumerate(labels)}
    totals=np.zeros((len(labels),31),dtype=np.float64)
    for row,g in zip(vectors,groups):totals[lookup[g]]+=row
    l,r,delta=decoded(totals.sum(0))
    need(abs(sum(delta["category_"+c+"_MRR_contribution"] for c in CATEGORIES)-delta["MRR"])<2e-12,"category_delta_sum")
    need(abs(l["in_pool_RR_contribution"]+l["out_of_pool_RR_contribution"]-l["MRR"])<2e-12,"coverage_RR_sum")
    rng=np.random.Generator(np.random.PCG64(random_seed));boot={k:[] for k in delta}
    for start in range(0,replicates,128):
        count=min(128,replicates-start)
        weights=rng.multinomial(len(labels),np.full(len(labels),1/len(labels)),size=count)
        sampled=weights@totals;_,_,sample_delta=decoded(sampled)
        for k,x in sample_delta.items():boot[k].extend(np.atleast_1d(x).tolist())
    intervals={}
    for k,values in boot.items():
        valid=np.array([v for v in values if math.isfinite(v)])
        intervals[k]=dict(lower=float(np.quantile(valid,.025,method="linear")) if len(valid) else None,
            upper=float(np.quantile(valid,.975,method="linear")) if len(valid) else None,
            confidence_level=.95,valid_replicates=len(valid),undefined_replicates=replicates-len(valid))
    category={c:dict(mean_query_count_per_seed=vectors[:,26+i].sum(),fraction=vectors[:,26+i].mean(),
        contribution_to_total_delta_MRR=delta["category_"+c+"_MRR_contribution"],
        conditional_delta_MRR=conditional(vectors[:,22+i].sum(),vectors[:,26+i].sum()),
        queries_by_seed={s:sum({(True,True):"common",(True,False):"new",(False,True):"lost",(False,False):"neither"}[(left[s][k]["target_in_coarse_pool"],right[s][k]["target_in_coarse_pool"])]==c for k in keys) for s in SEEDS}) for i,c in enumerate(CATEGORIES)}
    optional=None
    if pool:
        optional={k:sum(x[k] for x in pool if x[k] is not None)/sum(x[k] is not None for x in pool) if any(x[k] is not None for x in pool) else None for k in pool[0]}
        optional["available_query_seed_pairs"]=len(pool)
    return clean(dict(status="COMPLETE",queries_per_seed=n,connectivity_clusters=len(labels),per_seed=per_seed,
        two_seed=dict(left=l,right=r,delta=delta),categories=category,paired95CI=intervals,optional_common_pool=optional,
        conditional_estimand="primary in_pool_MRR: equal mean of two seed-specific ratios in point and every bootstrap; pooled ratio separately named"))
def analyze_pair(left_rows,right_rows,replicates=5000,random_seed=317,legacy_right=False):
    need(legacy_right is False,"legacy_mask_exemption_disallowed")
    left={s:index([normalize(x) for x in left_rows[s]]) for s in SEEDS};right={s:index([normalize(x) for x in right_rows[s]]) for s in SEEDS}
    keys=set(left[193]);need(all(set(z)==keys for z in list(left.values())+list(right.values())),"exact_paired_identity_required_no_intersection")
    for k in keys:
        rows=[left[s][k] for s in SEEDS]+[right[s][k] for s in SEEDS]
        need(all(type(r.get("mask_content_sha256")) is str and len(r["mask_content_sha256"])==64 and set(r["mask_content_sha256"])<=set("0123456789abcdef") for r in rows) and len({r["mask_content_sha256"] for r in rows})==1,"paired_mask_mismatch")
    order=sorted(keys,key=lambda k:json.dumps(k,sort_keys=True))
    result=[]
    for variant in VARIANTS:
        for source in ("all",)+SOURCES:
            subset=[k for k in order if left[193][k]["variant"]==variant and (source=="all" or left[193][k]["source"]==source)]
            result.append(dict(variant=variant,source=source,**stratum(left,right,subset,replicates,random_seed)))
    return result

def normalize(row):
    out=dict(row)
    if "query_identity" in row:
        query=row["query_identity"];need(isinstance(query,dict),"query_identity_dict")
        need(set(query)==set(FIELDS),"exact_query_identity_fields")
        for key in FIELDS:out[key]=query[key]
        need(type(row.get("target_in_pool")) is bool,"new_hit_bool")
        out["target_in_coarse_pool"]=row["target_in_pool"]
    need(type(out.get("query_order")) is int and out["query_order"]>=0,"query_order_int")
    need(out.get("variant") in VARIANTS and out.get("source") in SOURCES,"registered_variant_source")
    need(all(type(out.get(k)) is str and bool(out[k]) for k in ("smiles","group")),"identity_strings")
    need(type(out.get("record_id")) in (str,int),"record_id_scalar")
    need(type(out.get("rank")) is int and 1<=out["rank"]<=114982,"rank_int_range")
    need(type(out.get("target_in_coarse_pool")) is bool and (out["rank"]<=512)==out["target_in_coarse_pool"],"rank_pool_hit_consistency")
    if "RR" in out:need(finite(out["RR"]) and abs(out["RR"]-1/out["rank"])<1e-15,"RR_matches_rank")
    return out


"""功能概述：本轮6配置、2seed冻结test矩阵的纯内存统计，不读取预测文件。
validate_matrix核12cells、1371身份/variant数、完整mask配对及rank/池命中语义；
matrix_metrics生成6×(2seed+均值)×3variant×6source=324条完整指标；
analyze_matrix仅执行预注册10比较×18strata=180，调用sealed_cap_pairmath的簇bootstrap。
代码库关系：由独立冻结及执行收据gate之后的结果汇总入口调用，不依赖val模块。
输入输出与副作用：输入config→seed→rawrows与protocol.test_preregistration，返回
可JSON序列化的完整点估计和名义95%配对区间。无文件I/O、模型、筛选或正式系统修改。
"""
import collections,itertools,math
import sealed_cap_pairmath as pm
CONFIGS=('fixed_single','fixed_dual','cap5_single','cap5_dual','cap10_single','cap10_dual')
COMPARISONS=(
 ('cap10dual_minus_cap5dual','cap10_dual','cap5_dual'),
 ('cap10dual_minus_fixedsingle','cap10_dual','fixed_single'),
 ('cap5dual_minus_fixeddual','cap5_dual','fixed_dual'),
 ('cap10dual_minus_fixeddual','cap10_dual','fixed_dual'),
 ('cap5single_minus_fixedsingle','cap5_single','fixed_single'),
 ('cap10single_minus_cap5single','cap10_single','cap5_single'),
 ('cap10dual_minus_cap10single','cap10_dual','cap10_single'),
 ('cap5dual_minus_cap5single','cap5_dual','cap5_single'),
 ('fixeddual_minus_fixedsingle','fixed_dual','fixed_single'),
 ('cap10single_minus_fixedsingle','cap10_single','fixed_single'))
METRICS=('MRR','top1','top5','top10','recall512','in_pool_RR_contribution','out_of_pool_RR_contribution','in_pool_MRR')

def validate_registration(r):
    pm.need(tuple(r['matrix_configs'])==CONFIGS and r['seeds']==list(pm.SEEDS) and r['matrix_cells']==12,'exact12_registration')
    pm.need(r['variant_counts']==pm.VARIANTS and r['queries_per_cell']==1371 and r['candidate_library_size']==114982,'fixed_test_cohort')
    pm.need(r['strata']==['all',*pm.SOURCES],'fixed6_strata')
    expected=[dict(id=i,left=l,right=rr) for i,l,rr in COMPARISONS]
    pm.need(r['comparisons']==expected and r['comparison_strata']==180,'exact10_comparisons')
    u=r['uncertainty'];pm.need(u['bootstrap_seed']==317 and u['replicates']==5000 and u['confidence_level']==.95,'fixed_uncertainty')
    pm.need(r['test_reselection_allowed'] is False and r['additional_comparisons_after_results_allowed'] is False,'no_reselection_or_expansion')

def validate_matrix(raw):
    pm.need(set(raw)==set(CONFIGS),'all6_configs')
    matrix={};reference=None;mask_reference=None
    for config in CONFIGS:
        pm.need(set(raw[config])==set(pm.SEEDS),'all2_seeds')
        matrix[config]={}
        for seed in pm.SEEDS:
            rows=[pm.normalize(x) for x in raw[config][seed]]
            pm.need(len(rows)==1371 and dict(collections.Counter(x['variant'] for x in rows))==pm.VARIANTS,'exact1371_variant_counts')
            indexed=pm.index(rows);keys=set(indexed);mask={k:x.get('mask_content_sha256') for k,x in indexed.items()}
            pm.need(all(type(x) is str and len(x)==64 and set(x)<=set('0123456789abcdef') for x in mask.values()),'complete64hex_masks')
            if reference is None:reference=keys;mask_reference=mask
            pm.need(keys==reference and mask==mask_reference,'complete_matrix_exact_identities_masks')
            matrix[config][seed]=rows
    return matrix

def one_metrics(rows):
    n=len(rows);hit=[r for r in rows if r['target_in_coarse_pool']]
    if not n:return dict(queries=0,in_pool_queries=0,**{k:None for k in METRICS})
    rr=math.fsum(1/r['rank'] for r in rows);inside=math.fsum(1/r['rank'] for r in hit)
    return dict(queries=n,in_pool_queries=len(hit),MRR=rr/n,
        **{'top'+str(k):sum(r['rank']<=k for r in rows)/n for k in (1,5,10)},
        recall512=len(hit)/n,in_pool_RR_contribution=inside/n,
        out_of_pool_RR_contribution=math.fsum(1/r['rank'] for r in rows if not r['target_in_coarse_pool'])/n,
        in_pool_MRR=inside/len(hit) if hit else None)

def matrix_metrics(matrix):
    result=[]
    for config,variant,source in itertools.product(CONFIGS,pm.VARIANTS,('all',*pm.SOURCES)):
        per=[]
        for seed in pm.SEEDS:
            rows=[r for r in matrix[config][seed] if r['variant']==variant and (source=='all' or r['source']==source)]
            values=one_metrics(rows);per.append(values);result.append(dict(config=config,seed=str(seed),variant=variant,source=source,**values))
        pm.need(per[0]['queries']==per[1]['queries'],'paired_denominator')
        mean={k:(None if any(p[k] is None for p in per) else (per[0][k]+per[1][k])/2) for k in METRICS}
        result.append(dict(config=config,seed='two_seed_mean',variant=variant,source=source,
          queries=per[0]['queries'],in_pool_queries_mean=(per[0]['in_pool_queries']+per[1]['in_pool_queries'])/2,**mean))
    pm.need(len(result)==324,'complete324_metrics')
    return result

def analyze_matrix(raw,registration):
    validate_registration(registration);matrix=validate_matrix(raw);comparisons=[]
    for cid,left,right in COMPARISONS:
        values=pm.analyze_pair(matrix[left],matrix[right],replicates=5000,random_seed=317)
        pm.need(len(values)==18,'complete18_pair_strata')
        comparisons.extend(dict(comparison=cid,left=left,right=right,**x) for x in values)
    pm.need(len(comparisons)==180,'complete180_comparisons')
    return dict(schema='cap10_sealed_matrix_statistics_20261025',status='PASS',metrics=matrix_metrics(matrix),
        comparisons=comparisons,metric_rows=324,comparison_strata=180,bootstrap_seed=317,
        bootstrap_replicates=5000,selection_performed=False,promotion=False,
        conditional_estimand='equal mean of two seed-specific conditional ratios in point and each connectivity-cluster replicate',
        interval_interpretation='nominal 95 percent; all registered strata and null/undefined denominators retained')

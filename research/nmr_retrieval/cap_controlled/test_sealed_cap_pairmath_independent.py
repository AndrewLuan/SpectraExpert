"""功能概述：新sealed纯算术模块的独立query-list/whole-connectivity bootstrap oracle。
所有输入是本文件生成的内存字典；不打开预测、缓存、模型或任何真实test文件。
oracle逐query计算每seed比值并用同一整簇draw配对左右模型，独立核257次非线性CI与
undefined计数；另核同ID跨variant整簇、RR四类贡献、合法rank/mask与精确身份、候选
位移的含义及10comparison×18strata形状。不会执行冻结、评分、选型或调用远端。
"""
import copy
import math
import unittest

import numpy as np
import sealed_cap_pairmath as m
import sealed_cap_matrix as mx
from test_sealed_cap_matrix import fixture as matrix_fixture, registration

MASK='0123456789abcdef'*4


def row(i,rank,hit,group,variant='observed'):
    return dict(query_identity=dict(query_order=i,variant=variant,source='HMDB',record_id=str(i),
        smiles='C',group=group),rank=rank,RR=1/rank,target_in_pool=hit,mask_content_sha256=MASK)


def data():
    groups=['A','A','B','C']
    leftspec={193:[(1,True),(2,True),(700,False),(800,False)],
              211:[(4,True),(600,False),(701,False),(801,False)]}
    rightspec={193:[(2,True),(601,False),(3,True),(802,False)],
               211:[(5,True),(602,False),(6,True),(803,False)]}
    return ({s:[row(i,r,h,groups[i]) for i,(r,h) in enumerate(leftspec[s])] for s in m.SEEDS},
            {s:[row(i,r,h,groups[i]) for i,(r,h) in enumerate(rightspec[s])] for s in m.SEEDS})


def point_oracle(left,right,weights):
    n=sum(weights)
    def side(rows):
        per=[]
        for seed in m.SEEDS:
            r=rows[seed]
            hit=sum(w*x['target_in_pool'] for w,x in zip(weights,r))
            hit_rr=math.fsum(w/x['rank'] for w,x in zip(weights,r) if x['target_in_pool'])
            per.append(dict(MRR=math.fsum(w/x['rank'] for w,x in zip(weights,r))/n,
                            in_pool_MRR=hit_rr/hit if hit else None))
        return {key:math.fsum(x[key] for x in per)/2 if all(x[key] is not None for x in per) else None
                for key in ('MRR','in_pool_MRR')}
    a,b=side(left),side(right)
    return {k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in a}


def bootstrap_oracle(left,right,reps=257,seed=317,per_query_instead=False):
    groups=[x['query_identity']['group'] for x in left[193]]
    if per_query_instead:groups=list(map(str,range(len(groups))))
    labels=sorted(set(groups));lookup={g:i for i,g in enumerate(labels)}
    draws=np.random.Generator(np.random.PCG64(seed)).multinomial(
        len(labels),np.full(len(labels),1/len(labels)),size=reps)
    samples={k:[] for k in ('MRR','in_pool_MRR')}
    for draw in draws:
        weights=[int(draw[lookup[g]]) for g in groups]
        for k,v in point_oracle(left,right,weights).items():samples[k].append(v)
    result={}
    for k,values in samples.items():
        valid=[x for x in values if x is not None]
        result[k]=dict(lower=float(np.quantile(valid,.025,method='linear')) if valid else None,
                       upper=float(np.quantile(valid,.975,method='linear')) if valid else None,
                       valid_replicates=len(valid),undefined_replicates=reps-len(valid))
    return result


def observed(left,right,reps=257):
    return m.analyze_pair(left,right,replicates=reps,random_seed=317)[0]


class PureMathIndependentTests(unittest.TestCase):
    def test_query_list_oracle_point_CI_nonlinear_ratio_and_undefined_counts(self):
        left,right=data();actual=observed(left,right);oracle=bootstrap_oracle(left,right)
        point=point_oracle(left,right,[1]*4)
        self.assertAlmostEqual(actual['two_seed']['left']['in_pool_MRR'],.5)
        self.assertNotEqual(actual['two_seed']['left']['in_pool_MRR_pooled'],.5)
        for metric in ('MRR','in_pool_MRR'):
            self.assertAlmostEqual(actual['two_seed']['delta'][metric],point[metric],places=15)
            for key in ('lower','upper'):
                self.assertAlmostEqual(actual['paired95CI'][metric][key],oracle[metric][key],places=15)
            for key in ('valid_replicates','undefined_replicates'):
                self.assertEqual(actual['paired95CI'][metric][key],oracle[metric][key])
        self.assertGreater(oracle['in_pool_MRR']['undefined_replicates'],0)

    def test_same_cluster_same_ID_across_variant_is_resampled_as_whole(self):
        left,right=data()
        for s in m.SEEDS:
            for side in (left,right):
                side[s][1]['query_identity']=dict(side[s][0]['query_identity'],variant='frequency_crop')
        li={s:m.index([m.normalize(r) for r in left[s]]) for s in m.SEEDS}
        ri={s:m.index([m.normalize(r) for r in right[s]]) for s in m.SEEDS}
        keys=[m.identity(m.normalize(r)) for r in left[193]]
        actual=m.stratum(li,ri,keys,replicates=257,random_seed=317)
        oracle=bootstrap_oracle(left,right)
        naive=bootstrap_oracle(left,right,per_query_instead=True)
        self.assertEqual(actual['connectivity_clusters'],3)
        for key in ('lower','upper'):
            self.assertAlmostEqual(actual['paired95CI']['MRR'][key],oracle['MRR'][key],places=15)
        self.assertNotEqual(oracle['in_pool_MRR']['undefined_replicates'],naive['in_pool_MRR']['undefined_replicates'])

    def test_same_cluster_draw_is_paired_between_models(self):
        left,_=data();right=copy.deepcopy(left);actual=observed(left,right)
        for key,value in actual['two_seed']['delta'].items():self.assertEqual(value,0.)
        for value in actual['paired95CI'].values():
            self.assertEqual(value['lower'],0.);self.assertEqual(value['upper'],0.)

    def test_four_categories_partition_fullN_RR_not_conditional_means(self):
        left,right=data();actual=observed(left,right)
        names={(True,True):'common',(True,False):'new',(False,True):'lost',(False,False):'neither'}
        expected={name:0. for name in names.values()};counts={name:{s:0 for s in m.SEEDS} for name in names.values()}
        for seed in m.SEEDS:
            for a,b in zip(left[seed],right[seed]):
                name=names[a['target_in_pool'],b['target_in_pool']]
                expected[name]+=(1/a['rank']-1/b['rank'])/2/4;counts[name][seed]+=1
        for name,value in expected.items():
            self.assertAlmostEqual(actual['categories'][name]['contribution_to_total_delta_MRR'],value,places=15)
            self.assertEqual(actual['categories'][name]['queries_by_seed'],{str(k):v for k,v in counts[name].items()})
        self.assertAlmostEqual(sum(expected.values()),actual['two_seed']['delta']['MRR'],places=15)
        for side in ('left','right'):
            x=actual['two_seed'][side]
            self.assertAlmostEqual(x['in_pool_RR_contribution']+x['out_of_pool_RR_contribution'],x['MRR'],places=15)

    def test_no_hit_in_one_seed_keeps_primary_and_all_CI_replicates_undefined(self):
        left,right=data()
        for r in left[211]:r.update(rank=900,RR=1/900,target_in_pool=False)
        actual=observed(left,right,73)
        self.assertIsNone(actual['two_seed']['left']['in_pool_MRR'])
        self.assertIsNone(actual['two_seed']['delta']['in_pool_MRR'])
        ci=actual['paired95CI']['in_pool_MRR'];self.assertEqual(ci['valid_replicates'],0)
        self.assertEqual(ci['undefined_replicates'],73);self.assertIsNone(ci['lower']);self.assertIsNone(ci['upper'])

    def test_strict_full_identity_rejects_intersection_and_duplicate(self):
        for attack in ('missing','duplicate','changed_group'):
            left,right=data()
            if attack=='missing':right[211].pop()
            elif attack=='duplicate':left[193].append(copy.deepcopy(left[193][0]))
            else:right[193][0]['query_identity']['group']='changed'
            with self.assertRaises(ValueError):observed(left,right,11)

    def test_current_masks_strict_hex_and_legacy_mode_cannot_skip(self):
        for mask in (None,'z'*64,'A'*64,'a'*63):
            left,right=data()
            for s in m.SEEDS:
                for r in left[s]+right[s]:
                    if mask is None:r.pop('mask_content_sha256')
                    else:r['mask_content_sha256']=mask
            with self.assertRaises(ValueError):observed(left,right,11)
        left,right=data()
        with self.assertRaises(ValueError):m.analyze_pair(left,right,replicates=11,legacy_right=True)

    def test_rank_pool_semantics_and_types_strict(self):
        for rank,hit in ((700,True),(2,False),(0,False),(114983,False),(True,True),(2.,True)):
            left,right=data();r=left[193][0];r.update(rank=rank,RR=1/rank if rank else 0,target_in_pool=hit)
            with self.assertRaises(ValueError):observed(left,right,11)

    def test_flat_and_new_schema_equal_no_string_coercion_of_record_ID(self):
        left,right=data();flat={}
        for seed in m.SEEDS:
            flat[seed]=[dict(r['query_identity'],rank=r['rank'],RR=r['RR'],target_in_coarse_pool=r['target_in_pool'],
                            mask_content_sha256=r['mask_content_sha256']) for r in right[seed]]
        self.assertEqual(observed(left,right,29),observed(left,flat,29))
        flat[193][0]['record_id']=0
        with self.assertRaises(ValueError):observed(left,flat,11)

    def test_optional_pool_displacement_aligns_ID_and_is_not_pairwise_flip(self):
        left=dict(natural_candidate_indices=list(range(512)),natural_candidate_costs=list(map(float,range(512))))
        right=dict(natural_candidate_indices=list(range(256,768)),natural_candidate_costs=[-float(x) for x in range(256,768)])
        value=m.pool_change(left,right)
        self.assertEqual(value['common_candidates'],256);self.assertEqual(value['new_candidates'],256)
        self.assertEqual(value['lost_candidates'],256);self.assertEqual(value['common_candidate_mean_abs_rank_change'],128.)
        left['natural_candidate_costs']=[0.]*512;right['natural_candidate_costs']=[0.]*512
        self.assertEqual(m.pool_change(left,right)['common_candidate_mean_abs_rank_change'],0.)

    def test_180_strata_shape_and_local_rng_independent_of_global_numpy(self):
        left,right=data();first=m.analyze_pair(left,right,replicates=23,random_seed=317)
        np.random.seed(999);np.random.random(1000)
        self.assertEqual(first,m.analyze_pair(left,right,replicates=23,random_seed=317))
        ten=[m.analyze_pair(left,right,replicates=7,random_seed=317) for _ in range(10)]
        self.assertEqual(sum(map(len,ten)),180)
        self.assertEqual(sum(x['status']=='EMPTY_STRATUM' for pair in ten for x in pair),160)


class MatrixIndependentTests(unittest.TestCase):
    def test_full12matrix_nonzero_comparison_and5000_paired_CI_oracle(self):
        raw=matrix_fixture()
        for config,rank in (('cap10_dual',1),('cap5_dual',4)):
            for seed in m.SEEDS:
                for r in raw[config][seed]:
                    if r['target_in_pool']:r.update(rank=rank,RR=1/rank)
        actual=mx.analyze_matrix(raw,registration())
        self.assertEqual(len(actual['metrics']),324);self.assertEqual(len(actual['comparisons']),180)
        x=next(z for z in actual['comparisons'] if z['comparison']=='cap10dual_minus_cap5dual' and z['variant']=='observed' and z['source']=='all')
        self.assertEqual(x['two_seed']['delta']['MRR'],.28125)
        self.assertEqual(x['two_seed']['delta']['in_pool_MRR'],.75)
        ci=x['paired95CI']['in_pool_MRR'];self.assertEqual(ci['lower'],.75);self.assertEqual(ci['upper'],.75)
        draws=np.random.Generator(np.random.PCG64(317)).multinomial(2,[.5,.5],size=5000)
        self.assertEqual(ci['valid_replicates'],int((draws[:,0]>0).sum()))
        self.assertEqual(ci['undefined_replicates'],int((draws[:,0]==0).sum()))
        self.assertFalse(actual['selection_performed']);self.assertFalse(actual['promotion'])

    def test_registration_rejects_comparison_direction_or_extra_or_scope_expansion(self):
        for change in ('direction','extra','replicates','strata','reselection'):
            r=registration()
            if change=='direction':r['comparisons'][0]['left'],r['comparisons'][0]['right']=r['comparisons'][0]['right'],r['comparisons'][0]['left']
            elif change=='extra':r['comparisons'].append(copy.deepcopy(r['comparisons'][0]))
            elif change=='replicates':r['uncertainty']['replicates']=4999
            elif change=='strata':r['strata'].append('new_source')
            else:r['additional_comparisons_after_results_allowed']=True
            with self.assertRaises(ValueError):mx.validate_registration(r)

    def test_matrix_exact1371_identity_no_intersection_duplicate_or_group_relabel(self):
        for change in ('duplicate','group','source'):
            raw=matrix_fixture();rows=raw['cap10_single'][211]
            if change=='duplicate':rows[1]=copy.deepcopy(rows[0])
            elif change=='group':rows[0]['query_identity']['group']='different'
            else:rows[0]['query_identity']['source']='unregistered'
            with self.assertRaises(ValueError):mx.validate_matrix(raw)

    def test_matrix_empty_seed_conditional_is_null_not_pooled(self):
        raw=matrix_fixture()
        for r in raw['cap10_single'][211]:r.update(rank=900,RR=1/900,target_in_pool=False)
        rows=mx.matrix_metrics(mx.validate_matrix(raw))
        x=next(z for z in rows if z['config']=='cap10_single' and z['seed']=='two_seed_mean' and z['variant']=='observed' and z['source']=='all')
        self.assertIsNone(x['in_pool_MRR'])
        self.assertAlmostEqual(x['in_pool_RR_contribution']+x['out_of_pool_RR_contribution'],x['MRR'],places=15)


if __name__=='__main__':unittest.main()

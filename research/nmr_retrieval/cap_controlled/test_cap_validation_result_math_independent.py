"""功能概述：只用人工记录验证独立结果算术，覆盖dtype边界、条件分母与固定预选门槛。
不读科学缓存、权重、旧test或真实评估输出；所有测试同步、无外部进程和模型调用。
"""
import copy
import unittest
import cap_validation_result_math_independent as m

def row(i=0,source='HMDB',variant='observed',rank=1,hit=True):
    return dict(query_identity=dict(query_order=i,variant=variant,source=source,record_id=str(i),smiles='s'+str(i),group='g'+str(i)),
      rank=rank,RR=1/rank,target_in_pool=hit,in_pool_target_rank=rank if hit else None,
      cached_outside_pool_rank=None if hit else rank,target_candidate_index=0,target_cost=1.,
      natural_candidate_indices=list(range(512)),natural_candidate_costs=[1.]+[10.]*511,
      natural_candidate_coarse_costs=[1.]+[10.]*511,natural_candidate_trace_sha256=[['a'*64]*512,['b'*64]*512],
      mask_content_sha256='c'*64,gamma=.25,top1=rank<=1,top5=rank<=5,top10=rank<=10)

class IndependentMathTests(unittest.TestCase):
    def test_epsilon_uses_native_float64_not_float32(self):
        r=row();r['natural_candidate_costs'][1]=m.cast(1+2**-23,'torch.float32')
        self.assertEqual(m.audit_row(r,'torch.float64')['rank'],1)
        with self.assertRaisesRegex(ValueError,'in_pool_cost_rank'):m.audit_row(r,'torch.float32')
        r.update(rank=2,RR=.5,in_pool_target_rank=2,top1=False)
        self.assertEqual(m.audit_row(r,'torch.float32')['rank'],2)

    def test_miss_keeps_native_full_library_rank(self):
        r=row(rank=113000,hit=False)
        self.assertEqual(m.audit_row(r,'torch.float64')['rank'],113000)
        r['cached_outside_pool_rank']=512
        with self.assertRaisesRegex(ValueError,'native_miss_fallback'):m.audit_row(r,'torch.float64')

    def test_float32_epsilon_scalar_cast_precedes_add(self):
        r=row(rank=2);r['natural_candidate_costs'][0]=-m.cast(1e-7,'torch.float32')
        r['natural_candidate_costs'][1]=0.;r['target_cost']=r['natural_candidate_costs'][0]
        self.assertEqual(m.audit_row(r,'torch.float32')['rank'],2)
        self.assertLess(m.cast(r['target_cost']+1e-7,'torch.float32'),0.)

    def test_nonfinite_and_appended_pool_reject(self):
        r=row();r['natural_candidate_costs'][20]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite_native_costs'):m.audit_row(r,'torch.float64')
        r=row();r['natural_candidate_indices'].append(512)
        with self.assertRaisesRegex(ValueError,'natural512'):m.audit_row(r,'torch.float64')

    def test_conditional_mean_is_equal_seed_ratio_not_pooled(self):
        per={193:[row(1,rank=1),row(2,rank=900,hit=False)],211:[row(1,rank=2),row(2,rank=2)]}
        self.assertEqual(m.equal_seed_metrics(per)['in_pool_MRR'],.75)
        self.assertNotEqual(m.equal_seed_metrics(per)['in_pool_MRR'],2/3)
        self.assertEqual(m.equal_seed_metrics(per)['recall512'],.75)

    def test_merge_then_source_macro_equal_seed(self):
        rows=[row(i,source=s,rank=i+1) for i,s in enumerate(m.SOURCES)]
        rows += [row(7,variant='frequency_crop',rank=2),row(8,variant='peak_dropout',rank=4)]
        signals,parts=m.validation_signals({193:rows,211:copy.deepcopy(rows)},5)
        self.assertAlmostEqual(signals['macro_MRR'],sum(1/(i+1) for i in range(5))/5)
        self.assertEqual(signals['frequency_crop_MRR'],.5)
        self.assertEqual(parts['193'],parts['211'])

    def test_gate_mrr_strict_epsilon_and_four_guards(self):
        control=dict(m.REFERENCE);c=dict(m.REFERENCE)
        c['MRR']+=1e-12
        self.assertFalse(m.selection(c,control)['eligible'])
        c['MRR']+=1e-8
        self.assertTrue(m.selection(c,control)['eligible'])
        self.assertEqual(m.selection(c,control)['preselected'],'cap10_dual')
        for field in ('top1','frequency_crop_MRR','peak_dropout_MRR','macro_MRR'):
            broken=dict(c);broken[field]-=1e-8
            result=m.selection(broken,control)
            self.assertFalse(result['eligible']);self.assertEqual(result['preselected'],m.BASELINE)

    def test_common_candidate_reordering_is_invariant(self):
        a=row();b=copy.deepcopy(a)
        for key in ('natural_candidate_indices','natural_candidate_costs','natural_candidate_coarse_costs'):
            b[key]=list(reversed(b[key]))
        b['natural_candidate_trace_sha256']=[list(reversed(r)) for r in b['natural_candidate_trace_sha256']]
        self.assertEqual(m.common_invariance([a],[b])['common_candidate_pairs'],512)
        b['natural_candidate_trace_sha256'][1][2]='d'*64
        with self.assertRaisesRegex(ValueError,'common_trace'):m.common_invariance([a],[b])

    def test_fixed_pair_population_cannot_be_subselected(self):
        with self.assertRaisesRegex(ValueError,'fixed400'):m.fixed_pair_checks([], {5:[],10:[]})

    def test_native_pair_replay_matches_torch_mixed_dtype(self):
        import torch
        for dtype,td in (('torch.float32',torch.float32),('torch.float64',torch.float64)):
            d0,d1,s0,s1=[m.cast(v,'torch.float32') for v in (-4.9,3.8,1.0003,.75003)]
            a=torch.tensor(d0,dtype=torch.float32)*torch.tensor(s0,dtype=torch.float32)
            b=torch.tensor(d1,dtype=torch.float32)*torch.tensor(s1,dtype=torch.float32)
            native=(torch.tensor(20000001.,dtype=td)+.75*a+.25*b).item()
            self.assertEqual(m.native_pair_cost(20000001.,d0,d1,s0,s1,.25,dtype),native)

    def test_full400_original87_summary_and_saturation_denominators(self):
        records=[];per={5:[],10:[]}
        for seed in m.SEEDS:
            for i in range(200):
                obstructed=(i<44 if seed==193 else i<43)
                gap=15. if obstructed else 5.
                base=row(i,rank=2)
                old=dict(seed=seed,query_identity=base['query_identity'],original_other_candidate_index=1,
                  target_candidate_index=0,original_coarse_gap=gap,original_pairwise_bound=10.,
                  original_bound_obstructed=obstructed,original_rank=2,original_final_gap=.1)
                records.append(old)
                for cap in (5,10):
                    delta=m.cast(-.98*cap,'torch.float32')
                    value=m.native_pair_cost(gap,delta,delta,1.,1.,.25,'torch.float64')
                    r=copy.deepcopy(base);r.update(seed=seed,coarse_dtype='torch.float64',correction_dtypes=['torch.float32']*2,final_cost_dtype='torch.float64')
                    r['natural_candidate_costs']=[value,0.]+[100.]*510;r['natural_candidate_coarse_costs']=[gap,0.]+[100.]*510
                    e={k:old[k] for k in ('original_other_candidate_index','target_candidate_index','original_coarse_gap','original_pairwise_bound','original_bound_obstructed','original_rank')}
                    margin=r['natural_candidate_costs'][0]
                    e.update(new_pairwise_bound=2.*cap,role_scale0=[1.,1.],bound_released=gap<=2*cap,
                      coarse_dtype='torch.float64',correction_dtypes=['torch.float32']*2,final_cost_dtype='torch.float64',
                      target_coarse_cost=gap,original_other_coarse_cost=0.,target_native_cost=value,original_other_native_cost=0.,
                      target_native_replay=value,original_other_native_replay=0.,native_margin=margin,
                      actual_fixed_pair_margin=margin,real_arithmetic_margin=gap+delta,native_minus_real_margin=margin-(gap+delta),real_arithmetic_correction=delta,
                      role_saturation=[dict(role=role,target_delta=delta,original_other_delta=0.,target_saturation=abs(delta)/cap,
                        original_other_saturation=0.,target_saturated95=True,original_other_saturated95=False) for role in ('no_aug','aug')])
                    r['fixed_train_pair']=e;per[cap].append(r)
        report=m.fixed_pair_summary(records,per)
        self.assertEqual(report['all400']['query_seed_pairs'],400)
        self.assertEqual(report['original87']['query_seed_pairs'],87)
        self.assertEqual(report['all400']['total_bound_release_count'],{'5':313,'10':400})
        self.assertEqual(report['original87']['total_bound_release_count'],{'5':0,'10':87})
        sat=report['original87']['per_seed']['211']['caps']['10']['saturation']['aug']['target']
        self.assertEqual((sat['saturated95'],sat['denominator'],sat['fraction']),(43,43,1.))
        per[10][0]['fixed_train_pair']['original_other_candidate_index']=3
        with self.assertRaisesRegex(ValueError,'original_fixed_other_unchanged'):m.fixed_pair_summary(records,per)

if __name__=='__main__':unittest.main()

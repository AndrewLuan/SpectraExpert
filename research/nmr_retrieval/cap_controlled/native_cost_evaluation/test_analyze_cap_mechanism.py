"""功能概述：以固定400/87纯标量fixture核机制分母、上界、饱和度和不可替换原other。
不读取真实训练/验证数据，不调用模型。共享候选fixture对齐candidateID而非数组位置；
篡改共同cost/trace/观测路由必须失败。空类别比值保留null。
"""
import copy
import hashlib
import struct
import unittest
import analyze_cap_mechanism as a
import cap_evaluation_common as c

def query(i):
    return dict(query_order=i,variant='observed',source=c.SOURCES[i%5],record_id=str(i),smiles='C',group='g'+str(i))
def pairs():
    original=[];per={5:[],10:[]}
    for seed in c.SEEDS:
        for i in range(200):
            blocked=i<(43 if seed==193 else 44)
            old=dict(seed=seed,query_identity=query(i),target_candidate_index=2,original_other_candidate_index=10,
              original_coarse_gap=15. if blocked else 5.,original_pairwise_bound=10.,
              original_bound_obstructed=blocked,original_rank=20,original_final_gap=1.25)
            original.append(old)
            for cap in c.CAPS:
                td=-4.9 if cap==5 else -5.1;od=.5 if cap==5 else 9.99
                f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
                native=lambda x,d:f32(f32(f32(x)+f32(.75*f32(f32(d)*1.)))+f32(.25*f32(f32(d)*1.)))
                costs=[50.]*512;costs[0]=0.;costs[10]=native(0.,od);costs[2]=native(old['original_coarse_gap'],td)
                fp={k:old[k] for k in ('target_candidate_index','original_other_candidate_index','original_coarse_gap','original_pairwise_bound','original_bound_obstructed','original_rank')}
                fp.update(coarse_dtype='torch.float32',correction_dtypes=['torch.float32']*2,final_cost_dtype='torch.float32',new_pairwise_bound=2.*cap,bound_released=old['original_coarse_gap']<=2*cap,role_scale0=[1.,1.],
                   actual_fixed_pair_margin=costs[2]-costs[10],target_coarse_cost=old['original_coarse_gap'],original_other_coarse_cost=0.,
                   target_native_cost=costs[2],original_other_native_cost=costs[10],
                   role_saturation=[dict(role=role,target_delta=td,original_other_delta=od,
                      target_saturation=abs(td)/cap,original_other_saturation=abs(od)/cap,
                      target_saturated95=abs(td)/cap>=.95,original_other_saturated95=abs(od)/cap>=.95) for role in ('no_aug','aug')])
                fp.update(c.fixed_pair_numerics(fp,.25))
                rank=sum(x<=f32(costs[2]+1e-7) for x in costs)
                per[cap].append(dict(coarse_dtype='torch.float32',correction_dtypes=['torch.float32']*2,final_cost_dtype='torch.float32',seed=seed,query_identity=query(i),target_in_pool=True,target_candidate_index=2,
                   rank=rank,RR=1/rank,gamma=.25,fixed_train_pair=fp,natural_candidate_indices=list(range(512)),natural_candidate_costs=costs))
    return original,per

def paired_row(ids):
    return dict(coarse_dtype='torch.float32',correction_dtypes=['torch.float32']*2,final_cost_dtype='torch.float32',query_identity=query(1),gamma=.25,mask_content_sha256='a',natural_candidate_indices=ids,
      natural_candidate_costs=[x*.01 for x in ids],natural_candidate_coarse_costs=[x*.02 for x in ids],
      natural_candidate_trace_sha256=[[hashlib.sha256((str(role)+':'+str(x)).encode()).hexdigest() for x in ids] for role in range(2)])
class MechanismTests(unittest.TestCase):
    def test_all400_original87_exact_denominators(self):
        old,per=pairs();v=a.fixed_pair_analysis(old,per)
        self.assertEqual(v['all400']['query_seed_pairs'],400);self.assertEqual(v['original87']['query_seed_pairs'],87)
        self.assertEqual(v['original87']['total_bound_release_count'],{'5':0,'10':87})
        self.assertEqual(v['original87']['per_seed']['193']['queries'],43)
        self.assertEqual(v['original87']['per_seed']['211']['queries'],44)
    def test_saturation_is_normalized_by_each_cap(self):
        old,per=pairs();v=a.fixed_pair_analysis(old,per)['all400']['per_seed']['193']['caps']
        self.assertEqual(v['5']['saturation']['no_aug']['target']['saturated95'],200)
        self.assertEqual(v['10']['saturation']['no_aug']['target']['saturated95'],0)
        self.assertAlmostEqual(v['10']['saturation']['no_aug']['target']['distribution']['mean'],.51)
    def test_paired_metrics_not_ratio_of_ranks(self):
        old,per=pairs();v=a.fixed_pair_analysis(old,per)['all400']['equal_two_seed_mean_C10_minus_C5']
        expected=sum(r['RR'] for r in per[10])/400-sum(r['RR'] for r in per[5])/400
        self.assertAlmostEqual(v['MRR'],expected)
    def test_changed_original_other_rejected(self):
        old,per=pairs();per[10][0]['fixed_train_pair']['original_other_candidate_index']=11
        with self.assertRaisesRegex(ValueError,'original_pair_unchanged'):a.fixed_pair_analysis(old,per)
    def test_bound_release_cannot_be_claimed_from_rank(self):
        old,per=pairs();per[5][0]['fixed_train_pair']['bound_released']=True
        with self.assertRaisesRegex(ValueError,'bound_release'):a.fixed_pair_analysis(old,per)
    def test_saturation_tamper_rejected(self):
        old,per=pairs();per[10][0]['fixed_train_pair']['role_saturation'][0]['target_saturation']=.99
        with self.assertRaisesRegex(ValueError,'saturation_ratio'):a.fixed_pair_analysis(old,per)
    def test_missing_pair_rejected(self):
        old,per=pairs();per[10].pop()
        with self.assertRaisesRegex(ValueError,'fixed_train_identity'):a.fixed_pair_analysis(old,per)
    def test_fixed_margin_recomputed_from_original_pair_costs(self):
        old,per=pairs();per[10][0]['fixed_train_pair']['actual_fixed_pair_margin']=0
        with self.assertRaisesRegex(ValueError,'actual_native_pair_margin'):a.fixed_pair_analysis(old,per)
    def test_shared_candidates_align_by_ID(self):
        x=paired_row(list(range(512)));y=paired_row(list(reversed(range(1,513))))
        v=a.common_invariance([x],[y]);self.assertEqual(v['common_candidate_pairs'],511);self.assertTrue(v['all_relative_orders_equal'])
    def test_shared_cost_change_rejected(self):
        x=paired_row(list(range(512)));y=copy.deepcopy(x);y['natural_candidate_costs'][10]+=.01
        with self.assertRaisesRegex(ValueError,'cost_changed'):a.common_invariance([x],[y])
    def test_shared_coarse_change_rejected(self):
        x=paired_row(list(range(512)));y=copy.deepcopy(x);y['natural_candidate_coarse_costs'][10]+=.01
        with self.assertRaisesRegex(ValueError,'coarse_changed'):a.common_invariance([x],[y])
    def test_shared_trace_change_rejected(self):
        x=paired_row(list(range(512)));y=copy.deepcopy(x);y['natural_candidate_trace_sha256'][0][10]='b'*64
        with self.assertRaisesRegex(ValueError,'trace_changed'):a.common_invariance([x],[y])
    def test_empty_distribution_unknown_never_zero(self):
        x=a.distribution([None,None]);self.assertEqual(x['unknown'],2);self.assertIsNone(x['mean']);self.assertIsNone(x['quantiles']['median'])
if __name__=='__main__':unittest.main()

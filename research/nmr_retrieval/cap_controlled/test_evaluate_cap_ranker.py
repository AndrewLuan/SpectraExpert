"""功能概述：用513候选纯合成tensor核本轮评估标签隔离、native公式和选择门槛。
FakeObservation/FakeFixed提供不含真实模型和谱的runtime；CapEvaluationTests覆盖自然512
裁剪、gamma、final64/alpha1、outside原fallback、固定other和hash不变性。
CommonTests覆盖先合并query再等seed、guard失败保留原预选、空分母和路径边界。
仅本地同步单元测试，不连接JJ、不读取科学cache或任何真实test内容。
"""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
import cap_evaluation_common as c
import evaluate_cap_ranker as e

class FakeObservation(nn.Module):
    def forward(self,ppm,tags,obs,H=None,K=None,per_quantile_sigma=None):return ppm[:,:3]*.1
class FakeFixed(nn.Module):
    def __init__(self,sigmas,x,o):
        super().__init__();self.observation_model=FakeObservation();self.network=nn.Linear(36,4)
        with torch.no_grad():
            self.network.weight.copy_(torch.arange(144).reshape(4,36)/100)
            self.network.bias.copy_(torch.tensor([.1,.2,.3,.4]))
        self.register_buffer('fixed_x_center',torch.as_tensor(x,dtype=torch.float32))
        self.register_buffer('fixed_o_center',torch.as_tensor(o,dtype=torch.float32))
def cap_module():
    path=Path(__file__).with_name('cap_ranker.py');sp=importlib.util.spec_from_file_location('_test_real_cap_module',path)
    x=importlib.util.module_from_spec(sp);sp.loader.exec_module(x);return x
def row():
    g='group'
    return dict(query_order=1,variant='observed',source='HMDB',record_id='r',smiles='C',group=g,
     evaluation_pool_size=512,candidate_indices=torch.arange(513),kind=1,
     features=torch.arange(513*33,dtype=torch.float32).reshape(513,33)/10000,
     coarse_scores=torch.arange(513,dtype=torch.float32)/100,
     predicted_ppm=torch.ones(513,4),tags=torch.ones(513,4),H=torch.ones(513),K=torch.ones(513),
     observation_input={'windows':[[1,2]],'xs':[1.],'semantics':'line_positions','weights':[1.]},
     query_observation={'windows':[],'partial_peaks':False},
     target_in_coarse_pool=True,target_position=2,outside_pool_rank=None,coarse_rank=3,
     negative_exclusion_mask=torch.zeros(513,dtype=torch.bool),mask_content_sha256='mask')
def models(cap=5):
    module=cap_module();rt=SimpleNamespace(fixed=SimpleNamespace(FixedCenterRanker=FakeFixed))
    result=[]
    for role in ('no_aug','aug'):
        model=module.CapRanker([.1],torch.zeros(33),torch.zeros(3),cap,rt)
        result.append((model,torch.ones(33),dict(alpha=1.0,epoch=64,delta_cap=cap,seed=193,role=role,center_estimate_sha256='a')))
    return module,result
class CapEvaluationTests(unittest.TestCase):
    def test_513_trim_and_no_label_input(self):
        r=row();n=e.natural_row(r)
        self.assertEqual(n['features'].shape,(512,33))
        self.assertFalse(set(n)&{'source','group','target_position','candidate_indices','negative_exclusion_mask'})
        self.assertTrue(all(len(n[k])==512 for k in e.CANDIDATE_FIELDS if k in n))
    def test_appended_positive_cannot_change_score(self):
        _,ms=models();r=row();a=e.score_natural(ms,e.natural_row(r),.25,'cpu',5)
        r['features'][-1]=1e10;r['coarse_scores'][-1]=-1e10
        b=e.score_natural(ms,e.natural_row(r),.25,'cpu',5)
        self.assertTrue(torch.equal(a['cost'],b['cost']))
    def test_real_module_score_pair_interoperability(self):
        module,ms=models(10);r=row();n=e.natural_row(r)
        native=dict(n,query_observation={k:r['query_observation'][k] for k in ('windows','partial_peaks')})
        costs,gamma=module.score_pair(ms,native,'cpu')
        actual=e.score_natural(ms,n,gamma,'cpu',10)
        self.assertTrue(torch.equal(costs,actual['cost']))
    def test_gamma_uses_native_public_query_observation(self):
        r=row();self.assertEqual(e.route_gamma(r['query_observation']),.25)
        self.assertEqual(e.route_gamma({'windows':[[1,2]]}),.75)
        self.assertEqual(e.route_gamma({'partial_peaks':True}),.75)
    def test_nonfinal_checkpoint_rejected(self):
        _,ms=models();ms[0][2]['epoch']=63
        with self.assertRaisesRegex(ValueError,'final_fixed_role'):e.score_natural(ms,e.natural_row(row()),.25,'cpu',5)
    def test_alpha_grid_forbidden(self):
        _,ms=models();ms[1][2]['alpha']=.5
        with self.assertRaisesRegex(ValueError,'final_fixed_role'):e.score_natural(ms,e.natural_row(row()),.25,'cpu',5)
    def test_cap_formula_and_bound(self):
        _,ms=models(10);x=e.score_natural(ms,e.natural_row(row()),.75,'cpu',10)
        self.assertEqual(x['total_bound'],10)
        for t in x['traces']:self.assertTrue(torch.equal(t['delta'],10*torch.tanh(t['logits']/10)))
    def test_outside_target_never_scored(self):
        _,ms=models();r=row();r.update(target_in_coarse_pool=False,target_position=512,outside_pool_rank=1001)
        x=e.outcome(r,e.score_natural(ms,e.natural_row(r),.25,'cpu',5))
        self.assertEqual(x['rank'],1001);self.assertEqual(x['RR'],1/1001)
        self.assertIsNone(x['target_cost']);self.assertIsNone(x['in_pool_target_rank'])
    def test_hit_unused_none_fallback(self):
        _,ms=models();r=row();x=e.outcome(r,e.score_natural(ms,e.natural_row(r),.25,'cpu',5))
        self.assertTrue(x['target_in_pool']);self.assertIsNone(x['cached_outside_pool_rank'])
    def test_outside_rank_crossing_recall_rejected(self):
        _,ms=models();r=row();r.update(target_in_coarse_pool=False,target_position=512,outside_pool_rank=20)
        with self.assertRaisesRegex(ValueError,'outside_rank'):e.outcome(r,e.score_natural(ms,e.natural_row(r),.25,'cpu',5))
    def test_scoring_identity_leak_rejected(self):
        _,ms=models();n=e.natural_row(row());n['source']='HMDB'
        with self.assertRaisesRegex(ValueError,'identity_leak'):e.score_natural(ms,n,.25,'cpu',5)
    def test_state_digest_and_candidate_traces_stable(self):
        _,ms=models();before=e.model_digest(ms)
        a=e.score_natural(ms,e.natural_row(row()),.25,'cpu',5);b=e.score_natural(ms,e.natural_row(row()),.25,'cpu',5)
        self.assertEqual(e.model_digest(ms),before);self.assertEqual(e.trace_hashes(a),e.trace_hashes(b))
    def test_fixed_other_retained_and_saturation_defined(self):
        _,ms=models(10);r=row();x=e.score_natural(ms,e.natural_row(r),.25,'cpu',10)
        other=10;gap=float(r['coarse_scores'][2].double()-r['coarse_scores'][other].double())
        z=e.fixed_pair_evidence(r,x,dict(original_other_candidate_index=other,original_coarse_gap=gap,
             original_pairwise_bound=10,original_bound_obstructed=False,original_rank=3),10)
        self.assertEqual(z['original_other_candidate_index'],other);self.assertTrue(z['bound_released'])
        for t in z['role_saturation']:self.assertEqual(t['target_saturation'],abs(t['target_delta'])/10)
    def test_fixed_other_gap_drift_rejected(self):
        _,ms=models();r=row();x=e.score_natural(ms,e.natural_row(r),.25,'cpu',5)
        with self.assertRaisesRegex(ValueError,'coarse_gap'):
            e.fixed_pair_evidence(r,x,dict(original_other_candidate_index=10,original_coarse_gap=20),5)
    def test_large_finite_coarse_native_FP32_pair_keeps_rounding_residual(self):
        r=row();r['coarse_scores'][2]=10000015.;r['coarse_scores'][10]=10000000.
        delta=torch.zeros(512);delta[2]=-4.9;delta[10]=.5
        cost=r['coarse_scores'][:512]+.75*delta+.25*delta
        scored=dict(cost=cost,traces=[{'delta':delta},{'delta':delta}],role_bounds=[5.,5.],total_bound=5.,gamma=.25)
        z=e.fixed_pair_evidence(r,scored,dict(original_other_candidate_index=10,original_coarse_gap=15.,
          original_pairwise_bound=10.,original_bound_obstructed=True,original_rank=20),5)
        self.assertEqual(z['actual_fixed_pair_margin'],10.)
        self.assertAlmostEqual(z['real_arithmetic_margin'],9.6,places=6)
        self.assertAlmostEqual(z['native_minus_real_margin'],.4,places=6)

class CommonTests(unittest.TestCase):
    def test_empty_conditional_ratio_unavailable(self):
        self.assertIsNone(c.ratio(0,0));self.assertIsNone(c.metrics([])['MRR'])
    def test_only_C10dual_can_be_selected(self):
        base=dict(c.BASELINE_SIGNALS);control=dict(base,MRR=base['MRR']+.001);candidate=dict(base,MRR=base['MRR']+.002)
        x=c.select(candidate,control,base);self.assertTrue(x['eligible']);self.assertEqual(x['preselected'],'cap10_dual')
    def test_MRR_must_exceed_both(self):
        b=dict(c.BASELINE_SIGNALS);x=c.select(dict(b,MRR=b['MRR']+.001),dict(b,MRR=b['MRR']+.002),b)
        self.assertFalse(x['eligible']);self.assertEqual(x['preselected'],c.BASELINE)
    def test_guard_failure_keeps_fixedsingle(self):
        b=dict(c.BASELINE_SIGNALS);x=c.select(dict(b,MRR=.7,top1=.4),dict(b,MRR=.6),b)
        self.assertFalse(x['eligible']);self.assertEqual(x['preselected'],c.BASELINE)
    def test_unregistered_baseline_rejected(self):
        b=dict(c.BASELINE_SIGNALS,MRR=.5)
        with self.assertRaisesRegex(ValueError,'exact_registered'):c.select(b,b,b)
    def test_test_paths_rejected(self):
        for p in ('/tmp/test/x.json','/tmp/sealed_resume_x/a.json','/tmp/x/sealed_test_report.md'):
            with self.assertRaises(ValueError):c.safe(p)
    def test_seed_mean_is_after_query_merge(self):
        rows=[]
        for i in range(1344):
            rows.append(dict(query_identity=dict(query_order=i,variant='observed',source=c.SOURCES[i%5],record_id=str(i),smiles='C',group=str(i)),rank=1 if i<320 else 2,RR=1 if i<320 else .5,target_in_pool=True))
        for v in ('frequency_crop','peak_dropout'):rows.append(dict(query_identity=dict(query_order=0,variant=v,source='HMDB',record_id='r',smiles='C',group='g'),rank=2,RR=.5,target_in_pool=True))
        actual=c.signal_for_seed(rows)
        self.assertEqual(actual['MRR'],(320+1024*.5)/1344)
        self.assertNotEqual(actual['MRR'],.75)
if __name__=='__main__':unittest.main()

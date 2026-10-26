"""功能概述：native dtype修复的独立PyTorch算术oracle回归，纯人工512候选。
覆盖32/64 coarse +32 correction原公式逐元素一致、真正tensor EPS负近零边界、
大coarse mixed-step固定pair、全val入口/reader、dtype元数据篡改及非有限值拒绝。
不读取真实cache/test，不连接服务器，不改旧源码。
"""
import copy,json,tempfile,unittest,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import torch
import cap_evaluation_common as c
import evaluate_cap_ranker as e
from test_evaluate_cap_ranker import models,row
from test_cap_evaluation_entry import EntryFixture
class NativePrecisionTests(unittest.TestCase):
    def test_runpy_subdirectory_resolves_native_helper(self):
        folder=Path(e.__file__).parent;parent=folder.parent
        for script in ('evaluate_cap_ranker.py','summarize_cap_validation.py','analyze_cap_mechanism.py'):
            code="import sys,runpy;from pathlib import Path;sys.path.insert(0,"+repr(str(parent))+");d=runpy.run_path("+repr(str(folder/script))+",run_name='_synthetic_import_only');assert Path(d['promoted_dtype'].__globals__['__file__']).resolve()==Path("+repr(str(folder/'cap_evaluation_common.py'))+").resolve()"
            proc=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,cwd=parent,timeout=30)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
    def test_stale_root_helper_in_sys_modules_is_rejected(self):
        folder=Path(e.__file__).parent;parent=folder.parent
        code="import sys,runpy;sys.path.insert(0,"+repr(str(parent))+");import cap_evaluation_common;runpy.run_path("+repr(str(folder/'evaluate_cap_ranker.py'))+",run_name='_synthetic_import_only')"
        proc=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,cwd=parent,timeout=30)
        self.assertNotEqual(proc.returncode,0);self.assertIn('native_contract_module_path_mismatch',proc.stderr)
    def test_native_score_pair_oracle_coarse32_and64_both_caps_and_gamma(self):
        for cap in (5,10):
            module,ms=models(cap)
            for dtype in (torch.float32,torch.float64):
                for partial in (False,True):
                    r=row();r['coarse_scores']=r['coarse_scores'].to(dtype);r['query_observation']['partial_peaks']=partial
                    natural=e.natural_row(r);gamma=e.route_gamma(r['query_observation'])
                    oracle,_=module.score_pair(ms,dict(natural,query_observation=r['query_observation']),'cpu')
                    actual=e.score_natural(ms,natural,gamma,'cpu',cap)
                    self.assertTrue(torch.equal(oracle,actual['cost']));self.assertEqual(actual['cost'].dtype,dtype)
                    self.assertEqual(actual['correction_dtypes'],['torch.float32']*2)
    def eps_row(self,dtype):
        _,ms=models();r=row();r['coarse_scores']=r['coarse_scores'].to(dtype);scored=e.score_natural(ms,e.natural_row(r),.25,'cpu',5)
        costs=torch.ones(512,dtype=dtype);costs[0]=0;costs[2]=float(torch.tensor(-1e-7,dtype=torch.float32));scored['cost']=costs
        return r,scored
    def test_native32_EPS_cast_before_add_negative_nearzero(self):
        r,s=self.eps_row(torch.float32);expected=int((s['cost']<=s['cost'][2]+1e-7).sum())
        self.assertEqual(expected,2);self.assertEqual(e.outcome(r,s)['rank'],expected)
        self.assertEqual(c.native_threshold(float(s['cost'][2]),'torch.float32'),0.)
    def test_native64_same_target_has_distinct_EPS_threshold(self):
        r,s=self.eps_row(torch.float64);self.assertEqual(e.outcome(r,s)['rank'],1)
        self.assertLess(c.native_threshold(float(s['cost'][2]),'torch.float64'),0.)
    def test_mixed64_fixed_pair_replays_fp32_corrections_then_double_add(self):
        r=row();r['coarse_scores']=r['coarse_scores'].double();r['coarse_scores'][2]=10000015.;r['coarse_scores'][10]=10000000.
        delta=torch.zeros(512);delta[2]=-4.9;delta[10]=.5
        cost=r['coarse_scores'][:512]+.75*delta+.25*delta
        scored=dict(cost=cost,traces=[{'delta':delta},{'delta':delta}],role_bounds=[5.,5.],total_bound=5.,gamma=.25,
          coarse_dtype='torch.float64',correction_dtypes=['torch.float32']*2,final_cost_dtype='torch.float64')
        fp=e.fixed_pair_evidence(r,scored,dict(original_other_candidate_index=10,original_coarse_gap=15.,original_pairwise_bound=10.,original_bound_obstructed=True,original_rank=20),5)
        self.assertEqual(fp['target_native_replay'],float(cost[2]));self.assertEqual(fp['original_other_native_replay'],float(cost[10]))
        self.assertEqual(fp['native_margin'],float(cost[2]-cost[10]));self.assertNotEqual(fp['native_margin'],10.)
    def test_full_entry_native64_reader_no_float32_cast(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=EntryFixture(tmp)
            for r in f.rows:r['coarse_scores']=r['coarse_scores'].double()+1.234567891e-8
            result=f.run();_,_,rows=c.read_evaluation(c.Pins(),f.out,5,193,'single','original_val',f.bindings,f.protocol,True)
            self.assertTrue(all(r['final_cost_dtype']=='torch.float64' for r in rows));self.assertEqual(result['records'],4)
            self.assertTrue(any(c.cast_native(v,'torch.float32')!=v for v in rows[0]['natural_candidate_costs']))
    def test_declared_promotion_cannot_be_forged(self):
        with self.assertRaisesRegex(ValueError,'promotion'):
            c.validate_cost_dtypes(dict(coarse_dtype='torch.float32',correction_dtypes=['torch.float32']*2,final_cost_dtype='torch.float64'))
    def test_native32_serialized_cost_is_exact32(self):
        self.assertNotEqual(c.cast_native(.123456789123,'torch.float32'),.123456789123)
        self.assertEqual(c.cast_native(.123456789123,'torch.float64'),.123456789123)
    def test_nonfinite_input_stays_rejected(self):
        _,ms=models();r=row();r['coarse_scores'][0]=float('inf')
        with self.assertRaisesRegex(ValueError,'finite'):e.natural_row(r)
if __name__=='__main__':unittest.main()

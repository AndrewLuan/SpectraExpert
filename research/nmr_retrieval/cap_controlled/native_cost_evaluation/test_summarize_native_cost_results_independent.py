"""功能概述：非作者核验新生产统计入口的native精度与完整12收据首预测读取边界。
仅用人工512成本和临时12子进程fixture；没有科学缓存、实际test预测或外部调用。
子进程生命周期复用已审fixture的显式完成状态/超时预算，不添加固定sleep或重试。
"""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import summarize_sealed_cap_results as s
from test_summarize_native_cost_results import row,native_full_fixture

class PrecisionIndependentTests(unittest.TestCase):
    def test_threshold_matches_real_torch_scalar_promotion(self):
        for dtype,td in ((np.float32,torch.float32),(np.float64,torch.float64)):
            for x in (-20000001.,-1e-7,-1e-8,0.,1e-8,1.,20000001.):
                actual=float((torch.tensor(x,dtype=td)+1e-7).item())
                self.assertEqual(s.native_threshold(float(dtype(x)),dtype),actual)
    def test_native64_does_not_hide_rank_perturbation(self):
        r=row('torch.float64',[0.,1.00000002e-7]+[1.]*510)
        self.assertEqual(r['rank'],1);s.validate_payload_row(r)
        wrong=copy.deepcopy(r);wrong.update(rank=2,RR=.5,in_pool_target_rank=2,top1=False)
        with self.assertRaisesRegex(ValueError,'native_rank_from_declared_dtype_costs'):s.validate_payload_row(wrong)
    def test_winner_and_margin_recomputed_from_natural512(self):
        for field,value,reason in (('winner_candidate_index',3,'native_winner_from_512'),('margin',999.,'native_margin_from_512')):
            r=row('torch.float64',target=7);r[field]=value
            with self.assertRaisesRegex(ValueError,reason):s.validate_payload_row(r)

class FullExecutionGateIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='native_stats_independent_')
        cls.f=native_full_fixture(cls.tmp.name)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def tearDown(self):self.f.restore()
    def test_last_cell_cache_identity_error_rejects_before_first_prediction(self):
        path=Path(self.f.queue['jobs'][-1]['job_out'])/'summary.json'
        value=json.loads(path.read_text());value['cache_sha256']='a'*64;path.write_text(json.dumps(value))
        seen=[];original=s.c.Pins.raw
        def guarded(pins,path,expected=None):
            if Path(path).name=='predictions.jsonl':seen.append(str(path));raise AssertionError('prediction_before_all12_valid')
            return original(pins,path,expected)
        with patch.object(s.c.Pins,'raw',guarded),self.assertRaisesRegex(ValueError,'leaf_cache_universe'):
            s.run(self.f.args,self.f.root/'never_written')
        self.assertEqual(seen,[])
    def test_exact_producer_evaluator_ref_required_before_first_prediction(self):
        path=Path(self.f.queue['jobs'][-1]['job_out'])/'summary.json'
        value=json.loads(path.read_text());value['evaluator']=dict(path='/unregistered/evaluator.py',sha256='a'*64)
        path.write_text(json.dumps(value));seen=[];original=s.c.Pins.raw
        def guarded(pins,path,expected=None):
            if Path(path).name=='predictions.jsonl':seen.append(str(path));raise AssertionError('prediction_before_all12_valid')
            return original(pins,path,expected)
        with patch.object(s.c.Pins,'raw',guarded),self.assertRaisesRegex(ValueError,'leaf_exact_evaluator_reference'):
            s.run(self.f.args,self.f.root/'never_written')
        self.assertEqual(seen,[])

if __name__=='__main__':unittest.main()

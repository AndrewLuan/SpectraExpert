"""功能概述：用人工512候选核诊断显示native fp32/fp64 promotion差异，不读科学数据。"""
import unittest,torch
import diagnose_validation_cost_contract as d
from test_evaluate_cap_ranker import models,row
import evaluate_cap_ranker as e
class ProbeTests(unittest.TestCase):
    def test_native_float32_shape_and_finiteness(self):
        _,ms=models();value=d.inspect_cost(ms,e.natural_row(row()),.25,'cpu')
        self.assertEqual(value['legacy_assertions'],dict(shape512=True,dtype_float32=True,all_finite=True))
    def test_double_coarse_preserves_native_promotion(self):
        _,ms=models();r=row();r['coarse_scores']=r['coarse_scores'].double()
        value=d.inspect_cost(ms,e.natural_row(r),.25,'cpu')
        self.assertEqual(value['native_cost']['dtype'],'torch.float64');self.assertEqual(value['legacy_assertions'],dict(shape512=True,dtype_float32=False,all_finite=True))
    def test_no_tensor_values_or_rank_exported(self):
        _,ms=models();value=d.inspect_cost(ms,e.natural_row(row()),.25,'cpu')
        self.assertNotIn('rank',value);self.assertEqual(set(value['native_cost']),{'shape','dtype','finite','nonfinite_count','sha256'})
if __name__=='__main__':unittest.main()

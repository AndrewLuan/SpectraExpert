"""功能概述：新sealed native dtype路径的纯合成全入口/alpha/EPS测试，无真实test读取。"""
import json,tempfile,unittest
import torch
import cap_test_evaluation_common as c
import cap_test_evaluate_ranker as e
from test_cap_test_evaluate_ranker import SealedFixture
from test_evaluate_cap_ranker import models,row
class TestPrecision(unittest.TestCase):
    def test_full_sealed_entry_mixed64_preserves_1371_contract_scaled_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SealedFixture(tmp)
            for r in f.data['rows']:r['coarse_scores']=r['coarse_scores'].double()+1.234567891e-8
            result=f.run();rows=[json.loads(x) for x in (f.out/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(result['records'],6);self.assertTrue(all(r['coarse_dtype']==r['final_cost_dtype']=='torch.float64' for r in rows))
            self.assertTrue(all(r['correction_dtypes']==['torch.float32']*2 for r in rows))
            self.assertTrue(any(c.cast_native(x,'torch.float32')!=x for x in rows[0]['natural_candidate_costs']))
    def test_native_baseline_original_alpha_with_mixed64(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SealedFixture(tmp,'fixed_dual');f.make_native_baseline()
            for r in f.data['rows']:r['coarse_scores']=r['coarse_scores'].double()+1.23e-8
            result=f.run();self.assertEqual(result['config'],'fixed_dual')
            rows=[json.loads(x) for x in (f.out/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['role_absolute_correction_bounds'],[1.25,2.5]);self.assertEqual(rows[0]['final_cost_dtype'],'torch.float64')
    def test_sealed_native_EPS32_boundary_is_tensor_native(self):
        _,ms=models();r=row();scored=e.score_natural(ms,e.natural_row(r),.25,'cpu',5)
        costs=torch.ones(512);costs[0]=0.;costs[2]=-1e-7;scored['cost']=costs
        self.assertEqual(e.outcome(r,scored)['rank'],2);self.assertEqual(c.native_threshold(float(costs[2]),'torch.float32'),0.)
    def test_native_input_promotion_required(self):
        with self.assertRaisesRegex(ValueError,'promotion'):
            c.validate_cost_dtypes(dict(coarse_dtype='torch.float64',correction_dtypes=['torch.float32']*2,final_cost_dtype='torch.float32'))
if __name__=='__main__':unittest.main()

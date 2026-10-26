"""双核融合验证集的同步合成边界检查，无真实数据、训练或外部副作用。
核对配对仅取val、源/记录hash稳定选择、输入顺序不影响配对和候选核种有效性。
另外核对零H权重直接返回C控制、等权公式、配对MRR差值和HMDB覆盖不影响其他来源。依赖
h_fusion_prepare与h_fusion_workflow，保证后续融合选参不读取test表现。
"""
import unittest
import torch
from .h_fusion_prepare import validation_pairs
from .h_fusion_workflow import combine,paired_bootstrap
from .h_fusion_workflow import routed_h_score
from unittest.mock import patch,Mock
from types import SimpleNamespace


class TestFusion(unittest.TestCase):
    def test_reporting_override_is_only_used_for_HMDB(self):
        context=SimpleNamespace(given={},counts=torch.tensor([1.]));inputs=({},None,None,None,{})
        override=Mock(return_value=torch.tensor([3.]));baseline=torch.tensor([2.])
        with patch('nmr_transfer.h_fusion_workflow.eligible',return_value=False),patch('nmr_transfer.h_fusion_workflow.query_score',return_value=baseline):
            self.assertIs(routed_h_score(dict(source='NMRexp'),context,{},inputs,override),baseline);override.assert_not_called()
            torch.testing.assert_close(routed_h_score(dict(source='HMDB'),context,{},inputs,override),torch.tensor([3.]));override.assert_called_once()
    def test_validation_pairing_is_order_independent_and_excludes_test(self):
        rows=[]
        for split,smi in [('val','CCO'),('test','CCN'),('train','CCC')]:
            for n in ['1H','13C']:
                for rid in ['a','b']:rows.append(dict(split=split,smiles=smi,domain='experimental',physics=[.1,.1],nucleus=n,record_id=rid,source='synthetic'))
        actual=validation_pairs(rows);other=validation_pairs(rows[::-1]);self.assertEqual(actual,other);self.assertEqual([s for s,_ in actual],['CCO'])
        self.assertEqual(set(actual[0][1]),{'1H','13C'})

    def test_zero_weight_preserves_C_even_for_infinite_H_and_equal_weight_formula(self):
        h=torch.tensor([float('inf'),1.]);c=torch.tensor([.1,2.]);self.assertIs(combine(h,c,0.),c);h=torch.tensor([3.,1.]);torch.testing.assert_close(combine(h,c,1.),.5*(h+c),rtol=0,atol=0)

    def test_paired_bootstrap_reports_same_query_difference(self):
        result=paired_bootstrap([dict(rank=1),dict(rank=20)],[dict(rank=2),dict(rank=20)]);self.assertAlmostEqual(result['MRR']['difference'],.25);self.assertAlmostEqual(result['top1']['difference'],.5);self.assertAlmostEqual(result['top10']['difference'],0.)


if __name__=='__main__':main()

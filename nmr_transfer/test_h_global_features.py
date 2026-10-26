"""功能概述：检查候选匹配特征的观测表示不变性和显式窗口隔离。

相同报告氢质量的重复/合并位置必须产生相同特征；改动窗外预测不能影响窗内匹配。
另检查空窗的有限编码、真结构身份字段不进入特征以及原始峰高不作为 HMDB 积分。
全部使用 CPU 合成谱，独立于真实训练、验证和测试数据，无网络或文件副作用。
"""
import unittest

import numpy as np
import torch

from .h_global_features import candidate_features, observed_support, FEATURE_NAMES


class GlobalFeatureTests(unittest.TestCase):
    def query(self,xs,weights,windows=None):
        return dict(source='nmrshiftdb2_2024',observed_xs=xs,observed_weights=weights,
            measure='observed_assigned_atom_equal_partial_possible',q=np.repeat(xs[0],128),
            observation=dict(kind='unknown' if not windows else 'partial_window',windows=windows or []))

    def features(self,pred,query):
        return candidate_features(torch.tensor([pred],dtype=torch.float32),torch.zeros(1,len(pred),dtype=torch.long),
            torch.tensor([4.]),torch.tensor([1.]),query,{}, {})[0]

    def test_duplicate_equivalent_hydrogen_representation_is_invariant(self):
        expanded=self.query([1.,1.,1.,2.],[1.,1.,1.,1.])
        grouped=self.query([1.,2.],[3.,1.])
        a=self.features([.1,.1,.2,.2],expanded); b=self.features([.1,.1,.2,.2],grouped)
        self.assertTrue(torch.equal(a,b))
        support=observed_support(expanded,{})
        self.assertEqual(support['reported'],4.)
        self.assertEqual(support['positions'],2)

    def test_window_features_ignore_outside_support_location(self):
        query=self.query([1.04],[1.],[[1.01,1.1]])
        a=self.features([.1,.102,.106,.8],query)
        b=self.features([.03,.102,.106,.9],query)
        self.assertTrue(torch.allclose(a,b,atol=1e-6),str(a-b))
        self.assertAlmostEqual(float(a[0,FEATURE_NAMES.index('log_supports_003_ratio')]),np.log(2),places=5)

    def test_empty_window_has_finite_explicit_indicator(self):
        result=self.features([.1,.2,.3,.4],self.query([7.],[1.],[[6.,8.]]))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(float(result[0,FEATURE_NAMES.index('empty_window_support')]),1.)
        self.assertGreaterEqual(float(result[0,FEATURE_NAMES.index('predicted_span')]),0.)
        self.assertGreater(float(result[0,FEATURE_NAMES.index('predicted_nearest_mean')]),0.)

    def test_true_identity_fields_do_not_change_features(self):
        query=self.query([1.,2.],[3.,1.])
        a=self.features([.1,.1,.2,.2],dict(query,smiles='CCO',record_id='A',group='a'))
        b=self.features([.1,.1,.2,.2],dict(query,smiles='c1ccccc1',record_id='B',group='b'))
        self.assertTrue(torch.equal(a,b))

    def test_hmdb_heights_are_not_integrals(self):
        query=dict(source='HMDB',record_id='x',q=np.array([1.,1.02,7.]),observation=dict(kind='unknown',windows=[]))
        raw={'x':dict(value=dict(x=[1.,1.02,7.],y=[1000.,3000.,1.]))}
        support=observed_support(query,raw)
        self.assertEqual(support['positions'],2)
        self.assertFalse(support['absolute_count'])
        self.assertTrue(np.array_equal(support['weights'],[1.,1.]))


if __name__=='__main__':
    unittest.main()

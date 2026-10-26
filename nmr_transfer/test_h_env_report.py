"""HMDB预测环境与报告过程的同步合成检查，无真实数据/训练/网络副作用。
核对最大簇宽与重复分位质量无关、空非交换谱回退、漏报环境可跳过且计数有代价。
依赖h_env_report与旧count_nll；未提供积分时不允许把峰高当H个数。
"""
import unittest
import itertools
import numpy as np
import torch
from .h_env_report import environment_points,visible_fallback,fit_count_mixture,environment_features,environment_score,soft_partial_match


class TestEnvironmentReport(unittest.TestCase):
    def test_bounded_clusters_and_duplicate_mass_does_not_change_centers(self):
        pred=torch.tensor([[.1,.1,.105,.11,.115],[.1,.105,.105,.11,.115]])
        atoms,mask=environment_points(pred);torch.testing.assert_close(atoms,torch.tensor([[1.025,1.125],[1.025,1.125]]),rtol=0,atol=2e-6);self.assertTrue(bool(mask.all()))

    def test_empty_fixed_visibility_falls_back_without_changing_other_rows(self):
        full=torch.tensor([[.3,.4],[.2,.8]]);fixed=torch.tensor([[1e6,1e6],[.2,.8]]);result,empty=visible_fallback(full,fixed);torch.testing.assert_close(result,full);self.assertEqual(empty.tolist(),[True,False])

    def test_partial_environment_matching_keeps_missing_peak_and_count_penalty_explicit(self):
        atoms=torch.tensor([[1.,3.,8.],[1.,3.,0.]]);mask=torch.tensor([[True,True,True],[True,True,False]]);entry=fit_count_mixture([(3,2),(3,2),(2,2)]);priors={p:{f:{v:entry for v in ['full','fixed']} for f in ['water','nonwater']} for p in ['normal','dropout']};q=dict(observation=dict(windows=[],kind='unknown'));ff=environment_features((atoms,mask),(atoms,mask),q,[1.,3.],priors,False);cfg=dict(visibility='full',shift_sigma=0.,beta=0.);torch.testing.assert_close(environment_score(ff,cfg),torch.zeros(2));penalized=environment_score(ff,dict(cfg,beta=1.));self.assertTrue(bool(torch.isfinite(penalized).all()));self.assertTrue(bool((penalized>0).all()))

    def test_soft_matching_equals_explicit_subset_average(self):
        atoms=torch.tensor([[1.,2.,3.,8.],[1.,3.,0.,0.]],dtype=torch.float64);mask=torch.tensor([[True,True,True,True],[True,True,False,False]]);observed=torch.tensor([1.,3.],dtype=torch.float64);actual=soft_partial_match(atoms,mask,observed)
        expected=[]
        for row,valid in zip(atoms,mask):
            values=row[valid];costs=torch.stack([(values[list(ids)]-observed).abs().sum()/.2 for ids in itertools.combinations(range(len(values)),2)]);expected.append((-torch.logsumexp(-costs,0)+np.log(len(costs)))/2)
        torch.testing.assert_close(actual,torch.stack(expected),rtol=1e-6,atol=1e-6);self.assertGreater(float(actual[0]),float(actual[1]))

    def test_soft_overreport_fallback_is_finite(self):
        atoms=torch.tensor([[1.,0.]]);mask=torch.tensor([[True,False]]);observed=torch.tensor([1.,1.2]);actual=soft_partial_match(atoms,mask,observed);torch.testing.assert_close(actual,torch.tensor([.5]),rtol=1e-6,atol=1e-6)


if __name__=='__main__':unittest.main()

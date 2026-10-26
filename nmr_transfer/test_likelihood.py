"""核密度排序的独立数学检查：概率质量、过宽谱惩罚及已知缺测窗口条件化。
仅人工张量，不读取真实val/test。确保窗外质量被条件化，而数值空窗口不能假得高分。
"""
import unittest
import numpy as np
import torch
from .likelihood import density,log_score,interval_mass


class TestLikelihood(unittest.TestCase):
    def test_overlapping_windows_count_probability_once(self):
        kernel=density(torch.full((1,128),.3),'1H',.1)
        torch.testing.assert_close(interval_mass(kernel,[[0.,4.],[2.,6.]]),
                                   interval_mass(kernel,[[0.,6.]]))

    def test_dense_coverage_does_not_beat_supported_peak(self):
        predictions=torch.stack([torch.full((128,),.3),torch.linspace(-.2,1.6,128)])
        kernel=density(predictions,'1H',.1)
        q=dict(q=np.full(128,3.,np.float32),observation={'kind':'unknown','windows':[]})
        score=log_score(kernel,q,'1H')
        self.assertLess(float(score[0]),float(score[1]))
        torch.testing.assert_close(kernel['cdf'][:,-1],torch.ones(2),rtol=1e-3,atol=1e-3)

    def test_unmeasured_half_mass_is_conditioned_out(self):
        p=torch.stack([torch.full((128,),.3),torch.cat([torch.full((64,),.3),torch.full((64,),1.)])])
        q=dict(q=np.full(128,3.,np.float32),observation={'kind':'partial_window','windows':[[0.,6.]]})
        score=log_score(density(p,'1H',.1),q,'1H')
        self.assertAlmostEqual(float(score[0]),float(score[1]),places=4)

    def test_no_mass_in_measured_window_is_rejected(self):
        p=torch.stack([torch.full((128,),.3),torch.full((128,),100.)])
        q=dict(q=np.full(128,3.,np.float32),observation={'kind':'window','windows':[[0.,6.]]})
        score=log_score(density(p,'1H',.1),q,'1H')
        self.assertTrue(torch.isfinite(score).all());self.assertGreater(float(score[1]),float(score[0])+100.)


if __name__=='__main__':unittest.main()

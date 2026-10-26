"""HMDB峰高置信度的合成同步测试，不使用真实失败样例。
核对零污染与原等峰似然一致、重复行不会变成积分、缺失峰高回退等峰处理，
以及均匀污染模型为低高度不匹配位置支付概率代价。无文件或网络副作用。
"""
import unittest
import numpy as np
import torch
from .h_line_confidence import line_table,confidence_nll
from .likelihood import density,log_score


class TestLineConfidence(unittest.TestCase):
    def test_zero_contamination_matches_equal_peak_likelihood(self):
        kernel=density(torch.tensor([[.1,.3,.8],[.12,.28,.81]]),'1H',.2);centers=np.array([1.,3.,8.]);q=dict(q=centers,observation=dict(kind='unknown',windows=[]))
        torch.testing.assert_close(confidence_nll(kernel,q,centers,np.array([1.,.2,.1]),0.,.5),log_score(kernel,q,'1H'),atol=2e-6,rtol=1e-6)

    def test_duplicate_lines_use_max_height_and_never_integrate(self):
        raw=dict(value=dict(x=[1.,1.,2.,2.02,8.],y=[10.,10.,9.,8.,1.]));x,h=line_table(raw);np.testing.assert_allclose(x,[1.,2.01,8.]);np.testing.assert_allclose(h,[1.,.9,.1]);x,h=line_table(raw,threshold=.15);np.testing.assert_allclose(x,[1.,2.01])

    def test_absent_height_has_no_contamination_and_keeps_lines(self):
        x,h=line_table(dict(value=dict(x=[1.,3.,8.])));np.testing.assert_equal(h,np.ones(3))
        kernel=density(torch.tensor([[.1,.3,.8]]),'1H',.2);q=dict(q=x,observation=dict(kind='unknown',windows=[]));torch.testing.assert_close(confidence_nll(kernel,q,x,h,.8,.5),log_score(kernel,q,'1H'),atol=2e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()

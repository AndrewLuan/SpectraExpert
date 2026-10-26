"""密度报告过程的同步合成检查，不读取真实谱、模型或远程资源。
验证纯交换候选回退只改变指定行、密度CDF契约保持，以及零计数系数精确保留旧分数。
依赖h_env_density，不启动训练、不写入数据或调用外部服务。
"""
import unittest
import torch
from .h_env_density import replace_empty_support,density_score


class TestEnvironmentDensity(unittest.TestCase):
    def test_visibility_fallback_changes_only_empty_candidate_rows(self):
        old=dict(logp=torch.tensor([[1.,2.],[3.,4.]]),cdf=torch.tensor([[0.,0.,0.],[0.,.5,1.]]),low=-10.,step=.1);full=dict(logp=torch.tensor([[5.,6.],[7.,8.]]),cdf=torch.tensor([[0.,.4,1.],[0.,.8,1.]]));new=replace_empty_support(old,full,torch.tensor([True,False]));torch.testing.assert_close(new['logp'][0],full['logp'][0],rtol=0,atol=0);torch.testing.assert_close(new['logp'][1],old['logp'][1],rtol=0,atol=0);torch.testing.assert_close(new['cdf'][1],old['cdf'][1],rtol=0,atol=0);self.assertEqual(new['low'],old['low']);torch.testing.assert_close(old['logp'],torch.tensor([[1.,2.],[3.,4.]]))

    def test_zero_reporting_weight_preserves_legacy_score(self):
        old=torch.tensor([-.4,1.]);features=dict(scores={(False,'legacy'):old},count=torch.tensor([2.,3.]));actual=density_score(features,dict(visibility_fallback=False,shift='legacy',beta=0.));torch.testing.assert_close(actual,old,rtol=0,atol=0)


if __name__=='__main__':unittest.main()

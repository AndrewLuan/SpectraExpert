"""功能概述：检查正观测容量匹配的单次使用、漏峰、频窗和聚簇边界。

使用小型可枚举反例验证动态规划，不读取真实验证/测试数据或访问网络。
"""
import itertools
import unittest

import torch

from .h_global_capacity import cluster_predictions,capacity_excess


class CapacityTests(unittest.TestCase):
    def test_one_prediction_cannot_freely_explain_two_observations(self):
        c=torch.tensor([[1.],[1.]]);valid=torch.ones_like(c,dtype=torch.bool)
        self.assertGreater(float(capacity_excess(c,valid,[.9,1.1])[0]),1.)
        c=torch.tensor([[.9,1.1]])
        self.assertAlmostEqual(float(capacity_excess(c,torch.ones_like(c,dtype=torch.bool),[.9,1.1])),0.,6)

    def test_unreported_predictions_are_free(self):
        c=torch.tensor([[0.,1.,2.,8.]])
        self.assertAlmostEqual(float(capacity_excess(c,torch.ones_like(c,dtype=torch.bool),[1.,2.])),0.,6)

    def test_explicit_window_discards_outside_capacity(self):
        c=torch.tensor([[1.,1.2]]);valid=torch.ones_like(c,dtype=torch.bool)
        unconstrained=capacity_excess(c,valid,[1.,1.1])
        penalty=capacity_excess(c,valid,[1.,1.1],[(.8,1.15)])
        self.assertGreater(float(penalty),float(unconstrained))
        one=capacity_excess(c[:,:1],valid[:,:1],[1.,1.1])
        self.assertTrue(torch.allclose(penalty,one))

    def test_max_width_clustering_does_not_chain(self):
        c,valid=cluster_predictions(torch.tensor([[0.,0.,.07,.14]]),.08)
        self.assertEqual(int(valid.sum()),2)
        self.assertTrue(torch.allclose(c[0,:2],torch.tensor([.07/3,.14])))

    def test_dynamic_program_matches_brute_force(self):
        p=torch.tensor([[.1,.4,.9],[.2,.5,1.]])
        obs=torch.tensor([.15,.3,.7]);sigma=.2;outlier=.5;missing=(outlier/sigma)**2
        actual=capacity_excess(p,torch.ones_like(p,dtype=torch.bool),obs,sigma=sigma,outlier=outlier)
        for row,value in zip(p,actual):
            independent=sum(min(missing,float(((row-x)/sigma).square().min())) for x in obs)
            best=float('inf')
            for choices in itertools.product(range(-1,len(row)),repeat=len(obs)):
                used=[i for i in choices if i>=0]
                if len(set(used))!=len(used):continue
                cost=sum(missing if j<0 else min(missing,float(((row[j]-obs[i])/sigma)**2)) for i,j in enumerate(choices))
                best=min(best,cost)
            self.assertAlmostEqual(float(value),(best-independent)/len(obs),5)


if __name__=='__main__':unittest.main()

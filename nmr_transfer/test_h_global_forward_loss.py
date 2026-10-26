"""功能概述：独立检查检索微调观测损失的窗口、排序方向和教师梯度契约。

CPU 合成例验证正确候选获得较低 NLL、显式窗外成分按窗口质量归一化、未知覆盖
允许未观察支持，排序梯度推动正例变好且负例变差，教师本身不接收锚梯度。
本文件不访问原始谱、模型权重、网络或 GPU；实际数据训练另由 smoke 执行。
"""
import unittest

import torch

from .h_global_forward_loss import observed_nll,pairwise_rank_loss,teacher_anchor


class GlobalForwardLossTests(unittest.TestCase):
    def test_matching_candidate_has_lower_observation_cost(self):
        p=torch.tensor([[.1,.1,.7,.7],[.2,.2,.8,.8]],requires_grad=True)
        q=torch.tensor([[.1,.1,.7,.7]]).expand(2,-1)
        costs=observed_nll(p,q,torch.zeros(2,1,2),torch.zeros(2,1,dtype=torch.bool))
        self.assertLess(float(costs[0].detach()),float(costs[1].detach()))
        costs.sum().backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_explicit_window_conditions_missing_band(self):
        p=torch.tensor([[.1,.1,.8,.8]])
        q=torch.tensor([[.1,.1,.1,.1]])
        windows=torch.tensor([[[0.,.4]]]);mask=torch.ones(1,1,dtype=torch.bool)
        partial=observed_nll(p,q,windows,mask)
        all_inside=observed_nll(torch.tensor([[.1,.1,.1,.1]]),q,windows,mask)
        self.assertTrue(torch.allclose(partial,all_inside,atol=1e-5))

    def test_unknown_observation_does_not_force_all_predicted_peaks(self):
        p=torch.tensor([[.1,.8,.8,.8]])
        q=torch.tensor([[.1,.1,.1,.1]])
        unknown=observed_nll(p,q,torch.zeros(1,1,2),torch.zeros(1,1,dtype=torch.bool))
        complete=observed_nll(p,q,torch.zeros(1,1,2),torch.zeros(1,1,dtype=torch.bool),complete=torch.ones(1,dtype=torch.bool))
        self.assertLess(float(unknown),float(complete))

    def test_overlapping_windows_equal_their_union(self):
        p=torch.tensor([[.1,.1,.8,.8]])
        q=torch.tensor([[.1,.1,.1,.1]])
        split=observed_nll(p,q,torch.tensor([[[0.,.3],[.2,.4]]]),torch.ones(1,2,dtype=torch.bool))
        union=observed_nll(p,q,torch.tensor([[[0.,.4],[0.,0.]]]),torch.tensor([[True,False]]))
        self.assertTrue(torch.allclose(split,union,atol=1e-6))

    def test_ranking_gradient_improves_positive_relative_to_negatives(self):
        costs=torch.tensor([[2.,1.,3.]],requires_grad=True)
        pairwise_rank_loss(costs).backward()
        self.assertGreater(float(costs.grad[0,0]),0.)
        self.assertTrue((costs.grad[0,1:]<0).all())

    def test_teacher_reference_is_frozen(self):
        p=torch.tensor([[.1,.2]],requires_grad=True);teacher=torch.tensor([[.2,.3]],requires_grad=True)
        teacher_anchor(p,teacher).backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(p.grad).all())


if __name__=='__main__':
    unittest.main()

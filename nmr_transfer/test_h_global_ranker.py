"""功能概述：验证训练难负样本重排的召回上界、基线回退和身份隔离。

小型合成缓存覆盖 alpha=0 的完整库排名精确保持、训练补正例不能进入评测短名单、
相同数值特征下零初始化残差、观测分组标准化不利用补入标签以及大粗分数保序。
不接触真实 train/val/test，不启动训练或访问网络；实际验证集稳定性由研究流程复评。
"""
import unittest

import torch

from .h_global_features import FEATURE_NAMES
from .h_global_ranker import ResidualRanker, rank_one, row_center, normalized_rows


class GlobalRankerTests(unittest.TestCase):
    def test_zero_alpha_retains_full_library_rank_and_top10(self):
        row=dict(coarse_rank=1234,coarse_top10=[8,4,1],candidate_indices=torch.tensor([8,4,1]),
                 target_in_coarse_pool=False,target_position=-1,rerank_eligible=True)
        rank,top=rank_one(row,torch.tensor([3.,2.,1.]),torch.tensor([-9.,0.,9.]),0.)
        self.assertEqual((rank,top),(1234,[8,4,1]))

    def test_training_appended_positive_does_not_break_recall_ceiling(self):
        row=dict(coarse_rank=1234,coarse_top10=[8,4,1],candidate_indices=torch.tensor([8,4,1,99]),
                 target_in_coarse_pool=False,target_position=3,rerank_eligible=True)
        rank,top=rank_one(row,torch.tensor([3.,2.,1.,-100.]),torch.tensor([-9.,0.,9.,-100.]),1.)
        self.assertEqual(rank,1234)
        self.assertNotIn(99,top)

    def test_zero_initialized_ranker_preserves_base_scores(self):
        model=ResidualRanker(list(range(len(FEATURE_NAMES))),typed=True)
        z=torch.randn(4,8,len(FEATURE_NAMES)); base,delta=model(z,torch.arange(4))
        self.assertTrue(torch.equal(base,z[:,:,0]))
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))

    def test_nonzero_alpha_zero_residual_does_not_change_raw_ties(self):
        raw=torch.tensor([1.,1.0000001192])
        row=dict(coarse_rank=2,coarse_top10=[1,0],candidate_indices=torch.tensor([0,1]),coarse_scores=raw,
                 target_in_coarse_pool=True,target_position=0,rerank_eligible=True,score_scale=.05)
        self.assertEqual(rank_one(row,raw/.05,torch.zeros(2),.25),(2,[1,0]))
        self.assertEqual(rank_one(row,raw/.05,torch.ones(2)*.2,.25),(2,[1,0]))

    def test_appended_positive_does_not_change_query_center(self):
        x=torch.zeros(4,len(FEATURE_NAMES)); x[:3,0]=torch.tensor([1.,2.,3.]); x[3]=100.
        row=dict(features=x,target_in_coarse_pool=False,target_position=3)
        self.assertEqual(float(row_center(row)[0]),2.)

    def test_standardization_keeps_large_base_score_order(self):
        x=torch.zeros(3,len(FEATURE_NAMES)); x[:,0]=torch.tensor([1.,100.,1000.])
        row=dict(features=x,target_in_coarse_pool=True,target_position=0)
        result=normalized_rows([row],torch.ones(len(FEATURE_NAMES)))[0]['z']
        self.assertTrue(torch.equal(result[:,0].argsort(),torch.tensor([0,1,2])))
        self.assertGreater(float(result[2,0]),20.)


if __name__=='__main__':
    unittest.main()

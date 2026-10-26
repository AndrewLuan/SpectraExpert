"""功能概述：用 CPU 合成候选验证新双通道召回的池成员与观测边界。

RecallTests 覆盖重叠去重后的交替补池、稳定平分、默认固定库门禁、全库精确 512、
明确窗口及 unknown 覆盖、分块计算与非有限输入。测试不读取数据/真实测试结果，
不调用模型前向或外部服务，也不写缓存；作为 h_research_recall 的同步单元测试。
"""
import unittest

import torch

from .h_research_recall import (FIXED_CANDIDATES, combine_shortlists,
                                directional_kernel_scores, directional_scores,
                                mixed_pool_details, stable_order)


class RecallTests(unittest.TestCase):
    def test_overlap_alternating_refill_has_exact_unique_size(self):
        coarse = torch.arange(12.).float()
        # 初始两路都是 0,1；补池 coarse 2, positive 5, coarse 3, positive 4。
        positive = torch.tensor([0., 1., 8., 9., 3., 2., 6., 7., 10., 11., 12., 13.])
        result = mixed_pool_details(coarse, positive, pool_size=6, per_channel=2,
                                    expected_candidates=None)
        self.assertEqual(result['indices'].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(result['initial_membership'].tolist(), [3, 3, 0, 0, 0, 0])
        self.assertEqual(result['refill_channel'].tolist(), [3, 3, 1, 1, 2, 2])

    def test_initial_positive_channel_can_recover_outside_coarse_prefix(self):
        coarse = torch.arange(12.).float()
        positive = coarse.clone(); positive[10:] = torch.tensor([-2., -1.])
        pool = combine_shortlists(coarse, positive, 4, 2, expected_candidates=None)
        self.assertEqual(pool.tolist(), [0, 1, 10, 11])

    def test_stable_ties_and_fixed_library_size_are_explicit(self):
        scores = torch.zeros(12)
        self.assertEqual(stable_order(scores).tolist(), list(range(12)))
        self.assertEqual(combine_shortlists(scores, scores, 8, 4, None).tolist(), list(range(8)))
        with self.assertRaisesRegex(ValueError, 'library_size_changed'):
            combine_shortlists(scores, scores)

    def test_full_frozen_library_emits_exact_512_without_truth_input(self):
        coarse = torch.arange(FIXED_CANDIDATES).float()
        pool = combine_shortlists(coarse, coarse.flip(0))
        self.assertEqual(pool.numel(), 512)
        self.assertEqual(pool.unique().numel(), 512)
        self.assertEqual(pool[:256].tolist(), list(range(256)))
        self.assertEqual(pool[256:].tolist(), list(range(FIXED_CANDIDATES - 256, FIXED_CANDIDATES)))

    def test_positive_scores_have_no_unreported_support_penalty(self):
        centers = torch.tensor([[1., 1.], [1., 10.]])
        tags = torch.zeros_like(centers, dtype=torch.long)
        score = directional_scores(centers, tags, [1.])
        # 第二候选因正观测密度质量稀释付 log2；没有对 10ppm 的距离加惩罚。
        self.assertAlmostEqual(float(score[1] - score[0]), float(torch.log(torch.tensor(2.))), places=5)
        changed = directional_scores(torch.tensor([[1., 30.]]), tags[:1], [1.])
        self.assertTrue(torch.equal(score[1:], changed))

    def test_explicit_window_filters_support_and_chunking_is_invariant(self):
        centers = torch.tensor([[1., 10.], [1.1, 9.], [1.2, -2.]])
        tags = torch.zeros_like(centers, dtype=torch.long)
        a = directional_scores(centers, tags, [1.], windows=[[.5, 1.5]], chunk_size=1)
        b = directional_scores(centers, tags, [1.], windows=[[.5, 1.5]], chunk_size=3)
        self.assertTrue(torch.equal(a, b))
        ref = directional_scores(centers[:, :1], tags[:, :1], [1.])
        self.assertTrue(torch.equal(a, ref))

    def test_kernel_unknown_does_not_infer_half_window(self):
        # 密度单位与 CDF 一致；在已知窗口内两个候选条件密度相同。
        kernel = dict(logp=torch.tensor([[.1, .1, .4, .4], [.4, .4, .1, .1]]).log(),
                      cdf=torch.tensor([[0., .1, .2, .6, 1.], [0., .4, .8, .9, 1.]]),
                      low=0., step=1.)
        unknown = directional_kernel_scores(kernel, dict(xs=[0.], weights=[1.], windows=[]))
        known = directional_kernel_scores(kernel, dict(xs=[0.], weights=[1.], windows=[[-.5, 1.5]]))
        self.assertAlmostEqual(float(unknown[0] - unknown[1]), float(torch.log(torch.tensor(4.))), places=5)
        self.assertAlmostEqual(float(known[0] - known[1]), 0., places=5)

    def test_invalid_inputs_and_outside_window_rejected(self):
        with self.assertRaises(ValueError):
            stable_order(torch.tensor([float('nan')]))
        with self.assertRaises(ValueError):
            directional_scores(torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.long),
                               [1.], windows=[[2., 3.]])
        with self.assertRaises(ValueError):
            combine_shortlists(torch.ones(8), torch.ones(8), 4, 3, None)


if __name__ == '__main__':
    unittest.main()

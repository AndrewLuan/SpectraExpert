"""功能概述：只用合成数值验证Top10纯数学的边界，不读取真实谱、checkpoint或历史结果。
测试覆盖native EPS/混合dtype舍入、真实双role上界与主桶、固定/当前对手和Top10集合差，
单rolelistwise与anchor有限差分、冻结末态饱和导数、完整36双role碰撞、采样近邻及空分母。
本文件仅导入top10_barrier_math；unittest入口无科学执行资格，也不调用模型或外部服务。
"""
import math
import unittest
from top10_barrier_math import (DTYPES, native_threshold, native_cost_replay, native_rank,
    diagnose_costs, loss_probe, two_role_collision, nearest_reference, distribution,
    availability, paired_cap_difference, native_array_digest)

D = dict(coarse_dtype="torch.float64", correction_dtypes=["torch.float32"] * 2,
         final_cost_dtype="torch.float64")


def sample(coarse, scales=(1., 1.), cap=5, gamma=.25, deltas=None, target=11):
    ids = list(range(len(coarse)))
    deltas = deltas or [[0.] * len(ids), [0.] * len(ids)]
    costs = [native_cost_replay(b, deltas[0][i], deltas[1][i], *scales, gamma,
                               D["coarse_dtype"], D["correction_dtypes"]) for i, b in enumerate(coarse)]
    return diagnose_costs(ids, coarse, costs, target, gamma, scales, cap, D,
                          role_deltas=deltas, old_other_id=0)


class MathTests(unittest.TestCase):
    def test_native_epsilon_and_dtype(self):
        self.assertEqual(native_threshold(2e8, DTYPES[0]), 2e8)
        self.assertGreater(native_threshold(1., DTYPES[1]), 1.)
        self.assertEqual(native_rank([1.] * 12, 0, DTYPES[0]), 12)
        self.assertNotEqual(native_array_digest([1.], DTYPES[0]), native_array_digest([1.], DTYPES[1]))
        x = native_cost_replay(1e8, 5., 5., 1., 1., .25, DTYPES[0], [DTYPES[0]] * 2)
        self.assertEqual(x, 1e8)

    def test_three_buckets_and_absent(self):
        success = sample([0.] * 11 + [-1.])
        blocked = sample([0.] * 11 + [100.])
        opened = sample([0.] * 12)
        absent = sample([0.] * 12, target=99)
        self.assertEqual(success["bucket"], "S0_actual_top10")
        self.assertEqual(blocked["bucket"], "S1_fail_static_blocked")
        self.assertEqual(blocked["static_unbeatable_count"], 11)
        self.assertEqual(opened["bucket"], "S2_fail_static_open")
        self.assertEqual(opened["still_need_to_beat"], 2)
        self.assertFalse(opened["top10_threshold_equivalence"])
        self.assertEqual(absent["bucket"], "P0_target_absent")
        self.assertIsNone(absent["static_unbeatable_count"])

    def test_two_role_scale_bound_not_single_head(self):
        row = sample([0.] * 11 + [20.], scales=(1., 3.), gamma=.75)
        self.assertEqual(row["pairwise_bound"], 25.)
        self.assertTrue(row["pairs"]["immutable_old_other"]["necessary_bound_released"])
        self.assertEqual(row["bucket"], "S2_fail_static_open")
        self.assertEqual(row["pairs"]["current_negative10"]["candidate_index"], 9)

    def test_role_gain_native_replay_and_pairdiff(self):
        deltas = [[0.] * 11 + [-2.], [0.] * 11 + [-4.]]
        a = sample([0.] * 11 + [3.], deltas=deltas)
        pair = a["pairs"]["immutable_old_other"]
        self.assertEqual(pair["role_gains"], [2., 4.])
        self.assertEqual(pair["combined_gain"], 2.5)
        self.assertEqual(pair["real_margin"], .5)
        b = dict(a, top10_candidate_ids=[*range(1, 10), 11], rank=10, bucket="S0_actual_top10")
        diff = paired_cap_difference(a, b)
        self.assertEqual(diff["C5_only"], [0])
        self.assertEqual(diff["C10_only"], [11])

    def test_loss_finite_difference_anchor_support(self):
        coarse, delta = [0., .4, 5.], [-1., 1., 0.]
        logits = [5 * math.atanh(v / 5) for v in delta]
        probe = loss_probe(coarse, delta, 2., 0, [True, True, False], logits, 5)
        self.assertEqual(probe["probabilities"][2], 0)
        self.assertEqual(probe["total_gradient_delta"][2], 0)
        self.assertAlmostEqual(sum(probe["probabilities"]), 1.)
        self.assertTrue(probe["anchor_opposes_listwise"][0])
        for index in range(2):
            h = 1e-6
            losses = []
            for sign in (-1, 1):
                changed = delta.copy()
                changed[index] += sign * h
                v = loss_probe(coarse, changed, 2., 0, [True, True, False], logits, 5)
                losses.append(v["listwise_loss"] + v["anchor_loss"])
            self.assertAlmostEqual((losses[1] - losses[0]) / (2 * h),
                                   probe["total_gradient_delta"][index], places=8)
        self.assertEqual(loss_probe([], [], 0, -1, [], [], 5, eligible=False)["status"],
                         "not_training_query")
        saturated = loss_probe([0., 0.], [-5., 5.], 1., 0, [True, True], [-1000., 1000.], 5)
        self.assertEqual(saturated["total_gradient_logit"], [0., 0.])

    def test_complete36_collision(self):
        a = dict(values=[0.] * 36, dtype=DTYPES[0], kind=0)
        b = dict(values=[0.] * 35 + [1.], dtype=DTYPES[0], kind=0)
        inputs = {"no_aug": {0: a, 1: a}, "aug": {0: a, 1: b}}
        self.assertFalse(two_role_collision(inputs, 0, 1)["both_roles_full36_equal"])
        inputs["aug"][1] = a
        self.assertTrue(two_role_collision(inputs, 0, 1)["both_roles_full36_equal"])
        self.assertIsNone(two_role_collision({"no_aug": inputs["no_aug"]}, 0, 1)
                          ["both_roles_full36_equal"])

    def test_nearest_reference_filter_and_unknown(self):
        query = dict(cap=5, role="aug", kind=0, query_group="q", centered36=[0.] * 36)
        same = dict(query, reference_hash="0", candidate_index=0)
        other = dict(query, query_group="other", centered36=[1.] * 36,
                     reference_hash="1", candidate_index=1)
        result = nearest_reference(query, [same, other])
        self.assertEqual(result["reference_count"], 1)
        self.assertEqual(result["centered36_RMS"], 1.)
        self.assertEqual(nearest_reference(query, [same])["status"], "unknown")
        self.assertIsNone(distribution([])["mean"])
        self.assertEqual(availability([None, "not_applicable", 0])["known"], 1)


if __name__ == "__main__":
    unittest.main()

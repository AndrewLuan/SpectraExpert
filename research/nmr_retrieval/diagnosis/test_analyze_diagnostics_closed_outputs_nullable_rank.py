"""功能概述：最小描述性副本的纯合成回归测试。
复用已通过的20个合成场景，并核outside_rank=None保留unknown及FP32原式loss标量消减；
所有值在内存构造，不读真实shard，不运行模型，不改变原容差/数学定义。
"""
import math
import unittest
import torch
import analyze_diagnostics_closed_outputs_nullable_rank as current
import test_analyze_diagnostics_closed_outputs as base

base.a=current


class NullableRankTests(unittest.TestCase):
    def test_missing_outside_rank_keeps_unknown(self):
        row,heldout=base.make_saved_row("external",miss=True)
        row["cached_outside_pool_rank"]=None
        current.validate_saved_shard(row,heldout)
        obj=row["caps"][5];roles=row["roles"][5]
        diag=current.bm.diagnose_costs(row["candidate_indices"].tolist(),row["coarse_cost"].tolist(),
            obj["natural_cost"].tolist(),None,row["gamma"],[float(roles[r]["scale"][0])for r in current.ROLES],
            5,obj["dtypes"],outside_rank=None)
        self.assertIsNone(diag["rank"])
        self.assertEqual(diag["bucket"],"P0_target_absent")

    def test_native_FP32_scalar_cancellation_preserves_original_tolerance(self):
        row,heldout=base.make_saved_row()
        coarse=torch.full((512,),1e8,dtype=torch.float32)
        row["coarse_cost"]=coarse
        for cap in (5,10):
            row["caps"][cap]["natural_cost"]=coarse.clone()
            row["caps"][cap]["dtypes"].update(coarse_dtype="torch.float32",final_cost_dtype="torch.float32")
            for trace in row["roles"][cap].values():
                target=511;allowed=~row["strict_excluded"].clone();allowed[target]=True
                delta=trace["delta"];logit=trace["logit"]
                cost=coarse/trace["scale"][0]+delta
                p=torch.zeros_like(cost);p[allowed]=torch.softmax(-cost[allowed]/.5,0)
                indicator=torch.zeros_like(p);indicator[target]=1
                gl=torch.where(allowed,(indicator-p)/.5,torch.zeros_like(p))
                ga=torch.where(allowed,.2*delta/int(allowed.sum()),torch.zeros_like(delta))
                deriv=1-torch.tanh(logit/cap).square()
                native_list=torch.logsumexp(-cost[allowed]/.5,0)+cost[target]/.5
                anchor=.1*delta[allowed].square().mean()
                trace["actual_train_loss"]=dict(actual_normalized_costs=cost,normalized_cost_dtype=str(cost.dtype),
                    allowed=allowed,probability=p,probability_dtype=str(p.dtype),listwise_gradient_delta=gl,
                    anchor_gradient_delta=ga,total_gradient_delta=gl+ga,tanh_local_derivative=deriv,
                    listwise_gradient_logit=gl*deriv,anchor_gradient_logit=ga*deriv,total_gradient_logit=(gl+ga)*deriv,
                    listwise_loss=native_list,anchor_loss=anchor,total_loss=native_list+anchor,
                    support_count=int(allowed.sum()),gradient_scope="current_single_role_row_before_batch_mean")
                self.assertEqual(float(native_list),0.)
                self.assertFalse(math.isclose(float(native_list),math.log(512),rel_tol=current.FIXED_RTOL,abs_tol=current.FIXED_ATOL))
        current.validate_saved_shard(row,heldout)
        self.assertEqual((current.FIXED_RTOL,current.FIXED_ATOL),(2e-5,2e-6))

    def test_native_scalar_contradiction_still_fails(self):
        row,heldout=base.make_saved_row()
        row["roles"][5]["aug"]["actual_train_loss"]["total_loss"]+=.1
        with self.assertRaisesRegex(ValueError,"saved_loss_scalar_native_dtype"):
            current.validate_saved_shard(row,heldout)


def load_tests(loader,tests,pattern):
    suite=unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(base.ClosedSavedTests))
    suite.addTests(loader.loadTestsFromTestCase(NullableRankTests))
    return suite


if __name__=="__main__":
    unittest.main()

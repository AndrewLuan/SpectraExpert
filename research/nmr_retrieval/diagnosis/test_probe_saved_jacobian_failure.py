"""功能概述：失败定位器的纯合成测试，证明扫描顺序、固定容差和后端舍入诊断口径。
仅在内存创建已有worker形状；不读真实tensor、request或执行记录，不调用模型/训练/API。
"""
import math
import unittest
import torch
import probe_saved_jacobian_failure as p
import test_analyze_diagnostics_closed_outputs as fixture


def perturb(trace,cap):
    jac=trace["input_jacobians"];positions=jac["positions"]
    position=positions[0];trace["logit"][position]=math.atanh(.999)*cap
    cpu_tanh=torch.tanh(trace["logit"][position]/cap)
    adjacent=torch.nextafter(cpu_tanh,torch.tensor(float("inf"),dtype=cpu_tanh.dtype))
    hypothetical_local=1-adjacent.square()
    jac["dlogit_dinput36"][0,0]=1000.
    jac["ddelta_dinput36"][0,0]=jac["dlogit_dinput36"][0,0]*hypothetical_local
    if jac["weighted_dcost_dinput36"]is not None:
        jac["weighted_dcost_dinput36"]=jac["ddelta_dinput36"]*trace["inference_role_weight"]*trace["scale"][0]
    if trace["actual_train_loss"]is not None:
        trace["actual_train_loss"]["tanh_local_derivative"][position]=hypothetical_local
    trace["delta"][position]=cap*cpu_tanh


class ProbeTests(unittest.TestCase):
    def test_typed_bit_equal_preserves_dtype_and_signed_zero(self):
        self.assertFalse(p.vectors(torch.tensor([-0.]),torch.tensor([0.]))["bit_equal"])
        self.assertFalse(p.vectors(torch.tensor([0.]),torch.tensor([0.],dtype=torch.float64))["bit_equal"])
        self.assertTrue(p.vectors(torch.tensor([0.]),torch.tensor([0.]))["bit_equal"])

    def test_valid_saved_rows_have_no_mismatch(self):
        for append in(False,True):
            row,_=fixture.make_saved_row(append=append)
            self.assertIsNone(p.first_mismatch(row))

    def test_single_ULP_tanh_difference_can_amplify_without_new_tolerance(self):
        row,_=fixture.make_saved_row()
        trace=row["roles"][5]["no_aug"];perturb(trace,5)
        mismatch=p.first_mismatch(row)
        self.assertEqual(mismatch[:3],(5,"no_aug","natural512"))
        detail=p.detailed_trace(trace,5,"no_aug","natural512","cpu")
        self.assertFalse(detail["original_CPU_guard"]["passes_fixed_tolerance"])
        self.assertTrue(detail["same_shape_saved_loss_local"]["delta_chain"]["passes_fixed_tolerance"])
        self.assertTrue(detail["weighted_native_multiplication_order"]["from_saved_DD"]["bit_equal"])
        self.assertIsNone(detail["CUDA_algebra"])
        self.assertEqual((p.a.FIXED_RTOL,p.a.FIXED_ATOL),(2e-5,2e-6))

    def test_order_is_cap_role_then_natural_loss_like_3cd(self):
        row,_=fixture.make_saved_row(append=True)
        perturb(row["loss_roles"][5]["no_aug"],5)
        perturb(row["roles"][5]["aug"],5)
        self.assertEqual(p.first_mismatch(row)[:3],(5,"no_aug","full513_loss_only"))

    def test_FP32_saturation_and_FP64_ideal_are_reported_separately(self):
        row,_=fixture.make_saved_row();trace=row["roles"][5]["no_aug"]
        pos=trace["input_jacobians"]["positions"][0]
        trace["logit"][pos]=60.
        trace["input_jacobians"]["dlogit_dinput36"][0,0]=100000.
        detail=p.detailed_trace(trace,5,"no_aug","natural512","cpu")
        self.assertTrue(detail["original_CPU_guard"]["passes_fixed_tolerance"])
        self.assertFalse(detail["FP64_analytic_comparison"]["passes_fixed_tolerance"])
        self.assertEqual(detail["local_native_CPU"][0],0.)
        self.assertGreater(detail["local_FP64"][0],0.)

    def test_nonfinite_and_shape_fail_stop(self):
        row,_=fixture.make_saved_row();row["roles"][5]["no_aug"]["input_jacobians"]["ddelta_dinput36"][0,0]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):p.first_mismatch(row)
        row,_=fixture.make_saved_row();row["roles"][5]["no_aug"]["logit"]=torch.zeros(513)
        with self.assertRaisesRegex(ValueError,"tensor_shape"):p.first_mismatch(row)


if __name__=="__main__":
    unittest.main()

"""功能概述：Mac定位子阶段的纯合成顺序、身份join与typed trace校验。
所有tensor即时构造；不读真实数据/metadata，不执行stage或SSH，不调用模型/训练。
"""
import copy
import io
import unittest
from unittest import mock
import torch
import probe_saved_jacobian_failure_mac_stage as p
import test_analyze_diagnostics_closed_outputs as f
from test_probe_saved_jacobian_failure import perturb


def metadata(row):
    receipt={k:row[k]for k in("seed","cohort","population","pool","query_identity")}
    selected={k:row[k]for k in("seed","cohort","population","query_identity")}
    old=None
    if row["fixed400_member"]:
        selected.update(fixed400_member=True,target_candidate_index=row["target_candidate_id"],original_other_candidate_index=0,original_bound_obstructed=True)
        old=dict(target_candidate_index=row["target_candidate_id"],original_other_candidate_index=0,original_bound_obstructed=True)
    return receipt,selected,old


def perturb_and_rehash(trace,cap):
    perturb(trace,cap)
    key="trace_sha256_natural512"if len(trace["logit"])==512 else"trace_sha256_loss513"
    trace[key]=p.a.saved_trace_hashes(trace)


class MacStageTests(unittest.TestCase):
    def test_fixed_fit_natural_and_appended_identity(self):
        for append in(False,True):
            row,heldout=f.make_saved_row(append=append);receipt,selected,old=metadata(row)
            join=p.check_identity_join(row,receipt,selected,old,heldout)
            self.assertTrue(join["fixed400_member"])
            self.assertEqual(join["loss_target_position"],512 if append else 511)
            self.assertEqual(join["loss_only_positive_used"],append)

    def test_nonfit_selected_record_omits_fixed_flag_and_keeps_nullable_loss(self):
        for population in("internal","external"):
            for miss in(False,True):
                row,heldout=f.make_saved_row(population,miss=miss);receipt,selected,old=metadata(row)
                self.assertNotIn("fixed400_member",selected)
                join=p.check_identity_join(row,receipt,selected,old,heldout)
                self.assertFalse(join["fixed400_member"])
                self.assertIsNone(join["loss_target_position"])

    def test_identity_target_and_old_other_contradictions_stop(self):
        for mutate in(lambda r:r.update(target_candidate_id=7),
                      lambda r:r.update(original_other_candidate_index=8),
                      lambda r:r.update(fixed400_member=False)):
            row,heldout=f.make_saved_row();receipt,selected,old=metadata(row);mutate(row)
            with self.assertRaises(ValueError):p.check_identity_join(row,receipt,selected,old,heldout)

    def test_scan_valid_and_fixed_order(self):
        row,_=f.make_saved_row(append=True)
        self.assertEqual(p.scan_shard(row),(None,8))
        perturb_and_rehash(row["loss_roles"][5]["no_aug"],5)
        perturb_and_rehash(row["roles"][5]["aug"],5)
        mismatch,passed=p.scan_shard(row)
        self.assertEqual(mismatch[:3],(5,"no_aug","full513_loss_only"))
        self.assertEqual(passed,1)

    def test_first_natural_failure_same_original_fixed_tolerance(self):
        row,_=f.make_saved_row();trace=row["roles"][5]["no_aug"]
        perturb_and_rehash(trace,5)
        mismatch,passed=p.scan_shard(row)
        self.assertEqual(mismatch[:3],(5,"no_aug","natural512"))
        self.assertEqual(passed,0)
        self.assertEqual((p.a.FIXED_RTOL,p.a.FIXED_ATOL),(2e-5,2e-6))

    def test_saved_trace_hash_and_dtype_shape_fail(self):
        row,_=f.make_saved_row()
        row["roles"][5]["aug"]["trace_sha256_natural512"][0]="f"*64
        with self.assertRaisesRegex(ValueError,"saved_typed_trace_hash_join"):p.scan_shard(row)
        row,_=f.make_saved_row();row["roles"][5]["no_aug"]["input36"]=torch.zeros(513,36)
        with self.assertRaisesRegex(ValueError,"tensor_shape"):p.scan_shard(row)

    def test_tensor_identity_includes_dtype_shape_and_signed_bytes(self):
        a=p.tensor_identity(torch.tensor([0.]))
        b=p.tensor_identity(torch.tensor([-0.]))
        c=p.tensor_identity(torch.tensor([0.],dtype=torch.float64))
        self.assertNotEqual(a["sha256"],b["sha256"])
        self.assertNotEqual(a,c)
        self.assertEqual(a["shape"],[1])

    def test_original_guard_sequence_does_not_skip_other_failure(self):
        row,heldout=f.make_saved_row();receipt,selected,old=metadata(row)
        with mock.patch.object(p.a,"validate_saved_shard",side_effect=ValueError("earlier_original_guard")):
            with self.assertRaisesRegex(ValueError,"earlier_original_guard"):
                p.original_row_mismatch(row,heldout,selected,old)
        self.assertEqual(p.original_row_mismatch(row,heldout,selected,old),(None,4))
        with mock.patch.object(p.a,"validate_saved_shard",side_effect=ValueError("fixed_tolerance:saved_delta_Jacobian")):
            with self.assertRaisesRegex(ValueError,"original_DD_guard_not_reproduced_stop"):
                p.original_row_mismatch(row,heldout,selected,old)

    def test_loss_target_requires_exact_integer(self):
        row,heldout=f.make_saved_row();receipt,selected,old=metadata(row)
        row["target_position"]=0;row["target_candidate_id"]=0;row["candidate_groups"][0]=row["query_identity"]["group"]
        selected["target_candidate_index"]=0;old["target_candidate_index"]=0;row["loss_target_position"]=False
        with self.assertRaisesRegex(ValueError,"nullable_loss_target_join"):
            p.check_identity_join(row,receipt,selected,old,heldout)

    def test_complete_synthetic_safe_weights_only_roundtrip(self):
        row,_=f.make_saved_row(append=True)
        buffer=io.BytesIO();torch.save(row,buffer);buffer.seek(0)
        loaded=torch.load(buffer,map_location="cpu",weights_only=True)
        self.assertEqual(loaded["schema"],row["schema"])
        self.assertEqual(p.scan_shard(loaded),(None,8))

    def test_nonfinite_unchanged_failure(self):
        row,_=f.make_saved_row();row["roles"][5]["no_aug"]["input_jacobians"]["dlogit_dinput36"][0,0]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):p.scan_shard(row)


if __name__=="__main__":
    unittest.main()

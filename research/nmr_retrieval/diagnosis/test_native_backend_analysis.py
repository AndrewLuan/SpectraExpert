"""功能概述：用纯合成保存量核验原dtype重构、严格传输和输入始末pin；不加载实验模型或数据。
TestNative检查饱和导数、mask、运算次序和dtype；TestTransport检查唯一映射、SHA/bytes及修改拒绝。
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import analyze_diagnostics_native_backend as a
import prepare_native_backend_analysis_request as p


class TestNative(unittest.TestCase):
    def test_unadmitted_device_rejected(self):
        with patch.object(a, "_NATIVE_DEVICE", None), self.assertRaisesRegex(ValueError,"admitted_native"):
            a.native_device()

    def test_dtype_mismatch_rejected_even_equal_values(self):
        with self.assertRaisesRegex(ValueError,"native_dtype"):
            a.tolerant(torch.ones(3,dtype=torch.float32),torch.ones(3,dtype=torch.float64),"same")

    def test_fixed_tolerance_not_relaxed(self):
        with self.assertRaisesRegex(ValueError,"fixed_tolerance"):
            a.tolerant(torch.tensor([3e-6]),torch.zeros(1),"saturated")

    def test_loss_native_offset_and_strict_mask(self):
        for dtype in (torch.float32,torch.float64):
            delta=torch.tensor([4.,-.5,1.,-3.],dtype=dtype)
            logits=torch.tensor([40.,-1.,2.,-40.],dtype=dtype)
            coarse=torch.tensor([1e8,1e8+8,1e8+16,1e8+24],dtype=dtype)
            strict=torch.tensor([True,False,True,False])
            tr=dict(delta=delta,logit=logits,scale=torch.ones(33,dtype=dtype))
            v=a.reconstruct_native_loss(tr,coarse,strict,0,5,torch.device("cpu"))
            allowed=torch.tensor([True,True,False,True]); c=coarse/tr["scale"][0]+delta
            prob=torch.zeros_like(c); prob[allowed]=torch.softmax(-c[allowed]/.5,0)
            gl=(torch.tensor([1.,0.,0.,0.],dtype=dtype)-prob)/.5; gl[~allowed]=0
            ga=torch.where(allowed,2*.1*delta/3,torch.zeros_like(delta))
            local=1-torch.tanh(logits/5).square()
            self.assertTrue(torch.equal(v["allowed"],allowed))
            for name,expected in (("actual_normalized_costs",c),("probability",prob),
                 ("total_gradient_logit",(gl+ga)*local),
                 ("listwise_loss",torch.logsumexp(-c[allowed]/.5,0)+c[0]/.5)):
                self.assertEqual(v[name].dtype,dtype);self.assertTrue(torch.equal(v[name],expected))
            self.assertEqual(v["probability"][2].item(),0)
            loss={k:x.clone() for k,x in v.items()}
            loss.update(support_count=3,normalized_cost_dtype=str(dtype),probability_dtype=str(dtype))
            tr["actual_train_loss"]=loss
            with patch.object(a,"_NATIVE_DEVICE",torch.device("cpu")):
                a.validate_saved_loss(tr,coarse,strict,0,5)
                loss["total_gradient_logit"][1] += 1
                with self.assertRaisesRegex(ValueError,"fixed_tolerance"):
                    a.validate_saved_loss(tr,coarse,strict,0,5)

    def test_full_trace_saved_DD_chain_and_order(self):
        n=512; dtype=torch.float32
        x=torch.arange(n*33,dtype=dtype).reshape(n,33)/100
        o=torch.zeros(n,3,dtype=dtype); scale=torch.full((33,),3.,dtype=dtype)
        logits=torch.linspace(-40,40,n,dtype=dtype); pos=[0,250,511]
        dx=torch.arange(108,dtype=dtype).reshape(3,36)/7
        local=1-torch.tanh(logits[pos]/5).square(); dd=dx*local[:,None]
        tr=dict(raw_X=x,raw_O=o,scale=scale,fixed_x_center=torch.ones(33),fixed_o_center=torch.zeros(3),
                input36=torch.cat((((x-1)/scale).clamp(-20,20),o),1),logit=logits,
                delta=5*torch.tanh(logits/5),shape_kind="natural512_inference",inference_role_weight=.25)
        tr["trace_sha256_natural512"]=a.saved_trace_hashes(tr)
        tr["input_jacobians"]=dict(positions=pos,dlogit_dinput36=dx,ddelta_dinput36=dd,
                                    weighted_dcost_dinput36=dd*.25*scale[0])
        with patch.object(a,"_NATIVE_DEVICE",torch.device("cpu")):
            a.validate_saved_trace(tr,n,5,torch.zeros(n),.25,pos)
            tr["input_jacobians"]["weighted_dcost_dinput36"][1,0] += 1
            with self.assertRaisesRegex(ValueError,"saved_weighted"):
                a.validate_saved_trace(tr,n,5,torch.zeros(n),.25,pos)


class TestTransport(unittest.TestCase):
    def fixture(self,root):
        root=root.resolve()
        physical=root/"saved.json";physical.write_text('{"x":1}')
        logical="/logical/train/old.json";digest=a.file_sha(physical)
        row=dict(logical_path=logical,physical_path=str(physical),sha256=digest,bytes=physical.stat().st_size)
        mapping=dict(schema="native_analysis_exact_transport_map_v1",entries=[row])
        mp=root/"map.json";mp.write_text(json.dumps(mapping))
        return a.Pins(mapping,mp,a.file_sha(mp)),row,physical,mp,mapping

    def test_exact_logical_identity_and_no_prefix_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            pins,row,physical,mp,mapping=self.fixture(Path(t))
            self.assertEqual(pins.json(dict(path=row["logical_path"],sha256=row["sha256"])),dict(x=1))
            self.assertEqual(pins.checked,{row["logical_path"]:row["sha256"]});pins.finish()
            with self.assertRaisesRegex(ValueError,"unmapped_input"):
                pins.verify(physical,row["sha256"])

    def test_map_and_saved_bytes_changed_at_finish(self):
        for change_map in (False,True):
            with tempfile.TemporaryDirectory() as t:
                pins,row,physical,mp,mapping=self.fixture(Path(t));pins.verify(row["logical_path"],row["sha256"])
                (mp if change_map else physical).write_text("changed")
                with self.assertRaisesRegex(ValueError,"changed"):
                    pins.finish()

    def test_logical_dotdot_alias_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            pins,row,*_=self.fixture(Path(t))
            with self.assertRaisesRegex(ValueError,"canonical_exact_path"):
                pins.verify("/logical/train/sub/../old.json",row["sha256"])

    def test_physical_symlink_rejected_at_admission_and_finish(self):
        with tempfile.TemporaryDirectory() as t:
            pins,row,physical,mp,mapping=self.fixture(Path(t))
            link=physical.with_name("link.json");link.symlink_to(physical)
            other=dict(row,physical_path=str(link))
            with self.assertRaisesRegex(ValueError,"physical_path_alias"):
                a.Pins(dict(schema=mapping["schema"],entries=[other]),mp,a.file_sha(mp))
            pins.verify(row["logical_path"],row["sha256"])
            replacement=physical.with_name("replacement.json");physical.rename(replacement);physical.symlink_to(replacement)
            with self.assertRaisesRegex(ValueError,"physical_path_alias"):
                pins.finish()

    def test_duplicate_and_byte_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            pins,row,physical,mp,mapping=self.fixture(Path(t));row["bytes"]+=1
            with self.assertRaisesRegex(ValueError,"mapped_SHA_and_bytes"):
                a.Pins(mapping,mp,a.file_sha(mp)).verify(row["logical_path"],row["sha256"])
            mapping["entries"].append(row.copy())
            with self.assertRaisesRegex(ValueError,"unique_exact"):
                a.Pins(mapping,mp,a.file_sha(mp))

    def test_inventory_ambiguity_rejected(self):
        with self.assertRaisesRegex(AssertionError,"no_candidate_choice"):
            p.exact_entries(dict(entries=[dict(candidates=[{},{}])]))


if __name__=="__main__":
    unittest.main()

"""功能概述：继承原8项后端/传输测试，并以合成512候选核验legacy未知与worker已知role证据的完整连接。
测试只投影四个legacy未知字段；共享字段篡改、未采样unknown、worker角色篡改和遗漏/重复连接均必须拒绝。
"""
import ast
import copy
import inspect
import unittest
import numpy as np
import torch
import independent_native_backend_checker_role_evidence as c
import test_independent_native_backend_checker as base

base.c=c
TestAlgebra=base.TestAlgebra
TestTransport=base.TestTransport


class TestRoleEvidence(unittest.TestCase):
    def setUp(self):c.np=np;c.torch=torch

    def pair(self,target=20,old=2):
        ids=list(range(512));coarse=np.arange(512,dtype=np.float64)
        dtypes=dict(coarse_dtype="torch.float64",correction_dtypes=["torch.float64"]*2,final_cost_dtype="torch.float64")
        delta=[np.linspace(-1,1,512),np.linspace(1,-1,512)]
        cost=c.native_cost(coarse,*delta,[1.,1.],.25,dtypes)
        args=(ids,coarse,cost,target,.25,[1.,1.],5,dtypes)
        return c.diagnostic(*args,old=old),c.diagnostic(*args,old=old,deltas=delta)

    def test_only_probed_role_fields_deferred_then_fully_checked(self):
        legacy,full=self.pair();frozen=copy.deepcopy(full)
        with self.assertRaisesRegex(AssertionError,"role_gains"):c.compare_diag(full,legacy)
        c.compare_legacy_diag(full,legacy,True);c.compare_diag(full,full)
        self.assertEqual(full,frozen)
        with self.assertRaisesRegex(AssertionError,"role_gains"):c.compare_legacy_diag(full,legacy,False)
        for pair in ("current_negative1","current_negative10","immutable_old_other"):
            for field in c.ROLE_FIELDS:
                changed=copy.deepcopy(full)
                if field=="role_gains":changed["pairs"][pair][field][0]+=1
                else:changed["pairs"][pair][field]+=1
                c.compare_legacy_diag(changed,legacy,True)
                with self.assertRaises(AssertionError):c.compare_diag(changed,full)

    def test_shared_fields_pair_null_and_missing_keys_stay_strict(self):
        legacy,full=self.pair()
        for name in ("coarse_gap","native_margin","candidate_index","pairwise_bound"):
            changed=copy.deepcopy(full);changed["pairs"]["current_negative1"][name]+=1
            with self.assertRaises(AssertionError):c.compare_legacy_diag(changed,legacy,True)
        legacy,full=self.pair(old=None)
        c.compare_legacy_diag(full,legacy,True)
        full["pairs"]["immutable_old_other"]={}
        with self.assertRaises(AssertionError):c.compare_legacy_diag(full,legacy,True)
        legacy,full=self.pair();del full["pairs"]["current_negative1"]["role_gains"]
        with self.assertRaisesRegex(AssertionError,"pair fields"):c.compare_legacy_diag(full,legacy,True)
        legacy,full=self.pair(target=-1);c.compare_legacy_diag(full,legacy,True)
        full["pairs"]={}
        with self.assertRaises(AssertionError):c.compare_legacy_diag(full,legacy,True)

    def test_pending_coverage_and_actual_same_key_completion_guard(self):
        k=(193,5,"dual","fake");pending={k};proofs={k:"proof"}
        with self.assertRaisesRegex(AssertionError,"complete same-key"):c.complete_legacy_role_join(pending,set(),proofs)
        c.complete_legacy_role_join(pending,{k},proofs)
        with self.assertRaises(AssertionError):c.complete_legacy_role_join(pending,{k,"orphan"},proofs)
        fn=ast.parse(inspect.getsource(c.main)).body[0]
        guard=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and
                   isinstance(n.test.left,ast.Name) and n.test.left.id=="k" and
                   any(isinstance(v,ast.Name) and v.id=="legacy_proofs" for v in n.test.comparators))
        code=compile(ast.Module(body=[guard],type_ignores=[]),"<actual-same-key-proof-guard>","exec")
        env=dict(k=k,legacy_proofs=proofs,pending_legacy_roles=pending,completed_legacy_roles=set(),need=c.need,
                 proof=lambda *args:"proof",ids=[],coarse=[],costs=[],d={},traces=[])
        exec(code,env);self.assertEqual(env["completed_legacy_roles"],{k})
        with self.assertRaisesRegex(AssertionError,"completed once"):exec(code,env)
        env["completed_legacy_roles"]=set();env["pending_legacy_roles"]=set()
        with self.assertRaisesRegex(AssertionError,"completed once"):exec(code,env)
        env["pending_legacy_roles"]=pending;env["proof"]=lambda *args:"tampered"
        with self.assertRaisesRegex(AssertionError,"cost/dtype/trace"):exec(code,env)


if __name__=="__main__":unittest.main()

"""功能概述：JJ同ref helper的纯合成边界测试；不读取真实shard，不调用模型或CUDA实测。
用内存512/513行核身份/dtype/hash，用CPU元素运算核逐步链与原容差，用N内假元数据核准入/闭包绑定。
真实JJ CUDA primitive与实际结果由独立A/B后的单次同ref阶段观察，本测试不冒充GPU实际证据。
"""
from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile,types,unittest
import numpy as np
import torch
torch.set_num_threads(4)
import synthetic_independent_closed_outputs_checker as fixtures
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("JJ_helper_synthetic",ROOT/"probe_saved_jacobian_JJ_algebra.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.torch=torch;m.np=np

def ref(p):return dict(path=str(p),sha256=m.sha(p),bytes=p.stat().st_size)
def write(p,v):p.write_text(json.dumps(v))
def fake_row(append=False,fit=True):
    row,entry=fixtures.row(append,fit)
    row.update(population="fixed_fit"if fit else"tune",kind=0,fixed400_member=fit,
        query_external_heldout=not fit,query_internal_group=False,
        original_bound_obstructed=True if fit else None,original_other_candidate_index=0 if fit else None,
        diagnostic_loss_only_positive_used=append,cache_loss_only_positive_present=append)
    entry.update(population=row["population"])
    selected=dict(seed=row["seed"],cohort=row["cohort"],population=row["population"],query_identity=row["query_identity"],
        fixed400_member=fit,external_heldout=not fit,internal_group=False,
        original_bound_obstructed=row["original_bound_obstructed"],original_other_candidate_index=row["original_other_candidate_index"])
    if fit:selected["target_candidate_index"]=row["target_candidate_id"]
    for cap in(5,10):
        for role in m.ROLES:
            row["roles"][cap][role]["trace_sha256_natural512"]=m.trace_hashes(row["roles"][cap][role])
            if append:row["loss_roles"][cap][role]["trace_sha256_loss513"]=m.trace_hashes(row["loss_roles"][cap][role])
    return row,entry,selected

class Tests(unittest.TestCase):
    def test_natural_fit_append_and_nonfit_join(self):
        for append,fit in((False,True),(True,True),(False,False)):
            r,e,s=fake_row(append,fit);join=m.identity_join(r,e,s)
            self.assertEqual(join["actual_train_query"],fit)
            self.assertEqual(join["loss_only_positive_used"],append)
    def test_synchronized_wrong_natural_loss_target_rejected(self):
        r,e,s=fake_row();r["loss_target_position"]=0
        with self.assertRaisesRegex(ValueError,"nullable_loss_positive"):m.identity_join(r,e,s)
    def test_bool_float_and_nonfit_target_rejected(self):
        for value in(True,511.,None):
            r,e,s=fake_row();r["loss_target_position"]=value
            with self.assertRaises(ValueError):m.identity_join(r,e,s)
        r,e,s=fake_row(fit=False);r["loss_target_position"]=511
        with self.assertRaises(ValueError):m.identity_join(r,e,s)
    def test_identity_index_and_fixed_target_rejected(self):
        r,e,s=fake_row();e["pool"]="single"
        with self.assertRaises(ValueError):m.identity_join(r,e,s)
        r,e,s=fake_row();s["target_candidate_index"]=0
        with self.assertRaises(ValueError):m.identity_join(r,e,s)
    def test_full513_ID_and_coarse_dtype_prefix_rejected(self):
        r,e,s=fake_row(True);r["loss_candidate_indices"]=r["loss_candidate_indices"].int()
        with self.assertRaises(ValueError):m.identity_join(r,e,s)
        r,e,s=fake_row(True);r["coarse_cost"]=torch.arange(512,dtype=torch.float64);r["loss_coarse_cost"]=torch.arange(513,dtype=torch.float32)
        self.assertTrue(torch.equal(r["coarse_cost"],r["loss_coarse_cost"][:512]))
        with self.assertRaisesRegex(ValueError,"physical_prefix"):m.identity_join(r,e,s)
    def test_trace_positions_hash_dtype_and_finite(self):
        for append in(False,True):
            r,e,s=fake_row(append);shape="full513_loss_only"if append else"natural512"
            tr=r["loss_roles"if append else"roles"][5]["no_aug"]
            sel=dict(cap=5,role="no_aug",shape=shape,positions=tr["input_jacobians"]["positions"])
            got,hs=m.selected_trace(r,sel);self.assertEqual(hs,m.trace_hashes(got))
            bad=copy.deepcopy(r);bad["loss_roles"if append else"roles"][5]["no_aug"]["logit"][0]=float("nan")
            with self.assertRaises(ValueError):m.selected_trace(bad,sel)
            badsel=dict(sel,positions=[0])
            with self.assertRaises(ValueError):m.selected_trace(r,badsel)
    def test_full513_calibration_dtype_rejected(self):
        r,e,s=fake_row(True);r["loss_roles"][5]["no_aug"]["scale"]=r["loss_roles"][5]["no_aug"]["scale"].double()
        tr=r["loss_roles"][5]["no_aug"];sel=dict(cap=5,role="no_aug",shape="full513_loss_only",positions=tr["input_jacobians"]["positions"])
        with self.assertRaisesRegex(ValueError,"same_model_cache_dtype"):m.selected_trace(r,sel)
    def test_native_steps_and_saturation_original_tolerance(self):
        r,e,s=fake_row();tr=r["roles"][5]["no_aug"];tr["logit"].fill_(60.0000038)
        j=tr["input_jacobians"];j["dlogit_dinput36"].fill_(100000.)
        path=m.native_path(tr,5,"cpu");self.assertTrue(bool((path["local"]==0).all()))
        self.assertTrue(bool((path["DD"]==0).all()))
        ideal=j["dlogit_dinput36"].double()*(1-torch.tanh(tr["logit"][j["positions"]].double()/5).square())[:,None]
        self.assertFalse(m.compare(path["DD"],ideal)["passes_fixed_tolerance"])
        self.assertEqual((m.RTOL,m.ATOL),(2e-5,2e-6))
    def test_saved_DD_weighted_link_separate(self):
        r,e,s=fake_row();tr=r["roles"][5]["no_aug"];tr["logit"].fill_(60.);tr["scale"][0]=1e6
        tr["input_jacobians"]["ddelta_dinput36"].fill_(1e-6)
        path=m.native_path(tr,5,"cpu")
        self.assertFalse(m.compare(path["weighted"],path["weighted_from_saved_DD"])["passes_fixed_tolerance"])
    def test_compare_witness_and_signed_zero(self):
        a=torch.tensor([0.,1.,2.]);b=torch.tensor([-0.,1.,3.])
        result=m.compare(a,b);self.assertFalse(result["bit_equal"]);self.assertEqual(result["mismatching_elements"],1)
        self.assertEqual(result["witnesses"][0]["coordinate"],[2])
        self.assertEqual(result["max_absolute_residual"],1.)
    def test_no_execute_before_metadata(self):
        old=m.Pins
        m.Pins=lambda:(_ for _ in()).throw(AssertionError("premature_metadata"))
        try:
            with self.assertRaisesRegex(ValueError,"explicit_execute"):m.admit(types.SimpleNamespace(execute=False))
        finally:m.Pins=old
    def test_unregistered_path_and_input_drift(self):
        with tempfile.TemporaryDirectory(dir=ROOT)as tmp:
            p=Path(tmp)/"fake.json";write(p,{})
            pins=m.Pins();pins.ref(ref(p));write(p,{"changed":True})
            with self.assertRaisesRegex(ValueError,"input_changed"):pins.finish()
        with self.assertRaises(ValueError):m.safe("relative.json")
    def test_metadata_positive_wrong_scope_and_boolean_exit(self):
        self._metadata_cases()
    def _metadata_cases(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix="synthetic-JJ-gate-")as tmp:
            d=Path(tmp);selection=dict(seed=193,cohort="train",population="fixed_fit",pool="dual",kind=0,
                query_identity=dict(query_order=0,variant="observed",source="synthetic",record_id="q",smiles="C",group="g"),
                cap=5,role="no_aug",shape="natural512",positions=[0,511])
            shard=dict(relative="tensor_shards/000000.pt",remote_path=str(d/"worker"/"tensor_shards"/"000000.pt"),sha256="a"*64,bytes=1)
            mac=dict(status="PASS_MAC_FIRST_FAILURE_LOCALIZED",historical_attempt_first_identity=None,historical_attempt_first_identity_status="UNKNOWN_NOT_RECORDED_IN_OLD_STDERR",selection=selection,identity_join={},shard=shard,trace_sha256_ordered_sha256="b"*64,selected_position_trace_sha256={"0":"c"*64})
            e=dict(actual_inner_exit=0,actual_external_exit=0,termination_reason=None,elapsed_seconds=10.)
            paths={}
            for name,v in(("mac",mac),("execution",e),("preflight",{}),("worker",{}),("index",{}),("manifest",{}),("query",{})):
                p=d/(name+".json");write(p,v);paths[name]=p
            decision=dict(status="AUTHORIZED_ONE_BOUNDED_CUDA_SAVED_VALUE_PROBE_AFTER_INDEPENDENT_GATES",runtime="original_JJ",algebra_device="cuda:0",fallback_authorized=False,new_NN_budget=0,new_weight_loads=0,fixed_rtol=m.RTOL,fixed_atol=m.ATOL)
            for k in("new_training","new_samples","test_read","scientific_definition_change_authorized","threshold_relaxation_authorized","automatic_retry_authorized"):decision[k]=False
            paths["decision"]=d/"decision.json";write(paths["decision"],decision)
            review=dict(schema="saved_jacobian_Mac_localization_independent_result_review_v1",status="PASS_MAC_LOCALIZATION_REVIEW",independent=True,mac_execution_sha256=m.sha(paths["execution"]),mac_localization_sha256=m.sha(paths["mac"]),selected_shard_sha256=shard["sha256"],selection_sha256=m.digest(selection))
            paths["review"]=d/"review.json";write(paths["review"],review)
            keys={"decision_ref":"decision","preflight_ref":"preflight","source_index_ref":"index","source_manifest_ref":"manifest","query_manifest_ref":"query","worker_source_ref":"worker","mac_localization_ref":"mac","mac_execution_ref":"execution","mac_review_ref":"review"}
            r=dict(schema=m.SCHEMA,execution_scope=m.SCOPE,helper_sha256=m.sha(m.__file__),algebra_device="cuda:0",fallback_authorized=False,new_NN_budget=0,new_weight_loads=0,training=False,test_read=False,max_seconds=120,max_cpu_threads=4,max_rss_bytes=4*1024**3,max_gpu_allocated_bytes=4*1024**3,max_output_bytes=4*1024**2,write_root=str(ROOT),selection=selection,identity_join={},shard=shard,trace_sha256_ordered_sha256=mac["trace_sha256_ordered_sha256"],selected_position_trace_sha256=mac["selected_position_trace_sha256"])
            r.update({k:ref(paths[v])for k,v in keys.items()})
            constants={"DECISION_SHA":"decision","PREFLIGHT_SHA":"preflight","INDEX_SHA":"index","MANIFEST_SHA":"manifest","QUERY_SHA":"query","WORKER_SHA":"worker"}
            old={k:getattr(m,k)for k in constants}
            try:
                for k,v in constants.items():setattr(m,k,m.sha(paths[v]))
                rp=d/"request.json";write(rp,r)
                A=dict(status="PASS",independent=True,execution_authorized=True,execution_scope=m.SCOPE,request_sha256=m.sha(rp),helper_sha256=m.sha(m.__file__),decision_sha256=m.DECISION_SHA,mac_localization_sha256=m.sha(paths["mac"]),mac_review_sha256=m.sha(paths["review"]))
                ap=d/"A.json";write(ap,A)
                def args():return types.SimpleNamespace(execute=True,request=str(rp),request_sha256=m.sha(rp),admission=str(ap),admission_sha256=m.sha(ap),out=str(d/"fresh"))
                got=m.admit(args());self.assertEqual(got[0]["_Mac_elapsed"],10.)
                A["execution_scope"]="WRONG";write(ap,A)
                with self.assertRaisesRegex(ValueError,"scope"):m.admit(args())
                A["execution_scope"]=m.SCOPE;write(ap,A);e["actual_inner_exit"]=False;write(paths["execution"],e)
                r["mac_execution_ref"]=ref(paths["execution"]);write(rp,r);A["request_sha256"]=m.sha(rp);write(ap,A)
                with self.assertRaisesRegex(ValueError,"Mac_strict_0_0"):m.admit(args())
            finally:
                for k,v in old.items():setattr(m,k,v)

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    receipt=dict(schema="saved_JJ_probe_synthetic_receipt_v1",status="PASS",source_sha256=m.sha(ROOT/"probe_saved_jacobian_JJ_algebra.py"),
        test_sha256=m.sha(__file__),tests=result.testsRun,real_payload_decodes=0,model_loads=0,new_NN_calls=0,
        CUDA_actual_execution=False,scope="pure metadata and fabricated CPU tensors")
    (ROOT/"saved_jacobian_JJ_algebra_synthetic_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))

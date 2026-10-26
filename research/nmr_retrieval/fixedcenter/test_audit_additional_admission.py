"""用途：纯合成校验 frozen admission 算术及真实 producer 形状的 driver/输入证据；不运行模型。"""
import copy,json,tempfile,unittest
from pathlib import Path
import audit_additional_admission as m

def plan():return {"schema":"fixed_train_center_conditional_preregistration_20261023","seeds":[193,211],"admission_gate":dict(m.GATE)}
def summary():
    s=dict(status="PASS",scope=m.SCOPE,seeds=[193,211],groups=64,epsilon=1e-7,inputs_unchanged=True,source_files_unchanged=True,
        initial_model_state_sha256={"193":"a"*64,"211":"b"*64},final_model_state_sha256={"193":"a"*64,"211":"b"*64},
        summary={"by_seed_variant":{}})
    for seed in m.SEEDS:
        s["summary"]["by_seed_variant"][str(seed)]={"observed":dict(queries=64,
            pair_changes={name:{"common":dict(queries=64,counts={"pairs":64000,"strict_reversals":reversals},pair_weighted_strict_reversal_rate=reversals/64000)}
                          for name,reversals in [(m.NATIVE,6400),(m.FIXED,3200)]},
            mean_RR={"D_native":.25,"D_use_both_Scenters":.5})}
    return s
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v)+"\n")
def fixture(root):
    run=root/"runs/mechanism";run.mkdir(parents=True)
    producer=root/"pool_normalization_counterfactual.py"
    producer.write_bytes((Path(__file__).parent/producer.name).read_bytes())
    assert m.sha(producer)==m.PRODUCER_SHA
    raw=root/"features/train/features.pt";raw.parent.mkdir(parents=True);raw.write_bytes(b"synthetic_train_cache_bytes")
    sourcepins={str(producer):m.sha(producer)}
    jobs=[];receipts={}
    for i,name in enumerate(("reuse_completed_paired_validation","fixed_train64_center_counterfactual")):
        out=root/"audits"/name;out.mkdir(parents=True);(out/"EXIT_CODE").write_text("0\n")
        argv=["python",str(producer),"--root",str(root),"--out",str(out)]
        inputs={} if name=="reuse_completed_paired_validation" else {str(raw):m.sha(raw)}
        j=dict(id=name,kind="audit",argv=argv,out=str(out),cwd=str(root),inputs_sha256=inputs)
        r=dict(state="complete",external_exit=0,internal_exit=0,argv=argv,output=str(out),cwd=str(root),pid=100+i,
            requested_inputs_sha256=inputs,started_utc="2026-10-23T11:01:00Z",finished_utc="2026-10-23T11:02:00Z")
        jobs.append(j);receipts[name]=r
    qp=root/"mechanism_queue.json";dump(qp,dict(jobs=jobs,source_inputs_sha256=sourcepins))
    qh=m.sha(qp);allpins={**sourcepins,str(qp):qh}
    for name,r in receipts.items():
        r["queue_source_sha256"]=allpins;dump(run/(name+".receipt.json"),r)
    launchargv=["python","-m","nmr_transfer.h_research_driver","--queue",str(qp),"--out",str(run)]
    dump(run/"launch_receipt.json",dict(internal_exit=0,external_exit=0,queue_sha256={str(qp):qh},source_sha256=sourcepins,
        bundle_sha256=sourcepins,child_pid=99,argv=launchargv,started_utc="2026-10-23T10:59:00Z",finished_utc="2026-10-23T11:04:00Z"))
    dump(run/"driver_start.json",dict(pid=99,argv=["driver.py"]+launchargv[3:],inputs_sha256=allpins,utc="2026-10-23T11:00:00Z"))
    dump(run/"completion.json",dict(success=True,inputs_unchanged=True,jobs=receipts,utc="2026-10-23T11:03:00Z"))
    for n in ("EXIT_CODE","EXTERNAL_EXIT_CODE"):(run/n).write_text("0\n")
    out=root/"audits/fixed_train64_center_counterfactual"
    selection=dict(selection_scope="train_only_no_internal",group_count=64,groups=["g"+str(i) for i in range(64)],
        fixed_seeds=[193,211],tuning=False,entries=[dict(variant="observed",group="g"+str(i)) for i in range(64)])
    dump(out/"selection.json",selection)
    pins={**sourcepins,str(raw):m.sha(raw),str(out/"selection.json"):m.sha(out/"selection.json")}
    r=receipts["fixed_train64_center_counterfactual"]
    dump(out/"protocol.json",dict(input_sha256={str(raw):m.sha(raw)},pid=r["pid"],argv=r["argv"][1:],training=False,search=False,groups=64))
    s=summary();s.update(input_sha256=pins,pid=r["pid"],argv=r["argv"][1:],selection_path=str(out/"selection.json"),
        selection_sha256=m.sha(out/"selection.json"),queries=128)
    dump(out/"summary.json",s)
    pp=root/"ADDITIONAL_PLAN.json";pp.write_bytes((Path(__file__).parent/"ADDITIONAL_PLAN.json").read_bytes())
    (root/"ADDITIONAL_PLAN_REVIEW.json").write_bytes((Path(__file__).parent/"ADDITIONAL_PLAN_REVIEW.json").read_bytes())
    return run,pp,qh

class Arithmetic(unittest.TestCase):
    def test_pass_exact_frozen_gate(self):self.assertTrue(m.arithmetic(plan(),summary())["admitted"])
    def test_one_seed_equal_rate_rejected(self):
        s=summary();c=s["summary"]["by_seed_variant"]["211"]["observed"]["pair_changes"][m.FIXED]["common"]
        c["counts"]["strict_reversals"]=6400;c["pair_weighted_strict_reversal_rate"]=.1
        self.assertFalse(m.arithmetic(plan(),s)["admitted"])
    def test_one_seed_rate_worse_rejected(self):
        s=summary();c=s["summary"]["by_seed_variant"]["193"]["observed"]["pair_changes"][m.FIXED]["common"]
        c["counts"]["strict_reversals"]=12800;c["pair_weighted_strict_reversal_rate"]=.2
        self.assertEqual(m.arithmetic(plan(),s)["status"],"REJECTED")
    def test_zero_two_seed_RR_delta_rejected(self):
        s=summary();s["summary"]["by_seed_variant"]["211"]["observed"]["mean_RR"]={"D_native":.5,"D_use_both_Scenters":.25}
        self.assertFalse(m.arithmetic(plan(),s)["admitted"])
    def test_negative_two_seed_RR_delta_rejected(self):
        s=summary()
        for seed in m.SEEDS:s["summary"]["by_seed_variant"][str(seed)]["observed"]["mean_RR"]={"D_native":.5,"D_use_both_Scenters":.25}
        self.assertFalse(m.arithmetic(plan(),s)["admitted"])
    def test_stress_and_overall_not_used(self):
        s=summary();s["summary"]["overall"]={"bad":"irrelevant"}
        for seed in m.SEEDS:s["summary"]["by_seed_variant"][str(seed)]["frequency_crop"]={"bad":"irrelevant"}
        self.assertTrue(m.arithmetic(plan(),s)["admitted"])
    def test_uses_pair_weighted_not_query_mean(self):
        s=summary()
        for seed in m.SEEDS:
            for name in (m.NATIVE,m.FIXED):s["summary"]["by_seed_variant"][str(seed)]["observed"]["pair_changes"][name]["common"]["query_mean_strict_reversal_rate"]=1
        self.assertTrue(m.arithmetic(plan(),s)["admitted"])
    def test_native_and_fixed_pair_universe_must_match(self):
        s=summary();c=s["summary"]["by_seed_variant"]["193"]["observed"]["pair_changes"][m.FIXED]["common"]
        c["counts"]={"pairs":128000,"strict_reversals":6400}
        with self.assertRaisesRegex(ValueError,"common_pair_universe_changed"):m.arithmetic(plan(),s)
    def test_rate_must_match_counts(self):
        s=summary();s["summary"]["by_seed_variant"]["193"]["observed"]["pair_changes"][m.FIXED]["common"]["pair_weighted_strict_reversal_rate"]=0
        with self.assertRaisesRegex(ValueError,"pair_weighted"):m.arithmetic(plan(),s)
    def test_observed_exact64(self):
        s=summary();s["summary"]["by_seed_variant"]["193"]["observed"]["queries"]=63
        with self.assertRaisesRegex(ValueError,"observed64"):m.arithmetic(plan(),s)
    def test_state_change_rejected(self):
        s=summary();s["final_model_state_sha256"]["193"]="c"*64
        with self.assertRaisesRegex(ValueError,"model_state"):m.arithmetic(plan(),s)
    def test_no_test_scope_required(self):
        s=summary();s["scope"]="test"
        with self.assertRaisesRegex(ValueError,"scope"):m.arithmetic(plan(),s)
    def test_plan_gate_not_mutable(self):
        p=plan();p["admission_gate"]["required_per_seed"]="mean across seeds"
        with self.assertRaisesRegex(ValueError,"frozen_gate"):m.arithmetic(p,summary())
    def test_nan_rejected(self):
        s=summary();s["summary"]["by_seed_variant"]["193"]["observed"]["mean_RR"]["D_native"]=float("nan")
        with self.assertRaisesRegex(ValueError,"natural_RR"):m.arithmetic(plan(),s)

class Envelope(unittest.TestCase):
    def test_complete_producer_shaped_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,p,q=fixture(Path(tmp).resolve());x=m.evaluate(r,p,q)
            self.assertEqual(x["status"],"PASS");self.assertTrue(x["inputs_unchanged"])
            self.assertTrue(x["driver_and_both_jobs_actual_double_exit0"])
    def test_legitimate_empty_requested_inputs_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,p,q=fixture(Path(tmp).resolve())
            queue=json.loads((Path(tmp).resolve()/"mechanism_queue.json").read_text())
            self.assertEqual(queue["jobs"][0]["inputs_sha256"],{})
            self.assertEqual(m.evaluate(r,p,q)["status"],"PASS")
    def test_missing_completion_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,p,q=fixture(Path(tmp).resolve());(r/"completion.json").unlink()
            x=m.evaluate(r,p,q);self.assertEqual(x["status"],"PENDING");self.assertFalse(x["admitted"])
    def test_bool_exit_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,p,q=fixture(Path(tmp).resolve());v=json.loads((r/"launch_receipt.json").read_text());v["internal_exit"]=False;dump(r/"launch_receipt.json",v)
            self.assertEqual(m.evaluate(r,p,q)["status"],"INVALID_EVIDENCE")
    def test_actual_input_hash_drift_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();r,p,q=fixture(root);(root/"features/train/features.pt").write_bytes(b"changed")
            self.assertEqual(m.evaluate(r,p,q)["status"],"INVALID_EVIDENCE")
    def test_non_gate_training_definition_byte_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,p,q=fixture(Path(tmp).resolve());v=json.loads(p.read_text());v["training"]["epochs"]=32;dump(p,v)
            x=m.evaluate(r,p,q);self.assertEqual(x["status"],"INVALID_EVIDENCE")
            self.assertEqual(x["reason"],"reviewed_plan_bytes_changed")
    def test_review_must_be_independent_PASS(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,p,q=fixture(Path(tmp).resolve());rp=p.parent/"ADDITIONAL_PLAN_REVIEW.json";v=json.loads(rp.read_text());v["independent"]=False;dump(rp,v)
            self.assertEqual(m.evaluate(r,p,q)["reason"],"plan_independent_review")
    def test_wrong_unique_queue_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,p,q=fixture(Path(tmp).resolve());self.assertEqual(m.evaluate(r,p,"0"*64)["status"],"INVALID_EVIDENCE")
if __name__=="__main__":unittest.main()

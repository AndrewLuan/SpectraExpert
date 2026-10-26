"""用途：独立核验已完成 train64 机制诊断，计算条件化追加实验的准入门槛。
只读 driver、summary、selection、protocol 和既有 query JSONL；逐项散列输入，不加载
模型或特征缓存。observed 每 seed64，翻转率采用共同候选对总数加权；两个 seed 各严格
下降，且两 seed 等权自然池 RR 改变量大于零才 PASS。缺失证据返回 PENDING。
"""
import argparse
import hashlib
import json
import math
from datetime import datetime,timezone
from pathlib import Path

SEEDS=(193,211)
NATIVE="S_native__to__D_native"
FIXED="S_native__to__D_use_both_Scenters"
PRODUCER_SHA="3069d3b40b11d7684c2dada1062f617388f4eceaed1b9afbdc94148bb2c32929"
SCOPE="fixed_train_subset_only_no_test_no_training_no_search"
PLAN_SHA="240690103d9bbd03e3d4bbbcf03342547d4f29f7da307b31c88ed5ccae734af0"
GATE={
"diagnostic":"unique already-reviewed actual train64 counterfactual",
"scope":"observed per seed only",
"flip_metric":"pair_weighted_strict_reversal_rate across common candidate pairs; epsilon1e-7; compare S_native__to__D_use_both_Scenters against S_native__to__D_native",
"required_per_seed":"fixed-reference strict flip rate strictly less than native-reference rate, for both193 and211",
"RR_metric":"mean_RR[D_use_both_Scenters] minus mean_RR[D_native] on natural pools",
"required_two_seed_RR":"equal mean of the two per-seed observed RR differences strictly >0",
"incomplete_or_failed":"do not launch training",
"meaning":"mechanism evidence admitting one hypothesis test; not empirical proof of fixed global center performance"}

def require(x,message):
    if not x:raise ValueError(message)
def close(a,b,message):
    require(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-12,message)
def utc(s):
    v=datetime.fromisoformat(s.replace("Z","+00:00"))
    require(v.tzinfo is not None,"timezone_required")
    return v.astimezone(timezone.utc)
def dual0(v):
    require(all(type(v.get(k)) is int and v[k]==0 for k in ("internal_exit","external_exit")),"actual_double_exit0_required")
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()

class Reader:
    def __init__(self):self.pins={}
    def pin(self,p,expected=None):
        p=str(Path(p).resolve())
        require(not any(x in ("test","sealed","verify") for x in Path(p).parts),"forbidden_test_path")
        digest=sha(p)
        if expected is not None:require(digest==expected,"input_sha_mismatch:"+p)
        if p in self.pins:require(self.pins[p]==digest,"input_drift:"+p)
        self.pins[p]=digest
        return digest
    def obj(self,p):
        self.pin(p);return json.loads(Path(p).read_text())
    def end(self):
        after={p:sha(p) for p in self.pins}
        require(after==self.pins,"input_changed_during_gate_audit")
        return after
    def pinmap(self,m,allow_empty=False):
        require(isinstance(m,dict) and (allow_empty or bool(m)),"empty_input_or_source_sha")
        for p,h in m.items():
            require(isinstance(h,str) and len(h)==64 and all(c in "0123456789abcdef" for c in h),"invalid_sha")
            self.pin(p,h)

def arithmetic(plan,summary):
    require(plan["schema"]=="fixed_train_center_conditional_preregistration_20261023" and plan["seeds"]==list(SEEDS),"plan_identity")
    require(plan["admission_gate"]==GATE,"frozen_gate_definition_changed")
    require(summary["status"]=="PASS" and summary["scope"]==SCOPE,"scope_or_summary_status")
    require(summary["seeds"]==list(SEEDS) and type(summary["groups"]) is int and summary["groups"]==64,"fixed64_two_seeds")
    require(summary["epsilon"]==1e-7,"epsilon")
    require(summary["inputs_unchanged"] is True and summary["source_files_unchanged"] is True,"source_or_inputs_changed")
    before,after=summary["initial_model_state_sha256"],summary["final_model_state_sha256"]
    require(set(before)==set(after)=={"193","211"} and before==after,"model_state_changed")
    require(all(isinstance(h,str) and len(h)==64 for h in before.values()),"state_sha")
    cells={}
    for seed in SEEDS:
        cell=summary["summary"]["by_seed_variant"][str(seed)]["observed"]
        require(type(cell["queries"]) is int and cell["queries"]==64,"observed64_per_seed")
        rates={}
        for name in (NATIVE,FIXED):
            x=cell["pair_changes"][name]["common"]; counts=x["counts"]
            require(x["queries"]==64,"observed_pair_query_count")
            pairs,reversals=counts["pairs"],counts["strict_reversals"]
            require(type(pairs) is int and pairs>0 and type(reversals) is int and 0<=reversals<=pairs,"pair_counts")
            rate=x["pair_weighted_strict_reversal_rate"]
            close(rate,reversals/pairs,"pair_weighted_rate_mismatch")
            rates[name]=rate
        require(cell["pair_changes"][NATIVE]["common"]["counts"]["pairs"]==
                cell["pair_changes"][FIXED]["common"]["counts"]["pairs"],"common_pair_universe_changed")
        rr=cell["mean_RR"]
        for name in ("D_native","D_use_both_Scenters"):
            require(type(rr[name]) in (int,float) and math.isfinite(rr[name]) and 0<rr[name]<=1,"natural_RR")
        cells[str(seed)]=dict(queries=64,native_pair_weighted_strict_reversal_rate=rates[NATIVE],
            fixed_reference_pair_weighted_strict_reversal_rate=rates[FIXED],
            strict_reversal_rate_decreased=rates[FIXED]<rates[NATIVE],
            native_mean_RR=rr["D_native"],fixed_reference_mean_RR=rr["D_use_both_Scenters"],
            delta_RR=rr["D_use_both_Scenters"]-rr["D_native"])
    delta=math.fsum(c["delta_RR"] for c in cells.values())/2
    passed=all(c["strict_reversal_rate_decreased"] for c in cells.values()) and delta>0
    return dict(status="PASS" if passed else "REJECTED",admitted=passed,per_seed=cells,
        equal_two_seed_delta_RR=delta,RR_gate_passed=delta>0,selection_scope="train_observed64_mechanism_only",
        promotion=False,training_launched=False)

def audit(run,plan_path,expected_queue_sha):
    read=Reader();run=Path(run).resolve()
    plan=read.obj(plan_path)
    require(read.pins[str(Path(plan_path).resolve())]==PLAN_SHA,"reviewed_plan_bytes_changed")
    plan_review=read.obj(Path(plan_path).parent/"ADDITIONAL_PLAN_REVIEW.json")
    require(plan_review["status"]=="PASS" and plan_review["independent"] is True and
            plan_review["scope"]=="conditional_additional_scientific_preregistration_only","plan_independent_review")
    plan_hashes=[h for p,h in plan_review["reviewed_sha256"].items() if Path(p).name=="ADDITIONAL_PLAN.json"]
    require(plan_hashes==[PLAN_SHA],"plan_review_binding")
    for n in ("EXIT_CODE","EXTERNAL_EXIT_CODE"):
        read.pin(run/n);require((run/n).read_text().strip()=="0","driver_exit")
    launch=read.obj(run/"launch_receipt.json");dual0(launch)
    require(len(launch["queue_sha256"])==1,"queue_count")
    qp,qh=next(iter(launch["queue_sha256"].items()))
    require(qh==expected_queue_sha,"wrong_unique_queue")
    read.pin(qp,qh);queue=read.obj(qp)
    read.pinmap(queue["source_inputs_sha256"]);read.pinmap(launch["source_sha256"])
    require(launch["bundle_sha256"]==queue["source_inputs_sha256"],"launch_bundle")
    start=read.obj(run/"driver_start.json");complete=read.obj(run/"completion.json")
    require(complete.get("success") is True and complete.get("inputs_unchanged") is True,"driver_not_complete")
    require(start["pid"]==launch["child_pid"] and start["argv"][1:]==launch["argv"][3:],"driver_identity")
    read.pinmap(start["inputs_sha256"])
    expected={**queue["source_inputs_sha256"],qp:qh}
    require(all(start["inputs_sha256"].get(p)==h for p,h in expected.items()),"driver_start_sha")
    require(utc(launch["started_utc"])<=utc(start["utc"])<=utc(complete["utc"])<=utc(launch["finished_utc"]),"driver_time")
    jobs={j["id"]:j for j in queue["jobs"]}
    require(len(jobs)==len(queue["jobs"])==2 and set(jobs)=={"reuse_completed_paired_validation","fixed_train64_center_counterfactual"},"unique_two_job_matrix")
    require(set(complete["jobs"])==set(jobs),"completion_job_set")
    receipts={}
    for name,j in jobs.items():
        r=read.obj(run/(name+".receipt.json"));dual0(r)
        require(r==complete["jobs"][name] and r["state"]=="complete","actual_job_receipt")
        require(r["argv"]==j["argv"] and r["cwd"]==j["cwd"] and r["output"]==j["out"],"job_invocation")
        require(r["requested_inputs_sha256"]==j["inputs_sha256"],"requested_inputs")
        read.pinmap(j["inputs_sha256"],allow_empty=True)
        require(all(r["queue_source_sha256"].get(p)==h for p,h in expected.items()),"job_source_binding")
        require(utc(start["utc"])<=utc(r["started_utc"])<=utc(r["finished_utc"])<=utc(complete["utc"]),"job_time")
        out=Path(j["out"]);read.pin(out/"EXIT_CODE");require((out/"EXIT_CODE").read_text().strip()=="0","job_internal_exit")
        receipts[name]=r
    receipt=receipts["fixed_train64_center_counterfactual"];out=Path(receipt["output"])
    summary=read.obj(out/"summary.json");selection=read.obj(out/"selection.json");protocol=read.obj(out/"protocol.json")
    source_path=receipt["argv"][1]
    require(queue["source_inputs_sha256"].get(source_path)==PRODUCER_SHA,"reviewed_producer_source")
    require(summary["input_sha256"].get(source_path)==PRODUCER_SHA,"summary_source_binding")
    read.pinmap(summary["input_sha256"])
    require(all(summary["input_sha256"].get(p)==h for p,h in protocol["input_sha256"].items()),"protocol_input_binding")
    require(summary["pid"]==protocol["pid"]==receipt["pid"] and summary["argv"]==protocol["argv"]==receipt["argv"][1:],"producer_identity")
    require(protocol["training"] is False and protocol["search"] is False and protocol["groups"]==64,"no_training_search")
    require(summary["selection_path"]==str(out/"selection.json") and summary["selection_sha256"]==read.pins[str((out/"selection.json").resolve())],"selection_sha_binding")
    require(selection["selection_scope"]=="train_only_no_internal" and selection["group_count"]==64 and len(selection["groups"])==len(set(selection["groups"]))==64 and selection["fixed_seeds"]==list(SEEDS) and selection["tuning"] is False,"selection_fixed64")
    observed=[e for e in selection["entries"] if e["variant"]=="observed"]
    require(len(observed)==64 and {e["group"] for e in observed}==set(selection["groups"]),"selection_observed64")
    require(summary["queries"]==len(selection["entries"])*2,"total_queries")
    result=arithmetic(plan,summary)
    before=dict(read.pins);after=read.end()
    result.update(schema="additional_admission_gate_20261023",utc=datetime.now(timezone.utc).isoformat(),
        unique_queue_sha256=expected_queue_sha,run=str(run),diagnostic_output=str(out),input_sha256_before=before,
        input_sha256_after=after,inputs_unchanged=True,driver_and_both_jobs_actual_double_exit0=True,
        summary_source_input_binding_verified=True,model_state_unchanged=True,no_test_scope=True)
    return result

def evaluate(run,plan,queue_sha):
    try:return audit(run,plan,queue_sha)
    except FileNotFoundError as e:
        return dict(status="PENDING",admitted=False,reason="required_completed_evidence_missing",missing=str(e.filename),training_launched=False)
    except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e:
        return dict(status="INVALID_EVIDENCE",admitted=False,reason=str(e),training_launched=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--plan",required=True);p.add_argument("--queue-sha",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();out=Path(a.out);require(not out.exists(),"exclusive_output_required")
    result=evaluate(a.run,a.plan,a.queue_sha)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("input_sha256_before","input_sha256_after")}))
if __name__=="__main__":main()

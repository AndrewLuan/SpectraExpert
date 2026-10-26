"""用途：核验扩展封存的24个真实任务，汇总main8+fixedcenter2+frozen单/双池消融。
只读取已经生成的封存产物；固定1371身份、双seed、24任务双EXIT、具体freeze review、
joint预选和新producer/中心来源闭环。保留matching/reranking原始计时，不调用choose，
不按test重选、不晋升。所有输入始末SHA核对，独占写summary/TSV/EXIT；旧工具不变。
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

VALIDATION_SHA="9eb90a2ba38860bd15661bfbf5c5eefbbde708f68c190eb4c12c0bda471044aa"
def need(x,message):
    if not x:raise ValueError(message)
def load_validation():
    path=Path(__file__).with_name("summarize_pool_validation.py")
    need(hashlib.sha256(path.read_bytes()).hexdigest()==VALIDATION_SHA,"validation_helper_changed")
    spec=importlib.util.spec_from_file_location("_extended_sealed_validation",path)
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v
validation=load_validation()
SCHEMA="pool_extended_sealed_ablation_results_20261023"
ADDED=("fixedcenter_mixed_listwise_single","fixedcenter_mixed_listwise_dual")
CONFIGS=tuple(validation.CONFIGS)+ADDED+("frozen_single","frozen_dual")
VARIANTS={"observed":1000,"frequency_crop":200,"peak_dropout":171}
EVIDENCE_KEYS={"main_validation_summary","training_audit","additional_training_audit","preregistration",
    "additional_preregistration","eligibility","builder","legacy_builder","joint_summarizer",
    "validation_summarizer","training_auditor","legacy_training_auditor","test_summarizer","validation_review"}
TOOL_KEYS={"builder","legacy_builder","joint_summarizer","validation_summarizer","training_auditor",
           "legacy_training_auditor","test_summarizer"}
SOURCE_NAMES=("h_pool_ranker.py","h_research_ranker.py","h_research_observation.py","h_research_contract.py",
              "h_global_ranker.py","h_global_features.py","h_research_driver.py","data.py")

def canonical(p):return str(Path(p).resolve())
def double0(record):
    return all(type(record.get(k)) is int and record[k]==0 for k in ("external_exit","internal_exit"))
def fixed_identity(x):
    return isinstance(x,dict) and set(x)=={"query_sha256","identity_content_sha256","variant_mask_content_sha256"} and all(
        isinstance(h,str) and len(h)==64 and all(c in "0123456789abcdef" for c in h) for h in x.values())
def reviewed(frozen,review,path,digest,require_evaluation=True):
    path=canonical(path)
    need(frozen["reviewed_inputs_sha256"].get(path)==review["reviewed_sha256"].get(path)==digest,"specific_review_binding:"+path)
    if require_evaluation:need(frozen["evaluation_inputs_sha256"].get(path)==digest,"frozen_evaluation_binding:"+path)

def frozen_contract(frozen_path,pins,verified):
    fp=Path(frozen_path).resolve();frozen=validation.read_json(fp,pins)
    need(frozen["schema"]=="nmr_pool_ranker_20261023" and frozen["selection_scope"]=="train_val_only"
         and frozen["promotion"] is False and frozen["verify_eligibility"]=="unavailable","invalid_extended_frozen")
    need(frozen["candidate_count"]==114982 and frozen["candidate_order_sha256"]==validation.CANDIDATE_SHA
         and frozen["test_variants"]==VARIANTS and fixed_identity(frozen["fixed_test_identity"]),"fixed_test_contract")
    need(set(frozen["configurations"])==set(CONFIGS) and len(frozen["configurations"])==12,"complete12_configuration_matrix")
    need(frozen["preselected"] is None or frozen["preselected"] in set(validation.CONFIGS)|set(ADDED),"unregistered_preselected")
    need(set(frozen["sealed_evidence_paths"])==EVIDENCE_KEYS,"mandatory14_evidence")
    approval=frozen["freeze_review"];rp=Path(approval["path"])
    need(canonical(rp)==str(rp),"noncanonical_freeze_review")
    validation.require_sha({str(rp):approval["sha256"]},verified)
    review=validation.read_json(rp,pins)
    need(review.get("status")=="PASS" and review.get("independent") is True and review.get("scope")=="freeze_plan_and_implementation" and bool(review.get("reviewer")),"specific_freeze_review_required")
    need(bool(frozen["reviewed_inputs_sha256"]),"empty_freeze_review_inputs")
    need(all(review["reviewed_sha256"].get(p)==h for p,h in frozen["reviewed_inputs_sha256"].items()),"full_freeze_review_binding")
    validation.require_sha(frozen["reviewed_inputs_sha256"],verified)
    validation.require_sha(frozen["evaluation_inputs_sha256"],verified)
    evidence=frozen["sealed_evidence_paths"]
    need(canonical(evidence["test_summarizer"])==canonical(__file__),"wrong_extended_summary_tool")
    for path in [__file__,validation.__file__,frozen["test_plan_path"],frozen["validation_summary_path"]]+list(evidence.values()):
        digest=validation.sha(path);reviewed(frozen,review,path,digest)
    need(frozen["tool_sha256"].get(canonical(__file__))==validation.sha(__file__),"summary_tool_not_frozen")
    plan=validation.read_json(frozen["test_plan_path"],pins)
    need(plan["schema"]=="pool_extended_sealed_test_plan_20261023" and plan["purpose"]=="sealed_test_ablation_only"
         and plan["selection_scope"]=="train_val_only" and plan["promotion"] is False
         and plan["preselected"]==frozen["preselected"] and plan["fixed_test_identity"]==frozen["fixed_test_identity"],"reviewed_extended_plan")
    need(plan["tool_sha256"]==frozen["tool_sha256"],"frozen_plan_tools")
    for key in TOOL_KEYS:need(plan["tool_sha256"].get(canonical(evidence[key]))==validation.sha(evidence[key]),"reviewed_tool_identity")
    need(set(Path(p).name for p in plan["source_sha256"])==set(SOURCE_NAMES),"scientific_source_matrix")
    for p,h in {**plan["tool_sha256"],**plan["source_sha256"]}.items():reviewed(frozen,review,p,h)
    joint=validation.read_json(frozen["validation_summary_path"],pins)
    need(joint["schema"]=="pool_joint_validation_20261023" and joint["selection"]["preselected"]==frozen["preselected"],"frozen_joint_selection_changed")
    need(validation.sha(frozen["validation_summary_path"])==frozen["validation_summary_sha256"],"joint_summary_SHA")
    need(validation.read_json(evidence["main_validation_summary"],pins)["schema"]=="pool_validation_selection_20261023","main_validation_schema")
    need(validation.read_json(evidence["training_audit"],pins)["status"]=="PASS","original_training_audit")
    extra=validation.read_json(evidence["additional_training_audit"],pins)
    need(extra["status"]=="PASS" and extra["scope"]=="fixedcenter_training_no_test","fixed_training_audit")
    named=frozen["ranker_configurations"]
    need(set(named)==set(CONFIGS) and all(set(z)=={"193","211"} for z in named.values()),"named24_matrix")
    flattened=[named[c][str(s)] for c in CONFIGS for s in validation.SEEDS]
    hashes=[validation.content_sha(x) for x in flattened]
    need(len(frozen["allowed_ranker_configs"])==24 and len(set(hashes))==24 and
         sorted(hashes)==sorted(validation.content_sha(x) for x in frozen["allowed_ranker_configs"]),"global_named_allowlist")
    fixed=frozen["fixed_center_configurations"]
    need(set(fixed)==set(ADDED) and all(set(z)=={"193","211"} for z in fixed.values()),"fixed_center4_matrix")
    entries={}
    for e in plan["evaluations"]:
        role=(e["config"],e["seed"]);need(type(e["seed"]) is int and role not in entries,"duplicate_plan_role")
        need(role in {(c,s) for c in CONFIGS for s in validation.SEEDS},"unregistered_plan_role")
        entries[role]=e
    need(len(entries)==24,"plan24_required")
    return frozen,review,joint,entries

def verify_run(frozen_path,queue_path,run,pins,verified):
    fp=Path(frozen_path).resolve();qp=Path(queue_path).resolve();run=Path(run).resolve()
    frozen,review,joint,entries=frozen_contract(fp,pins,verified)
    validation.exit0(fp.parent/"EXIT_CODE",pins)
    build=validation.read_json(fp.parent/"build_receipt.json",pins)
    need(build["schema"]=="pool_extended_freeze_build_20261023" and build["status"]=="BUILT_NOT_EXECUTED" and build["jobs"]==24 and build["promotion"] is False,"extended_builder_receipt")
    validation.require_sha(build["output_sha256"],verified)
    need(build["output_sha256"].get(str(fp))==validation.sha(fp) and build["output_sha256"].get(str(qp))==validation.sha(qp),"builder_output_SHA")
    queue=validation.read_json(qp,pins);need(queue["schema"]=="nmr_research_jobs_20261022","sealed_queue_schema")
    validation.require_sha(queue["source_inputs_sha256"],verified)
    validation.exit0(run/"EXIT_CODE",pins);validation.exit0(run/"EXTERNAL_EXIT_CODE",pins)
    launch=validation.read_json(run/"launch_receipt.json",pins);need(double0(launch),"actual_outer_double_exit0")
    need(launch["queue_sha256"]=={str(qp):validation.sha(qp)} and launch["bundle_sha256"]==queue["source_inputs_sha256"],"launch_bundle_queue_binding")
    need(bool(launch["source_sha256"]) and all(queue["source_inputs_sha256"].get(p)==h for p,h in launch["source_sha256"].items()),"launch_source_binding")
    validation.require_sha(launch["source_sha256"],verified)
    start=validation.read_json(run/"driver_start.json",pins);completion=validation.read_json(run/"completion.json",pins)
    expected={**queue["source_inputs_sha256"],str(qp):validation.sha(qp)}
    need(start["inputs_sha256"]==expected,"driver_start_source_matrix")
    need(type(start["pid"]) is int and start["pid"]>0 and start["pid"]==launch["child_pid"] and start["argv"][1:]==launch["argv"][3:],"driver_identity")
    validation.require_sha(start["inputs_sha256"],verified)
    need(completion["success"] is True and completion["inputs_unchanged"] is True,"driver_incomplete")
    times=[validation.utc(x) for x in [launch["started_utc"],start["utc"],completion["utc"],launch["finished_utc"]]]
    need(times==sorted(times) and times[-1]<=validation.datetime.now(validation.timezone.utc),"driver_time_order")
    jobs=queue["jobs"];need(len(jobs)==len({j["id"] for j in jobs})==len({j["out"] for j in jobs})==24,"actual24_job_matrix")
    need(set(completion["jobs"])=={j["id"] for j in jobs},"completion_matrix")
    roles=set()
    for j in jobs:
        need(j["kind"]=="sealed" and not j.get("deps"),"sealed_only")
        r=validation.read_json(run/(j["id"]+".receipt.json"),pins)
        need(r==completion["jobs"][j["id"]] and r["state"]=="complete" and double0(r),"job_actual_double_exit0")
        need(r["argv"]==j["argv"] and r["cwd"]==j["cwd"] and r["output"]==j["out"],"job_actual_invocation")
        need(r["queue_source_sha256"]==expected and r["requested_inputs_sha256"]==j["inputs_sha256"],"job_actual_source_inputs")
        need(times[1]<=validation.utc(r["started_utc"])<=validation.utc(r["finished_utc"])<=times[2],"job_time_order")
        validation.require_sha(j["inputs_sha256"],verified);validation.exit0(Path(j["out"])/"EXIT_CODE",pins)
        f=Path(j["out"]);role=(f.parent.parent.name,int(f.parent.name.removeprefix("s")))
        need(f.name=="test" and role in entries and role not in roles and j["id"]==f"sealed_{role[0]}_s{role[1]}","job_named_role")
        roles.add(role)
    need(roles==set(entries),"complete24_named_roles")
    return frozen,review,joint,entries,jobs

def read_test(job,frozen,review,joint,entry,pins,verified,frozen_path):
    out=Path(job["out"]).resolve();config=out.parent.parent.name;seed=int(out.parent.name.removeprefix("s"));pool=config.rsplit("_",1)[1]
    protocol=validation.read_json(out/"protocol.json",pins);args=protocol["args"];argv=job["argv"]
    added=config in ADDED
    names=("source","config","cache","no_aug_checkpoint","aug_checkpoint","seed","out","frozen_selection") if added else ("stage","cache","no_aug_checkpoint","aug_checkpoint","seed","out","frozen_selection")
    if added:
        need(len(argv)>=2 and Path(argv[1]).name=="evaluate_pool_fixed_center_sealed.py","new_producer_command")
        tail=argv[2:]
    else:
        need(argv[1:3]==["-m","nmr_transfer.h_pool_ranker"],"original_producer_command");tail=argv[3:]
    need(len(tail)==2*len(names),"exact_sealed_command")
    options={}
    for k,x in zip(tail[::2],tail[1::2]):
        need(k not in options,"duplicate_sealed_option");options[k]=x
    need(set(options)=={"--"+n.replace("_","-") for n in names},"sealed_command_options")
    need(all(str(args[n])==options["--"+n.replace("_","-")] for n in names),"protocol_actual_argv")
    need(type(args["seed"]) is int and args["seed"]==seed and canonical(args["out"])==str(out)
         and args["frozen_selection"]==canonical(frozen_path),"protocol_named_role")
    if not added:need(args["stage"]=="evaluate","evaluation_stage")
    need(args["cache"]==entry["cache"],"registered_test_cache_path")
    inp=protocol["inputs_sha256"];validation.require_sha(inp,verified)
    need(all(inp.get(p)==h for p,h in job["inputs_sha256"].items()),"protocol_requested_inputs")
    allowed=dict(no_aug_checkpoint=args["no_aug_checkpoint"],aug_checkpoint=args["aug_checkpoint"],seed=seed,cohort="test",recall=pool,cache_sha256=inp.get(args["cache"]))
    need(allowed==frozen["ranker_configurations"][config][str(seed)] and allowed in frozen["allowed_ranker_configs"],"named_model_allowlist")
    need(entry["cache_sha256"]==allowed["cache_sha256"],"test_plan_cache_SHA")
    summary=validation.read_json(out/"summary.json",pins);cache=validation.read_json(Path(args["cache"]).parent/"summary.json",pins)
    need(summary["input_sha256"]==inp,"summary_protocol_inputs")
    need(summary["checkpoints"]=={args[k]:inp[args[k]] for k in ("no_aug_checkpoint","aug_checkpoint")},"summary_model_SHA")
    need(summary["cohort"]==cache["cohort"]=="test" and summary["recall"]==cache["recall"]==pool,"test_cohort_pool")
    need(type(cache["records"]) is int and cache["records"]==1371 and type(cache["base_queries"]) is int and cache["base_queries"]==1000
         and cache["variants"]==VARIANTS and all(type(n) is int for n in cache["variants"].values())
         and cache["candidate_count"]==114982 and cache["pool_size"]==512 and cache["appended_training_positives"]==0,"fixed1371_test_contract")
    need(cache["feature_cache_sha256"]==inp[args["cache"]],"test_cache_producer_SHA")
    need(summary["candidate_order_sha256"]==cache["candidate_order_sha256"]==validation.CANDIDATE_SHA
         and summary["query_sha256"]==cache["query_sha256"],"test_candidate_query_identity")
    need(all(cache[k]==h for k,h in frozen["fixed_test_identity"].items()),"frozen_test_identity")
    if added:
        need(args["config"]==summary["config"]==config and type(summary["seed"]) is int and summary["seed"]==seed,"fixed_summary_named_role")
        need(args.get("device","cuda")=="cuda" and args["source"]==job["cwd"],"fixed_source_device")
        selected=frozen["fixed_center_configurations"][config][str(seed)]
        need(all(selected[k]==x for k,x in allowed.items()) and selected["cache"]==args["cache"],"fixed_allowlist")
        need(summary["center_estimate_sha256"]==selected["center_estimate_sha256"],"fixed_summary_centers")
        paths=[argv[1]]+[str(Path(args["source"])/"nmr_transfer"/n) for n in SOURCE_NAMES if n!="h_research_driver.py"]
        fixed_sources=[p for p in frozen["tool_sha256"] if Path(p).name=="h_pool_fixed_center_ranker.py"]
        need(len(fixed_sources)==1,"fixed_module_source");paths+=fixed_sources
        paths += [frozen["test_plan_path"],frozen["validation_summary_path"]]+list(frozen["sealed_evidence_paths"].values())
        for path in paths:
            path=canonical(path);digest=validation.sha(path)
            need(inp.get(path)==digest,"fixed_producer_source_or_evidence_closure:"+path)
            reviewed(frozen,review,path,digest)
        need(inp.get(canonical(frozen_path))==validation.sha(frozen_path) and inp.get(frozen["freeze_review"]["path"])==frozen["freeze_review"]["sha256"],"fixed_protocol_freeze_review")
        need(len(selected["center_artifacts"])==2 and selected["center_estimate_sha256"]==[a["centers_sha256"] for a in selected["center_artifacts"]],"fixed_center_roles")
        for aug,key,artifact in zip((False,True),("no_aug_checkpoint","aug_checkpoint"),selected["center_artifacts"]):
            parent=Path(args[key]).parent
            need(artifact["centers"]==str(parent/"centers.json") and artifact["row_audit"]==str(parent/"center_rows.jsonl"),"fixed_center_paths")
            for path,h in [(artifact["centers"],artifact["centers_sha256"]),(artifact["row_audit"],artifact["row_audit_sha256"])]:
                need(inp.get(path)==h,"fixed_center_protocol_SHA");reviewed(frozen,review,path,h)
            estimate=validation.read_json(artifact["centers"],pins)
            need(estimate["seed"]==seed and estimate["augment"] is aug and estimate["selection_scope"]=="fit_train_only"
                 and len(estimate["center33"])==33 and len(estimate["O_center3"])==3 and estimate["row_audit_sha256"]==artifact["row_audit_sha256"],"fixed_center_estimation_identity")
            need(all(type(x) in (int,float) and math.isfinite(x) for x in estimate["center33"]+estimate["O_center3"]),"finite_fixed_centers")
        weights={args["no_aug_checkpoint"]:selected["no_aug_sha256"],args["aug_checkpoint"]:selected["aug_sha256"]}
        need(weights==summary["checkpoints"] and all(joint["metadata"][config][str(seed)][c]["checkpoints"]==weights for c in validation.COHORTS),"fixed_joint_model_binding")
    rows=[json.loads(line) for line in validation.read_pinned(out/"predictions.jsonl",pins).splitlines() if line.strip()]
    validation.validate_rows_identity(rows,cache)
    need(len(rows)==1371 and len({validation.identity(r) for r in rows})==1371,"complete1371_rows")
    for row in rows:
        need(type(row["rank"]) is int and 1<=row["rank"]<=114982 and type(row["target_in_coarse_pool"]) is bool
             and row["evaluation_pool_size"]==512 and (row["rank"]<=512)==row["target_in_coarse_pool"],"natural_rank_membership")
        ids,costs=row["natural_candidate_indices"],row["natural_candidate_costs"]
        need(len(ids)==len(set(ids))==len(costs)==512 and all(type(i) is int and 0<=i<114982 for i in ids)
             and all(type(x) in (int,float) and math.isfinite(x) for x in costs),"natural512_costs")
    normal=[r for r in rows if r["variant"]=="observed"]
    computed=dict(overall=validation.metrics(normal),by_source={s:validation.metrics([r for r in normal if r["source"]==s]) for s in sorted(validation.SOURCES)},
        stress={w:validation.metrics([r for r in rows if r["variant"]==w]) for w in list(VARIANTS)[1:]})
    need(validation.close_metrics(summary["metrics"]["overall"],computed["overall"]),"summary_metrics")
    for k in ("by_source","stress"):
        need(summary["metrics"][k].keys()==computed[k].keys() and all(validation.close_metrics(summary["metrics"][k][n],x) for n,x in computed[k].items()),"summary_strata_metrics")
    matching,reranking=cache["matching_seconds_mean"],summary["reranking_seconds_mean"]
    need(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in (matching,reranking)),"raw_cost_records")
    raw_cost=dict(matching_seconds_mean=matching,reranking_seconds_mean=reranking,reranking_seconds_p95=summary.get("reranking_seconds_p95"),
        evaluation_elapsed_seconds=summary.get("elapsed_seconds"),cost_seconds_sum=matching+reranking,
        summary_path=str(out/"summary.json"),cache_summary_path=str(Path(args["cache"]).parent/"summary.json"))
    return config,seed,rows,raw_cost

def aggregate(data):
    table=[]
    for config,seeds in data.items():
        need(set(seeds)==set(validation.SEEDS),"complete_two_seeds")
        for variant in VARIANTS:
            for source in ["all"]+sorted(validation.SOURCES):
                pair=[]
                for seed in validation.SEEDS:
                    part=[r for r in seeds[seed] if r["variant"]==variant and (source=="all" or r["source"]==source)]
                    result=validation.metrics(part);pair.append(result);table.append(dict(config=config,seed=str(seed),variant=variant,source=source,**result))
                need(pair[0]["queries"]==pair[1]["queries"],"seed_stratum_query_count")
                mean={k:sum(p[k] for p in pair)/2 if all(p[k] is not None for p in pair) else None for k in pair[0]}
                mean["queries"]=pair[0]["queries"];table.append(dict(config=config,seed="two_seed_mean",variant=variant,source=source,**mean))
    return table

def run(args,out):
    pins={};verified={};validation.read_pinned(__file__,pins);validation.read_pinned(validation.__file__,pins)
    frozen,review,joint,entries,jobs=verify_run(args.frozen,args.queue,args.run,pins,verified)
    data={};raw_costs={};signature=None
    for job in jobs:
        folder=Path(job["out"]);role=(folder.parent.parent.name,int(folder.parent.name.removeprefix("s")))
        config,seed,rows,cost=read_test(job,frozen,review,joint,entries[role],pins,verified,args.frozen)
        current={validation.identity(r):r["mask_content_sha256"] for r in rows}
        if signature is None:signature=current
        need(current==signature,"cross24_query_mask_identity")
        data.setdefault(config,{})[seed]=rows;raw_costs.setdefault(config,{})[seed]=cost
    need(set(data)==set(CONFIGS),"complete24_only")
    table=aggregate(data);need(len(table)==648,"full_stratum_matrix")
    with (out/"metrics.tsv").open("x") as f:
        fields=["config","seed","variant","source","queries","top1","top5","top10","MRR","recall512","in_pool_MRR"]
        writer=csv.DictWriter(f,fieldnames=fields,delimiter="\t");writer.writeheader();writer.writerows(table)
    validation.require_sha(pins,{});validation.require_sha(verified,{})
    result=dict(schema=SCHEMA,scope="sealed_test_ablation_only",preselected=frozen["preselected"],frozen_joint_preselected=joint["selection"]["preselected"],
        selection_performed=False,promotion=False,independent_acceptance_status="unavailable",jobs=24,
        configurations=list(CONFIGS),fixed_test_identity=frozen["fixed_test_identity"],test_variants=VARIANTS,
        input_sha256=pins,scientific_inputs_sha256=verified,source_sha256={canonical(__file__):validation.sha(__file__),canonical(validation.__file__):VALIDATION_SHA},
        metric_rows=len(table),cost_seconds_by_seed={c:{s:x["cost_seconds_sum"] for s,x in seeds.items()} for c,seeds in raw_costs.items()},
        raw_cost_records=raw_costs,frozen_sha256=validation.sha(args.frozen),external_api_calls=0)
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result

def main():
    p=argparse.ArgumentParser()
    for n in ("frozen","queue","run","out"):p.add_argument("--"+n,required=True)
    a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        value=run(a,out);(out/"EXIT_CODE").write_text("0\n");print(json.dumps(dict(preselected=value["preselected"],jobs=value["jobs"],metric_rows=value["metric_rows"],promotion=False)))
    except BaseException as e:
        (out/"failure.json").write_text(json.dumps({"error":repr(e)})+"\n");(out/"EXIT_CODE").write_text("1\n");raise
if __name__=="__main__":main()

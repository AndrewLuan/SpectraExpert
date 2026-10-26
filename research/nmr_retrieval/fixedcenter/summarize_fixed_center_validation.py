"""用途：只汇总唯一预注册 fixedcenter_mixed_listwise 的完整4fit/8eval追加家族。
读取真实driver双EXIT与已独审训练audit PASS，再复用固定SHA旧验证器的read_eval与
aggregate。read_eval使用隔离globals注入新argv binder，旧模块全局与原8配置不变。
直接frozen_single、320+1024查询权重、双seed、五source与全部guard/成本并列保持原值。
输出在原1188 TSV后追加216行；保留原main8/旧incumbent，独占写新产物，始末核SHA。
不加载模型/特征对象、不训练、不读test、不freeze、不晋升。
"""
import argparse
import collections
import copy
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import time
import types

HELPER_SHA="9eb90a2ba38860bd15661bfbf5c5eefbbde708f68c190eb4c12c0bda471044aa"
MAIN_SHA="143df36ba19e96368b25578f987b2d685cc5d8896875b3e4d210935d5bcd8c27"
MAIN_TSV_SHA="04d3bb1f5cd1718c8630bc26ee0a7e6b811bb2f66542bc97b657bb29c54020e4"
PLAN_SHA="240690103d9bbd03e3d4bbbcf03342547d4f29f7da307b31c88ed5ccae734af0"
QUEUE_SHA="d1429dd48c70f0e85b840e605c4d0d15ef2c2bbfe21ac0b1687b7fc1c0522951"
MODULE_SHA="3b8a6510198e40a63da542705987960f5a75feb023445510dc06ffb80ce19d20"
GATE_SHA="6aa896462033df0f90b7d3537fea67fcfa855c5194511a2048a2d88abf95ad2f"
CONFIGS=("fixedcenter_mixed_listwise_single","fixedcenter_mixed_listwise_dual")
FIELDS=("config","seed","cohort","variant","source","queries","top1","top5","top10","MRR","recall512","in_pool_MRR")

def require(x,message):
    if not x:raise ValueError(message)

def load_helper():
    path=Path(__file__).with_name("summarize_pool_validation.py")
    import hashlib
    require(hashlib.sha256(path.read_bytes()).hexdigest()==HELPER_SHA,"old_helper_source_changed")
    spec=importlib.util.spec_from_file_location("_fixed_center_validation_old_helper",path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

v=load_helper()

def double0(value):
    require(all(type(value.get(k)) is int and value[k]==0 for k in ("internal_exit","external_exit")),"real_integer_double_exit0_required")

def expected_args(root,config,seed,cohort):
    root=Path(root);pool=config.rsplit("_",1)[1]
    old=root.parent/"nmr_global_retrieval_20261022"
    model=root/f"additional/rankers/fixedcenter_mixed_listwise/s{seed}"
    return dict(source=str(root.parent.parent),stage="evaluate",seed=seed,
        out=str(root/f"additional/fixedcenter_mixed_listwise/s{seed}"/pool/cohort),
        cache=str(old/f"features/inherited_s{seed}"/pool/cohort/"features.pt"),
        no_aug_checkpoint=str(model/"no_aug/best.pt"),aug_checkpoint=str(model/"aug/best.pt"))

def bind_fixed_evaluation(args,job,expected):
    argv=job["argv"]
    require(argv[1]==str(Path(expected["out"]).parents[4]/"h_pool_fixed_center_ranker.py"),"fixed_module_command")
    tail=argv[2:];require(len(tail)%2==0,"fixed_command_pairs")
    options={}
    for key,value in zip(tail[::2],tail[1::2]):
        require(key.startswith("--") and key not in options,"duplicate_fixed_arg")
        options[key]=value
    required={"--"+k.replace("_","-") for k in expected}
    require(set(options) in (required,required|{"--device"}),"unregistered_fixed_arg")
    if "--device" in options:require(options["--device"] in ("cuda","cpu"),"device")
    require(args["device"]==options.get("--device","cuda"),"device_protocol_mismatch")
    for key,value in expected.items():
        require(str(args[key])==str(value)==options["--"+key.replace("_","-")],"fixed_family_or_seed_or_cache:"+key)
    for key,value in dict(training_pool="mixed",loss="listwise",epochs=64,samples=8192,batch_size=32,anchor=.1,lr=.001,augment=False).items():
        require(args[key]==value,"fixed_eval_budget:"+key)

def fixed_read_eval(*args,**kwargs):
    # 只替换已审read_eval的argv绑定器；旧模块对象、CONFIGS和read_eval全局均保持原值。
    namespace={**v.read_eval.__globals__,"bind_evaluation":bind_fixed_evaluation}
    reader=types.FunctionType(v.read_eval.__code__,namespace,v.read_eval.__name__,v.read_eval.__defaults__)
    return reader(*args,**kwargs)

def verify_driver(run,queue_path,queue,pins,verified):
    run=Path(run)
    v.exit0(run/"EXIT_CODE",pins);v.exit0(run/"EXTERNAL_EXIT_CODE",pins)
    launch=v.read_json(run/"launch_receipt.json",pins);double0(launch)
    require(launch["queue_sha256"]=={str(queue_path):QUEUE_SHA},"unique_actual_queue")
    require(launch["bundle_sha256"]==queue["source_inputs_sha256"],"launch_bundle")
    require(bool(launch["source_sha256"]),"missing_launcher_source")
    v.require_sha(launch["source_sha256"],verified)
    require(all(queue["source_inputs_sha256"].get(p)==h for p,h in launch["source_sha256"].items()),"launcher_source_binding")
    driver=v.read_json(run/"driver_start.json",pins);completion=v.read_json(run/"completion.json",pins)
    expected={**queue["source_inputs_sha256"],str(queue_path):QUEUE_SHA}
    require(driver["inputs_sha256"]==expected,"driver_start_source_binding")
    v.require_sha(expected,verified)
    require(driver["pid"]==launch["child_pid"] and driver["argv"][1:]==launch["argv"][3:],"driver_identity")
    require(completion.get("success") is True and completion.get("inputs_unchanged") is True,"incomplete_driver")
    begin,end=v.utc(driver["utc"]),v.utc(completion["utc"])
    require(v.utc(launch["started_utc"])<=begin<=end<=v.utc(launch["finished_utc"]),"driver_time_order")
    jobs=queue["jobs"]
    require(len(jobs)==len({j["id"] for j in jobs})==12 and collections.Counter(j["kind"] for j in jobs)=={"training":4,"validation":8},"complete4fit8eval_required")
    require(set(completion["jobs"])=={j["id"] for j in jobs},"completion_job_set")
    require(len({j["out"] for j in jobs})==12,"duplicate_output")
    result={}
    for job in jobs:
        receipt=v.read_json(run/(job["id"]+".receipt.json"),pins);double0(receipt)
        require(receipt==completion["jobs"][job["id"]] and receipt["state"]=="complete","receipt_complete_binding")
        require(receipt["argv"]==job["argv"] and receipt["output"]==job["out"] and receipt["cwd"]==job["cwd"],"job_invocation_binding")
        require(receipt["queue_source_sha256"]==expected and receipt["requested_inputs_sha256"]==job["inputs_sha256"],"job_source_input_binding")
        v.require_sha(job["inputs_sha256"],verified)
        start,finish=v.utc(receipt["started_utc"]),v.utc(receipt["finished_utc"])
        require(begin<=start<=finish<=end,"job_time_order")
        for dep in job.get("deps",[]):require(v.utc(completion["jobs"][dep]["finished_utc"])<=start,"job_dependency_time")
        v.exit0(Path(job["out"])/"EXIT_CODE",pins)
        result[str(Path(job["out"]).resolve())]=job
    return result

def read_audit(root,folder,review_path,queue,queue_path,jobs,pins,verified):
    folder=Path(folder);v.exit0(folder/"EXIT_CODE",pins)
    audit=v.read_json(folder/"summary.json",pins)
    require(audit["status"]=="PASS" and audit["scope"]=="fixedcenter_training_no_test" and audit["training_jobs"]==4 and audit["validation_jobs"]==8 and audit["inputs_unchanged"] is True,"training_audit_required")
    checks={"same_initial_tensors","frozen_observation","frozen_centers","matched_exposure","finite","full_budget","real_driver_double_exit","complete_validation_metadata"}
    require(checks<=set(audit["checks"]) and all(audit["checks"][c] is True for c in checks),"training_audit_checks")
    require(audit["queue_sha256"]==QUEUE_SHA and audit["input_sha256"].get(str(queue_path))==QUEUE_SHA,"audit_unique_queue_binding")
    review=v.read_json(review_path,pins)
    require(review["status"]=="PASS" and review["independent"] is True and review["scope"]=="local_source_and_synthetic_training_audit_review_only" and bool(review["reviewer"]),"audit_implementation_review")
    sources=[(p,h) for p,h in audit["input_sha256"].items() if Path(p).name=="audit_pool_fixed_center_training.py"]
    hashes=[h for p,h in review["reviewed_sha256"].items() if Path(p).name=="audit_pool_fixed_center_training.py"]
    require(len(sources)==len(hashes)==1 and sources[0][1]==hashes[0],"audit_reviewed_source_binding")
    v.require_sha(audit["input_sha256"],verified)
    models={}
    for m in audit["models"]:
        seed,aug=m["role"];require(type(seed) is int and type(aug) is bool,"model_role_type")
        role=(seed,aug);require(role not in models and seed in v.SEEDS,"duplicate_or_invalid_model")
        expected=Path(root)/f"additional/rankers/fixedcenter_mixed_listwise/s{seed}"/("aug" if aug else "no_aug")
        require(m["output"]==str(expected) and str(expected) in jobs and jobs[str(expected)]["kind"]=="training","audit_model_output")
        require(m["draws"]==524288 and m["optimizer_steps"]==16384 and m["constant_fp32"]==36,"audit_full_budget_constants")
        for name,digest in [("centers.json",m["center_estimate_sha256"]),("center_rows.jsonl",m["center_rows_sha256"])]:
            require(audit["input_sha256"].get(str(expected/name))==digest,"audit_center_binding")
        require(str(expected/"best.pt") in audit["input_sha256"],"audit_model_best_binding")
        models[role]=m
    require(set(models)=={(s,a) for s in v.SEEDS for a in (False,True)},"incomplete_audit_models")
    evaluations={}
    for e in audit["evaluations"]:
        role=tuple(e["role"]);require(role not in evaluations,"duplicate_audit_eval")
        seed,pool,cohort=role
        require(seed in v.SEEDS and pool in ("single","dual") and cohort in v.COHORTS,"audit_eval_role")
        expected=expected_args(root,"fixedcenter_mixed_listwise_"+pool,seed,cohort)
        require(e["output"]==expected["out"] and e["output"] in jobs and jobs[e["output"]]["kind"]=="validation","audit_eval_output")
        require(e["checkpoints"]=={expected[r]:audit["input_sha256"][expected[r]] for r in ("no_aug_checkpoint","aug_checkpoint")},"audit_eval_checkpoint_binding")
        evaluations[role]=e
    require(set(evaluations)=={(s,p,c) for s in v.SEEDS for p in ("single","dual") for c in v.COHORTS},"incomplete_audit_evaluations")
    return audit,models

def extend_selection(main,additional):
    require(set(additional)==set(CONFIGS),"only_registered_fixedcenter2")
    require(main["selection"]["preselected"] is None and main["selection"]["eligible_configurations"]==[],"fixed_original_main_result")
    baseline=main["selection"]["baseline_signal"];results=copy.deepcopy(main["selection"]["configurations"]);eligible=[]
    for config in CONFIGS:
        signal=additional[config]
        require(set(signal)==set(baseline) and all(type(x) in (int,float) and math.isfinite(x) for x in signal.values()) and signal["cost_seconds"]>=0,"finite_selection_signals")
        gates=dict(normal_MRR=signal["MRR"]>baseline["MRR"],top1=signal["top1"]>=baseline["top1"],
            crop=signal["frequency_crop_MRR"]>=baseline["frequency_crop_MRR"],dropout=signal["peak_dropout_MRR"]>=baseline["peak_dropout_MRR"],source_macro=signal["macro_MRR"]>=baseline["macro_MRR"])
        results[config]=dict(signal=signal,gates=gates,eligible=all(gates.values()),delta={k:signal[k]-baseline[k] for k in baseline})
        if all(gates.values()):eligible.append(config)
    chosen=None
    if eligible:
        best=max(additional[c]["MRR"] for c in eligible)
        chosen=min((c for c in eligible if best-additional[c]["MRR"]<=1e-12),key=lambda c:(additional[c]["cost_seconds"],c))
    require({c:results[c] for c in v.CONFIGS}==main["selection"]["configurations"],"main8_changed")
    return dict(preselected=chosen,baseline="frozen_single",baseline_signal=baseline,configurations=results,
        eligible_configurations=eligible,promotion=False,selection_scope="train_val_only",main8_preserved=True)

def run(args,out):
    root=Path(args.root).resolve();pins={};verified={}
    for path in (__file__,v.__file__):v.read_pinned(path,pins)
    queue_path=Path(args.queue).resolve();queue=v.read_json(queue_path,pins)
    require(pins[str(queue_path)]==QUEUE_SHA,"wrong_actual_additional_queue")
    module=str(root/"h_pool_fixed_center_ranker.py")
    require(queue["source_inputs_sha256"].get(module)==MODULE_SHA,"fixed_ranker_source_changed")
    require(queue["source_inputs_sha256"].get(str(root/"ADDITIONAL_PLAN.json"))==PLAN_SHA and queue["source_inputs_sha256"].get(str(root/"ADDITIONAL_ADMISSION_GATE.json"))==GATE_SHA,"plan_and_admission_binding")
    jobs=verify_driver(args.run,queue_path,queue,pins,verified)
    audit,models=read_audit(root,args.training_audit,args.training_audit_review,queue,queue_path,jobs,pins,verified)
    main_dir=root/"validation/main_matrix";v.exit0(main_dir/"EXIT_CODE",pins)
    main=v.read_json(main_dir/"summary.json",pins)
    require(pins[str((main_dir/"summary.json").resolve())]==MAIN_SHA,"original_main_summary_changed")
    original_tsv=v.read_pinned(main_dir/"metrics.tsv",pins)
    require(pins[str((main_dir/"metrics.tsv").resolve())]==MAIN_TSV_SHA and main["metric_rows"]==1188,"original_main_TSV_changed")
    reference={}
    for cohort in v.COHORTS:
        path=root/f"diagnostics/frozen/s193/single"/cohort/"predictions.jsonl"
        raw=v.read_pinned(path,pins)
        require(pins[str(path.resolve())]==main["input_sha256"].get(str(path.resolve())),"direct_frozen_baseline_changed")
        reference[cohort]=[json.loads(line) for line in raw.splitlines() if line.strip()]
    data={};metadata={};expected_outputs=set()
    for config in CONFIGS:
        pool=config.rsplit("_",1)[1];data[config]={};metadata[config]={}
        for seed in v.SEEDS:
            data[config][seed]={};metadata[config][seed]={}
            for cohort in v.COHORTS:
                expected=expected_args(root,config,seed,cohort);folder=Path(expected["out"])
                rows,meta=fixed_read_eval(folder,seed,pool,cohort,jobs,pins,verified,expected)
                summary=v.read_json(folder/"summary.json",pins)
                require(summary["center_estimate_sha256"]==[models[(seed,a)]["center_estimate_sha256"] for a in (False,True)],"evaluation_center_role_SHA")
                for aug in (False,True):
                    m=models[(seed,aug)];model=Path(m["output"])
                    require(summary["checkpoints"][str(model/"best.pt")]==audit["input_sha256"][str(model/"best.pt")],"best_weights_audited")
                signature=lambda rr:{v.identity(r):r["mask_content_sha256"] for r in rr}
                require(signature(rows)==signature(reference[cohort]),"cross_family_or_seed_query_mask_changed")
                refmeta=main["metadata"]["frozen_single"]["193"][cohort]
                require(meta["query_sha256"]==refmeta["query_sha256"] and meta["candidate_order_sha256"]==refmeta["candidate_order_sha256"],"frozen_reference_identity")
                data[config][seed][cohort]=rows;metadata[config][seed][cohort]=meta;expected_outputs.add(str(folder))
    require(expected_outputs=={p for p,j in jobs.items() if j["kind"]=="validation"},"unexpected_or_missing_eval")
    table,signals=v.aggregate(data,metadata);require(len(table)==216,"additional_strata_count")
    selection=extend_selection(main,signals)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=FIELDS,delimiter="\t");writer.writerows(table)
    (out/"metrics.tsv").write_bytes(original_tsv+stream.getvalue().encode())
    v.require_sha(pins,{});v.require_sha(verified,{})
    result=dict(schema="pool_joint_validation_20261023",selection=selection,
        signals={**main["signals"],**signals},additional_signals=signals,metadata={**main["metadata"],**metadata},
        main_matrix_selection=main["selection"],main_matrix_summary_sha256=MAIN_SHA,main8_preserved=True,
        historical_incumbent=main["historical_incumbent"],metric_rows=1404,additional_metric_rows=216,
        training_audit=dict(status="PASS",path=str(Path(args.training_audit).resolve()),queue_sha256=QUEUE_SHA),
        main_validation_summary=dict(path=str((main_dir/"summary.json").resolve()),sha256=MAIN_SHA),
        fixed_center_training_audit=dict(path=str((Path(args.training_audit)/"summary.json").resolve()),sha256=pins[str((Path(args.training_audit)/"summary.json").resolve())]),
        additional_queue=dict(path=str(queue_path),sha256=QUEUE_SHA),additional_run=str(Path(args.run).resolve()),
        input_sha256={**main["input_sha256"],**pins},scientific_inputs_sha256={**main["scientific_inputs_sha256"],**verified},
        source_sha256={**main["source_sha256"],str(Path(__file__).resolve()):v.sha(__file__),str(Path(v.__file__).resolve()):HELPER_SHA},
        driver_evidence={**main["driver_evidence"],"additional_training_jobs":4,"additional_validation_jobs":8,"additional_all_dual_exit0":True},
        external_api_calls=0,independent_acceptance_status=main["independent_acceptance_status"],independent_eligibility_status="unavailable",promotion=False)
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result

def main():
    p=argparse.ArgumentParser()
    for n in ("root","queue","run","training-audit","training-audit-review","out"):p.add_argument("--"+n,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    try:
        value=run(a,out);(out/"EXIT_CODE").write_text("0\n")
        print(json.dumps(dict(preselected=value["selection"]["preselected"],metric_rows=value["metric_rows"],promotion=False)))
    except BaseException as e:
        (out/"EXIT_CODE").write_text("1\n");(out/"failure.json").write_text(json.dumps({"error":repr(e)})+"\n");raise
if __name__=="__main__":main()

"""功能概述：为已闭包的保存tensor分析登记精确输入和有界job，不解码tensor或计算科学统计。
build_request仅读取授权/闭包/输入库存JSON并散列逐字节文件；原NN inner0/external1保持FAIL，
closure inner0/external0只证明产物闭包。main独占创建analysis_closed_nullable_rank_request.json及unsigned job。
分析器和独立A/B凭证才控制真实科学执行；本文件不创建凭证、不加载模型、不新增样本或NN。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import analyze_diagnostics_closed_outputs_nullable_rank as analyzer

ROOT=Path(__file__).resolve().parent
PYTHON="/Users/andrewluan/.venvs/spectrexpert-local/bin/python"
SCOPE=analyzer.EXECUTION_SCOPE


def sha(path):
    return analyzer.file_sha(path)


def ref(path):
    p=analyzer.safe_path(path)
    return dict(path=str(p),sha256=sha(p))


def build_request(root=ROOT):
    root=Path(root).resolve()
    assert root==ROOT, "sole_diagnostic_write_root"
    inventory=json.loads((root/"input_inventory.json").read_text())
    stage=root/"collected"/"replay_device_indexing_attempt"
    manifest_ref=ref(root/"collected"/"replay_device_indexing_remote_manifest.json")
    manifest=json.loads(Path(manifest_ref["path"]).read_text())
    legacy=[]
    for record in inventory["prediction_files"]:
        value=ref(record["path"])
        assert value["sha256"]==record["registered_collection_sha256"], "legacy_registered_SHA"
        value.update(cap=record["cap"],seed=record["seed"],pool=record["pool"],
                     cohort="train" if record["cohort"]=="fixed_train400" else record["cohort"])
        legacy.append(value)
    assert len(legacy)==20, "exact20_legacy"
    binding=next(x for x in inventory["bindings"] if x["role"]=="fixed_mechanism")
    contract=inventory["cohort_contract"]
    sources={str(root/name):sha(root/name) for name in
             ("analyze_diagnostics_closed_outputs_nullable_rank.py","top10_barrier_math.py","prepare_closed_output_analysis_request_nullable_rank.py")}
    request=dict(schema=analyzer.SCHEMA+"_request",execution_scope=SCOPE,
                 execution_authorized=True,training=False,test_read=False,new_NN_budget=0,
                 write_root=str(root),fixed_manifest=ref(inventory["fixed_train_population"]["path"]),
                 query_manifest=ref(root/"query_manifest.json"),legacy_predictions=legacy,
                 qualification_contract=ref(contract["path"]),qualification_bindings=ref(binding["path"]),
                 worker_outputs=[dict(summary=ref(stage/"worker"/"summary.json"),index=ref(stage/"worker"/"rows.jsonl"))],
                 analysis_source_sha256=sources,
                 completed_output_provenance=dict(
                     original_decision=ref(root/"original_scientific_decoding_decision.json"),
                     original_decision_text=ref(root/"ORIGINAL_SCIENTIFIC_DECODING_DECISION.md"),
                     source_execution=ref(stage/"actual_execution.json"),
                     closure_execution=ref(root/"collected"/"failed_supervision_output_closure_attempt"/"actual_execution.json"),
                     closure_report=ref(root/"collected"/"failed_supervision_output_closure_attempt"/"worker"/"closure.json"),
                     closure_review=ref(root/"output_closure_strict_independent_result_review.json"),
                     sealed_manifest=manifest_ref,source_stage_root=str(stage)),
                 source_NN_stage_status="FAIL_SUPERVISION",source_actual_inner_exit=0,source_actual_external_exit=1,
                 source_RSS_gap_status="UNKNOWN_NOT_CERTIFIED",original_resource_envelope_recertified=False,
                 model_state_assertion_scope="SOURCE_BOUND_NOT_INDEPENDENTLY_REMEASURED",
                 reference_records_per_seed_cap_role_pool=4096,expected_C10_released_old87=66,
                 expected_query_seed_pairs=826,expected_shards=1652,max_seconds=3600,
                 max_artifact_bytes=4*1024**3)
    assert request["qualification_contract"]["sha256"]==contract["sha256"]
    assert request["qualification_bindings"]["sha256"]==binding["sha256"]
    for name,relative in (("summary","worker/summary.json"),("index","worker/rows.jsonl")):
        assert request["worker_outputs"][0][name]["sha256"]==manifest["files"][relative]["sha256"]
    # 只做元数据门禁和字节SHA核验；本函数绝不调用worker_rows/run_analysis。
    pins=analyzer.Pins()
    analyzer.validate_completed_provenance(request,pins)
    pins.finish()
    return {k:v for k,v in request.items() if not k.startswith("_")}


def unsigned_job(request_ref,root=ROOT):
    root=Path(root).resolve()
    source=root/"analyze_diagnostics_closed_outputs_nullable_rank.py"
    return dict(schema="top10_bounded_stage_v1",execution_scope=SCOPE,
                write_root=str(root),stage_dir=str(root/"analysis_closed_attempt1"),
                max_seconds=3600,max_rss_bytes=32*1024**3,max_artifact_bytes=4*1024**3,
                max_cpu_threads=4,cwd=str(root),
                expected_input_admission=str(root/"analysis_closed_input_independent_admission.json"),
                expected_stage_admission=str(root/"analysis_closed_stage_independent_admission.json"),
                source_sha256={str(p):sha(p) for p in
                    (source,root/"top10_barrier_math.py",Path(request_ref["path"]))},
                argv=[PYTHON,"-B",str(source),"--request",request_ref["path"],
                      "--request-sha256",request_ref["sha256"],"--admission",
                      str(root/"analysis_closed_input_independent_admission.json"),"--admission-sha256",
                      "REVIEW_SHA_TO_BE_BOUND","--out",str(root/"analysis_closed_attempt1"/"results"),"--execute"])


def save_exclusive(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args()
    assert args.prepare,"explicit_metadata_prepare_required"
    request=build_request()
    save_exclusive(ROOT/"analysis_closed_nullable_rank_request.json",request)
    request_ref=ref(ROOT/"analysis_closed_nullable_rank_request.json")
    save_exclusive(ROOT/"analysis_closed_nullable_rank_job_unsigned.json",unsigned_job(request_ref))
    print(json.dumps(dict(request=request_ref,unsigned_job=ref(ROOT/"analysis_closed_nullable_rank_job_unsigned.json"),
                          tensor_decodes=0,model_loads=0,new_NN_calls=0,aggregation_performed=False)))


if __name__=="__main__":
    main()

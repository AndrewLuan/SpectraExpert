"""功能概述：为单次composed保存量定位登记Mac≤180s子阶段request/job，仅元数据操作。
CPU定位之后必须以同一固定ref进入JJ≤120s cuda:0子阶段；本文件不执行定位、不自动重试或换设备。
原失败、旧source/request均保留；输入/阶段独审通过后仅root执行。新source只写当前诊断目录。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import probe_saved_jacobian_failure_mac_stage as p

ROOT=Path(__file__).resolve().parent
PYTHON="/Users/andrewluan/.venvs/spectrexpert-local/bin/python"


def ref(path):
    path=p.a.safe_path(path)
    return dict(path=str(path),sha256=p.a.file_sha(path),bytes=path.stat().st_size)


def save(name,value):
    with(ROOT/name).open("x")as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args();assert args.prepare,"explicit_metadata_prepare"
    sources={str(ROOT/name):p.a.file_sha(ROOT/name)for name in
        ("probe_saved_jacobian_failure_mac_stage.py","prepare_saved_jacobian_failure_mac_stage.py",
         "analyze_diagnostics_closed_outputs_nullable_rank.py","top10_barrier_math.py","runtime_device_indexing.py")}
    decision=ref(ROOT/"original_saved_jacobian_probe_decision.json")
    mac=ref(ROOT/"Mac_saved_algebra_runtime_preflight.json")
    jj=ref(ROOT/"JJ_saved_algebra_runtime_preflight.json")
    failure=ref(ROOT/"analysis_closed_attempt1_independent_failure_review.json")
    assert decision["sha256"]==p.PROBE_DECISION_SHA and mac["sha256"]==p.MAC_PREFLIGHT_SHA
    assert jj["sha256"]=="240f96ec017be245d3a60d763b7906724accd0f33205458270aed2f89a5f1814"
    assert failure["sha256"]==p.ANALYSIS_FAILURE_REVIEW_SHA
    failed=ref(ROOT/"analysis_closed_attempt1"/"actual_execution.json")
    outcome=json.loads(Path(failed["path"]).read_text())
    p.a.exact_exit(outcome,"actual_inner_exit",1);p.a.exact_exit(outcome,"actual_external_exit",1)
    request=dict(schema=p.SCHEMA+"_request",execution_scope=p.SCOPE,write_root=str(ROOT),
        parent_analysis_request=ref(ROOT/"analysis_closed_nullable_rank_request.json"),failed_analysis_execution=failed,
        probe_decision=decision,Mac_runtime_preflight=mac,JJ_runtime_preflight=jj,analysis_failure_review=failure,
        saved_worker_source=ref(ROOT/"runtime_device_indexing.py"),source_sha256=sources,
        algebra_device="cpu",campaign_algebra_device="cuda:0",JJ_fallback_authorized=False,
        max_shards=1652,stop_after_first_original_CPU_guard_failure=True,new_NN_budget=0,
        training=False,test_read=False,max_seconds=180,max_output_bytes=4*1024**2,
        no_model_load=True,no_population_aggregation=True,original_guard_released=False,
        campaign_max_execution_seconds=300,pending_same_ref_JJ_CUDA_substage_max_seconds=120)
    save("saved_jacobian_mac_stage_request.json",request)
    request_ref=ref(ROOT/"saved_jacobian_mac_stage_request.json")
    stage=ROOT/"saved_jacobian_mac_stage_attempt1"
    job=dict(schema="top10_bounded_stage_v1",execution_scope=p.SCOPE,write_root=str(stage),stage_dir=str(stage),
        max_seconds=180,max_rss_bytes=4*1024**3,max_artifact_bytes=8*1024**2,max_cpu_threads=4,cwd=str(ROOT),
        source_sha256={**sources,request_ref["path"]:request_ref["sha256"]},
        expected_input_admission=str(ROOT/"saved_jacobian_mac_input_independent_admission.json"),
        expected_stage_admission=str(ROOT/"saved_jacobian_mac_stage_independent_admission.json"),
        argv=[PYTHON,"-B",str(ROOT/"probe_saved_jacobian_failure_mac_stage.py"),"--request",request_ref["path"],
            "--request-sha256",request_ref["sha256"],"--admission",str(ROOT/"saved_jacobian_mac_input_independent_admission.json"),
            "--admission-sha256","REVIEW_SHA_TO_BE_BOUND","--out",str(stage/"results"),"--execute"])
    save("saved_jacobian_mac_stage_job_unsigned.json",job)
    print(json.dumps(dict(request=request_ref,unsigned_job=ref(ROOT/"saved_jacobian_mac_stage_job_unsigned.json"),
        campaign_algebra_device="cuda:0",current_substage="Mac CPU localization only",tensor_decodes=0,new_NN_calls=0)))


if __name__=="__main__":
    main()

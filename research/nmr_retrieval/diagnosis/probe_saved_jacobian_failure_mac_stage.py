"""功能概述：单次composed probe的Mac定位子阶段，在独审授权后对封存Jacobian定位；不运行网络或科学聚合。
按原index与3cd验证顺序扫描，首个原CPU链式校验失败即停止；所有输入先核封存SHA/bytes。
仅用保存logit/grad/delta/weightedJac核CPU原dtype、FP64与可选原CUDA纯tanh代数。
定位成功只说明找到同一guard失败，不解除原guard，不改变2e-5/2e-6或原NN/分析失败状态。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import analyze_diagnostics_closed_outputs_nullable_rank as a

SCHEMA="saved_jacobian_failure_mac_stage_v1"
SCOPE="SAVED_JACOBIAN_MAC_LOCALIZATION_STAGE_ONLY"
ROOT=Path(__file__).resolve().parent


def vectors(actual,expected):
    import torch
    actual=actual.detach().cpu();expected=expected.detach().cpu()
    typed_bits=(actual.dtype==expected.dtype and tuple(actual.shape)==tuple(expected.shape) and
        torch.equal(actual.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8)))
    actual=actual.to(torch.float64);expected=expected.to(torch.float64)
    diff=(actual-expected).abs();bound=a.FIXED_ATOL+a.FIXED_RTOL*expected.abs()
    bad=diff>bound
    return dict(passes_fixed_tolerance=bool(torch.allclose(actual,expected,rtol=a.FIXED_RTOL,atol=a.FIXED_ATOL)),
                bit_equal=bool(typed_bits),elements=actual.numel(),
                mismatching_elements=int(bad.sum()),max_absolute_residual=float(diff.max()),
                max_residual_over_fixed_bound=float((diff/bound).max()))


def expected_cpu(trace,cap):
    import torch
    positions=trace["input_jacobians"]["positions"]
    local=1-torch.tanh(trace["logit"][positions]/cap).square()
    return local,trace["input_jacobians"]["dlogit_dinput36"]*local[:,None]


def first_mismatch(shard):
    """顺序与3cd validate_saved_shard一致；本函数不读候选成本/排名。"""
    import torch
    for cap in (5,10):
        for role in a.ROLES:
            traces=[("natural512",a._cap(shard["roles"],cap)[role])]
            if shard.get("loss_roles"):
                traces.append(("full513_loss_only",a._cap(shard["loss_roles"],cap)[role]))
            for shape,trace in traces:
                jac=trace["input_jacobians"];positions=jac["positions"];m=len(positions)
                n=512 if shape=="natural512" else 513
                a.bm.require(positions==sorted(set(positions))and m>0 and
                    all(type(p)is int and 0<=p<n for p in positions),"fixed_saved_jacobian_positions")
                for name,shape_expected in (("logit",(n,)),):
                    a.check_tensor(trace[name],shape_expected,name,(torch.float32,torch.float64))
                for name in ("dlogit_dinput36","ddelta_dinput36"):
                    a.check_tensor(jac[name],(m,36),name,(torch.float32,torch.float64))
                a.finite_present(dict(logit=trace["logit"],delta=trace["delta"],scale=trace["scale"],jac=jac))
                local,expected=expected_cpu(trace,cap)
                if not vectors(jac["ddelta_dinput36"],expected)["passes_fixed_tolerance"]:
                    return cap,role,shape,trace,local,expected
    return None


def detailed_trace(trace,cap,role,shape,algebra_device):
    import torch
    jac=trace["input_jacobians"];positions=jac["positions"]
    logits=trace["logit"][positions];grad=jac["dlogit_dinput36"];saved=jac["ddelta_dinput36"]
    local,expected=expected_cpu(trace,cap)
    local64=1-torch.tanh(logits.to(torch.float64)/cap).square()
    out=dict(cap=cap,role=role,shape=shape,positions=positions,
        dtype=dict(logit=str(logits.dtype),dlogit=str(grad.dtype),ddelta=str(saved.dtype)),
        original_CPU_guard=vectors(saved,expected),FP64_analytic_comparison=vectors(saved,grad.to(torch.float64)*local64[:,None]),
        local_native_CPU=local.tolist(),local_FP64=local64.tolist(),saved_logit=logits.tolist(),
        algebra_device=algebra_device,CUDA_algebra=None,
        interpretation="Saved-value algebra only; CPU/CUDA equivalence and actual failure cause are not assumed.")
    nonzero=grad!=0
    out["DD_over_dlogit"]=dict(nonzero_denominator_components=int(nonzero.sum()),zero_denominator_components=int((~nonzero).sum()),
        zero_grad_nonzero_DD_components=int(((~nonzero)&(saved!=0)).sum()),by_candidate=[])
    for i,pos in enumerate(positions):
        ratio=(saved[i][nonzero[i]].to(torch.float64)/grad[i][nonzero[i]].to(torch.float64))
        out["DD_over_dlogit"]["by_candidate"].append(dict(position=pos,
            minimum=float(ratio.min())if ratio.numel()else None,maximum=float(ratio.max())if ratio.numel()else None,
            saved_delta_div_cap=float(trace["delta"][pos]/cap),delta_div_cap_semantics="rounded_estimate_not_saved_tanh",
            tanh_CPU=float(torch.tanh(logits[i]/cap))))
    loss=trace.get("actual_train_loss")
    if loss is not None:
        saved_local=loss["tanh_local_derivative"][positions]
        out["same_shape_saved_loss_local"]=dict(local=saved_local.tolist(),delta_chain=vectors(saved,grad*saved_local[:,None]))
    else:
        out["same_shape_saved_loss_local"]=None
    weighted=jac.get("weighted_dcost_dinput36")
    if weighted is not None:
        weight=trace["inference_role_weight"];scale0=trace["scale"][0]
        out["weighted_native_multiplication_order"]=dict(weight=weight,scale0=float(scale0),
            from_saved_DD=vectors(weighted,saved*weight*scale0),
            from_CPU_tanh=vectors(weighted,grad*local[:,None]*weight*scale0))
    else:out["weighted_native_multiplication_order"]=None
    if algebra_device=="cuda:0":
        z=logits.to(algebra_device);g=grad.to(algebra_device)
        cuda_local=1-torch.tanh(z/cap).square();cuda_expected=g*cuda_local[:,None]
        out["CUDA_algebra"]=dict(local=cuda_local.cpu().tolist(),
            delta_chain=vectors(saved,cuda_expected.cpu()),tanh=torch.tanh(z/cap).cpu().tolist(),
            cuda_device_name=torch.cuda.get_device_name(0),torch_version=torch.__version__,cuda_version=torch.version.cuda,
            semantics="Current requested CUDA saved-value operations; no model call and no claim of identical historical runtime.")
        if weighted is not None:
            out["CUDA_algebra"]["weighted_chain"]=vectors(weighted,(cuda_expected*weight*scale0.to(algebra_device)).cpu())
    return out



PROBE_DECISION_SHA="41e85949a7991ad3e7cf1824b5308d3891fab60cde85adf5a712cb52366dc4b8"
MAC_PREFLIGHT_SHA="b4ed369c30b7ab4f320d83552f7dc984753c3268f00e0882987b0ac675181676"
ANALYSIS_FAILURE_REVIEW_SHA="507a662652f92b9c71c55f73c5946af804a024eeeb033103492586971de360a9"


def tensor_identity(value):
    import torch
    a.bm.require(isinstance(value,torch.Tensor),"tensor_identity_required")
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def check_identity_join(shard,receipt,selected,old,heldout):
    """只核冻结身份/正例连接，不从当前失败新选query或修补身份。"""
    import torch
    qid=a.identity(shard);fixed=old is not None
    a.bm.require(a.identity(receipt)==qid==a.identity(selected)and
        all(shard[k]==receipt[k]for k in("seed","cohort","population","pool"))and
        shard["seed"]==selected["seed"]and shard["cohort"]==selected["cohort"]and
        shard["population"]==selected["population"],"exact_frozen_query_join")
    a.bm.require(shard["cohort"]in a.COHORTS and qid[1]=="observed"and
        shard["fixed400_member"]is fixed and selected.get("fixed400_member",False)is fixed,"exact_fixed400_join")
    q_external=qid[-1]in heldout;q_internal=a.internal_qualification(qid[-1])
    fit=shard["cohort"]=="train"and not q_external and not q_internal
    a.bm.require(shard["actual_train_query"]is fit and shard["query_external_heldout"]is q_external and
        shard["query_internal_group"]is q_internal,"query_eligibility_join")
    ids=shard["candidate_indices"];a.check_tensor(ids,(512,),"natural_ids",(torch.int32,torch.int64))
    a.bm.require(len(ids.unique())==512,"natural_ids_unique")
    target=shard["target_position"]
    a.bm.require(type(target)is int and -1<=target<512 and shard["target_natural"]is(target>=0),"natural_target_join")
    expected=old["target_candidate_index"]if fixed else selected.get("target_candidate_index")
    if target>=0:
        cid=int(ids[target])
        a.bm.require(shard["candidate_groups"][target]==qid[-1]and shard["target_candidate_id"]==cid,
                     "natural_target_group_ID_join")
        a.bm.require(expected is None or cid==expected,"fixed_target_ID_join")
    else:cid=shard.get("target_candidate_id")
    appended=bool(shard.get("loss_roles"));loss_target=shard["loss_target_position"]
    a.bm.require(shard["diagnostic_loss_only_positive_used"]is appended,"loss_shape_join")
    loss_ids=shard.get("loss_candidate_indices")
    if appended:
        a.bm.require(fit and fixed and shard["pool"]=="single"and target==-1 and type(loss_target)is int and loss_target==512 and
            shard["cache_loss_only_positive_present"]is True,"fit_single_only_loss_append")
        a.check_tensor(loss_ids,(513,),"loss_ids",(ids.dtype,))
        a.bm.require(torch.equal(loss_ids[:512],ids)and len(loss_ids.unique())==513 and
            int(loss_ids[512])==expected==cid and shard["loss_candidate_groups"][512]==qid[-1],
            "exact_full513_positive_join")
    else:
        a.bm.require(loss_ids is None and ((fit and target>=0 and type(loss_target)is int and loss_target==target)or
            (not fit and loss_target is None)),"nullable_loss_target_join")
    if fixed:
        a.bm.require(shard["original_bound_obstructed"]is old["original_bound_obstructed"]and
            shard.get("original_other_candidate_index")==old["original_other_candidate_index"]==
            selected.get("original_other_candidate_index"),"old87_immutable_other_join")
    return dict(status="PASS_EXACT_IDENTITY_JOIN",fixed400_member=fixed,
        original87=bool(fixed and old["original_bound_obstructed"]),target_candidate_id=cid,
        natural_target_position=target,loss_target_position=loss_target,
        original_other_candidate_index=shard.get("original_other_candidate_index"),
        natural_candidate_id_tensor=tensor_identity(ids),
        loss_candidate_id_tensor=tensor_identity(loss_ids)if appended else None,
        actual_train_query=fit,query_external_heldout=q_external,query_internal_group=q_internal,
        target_in_natural512=target>=0,loss_only_positive_used=appended)


def validate_probe_trace(trace,n):
    import torch
    for name,shape in (("raw_X",(n,33)),("raw_O",(n,3)),("input36",(n,36)),("logit",(n,)),
                       ("delta",(n,)),("scale",(33,)),("fixed_x_center",(33,)),("fixed_o_center",(3,))):
        a.check_tensor(trace[name],shape,name,(torch.float32,torch.float64))
    a.finite_present(trace)
    hashes=trace["trace_sha256_natural512"if n==512 else"trace_sha256_loss513"]
    a.bm.require(hashes==a.saved_trace_hashes(trace),"saved_typed_trace_hash_join")
    return hashes


def scan_shard(shard):
    import torch
    a.bm.require(set(map(int,shard["roles"]))=={5,10}and len(shard["roles"])==2,"exact_caps")
    passed=0
    for cap in(5,10):
        a.bm.require(set(a._cap(shard["roles"],cap))==set(a.ROLES),"exact_roles")
        for role in a.ROLES:
            traces=[("natural512",a._cap(shard["roles"],cap)[role])]
            if shard.get("loss_roles"):
                a.bm.require(set(map(int,shard["loss_roles"]))=={5,10}and
                    set(a._cap(shard["loss_roles"],cap))==set(a.ROLES),"exact_loss_caps_roles")
                traces.append(("full513_loss_only",a._cap(shard["loss_roles"],cap)[role]))
            for shape,trace in traces:
                n=512 if shape=="natural512"else 513
                validate_probe_trace(trace,n)
                jac=trace["input_jacobians"];positions=jac["positions"];m=len(positions)
                a.bm.require(positions==sorted(set(positions))and m>0 and
                    all(type(p)is int and 0<=p<n for p in positions),"fixed_saved_jacobian_positions")
                for name in("dlogit_dinput36","ddelta_dinput36"):
                    a.check_tensor(jac[name],(m,36),name,(torch.float32,torch.float64))
                local,expected=expected_cpu(trace,cap)
                if not vectors(jac["ddelta_dinput36"],expected)["passes_fixed_tolerance"]:
                    return (cap,role,shape,trace,local,expected),passed
                passed+=1
    return None,passed


def first_mismatch(shard):
    return scan_shard(shard)[0]


def original_row_mismatch(shard,heldout,selected,old):
    """逐row先原样执行3cd全部前置门；仅捕获精确DD guard失败用于定位，其他失败不跳过。"""
    try:
        a.validate_saved_shard(shard,heldout,{**selected,"_fixed400_member":old is not None})
    except ValueError as exc:
        if str(exc)!="fixed_tolerance:saved_delta_Jacobian":
            raise
        mismatch,passed=scan_shard(shard)
        a.bm.require(mismatch is not None,"original_DD_guard_not_reproduced_stop")
        return mismatch,passed
    return None,8 if shard.get("loss_roles")else 4




def load_gate(args):
    import sys
    pins=a.Pins();request=pins.json(dict(path=args.request,sha256=args.request_sha256))
    admission=pins.json(dict(path=args.admission,sha256=args.admission_sha256))
    a.bm.require(args.execute and request["schema"]==SCHEMA+"_request"and request["execution_scope"]==SCOPE and
        admission["status"]=="PASS"and admission["independent"]is True and admission["execution_authorized"]is True and
        admission["execution_scope"]==SCOPE and admission["request_sha256"]==args.request_sha256,"request_specific_independent_PASS")
    for path,digest in request["source_sha256"].items():
        pins.verify(path,digest)
        a.bm.require(admission["reviewed_source_sha256"].get(path)==digest,"reviewed_exact_probe_source")
    for key,digest in(("probe_decision",PROBE_DECISION_SHA),("Mac_runtime_preflight",MAC_PREFLIGHT_SHA),
                     ("analysis_failure_review",ANALYSIS_FAILURE_REVIEW_SHA)):
        a.bm.require(request[key]["sha256"]==digest,"sole_probe_metadata:"+key)
    decision=pins.json(request["probe_decision"])
    a.bm.require(decision["status"]=="AUTHORIZED_ONE_BOUNDED_CUDA_SAVED_VALUE_PROBE_AFTER_INDEPENDENT_GATES"and
        decision["algebra_device"]=="cuda:0"and decision["runtime"]=="original_JJ"and
        decision["fallback_authorized"]is False and decision["new_NN_budget"]==0 and
        decision["new_weight_loads"]==0 and decision["fixed_rtol"]==a.FIXED_RTOL and
        decision["fixed_atol"]==a.FIXED_ATOL and decision["threshold_relaxation_authorized"]is False,
        "fixed_CUDA_campaign_decision")
    for flag in("new_training","new_samples","test_read","scientific_definition_change_authorized","automatic_retry_authorized"):
        a.bm.require(decision[flag]is False,"probe_decision_forbids:"+flag)
    preflight=pins.json(request["Mac_runtime_preflight"]);review=pins.json(request["analysis_failure_review"])
    a.bm.require(request["JJ_runtime_preflight"]["sha256"]=="240f96ec017be245d3a60d763b7906724accd0f33205458270aed2f89a5f1814",
                 "sole_JJ_runtime_preflight")
    jj=pins.json(request["JJ_runtime_preflight"])
    a.bm.require(jj["cuda_available"]is True and jj["cuda0_device"]=="NVIDIA A800 80GB PCIe","JJ_CUDA_campaign_required")
    a.bm.require(review["status"]=="FAIL_ANALYSIS_PRESERVED_ACCOUNTING_PASS"and review["independent"]is True and
        review["failure"]=="fixed_tolerance:saved_delta_Jacobian","original_failure_review")
    parent=pins.json(request["parent_analysis_request"])
    provenance=a.validate_completed_provenance(parent,pins)
    failed=pins.json(request["failed_analysis_execution"])
    a.exact_exit(failed,"actual_inner_exit",1);a.exact_exit(failed,"actual_external_exit",1)
    a.bm.require(request["max_shards"]==1652 and request["stop_after_first_original_CPU_guard_failure"]is True and
        request["algebra_device"]=="cpu"and request["campaign_algebra_device"]=="cuda:0"and
        request["JJ_fallback_authorized"]is False and type(request["new_NN_budget"])is int and request["new_NN_budget"]==0 and
        request["max_seconds"]==180 and request["max_output_bytes"]==4*1024**2 and
        request["training"]is False and request["test_read"]is False,"fixed_Mac_substage_bounds")
    closed=pins.json(parent["completed_output_provenance"]["closure_report"])
    a.bm.require(request["saved_worker_source"]["sha256"]==closed["source_refs"]["worker"]["sha256"],
                 "exact_saved_worker_source")
    pins.verify(request["saved_worker_source"]["path"],request["saved_worker_source"]["sha256"])
    selection=pins.json(parent["query_manifest"]);fixed=pins.json(parent["fixed_manifest"])
    selected={(r["seed"],r["cohort"],a.identity(r)):r for r in selection["records"]}
    old={(r["seed"],a.identity(r)):r for r in fixed["records"]}
    a.bm.require(len(selected)==826 and len(old)==400 and
        sum(bool(r["original_bound_obstructed"])for r in old.values())==87,"original_frozen_denominators")
    parent["_selected"]=selected;parent["_old"]=old
    request["_preflight"]=preflight
    return request,parent,admission,provenance,pins


def run(request,parent,admission,provenance,pins,out):
    import torch,sys,platform
    torch.set_num_threads(4)
    preflight=request["_preflight"]
    a.bm.require(str(torch.__version__)==preflight["torch"]and sys.executable==preflight["python"]and
        platform.machine()=="arm64","current_Mac_runtime_changed_no_fallback")
    started=time.monotonic();seen=0;trace_count=0;found=None
    entry=parent["worker_outputs"][0]
    summary=pins.json(entry["summary"])
    a.bm.require(summary["status"]=="PASS"and summary["model_state_unchanged"]is True and
        summary["inputs_unchanged"]is True and summary["test_read"]is False,"worker_summary_integrity")
    for name in("optimizer_instances","optimizer_steps","weight_updates","new_spectrum_forwards",
                "forward_bank_calls","external_api_calls"):
        a.bm.require(summary[name]==0,"worker_forbidden_operation")
    a.bm.require(summary["natural512_inference_role_row_forwards"]==summary["natural_role_row_forwards"]and
        summary["full513_loss_role_row_forwards"]==summary["loss_only_role_row_forwards"]and
        summary["role_row_forwards"]==summary["natural_role_row_forwards"]+summary["loss_only_role_row_forwards"],
        "worker_shape_counter_aliases")
    a.bm.require(summary["natural512_inference_role_row_forwards"]<=7040 and
        summary["full513_loss_role_row_forwards"]<=1600 and summary["role_row_forwards"]<=8640,
        "global_double_shape_forward_budget")
    index=pins.verify(entry["index"]["path"],entry["index"]["sha256"])
    lines=index.read_text().splitlines();a.bm.require(len(lines)==1652,"exact_index_rows")
    for line in lines:
        seen+=1;a.bm.require(seen<=request["max_shards"],"fixed_shard_bound")
        receipt=json.loads(line);relative=Path(receipt["tensor_file"])
        a.bm.require(not relative.is_absolute()and".."not in relative.parts,"relative_sealed_shard")
        path=pins.verify(index.parent/relative,receipt["tensor_sha256"])
        closed=parent["_sealed_files"]["worker/"+str(relative)]
        a.bm.require(closed["sha256"]==receipt["tensor_sha256"]and path.stat().st_size==closed["bytes"],"closed_shard_bytes")
        shard=torch.load(path,map_location="cpu",weights_only=True)
        a.bm.require(a.identity(shard)==a.identity(receipt)and
            all(shard[z]==receipt[z]for z in("seed","cohort","pool","population")),"shard_receipt_identity")
        selected=parent["_selected"][(shard["seed"],shard["cohort"],a.identity(shard))]
        mismatch,passed=original_row_mismatch(shard,parent["_heldout_groups"],selected,parent["_old"].get((shard["seed"],a.identity(shard))));trace_count+=passed
        if mismatch:
            join=check_identity_join(shard,receipt,selected,parent["_old"].get((shard["seed"],a.identity(shard))),parent["_heldout_groups"])
            cap,role,shape,trace,local,expected=mismatch
            hashes=trace["trace_sha256_natural512"if shape=="natural512"else"trace_sha256_loss513"]
            a.bm.require(time.monotonic()-started<=request["max_seconds"],"Mac_wall_budget")
            found=dict(shard_index_one_based=seen,shard_ref=dict(path=str(path),relative=str(relative),
                sha256=receipt["tensor_sha256"],bytes=closed["bytes"]),seed=shard["seed"],cohort=shard["cohort"],
                population=shard["population"],pool=shard["pool"],kind=shard["kind"],query_identity=shard["query_identity"],
                identity_join=join,selection=dict(cap=cap,role=role,shape=shape,positions=trace["input_jacobians"]["positions"]),
                trace_sha256_ordered_sha256=hashlib.sha256(a.canonical(hashes).encode()).hexdigest(),
                selected_trace_sha256={str(p):hashes[p]for p in trace["input_jacobians"]["positions"]},
                evidence=detailed_trace(trace,cap,role,shape,"cpu"))
            break
        a.bm.require(time.monotonic()-started<=request["max_seconds"],"Mac_wall_budget")
    a.bm.require(found is not None,"Mac_failure_not_reproduced_no_refill")
    root=a.safe_path(request["write_root"]);output=Path(out).resolve()
    a.bm.require(output.is_relative_to(root)and not output.exists(),"exclusive_Mac_output")
    pins.finish();output.mkdir(parents=True,exist_ok=False)
    result=dict(schema=SCHEMA,status="PASS_MAC_FIRST_FAILURE_LOCALIZED",execution_scope=SCOPE,
        campaign_status="PENDING_JJ_SAME_REF_CUDA_SUBSTAGE",original_analysis_status="FAIL_GUARD_PRESERVED",
        original_guard_released=False,historical_attempt_first_identity=None,
        historical_attempt_first_identity_status="UNKNOWN_NOT_RECORDED_IN_OLD_STDERR",
        current_Mac_first_failure=found,current_Mac_same_source_order_reproduction=True,
        selection={**{k:found[k]for k in("seed","cohort","population","pool","kind","query_identity")},**found["selection"]},
        identity_join=found["identity_join"],
        shard=dict(relative=found["shard_ref"]["relative"],local_path=found["shard_ref"]["path"],
            remote_path="/home/user/IRfingerprints/SpectrExpert/outputs/nmr_top10_barrier_diagnosis_20261025/replay_device_indexing_attempt/worker/"+found["shard_ref"]["relative"],
            sha256=found["shard_ref"]["sha256"],bytes=found["shard_ref"]["bytes"]),
        trace_sha256_ordered_sha256=found["trace_sha256_ordered_sha256"],
        selected_position_trace_sha256=found["selected_trace_sha256"],
        runtime=dict(python=sys.executable,python_version=sys.version,torch=str(torch.__version__),
                     platform=platform.platform(),machine=platform.machine()),
        source_NN_provenance=provenance,probe_decision=request["probe_decision"],JJ_runtime_preflight=request["JJ_runtime_preflight"],
        request_sha256=admission["request_sha256"],independent_admission=admission,
        examined_shards=seen,preceding_passing_traces=trace_count,
        fixed_rtol=a.FIXED_RTOL,fixed_atol=a.FIXED_ATOL,saved_VJP_independently_recertified=False,
        model_loads=0,new_NN_calls=0,training=False,test_read=False,external_API_calls=0,
        scientific_population_statistics=False,input_sha256=pins.checked,elapsed_seconds=time.monotonic()-started)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    a.bm.require(len(payload.encode())<=request["max_output_bytes"],"Mac_output_bound")
    (output/"mac_localization.json").write_text(payload)
    pins.finish();(output/"EXIT_CODE").write_text("0\n")
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in("request","request-sha256","admission","admission-sha256","out"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    request,parent,admission,provenance,pins=load_gate(args)
    result=run(request,parent,admission,provenance,pins,args.out)
    print(json.dumps(dict(status=result["status"],new_NN_calls=0,original_guard_released=False)))


if __name__=="__main__":
    main()

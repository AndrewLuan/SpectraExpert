"""功能概述：在独审授权后，对封存Jacobian做一次有界失败定位；不运行网络或科学聚合。
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

SCHEMA="saved_jacobian_failure_probe_v1"
SCOPE="SAVED_JACOBIAN_FAILURE_LOCALIZATION_ONLY"
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


def load_gate(args):
    pins=a.Pins();request=pins.json(dict(path=args.request,sha256=args.request_sha256))
    admission=pins.json(dict(path=args.admission,sha256=args.admission_sha256))
    a.bm.require(args.execute and request["schema"]==SCHEMA+"_request" and request["execution_scope"]==SCOPE and
        admission["status"]=="PASS" and admission["independent"] is True and admission["execution_authorized"] is True and
        admission["execution_scope"]==SCOPE and admission["request_sha256"]==args.request_sha256,"request_specific_independent_PASS")
    for path,digest in request["source_sha256"].items():
        pins.verify(path,digest)
        a.bm.require(admission["reviewed_source_sha256"].get(path)==digest,"reviewed_exact_probe_source")
    parent=pins.json(request["parent_analysis_request"])
    provenance=a.validate_completed_provenance(parent,pins)
    closed=pins.json(parent["completed_output_provenance"]["closure_report"])
    a.bm.require(request["saved_worker_source"]["sha256"]==closed["source_refs"]["worker"]["sha256"],
                 "exact_saved_worker_source")
    pins.verify(request["saved_worker_source"]["path"],request["saved_worker_source"]["sha256"])
    failed=pins.json(request["failed_analysis_execution"])
    a.exact_exit(failed,"actual_inner_exit",1);a.exact_exit(failed,"actual_external_exit",1)
    a.bm.require(type(request["max_shards"])is int and request["max_shards"]==1652 and
        request["stop_after_first_original_CPU_guard_failure"]is True and
        request["algebra_device"]in("cpu","cuda:0")and type(request["new_NN_budget"])is int and request["new_NN_budget"]==0 and
        request["max_seconds"]==300 and request["max_output_bytes"]==4*1024**2 and
        request["training"]is False and request["test_read"]is False,"fixed_probe_bounds")
    return request,parent,admission,provenance,pins


def run(request,parent,admission,provenance,pins,out):
    import torch
    torch.set_num_threads(4)
    device=request["algebra_device"]
    a.bm.require(device!="cuda:0"or torch.cuda.is_available(),"requested_CUDA_unavailable_no_fallback")
    started=time.monotonic();seen=0;trace_count=0;found=None
    entry=parent["worker_outputs"][0]
    index=pins.verify(entry["index"]["path"],entry["index"]["sha256"])
    for line in index.read_text().splitlines():
        seen+=1;a.bm.require(seen<=request["max_shards"],"fixed_shard_bound")
        receipt=json.loads(line);relative=Path(receipt["tensor_file"])
        a.bm.require(not relative.is_absolute()and ".."not in relative.parts,"relative_sealed_shard")
        path=pins.verify(index.parent/relative,receipt["tensor_sha256"])
        closed=parent["_sealed_files"]["worker/"+str(relative)]
        a.bm.require(closed["sha256"]==receipt["tensor_sha256"]and path.stat().st_size==closed["bytes"],"closed_shard_bytes")
        shard=torch.load(path,map_location="cpu",weights_only=False)
        a.bm.require(a.identity(shard)==a.identity(receipt)and
            all(shard[k]==receipt[k]for k in("seed","cohort","population","pool")),"receipt_identity")
        mismatch=first_mismatch(shard)
        if mismatch:
            cap,role,shape,trace,local,expected=mismatch
            a.bm.require(time.monotonic()-started<=request["max_seconds"],"probe_wall_budget")
            found=dict(shard_index_one_based=seen,shard_ref=dict(path=str(path),sha256=receipt["tensor_sha256"]),
                seed=shard["seed"],cohort=shard["cohort"],population=shard["population"],pool=shard["pool"],
                query_identity=shard["query_identity"],evidence=detailed_trace(trace,cap,role,shape,device))
            break
        trace_count+=8 if shard.get("loss_roles")else 4
        a.bm.require(time.monotonic()-started<=request["max_seconds"],"probe_wall_budget")
    a.bm.require(found is not None,"original_failure_not_reproduced_stop_without_refill")
    root=a.safe_path(request["write_root"]);output=Path(out).resolve()
    a.bm.require(output.is_relative_to(root)and not output.exists(),"exclusive_probe_output")
    pins.finish();output.mkdir(parents=True,exist_ok=False)
    result=dict(schema=SCHEMA,status="PASS_FAILURE_LOCALIZED",execution_scope=SCOPE,
        original_analysis_status="FAIL_GUARD_PRESERVED",original_guard_released=False,
        source_NN_stage_status="FAIL_SUPERVISION",source_NN_provenance=provenance,
        request_sha256=admission["request_sha256"],independent_admission=admission,
        examined_shards=seen,preceding_passing_traces=trace_count,first_failure=found,
        fixed_rtol=a.FIXED_RTOL,fixed_atol=a.FIXED_ATOL,model_loads=0,new_NN_calls=0,
        saved_VJP_independently_recertified=False,
        saved_worker_formula="one logit VJP; saved DD=grad*(1-tanh(logit/C)^2); no second delta VJP",
        training=False,test_read=False,external_API_calls=0,scientific_population_statistics=False,
        input_sha256=pins.checked,elapsed_seconds=time.monotonic()-started)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    a.bm.require(len(payload.encode())<=request["max_output_bytes"],"probe_output_bound")
    (output/"failure_localization.json").write_text(payload)
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

"""功能概述：JJ仅对Mac已关闭定位的同一封存shard重算CPU/CUDA保存量；不运行模型或科学聚合。
admit核独审、41e授权、240f预检、Mac0/0和独审；locate核index/manifest/query及唯一ref字节。
identity_join核正例、512/513物理dtype；algebra分开division/tanh/square/local/DD/weighted，
并做同输入tanh跨backend对照。FP64理想链仅作旁证，不替代原dtype与原2e-5/2e-6容差。
仅一次安全tensor解码；CUDA不可用即失败，不fallback/重试/refill，不回填历史Mac首失败身份。
只在已授权阶段输出有界JSON；source/result独审另行控制执行。
"""
from pathlib import Path
import argparse,hashlib,json,os,platform,resource,sys,time
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[_k]="4"
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
SCHEMA="saved_jacobian_JJ_algebra_request_v1"
SCOPE="SAVED_JACOBIAN_JJ_CUDA_ALGEBRA_ONLY"
DECISION_SHA="41e85949a7991ad3e7cf1824b5308d3891fab60cde85adf5a712cb52366dc4b8"
PREFLIGHT_SHA="240f96ec017be245d3a60d763b7906724accd0f33205458270aed2f89a5f1814"
INDEX_SHA="40fe645c4b8e01a1b7092c7f52fc9c506d797ac03d2f520d5ae3f4cb1354ff76"
MANIFEST_SHA="8cc079d4b9a936c82994eb0c888010d2145d71dc3a8ffd6ff49675ea6ed953b7"
QUERY_SHA="73916f58e8c5e4582039053136186c129813b70a59736fd2a5e3693df7635725"
WORKER_SHA="c8b89568d3db99537db8933bfc311c891fa005073f9c474751bc91256e252552"
FIELDS=("query_order","variant","source","record_id","smiles","group")
ROLES=("no_aug","aug")
RTOL=2e-5
ATOL=2e-6

def need(v,why):
    if not v:raise ValueError(why)
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb")as f:
        for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
    return h.hexdigest()
def safe(p):
    p=Path(p);need(p.is_absolute()and".."not in p.parts and not p.is_symlink(),"registered_absolute_path")
    need(not any(x in("test","sealed")or x.startswith("sealed-")for x in p.parts),"forbidden_path")
    need(p.name not in("research_report.md","supplementary_report.md","cohorts.pt"),"forbidden_payload")
    return p.resolve()
class Pins:
    def __init__(self):self.checked={}
    def ref(self,r):
        p=safe(r["path"]);need(type(r["sha256"])is str and len(r["sha256"])==64 and sha(p)==r["sha256"],"input_SHA")
        if"bytes"in r:need(type(r["bytes"])is int and p.stat().st_size==r["bytes"],"input_bytes")
        self.checked[str(p)]=r["sha256"];return p
    def json(self,r):return json.loads(self.ref(r).read_text())
    def finish(self):
        for p,h in self.checked.items():need(sha(p)==h,"input_changed")
def identity(r):
    q=r.get("query_identity",r);return tuple(q[k]for k in FIELDS)
def capitem(r,c):return r[c]if c in r else r[str(c)]

def admit(args):
    need(args.execute,"explicit_execute_required");pins=Pins()
    A=pins.json(dict(path=args.admission,sha256=args.admission_sha256))
    need(A.get("status")=="PASS"and A.get("independent")is True and A.get("execution_authorized")is True and A.get("execution_scope")==SCOPE,"independent_same_ref_scope")
    r=pins.json(dict(path=args.request,sha256=args.request_sha256));own=sha(__file__)
    need(A.get("request_sha256")==args.request_sha256 and A.get("helper_sha256")==own and r.get("helper_sha256")==own,"exact_helper_request")
    need(r["schema"]==SCHEMA and r["execution_scope"]==SCOPE and r["algebra_device"]=="cuda:0"and r["fallback_authorized"]is False and type(r["new_NN_budget"])is int and r["new_NN_budget"]==0 and r["new_weight_loads"]==0 and r["training"]is False and r["test_read"]is False,"saved_only_request")
    need(r["max_seconds"]==120 and r["max_cpu_threads"]==4 and r["max_rss_bytes"]==4*1024**3 and r["max_gpu_allocated_bytes"]==4*1024**3 and r["max_output_bytes"]==4*1024**2,"fixed_bounds")
    need(safe(r["write_root"])==ROOT,"registered_root");out=safe(args.out)
    need(out.is_relative_to(ROOT)and not out.exists(),"fresh_output")
    for key,h in(("decision_ref",DECISION_SHA),("preflight_ref",PREFLIGHT_SHA),("source_index_ref",INDEX_SHA),("source_manifest_ref",MANIFEST_SHA),("query_manifest_ref",QUERY_SHA),("worker_source_ref",WORKER_SHA)):
        need(r[key]["sha256"]==h,"frozen_ref:"+key)
    d=pins.json(r["decision_ref"]);pre=pins.json(r["preflight_ref"]);pins.ref(r["worker_source_ref"])
    need(d["status"]=="AUTHORIZED_ONE_BOUNDED_CUDA_SAVED_VALUE_PROBE_AFTER_INDEPENDENT_GATES"and d["runtime"]=="original_JJ"and d["algebra_device"]=="cuda:0"and d["fallback_authorized"]is False and d["new_NN_budget"]==0 and d["new_weight_loads"]==0 and d["fixed_rtol"]==RTOL and d["fixed_atol"]==ATOL,"campaign_decision")
    for k in("new_training","new_samples","test_read","scientific_definition_change_authorized","threshold_relaxation_authorized","automatic_retry_authorized"):need(d[k]is False,"decision_forbids:"+k)
    mac=pins.json(r["mac_localization_ref"]);e=pins.json(r["mac_execution_ref"]);review=pins.json(r["mac_review_ref"])
    need(mac["status"]=="PASS_MAC_FIRST_FAILURE_LOCALIZED"and mac["historical_attempt_first_identity"]is None and mac["historical_attempt_first_identity_status"]=="UNKNOWN_NOT_RECORDED_IN_OLD_STDERR","Mac_boundary")
    need(type(e["actual_inner_exit"])is int and e["actual_inner_exit"]==0 and type(e["actual_external_exit"])is int and e["actual_external_exit"]==0 and e["termination_reason"]is None and e["elapsed_seconds"]<=180,"Mac_strict_0_0")
    need(review["schema"]=="saved_jacobian_Mac_localization_independent_result_review_v1"and review["status"]=="PASS_MAC_LOCALIZATION_REVIEW"and review["independent"]is True and review["mac_execution_sha256"]==r["mac_execution_ref"]["sha256"]and review["mac_localization_sha256"]==r["mac_localization_ref"]["sha256"]and review["selected_shard_sha256"]==r["shard"]["sha256"]and review["selection_sha256"]==digest(r["selection"]),"Mac_review_binding")
    for k in("selection","identity_join","trace_sha256_ordered_sha256","selected_position_trace_sha256"):need(r[k]==mac[k],"same_Mac_field:"+k)
    for k in("relative","sha256","bytes","remote_path"):need(r["shard"][k]==mac["shard"][k],"same_ref:"+k)
    need(A.get("decision_sha256")==DECISION_SHA and A.get("mac_localization_sha256")==r["mac_localization_ref"]["sha256"]and A.get("mac_review_sha256")==r["mac_review_ref"]["sha256"],"A_provenance")
    r["_Mac_elapsed"]=e["elapsed_seconds"];r["_preflight"]=pre
    return r,A,pins,out

def locate(r,pins):
    index=pins.ref(r["source_index_ref"]);rows=[json.loads(x)for x in index.read_text().splitlines()]
    need(len(rows)==1652,"index_population");s=r["selection"]
    matches=[(i,x)for i,x in enumerate(rows)if(x["seed"],x["cohort"],x["population"],x["pool"],identity(x))==(s["seed"],s["cohort"],s["population"],s["pool"],identity(s))]
    need(len(matches)==1,"unique_ref_identity");i,entry=matches[0];rel=Path(entry["tensor_file"])
    need(not rel.is_absolute()and".."not in rel.parts and str(rel)==r["shard"]["relative"],"relative_shard")
    path=safe(index.parent/rel);need(path==safe(r["shard"]["remote_path"]),"same_physical_ref")
    manifest=pins.json(r["source_manifest_ref"]);need(safe(manifest["remote_stage"])==index.parent.parent,"same_original_stage")
    ref=manifest["files"]["worker/"+str(rel)]
    need(ref["sha256"]==entry["tensor_sha256"]==r["shard"]["sha256"]and ref["bytes"]==r["shard"]["bytes"],"sealed_ref")
    pins.ref(dict(path=str(path),sha256=ref["sha256"],bytes=ref["bytes"]))
    query=pins.json(r["query_manifest_ref"])
    selected=[x for x in query["records"]if(x["seed"],x["cohort"],identity(x))==(s["seed"],s["cohort"],identity(s))]
    need(len(query["records"])==826 and len(selected)==1 and i+1==r["mac_index_one_based"],"same_presampled_ref")
    return path,entry,selected[0]

def tensor_identity(v):
    v=v.detach().cpu().contiguous()
    return dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest())
def check_tensor(v,shape,label,integer=False):
    kinds=(torch.int32,torch.int64)if integer else(torch.float32,torch.float64)
    need(isinstance(v,torch.Tensor)and v.device.type=="cpu"and tuple(v.shape)==shape and v.dtype in kinds and bool(torch.isfinite(v).all()),"saved_tensor:"+label)
def identity_join(row,entry,selected):
    need(identity(row)==identity(entry)==identity(selected)and all(row[k]==entry[k]for k in("seed","cohort","population","pool")),"row_index_query_join")
    ids=row["candidate_indices"];check_tensor(ids,(512,),"natural_ID",True)
    need(len(ids.unique())==512 and bool(((ids>=0)&(ids<114982)).all()),"natural_ID_values")
    fixed=selected.get("fixed400_member",False);external=selected["external_heldout"];internal=selected["internal_group"]
    need(type(fixed)is bool and row["fixed400_member"]is fixed,"fixed_join")
    fit=row["cohort"]=="train"and not external and not internal
    need(row["actual_train_query"]is fit and row["query_external_heldout"]is external and row["query_internal_group"]is internal,"eligibility_join")
    target=row["target_position"];need(type(target)is int and -1<=target<512 and row["target_natural"]is(target>=0),"target_position")
    cid=row.get("target_candidate_id");group=identity(row)[-1]
    if target>=0:need(cid==int(ids[target])and row["candidate_groups"][target]==group,"natural_positive_ID_group")
    true_id=selected.get("target_candidate_index")
    if true_id is not None:need(cid==true_id,"fixed_positive_ID")
    append=bool(row.get("loss_roles"));lt=row["loss_target_position"];lossids=row.get("loss_candidate_indices")
    need(row["diagnostic_loss_only_positive_used"]is append,"loss_shape_flag")
    if append:
        need(fit and fixed and row["pool"]=="single"and target==-1 and type(lt)is int and lt==512 and row["cache_loss_only_positive_present"]is True,"fit_single_append")
        check_tensor(lossids,(513,),"loss_ID",True)
        need(lossids.dtype==ids.dtype and torch.equal(lossids[:512],ids)and len(lossids.unique())==513 and bool(((lossids>=0)&(lossids<114982)).all())and int(lossids[512])==cid==true_id and row["loss_candidate_groups"][512]==group,"loss_ID_prefix")
        need(row["loss_coarse_cost"].dtype==row["coarse_cost"].dtype and torch.equal(row["loss_coarse_cost"][:512],row["coarse_cost"])and row["loss_candidate_groups"][:512]==row["candidate_groups"]and row["loss_candidate_smiles"][:512]==row["candidate_smiles"],"physical_prefix")
    else:need(row.get("loss_roles")is None and lossids is None and ((fit and type(lt)is int and lt==target>=0)or(not fit and lt is None)),"nullable_loss_positive")
    if fixed:need(row["original_bound_obstructed"]is selected["original_bound_obstructed"]and row.get("original_other_candidate_index")==selected.get("original_other_candidate_index"),"old87_other_join")
    return dict(status="PASS_EXACT_IDENTITY_JOIN",fixed400_member=fixed,original87=bool(fixed and row["original_bound_obstructed"]),target_candidate_id=cid,natural_target_position=target,loss_target_position=lt,original_other_candidate_index=row.get("original_other_candidate_index"),natural_candidate_id_tensor=tensor_identity(ids),loss_candidate_id_tensor=tensor_identity(lossids)if append else None,actual_train_query=fit,query_external_heldout=external,query_internal_group=internal,target_in_natural512=target>=0,loss_only_positive_used=append)
def trace_hashes(tr):
    arrays=dict(raw_X=tr["raw_X"].numpy(),raw_O=tr["raw_O"].numpy(),centered_X=tr["input36"][:,:33].numpy(),centered_O=tr["input36"][:,33:].numpy(),logits=tr["logit"].numpy(),delta=tr["delta"].numpy())
    output=[]
    for i in range(len(tr["logit"])):
        h=hashlib.sha256()
        for k,a in sorted(arrays.items()):
            v=np.asarray(a[i]);h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
        output.append(h.hexdigest())
    return output
def selected_trace(row,s):
    cap=s["cap"];role=s["role"];kind=s["shape"]
    need(type(cap)is int and cap in(5,10)and role in ROLES and kind in("natural512","full513_loss_only"),"selected_head")
    n=512 if kind=="natural512" else 513
    tr=capitem(row["roles"]if n==512 else row["loss_roles"],cap)[role]
    need(tr["shape_kind"]==("natural512_inference"if n==512 else"full513_loss_only"),"shape_kind")
    for k,sz in(("raw_X",(n,33)),("raw_O",(n,3)),("input36",(n,36)),("logit",(n,)),("delta",(n,)),("scale",(33,)),("fixed_x_center",(33,)),("fixed_o_center",(3,))):check_tensor(tr[k],sz,k)
    need(bool((tr["scale"]>0).all())and bool((tr["delta"].abs()<=cap).all()),"scale_cap")
    j=tr["input_jacobians"];positions=j["positions"]
    need(type(positions)is list and positions==sorted(set(positions))and positions==s["positions"]and positions and all(type(i)is int and 0<=i<n for i in positions),"positions")
    for k in("dlogit_dinput36","ddelta_dinput36"):check_tensor(j[k],(len(positions),36),k)
    if n==512:
        need(row["gamma"]in(.25,.75)and tr["inference_role_weight"]==((1-row["gamma"])if role=="no_aug"else row["gamma"]),"role_weight")
        check_tensor(j["weighted_dcost_dinput36"],(len(positions),36),"weighted")
    else:
        need(tr["inference_role_weight"]is None and j["weighted_dcost_dinput36"]is None,"loss_not_inference")
        natural=capitem(row["roles"],cap)[role]
        need(tr["raw_X"].dtype==natural["raw_X"].dtype and torch.equal(tr["raw_X"][:512],natural["raw_X"])and all(tr[k].dtype==natural[k].dtype and torch.equal(tr[k],natural[k])for k in("scale","fixed_x_center","fixed_o_center")),"same_model_cache_dtype")
    hashes=trace_hashes(tr);need(hashes==tr["trace_sha256_natural512"if n==512 else"trace_sha256_loss513"],"trace_SHA")
    return tr,hashes
def compare(actual,expected):
    a=actual.detach().cpu();b=expected.detach().cpu()
    need(tuple(a.shape)==tuple(b.shape)and a.numel()>0 and bool(torch.isfinite(a).all())and bool(torch.isfinite(b).all()),"finite_comparison")
    bits=a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    af=a.to(torch.float64);bf=b.to(torch.float64);diff=(af-bf).abs();bound=ATOL+RTOL*bf.abs();bad=diff>bound
    maximum=[int(x)for x in np.unravel_index(int(diff.argmax()),tuple(diff.shape))]
    coords=[maximum]+[c for c in bad.nonzero().tolist()[:16]if c!=maximum];w=[]
    for c in coords:
        at=tuple(c);w.append(dict(coordinate=c,actual=float(af[at]),expected=float(bf[at]),absolute_residual=float(diff[at]),fixed_bound=float(bound[at])))
    nz=bf!=0
    relative=float((diff[nz]/bf.abs()[nz]).max())if bool(nz.any())else None
    return dict(bit_equal=bool(bits),passes_fixed_tolerance=bool(torch.allclose(af,bf,rtol=RTOL,atol=ATOL)),actual=tensor_identity(a),expected=tensor_identity(b),elements=a.numel(),mismatching_elements=int(bad.sum()),max_absolute_residual=float(diff.max()),max_residual_over_fixed_bound=float((diff/bound).max()),max_relative_nonzero_expected=relative,zero_expected_nonzero_actual=int(((bf==0)&(af!=0)).sum()),witnesses=w)
def native_path(tr,cap,device):
    j=tr["input_jacobians"];pos=j["positions"];z=tr["logit"][pos].to(device);g=j["dlogit_dinput36"].to(device)
    divided=z/cap;t=torch.tanh(divided);square=t.square();local=1-square;dd=g*local[:,None]
    output=dict(logit=z,division=divided,tanh=t,square=square,local=local,DD=dd)
    weight=tr["inference_role_weight"];scale0=tr["scale"][0].to(device)
    if weight is not None:
        output["DD_times_weight"]=dd*weight;output["weighted"]=output["DD_times_weight"]*scale0
        output["weighted_from_saved_DD"]=j["ddelta_dinput36"].to(device)*weight*scale0
    return output
def algebra(tr,cap):
    cpu=native_path(tr,cap,"cpu");gpu=native_path(tr,cap,"cuda:0");torch.cuda.synchronize(0)
    j=tr["input_jacobians"];saved=j["ddelta_dinput36"];pos=j["positions"]
    out=dict(cpu_vs_cuda={k:compare(cpu[k],gpu[k])for k in cpu},saved_DD_vs_CPU=compare(saved,cpu["DD"]),saved_DD_vs_CUDA=compare(saved,gpu["DD"]),same_input_tanh=dict(CPU_division_input=compare(torch.tanh(cpu["division"]),torch.tanh(cpu["division"].to("cuda:0"))),CUDA_division_input=compare(torch.tanh(gpu["division"].cpu()),torch.tanh(gpu["division"]))),saved_logit=tr["logit"][pos].tolist(),saved_delta=tr["delta"][pos].tolist(),native_CPU_local=cpu["local"].tolist(),native_CUDA_local=gpu["local"].cpu().tolist(),role_weight=tr["inference_role_weight"],scale0=float(tr["scale"][0]),saturation=dict(CPU_abs_tanh_equal_one=int((cpu["tanh"].abs()==1).sum()),CUDA_abs_tanh_equal_one=int((gpu["tanh"].abs()==1).sum()),CPU_local_zero=int((cpu["local"]==0).sum()),CUDA_local_zero=int((gpu["local"]==0).sum()),saved_abs_delta_over_cap_ge095=int((tr["delta"][pos].abs()/cap>=.95).sum())))
    w=j["weighted_dcost_dinput36"]
    out["saved_weighted"]=dict(from_CPU_native_chain=compare(w,cpu["weighted"]),from_CUDA_native_chain=compare(w,gpu["weighted"]),from_CPU_saved_DD=compare(w,cpu["weighted_from_saved_DD"]),from_CUDA_saved_DD=compare(w,gpu["weighted_from_saved_DD"]))if w is not None else None
    z=tr["logit"][pos].to(torch.float64);ideal=j["dlogit_dinput36"].to(torch.float64)*(1-torch.tanh(z/cap).square())[:,None]
    out["FP64_ideal_diagnostic_not_native_guard"]=compare(saved,ideal)
    out["saved_zero_dlogit_nonzero_DD_components"]=int(((j["dlogit_dinput36"]==0)&(saved!=0)).sum())
    loss=tr.get("actual_train_loss");out["saved_loss_local_comparison"]=compare(loss["tanh_local_derivative"][pos],gpu["local"])if loss is not None else None
    return out

def configure(r):
    global torch,np
    import torch
    import numpy as np
    torch.set_num_threads(4);torch.set_num_interop_threads(4)
    p=r["_preflight"]
    need(torch.cuda.is_available()and str(torch.__version__)==p["torch"]and torch.version.cuda==p["cuda_runtime"]and torch.cuda.get_device_name(0)==p["cuda0_device"]and platform.machine()=="x86_64","JJ_CUDA_runtime_no_fallback")
    need(Path(sys.executable).resolve()==Path(p["python"]).resolve()and torch.get_num_threads()==4,"JJ_python_threads")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(r["max_gpu_allocated_bytes"]/torch.cuda.get_device_properties(0).total_memory,0)
    torch.cuda.reset_peak_memory_stats(0)
    return dict(python=sys.executable,python_version=sys.version,platform=platform.platform(),torch=str(torch.__version__),cuda_runtime=torch.version.cuda,cuda0_device=torch.cuda.get_device_name(0),cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())

def run(r,A,pins,out,started):
    runtime=configure(r);path,entry,selected=locate(r,pins)
    row=torch.load(path,map_location="cpu",weights_only=True)
    join=identity_join(row,entry,selected);need(join==r["identity_join"],"same_Mac_identity_join")
    tr,hashes=selected_trace(row,r["selection"])
    need(digest(hashes)==r["trace_sha256_ordered_sha256"]and {str(i):hashes[i]for i in r["selection"]["positions"]}==r["selected_position_trace_sha256"],"same_Mac_trace_fingerprints")
    evidence=algebra(tr,r["selection"]["cap"]);torch.cuda.synchronize(0)
    elapsed=time.monotonic()-started;peak=int(torch.cuda.max_memory_allocated(0));rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    need(elapsed<=120 and elapsed+r["_Mac_elapsed"]<=300 and peak<=r["max_gpu_allocated_bytes"]and rss<=r["max_rss_bytes"],"JJ_resource_bound")
    pins.finish()
    result=dict(schema="saved_jacobian_JJ_algebra_result_v1",status="PASS_SAME_REF_ALGEBRA_PROBE_COMPLETED",execution_scope=SCOPE,
        request_sha256=A["request_sha256"],independent_admission=A,selection=r["selection"],identity_join=join,shard=r["shard"],
        trace_sha256_ordered_sha256=digest(hashes),selected_position_trace_sha256=r["selected_position_trace_sha256"],
        historical_Mac_first_identity_status="UNRECORDED_NOT_RETROACTIVELY_ESTABLISHED",
        current_Mac_localization_sha256=r["mac_localization_ref"]["sha256"],Mac_review_sha256=r["mac_review_ref"]["sha256"],
        original_NN_status="FAIL_SUPERVISION_PRESERVED",original_analysis_status="FAIL_GUARD_PRESERVED",original_RSS_gap_status="UNKNOWN_NOT_CERTIFIED",original_guard_released=False,
        runtime=runtime,evidence=evidence,fixed_rtol=RTOL,fixed_atol=ATOL,model_VJP_state_assertion_scope="SOURCE_BOUND_NOT_INDEPENDENTLY_REMEASURED",
        single_shard_decodes=1,model_loads=0,new_NN_calls=0,training=False,test_read=False,external_API_calls=0,new_samples=0,
        scientific_population_statistics=False,automatic_retry=False,fallback_used=False,
        Mac_supervised_elapsed_seconds=r["_Mac_elapsed"],JJ_pre_exit_elapsed_seconds=elapsed,pre_exit_active_sum_seconds=r["_Mac_elapsed"]+elapsed,
        final_campaign_total_status="REQUIRES_CLOSED_JJ_SUPERVISOR_RECEIPT; interstage review/deployment excluded from active budget",
        JJ_peak_GPU_allocator_bytes=peak,JJ_peak_RSS_bytes=rss,input_sha256=pins.checked,inputs_unchanged=True,
        interpretation="Same saved ref primitive evidence only; independent actual review decides cause. Historical first identity and model/VJP validity are not remeasured.")
    encoded=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n";need(len(encoded.encode())<=r["max_output_bytes"],"JJ_output_bound")
    (out/"JJ_algebra.json").write_text(encoded);pins.finish();(out/"EXIT_CODE").write_text("0\n")
    return result

def main():
    p=argparse.ArgumentParser()
    for k in("request","request-sha256","admission","admission-sha256","out"):p.add_argument("--"+k,required=True)
    p.add_argument("--execute",action="store_true");args=p.parse_args();started=time.monotonic()
    r,A,pins,out=admit(args);out.mkdir(parents=True,exist_ok=False)
    try:result=run(r,A,pins,out,started)
    except Exception as error:
        failure=dict(schema="saved_jacobian_JJ_algebra_failure_v1",status="FAIL",execution_scope=SCOPE,request_sha256=args.request_sha256,
            error=type(error).__name__+":"+str(error),elapsed_seconds=time.monotonic()-started,original_guard_released=False,
            new_NN_calls=0,model_loads=0,training=False,test_read=False,external_API_calls=0,automatic_retry=False,fallback_used=False,extra_refill=False)
        (out/"FAILURE.json").write_text(json.dumps(failure,ensure_ascii=False,indent=2)+"\n")
        raise
    print(json.dumps(dict(status=result["status"],new_NN_calls=0,original_guard_released=False)))

if __name__=="__main__":main()

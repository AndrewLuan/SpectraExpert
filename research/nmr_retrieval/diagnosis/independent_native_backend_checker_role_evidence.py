"""功能概述：保留8f失败源，仅修正legacy未知role证据与保存量已知role证据的比对时机。
compare_legacy_diag只投影4个当时无法重建的role字段；同key worker全树与字节证明通过后才完成延期验证。
unprobed、P0、pair-null和其余legacy字段保持严格；所有科学算法、容差和独审执行门不变。
本副本独立实现后端重构及exact transport，不导入作者聚合器或数学层；门禁绑定41e/79cb/实际0/0。
主要职责：closed/metadata 核退出与输入字节；diagnostic 自行重算 native 成本、EPS排名、
双角色box桶；loss_check核当前loss，main逐shard核资格、512/513与typed碰撞，并独立
预hash选参考及核完整重要候选的参考/碰撞计数；距离仅复算预定义身份hash子集。
summary_check核全量分母，budget核wall、RSS和N输出体积。
辅助接口：sha/read核文件与JSON指纹；qid/key/capped/canon规范身份；trace_hash/typed_hash/proof核字节；
MaxItem/retain实现有界最小hash集合；native_cost重放数值运算，summary_check核完整统计分母。
代码库关系：消费冻结协议、request、原固定manifest、获准20个预测以及已关闭worker和
analysis产物；不导入作者聚合器或数学层，不载入checkpoint，不调用任何神经网络。
输入输出与副作用：main 先核原0/1、闭包0/0、57fa独审和原规划授权及独立checker admission，再核analysis0/0；之后才导入数值库与解码科学数据；所有
文件须匹配已声明SHA。只写N内审阅receipt；0训练/模型forward/API。CPU库限4线程，
wall3600秒；NN数值验证按独立预声明盐每seed/cap/role/pool取最小256，完整参考选择、
raw排名/边界/loss/资格全量核验，receipt明确覆盖。历史实际曝光保持unknown。
"""
from pathlib import Path
import argparse
import collections
import hashlib
import heapq
import json
import math
import os
import struct
import sys
import time

for _name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="4"
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
FIELDS=("query_order","variant","source","record_id","smiles","group")
ROLES=("no_aug","aug")
NN_SALT="independent-nn-check-v1"
NN_LIMIT=256
_MAPPING={}
_BOOTSTRAP={}
_CONSUMED={}
_DEVICE=None
_WRITE_ROOT=None
_MAP_PIN=None


def need(v,reason):
    if not v:
        raise AssertionError(reason)


def raw_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024**2),b""):
            h.update(block)
    return h.hexdigest()


def exact_logical(path):
    raw=str(path);p=Path(raw)
    need(p.is_absolute() and str(p)==raw and ".." not in p.parts,"exact logical identity")
    need(not any(v in ("test","sealed") or v.startswith("sealed-") for v in p.parts) and
         p.name not in ("research_report.md","supplementary_report.md","cohorts.pt"),"forbidden input")
    return raw


def io_path(path):
    logical=exact_logical(path)
    row=_MAPPING.get(logical,_BOOTSTRAP.get(logical))
    need(row is not None,"unregistered IO:"+logical)
    valid_transport_row(row)
    p=Path(exact_logical(row["physical_path"]))
    need(p.is_file() and str(p.resolve(strict=True))==str(p),"physical alias or missing")
    need(p.stat().st_size==row["bytes"] and raw_sha(p)==row["sha256"],"exact bytes/SHA:"+logical)
    _CONSUMED[logical]=row.copy()
    return p


def valid_transport_row(row):
    need(type(row["bytes"]) is int and row["bytes"]>=0 and type(row["sha256"]) is str and
         len(row["sha256"])==64 and all(c in "0123456789abcdef" for c in row["sha256"]),"transport row types")
    need(type(row["physical_path"]) is str,"physical path string")
    exact_logical(row["physical_path"])


def sha(path):
    io_path(path)
    return _CONSUMED[str(path)]["sha256"]


def initialize_transport(args):
    """CLI bootstrap先验SHA；其余读取只走已签A列出的逐文件表。"""
    global _MAP_PIN
    for path,digest in ((args.checker_admission,args.checker_admission_sha256),
                        (args.transport_map,args.transport_map_sha256)):
        p=Path(exact_logical(path))
        need(p.is_file() and str(p.resolve(strict=True))==str(p) and raw_sha(p)==digest,"CLI bootstrap SHA/identity")
    admission=json.loads(Path(args.checker_admission).read_text())
    need(admission["transport_map_sha256"]==args.transport_map_sha256 and
         admission["backend_decision_sha256"]=="41e85949a7991ad3e7cf1824b5308d3891fab60cde85adf5a712cb52366dc4b8" and
         admission["backend_actual_review_sha256"]=="79cbdf70743f13c7499eec49c94ecde61216485c89c8375572faf80023396a30",
         "checker backend/map admission")
    mapping=json.loads(Path(args.transport_map).read_text())
    need(mapping["schema"]=="native_analysis_exact_transport_map_v1","checker exact map schema")
    _MAPPING.clear();_BOOTSTRAP.clear();_CONSUMED.clear()
    for row in mapping["entries"]:
        key=exact_logical(row["logical_path"])
        need(key not in _MAPPING and type(row["bytes"]) is int and row["bytes"]>=0,"unique mapped file")
        _MAPPING[key]={k:row[k] for k in ("physical_path","sha256","bytes")}
    for path,row in admission["bootstrap_files"].items():
        valid_transport_row(row)
        need(exact_logical(path)==row["physical_path"] and path not in _MAPPING,"exact bootstrap entry")
        _BOOTSTRAP[path]=row
    _BOOTSTRAP[args.checker_admission]=dict(physical_path=args.checker_admission,
        sha256=args.checker_admission_sha256,bytes=Path(args.checker_admission).stat().st_size)
    _BOOTSTRAP[args.transport_map]=dict(physical_path=args.transport_map,
        sha256=args.transport_map_sha256,bytes=Path(args.transport_map).stat().st_size)
    _MAP_PIN=(args.transport_map,args.transport_map_sha256)


def finish_transport():
    for logical in list(_CONSUMED):io_path(logical)
    need(sha(_MAP_PIN[0])==_MAP_PIN[1],"checker map changed")


def read(ref):
    need(sha(ref["path"])==ref["sha256"],"input SHA:"+ref["path"])
    return json.loads(io_path(ref["path"]).read_text())


def qid(row):
    q=row.get("query_identity",row)
    return tuple(q[k] for k in FIELDS)


def key(seed,cap,pool,row):
    return (seed,cap,pool,qid(row))


def capped(obj,cap):
    return obj[cap] if cap in obj else obj[str(cap)]


def canon(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(",",":"))


def intern(group):
    return int(hashlib.sha256(("ranker_internal_317"+str(group)).encode()).hexdigest()[:12],16)%5==0


def closed(stage):
    p=Path(stage)
    e=json.loads(io_path(p/"actual_execution.json").read_text())
    need(type(e["actual_inner_exit"]) is int and e["actual_inner_exit"]==0,"inner not success")
    need(type(e["actual_external_exit"]) is int and e["actual_external_exit"]==0,"outer not success")
    need(e["termination_reason"] is None,"terminated stage")
    need(io_path(p/"EXIT_CODE").read_text()=="0\n","stage EXIT_CODE")
    need(e["elapsed_seconds"]<=e["max_seconds"]<=3600,"stage wall")
    need(e["peak_observed_rss_bytes"]<=e["max_rss_bytes"]<=32*1024**3,"stage RSS")
    need(e["final_output_bytes_before_receipt"]<=e["max_artifact_bytes"]<=4*1024**3,"stage bytes")
    return e



def budget(started,full=False):
    import resource
    need(time.monotonic()-started<=3600,"review wall")
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes=int(peak) if sys.platform=="darwin" else int(peak)*1024
    need(peak_bytes<=32*1024**3,"review RSS")
    if full:
        total=sum(p.stat().st_size for p in Path(_WRITE_ROOT).rglob("*") if p.is_file() and not p.is_symlink())
        need(total<=4*1024**3,"global N bytes")

def dtype(name):
    need(name in ("torch.float32","torch.float64"),"native cost float32/64 required")
    return np.dtype(name.split(".")[-1])


def native_cost(base,d0,d1,scales,gamma,dtypes):
    t0,t1=map(dtype,dtypes["correction_dtypes"])
    a=np.multiply(np.asarray(d0,dtype=t0),np.asarray(scales[0],dtype=t0),dtype=t0)
    b=np.multiply(np.asarray(d1,dtype=t1),np.asarray(scales[1],dtype=t1),dtype=t1)
    a=np.multiply(a,1-gamma,dtype=t0); b=np.multiply(b,gamma,dtype=t1)
    first=np.result_type(dtype(dtypes["coarse_dtype"]),t0)
    return np.add(np.add(np.asarray(base,dtype=dtype(dtypes["coarse_dtype"])),a,dtype=first),
                  b,dtype=dtype(dtypes["final_cost_dtype"]))


def diagnostic(ids,coarse,cost,target,gamma,scales,cap,dtypes,old=None,outside=None,deltas=None):
    cost=np.asarray(cost,dtype=dtype(dtypes["final_cost_dtype"]))
    coarse=np.asarray(coarse,dtype=dtype(dtypes["coarse_dtype"]))
    need(len(ids)==len(set(ids))==len(cost)==len(coarse)==512,"natural512 unique")
    order=sorted(range(512),key=lambda i:(float(cost[i]),ids[i]))
    rb=[cap*scales[i] for i in (0,1)]
    bound=math.fsum((1-gamma if i==0 else gamma)*rb[i] for i in (0,1))
    base=dict(top10_candidate_ids=[ids[i] for i in order[:10]],role_absolute_bounds=rb,
              combined_absolute_bound=bound,pairwise_bound=2*bound,actual_exposure="unknown",
              target_in_pool=target>=0,required_fields_unknown=[])
    if target<0:
        need(outside is None or (type(outside) is int and outside>512),"outside rank")
        return dict(base,rank=outside,bucket="P0_target_absent",negative_counted=None,still_need_to_beat=None,
                    static_unbeatable_count=None,real_static_unbeatable_count=None,pairs=None)
    threshold=np.add(cost[target],np.asarray(1e-7,dtype=cost.dtype),dtype=cost.dtype)
    rank=int(np.count_nonzero(cost<=threshold))
    tmin=native_cost(coarse[target],-cap,-cap,scales,gamma,dtypes)
    maximum=native_cost(coarse,cap,cap,scales,gamma,dtypes)
    blocked=maximum<=np.add(tmin,np.asarray(1e-7,dtype=cost.dtype),dtype=cost.dtype)
    blocked[target]=False
    nblocked=int(np.count_nonzero(blocked))
    negative=[i for i in order if i!=target]
    realblocked=sum(float(coarse[j])+bound<=float(coarse[target])-bound+1e-7 for j in negative)
    pairs={}
    for name,j in (("current_negative1",negative[0]),("current_negative10",negative[9]),
                   ("immutable_old_other",ids.index(old) if old in ids and ids.index(old)!=target else None)):
        if j is None:
            pairs[name]=None;continue
        gap=float(coarse[target])-float(coarse[j]);margin=float(cost[target])-float(cost[j])
        pair=dict(candidate_index=ids[j],coarse_gap=gap,native_margin=margin,pairwise_bound=2*bound,
                  necessary_bound_released=gap<=2*bound,role_gains=None,combined_gain=None,real_margin=None,native_minus_real_margin=None)
        if deltas is not None:
            gains=[scales[r]*(float(deltas[r][j])-float(deltas[r][target])) for r in (0,1)]
            gain=math.fsum((1-gamma if r==0 else gamma)*gains[r] for r in (0,1))
            pair.update(role_gains=gains,combined_gain=gain,real_margin=gap-gain,native_minus_real_margin=margin-(gap-gain))
        pairs[name]=pair
    bucket="S0_actual_top10" if rank<=10 else "S1_fail_static_blocked" if nblocked>=10 else "S2_fail_static_open"
    return dict(base,rank=rank,bucket=bucket,negative_counted=rank-1,still_need_to_beat=max(0,rank-10),
                static_unbeatable_count=nblocked,static_unbeatable_candidate_ids=[ids[j] for j in negative if blocked[j]],
                real_static_unbeatable_count=realblocked,native_real_static_boundary_differs=nblocked!=realblocked,
                native_target_minimum=float(tmin),top10_threshold_equivalence=bool(cost[negative[9]]>threshold),pairs=pairs)


def equal_tree(actual,expected,label):
    if isinstance(expected,dict):
        need(isinstance(actual,dict) and set(actual)==set(expected),label+": keys")
        for k,v in expected.items():equal_tree(actual[k],v,label+"."+str(k))
    elif isinstance(expected,(list,tuple)):
        need(isinstance(actual,(list,tuple)) and len(actual)==len(expected),label+": sequence")
        for i,(a,e) in enumerate(zip(actual,expected)):equal_tree(a,e,label+"."+str(i))
    elif expected is None or type(expected) in (str,bool,int):
        need(actual==expected and (expected is None or type(actual) is type(expected)),label)
    else:
        need(actual is not None and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),label)


def compare_diag(actual,expected):
    # P0 does not publish target-only numeric fields; natural rows publish the full frozen contract.
    equal_tree(actual,expected,"diagnostic")


ROLE_FIELDS=("role_gains","combined_gain","real_margin","native_minus_real_margin")


def compare_legacy_diag(actual, expected, has_saved_role_evidence):
    """只延后同身份worker已有保存role的四字段，其余树和空值仍完整核验。"""
    need(type(has_saved_role_evidence) is bool,"exact saved-role evidence flag")
    if not has_saved_role_evidence or expected["pairs"] is None:
        compare_diag(actual,expected)
        return
    need(isinstance(actual,dict) and set(actual)==set(expected),"legacy diagnostic keys")
    need(isinstance(actual["pairs"],dict) and set(actual["pairs"])==set(expected["pairs"]),"legacy pair keys")
    left,right=dict(actual),dict(expected)
    left["pairs"],right["pairs"]={},{}
    for name,pair in expected["pairs"].items():
        got=actual["pairs"][name]
        if pair is None:
            equal_tree(got,None,"legacy pair null:"+name)
            left["pairs"][name]=right["pairs"][name]=None
            continue
        need(isinstance(got,dict) and set(got)==set(pair),"legacy pair fields:"+name)
        need(all(pair[field] is None for field in ROLE_FIELDS),"legacy role fields truly unavailable")
        left["pairs"][name]={k:v for k,v in got.items() if k not in ROLE_FIELDS}
        right["pairs"][name]={k:v for k,v in pair.items() if k not in ROLE_FIELDS}
    compare_diag(left,right)


def complete_legacy_role_join(pending, completed, proofs):
    need(set(proofs)==pending==completed,"complete same-key legacy-role verification")


def trace_hash(trace):
    arrays=dict(raw_X=trace["raw_X"].numpy(),raw_O=trace["raw_O"].numpy(),
                centered_X=trace["input36"][:,:33].numpy(),centered_O=trace["input36"][:,33:].numpy(),
                logits=trace["logit"].numpy(),delta=trace["delta"].numpy())
    result=[]
    for i in range(len(trace["raw_X"])):
        h=hashlib.sha256()
        for name,array in sorted(arrays.items()):
            v=np.asarray(array[i]);h.update(name.encode());h.update(str(v.dtype).encode())
            h.update(str(v.shape).encode());h.update(v.tobytes())
        result.append(h.hexdigest())
    return result


def typed_hash(values,name):
    h=hashlib.sha256(json.dumps([name,len(values)],separators=(",",":")).encode())
    fmt="<f" if name=="torch.float32" else "<d"
    for v in values:
        h.update(struct.pack(fmt,float(v)))
    return h.hexdigest()


def proof(ids,coarse,cost,dtypes,traces):
    return hashlib.sha256(canon([ids,list(map(float,coarse)),list(map(float,cost)),dtypes,traces]).encode()).hexdigest()


def native_close(saved, expected, label, rtol=2e-5):
    need(saved.dtype==expected.dtype and tuple(saved.shape)==tuple(expected.shape),"native dtype/shape:"+label)
    need(torch.allclose(saved.detach().cpu().to(torch.float64),expected.detach().cpu().to(torch.float64),
                        rtol=rtol,atol=2e-6),"native algebra:"+label)


def loss_check(trace,coarse,target,excluded):
    obj=trace["actual_train_loss"]
    for field in ("actual_normalized_costs","probability","listwise_gradient_delta","anchor_gradient_delta","total_gradient_delta","total_gradient_logit","tanh_local_derivative","listwise_gradient_logit","anchor_gradient_logit"):
        need(isinstance(obj[field],torch.Tensor) and obj[field].dtype in (torch.float32,torch.float64) and tuple(obj[field].shape)==(len(coarse),),"loss vector dtype/shape:"+field)
    need(type(target) is int and 0<=target<len(coarse),"loss target position")
    need(obj["allowed"].dtype==torch.bool and tuple(obj["allowed"].shape)==(len(coarse),),"loss bool support shape")
    allowed=~np.asarray(excluded,dtype=bool);allowed[target]=True
    need(np.array_equal(obj["allowed"].numpy(),allowed),"actual loss support")
    need(_DEVICE is not None,"admitted native backend")
    delta=trace["delta"].to(_DEVICE);logit=trace["logit"].to(_DEVICE);scale=trace["scale"].to(_DEVICE)[0]
    mask=torch.as_tensor(allowed,device=_DEVICE)
    native=torch.as_tensor(coarse,device=_DEVICE)/scale+delta
    need(native.dtype==obj["actual_normalized_costs"].dtype and
         torch.equal(native.cpu(),obj["actual_normalized_costs"]),"loss native normalized cost")
    p=torch.zeros_like(native);p[mask]=torch.softmax(-native[mask]/.5,dim=0)
    target_indicator=torch.zeros_like(p);target_indicator[target]=1
    gl=torch.where(mask,(target_indicator-p)/.5,torch.zeros_like(p))
    ga=torch.where(mask,2*.1*delta/int(mask.sum()),torch.zeros_like(delta))
    derivative=1-torch.tanh(logit/CAP).square()
    for field,want in (("probability",p),("listwise_gradient_delta",gl),("anchor_gradient_delta",ga),
                       ("total_gradient_delta",gl+ga),("total_gradient_logit",(gl+ga)*derivative)):
        native_close(obj[field],want,"analytical loss:"+field)
        need(np.all(obj[field].numpy()[~allowed]==0),"masked zero:"+field)
    need(type(obj["support_count"]) is int and obj["support_count"]==int(allowed.sum())>=2,"loss support count")
    need(obj["normalized_cost_dtype"]==str(obj["actual_normalized_costs"].dtype) and obj["probability_dtype"]==str(obj["probability"].dtype),"loss dtype aliases")
    for field,want in (("tanh_local_derivative",derivative),("listwise_gradient_logit",gl*derivative),
                       ("anchor_gradient_logit",ga*derivative)):
        native_close(obj[field],want,"loss component:"+field)
        if field!="tanh_local_derivative":need(np.all(obj[field].numpy()[~allowed]==0),"masked component exact zero")
    # Scalars retain the worker's native dtype and operation order; FP64 ideal logZ can differ by cancellation.
    # 原JJ后端完成原dtype运算后再比较；不以FP64理想式替换原worker scalar。
    saved_cost=obj["actual_normalized_costs"].to(_DEVICE)
    expected_list=torch.logsumexp(-saved_cost[mask]/.5,0)+saved_cost[target]/.5
    expected_anchor=.1*delta[mask].square().mean()
    for field,want in (("listwise_loss",expected_list),("anchor_loss",expected_anchor),("total_loss",expected_list+expected_anchor)):
        need(math.isclose(float(obj[field]),float(want),rel_tol=2e-5,abs_tol=2e-6),"native saved loss scalar:"+field)
    gl_cpu=gl.detach().cpu().numpy()
    need(gl_cpu[target]>=-2e-6 and np.all(np.delete(gl_cpu,target)<=2e-6),"listwise gradient sign")
    return allowed


class MaxItem:
    def __init__(self,k,v):self.k,self.v=k,v
    def __lt__(self,other):return self.k>other.k


def retain(heaps,group,order,value,limit):
    heap=heaps.setdefault(group,[])
    if len(heap)<limit:heapq.heappush(heap,MaxItem(order,value))
    elif order<heap[0].k:heapq.heapreplace(heap,MaxItem(order,value))



def failed_source(stage):
    p=Path(stage);e=json.loads(io_path(p/"actual_execution.json").read_text())
    need(type(e["actual_inner_exit"]) is int and e["actual_inner_exit"]==0,"source inner0 required")
    need(type(e["actual_external_exit"]) is int and e["actual_external_exit"]==1,"source external1 preserved")
    need(e["termination_reason"]=="rss_monitor_unavailable" and e.get("rss_missing_while_alive") is True,"source RSS failure preserved")
    need(io_path(p/"EXIT_CODE").read_text()=="0\n","source inner exit file")
    return e


def admit_checker(args):
    admission=read(dict(path=args.checker_admission,sha256=args.checker_admission_sha256))
    need(admission.get("status")=="PASS" and admission.get("independent") is True and
         admission.get("execution_authorized") is True and
         admission.get("execution_scope")=="INDEPENDENT_CLOSED_OUTPUT_RESULT_CHECK_ONLY","independent checker admission")
    for field,wanted in (("checker_sha256",sha(__file__)),("request_sha256",args.request_sha256),
                         ("protocol_sha256",args.protocol_sha256),
                         ("decision_sha256",args.original_decision_sha256),
                         ("closure_review_sha256",args.closure_review_sha256),("analysis_manifest_sha256",args.analysis_manifest_sha256)):
        need(admission.get(field)==wanted,"checker admission binding:"+field)
    review_ref=admission["role_evidence_failure_review"]
    failed_ref=admission["prior_failed_checker_manifest"]
    need(review_ref["sha256"]=="83c3a2bd9f5b89d0597153c84745a44dc2a8ed206030c12dd2d33811dce81b5f" and
         failed_ref["sha256"]=="6b97fdeedac6b867b06b166feadf66edc5305b5a527d50a573e9e5608a40f199",
         "sole reviewed role-evidence correction")
    review,failed=read(review_ref),read(failed_ref)
    need(review["status"]=="PASS_IMPLEMENTATION_FAILURE_SOURCE_REVIEW" and review["independent"] is True and
         review["minimal_evidence_join_correction_eligible"] is True and
         review["threshold_or_scientific_definition_change"] is False,"reviewed minimal evidence join")
    need(type(failed["actual"]["actual_inner_exit"]) is int and failed["actual"]["actual_inner_exit"]==1 and
         type(failed["actual"]["actual_external_exit"]) is int and failed["actual"]["actual_external_exit"]==1 and
         failed["checker_source_sha256"]=="8f3340cecccd668ae6917d2f1a7fa3f1a586b36188da07b6e9eebfb723a329ec",
         "prior failed checker preserved")
    return admission



def analysis_chain(args):
    package=read(dict(path=args.analysis_manifest,sha256=args.analysis_manifest_sha256))
    root=Path(args.analysis_stage).resolve()
    need(package.get("schema")=="closed_analysis_result_manifest_v1" and
         Path(package["stage_root"]).resolve()==root,"closed analysis manifest root")
    need(not Path(args.analysis_manifest).resolve().is_relative_to(root),"analysis manifest separate")
    files=package["files"]
    current={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    need(current==set(files) and not any(p.is_symlink() for p in root.rglob("*")),"complete analysis result files")
    for relative,ref in files.items():
        rel=Path(relative);path=root/rel
        need(not rel.is_absolute() and ".." not in rel.parts and str(rel)==relative and
             path.resolve().is_relative_to(root) and type(ref["bytes"]) is int and
             path.stat().st_size==ref["bytes"] and sha(path)==ref["sha256"],"analysis result byte pin")
    e=closed(root)
    refs=package["refs"]
    need(set(refs)=={"analysis_job","input_admission","stage_admission"},"analysis gate refs")
    job=read(refs["analysis_job"]);A=read(refs["input_admission"]);B=read(refs["stage_admission"])
    scope="CLOSED_OUTPUT_SAVED_TENSOR_SCIENTIFIC_ANALYSIS_ONLY"
    for proof in (A,B):
        need(proof.get("status")=="PASS" and proof.get("independent") is True and
             proof.get("execution_authorized") is True and proof.get("execution_scope")==scope,"analysis A/B scope")
    need(e["job_sha256"]==refs["analysis_job"]["sha256"]==B["pins"]["job"] and
         e["review_sha256"]==refs["stage_admission"]["sha256"],"analysis actual B binding")
    need(A["request_sha256"]==args.request_sha256,"analysis A request")
    need(Path(job["stage_dir"]).resolve()==root,"analysis job exact stage")
    argv=job["argv"]
    def argument(name):return argv[argv.index(name)+1]
    need(Path(argument("--request")).resolve()==Path(args.request).resolve() and
         argument("--request-sha256")==args.request_sha256 and
         Path(argument("--admission")).resolve()==Path(refs["input_admission"]["path"]).resolve() and
         argument("--admission-sha256")==refs["input_admission"]["sha256"] and
         Path(argument("--out")).resolve()==root/"results","analysis exact argv")
    scripts=[p for p in argv[1:] if str(p).endswith(".py")]
    need(len(scripts)==1,"analysis script identity")
    need(scripts[0] in A["reviewed_source_sha256"] and
         job["source_sha256"].get(scripts[0])==A["reviewed_source_sha256"][scripts[0]],"analysis author source binding")
    for p,h in A["reviewed_source_sha256"].items():
        need(sha(p)==h and job["source_sha256"].get(p)==h,"analysis reviewed source byte pin")
    result=json.loads(io_path(root/"results/summary.json").read_text())
    need(result.get("status")=="PASS" and result.get("independent_admission")==A and
         result["bootstrap_sha256"].get(str(Path(args.request).resolve()))==args.request_sha256 and
         result.get("inputs_unchanged") is True,"author result A/request")
    request=read(dict(path=args.request,sha256=args.request_sha256))
    need(A["reviewed_source_sha256"]==request["analysis_source_sha256"] and
         A["transport_map_sha256"]==request["transport_map"]["sha256"] and
         argument("--transport-map")==request["transport_map"]["path"] and
         argument("--transport-map-sha256")==request["transport_map"]["sha256"],"analysis exact source/map gate")
    need(result["transport_map_sha256"]==request["transport_map"]["sha256"] and
         result["algebra_device"]=="cuda:0" and result["runtime"]=="original_JJ" and
         result["source_git_commit"]==request["source_git_commit"],"author native provenance")
    author_map=read(request["transport_map"])
    author_rows={r["logical_path"]:dict(path=r["physical_path"],sha256=r["sha256"],bytes=r["bytes"])
                 for r in author_map["entries"]}
    direct=[request[k] for k in ("fixed_manifest","query_manifest","qualification_contract","qualification_bindings",
                                "backend_decision","backend_actual_review")]
    direct+=request["legacy_predictions"]
    direct+=[ref for entry in request["worker_outputs"] for ref in entry.values()]
    direct+=[ref for ref in request["completed_output_provenance"].values() if type(ref) is dict]
    need(len(direct)==35,"author35 direct read refs")
    expected_reads={ref["path"]:ref["sha256"] for ref in direct}
    sealed=read(request["completed_output_provenance"]["sealed_manifest"])
    logical_stage=Path(exact_logical(request["completed_output_provenance"]["source_stage_root"]))
    shards={str(logical_stage/relative):ref["sha256"] for relative,ref in sealed["files"].items()
            if relative.startswith("worker/tensor_shards/")}
    need(len(shards)==1652,"author1652 exact saved reads");expected_reads.update(shards)
    need(result["input_sha256"]==expected_reads and
         all(author_rows.get(k)==v for k,v in result["physical_input_transport"].items()) and
         set(result["physical_input_transport"])==set(expected_reads),"author complete exact transport provenance")
    expected_bootstrap={args.request:args.request_sha256,refs["input_admission"]["path"]:refs["input_admission"]["sha256"],
                        **request["analysis_source_sha256"]}
    need(len(expected_bootstrap)==5 and result["bootstrap_sha256"]==expected_bootstrap,"author exact bootstrap ledger")
    for field in ("model_calls","new_spectrum_forward_calls","forward_bank_calls","optimizer_steps","weight_updates","external_api_calls"):
        need(result[field]==0,"author analysis forbidden:"+field)
    need(result["test_read"] is False,"analysis no test")
    return package


def recheck_analysis_files(package):
    root=Path(package["stage_root"])
    need({str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}==set(package["files"]) and not any(p.is_symlink() for p in root.rglob("*")),"analysis files changed")
    for ref in package["refs"].values():need(sha(ref["path"])==ref["sha256"],"analysis gate ref changed")
    for relative,ref in package["files"].items():
        path=root/relative
        need(path.stat().st_size==ref["bytes"] and sha(path)==ref["sha256"],"analysis bytes changed")


def closed_output_provenance(args,request,protocol):
    decision=read(dict(path=args.original_decision,sha256=args.original_decision_sha256))
    need(decision.get("schema")=="original_planning_scientific_decoding_decision_v1" and
         decision.get("status")=="AUTHORIZED_CLOSED_SAVED_TENSOR_SCIENTIFIC_ANALYSIS","original decision required")
    need(decision.get("decision_owner")=="original_planning_session" and
         decision.get("saved_tensor_scientific_decode_authorized") is True and
         decision.get("saved_tensor_algebra_statistics_authorized") is True,"saved-output authorization")
    for field in ("new_model_forward_authorized","training_authorized","test_read_authorized",
                  "external_API_authorized","additional_sample_authorized","new_definition_authorized","formal_system_changed"):
        need(decision.get(field) is False,"decision forbidden:"+field)
    need(type(decision.get("new_NN_budget")) is int and decision["new_NN_budget"]==0 and
         decision.get("query_seed_pairs")==826 and decision.get("pool_count")==2,"frozen zeroNN decision")
    need(decision.get("source_NN_stage_status")=="FAIL_SUPERVISION" and
         decision.get("source_actual_inner_exit")==0 and decision.get("source_actual_external_exit")==1 and
         decision.get("source_RSS_gap_status")=="UNKNOWN_NOT_CERTIFIED" and
         decision.get("model_state_assertion_scope")=="SOURCE_BOUND_NOT_INDEPENDENTLY_REMEASURED","decision failure boundary")
    need(sha(decision["decision_text_path"])==decision["decision_text_sha256"],"original decision text SHA")
    review=read(dict(path=args.closure_review,sha256=args.closure_review_sha256))
    need(decision["closure_independent_review_sha256"]==args.closure_review_sha256 and
         review.get("status")=="PASS_RAW_CLOSURE_REVIEW" and review.get("independent") is True,"closure independent review")
    need(review.get("closure_actual_inner_exit")==0 and review.get("closure_actual_external_exit")==0 and
         review.get("source_stage_actual_inner_exit")==0 and review.get("source_stage_actual_external_exit")==1 and
         review.get("source_stage_status")=="FAIL_SUPERVISION_PRESERVED" and
         review.get("original_resource_envelope_recertified") is False and
         review.get("original_RSS_gap_status")=="UNKNOWN_NOT_CERTIFIED","closure review failure boundary")
    source=failed_source(args.replay_stage)
    closure_execution=closed(args.closure_stage)
    analysis_package=analysis_chain(args)
    need(source["job_sha256"]==sha(args.replay_job) and
         source["review_sha256"]==sha(args.replay_admission),"original source job/admission")
    closure_admission=json.loads(io_path(args.closure_stage_admission).read_text())
    need(closure_admission.get("status")=="PASS" and closure_admission.get("independent") is True and
         closure_admission.get("execution_authorized") is True and
         closure_admission.get("execution_scope")=="FAILED_SUPERVISION_OUTPUT_CLOSURE_ONLY" and
         closure_execution["job_sha256"]==sha(args.closure_job)==closure_admission["pins"]["job"] and
         closure_execution["review_sha256"]==sha(args.closure_stage_admission),"closure stage exact admission")
    raw_path=Path(args.closure_stage)/"worker/closure.json"
    for path in (Path(args.closure_stage)/"actual_execution.json",raw_path):
        need(review["evidence_sha256"].get(exact_logical(path))==sha(path),"reviewed closure artifact SHA")
    raw=json.loads(io_path(raw_path).read_text())
    need(raw.get("status")=="PASS_RAW_OUTPUT_CLOSURE" and
         raw.get("execution_scope")=="FAILED_SUPERVISION_OUTPUT_CLOSURE_ONLY" and
         raw.get("original_actual_inner_exit")==0 and raw.get("original_actual_external_exit")==1 and
         raw.get("original_stage_status")=="FAIL_SUPERVISION" and
         raw.get("original_resource_envelope_recertified") is False and
         raw.get("original_rss_gap_status")=="unknown_not_certified","raw closure preserved failure")
    need(raw["closed_output_files"]==1661 and raw["raw_shards_hashed"]==1652 and
         raw["checked_input_files"]==138 and raw["input_raw_hash_passes_min"]>=2 and
         raw["output_raw_hash_passes_min"]>=2,"complete raw closure coverage")
    need(raw["index_counts"]==dict(rows=1652,query_seed_pairs=826,natural=6608,loss513=1600,total=8208),"closure count aliases")
    for field in ("tensor_decode_loads","model_loads","NN_calls","training_steps","external_API_calls"):
        need(raw[field]==0,"closure zeroNN:"+field)
    need(raw["source_budget"]==dict(prior_consumed=2,completed_stage=8208,cumulative_consumed=8210,
                                   global_ceiling=8640,remaining=430,new_role_row_forwards=0),"cumulative NN budget")
    for key,wanted in (("protocol",args.protocol_sha256),("job",sha(args.replay_job)),
                       ("replay_admission",sha(args.replay_admission)),
                       ("query_manifest",request["query_manifest"]["sha256"])):
        need(raw["source_refs"][key]["sha256"]==wanted,"closure original ref:"+key)
    need(raw["source_refs"]["protocol"]["sha256"]==PROTOCOL_SHA and
         raw["source_refs"]["query_manifest"]["sha256"]==protocol["query_manifest_sha256"],"protocol/query closure")
    return dict(source_stage_status="FAIL_SUPERVISION",source_actual_inner_exit=0,source_actual_external_exit=1,
                original_RSS_gap_status="UNKNOWN_NOT_CERTIFIED",original_resource_envelope_recertified=False,
                closure_actual_inner_exit=0,closure_actual_external_exit=0,
                closure_review_sha256=args.closure_review_sha256,decision_sha256=args.original_decision_sha256,
                cumulative_NN_calls=8210,new_NN_calls=0,analysis_package=analysis_package)


def cap_keys(values):
    need(sorted(map(int,values))==[5,10] and len(values)==2,"exact caps5and10")


def finite_payload(value):
    if isinstance(value,torch.Tensor):
        need(value.device.type=="cpu" and not value.requires_grad,"saved tensor CPU detached")
        need(bool(torch.isfinite(value).all()),"nonfinite saved tensor")
    elif isinstance(value,float):
        need(math.isfinite(value),"nonfinite saved scalar")
    elif isinstance(value,dict):
        for item in value.values():finite_payload(item)
    elif isinstance(value,(list,tuple)):
        for item in value:finite_payload(item)


def trace_algebra(trace,n,cap,weight):
    for field,shape in (("raw_X",(n,33)),("raw_O",(n,3)),("input36",(n,36)),
                        ("logit",(n,)),("delta",(n,)),("scale",(33,)),
                        ("fixed_x_center",(33,)),("fixed_o_center",(3,))):
        tensor=trace[field]
        need(isinstance(tensor,torch.Tensor) and tuple(tensor.shape)==shape and
             tensor.dtype in (torch.float32,torch.float64),"trace dtype/shape:"+field)
    need(bool((trace["scale"]>0).all()),"positive scale")
    need(bool((trace["delta"].abs()<=cap).all()),"cap absolute bound")
    need(_DEVICE is not None,"admitted native backend")
    tr={name:trace[name].to(_DEVICE) for name in ("logit","raw_X","raw_O","fixed_x_center","fixed_o_center","scale")}
    expected=cap*torch.tanh(tr["logit"]/cap)
    native_close(trace["delta"],expected,"saved cap-tanh algebra",rtol=2e-6)
    centered=torch.cat((((tr["raw_X"]-tr["fixed_x_center"])/tr["scale"]).clamp(-20,20),
                        (tr["raw_O"]-tr["fixed_o_center"]).clamp(-20,20)),1)
    native_close(trace["input36"],centered,"saved centered36 algebra",rtol=2e-6)
    jac=trace["input_jacobians"];positions=jac["positions"]
    need(all(type(i) is int and 0<=i<n for i in positions) and positions==sorted(set(positions)),"Jacobian positions")
    for field in ("dlogit_dinput36","ddelta_dinput36"):
        need(isinstance(jac[field],torch.Tensor) and jac[field].dtype in (torch.float32,torch.float64) and tuple(jac[field].shape)==(len(positions),36),"Jacobian dtype/shape")
    # Preserve selected_jacobians' saved dtype and operation order. Promoting tanh to FP64
    # can create a nonzero derivative where the original FP32 primitive has saturated.
    # 原JJ cuda:0原dtype重构，保持source-bound VJP边界。
    local=1-torch.tanh(tr["logit"][positions]/cap).square()
    expected=jac["dlogit_dinput36"].to(_DEVICE)*local[:,None]
    native_close(jac["ddelta_dinput36"],expected,"saved tanh-Jacobian chain")
    if weight is None:
        need(jac["weighted_dcost_dinput36"] is None,"loss-only no inference weight")
    else:
        need(trace["inference_role_weight"]==weight,"natural route weight")
        need(isinstance(jac["weighted_dcost_dinput36"],torch.Tensor) and jac["weighted_dcost_dinput36"].dtype in (torch.float32,torch.float64) and tuple(jac["weighted_dcost_dinput36"].shape)==(len(positions),36),"weighted Jacobian dtype/shape")
        native_close(jac["weighted_dcost_dinput36"],jac["ddelta_dinput36"].to(_DEVICE)*weight*tr["scale"][0],
                     "saved weighted-cost Jacobian chain")


def row_contract(row,entry):
    need((row["seed"],row["cohort"],row["pool"],qid(row))==
         (entry["seed"],entry["cohort"],entry["pool"],qid(entry)),"tensor/index identity")
    finite_payload(row)
    ids=row["candidate_indices"]
    need(tuple(ids.shape)==(512,) and ids.dtype in (torch.int32,torch.int64) and
         len(set(ids.tolist()))==512 and bool(((ids>=0)&(ids<114982)).all()),"natural candidate IDs")
    need(type(row["target_position"]) is int and -1<=row["target_position"]<512,"natural target position")
    need(row["target_natural"] is (row["target_position"]>=0),"natural target flag")
    if row["target_position"]>=0:
        need(row["target_candidate_id"]==int(ids[row["target_position"]]) and row["candidate_groups"][row["target_position"]]==qid(row)[-1],"natural target ID/group")
    need(len(row["candidate_groups"])==len(row["candidate_smiles"])==512 and
         tuple(row["coarse_cost"].shape)==(512,),"natural metadata/score shape")
    need(type(row["target_position"]) is int and -1<=row["target_position"]<512,"natural target position")
    for name in ("same_query_group","external_heldout","internal_group"):
        mask=row["candidate_qualification_bits"][name]
        need(mask.dtype==torch.bool and tuple(mask.shape)==(512,),"qualification mask shape")
    need(row["strict_excluded"].dtype==torch.bool and tuple(row["strict_excluded"].shape)==(512,),"strict mask shape")
    observation=row["public_observation"]
    gamma=.75 if observation.get("windows") or observation.get("partial_peaks") else .25
    need(row["gamma"]==gamma,"gamma from public observation")
    cap_keys(row["caps"]);cap_keys(row["roles"])
    append=bool(row["loss_roles"])
    need(append==(entry["loss_only_role_row_forwards"]==4),"loss role/index shape")
    if append:
        cap_keys(row["loss_roles"])
        need(row["loss_candidate_indices"].dtype==ids.dtype and
             row["loss_coarse_cost"].dtype==row["coarse_cost"].dtype,"full513 physical dtype prefix")
        need(row["loss_candidate_indices"].dtype in (torch.int32,torch.int64) and bool(((row["loss_candidate_indices"]>=0)&(row["loss_candidate_indices"]<114982)).all()) and
             tuple(row["loss_candidate_indices"].shape)==(513,) and
             tuple(row["loss_coarse_cost"].shape)==(513,) and
             len(row["loss_candidate_groups"])==len(row["loss_candidate_smiles"])==513 and
             torch.equal(row["loss_candidate_indices"][:512],ids) and
             len(set(row["loss_candidate_indices"].tolist()))==513 and
             row["loss_candidate_groups"][:512]==row["candidate_groups"] and
             row["loss_candidate_smiles"][:512]==row["candidate_smiles"] and
             torch.equal(row["loss_coarse_cost"][:512],row["coarse_cost"]),"separate513 candidate shape")
        need(row["loss_strict_excluded"].dtype==torch.bool and tuple(row["loss_strict_excluded"].shape)==(513,),"loss513 exclusion shape")
        need(row["target_position"]==-1 and row["loss_target_position"]==512 and
             row["target_candidate_id"]==int(row["loss_candidate_indices"][512]) and
             row["loss_candidate_groups"][512]==qid(row)[-1],"appended target ID/group")
    else:
        need(row["loss_roles"] is None,"null loss roles")
        if row["actual_train_query"]:
            need(type(row["loss_target_position"]) is int and
                 row["loss_target_position"]==row["target_position"]>=0,"fit natural loss target identity")
        else:
            need(row["loss_target_position"] is None,"nontraining loss target None")
        need(all(row[name] is None for name in ("loss_candidate_indices","loss_candidate_groups","loss_candidate_smiles","loss_coarse_cost","loss_strict_excluded")),"null extra loss data")
    for cap in (5,10):
        roles=capped(row["roles"],cap)
        need(set(roles)==set(ROLES),"exact natural roles")
        cost=capped(row["caps"],cap)
        need(row["coarse_cost"].dtype in (torch.float32,torch.float64) and cost["natural_cost"].dtype in (torch.float32,torch.float64),"native score dtype")
        need(tuple(cost["natural_cost"].shape)==(512,) and
             cost["top10_positions"]==sorted(range(512),key=lambda i:(float(cost["natural_cost"][i]),int(ids[i])))[:10],
             "natural top10 stable positions")
        need(cost["dtypes"]["coarse_dtype"]==str(row["coarse_cost"].dtype) and
             cost["dtypes"]["final_cost_dtype"]==str(cost["natural_cost"].dtype) and
             cost["dtypes"]["correction_dtypes"]==[str((roles[r]["delta"]*roles[r]["scale"][0]).dtype) for r in ROLES],"native cost dtype descriptors")
        need(dtype(cost["dtypes"]["final_cost_dtype"])==np.result_type(dtype(cost["dtypes"]["coarse_dtype"]),*[dtype(t) for t in cost["dtypes"]["correction_dtypes"]]),"native promoted final dtype")
        if append:need(set(capped(row["loss_roles"],cap))==set(ROLES),"exact full513 roles")
        for role,weight in zip(ROLES,(1-gamma,gamma)):
            trace=roles[role]
            need(trace["shape_kind"]=="natural512_inference","natural shape label")
            trace_algebra(trace,512,cap,weight)
            status="separate_loss_only_full513_shape" if append else "fit_current_frozen_loss_reuses_natural512" if row["actual_train_query"] else "not_training_query"
            need(trace["actual_train_loss_status"]==status and (trace["actual_train_loss"] is not None)==(row["actual_train_query"] and not append),"natural actual-loss location/status")
            if append:
                loss=capped(row["loss_roles"],cap)[role]
                need(loss["shape_kind"]=="full513_loss_only" and loss["inference_role_weight"] is None and
                     loss["raw_X"].dtype==trace["raw_X"].dtype and torch.equal(loss["raw_X"][:512],trace["raw_X"]),"distinct full513 loss trace")
                need(all(loss[field].dtype==trace[field].dtype and torch.equal(loss[field],trace[field]) for field in ("scale","fixed_x_center","fixed_o_center")),"same model scaling/centers")
                trace_algebra(loss,513,cap,None)
                need(loss["actual_train_loss_status"]=="fit_current_frozen_loss_full513" and loss["actual_train_loss"] is not None,"full513 actual-loss location/status")
                need(loss["trace_sha256_loss513"]==trace_hash(loss),"full513 typed trace SHA")


def fact_aliases(row,cap,f,oldrow,critical):
    ids=row["candidate_indices"].tolist();target=row["target_position"];append=bool(row["loss_roles"])
    traces=capped(row["roles"],cap);fit=row["actual_train_query"]
    features=f["feature_evidence"]
    need(features["input36_dtype_by_role"]=={r:str(traces[r]["input36"].dtype) for r in ROLES},"feature dtype aliases")
    if target>=0:
        for name,pair in f["diagnostic"]["pairs"].items():
            if pair is None:continue
            j=ids.index(pair["candidate_index"]);flags={};blocks={b:{} for b in ("raw33_RMS","rawO3_RMS")}
            for role in ROLES:
                trace=traces[role];a=trace["input36"].numpy().view(np.uint8).reshape(512,-1)
                flags[role]=bool(np.array_equal(a[target],a[j]))
                for block,field in (("raw33_RMS","raw_X"),("rawO3_RMS","raw_O")):
                    x=trace[field][target].tolist();y=trace[field][j].tolist()
                    blocks[block][role]=math.sqrt(math.fsum((u-v)**2 for u,v in zip(x,y))/len(x))
            equal_tree(features[name],dict(full36_equal_by_role=flags,both_roles_equal=all(flags.values()),**blocks),"feature pair")
        equal_tree(f["forward_matching_proxy"],{r:dict(target_kde020=float(traces[r]["raw_X"][target,2]),target_observed_nearest_mean=float(traces[r]["raw_X"][target,9]),target_observation_positive_nll=float(traces[r]["raw_O"][target,0]),boundary="observation_matching_proxy_not_aligned_forward_error")for r in ROLES},"matching proxy")
    else:need(f["forward_matching_proxy"] is None,"natural missing proxy null")
    supplied=row.get("structural_evidence")
    if supplied is None:need(f["structural_evidence"] is None,"structure unknown")
    else:
        want={k:supplied[k] for k in ("schema","rdkit_version","fingerprint","target_smiles_source")}
        want["pairs"]=capped(supplied["caps"],cap)
        equal_tree(f["structural_evidence"],want,"saved structural evidence")
    for role in ROLES:
        tr=traces[role];lt=capped(row["loss_roles"],cap)[role] if append else tr
        loss=lt["actual_train_loss"];info=f["loss_evidence"][role]
        need(info["status"]==lt["actual_train_loss_status"] and info["actual_historical_exposure"]=="unknown","loss status/exposure alias")
        t=row["loss_target_position"] if fit else target
        delta=float(lt["delta"][t]) if fit else float(tr["delta"][target]) if target>=0 else None
        equal_tree(info["target_saturated95"],abs(delta)/cap>=.95 if delta is not None else None,"saturated95 alias")
        points={"target":int((row["loss_candidate_indices"] if append else row["candidate_indices"])[t]) if t is not None and t>=0 else None,
                "immutable_old_other":oldrow["original_other_candidate_index"] if oldrow else None}
        if f["diagnostic"]["pairs"]:points.update({k:v["candidate_index"] for k,v in f["diagnostic"]["pairs"].items() if v})
        need(set(info["candidate_points"])==set(points),"complete important loss points")
        li=row["loss_candidate_indices"].tolist() if append else ids;positions={cid:i for i,cid in enumerate(li)}
        for name,cid in points.items():
            point=info["candidate_points"][name];pos=positions.get(cid);known=fit and pos is not None
            need(point["candidate_index"]==cid and point["in_loss_row"]==(pos is not None),"loss point ID")
            fields=("probability","listwise_gradient_delta","anchor_gradient_delta","total_gradient_delta","tanh_local_derivative","listwise_gradient_logit","anchor_gradient_logit","total_gradient_logit")
            for field in fields:equal_tree(point[field],float(loss[field][pos]) if known else None,"loss point."+field)
            equal_tree(point["support"],bool(loss["allowed"][pos]) if known else None,"loss point support")
            need(point["status"]==("known" if known else "not_training_query" if not fit else "absent_from_actual_loss_row"),"loss point status")
            if fit:
                gl=float(loss["listwise_gradient_delta"][pos]) if known else None
                ga=float(loss["anchor_gradient_delta"][pos]) if known else None
                equal_tree(point["absolute_anchor_to_listwise_delta_ratio"],abs(ga/gl) if gl not in (0,None) else None,"point anchor ratio")
                need(point["ratio_undefined_reason"]==("not_in_actual_loss" if gl is None else "zero_listwise_gradient" if gl==0 else None),"point ratio reason")
        if fit:
            for field in ("listwise_loss","anchor_loss","total_loss","support_count"):
                equal_tree(info[field],float(loss[field]) if field!="support_count" else loss[field],"loss alias."+field)
            aliases={"target_probability":"probability","target_listwise_gradient":"listwise_gradient_delta","target_anchor_gradient":"anchor_gradient_delta",
                     "target_total_logit_gradient":"total_gradient_logit","target_total_gradient_delta":"total_gradient_delta",
                     "target_tanh_local_derivative":"tanh_local_derivative","target_listwise_gradient_logit":"listwise_gradient_logit","target_anchor_gradient_logit":"anchor_gradient_logit"}
            for alias,field in aliases.items():equal_tree(info[alias],float(loss[field][t]),"target loss alias."+alias)
            equal_tree(info["anchor_opposes_target"],bool(float(loss["listwise_gradient_delta"][t])*float(loss["anchor_gradient_delta"][t])<0),"anchor opposition")
            need(info["actual_loss_formula_verified"] is True,"loss formula alias")
            need(info["current_gradient_scope"]==loss.get("gradient_scope") and info["target_in_natural512"]==row["target_natural"] and info["loss_only_positive_used"]==row["diagnostic_loss_only_positive_used"],"loss context alias")
            for a,b in (("target_absolute_anchor_to_listwise_delta_ratio","absolute_anchor_to_listwise_delta_ratio"),("target_ratio_undefined_reason","ratio_undefined_reason")):equal_tree(info[a],info["candidate_points"]["target"][b],"target ratio alias")
        membership=[]
        for cid in critical:
            pos=positions.get(cid);natural=ids.index(cid) if cid in ids else None
            bits={k:bool(v[natural]) for k,v in row["candidate_qualification_bits"].items()} if natural is not None else None
            allowed=bool(loss["allowed"][pos]) if fit and pos is not None else None
            reason="not_training_query" if not fit else "absent_from_actual_loss_row" if pos is None else "positive_override" if pos==t else "eligible_negative" if allowed else "same_query_group" if bits and bits["same_query_group"] else "external_heldout" if bits and bits["external_heldout"] else "internal_group" if bits and bits["internal_group"] else "excluded_missing_bit_evidence"
            membership.append(dict(candidate_index=cid,in_natural=natural is not None,in_loss=(pos is not None) if fit else None,
                                   allowed=allowed,exclusion_bits=bits,exclusive_reason=reason,exclusive_reason_priority=["positive_override","same_query_group","external_heldout","internal_group"]))
        equal_tree(info["critical_dual_membership_table"],membership,"critical membership")
        if fit:
            equal_tree(info["critical_dual_qualification"],[dict(candidate_index=v["candidate_index"],in_loss_row=v["in_loss"],in_support=bool(v["allowed"]) if v["in_loss"] else False,exclusion_bits=v["exclusion_bits"])for v in membership],"critical qualification alias")
        jac=tr["input_jacobians"];expected=[]
        for pos,values in zip(jac["positions"],jac["weighted_dcost_dinput36"].tolist()):
            expected.append(dict(candidate_index=ids[pos],centered33_RMS=math.sqrt(math.fsum(v*v for v in values[:33])/33),centeredO3_RMS=math.sqrt(math.fsum(v*v for v in values[33:])/3)))
        equal_tree(features[role+"_input_gradient"],expected,"input gradient RMS aliases")


def metadata(request,protocol,replay_stage):
    e=failed_source(replay_stage)
    w=read(request["worker_outputs"][0]["summary"])
    need(w["status"]=="PASS" and w["model_state_unchanged"] and w["inputs_unchanged"],"worker summary")
    need(w["protocol_sha256"]==PROTOCOL_SHA,"worker protocol")
    for name in ("optimizer_instances","optimizer_steps","weight_updates","new_spectrum_forwards","forward_bank_calls","external_api_calls"):
        need(w[name]==0,"worker forbidden:"+name)
    need(w["test_read"] is False and w["model_loads"]==8,"worker operation scope")
    need(w["natural_role_row_forwards"]==w["natural512_inference_role_row_forwards"]==6608,"natural forward count")
    need(w["full513_loss_role_row_forwards"]==w["loss_only_role_row_forwards"]<=1600,"loss forwards")
    need(w["role_row_forwards"]==6608+w["loss_only_role_row_forwards"]<=8640,"total forwards")
    need(w["peak_gpu_allocated_bytes"]<=4*1024**3,"GPU budget")
    need(w["measured_runtime"]==protocol["worker"]["frozen_runtime"],"runtime")
    selection=read(request["query_manifest"]);need(selection["query_seed_pairs"]==826,"826 identities")
    need(request["query_manifest"]["sha256"]==protocol["query_manifest_sha256"],"query pin")
    fixed=read(request["fixed_manifest"])["records"]
    old={(r["seed"],qid(r)):r for r in fixed}
    need(len(old)==400 and sum(r["original_bound_obstructed"] for r in fixed)==87,"fixed400/87")
    index=request["worker_outputs"][0]["index"];need(sha(index["path"])==index["sha256"],"index pin")
    entries=[json.loads(line) for line in io_path(index["path"]).read_text().splitlines()]
    wanted={(r["seed"],r["cohort"],pool,qid(r)) for r in selection["records"] for pool in ("single","dual")}
    actual={(r["seed"],r["cohort"],r["pool"],qid(r)) for r in entries}
    need(actual==wanted and len(entries)==len(actual)==1652,"complete worker identities")
    for row in entries:
        rel=Path(row["tensor_file"]);need(not rel.is_absolute() and ".." not in rel.parts,"relative shard")
    for ref in request["legacy_predictions"]:
        need("test" not in Path(ref["path"]).parts and "sealed" not in str(ref["path"]),"forbidden source")
        need(sha(ref["path"])==ref["sha256"],"legacy raw hash")
    need(w["role_row_forwards"]==8208 and w["loss_only_role_row_forwards"]==1600,"exact completed stage8208")
    return w,entries,old


def population_labels(f,released):
    labels=[]
    if f["fixed400_member"]:
        labels.append("fixed400")
        if f["original87"]:
            labels.append("fixed400_old87")
            if (f["seed"],qid(f)) in released:labels.append("fixed400_old87_C10_oldpair_released")
    if f["population_source"]=="internal":labels.append("internal_hash_probe")
    if f["cohort"] in ("original_val","tune"):
        labels.append("external_observed_full")
        if f["worker_probe"]:labels.append("external_hash_probe")
    return labels


def distribution(values):
    ordered=sorted(v for v in values if v is not None)
    need(all(math.isfinite(v) for v in ordered),"finite distribution")
    def quantile(p):
        if not ordered:return None
        at=(len(ordered)-1)*p;i=int(at)
        return ordered[i]+(ordered[min(i+1,len(ordered)-1)]-ordered[i])*(at-i)
    return dict(available=len(ordered),unknown=len(values)-len(ordered),
                mean=math.fsum(ordered)/len(ordered) if ordered else None,
                quantiles={k:quantile(p) for k,p in (("min",0),("p10",.1),("p25",.25),("median",.5),("p75",.75),("p90",.9),("p95",.95),("max",1))})


def availability(states):
    return dict(total=len(states),known=sum(v=="known" for v in states),
                unknown=sum(v is None for v in states),not_applicable=sum(v=="not_applicable" for v in states))


def nn_state(f,target=False):
    if target and not f["diagnostic"]["target_in_pool"]:return "not_applicable"
    return "known" if any(v.get("status")=="known" for v in (f.get("nn_evidence") or []) if not target or v.get("is_target")) else None


def structure_state(f):
    statuses=[v["status"] for v in (f.get("structural_evidence") or {}).get("pairs",{}).values()]
    return "known" if "known" in statuses else "not_applicable" if statuses and all(v=="not_applicable" for v in statuses) else None


CROSS_NAMES=("old_pair_released_but_actual_top10_failed","released_failed_S1_fail_static_blocked",
 "released_failed_S2_fail_static_open","released_failed_P0_target_absent","S2_exact_collision_unknown",
 "S2_exact_collision_known","S2_any_exact_collision_constraint","S2_collision_at_least10","S2_loss_evidence_unknown",
 "S2_actual_loss_not_applicable","S2_actual_loss_known","S2_anchor_opposes_target_any_role","S2_target_saturated95_any_role",
 "S2_critical_dual_loss_support_known","S2_observation_matching_proxy_available","S2_measurement_mask_unknown",
 "S2_true_forward_error_unknown","S2_crossquery_opposite_loss_label_exact_constraint","S2_crossquery_collision_reference_unknown")


def core_counts(rows):
    c=dict.fromkeys(CROSS_NAMES,0)
    for f in rows:
        d=f["diagnostic"];o=f.get("old_manifest")
        if not o or o["original_coarse_gap"]>d["pairwise_bound"] or d["rank"] is None or d["rank"]<=10:continue
        c["old_pair_released_but_actual_top10_failed"]+=1;c["released_failed_"+d["bucket"]]+=1
        if d["bucket"]!="S2_fail_static_open":continue
        value=(f.get("feature_evidence") or {}).get("both_role_full36_equal_ahead_count")
        c["S2_exact_collision_unknown" if value is None else "S2_exact_collision_known"]+=1
        if value is not None:
            c["S2_any_exact_collision_constraint"]+=int(value>0);c["S2_collision_at_least10"]+=int(value>=10)
        loss=f.get("loss_evidence")
        if loss is None:c["S2_loss_evidence_unknown"]+=1
        elif f["query_qualification"]!="fit":c["S2_actual_loss_not_applicable"]+=1
        else:
            c["S2_actual_loss_known"]+=1
            c["S2_anchor_opposes_target_any_role"]+=int(any(loss[r]["anchor_opposes_target"] is True for r in ROLES))
            c["S2_target_saturated95_any_role"]+=int(any(loss[r]["target_saturated95"] is True for r in ROLES))
            c["S2_critical_dual_loss_support_known"]+=int(all(loss[r]["critical_dual_negatives_in_loss_support"] is not None for r in ROLES))
        if nn_state(f) is None:c["S2_crossquery_collision_reference_unknown"]+=1
        else:c["S2_crossquery_opposite_loss_label_exact_constraint"]+=int(any((v.get("opposite_actual_loss_label_reference_count") or 0)>0 for v in f["nn_evidence"]))
        c["S2_observation_matching_proxy_available"]+=int(f.get("forward_matching_proxy") is not None)
        c["S2_measurement_mask_unknown"]+=int(f["missing_measurement_mask_status"]=="unknown")
        c["S2_true_forward_error_unknown"]+=int(f["true_forward_RMSE_status"]=="unknown")
    return c


def weak_distributions(rows):
    out={}
    for role in ROLES:
        for field in ("target_probability","target_listwise_gradient","target_anchor_gradient","target_total_logit_gradient","critical_dual_negatives_in_loss_support","critical_dual_probability_mass"):
            out[role+".loss."+field]=distribution([(f.get("loss_evidence") or {}).get(role,{}).get(field) for f in rows])
        for field in ("centered36_RMS","raw33_RMS","rawO3_RMS","centered33_RMS","centeredO3_RMS"):
            values=[]
            for f in rows:
                target=[v for v in (f.get("nn_evidence") or []) if v["role"]==role and v.get("is_target")]
                values.append(target[0].get(field) if target and target[0].get("status")=="known" else None)
            out[role+".target_NN."+field]=distribution(values)
        for field in ("target_kde020","target_observed_nearest_mean","target_observation_positive_nll"):
            out[role+".matching_proxy."+field]=distribution([(f.get("forward_matching_proxy") or {}).get(role,{}).get(field) for f in rows])
        for pair in ("immutable_old_other","current_negative1","current_negative10"):
            for block in ("raw33_RMS","rawO3_RMS"):
                out[role+"."+pair+"."+block]=distribution([((f.get("feature_evidence") or {}).get(pair) or {}).get(block,{}).get(role) for f in rows])
    for pair in ("immutable_old_other","current_negative1","current_negative10"):
        objs=[(f.get("structural_evidence") or {}).get("pairs",{}).get(pair) for f in rows]
        out["structure."+pair+".Morgan_Tanimoto"]=distribution([o["tanimoto"] if o and o["status"]=="known" else None for o in objs])
        out["structure."+pair+".availability"]=availability(["known" if o and o["status"]=="known" else "not_applicable" if o and o["status"]=="not_applicable" else None for o in objs])
    return out


def summary_extra(rows):
    return dict(core_crosscounts=core_counts(rows),weak_proxy_distributions=weak_distributions(rows),
                rank=distribution([f["diagnostic"]["rank"] for f in rows]),
                static_unbeatable=distribution([f["diagnostic"].get("static_unbeatable_count") for f in rows]),
                feature_evidence_availability=availability(["known" if f.get("feature_evidence") is not None else None for f in rows]),
                loss_evidence_availability=availability(["not_applicable" if f["query_qualification"]!="fit" else "known" if f.get("loss_evidence") is not None else None for f in rows]),
                nn_availability=availability([nn_state(f) for f in rows]),
                natural_target_nn_availability=availability([nn_state(f,True) for f in rows]),
                structural_availability=availability([structure_state(f) for f in rows]),
                actual_historical_exposure=dict(known=0,unknown=len(rows),not_applicable=0))


def summary_check(summary,facts,released):
    need(summary["fixed400"]==400 and summary["original87"]==87 and summary["C10_released_old87"]==len(released)==66,"core denominators")
    expected_groups=set()
    for f in facts.values():
        for label in population_labels(f,released):
            group=(label,f["cap"],f["seed"],f["pool"],f["cohort"],f["query_qualification"])
            expected_groups.add(group)
            expected_groups.add(group+("source",f["source"]))
            for axis in ("observation_semantics","gamma","public_coverage","positive_membership"):
                expected_groups.add(group+(axis,json.dumps(f.get(axis),ensure_ascii=False,sort_keys=True,separators=(",",":"))))
    actual_groups=[tuple(part["group"]) for part in summary["summaries"]]
    grouped={}
    need(len(actual_groups)==len(set(actual_groups)) and set(actual_groups)==expected_groups,"complete predefined summary strata")
    for part in summary["summaries"]:
        g=part["group"];label,cap,seed,pool,cohort,qualification=g[:6]
        rows=[]
        for f in facts.values():
            if (f["cap"],f["seed"],f["pool"],f["cohort"],f["query_qualification"])!=(cap,seed,pool,cohort,qualification):continue
            ik=(f["seed"],qid(f))
            ok=(label=="fixed400" and f["fixed400_member"] or label=="fixed400_old87" and f["original87"] or
                label=="fixed400_old87_C10_oldpair_released" and ik in released and f["original87"] or
                label=="internal_hash_probe" and f["population_source"]=="internal" or
                label=="external_observed_full" and cohort in ("original_val","tune") or
                label=="external_hash_probe" and cohort in ("original_val","tune") and f["worker_probe"])
            if ok and (len(g)==6 or (f.get(g[6])==g[7] if g[6]=="source" else
                                    json.dumps(f.get(g[6]),ensure_ascii=False,sort_keys=True,separators=(",",":"))==g[7])):
                rows.append(f)
        need(part["query_seed_pairs"]==len(rows),"summary count:"+str(g))
        ranks=[f["diagnostic"]["rank"] for f in rows]
        if not ranks or any(v is None for v in ranks):
            need(all(v is None for v in part["metrics"].values()),"unknown rank metric null")
        else:
            for name in ("top1","top5","top10"):
                need(math.isclose(part["metrics"][name],sum(v<=int(name[3:]) for v in ranks)/len(ranks),abs_tol=1e-14),"summary metric")
            need(math.isclose(part["metrics"]["MRR"],math.fsum(1/v for v in ranks)/len(ranks),abs_tol=1e-14),"summary MRR")
        need(part["buckets"]==dict(collections.Counter(f["diagnostic"]["bucket"] for f in rows)),"summary buckets")
        need(part["unique_query_identities"]==len({qid(f) for f in rows}),"unique query")
        need(part["unique_connectivity_groups"]==len({qid(f)[-1] for f in rows}),"unique connectivity")
        for name,want in summary_extra(rows).items():equal_tree(part[name],want,"summary."+name)
        grouped.setdefault(tuple(g[:2]+g[3:]),{})[seed]=(part,rows)
    means=[]
    for g,seeds in sorted(grouped.items()):
        parts=[v[0] for v in seeds.values()];rows=[f for _,fs in seeds.values() for f in fs]
        mean={name:math.fsum(seeds[z][0]["metrics"][name] for z in (193,211))/2 if set(seeds)=={193,211} and all(seeds[z][0]["metrics"][name] is not None for z in (193,211)) else None for name in ("MRR","top1","top5","top10")}
        counts={name:sum(p["core_crosscounts"][name] for p in parts) for name in CROSS_NAMES}
        means.append(dict(group=list(g),seeds_available=sorted(seeds),query_seed_pairs=len(rows),
                          unique_query_identities=len({qid(f) for f in rows}),unique_connectivity_groups=len({qid(f)[-1] for f in rows}),
                          equal_two_seed_mean=mean,summed_core_crosscounts=counts))
    equal_tree(summary["equal_seed_summaries"],means,"equal seed summaries")


def main():
    global np,torch,PROTOCOL_SHA,CAP,_DEVICE,_WRITE_ROOT
    ap=argparse.ArgumentParser()
    for name in ("request","request-sha256","protocol","protocol-sha256","replay-stage","replay-job","replay-admission","analysis-stage","contract","out","closure-stage","closure-job","closure-stage-admission","closure-review","closure-review-sha256","original-decision","original-decision-sha256","checker-admission","checker-admission-sha256","bindings","analysis-manifest","analysis-manifest-sha256","transport-map","transport-map-sha256"):
        ap.add_argument("--"+name,required=True)
    ap.add_argument("--execute-after-closed",action="store_true")
    args=ap.parse_args();need(args.execute_after_closed,"explicit closed-stage execution required")
    initialize_transport(args)
    started=time.monotonic();PROTOCOL_SHA=args.protocol_sha256
    checker_admission=admit_checker(args)
    need(sha(args.protocol)==PROTOCOL_SHA,"protocol pin")
    protocol=json.loads(io_path(args.protocol).read_text())
    need(sha(args.request)==args.request_sha256,"request pin")
    request=json.loads(io_path(args.request).read_text())
    need(request["algebra_device"]=="cuda:0" and request["runtime"]=="original_JJ" and
         request["fallback_authorized"] is False and request["new_NN_budget"]==0 and
         request["fixed_rtol"]==2e-5 and request["fixed_atol"]==2e-6,"fixed native request")
    need(checker_admission["source_git_commit"]==request["source_git_commit"],"checker author commit binding")
    for ref,digest in ((request["backend_decision"],"41e85949a7991ad3e7cf1824b5308d3891fab60cde85adf5a712cb52366dc4b8"),
                       (request["backend_actual_review"],"79cbdf70743f13c7499eec49c94ecde61216485c89c8375572faf80023396a30")):
        need(ref["sha256"]==digest,"backend evidence identity");read(ref)
    _WRITE_ROOT=request["write_root"]
    need(Path(_WRITE_ROOT).resolve()==ROOT.parent,"explicit scientific write/budget root")
    provenance=closed_output_provenance(args,request,protocol)
    admission=json.loads(io_path(args.replay_admission).read_text())
    need(admission["status"]=="PASS" and admission["independent"] and admission["protocol_sha256"]==PROTOCOL_SHA,"review admission")
    w,entries,old=metadata(request,protocol,args.replay_stage)
    need(w["admission"]==admission,"worker exact admission")
    # All source/closure/decision/independent gates have passed before numerical imports or payload decoding.
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    need(sys.platform=="linux" and str(torch.__version__)=="2.14.0+cu130" and torch.version.cuda=="13.0" and
         torch.cuda.is_available() and torch.cuda.get_device_name(0)=="NVIDIA A800 80GB PCIe","original JJ cuda0 required")
    _DEVICE=torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(0)
    contract_hash=next(h for p,h in protocol["worker"]["profiles"]["remote_replay"]["allowed_files"].items() if p.endswith("/cohorts/cohort_contract.json"))
    need(sha(args.contract)==contract_hash,"contract pin")
    heldout=set(json.loads(io_path(args.contract).read_text())["training_excluded_groups"])
    need(request["qualification_contract"]["sha256"]==sha(args.contract) and request["qualification_bindings"]["sha256"]==sha(args.bindings),"request qualification refs")
    need(sha(args.bindings)==admission["bindings_sha256"],"frozen bindings SHA")
    binding=json.loads(io_path(args.bindings).read_text())
    need(hashlib.sha256("\n".join(sorted(heldout)).encode()).hexdigest()==binding["heldout_groups_sha256"],"heldout group content SHA")
    results=Path(args.analysis_stage)/"results"
    need(io_path(results/"EXIT_CODE").read_text()=="0\n","analysis result exit")
    summary=json.loads(io_path(results/"summary.json").read_text())
    facts={}
    with io_path(results/"diagnostics.jsonl").open() as stream:
        for line in stream:
            f=json.loads(line);k=key(f["seed"],f["cap"],f["pool"],f)
            need(k not in facts,"duplicate fact");facts[k]=f
    selected={(r["seed"],qid(r)) for r in read(request["query_manifest"])["records"]}
    for f in facts.values():
        expected_old=old.get((f["seed"],qid(f))) if f["cohort"]=="train" else None
        need(f["old_manifest"]==expected_old,"exact old manifest provenance")
        need(f["fixed400_member"]==(expected_old is not None),"fact fixed identity")
        need(f["original87"]==bool(expected_old and expected_old["original_bound_obstructed"]),"fact87 identity")
        expected_probe=(f["seed"],qid(f)) in selected
        need(f["worker_probe"]==expected_probe,"worker probe coverage")
        if not expected_probe:
            need(f["cohort"] in ("original_val","tune") and f["query_qualification"]=="external","unprobed cohort")
            need(all(f[name] is None for name in ("feature_evidence","loss_evidence","nn_evidence","structural_evidence")),"unprobed evidence unknown")
        need(f["missing_measurement_mask_status"]=="unknown" and f["true_forward_RMSE_status"]=="unknown","proxy unknown boundary")
    legacy_proofs={};raw_count=0;raw_keys=set();pending_legacy_roles=set();completed_legacy_roles=set()
    for spec in request["legacy_predictions"]:
        with io_path(spec["path"]).open() as stream:
            count=0
            for line in stream:
                r=json.loads(line)
                if qid(r)[1]!="observed":continue
                ids=r["natural_candidate_indices"];coarse=r["natural_candidate_coarse_costs"];cost=r["natural_candidate_costs"]
                target=ids.index(r["target_candidate_index"]) if r["target_candidate_index"] in ids else -1
                d={k:r[k] for k in ("coarse_dtype","correction_dtypes","final_cost_dtype")}
                scales=[v/spec["cap"] for v in r["role_absolute_correction_bounds"]]
                previous=old.get((spec["seed"],qid(r))) if spec["cohort"]=="train" else None
                diag=diagnostic(ids,coarse,cost,target,r["gamma"],scales,spec["cap"],d,
                    old=previous["original_other_candidate_index"] if previous else None,outside=r["cached_outside_pool_rank"])
                need(diag["rank"]==r["rank"],"raw native rank")
                k=key(spec["seed"],spec["cap"],spec["pool"],r);need(k in facts,"legacy missing fact")
                need(k not in raw_keys,"duplicate legacy identity");raw_keys.add(k)
                need(facts[k]["gamma"]==r["gamma"],"legacy gamma")
                has_saved_roles=(spec["seed"],qid(r)) in selected
                need(facts[k]["worker_probe"] is has_saved_roles,"legacy selected evidence join")
                if has_saved_roles:
                    need(k not in pending_legacy_roles,"duplicate deferred legacy role key")
                    pending_legacy_roles.add(k)
                    legacy_proofs[k]=proof(ids,coarse,cost,d,r["natural_candidate_trace_sha256"])
                compare_legacy_diag(facts[k]["diagnostic"],diag,has_saved_roles)
                count+=1;raw_count+=1
        need(count==(200 if spec["cohort"]=="train" else 320 if spec["cohort"]=="original_val" else 1024),"full validation denominator")
        budget(started,full=True)
    refs={};nn={};all_queries=[];loss_rows=0;worker_count=0;calibrations={}
    index_parent=Path(request["worker_outputs"][0]["index"]["path"]).parent
    worker_keys=set()
    for entry in entries:
        path=index_parent/entry["tensor_file"];need(sha(path)==entry["tensor_sha256"],"shard SHA")
        row=torch.load(io_path(path),map_location="cpu",weights_only=True)
        row_contract(row,entry)
        q=qid(row);seed=row["seed"];pool=row["pool"];cohort=row["cohort"]
        fit=cohort=="train" and q[-1] not in heldout and not intern(q[-1])
        need(row["actual_train_query"]==fit and row["query_external_heldout"]==(q[-1] in heldout) and row["query_internal_group"]==intern(q[-1]),"query qualification")
        ids=row["candidate_indices"].tolist();groups=row["candidate_groups"];t=row["target_position"]
        bits=dict(same_query_group=[g==q[-1] for g in groups],external_heldout=[g in heldout for g in groups],internal_group=[intern(g) for g in groups])
        for name,values in bits.items():need(row["candidate_qualification_bits"][name].tolist()==values,"candidate bits")
        excluded=np.logical_or.reduce(list(map(np.asarray,bits.values())))
        need(np.array_equal(excluded,row["strict_excluded"].numpy()),"strict mask")
        append=bool(row["loss_roles"]);need(not append or fit and pool=="single" and t==-1 and row["loss_target_position"]==512,"513 scope")
        oldrow=old.get((seed,q));other=oldrow["original_other_candidate_index"] if oldrow else None
        need(row["fixed400_member"]==(oldrow is not None),"row fixed400 membership")
        if oldrow:
            need(row["target_candidate_id"]==oldrow["target_candidate_index"] and
                 row["original_bound_obstructed"]==oldrow["original_bound_obstructed"],"fixed target/old87 identity")
        required=set()
        for c in (5,10):
            raw_cost=capped(row["caps"],c)["natural_cost"].tolist()
            required.update(sorted(range(512),key=lambda pos:(raw_cost[pos],ids[pos]))[:10])
        if t>=0:required.add(t)
        if other in ids:required.add(ids.index(other))
        for CAP in (5,10):
            capdata=capped(row["caps"],CAP);roles=capped(row["roles"],CAP);d=capdata["dtypes"]
            scales=[float(roles[r]["scale"][0]) for r in ROLES];deltas=[roles[r]["delta"].numpy() for r in ROLES]
            costs=capdata["natural_cost"].numpy();coarse=row["coarse_cost"].numpy()
            need(np.array_equal(native_cost(coarse,*deltas,scales,row["gamma"],d),costs),"native role cost replay")
            oldrow=old.get((seed,q));other=oldrow["original_other_candidate_index"] if oldrow else None
            diag=diagnostic(ids,coarse,costs,t,row["gamma"],scales,CAP,d,old=other,outside=row["cached_outside_pool_rank"],deltas=deltas)
            k=key(seed,CAP,pool,row);f=facts[k];compare_diag(f["diagnostic"],diag)
            need(k not in worker_keys,"duplicate worker fact");worker_keys.add(k)
            qualification="fit" if fit else "internal" if cohort=="train" else "external"
            need(f["query_qualification"]==qualification,"fact qualification")
            need(f["gamma"]==row["gamma"] and f["observation_semantics"]==row["observation_semantics"] and f["public_coverage"]==row["public_coverage"],"fact observation strata")
            need(f["positive_membership"]==("natural512_positive" if t>=0 else "loss_only_appended" if append else "natural_miss_no_loss_append"),"positive membership")
            dual_diag=facts[key(seed,CAP,"dual",row)]["diagnostic"]
            target_id=row["target_candidate_id"]
            critical=[cid for cid in dual_diag["top10_candidate_ids"] if cid!=target_id] if dual_diag["target_in_pool"] else []
            if dual_diag["pairs"] and dual_diag["pairs"]["current_negative10"]["candidate_index"] not in critical:critical.append(dual_diag["pairs"]["current_negative10"]["candidate_index"])
            fact_aliases(row,CAP,f,oldrow,critical)
            traces=[trace_hash(roles[r]) for r in ROLES]
            need(traces==[roles[r]["trace_sha256_natural512"] for r in ROLES],"raw trace SHA")
            if k in legacy_proofs:
                need(proof(ids,coarse,costs,d,traces)==legacy_proofs[k],"exact existing native cost/dtype/trace")
                need(k in pending_legacy_roles and k not in completed_legacy_roles,"same-key deferred role completed once")
                completed_legacy_roles.add(k)
            equal=np.ones(512,dtype=bool)
            if t>=0:
                for r in ROLES:
                    a=roles[r]["input36"].numpy().view(np.uint8).reshape(512,-1);equal &= np.all(a==a[t],axis=1)
                threshold=np.add(costs[t],np.asarray(1e-7,dtype=costs.dtype),dtype=costs.dtype)
                collisions=[ids[i] for i in range(512) if i!=t and equal[i] and costs[i]<=threshold]
                need(f["feature_evidence"]["both_role_full36_equal_ahead_ids"]==collisions and f["feature_evidence"]["both_role_full36_equal_ahead_count"]==len(collisions) and f["feature_evidence"]["representation_collision_top10_blocked"] is (len(collisions)>=10),"typed full36 collision")
            else:need(f["feature_evidence"]["both_role_full36_equal_ahead_count"] is None,"missing target collision null")
            nnkeys=[(v["role"],v["candidate_index"])for v in f["nn_evidence"]]
            need(len(nnkeys)==len(set(nnkeys)) and set(nnkeys)=={(r,ids[pos])for r in ROLES for pos in required},"complete NN important-candidate evidence")
            for role in ROLES:
                tr=roles[role];lt=capped(row["loss_roles"],CAP)[role] if append else tr
                calibration=tuple((str(tr[field].dtype),tr[field].numpy().tobytes()) for field in ("scale","fixed_x_center","fixed_o_center"))
                calibration_key=(seed,CAP,role)
                need(calibration_key not in calibrations or calibrations[calibration_key]==calibration,"saved calibration consistency")
                calibrations[calibration_key]=calibration
                need(tr["input_jacobians"]["positions"]==sorted(required),"frozen important candidate set")
                hashes={v["candidate_index"]:v for v in f["feature_evidence"][role+"_important_candidate_hashes"]}
                need(set(hashes)=={ids[pos] for pos in required},"typed hash coverage")
                for pos in required:
                    got=hashes[ids[pos]]
                    for name,field in (("raw33","raw_X"),("rawO3","raw_O"),("full36","input36")):
                        need(got[name]==typed_hash(tr[field][pos].tolist(),str(tr[field].dtype)),"typed input SHA")
                need(tuple(tr["input36"].shape)==(512,36) and tuple(tr["raw_X"].shape)==(512,33) and tuple(tr["raw_O"].shape)==(512,3),"36 shape")
                if fit:
                    li=row["loss_candidate_indices"].tolist() if append else ids
                    lg=row["loss_candidate_groups"] if append else groups
                    lc=row["loss_coarse_cost"].numpy() if append else coarse
                    target=row["loss_target_position"]
                    ex=np.asarray([g==q[-1] or g in heldout or intern(g) for g in lg])
                    if append:need(lt["input_jacobians"]["positions"]==sorted(required|{512}),"loss important candidate set")
                    allowed=loss_check(lt,lc,target,ex);loss_rows+=1
                    if append:need(np.array_equal(ex,row["loss_strict_excluded"].numpy()),"full513 exclusion")
                    info=f["loss_evidence"][role]
                    need(info["actual_loss_shape"]==len(li),"actual loss shape field")
                    need(info["target_probability"]==float(lt["actual_train_loss"]["probability"][target]),"target probability field")
                    dual=facts[key(seed,CAP,"dual",row)]["diagnostic"]
                    target_id=oldrow["target_candidate_index"]
                    critical=[cid for cid in dual["top10_candidate_ids"] if cid!=target_id]
                    j10=dual["pairs"]["current_negative10"]["candidate_index"]
                    if j10 not in critical:critical.append(j10)
                    need(len(critical)==10,"critical dual ten negatives")
                    positions={cid:pos for pos,cid in enumerate(li)}
                    supported=[positions[cid] for cid in critical if cid in positions and allowed[positions[cid]]]
                    need(info["critical_dual_negatives_in_loss_support"]==len(supported),"critical dual support count")
                    mass=math.fsum(float(lt["actual_train_loss"]["probability"][pos]) for pos in supported)
                    need(math.isclose(info["critical_dual_probability_mass"],mass,rel_tol=1e-14,abs_tol=1e-14),"critical dual probability mass")
                    for point in info["candidate_points"].values():
                        cid=point["candidate_index"];pos=positions.get(cid)
                        need(point["in_loss_row"]==(pos is not None),"loss point membership")
                        if pos is None:
                            need(point["probability"] is None and point["support"] is None,"absent loss point null")
                        else:
                            need(point["support"]==bool(allowed[pos]),"loss point support")
                            for field in ("probability","listwise_gradient_delta","anchor_gradient_delta","total_gradient_logit"):
                                need(math.isclose(point[field],float(lt["actual_train_loss"][field][pos]),rel_tol=1e-14,abs_tol=1e-14),"loss point value")
                    if (seed,q) in old:
                        for pos in np.flatnonzero(allowed):
                            digest=hashlib.sha256(canon(["top10-barrier-training-input-reference-v1",seed,CAP,role,pool,list(q),li[pos]]).encode()).hexdigest()
                            group=(seed,CAP,role,pool);order=(digest,q,li[pos]);heap=refs.setdefault(group,[])
                            if len(heap)>=4096 and order>=heap[0].k:continue
                            payload=dict(reference_hash=digest,candidate_index=li[pos],query_group=q[-1],kind=row["kind"],
                                centered36=lt["input36"][pos].numpy().copy(),raw33=lt["raw_X"][pos].numpy().copy(),rawO3=lt["raw_O"][pos].numpy().copy(),
                                dtype=str(lt["input36"].dtype),label="target" if pos==target else "eligible_negative",input_shape=len(li))
                            retain(refs,group,order,payload,4096)
                else:
                    need(lt["actual_train_loss"] is None,"heldout/internal actual loss null")
                    need(f["loss_evidence"][role]["target_probability"] is None,"fact heldout loss null")
                    for point in f["loss_evidence"][role]["candidate_points"].values():
                        need(point["status"]=="not_training_query" and point["probability"] is None and point["support"] is None,"nontraining point null")
                for pos in tr["input_jacobians"]["positions"]:
                    digest=hashlib.sha256(canon([NN_SALT,seed,CAP,role,pool,list(q),ids[pos]]).encode()).hexdigest()
                    label=("target" if li.index(ids[pos])==target else "eligible_negative") if fit and ids[pos] in li and allowed[li.index(ids[pos])] else None
                    query_meta=dict(fact_key=k,role=role,candidate_index=ids[pos],query_group=q[-1],kind=row["kind"],
                                    group=(seed,CAP,role,pool),dtype=str(tr["input36"].dtype),
                                    input_hash=typed_hash(tr["input36"][pos].tolist(),str(tr["input36"].dtype)),
                                    label=label,is_target=pos==t)
                    all_queries.append(query_meta)
                    payload=dict(query_meta,
                                 centered36=tr["input36"][pos].numpy().copy(),raw33=tr["raw_X"][pos].numpy().copy(),rawO3=tr["raw_O"][pos].numpy().copy())
                    retain(nn,(seed,CAP,role,pool),(digest,q,ids[pos]),payload,NN_LIMIT)
            need(f["actual_historical_exposure"] is None and f["actual_historical_exposure_status"]=="unknown","historical exposure")
        worker_count+=1
        budget(started,full=worker_count%64==0)
    complete_legacy_role_join(pending_legacy_roles,completed_legacy_roles,legacy_proofs)
    need(set(facts)==raw_keys|worker_keys,"complete fact population")
    ref_rows={g:[x.v for x in sorted(heap,key=lambda x:x.k)] for g,heap in refs.items()}
    expected_ref={tuple(x["stratum"]):x for x in summary["training_reference"]}
    need(set(expected_ref)==set(ref_rows) and len(expected_ref)==len(summary["training_reference"]),"reference strata")
    exact_map=collections.defaultdict(list);kind_counts=collections.Counter();group_counts=collections.Counter()
    for g,values in ref_rows.items():
        for r in values:
            r["input_hash"]=typed_hash(r["centered36"],r["dtype"])
            exact_map[(g,r["kind"],r["dtype"],r["input_hash"])].append(r)
            kind_counts[(g,r["kind"])]+=1;group_counts[(g,r["kind"],r["query_group"])]+=1
    full_nn_count_checks=0
    for q in all_queries:
        rows=facts[q["fact_key"]]["nn_evidence"]
        got=next(r for r in rows if r["role"]==q["role"] and r["candidate_index"]==q["candidate_index"])
        need(got["is_target"]==q["is_target"],"NN target identity")
        count=kind_counts[(q["group"],q["kind"])]-group_counts[(q["group"],q["kind"],q["query_group"])]
        if not count:
            need(got["status"]=="unknown","no sampled reference unknown")
        else:
            need(got["status"]=="known" and got["reference_count"]==count,"full NN reference coverage")
            same=[r for r in exact_map[(q["group"],q["kind"],q["dtype"],q["input_hash"])] if r["query_group"]!=q["query_group"]]
            need(got["current_frozen_input_exact_reference_count"]==len(same),"full crossquery typed collision count")
            opposite=sum(r["label"]!=q["label"] for r in same) if q["label"] else None
            need(got["opposite_actual_loss_label_reference_count"]==opposite,"full crossquery opposite-label count")
        full_nn_count_checks+=1
    for g,rows in ref_rows.items():
        got=expected_ref[g];need(got["count"]==len(rows),"reference count")
        need(got["ordered_reference_hashes_sha256"]==hashlib.sha256("\n".join(x["reference_hash"] for x in rows).encode()).hexdigest(),"reference exact prehash selection")
    checked_nn=0
    for g,heap in nn.items():
        for item in heap:
            q=item.v;eligible=sorted((r for r in ref_rows[g] if r["kind"]==q["kind"] and r["query_group"]!=q["query_group"]),key=lambda r:(r["reference_hash"],r["candidate_index"],r["query_group"]))
            got=next(r for r in facts[q["fact_key"]]["nn_evidence"] if r["role"]==q["role"] and r["candidate_index"]==q["candidate_index"])
            if not eligible:need(got["status"]=="unknown","empty NN unknown");continue
            matrix=np.asarray([r["centered36"] for r in eligible],dtype=np.float64)
            diff=matrix-np.asarray(q["centered36"],dtype=np.float64)
            squared=np.einsum("ij,ij->i",diff,diff)/36;at=int(np.argmin(squared));best=eligible[at]
            need(got["reference_count"]==len(eligible) and got["reference_hash"]==best["reference_hash"] and got["reference_candidate_index"]==best["candidate_index"],"NN identity/tie/reference coverage")
            need(math.isclose(got["centered36_RMS"],math.sqrt(float(squared[at])),rel_tol=1e-12,abs_tol=1e-12),"NN numeric distance")
            need(got["reference_input_shape"]==best["input_shape"],"NN reference shape")
            for field,left,right in (("raw33_RMS",q["raw33"],best["raw33"]),("rawO3_RMS",q["rawO3"],best["rawO3"]),
                                     ("centered33_RMS",q["centered36"][:33],best["centered36"][:33]),
                                     ("centeredO3_RMS",q["centered36"][33:],best["centered36"][33:])):
                rms=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(left,right))/len(left))
                need(math.isclose(got[field],rms,rel_tol=1e-12,abs_tol=1e-12),"NN block distance")
            checked_nn+=1
        budget(started)
    released={(f["seed"],qid(f)) for f in facts.values() if f["fixed400_member"] and f["original87"] and f["cap"]==10 and f["pool"]=="dual" and f["old_manifest"]["original_coarse_gap"]<=f["diagnostic"]["pairwise_bound"]}
    summary_check(summary,facts,released)
    paired_count=0;paired_keys=set()
    with io_path(results/"paired_cap_changes.jsonl").open() as stream:
        for line in stream:
            pair=json.loads(line);pk=key(pair["seed"],5,pair["pool"],pair)
            need(pk not in paired_keys and pair["cohort"]==facts[pk]["cohort"],"paired unique identity/cohort");paired_keys.add(pk)
            left=facts[pk]["diagnostic"]
            right=facts[key(pair["seed"],10,pair["pool"],pair)]["diagnostic"]
            a,b=left["top10_candidate_ids"],right["top10_candidate_ids"]
            need(pair["C5_only"]==[i for i in a if i not in set(b)] and pair["C10_only"]==[i for i in b if i not in set(a)],"cap paired identities")
            expected_difference=right["rank"]-left["rank"] if left["rank"] is not None and right["rank"] is not None else None
            need(pair["rank_C10_minus_C5"]==expected_difference,"cap paired rank")
            need(pair["common"]==[i for i in a if i in set(b)] and pair["overlap_count"]==len(set(a)&set(b)) and pair["actual_bucket_C5"]==left["bucket"] and pair["actual_bucket_C10"]==right["bucket"],"cap paired full fields")
            paired_count+=1
    need(paired_count*2==len(facts) and paired_keys=={k for k in facts if k[1]==5},"paired complete denominator")
    budget(started,full=True)
    recheck_analysis_files(provenance["analysis_package"])
    finish_transport()
    output=Path(args.out).resolve();need(output.is_relative_to(Path(_WRITE_ROOT)) and not output.exists(),"new review output root")
    output.parent.mkdir(parents=True,exist_ok=True)
    gpu_peak=torch.cuda.max_memory_allocated(0);need(gpu_peak<=4*1024**3,"saved algebra GPU budget")
    receipt=dict(schema="independent_actual_diagnosis_result_check_v1",status="PASS",independent=True,
        request_sha256=args.request_sha256,protocol_sha256=PROTOCOL_SHA,reviewer_source_sha256=sha(__file__),
        transport_map_sha256=args.transport_map_sha256,algebra_device="cuda:0",runtime="original_JJ",
        runtime_versions=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cuda0=torch.cuda.get_device_name(0)),
        physical_input_transport=_CONSUMED,inputs_unchanged=True,peak_GPU_allocator_bytes=gpu_peak,
        provenance=provenance,checker_admission_sha256=args.checker_admission_sha256,
        scalar_trace_tolerances=dict(center_and_tanh_rtol=2e-6,center_and_tanh_atol=2e-6,Jacobian_rtol=2e-5,Jacobian_atol=2e-6),
        raw_legacy_rows=raw_count,raw_worker_shards=worker_count,loss_role_rows=loss_rows,
        legacy_role_evidence_deferred=len(pending_legacy_roles),legacy_role_evidence_fully_verified=len(completed_legacy_roles),
        role_evidence_failure_review_sha256=checker_admission["role_evidence_failure_review"]["sha256"],
        prior_failed_checker_manifest_sha256=checker_admission["prior_failed_checker_manifest"]["sha256"],
        fixed400=400,old87=87,released_old87=len(released),facts=len(facts),
        exact_reference_vectors=sum(map(len,ref_rows.values())),nn_numeric_checked=checked_nn,full_nn_reference_and_collision_count_checks=full_nn_count_checks,paired_rows=paired_count,
        nn_numeric_scope=dict(salt=NN_SALT,per_seed_cap_role_pool_limit=NN_LIMIT,full_nn_numeric_validation=False,
                             full_count_scope="all sampled worker important candidates; unprobed legacy features remain unknown"),
        scientific_model_calls=0,optimizer_steps=0,external_api_calls=0,elapsed_seconds=time.monotonic()-started,
        checks=["closure/decision chain preserving original0/1FAIL", "all saved tensor finite/cap/roles/512-513 shapes/index identity/gamma", "saved cap-center-Jacobian algebra without model/weights", "all raw ranks/native two-role bounds/current-top10","all worker native cost/dtype/trace","all loss and qualification",
                "512 inference / fit-single513-only loss / heldout nulls","typed36 collision IDs","full prehash reference selection",
                "full crossquery typed/label collision and NN reference counts","predeclared NN numeric subset","full validation and fixed-subset denominators/metrics"],
        proposal_review_status="PENDING",training_authorized=False)
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=="__main__":
    main()

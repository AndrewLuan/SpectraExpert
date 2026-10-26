#!/usr/bin/env python3
"""功能概述：为已完成 NN 输出但监督器退出失败的阶段建立独立 raw-byte 闭合凭证。
主要职责：run 核独审 admission/request/source SHA，保留原阶段 inner0/external1 与 RSS 缺口；
从冻结 manifest/rows/query identities 核1652条唯一输出、全文件字节与SHA、826×2覆盖；
重建138项实际输入闭包，逐字节前后复核模型/cache/源码/元数据，并复核两个缺失路径。
代码库关系：只读取 root 新建的 FAILED_SUPERVISION_OUTPUT_CLOSURE_ONLY 请求及其中绑定的
protocol/job/replay admission/worker/supervisor/query manifest/remote manifest；不导入或执行这些
源码。worker 的模型状态与输入未变字段只作为绑定源码产生的断言，未重新测量模型张量。
输入输出与副作用：必须 --execute、threads4，stdlib-only；仅写新的 --out/closure.json 与
EXIT_CODE。模型/cache会发生raw-byte哈希读取，tensor decode、模型载入、NN、训练、API均为0。
本严格来源链修订新增 exact-int 退出码、原 replay 授权/job/supervisor pin 与摘要 protocol/admission 绑定；
本凭证不把原0/1改成PASS，不认证未知RSS区间，不授权聚合或后续科学执行。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True

SCOPE = "FAILED_SUPERVISION_OUTPUT_CLOSURE_ONLY"
SCHEMA = "failed_supervision_output_closure_request_v1"
IDENTITY = ("query_order","variant","source","record_id","smiles","group")
REFS = frozenset(("protocol","job","replay_admission","worker","supervisor","query_manifest","remote_manifest"))
COUNTS = dict(query_seed_pairs=826,row_shards=1652,natural_role_row_forwards=6608,
              natural512_inference_role_row_forwards=6608,loss_only_role_row_forwards=1600,
              full513_loss_role_row_forwards=1600,role_row_forwards=8208,model_loads=8)
ZERO = ("optimizer_instances","optimizer_steps","weight_updates","new_spectrum_forwards",
        "forward_bank_calls","external_api_calls")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(path):
    raw = os.fspath(path)
    p = Path(raw)
    need(p.is_absolute() and str(p) == raw and str(p.resolve()) == raw, "canonical_absolute_path:" + raw)
    need(not any(x in ("test","sealed") or x.startswith("sealed-") for x in p.parts),"forbidden_source_path")
    return p


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


class Reader:
    """只对已登记的精确文件作raw读取；没有反序列化tensor或执行源码接口。"""
    def __init__(self):
        self.allowed = {}
        self.passes = {}
        self.bytes_read = {}
    def add(self,path,expected):
        p = canonical(path)
        need(valid_sha(expected),"invalid_SHA")
        key = str(p)
        need(key not in self.allowed or self.allowed[key]==expected,"conflicting_pin")
        self.allowed[key] = expected
        return p
    def check(self,path,expected=None,keep=False):
        key = str(canonical(path))
        need(key in self.allowed and (expected is None or expected==self.allowed[key]),"unlisted_raw_read")
        p = Path(key)
        need(p.is_file() and not p.is_symlink(),"regular_input_required:" + key)
        before = p.stat()
        h = hashlib.sha256(); chunks=[]; count=0
        if keep:
            need(before.st_size <= 32*1024**2,"metadata_size_limit")
        with p.open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):
                h.update(block); count += len(block)
                if keep: chunks.append(block)
        after = p.stat()
        need(before.st_size==after.st_size==count and before.st_mtime_ns==after.st_mtime_ns and
             before.st_ino==after.st_ino,"file_changed_during_raw_hash:" + key)
        need(h.hexdigest()==self.allowed[key],"raw_SHA:" + key)
        self.passes[key] = self.passes.get(key,0)+1
        self.bytes_read[key] = self.bytes_read.get(key,0)+count
        return b"".join(chunks) if keep else count
    def json_ref(self,ref):
        p=self.add(ref["path"],ref["sha256"])
        return json.loads(self.check(p,keep=True))
    def finish(self):
        for path in sorted(self.allowed):
            self.check(path)


def identity(row):
    q=row.get("query_identity",row)
    values=tuple(q[k] for k in IDENTITY)
    need(type(values[0]) is int and all(isinstance(x,str) for x in values[1:]),"identity_types")
    return (row["seed"],row["cohort"],*values)


def absent(paths):
    need(len(paths)==2 and len(set(paths))==2,"two_exact_absences")
    for value in paths:
        p=canonical(value)
        need(not os.path.lexists(p),"registered_absence_changed:" + value)


def stage_files(stage):
    files={}
    for root,dirs,names in os.walk(stage,followlinks=False):
        for name in dirs:
            p=Path(root)/name
            need(not p.is_symlink(),"stage_directory_symlink")
        for name in names:
            p=canonical(str(Path(root)/name))
            need(p.is_file() and not p.is_symlink(),"stage_nonregular_file")
            files[str(p.relative_to(stage))]=p
    return files


def mandatory_inputs(binding,w,query_ref):
    values={}
    def add(path,expected=None):
        key=str(canonical(path))
        expected=w["allowed_files"].get(key) if expected is None else expected
        need(valid_sha(expected) and w["allowed_files"].get(key,query_ref["sha256"] if key==query_ref["path"] else None)==expected,
             "mandatory_pin_not_authorized")
        need(key not in values or values[key]==expected,"mandatory_conflict")
        values[key]=expected
    for path,h in binding["source_sha256"].items(): add(path,h)
    for field in ("cap_module","fixed_module"): add(binding[field]["path"],binding[field]["sha256"])
    add(binding["cohort_contract"])
    for cap in ("5","10"):
        for seed in ("193","211"):
            for role in ("no_aug","aug"):
                r=binding["roles"][cap][seed][role]
                add(r["checkpoint"],r["checkpoint_sha256"])
                add(r["centers"],r["centers_sha256"])
                add(r["center_rows"],r["center_rows_sha256"])
                add(r["producer_summary"]);add(str(Path(r["checkpoint"]).parent/"EXIT_CODE"))
    for seed in ("193","211"):
        for cohort in ("train","original_val","tune"):
            for pool in ("single","dual"):
                r=binding["caches"][seed][cohort][pool]
                add(r["path"],r["sha256"])
                producer=r["producer"] if isinstance(r["producer"],dict) else {"path":r["producer"]}
                add(producer["path"],producer.get("sha256"))
                add(str(Path(r["path"]).parent/"EXIT_CODE"))
    for field in ("bindings","validation_bindings"):add(w[field]["path"],w[field]["sha256"])
    for ref in w["reference_predictions"].values():add(ref["path"],ref["sha256"])
    for path,h in w["runtime_metadata_files"].items():add(path,h)
    add(query_ref["path"],query_ref["sha256"])
    need(len(w["runtime_metadata_files"])==5,"exact_five_runtime_metadata")
    return values


def check_rows(blob,manifest,output_manifest):
    records=manifest["records"]
    need(len(records)==826,"frozen_826_records")
    expected={}
    for row in records:
        key=identity(row)
        need(key not in expected,"duplicate_query_manifest_identity")
        expected[key]=row
    seen=set(); shards=set(); natural=loss=0; fixed=0
    lines=blob.decode("utf-8").splitlines()
    need(len(lines)==1652,"exact_1652_index_rows")
    for number,line in enumerate(lines):
        row=json.loads(line);key=identity(row);pool=row["pool"]
        need(key in expected and pool in ("single","dual") and (key,pool) not in seen,"index_identity_coverage")
        selected=expected[key]
        need(row["population"]==selected["population"] and
             row["fixed400_member"]==selected.get("fixed400_member",False) and
             row["query_external_heldout"]==selected["external_heldout"] and
             row["query_internal_group"]==selected["internal_group"],"index_manifest_metadata")
        filename="tensor_shards/"+format(number,"06d")+".pt"
        need(row["tensor_file"]==filename and filename not in shards,"exact_shard_sequence")
        ref=output_manifest["worker/"+filename]
        need(row["tensor_sha256"]==ref["sha256"],"index_manifest_shard_SHA")
        need(row["raw_candidates"]==512 and row["natural_role_row_forwards"]==4,"natural512_row_count")
        extra=row["loss_only_role_row_forwards"]
        need(type(extra) is int and extra in (0,4) and row["role_row_forwards"]==4+extra,"row_forward_alias")
        if extra:
            need(row["cohort"]=="train" and row["population"]=="fixed_fit" and pool=="single" and
                 row["fixed400_member"] is True and row["actual_train_query"] is True and
                 row["loss_candidates"]==513 and row["target_position"]==-1,"loss513_fit_single_only")
            fixed+=1
        else:
            need(row["loss_candidates"] is None,"unexpected_loss_shape")
        need(row["actual_train_query"]==(row["population"]=="fixed_fit"),"actual_train_query_scope")
        natural+=4;loss+=extra;seen.add((key,pool));shards.add(filename)
    need(seen=={(key,pool) for key in expected for pool in ("single","dual")},"complete_826_times2")
    need(natural==6608 and loss==1600 and fixed==400,"exact_forward_sum")
    need({p for p in output_manifest if p.startswith("worker/tensor_shards/")}==
         {"worker/"+p for p in shards},"shard_manifest_coverage")
    return dict(rows=len(lines),query_seed_pairs=len(records),natural=natural,loss513=loss,total=natural+loss)


def run(args):
    need(args.execute and args.threads==4,"explicit_execute_threads4_required")
    reader=Reader()
    admission=reader.json_ref({"path":str(canonical(args.admission)),"sha256":args.admission_sha256})
    need(admission.get("status")=="PASS" and admission.get("independent") is True and
         admission.get("execution_authorized") is True and admission.get("execution_scope")==SCOPE and
         admission.get("request_sha256")==args.request_sha256 and
         admission.get("helper_sha256")==digest(__file__),"independent_closure_admission")
    request=reader.json_ref({"path":str(canonical(args.request)),"sha256":args.request_sha256})
    need(request.get("schema")==SCHEMA and request.get("execution_scope")==SCOPE and
         request.get("execution_authorized") is True and request.get("helper_sha256")==digest(__file__) and
         request.get("prior_role_row_forwards")==2 and set(request["refs"])==REFS,"frozen_closure_request")
    reader.add(str(Path(__file__).resolve()),request["helper_sha256"])
    refs=request["refs"]
    for ref in refs.values():reader.add(ref["path"],ref["sha256"])
    # 只hash源码，不import、compile、exec或torch.load。
    reader.check(refs["worker"]["path"]);reader.check(refs["supervisor"]["path"])
    protocol=reader.json_ref(refs["protocol"]);job=reader.json_ref(refs["job"])
    replay_admission=reader.json_ref(refs["replay_admission"])
    manifest=reader.json_ref(refs["query_manifest"]);remote=reader.json_ref(refs["remote_manifest"])
    w=dict(protocol["worker"],**protocol["worker"]["profiles"]["remote_replay"])
    stage=canonical(request["stage_root"]);out=canonical(args.out);write_root=canonical(request["write_root"])
    need(stage==canonical(remote["remote_stage"]) and job["stage_dir"]==str(stage) and
         write_root==canonical(w["write_root"])==canonical(job["write_root"]) and
         out.is_relative_to(write_root) and not out.is_relative_to(stage) and not out.exists(),
         "new_separate_output_and_exact_stage")
    for key in ("worker","protocol","query_manifest"):
        need(job["source_sha256"].get(refs[key]["path"])==refs[key]["sha256"],"original_job_source_binding")
    need(job["argv"][1]==refs["worker"]["path"] and
         job["argv"][job["argv"].index("--out")+1]==str(stage/"worker"),"original_worker_argv_binding")
    need(replay_admission.get("status")=="PASS" and replay_admission.get("independent") is True and
         replay_admission.get("execution_authorized") is True and
         replay_admission.get("execution_scope")=="FROZEN_RANKER_REPLAY" and
         replay_admission.get("pins",{}).get("job")==refs["job"]["sha256"] and
         replay_admission.get("pins",{}).get("supervisor")==refs["supervisor"]["sha256"] and
         replay_admission["worker_sha256"]==refs["worker"]["sha256"] and
         replay_admission["protocol_sha256"]==refs["protocol"]["sha256"] and
         replay_admission["query_manifest_sha256"]==refs["query_manifest"]["sha256"] and
         w["query_manifest_path"]==refs["query_manifest"]["path"],"original_replay_source_admission")
    need(refs["supervisor"]["sha256"]=="978700874f222d14c578c6b59640636a76afa43821808fc54194e2267c9aa0c8",
         "active_025_grace_supervisor_required")
    files=remote["files"]
    need(len(files)==1661 and set(remote)=={"remote_stage","files"},"exact_closed_manifest_format")
    actual=stage_files(stage)
    need(set(actual)==set(files),"stage_exact_file_membership")
    for relative,ref in files.items():
        rel=Path(relative)
        need(not rel.is_absolute() and ".." not in rel.parts and str(rel)==relative and
             type(ref["bytes"]) is int and ref["bytes"]>=0,"manifest_relative_path_or_size")
        path=reader.add(str(stage/rel),ref["sha256"])
        need(path.stat().st_size==ref["bytes"],"manifest_byte_size")
    execution=reader.json_ref({"path":str(stage/"actual_execution.json"),"sha256":files["actual_execution.json"]["sha256"]})
    summary=reader.json_ref({"path":str(stage/"worker/summary.json"),"sha256":files["worker/summary.json"]["sha256"]})
    need(type(execution["actual_inner_exit"]) is int and type(execution["actual_external_exit"]) is int and
         execution["actual_inner_exit"]==0 and execution["actual_external_exit"]==1 and
         execution["termination_reason"]=="rss_monitor_unavailable" and execution.get("rss_missing_while_alive") is True and
         execution["job_sha256"]==refs["job"]["sha256"] and
         execution["review_sha256"]==refs["replay_admission"]["sha256"],"original_failed_supervision_preserved")
    for relative in ("EXIT_CODE","worker/EXIT_CODE"):
        need(reader.check(stage/relative,keep=True)==b"0\n","original_inner_exit_file")
    need(summary.get("protocol_sha256")==refs["protocol"]["sha256"] and
         summary.get("admission")==replay_admission and
         summary["status"]=="PASS" and summary["model_state_unchanged"] is True and
         summary["inputs_unchanged"] is True and summary["test_read"] is False and
         all(type(summary[k]) is int and summary[k]==v for k,v in COUNTS.items()) and
         all(summary[k]==0 for k in ZERO),"bound_worker_summary_claims")
    binding=reader.json_ref(w["bindings"]);validation=reader.json_ref(w["validation_bindings"])
    need(replay_admission["bindings_sha256"]==w["bindings"]["sha256"] and
         binding["roles"]==validation["roles"] and binding["caches"]==validation["caches"],"bindings_exact")
    checked=summary["input_sha256"]
    expected=mandatory_inputs(binding,w,refs["query_manifest"])
    authorized=dict(w["allowed_files"]);authorized[refs["query_manifest"]["path"]]=refs["query_manifest"]["sha256"]
    need(len(checked)==138 and all(authorized.get(path)==h for path,h in checked.items()) and
         all(checked.get(path)==h for path,h in expected.items()) and
         all(Path(path).suffix==".py" for path in set(checked)-set(expected)),
         "exact_checked_input_whitelist")
    for path,h in checked.items():reader.add(path,h)
    absent(w["registered_absent_runtime_metadata"])
    for path in sorted(checked):reader.check(path)
    for relative,path in sorted(actual.items()):reader.check(path)
    rows=reader.check(stage/"worker/rows.jsonl",keep=True)
    row_counts=check_rows(rows,manifest,files)
    shard_bytes=sum(ref["bytes"] for name,ref in files.items() if name.startswith("worker/tensor_shards/"))
    need(shard_bytes==summary["artifact_bytes"] and sum(x["bytes"] for x in files.values())<=4*1024**3,
         "artifact_byte_contract")
    # 第二遍所有已登记文件和缺失路径；所有原输出与实际输入都在前后byte核验中。
    reader.finish();absent(w["registered_absent_runtime_metadata"])
    need(set(stage_files(stage))==set(files),"stage_membership_changed_at_finish")
    report={
        "schema":"failed_supervision_output_closure_v1","status":"PASS_RAW_OUTPUT_CLOSURE",
        "execution_scope":SCOPE,"request_sha256":args.request_sha256,"admission_sha256":args.admission_sha256,
        "helper_sha256":digest(__file__),"source_refs":refs,"source_stage":str(stage),
        "original_stage_status":"FAIL_SUPERVISION","original_actual_inner_exit":0,"original_actual_external_exit":1,
        "original_termination_reason":"rss_monitor_unavailable","original_rss_gap_status":"unknown_not_certified",
        "original_resource_envelope_recertified":False,
        "worker_state_assertions":{"model_state_unchanged":True,"inputs_unchanged":True,
                                   "evidence":"source-bound worker summary assertions; no independent tensor state remeasurement"},
        "closed_output_files":len(files),"closed_output_bytes":sum(x["bytes"] for x in files.values()),
        "raw_shards_hashed":1652,"shard_bytes":shard_bytes,"index_counts":row_counts,
        "coverage_evidence":{"independent":"826 identities x 2 pools and 1652 indexed shard bytes",
                             "cap_role_shape_coverage":"source-bound worker loops and row metadata; tensor payload coverage not independently recomputed"},
        "checked_input_files":len(checked),"checked_input_sha256":checked,
        "input_raw_hash_passes_min":min(reader.passes[p] for p in checked),
        "input_raw_hash_bytes":sum(reader.bytes_read[p] for p in checked),
        "output_raw_hash_passes_min":min(reader.passes[str(p)] for p in actual.values()),
        "output_raw_hash_bytes":sum(reader.bytes_read[str(p)] for p in actual.values()),
        "source_budget":{"prior_consumed":2,"completed_stage":8208,"cumulative_consumed":8210,
                         "global_ceiling":8640,"remaining":430,"new_role_row_forwards":0},
        "tensor_decode_loads":0,"model_loads":0,"NN_calls":0,"training_steps":0,"external_API_calls":0,
        "aggregation_authorized":False,"scientific_content_decoded_or_aggregated":False,
        "closure_supervision":"Fresh helper exit must independently be observed 0/0; this JSON does not certify its future supervisor result."
    }
    out.mkdir()
    (out/"closure.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    (out/"EXIT_CODE").write_text("0\n")
    print(json.dumps({"status":report["status"],"closed_output_files":1661,"raw_shards_hashed":1652,
                      "NN_calls":0,"receipt":str(out/"closure.json")}))


def main():
    parser=argparse.ArgumentParser()
    for name in ("request","request-sha256","admission","admission-sha256","out"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--threads",type=int,default=4);parser.add_argument("--execute",action="store_true")
    try:run(parser.parse_args())
    except Exception as error:
        print(json.dumps({"status":"FAIL_RAW_OUTPUT_CLOSURE","error":type(error).__name__+": "+str(error)}),file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())

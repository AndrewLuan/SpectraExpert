#!/usr/bin/env python3
"""功能概述：闭合验证器的纯合成metadata/raw-byte契约测试，不读取真实科学产物。
主要职责：fixture 建826身份、1652个明确不可反序列化的假shard、138个小型假输入；
run_case 以独立进程核正常闭合及字节损坏、身份、退出类型、输入pin、计数和admission拒绝。
代码库关系：依赖同目录 close_failed_supervision_outputs_strict.py；只raw-hash当前监督器源码作为
固定9787 provenance，绝不执行它。helper自身仅stdlib，正例假.pt内容故意不是pickle/tensor。
输入输出与副作用：临时fixture均在本目录创建后清理，最终仅写synthetic receipt；
无真实worker运行、tensor decode、NN、训练、API或科学聚合。
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
N=Path(__file__).resolve().parent
HELPER=N/"close_failed_supervision_outputs_strict.py"
SUPERVISOR=N/"supervise_stage_rss_exit_boundary.py"
SCOPE="FAILED_SUPERVISION_OUTPUT_CLOSURE_ONLY"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,separators=(",",":"))+"\n")


def need(value,reason):
    if not value:raise AssertionError(reason)


def fixture(root,case):
    inputs=root/"inputs";inputs.mkdir()
    checked={}
    def data(path,content=b"synthetic raw bytes"):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        checked[str(path)]=sha(path)
        return str(path)
    sources=[data(inputs/"sources"/("source_"+str(i)+".py"),b"# synthetic source; never executed\n")
             for i in range(33)]
    roles={}
    for cap in ("5","10"):
        roles[cap]={}
        for seed in ("193","211"):
            roles[cap][seed]={}
            for role in ("no_aug","aug"):
                d=inputs/"models"/cap/seed/role
                checkpoint=data(d/"final.pt",b"NOT A CHECKPOINT")
                centers=data(d/"centers.json",b"{}")
                rows=data(d/"center_rows.jsonl",b"{}\n")
                summary=data(d/"summary.json",b"{}")
                data(d/"EXIT_CODE",b"0\n")
                roles[cap][seed][role]={"checkpoint":checkpoint,"checkpoint_sha256":checked[checkpoint],
                    "centers":centers,"centers_sha256":checked[centers],
                    "center_rows":rows,"center_rows_sha256":checked[rows],"producer_summary":summary}
    caches={}
    for seed in ("193","211"):
        caches[seed]={}
        for cohort in ("train","original_val","tune"):
            caches[seed][cohort]={}
            for pool in ("single","dual"):
                d=inputs/"cache"/seed/cohort/pool
                path=data(d/"features.pt",b"NOT A CACHE")
                producer=data(d/"producer.json",b"{}")
                data(d/"EXIT_CODE",b"0\n")
                caches[seed][cohort][pool]={"path":path,"sha256":checked[path],
                    "producer":{"path":producer,"sha256":checked[producer]}}
    references={}
    for cap in ("5","10"):
        for seed in ("193","211"):
            for cohort,pool in (("train","dual"),("original_val","single"),("original_val","dual"),
                                ("tune","single"),("tune","dual")):
                key=":".join((cap,seed,cohort,pool));path=data(inputs/"references"/(key+".jsonl"),b"{}\n")
                references[key]={"path":path,"sha256":checked[path]}
    contract=data(inputs/"contract.json",b'{"training_excluded_groups":[]}')
    binding={"roles":roles,"caches":caches,"source_sha256":{p:checked[p] for p in sources[:16]},
             "cap_module":{"path":sources[0],"sha256":checked[sources[0]]},
             "fixed_module":{"path":sources[1],"sha256":checked[sources[1]]},"cohort_contract":contract}
    binding_path=inputs/"bindings.json";write_json(binding_path,binding);checked[str(binding_path)]=sha(binding_path)
    validation=inputs/"validation.json";write_json(validation,binding);checked[str(validation)]=sha(validation)
    metadata={}
    for name in ("PKG-INFO","SOURCES.txt","dependency_links.txt","requires.txt","top_level.txt"):
        path=data(inputs/"spectraexpert.egg-info"/name,b"synthetic metadata")
        metadata[path]=checked[path]
    absences=[str(inputs/"spectraexpert.egg-info"/name) for name in ("METADATA","entry_points.txt")]
    records=[]
    counts=(("fixed_fit","train",400),("internal","train",150),("original_val","original_val",160),("tune","tune",116))
    for population,cohort,count in counts:
        for number in range(count):
            seed=193 if number<count//2 else 211
            q={"query_order":len(records),"variant":"observed","source":"synthetic",
               "record_id":"record_"+str(len(records)),"smiles":"synthetic_identity_"+str(len(records)),
               "group":"group_"+str(len(records))}
            records.append({"seed":seed,"cohort":cohort,"population":population,"query_identity":q,
                            "fixed400_member":population=="fixed_fit","external_heldout":cohort!="train",
                            "internal_group":population=="internal"})
    query=root/"selected_query_manifest.json";write_json(query,{"records":records});checked[str(query)]=sha(query)
    need(len(checked)==138,"fixture138")
    old_worker=root/"synthetic_worker.py";old_worker.write_text("# fake provenance source; never executed\n")
    protocol=root/"protocol.json"
    w={"allowed_files":{p:h for p,h in checked.items() if p!=str(query)},
       "write_root":str(root),"query_manifest_path":str(query),
       "bindings":{"path":str(binding_path),"sha256":sha(binding_path)},
       "validation_bindings":{"path":str(validation),"sha256":sha(validation)},
       "runtime_metadata_files":metadata,"registered_absent_runtime_metadata":absences,
       "reference_predictions":references}
    write_json(protocol,{"source_sha256":{"/LOCAL/DO_NOT_READ.py":"0"*64},"worker":{"profiles":{"remote_replay":w}}})
    stage=root/"failed_original_stage";stage.mkdir();(stage/"worker/tensor_shards").mkdir(parents=True)
    job=root/"original_job.json"
    write_json(job,{"stage_dir":str(stage),"write_root":str(root),
        "source_sha256":{str(old_worker):sha(old_worker),str(protocol):sha(protocol),str(query):sha(query)},
        "argv":[sys.executable,str(old_worker),"replay","--out",str(stage/"worker")]})
    replay_admission=root/"original_admission.json"
    original_proof={"status":"PASS","independent":True,"execution_scope":"FROZEN_RANKER_REPLAY",
        "execution_authorized":True,"pins":{"job":sha(job),"supervisor":sha(SUPERVISOR)},
        "worker_sha256":sha(old_worker),"protocol_sha256":sha(protocol),
        "query_manifest_sha256":sha(query),"bindings_sha256":sha(binding_path)}
    if case=="replay_authorization":original_proof["execution_authorized"]=False
    if case=="job_pin":original_proof["pins"]["job"]="0"*64
    if case=="supervisor_pin":original_proof["pins"]["supervisor"]="0"*64
    write_json(replay_admission,original_proof)
    index=[]
    shard_bytes=0
    for record in records:
        for pool in ("single","dual"):
            number=len(index);name="tensor_shards/"+format(number,"06d")+".pt"
            path=stage/"worker"/name;path.write_bytes(b"UNDECODABLE NON-TENSOR SHARD "+str(number).encode())
            shard_bytes+=path.stat().st_size
            extra=4 if record["population"]=="fixed_fit" and pool=="single" else 0
            row=dict(record,pool=pool,tensor_file=name,tensor_sha256=sha(path),raw_candidates=512,
                natural_role_row_forwards=4,loss_only_role_row_forwards=extra,role_row_forwards=4+extra,
                loss_candidates=513 if extra else None,target_position=-1 if extra else 0,
                actual_train_query=record["population"]=="fixed_fit",
                query_external_heldout=record["external_heldout"],query_internal_group=record["internal_group"])
            index.append(row)
    if case=="identity":
        index[1]["query_identity"]=dict(index[0]["query_identity"]);index[1]["pool"]="single"
    if case=="sequence":index[0]["tensor_file"],index[1]["tensor_file"]=index[1]["tensor_file"],index[0]["tensor_file"]
    (stage/"worker/rows.jsonl").write_text("".join(json.dumps(row)+"\n" for row in index))
    summary={"status":"PASS","model_state_unchanged":True,"inputs_unchanged":True,"test_read":False,
        "protocol_sha256":sha(protocol),"admission":json.loads(replay_admission.read_text()),
        "query_seed_pairs":826,"row_shards":1652,"natural_role_row_forwards":6608,
        "natural512_inference_role_row_forwards":6608,"loss_only_role_row_forwards":1600,
        "full513_loss_role_row_forwards":1600,"role_row_forwards":8208,"model_loads":8,
        "optimizer_instances":0,"optimizer_steps":0,"weight_updates":0,"new_spectrum_forwards":0,
        "forward_bank_calls":0,"external_api_calls":0,"artifact_bytes":shard_bytes,"input_sha256":dict(checked)}
    if case=="summary_protocol":summary["protocol_sha256"]="0"*64
    if case=="summary_admission":summary["admission"]["independent"]=False
    if case=="count":summary["role_row_forwards"]=8207
    if case=="pin":summary["input_sha256"][sources[-1]]="0"*64
    if case=="missing_pin":summary["input_sha256"].pop(sources[-1])
    write_json(stage/"worker/summary.json",summary)
    execution={"actual_inner_exit":0,"actual_external_exit":1,"termination_reason":"rss_monitor_unavailable",
        "rss_missing_while_alive":True,"job_sha256":sha(job),"review_sha256":sha(replay_admission)}
    if case=="exit_type":execution["actual_external_exit"]=0
    if case=="inner_type":execution["actual_inner_exit"]=1
    if case=="bool_inner_exit":execution["actual_inner_exit"]=False
    if case=="bool_external_exit":execution["actual_external_exit"]=True
    write_json(stage/"actual_execution.json",execution)
    for name,content in (("EXIT_CODE","0\n"),("worker/EXIT_CODE","0\n"),("stdout.log","synthetic\n"),
                         ("stderr.log",""),("running.json","{}"),("started.json","{}")):
        (stage/name).write_text(content)
    remote=root/"remote_manifest.json"
    filemap={str(p.relative_to(stage)):{"bytes":p.stat().st_size,"sha256":sha(p)}
             for p in stage.rglob("*") if p.is_file()}
    write_json(remote,{"remote_stage":str(stage),"files":filemap})
    refs={"protocol":protocol,"job":job,"replay_admission":replay_admission,"worker":old_worker,
          "supervisor":SUPERVISOR,"query_manifest":query,"remote_manifest":remote}
    request=root/"closure_request.json"
    write_json(request,{"schema":"failed_supervision_output_closure_request_v1","execution_scope":SCOPE,
        "execution_authorized":True,"helper_sha256":sha(HELPER),"stage_root":str(stage),"write_root":str(root),
        "prior_role_row_forwards":2,"refs":{k:{"path":str(p),"sha256":sha(p)} for k,p in refs.items()}})
    admission=root/"closure_admission.json"
    proof={"status":"PASS","independent":True,"execution_scope":SCOPE,"execution_authorized":True,
           "request_sha256":sha(request),"helper_sha256":sha(HELPER)}
    if case=="admission":proof["execution_scope"]="FROZEN_RANKER_REPLAY"
    write_json(admission,proof)
    if case=="corruption":(stage/"worker/tensor_shards/000000.pt").write_bytes(b"CORRUPTED")
    if case=="extra_file":(stage/"unexpected.txt").write_text("extra")
    if case=="absence":Path(absences[0]).write_text("appeared")
    if case=="input_corruption":Path(sources[-1]).write_text("changed")
    out=root/"fresh_closure_worker"
    args=[str(HELPER),"--request",str(request),"--request-sha256",sha(request),"--admission",str(admission),
          "--admission-sha256",sha(admission),"--out",str(out),"--execute","--threads","4"]
    if case=="no_execute":args.remove("--execute")
    return args,out


def run_case(case):
    with tempfile.TemporaryDirectory(prefix="closure-synthetic-",dir=N) as temporary:
        args,out=fixture(Path(temporary),case)
        process=subprocess.run([sys.executable,*args],capture_output=True,text=True,timeout=30)
        expected=0 if case=="success" else 1
        need(process.returncode==expected,case+":"+process.stdout+process.stderr)
        if case=="success":
            report=json.loads((out/"closure.json").read_text())
            need(report["status"]=="PASS_RAW_OUTPUT_CLOSURE" and report["raw_shards_hashed"]==1652,"positive_closure")
            need(report["original_actual_inner_exit"]==0 and report["original_actual_external_exit"]==1 and
                 report["original_stage_status"]=="FAIL_SUPERVISION" and
                 report["original_resource_envelope_recertified"] is False and
                 report["original_rss_gap_status"]=="unknown_not_certified","failure_preserved")
            need(report["tensor_decode_loads"]==report["NN_calls"]==report["model_loads"]==0,"zero_decode_NN")
            need(report["input_raw_hash_passes_min"]>=2 and report["output_raw_hash_passes_min"]>=2 and
                 report["input_raw_hash_bytes"]>0 and report["output_raw_hash_bytes"]>0,"actual_raw_hashes")
            need(report["source_budget"]["cumulative_consumed"]==8210 and report["source_budget"]["remaining"]==430,
                 "global_budget")
            need(report["aggregation_authorized"] is False and "source-bound" in
                 report["coverage_evidence"]["cap_role_shape_coverage"],"scope_boundary")
        else:
            need(not out.exists(),case+":failure_output")
        return {"case":case,"status":"PASS","actual_exit":process.returncode}


def main():
    tree=ast.parse(HELPER.read_text())
    imports={node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    imports|={alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
    need(imports<={"__future__","argparse","hashlib","json","os","pathlib","sys"},"stdlib_only_source")
    cases=("success","corruption","identity","sequence","exit_type","inner_type","pin","missing_pin",
           "count","admission","no_execute","extra_file","absence","input_corruption",
           "bool_inner_exit","bool_external_exit","replay_authorization","job_pin","supervisor_pin",
           "summary_protocol","summary_admission")
    results=[run_case(case) for case in cases]
    # Deterministic end-change rejection without introducing a race or asynchronous test.
    spec=importlib.util.spec_from_file_location("_closure_synthetic",HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="closure-finish-synthetic-",dir=N) as temporary:
        path=Path(temporary)/"raw.pt";path.write_bytes(b"NOT A TENSOR")
        reader=module.Reader();reader.add(str(path),sha(path));reader.check(path)
        path.write_bytes(b"CHANGED")
        try:reader.finish()
        except ValueError:pass
        else:raise AssertionError("finish_drift_not_rejected")
    results.append({"case":"postcheck_input_drift","status":"PASS"})
    receipt={"schema":"failed_supervision_output_closure_synthetic_v1","status":"PASS",
        "helper_sha256":sha(HELPER),"synthetic_source_sha256":sha(__file__),"cases":results,
        "real_scientific_stage_reads":0,"real_tensor_decode_loads":0,"NN_calls":0,"external_API_calls":0,
        "scope":"Synthetic metadata/raw bytes only; real closure not executed."}
    output=N/"output_closure_strict_synthetic_receipt.json";need(not output.exists(),"receipt_exists")
    write_json(output,receipt)
    print(json.dumps({"status":"PASS","cases":len(results),"receipt":str(output)}))


if __name__=="__main__":main()

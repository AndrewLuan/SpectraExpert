"""功能概述：为root后续锁定流程构造源码registry和声明型metadata aliases。
只读取spec及显式.py源码；不打开任何test plan、producer、缓存、预测或旧报告。
metadata aliases只做路径/SHA声明合并，实际字节由锁后builder Reader再次核验。
不调用prepare/freeze/qualifier，不写远端，不选型，不导入科学模块。
"""
import argparse,hashlib,json
from pathlib import Path

RUNTIME={"h_pool_ranker.py","h_research_ranker.py","h_research_contract.py","h_research_observation.py","h_global_ranker.py","h_global_features.py","data.py","h_pool_fixed_center_ranker.py","cap_ranker.py","train_cap_ranker.py"}
TOOLS={"evaluator","common","statistics","pairmath","matrix","cap_module","fixed_module","builder"}

def need(ok,label):
    if not ok:raise ValueError(label)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def hexsha(v):return type(v) is str and len(v)==64 and all(x in "0123456789abcdef" for x in v)
def canonical(v):
    p=Path(v);need(type(v) is str and p.is_absolute() and ".." not in p.parts and str(p)==v,"canonical_path");return p
def encoded(v):return (json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
def ref(v):
    need(type(v) is dict and set(v)=={"path","sha256"} and hexsha(v["sha256"]),"exact_source_ref");canonical(v["path"]);return v
def alias_allowed(v):
    p=canonical(v);need(p.suffix in (".json",".py") or p.name in ("EXIT_CODE","EXTERNAL_EXIT_CODE"),"alias_metadata_or_source_only")
    need(not any(x in p.name.lower() for x in ("prediction","report","metrics")),"alias_payload_or_report");return p

def construct(spec,out):
    need(spec["schema"]=="cap10_source_registry_alias_spec_20261025","spec_schema")
    root=canonical(spec["remote_root"]);tools=spec["tool_paths"]
    need(set(tools)==TOOLS,"exact_tool_roles")
    for r in tools.values():ref(r)
    expected={"evaluator":root/"native_cost_evaluation/cap_test_evaluate_ranker.py","common":root/"native_cost_evaluation/cap_test_evaluation_common.py","statistics":root/"resource_ready_analysis/summarize_sealed_cap_results.py","pairmath":root/"sealed_cap_pairmath.py","matrix":root/"sealed_cap_matrix.py","cap_module":root/"cap_ranker.py","builder":root/"prepare_cap_test_metadata.py"}
    need(all(tools[k]["path"]==str(v) for k,v in expected.items()),"exact_tool_role_paths")
    need(len({r["path"] for r in tools.values()})==len(tools),"unique_tool_role_paths")
    leaf=spec["leaf_source_sha256"];down=spec["downstream_source_sha256"];ops=spec["operator_source_sha256"]
    need(len(leaf)==12 and {Path(x).name for x in leaf}==RUNTIME|{"cap_test_evaluate_ranker.py","cap_test_evaluation_common.py"},"exact12_leaf_sources")
    need(set(down)=={tools[k]["path"] for k in ("statistics","pairmath","matrix")},"exact_downstream")
    need(set(ops)=={str(root/x) for x in ("isolation_entry.py","supervise_resource_ready_queue.py","launch_resource_ready_stage.py")},"exact_new_operators")
    need(not(set(leaf)&set(down) or set(leaf)&set(ops) or set(down)&set(ops)),"source_roles_disjoint")
    sources={}
    for d in (leaf,down,ops,{tools["builder"]["path"]:tools["builder"]["sha256"]}):
        for path,h in d.items():
            need(hexsha(h) and (path not in sources or sources[path]==h),"source_SHA_conflict");canonical(path);sources[path]=h
    need(all(sources.get(r["path"])==r["sha256"] for r in tools.values()),"tool_SHA_in_source_map")
    need(all(leaf.get(tools[k]["path"])==tools[k]["sha256"] for k in ("evaluator","common","cap_module","fixed_module")),"tool_in_leaf")
    source_alias=spec["source_aliases"];need(set(source_alias)==set(sources),"exact_source_alias_set")
    consumed={}
    for path,h in sources.items():
        local=canonical(source_alias[path]);need(local.suffix==".py" and not local.is_symlink(),"only_real_py_source")
        resolved=local.resolve(strict=True);need(resolved==local and resolved.suffix==".py","source_canonical_no_redirect")
        raw=local.read_bytes();need(digest(raw)==h,"actual_source_SHA");consumed[str(local)]=h
    declared=spec.get("metadata_aliases",{"files":{},"sha256":{}})
    need(set(declared)=={"files","sha256"} and set(declared["files"])==set(declared["sha256"]),"exact_declared_alias_sets")
    files=dict(source_alias);hashes=dict(sources)
    for path,local in declared["files"].items():
        alias_allowed(path);alias_allowed(local);h=declared["sha256"][path];need(hexsha(h),"declared_SHA")
        need(path not in hashes or (hashes[path]==h and files[path]==local),"alias_conflict")
        files[path]=local;hashes[path]=h
    registry={"tool_paths":tools,"leaf_source_sha256":leaf,"downstream_source_sha256":down,"operator_source_sha256":ops}
    registry_path=str(root/"test_source_registry.json");registry_local=str(out/"test_source_registry.json")
    need(registry_path not in files,"registry_self_alias_reserved")
    files[registry_path]=registry_local;hashes[registry_path]=digest(encoded(registry))
    need(all(digest(Path(local).read_bytes())==h for local,h in consumed.items()),"source_changed_at_finish")
    return registry,{"files":files,"sha256":hashes},consumed

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--spec",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    source=Path(__file__).resolve();own=digest(source.read_bytes())
    specpath=canonical(a.spec);need(specpath.suffix==".json" and not specpath.is_symlink() and not any(v in ("test","sealed") or v.startswith(("sealed-","sealed_resume_")) for v in specpath.parts) and specpath.name not in ("summary.json","test.json","artifact_map.json") and not any(v in specpath.name.lower() for v in ("prediction","report","metrics")),"spec_json_required");raw=specpath.read_bytes()
    out=canonical(a.out);need(not out.exists(),"unique_output_required")
    registry,aliases,sources=construct(json.loads(raw),out)
    need(digest(source.read_bytes())==own and specpath.read_bytes()==raw,"source_or_spec_changed")
    out.mkdir(parents=True,exist_ok=False)
    for name,value in (("test_source_registry.json",registry),("test_metadata_alias_map.json",aliases)):(out/name).write_bytes(encoded(value))
    receipt={"status":"SOURCE_REGISTRY_AND_DECLARED_ALIASES_ONLY","source_sha256":{str(source):own,**sources},"spec_sha256":digest(raw),"metadata_alias_bytes_verified":False,"actual_prepare_or_freeze":False,"test_payload_or_producer_opened":False,"scientific_dispatch":0}
    (out/"receipt.json").write_bytes(encoded(receipt))
if __name__=="__main__":main()

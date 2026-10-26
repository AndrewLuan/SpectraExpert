"""功能概述：只用已封存元数据生成原JJ保存量分析的逐文件map、request和unsigned job。
build保留旧request科学字段及逻辑身份，选择库存中唯一已登记物理路径；只复制小元数据原字节。
本程序不打开预测内容、不解码tensor、不运行模型，也不签发独审A/B；实际执行由分析器和supervisor控制。
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

PARENT_SHA = "fe3c835cd39ad098a6a437ed47e79de8980a6b4c7045df9b288131f906aa24f0"
INVENTORY_SHA = "7f781a48fd822f22a775a462e35b91d3a2c183c94834083c45be55642a7789f3"
REMOTE_ROOT = "/home/user/IRfingerprints/SpectrExpert/outputs/nmr_top10_barrier_diagnosis_20261025"
PYTHON = "/home/user/.venvs/spectrexpert/bin/python"
NAMES = ("analyze_diagnostics_native_backend.py", "top10_barrier_math.py", "prepare_native_backend_analysis_request.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")


def exact_entries(inventory):
    rows, seen = [], set()
    for e in inventory["entries"]:
        candidates = e["candidates"]
        assert len(candidates) <= 1, "no_candidate_choice"
        physical = e["preferred_physical_path"] if candidates else e["proposed_physical_path"]
        if candidates:
            c = candidates[0]
            assert c["physical_path"] == physical and c["sha256"] == e["sha256"] and c["bytes"] == e["bytes"]
        assert e["logical_path"] not in seen and type(e["bytes"]) is int and e["bytes"] >= 0
        seen.add(e["logical_path"])
        rows.append(dict(logical_path=e["logical_path"], physical_path=physical,
                         sha256=e["sha256"], bytes=e["bytes"]))
    return rows


def build(parent_path, inventory_path, bundle_path, source_git_commit):
    assert sha(parent_path) == PARENT_SHA and sha(inventory_path) == INVENTORY_SHA
    assert len(source_git_commit) == 40 and all(c in "0123456789abcdef" for c in source_git_commit)
    parent = json.loads(Path(parent_path).read_text())
    inventory = json.loads(Path(inventory_path).read_text())
    bundle, remote = Path(bundle_path).resolve(), Path(REMOTE_ROOT)
    assert not bundle.exists(), "new_bundle_only"
    bundle.mkdir()
    metadata = bundle / "native_analysis_transport_metadata"; metadata.mkdir()
    inputs = bundle / "native_analysis_inputs"; inputs.mkdir()
    for e in inventory["entries"]:
        if not e["candidates"]:
            p = Path(e["logical_path"])
            assert p.stat().st_size == e["bytes"] and sha(p) == e["sha256"]
            assert Path(e["proposed_physical_path"]).parent == remote / metadata.name
            shutil.copyfile(p, metadata / Path(e["proposed_physical_path"]).name)
    for name in NAMES:
        shutil.copyfile(Path(__file__).resolve().with_name(name), inputs / name)
    mapping = dict(schema="native_analysis_exact_transport_map_v1", entries=exact_entries(inventory),
                   inventory_sha256=INVENTORY_SHA, metadata_only=True, prefix_mapping=False, fallback=False)
    map_local = inputs / "exact_transport_map.json"; save(map_local, mapping)
    request = copy.deepcopy(parent)
    request.update(write_root=str(remote), algebra_device="cuda:0", runtime="original_JJ", fallback_authorized=False,
                   fixed_rtol=2e-5, fixed_atol=2e-6, source_git_commit=source_git_commit,
                   baseline_analyzer_sha256="3cd958d15c1f34d672f744bbac78984563ee21b9e09d20459d7d79651dac4c20",
                   parent_request_sha256=PARENT_SHA,
                   transport_map=dict(path=str(remote/inputs.name/map_local.name), sha256=sha(map_local)),
                   backend_decision=dict(path=str(Path(parent_path).parent/"original_saved_jacobian_probe_decision.json"),
                                         sha256="41e85949a7991ad3e7cf1824b5308d3891fab60cde85adf5a712cb52366dc4b8"),
                   backend_actual_review=dict(path=str(Path(parent_path).parent/"saved_jacobian_JJ_algebra_independent_review.json"),
                                              sha256="79cbdf70743f13c7499eec49c94ecde61216485c89c8375572faf80023396a30"))
    request["analysis_source_sha256"] = {str(remote/inputs.name/name):sha(inputs/name) for name in NAMES}
    rp = inputs/"analysis_native_request.json"; save(rp, request)
    stage = remote/"analysis_native_attempt1"
    a, b = remote/inputs.name/"input_admission.json", remote/inputs.name/"stage_admission.json"
    job = dict(schema="top10_bounded_stage_v1", execution_scope=parent["execution_scope"], write_root=str(remote),
               stage_dir=str(stage), max_seconds=3600, max_rss_bytes=32*1024**3, max_artifact_bytes=4*1024**3,
               max_cpu_threads=4, cwd=str(remote/inputs.name), expected_input_admission=str(a), expected_stage_admission=str(b),
               source_sha256={**request["analysis_source_sha256"], str(remote/inputs.name/rp.name):sha(rp),
                              request["transport_map"]["path"]:request["transport_map"]["sha256"]},
               argv=[PYTHON,"-B",str(remote/inputs.name/NAMES[0]),"--request",str(remote/inputs.name/rp.name),
                     "--request-sha256",sha(rp),"--admission",str(a),"--admission-sha256","REVIEW_SHA_TO_BE_BOUND",
                     "--transport-map",request["transport_map"]["path"],"--transport-map-sha256",sha(map_local),
                     "--out",str(stage/"results"),"--execute"])
    save(inputs/"analysis_native_job_unsigned.json", job)
    return dict(bundle=str(bundle), request_sha256=sha(rp), transport_map_sha256=sha(map_local), new_NN_calls=0,
                tensor_decodes=0, prediction_content_reads=0, independent_execution_admission=False)


def main():
    p = argparse.ArgumentParser()
    for name in ("parent", "inventory", "bundle", "source-git-commit"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.parent,args.inventory,args.bundle,args.source_git_commit),sort_keys=True))


if __name__ == "__main__":
    main()

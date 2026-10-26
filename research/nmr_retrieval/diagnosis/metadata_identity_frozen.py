"""功能概述：Top10 屏障诊断的身份投影与冻结 ranker 重放 worker。
主要职责：identities_main 仅从已绑定 train/val JSONL 投影身份并冻结无条件 SHA 样本；
run_replay 在独审 PASS、精确文件白名单及预算约束下加载八个最终模型各一次，重放
已选缓存行，保存实际 raw33/O3/centered36/logit/delta、训练资格、当前 loss 解析梯度
及少量候选的局部输入 Jacobian。没有优化器、权重更新、谱前向或 bank 抽取入口。
代码库关系：由本目录冻结 protocol.worker、旧 mechanism/validation bindings、
固定400 manifest 和独立 admission 驱动；只读旧 cap_ranker/fixed runtime/cache。
输入输出与副作用：身份入口不导入 torch/科学模块；真实入口只写全新 out 目录，
读取均核字节 SHA，并以审计 hook 拒绝科学输入根的非白名单文件和目录外写入。
自然推理始终只取前512；固定 fit 行可将原缓存第513正例仅用于当前训练 loss。
internal/external 不追加目标；其 actual_train_loss 为 null/not_training_query。
O3 是 MLP 输入，不是额外相加的成本。所有梯度只描述当前冻结函数，实际历史曝光 unknown。
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = "top10_frozen_worker_v1"
IDENTITY = ("query_order", "variant", "source", "record_id", "smiles", "group")
SOURCES = ("HMDB", "NMRexp", "WanJuan", "nmrshiftdb2_2018", "nmrshiftdb2_2024")
COHORT_ALIASES = {"train": "train", "original320": "original_val", "tune1024": "tune",
                  "original_val": "original_val", "tune": "tune"}
CANDIDATE_FIELDS = ("features", "coarse_scores", "predicted_ppm", "tags", "H", "K",
                    "per_quantile_sigma")
OBS_FIELDS = ("semantics", "kind", "xs", "weights", "reported_count", "windows", "partial_peaks")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(row):
    q = row.get("query_identity", row)
    need(all(k in q for k in IDENTITY), "incomplete_identity")
    need(type(q["query_order"]) is int and q["variant"] == "observed" and
         q["source"] in SOURCES, "only_observed_registered_identity")
    return tuple(q[k] for k in IDENTITY)


def identity_dict(key):
    return dict(zip(IDENTITY, key))


def internal_group(group):
    return int(hashlib.sha256(("ranker_internal_317" + str(group)).encode()).hexdigest()[:12], 16) % 5 == 0


def sample_hash(salt, seed, cohort, key):
    return hashlib.sha256(canonical([salt, seed, cohort, key[2], list(key)]).encode()).hexdigest()


def prehash_sample(records, salt, limit):
    """按 seed/cohort/source 无条件排序；记录不得含基于结果的筛选条件。"""
    strata = {}
    for record in records:
        key = identity(record)
        stratum = (record["seed"], record["cohort"], key[2])
        strata.setdefault(stratum, []).append(record)
    chosen = []
    for stratum in sorted(strata):
        seed, cohort, _ = stratum
        ordered = sorted(strata[stratum],
                         key=lambda r: (sample_hash(salt, seed, cohort, identity(r)), identity(r)))
        chosen.extend(ordered[:limit])
    return chosen


def deny_scientific_path(path):
    p = Path(path).resolve()
    parts = [x.lower() for x in p.parts]
    need(not any(x == "test" or "sealed" in x or x.startswith("test_") for x in parts),
         "forbidden_test_or_sealed_path")
    need(p.name not in {"research_report.md", "paired_comparisons.tsv", "completion.json",
                        "diagnosis_report.md"}, "forbidden_old_report_path")
    return p


def read_protocol(path, expected):
    p = deny_scientific_path(path)
    need(len(expected) == 64 and digest_file(p) == expected, "protocol_SHA")
    obj = json.loads(p.read_text())
    worker = obj["worker"]
    need(worker["schema"] == SCHEMA and worker["temperature"] == .5 and
         worker["anchor"] == .1, "worker_frozen_math")
    need(worker["max_query_seed_pairs"] == 880 and worker["max_role_row_forwards"] == 8640,
         "worker_frozen_budget")
    need(worker["max_validation_per_stratum"] == worker["max_internal_per_source"] == 16,
         "worker_frozen_sampling")
    need(worker["sampling_salt"] == "top10-barrier-20261025-external-observed" and
         worker["internal_sampling_salt"] == "top10-barrier-20261025-internal-observed", "sole_frozen_salts")
    need(worker.get("hypothetical_loss_enabled", False) is False, "hypothetical_loss_not_admitted")
    return obj, worker


def atomic_json(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, p)


class PinnedInputs:
    """显式路径/SHA闭包；不从任意目录搜索输入，不自动重映射远端路径。"""
    def __init__(self, allowed):
        self.allowed = {str(deny_scientific_path(p)): h for p, h in allowed.items()}
        need(self.allowed and all(isinstance(h, str) and len(h) == 64 for h in self.allowed.values()),
             "nonempty_exact_input_allowlist")
        self.checked = {}

    def verify(self, path, expected=None):
        p = deny_scientific_path(path)
        key = str(p)
        need(key in self.allowed, "input_not_allowlisted:" + key)
        h = self.allowed[key]
        need(expected is None or expected == h, "input_pin_conflict:" + key)
        if key not in self.checked:
            need(digest_file(p) == h, "input_SHA:" + key)
            self.checked[key] = h
        return p

    def read_json(self, path, expected=None):
        return json.loads(self.verify(path, expected).read_text())

    def finish(self):
        for path, expected in self.checked.items():
            need(digest_file(path) == expected, "input_changed:" + path)


def install_io_guard(allowed, protected_roots, out, checked=None):
    """真实执行进程的 open 审计；运行时库只读，科学输入只准白名单，写入只准 out。"""
    expected = {str(Path(p).resolve()): h for p, h in allowed.items()}
    allowed = set(expected)
    checked = {} if checked is None else checked
    roots = [Path(p).resolve() for p in protected_roots]
    out = Path(out).resolve()

    def audit(event, args):
        if event in ("socket.connect", "subprocess.Popen", "os.system"):
            raise ValueError("external_process_or_network_forbidden")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        mode = args[1] or ""
        flags = args[2] or 0
        writing = any(x in mode for x in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing:
            need(p.is_relative_to(out), "write_outside_new_output:" + str(p))
        elif p.is_relative_to(out):
            return
        elif any(p.is_relative_to(root) for root in roots):
            deny_scientific_path(p)
            need(str(p) in allowed, "unlisted_scientific_open:" + str(p))
    # 强制科学源码从已 pin 的 .py 编译，拒绝执行未 pin 的旧 __pycache__。
    import importlib.machinery
    original_get_code = importlib.machinery.SourceFileLoader.get_code
    def source_only(loader, fullname):
        p = Path(loader.path).resolve()
        if any(p.is_relative_to(root) for root in roots):
            need(str(p) in allowed and p.suffix == ".py", "unlisted_source_loader:" + str(p))
            source = loader.get_data(str(p))
            need(hashlib.sha256(source).hexdigest() == expected[str(p)], "import_source_SHA")
            checked[str(p)] = expected[str(p)]
            return loader.source_to_code(source, str(p))
        return original_get_code(loader, fullname)
    importlib.machinery.SourceFileLoader.get_code = source_only
    sys.addaudithook(audit)


def identities_main(args):
    """只读 JSON 元数据，不加载 tensor/model；结束核 SHA 后才写身份 manifest。"""
    _, w = read_protocol(args.protocol, args.protocol_sha256)
    w = dict(w, **w.get("profiles", {}).get("local_identity", {}))
    pins = PinnedInputs(w["allowed_files"])
    source = w["identity_metadata_source"]
    path = pins.verify(source["path"], source["sha256"])
    fixed = w["training_fixed_pairs"]
    fixed_obj = pins.read_json(fixed["path"], fixed["sha256"])
    wanted = {(r["seed"], identity(r)): r for r in fixed_obj["records"]}
    need(len(wanted) == 400 and sum(r["original_bound_obstructed"] for r in wanted.values()) == 87,
         "fixed400_old87")
    chosen_fixed, internals, externals, seen = {}, [], [], set()
    counts = {}
    with path.open("r") as stream:
        for line in stream:
            row = json.loads(line)
            # 不读取或解释 single/dual/category/rank/cost 等结果字段。
            cohort = COHORT_ALIASES.get(row["cohort"])
            need(cohort in ("train", "original_val", "tune"), "metadata_forbidden_cohort")
            q = row["query_identity"]
            if q["variant"] != "observed":
                continue
            key = identity(q); seed = row["seed"]
            need(seed in (193, 211), "metadata_seed")
            unique = (seed, cohort, key)
            need(unique not in seen, "duplicate_metadata_identity"); seen.add(unique)
            counts[(seed, cohort)] = counts.get((seed, cohort), 0) + 1
            is_internal = internal_group(key[-1])
            record = dict(seed=seed, cohort=cohort, query_identity=identity_dict(key),
                          cache_refs={pool: {"path": row["cache_refs"][pool]["path"],
                                             "row_index": row["cache_refs"][pool]["row_index"]}
                                      for pool in ("single", "dual")},
                          external_heldout=row["external_heldout"],
                          internal_group=is_internal, actual_exposure=None,
                          actual_exposure_status="unknown")
            if cohort == "train":
                need(row["external_heldout"] is False and
                     row["training_population"] == ("internal" if is_internal else "fit"),
                     "metadata_train_qualification")
                if (seed, key) in wanted:
                    need(not is_internal, "fixed400_not_fit")
                    old = wanted[(seed, key)]
                    record.update(population="fixed_fit", fixed400_member=True,
                                  target_candidate_index=old["target_candidate_index"],
                                  original_other_candidate_index=old["original_other_candidate_index"],
                                  original_bound_obstructed=old["original_bound_obstructed"])
                    chosen_fixed[(seed, key)] = record
                elif is_internal:
                    record["population"] = "internal"; internals.append(record)
            else:
                need(row["external_heldout"] is True and row["training_population"] == "validation",
                     "metadata_external_qualification")
                record["population"] = cohort; externals.append(record)
    need(set(chosen_fixed) == set(wanted), "fixed400_metadata_missing")
    for seed in (193, 211):
        need(counts[(seed, "train")] == 8192 and counts[(seed, "original_val")] == 320 and
             counts[(seed, "tune")] == 1024, "complete_observed_metadata")
    internal = prehash_sample(internals, w["internal_sampling_salt"], 16)
    external = prehash_sample(externals, w["sampling_salt"], 16)
    records = list(chosen_fixed.values()) + internal + external
    records.sort(key=lambda r: (r["seed"], r["cohort"], identity(r)))
    need(len(internal) <= 160 and len(external) <= 320 and len(records) <= 880, "identity_budget")
    pins.finish()
    output = Path(args.out).resolve()
    need(output.is_relative_to(Path(w["write_root"]).resolve()), "metadata_output_root")
    need(not output.exists(), "metadata_output_exists")
    value = dict(schema=SCHEMA + "_query_manifest", protocol_sha256=args.protocol_sha256,
                 metadata_source=source, records=records, fixed_fit_count=400,
                 fixed400_all_fit=True, internal_count=len(internal), external_count=len(external),
                 query_seed_pairs=len(records), role_row_forward_ceiling=len(records) * 8 + 1600,
                 natural_role_row_forward_ceiling=len(records) * 8,
                 loss_only_role_row_forward_ceiling=1600,
                 selection="predeclared_unconditioned_identity_SHA", input_sha256=pins.checked,
                 tensor_loads=0, ranker_loads=0, ranker_forwards=0, scientific_values_interpreted=False,
                 test_read=False, external_api_calls=0)
    atomic_json(output, value)
    return value


def analytical_loss(coarse, delta, logits, scale0, target, excluded, cap, anchor=.1, temperature=.5):
    """单 role 当前训练 loss 的解析导数；不构建优化器，不反传到参数或 coarse/scale。"""
    import torch
    need(excluded.dtype == torch.bool and excluded.shape == delta.shape and
         0 <= target < len(delta), "loss_input_shape")
    excluded = excluded.to(delta.device).clone(); excluded[target] = True
    need(bool((~excluded).any()), "loss_no_negative")
    allowed = ~excluded; allowed[target] = True
    cost = coarse.to(delta.device) / scale0 + delta
    p = torch.zeros_like(cost); p[allowed] = torch.softmax(-cost[allowed] / temperature, 0)
    indicator = torch.zeros_like(p); indicator[target] = 1
    gl = torch.where(allowed, (indicator - p) / temperature, torch.zeros_like(p))
    ga = torch.where(allowed, 2 * anchor * delta / int(allowed.sum()), torch.zeros_like(delta))
    derivative = 1 - torch.tanh(logits / cap).square()
    list_loss = torch.logsumexp(-cost[allowed] / temperature, 0) + cost[target] / temperature
    anchor_loss = anchor * delta[allowed].square().mean()
    return dict(actual_normalized_costs=cost, normalized_cost_dtype=str(cost.dtype),
                allowed=allowed, probability=p, probability_dtype=str(p.dtype), listwise_gradient_delta=gl,
                anchor_gradient_delta=ga, total_gradient_delta=gl + ga,
                tanh_local_derivative=derivative, listwise_gradient_logit=gl * derivative,
                anchor_gradient_logit=ga * derivative, total_gradient_logit=(gl + ga) * derivative,
                listwise_loss=list_loss, anchor_loss=anchor_loss, total_loss=list_loss + anchor_loss,
                support_count=int(allowed.sum()), gradient_scope="current_single_role_row_before_batch_mean",
                actual_historical_exposure=None, actual_historical_exposure_status="unknown")


def shape_plan(row, fit, pool, fixed400_member):
    """注册 shape 只由原缓存正例位置决定；不根据当前科学结果增加前向。"""
    n = len(row["candidate_indices"]); target = int(row["target_position"])
    need(n in (512, 513), "shape_candidate_count")
    appended = fit and n == 513
    need(not (fit and pool == "dual" and appended), "fixed_dual_appended_manifest_conflict")
    need(not appended or (pool == "single" and fixed400_member and target == 512),
         "unregistered_loss513_shape")
    need(not fit or (0 <= target < n), "fit_target_shape")
    return dict(natural_candidates=512, natural_target_position=target if 0<=target<512 else -1,
                loss_candidates=513 if appended else (512 if fit else None),
                loss_target_position=target if fit else None, extra_loss_forward=appended)


def scoring_input(row, fit):
    """评分只接收物理输入；非 fit 行裁去缓存 loss-only 正例。"""
    import torch
    n = len(row["candidate_indices"]) if fit else 512
    need(row["evaluation_pool_size"] == 512 and n in (512, 513), "natural512_loss513")
    out = {"kind": row["kind"], "observation_input": {k: row["observation_input"][k]
            for k in OBS_FIELDS if k in row["observation_input"]}}
    for field in CANDIDATE_FIELDS:
        if field in row:
            need(len(row[field]) == len(row["candidate_indices"]), "candidate_field_length")
            out[field] = row[field][:n]
            need(bool(torch.isfinite(out[field]).all()), "nonfinite_candidate")
    need(out["features"].shape == (n, 33), "raw33")
    return out


def capture_role(model, scale, saved, row_input, device):
    """同一次实际 row_delta 捕获 O3 与输入36；仅输入节点允许 autograd。"""
    import torch
    for parameter in model.parameters():
        need(not parameter.requires_grad and parameter.grad is None, "model_not_frozen")
    captured = {}
    def observation_hook(module, args, output):
        captured["raw_O"] = output.detach()
    def network_pre(module, args):
        value = args[0].detach().requires_grad_(True)
        captured["input36"] = value
        return (value,)
    def network_post(module, args, output):
        captured["logit"] = output[:, int(row_input["kind"])]
    handles = [model.observation_model.register_forward_hook(observation_hook),
               model.network.register_forward_pre_hook(network_pre),
               model.network.register_forward_hook(network_post)]
    try:
        with torch.enable_grad():
            delta = model.row_delta(row_input, scale, device)
    finally:
        for handle in handles:
            handle.remove()
    x = row_input["features"].to(device)
    expected = torch.cat((((x - model.fixed_x_center) / scale).clamp(-20, 20),
                          (captured["raw_O"] - model.fixed_o_center).clamp(-20, 20)), 1)
    need(torch.equal(captured["input36"].detach(), expected), "actual_centered36_mismatch")
    need(torch.equal(delta.detach(), saved["delta_cap"] *
                     torch.tanh(captured["logit"].detach() / saved["delta_cap"])), "cap_formula")
    need(captured["raw_O"].shape == (len(x), 3), "raw_O3")
    return dict(raw_X=x.detach(), raw_O=captured["raw_O"], input36=captured["input36"],
                logit=captured["logit"], delta=delta, scale=scale.detach(),
                fixed_x_center=model.fixed_x_center.detach(), fixed_o_center=model.fixed_o_center.detach())


def selected_jacobians(trace, positions, weight, scale0, cap):
    """逐候选独立 MLP 的 selected logit sum VJP；无额外 forward。"""
    import torch
    positions = sorted(set(positions))
    if not positions:
        return {"positions": [], "dlogit_dinput36": None, "ddelta_dinput36": None,
                "weighted_dcost_dinput36": None}
    grad = torch.autograd.grad(trace["logit"][positions].sum(), trace["input36"],
                               retain_graph=False, create_graph=False)[0][positions]
    local = 1 - torch.tanh(trace["logit"].detach()[positions] / cap).square()
    return dict(positions=positions, dlogit_dinput36=grad.detach(),
                ddelta_dinput36=(grad * local[:, None]).detach(),
                weighted_dcost_dinput36=(grad * local[:, None] * weight * scale0).detach())


def native_top10(ids, costs):
    import torch
    by_id = torch.argsort(ids[:512], stable=True)
    return by_id[torch.argsort(costs[:512][by_id], stable=True)][:10].tolist()


def tensor_trace_hashes(trace, natural_only=True):
    """与已封存 native evaluator 的逐候选 trace SHA 字节契约一致。"""
    import numpy as np
    fields = dict(raw_X=trace["raw_X"], raw_O=trace["raw_O"],
                  centered_X=trace["input36"][:, :33], centered_O=trace["input36"][:, 33:],
                  logits=trace["logit"], delta=trace["delta"])
    arrays = [(k, v.detach().cpu().contiguous().numpy()) for k, v in sorted(fields.items())]
    result = []
    for i in range(512 if natural_only else len(trace["raw_X"])):
        h = hashlib.sha256()
        for k, arr in arrays:
            value = np.asarray(arr[i])
            h.update(k.encode()); h.update(str(value.dtype).encode()); h.update(str(value.shape).encode())
            h.update(value.tobytes())
        result.append(h.hexdigest())
    return result


def detached(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [detached(v) for v in value]
    return value


def load_module(path):
    spec = importlib.util.spec_from_file_location("_top10_bound_cap_runtime", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def admit(args, w):
    receipt = json.loads(Path(args.admission).read_text())
    need(receipt["status"] == "PASS" and receipt["execution_authorized"] is True and
         receipt["protocol_sha256"] == args.protocol_sha256 and
         receipt["worker_sha256"] == digest_file(__file__) and
         receipt["query_manifest_sha256"] == digest_file(args.query_manifest) and
         receipt["bindings_sha256"] == digest_file(args.bindings), "independent_admission_required")
    return receipt


_STRUCTURE_CACHE = {}
def structure_pair(target_smiles, other_smiles, identity_fn):
    """复用项目 identity，再用固定非手性 Morgan 位指纹；非法/多组分保留 unknown。"""
    import rdkit
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    def one(smiles):
        if smiles not in _STRUCTURE_CACHE:
            try:
                registered = identity_fn(smiles)
                iso, group = registered[0], registered[1]
                mol = Chem.RemoveHs(Chem.MolFromSmiles(iso))
                need(mol is not None and "." not in Chem.MolToSmiles(mol), "invalid_or_multicomponent")
                fp = rdFingerprintGenerator.GetMorganGenerator(
                    radius=2, fpSize=2048, includeChirality=False).GetFingerprint(mol)
                _STRUCTURE_CACHE[smiles] = dict(status="known", canonical_isomeric_smiles=iso,
                                               connectivity_group=group, fingerprint=fp)
            except Exception as exc:
                _STRUCTURE_CACHE[smiles] = dict(status="unknown", reason=str(exc))
        return _STRUCTURE_CACHE[smiles]
    a, b = one(target_smiles), one(other_smiles)
    common = dict(rdkit_version=rdkit.__version__, radius=2, nBits=2048, useChirality=False)
    if a["status"] != "known" or b["status"] != "known":
        return dict(common, status="unknown", tanimoto=None, same_connectivity=None,
                    reason="target:"+a.get("reason","ok")+";other:"+b.get("reason","ok"))
    return dict(common, status="known", tanimoto=float(DataStructs.TanimotoSimilarity(
                    a["fingerprint"], b["fingerprint"])),
                same_connectivity=a["connectivity_group"]==b["connectivity_group"],
                target_canonical_isomeric_smiles=a["canonical_isomeric_smiles"],
                other_canonical_isomeric_smiles=b["canonical_isomeric_smiles"],
                target_connectivity_group=a["connectivity_group"],
                other_connectivity_group=b["connectivity_group"])


def structural_evidence(row, selected, ids, caps, smiles, identity_fn):
    """每 cap/pool 仅 target-old/j1/j10 三对；不进行额外 ranker forward 或采样。"""
    import torch, rdkit
    target_pos = int(row["target_position"])
    target_id = selected.get("target_candidate_index",
                             int(ids[target_pos]) if 0<=target_pos<512 else None)
    target_smiles = smiles[target_id] if target_id is not None else row["smiles"]
    result = dict(schema="top10_structural_evidence_v1", rdkit_version=rdkit.__version__,
                  fingerprint=dict(algorithm="Morgan",radius=2,nBits=2048,useChirality=False),
                  target_smiles_source="candidate_library" if target_id is not None else "query_identity",
                  caps={})
    for cap in (5,10):
        by_id = torch.argsort(ids, stable=True)
        order = by_id[torch.argsort(caps[cap]["natural_cost"].cpu()[by_id], stable=True)].tolist()
        negatives = [p for p in order if int(ids[p]) != target_id]
        comparators = dict(immutable_old_other=selected.get("original_other_candidate_index"),
                           current_negative1=int(ids[negatives[0]]),
                           current_negative10=int(ids[negatives[9]]))
        result["caps"][cap] = {}
        for label, other_id in comparators.items():
            if other_id is None:
                value = dict(status="not_applicable", tanimoto=None, same_connectivity=None,
                             reason="immutable_old_other_not_registered")
            else:
                value = structure_pair(target_smiles, smiles[other_id], identity_fn)
            result["caps"][cap][label] = dict(value, target_candidate_id=target_id,
                                              other_candidate_id=other_id)
    return result


def cuda_runtime_snapshot(torch):
    prop = torch.cuda.get_device_properties(0)
    return dict(torch_version=str(torch.__version__),torch_cuda=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cpu_threads=torch.get_num_threads(),torch_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                device0=dict(name=prop.name,total_memory=prop.total_memory,major=prop.major,minor=prop.minor,
                             multi_processor_count=prop.multi_processor_count))


def run_replay(args):
    """真实入口；只在独审凭证通过后导入科学模块和读取大型缓存。"""
    args._started_monotonic = time.monotonic()
    protocol, w = read_protocol(args.protocol, args.protocol_sha256)
    w = dict(w, **w.get("profiles", {}).get("remote_replay", {}))
    need(args.execute and args.admission, "explicit_execute_and_PASS_required")
    admission = admit(args, w)
    allowed = dict(w["allowed_files"])
    need(str(Path(args.query_manifest).resolve()) == w["query_manifest_path"], "frozen_query_manifest_path")
    allowed[str(Path(args.query_manifest).resolve())] = admission["query_manifest_sha256"]
    pins = PinnedInputs(allowed)
    need(str(Path(args.bindings).resolve()) == w["bindings"]["path"], "frozen_mechanism_bindings_path")
    bindings = pins.read_json(args.bindings, w["bindings"]["sha256"])
    vb = w["validation_bindings"]
    validation_bindings = pins.read_json(vb["path"], vb["sha256"])
    for seed in ("193", "211"):
        for cohort in ("train", "original_val", "tune"):
            for pool in ("single", "dual"):
                a, b = bindings["caches"][seed][cohort][pool], validation_bindings["caches"][seed][cohort][pool]
                need((a["path"], a["sha256"]) == (b["path"], b["sha256"]), "validation_cache_binding_conflict")
    for cap in ("5", "10"):
        for seed in ("193", "211"):
            for role in ("no_aug", "aug"):
                a, b = bindings["roles"][cap][seed][role], validation_bindings["roles"][cap][seed][role]
                need(all(a[k] == b[k] for k in ("checkpoint","checkpoint_sha256","centers_sha256","center_rows_sha256")),
                     "validation_model_binding_conflict")
    need(all(validation_bindings["source_sha256"].get(k, v) == v for k, v in bindings["source_sha256"].items()),
         "validation_source_binding_conflict")
    selection = pins.read_json(args.query_manifest)
    need(selection["schema"] == SCHEMA + "_query_manifest" and
         selection["protocol_sha256"] == args.protocol_sha256 and selection["fixed400_all_fit"],
         "frozen_query_manifest")
    records = selection["records"]
    w["_selected_records"] = records
    need(len(records) <= 880 and len({(r["seed"], r["cohort"], identity(r)) for r in records}) == len(records),
         "query_population_budget")
    out = Path(args.out).resolve()
    need(out.is_relative_to(Path(w["write_root"]).resolve()) and not out.exists(), "new_output_required")
    # Admission, protocol and manifest are already read before process-local open audit installation.
    out.mkdir(); args._created_out = True; (out / "tensor_shards").mkdir()
    install_io_guard(pins.allowed, w["protected_read_roots"], out, pins.checked)
    import torch
    need(args.device == w["device"] == "cuda:0" and args.threads == w["max_cpu_threads"] == 4,
         "frozen_cuda_device_and_threads")
    torch.set_num_threads(4); device = torch.device("cuda:0")
    need(torch.cuda.is_available(), "frozen_CUDA_unavailable_no_fallback")
    total_gpu = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., w["max_gpu_allocated_bytes"]/total_gpu), device)
    torch.cuda.reset_peak_memory_stats(device)
    measured_runtime = cuda_runtime_snapshot(torch)
    need(measured_runtime == w["frozen_runtime"], "frozen_CUDA_runtime_mismatch_no_fallback")
    module_info = bindings["cap_module"]; pins.verify(module_info["path"], module_info["sha256"])
    fixed = bindings["fixed_module"]; pins.verify(fixed["path"], fixed["sha256"])
    for path, digest in bindings["source_sha256"].items():
        pins.verify(path, digest)
    module = load_module(module_info["path"])
    runtime = module.configure_runtime(bindings["source_checkout"], fixed["path"])
    for path in runtime.source_paths():
        pins.verify(path)
    contract = pins.read_json(bindings["cohort_contract"])
    heldout = set(contract["training_excluded_groups"])
    need(hashlib.sha256("\n".join(sorted(heldout)).encode()).hexdigest() ==
         bindings["heldout_groups_sha256"], "heldout_contract_SHA")
    models, before = {}, {}
    for cap in (5, 10):
        for seed in (193, 211):
            for role in ("no_aug", "aug"):
                record = bindings["roles"][str(cap)][str(seed)][role]
                path = pins.verify(record["checkpoint"], record["checkpoint_sha256"])
                for name in ("centers", "center_rows", "producer_summary"):
                    pins.verify(record[name])
                pins.verify(path.parent / "EXIT_CODE")
                model, scale, saved = module.load_checkpoint(path, device, expected_cap=cap,
                                                             require_final=True, runtime=runtime)
                need(saved["seed"] == seed and saved["role"] == role and saved["alpha"] == 1 and
                     saved["epoch"] == 64, "final_model_identity")
                for p in model.parameters():
                    p.requires_grad_(False); need(p.grad is None, "existing_parameter_grad")
                model.eval(); key = (cap, seed, role)
                models[key] = (model, scale, saved); before[key] = module.tensor_hashes(model, scale)
    # 其余 IO/逐行逻辑显式在下方函数；没有训练或谱模型入口。
    return replay_rows(args, w, pins, bindings, selection, runtime, module, models, before, heldout, out, admission)


def replay_rows(args, w, pins, bindings, selection, runtime, module, models, before, heldout, out, admission):
    """两种已注册真实 shape：natural512；fit single append 时另做 loss-only full513。"""
    import torch
    started = getattr(args, "_started_monotonic", time.monotonic()); natural_forwards = 0; loss_forwards = 0
    written = 0; shard_number = 0; groups = None
    records = selection["records"]; seen = set()
    with (out / "rows.jsonl").open("x") as index:
        for seed in (193, 211):
            for cohort in ("train", "original_val", "tune"):
                selected = [r for r in records if r["seed"] == seed and r["cohort"] == cohort]
                for pool in ("single", "dual"):
                    cache = bindings["caches"][str(seed)][cohort][pool]
                    path = pins.verify(cache["path"], cache["sha256"])
                    producer = cache["producer"]
                    if isinstance(producer, str):
                        producer = {"path": producer}
                    producer_obj = pins.read_json(producer["path"], producer.get("sha256"))
                    pins.verify(path.parent / "EXIT_CODE")
                    data, actual = runtime.read_feature_cache(path, known_sha=cache["sha256"])
                    need(actual == cache["sha256"] and data["manifest"]["cohort"] == cohort and
                         data["manifest"]["recall"] == pool, "cache_identity")
                    need(all(cache["manifest"].get(k) == v and producer_obj.get(k) == v
                             for k, v in data["manifest"].items()), "cache_manifest_exact")
                    need(len(data["smiles"]) == 114982 and
                         hashlib.sha256("\n".join(data["smiles"]).encode()).hexdigest() ==
                         bindings["candidate_order_sha256"], "candidate_universe")
                    if groups is None:
                        groups = [runtime.fixed.identity_source.identity(s)[1] for s in data["smiles"]]
                        group_sha = hashlib.sha256("\n".join(groups).encode()).hexdigest()
                        expected = bindings["roles"]["5"][str(seed)]["no_aug"]["qualification"]
                        need(group_sha == expected["derived_candidate_groups_sha256"], "candidate_groups")
                    for selected_record in selected:
                        key = identity(selected_record); ref = selected_record["cache_refs"][pool]
                        need(str(Path(ref["path"]).resolve()) == str(path), "selected_cache_ref_path")
                        row = data["rows"][ref["row_index"]]
                        need(identity(row) == key, "selected_cache_row_identity")
                        runtime.base.validate_row(row, cohort == "train")
                        fixed400_member = selected_record.get("fixed400_member", False)
                        q_external = key[-1] in heldout; q_internal = internal_group(key[-1])
                        fit = cohort == "train" and not q_external and not q_internal
                        need(selected_record["population"] != "fixed_fit" or fit, "fixed400_not_actual_fit")
                        need(selected_record["external_heldout"] == q_external and
                             selected_record["internal_group"] == q_internal, "query_qualification_changed")
                        need(selected_record["population"] != "internal" or
                             (not q_external and q_internal), "internal_query_qualification")
                        original_ids = row["candidate_indices"].cpu()
                        original_target = int(row["target_position"])
                        shape = shape_plan(row, fit, pool, fixed400_member)
                        natural_target = shape["natural_target_position"]
                        appended_loss = shape["extra_loss_forward"]
                        if fit:
                            need(0 <= original_target < len(original_ids) and
                                 groups[int(original_ids[original_target])] == key[-1], "fit_target")
                            need(int(original_ids[original_target]) == selected_record["target_candidate_index"],
                                 "fixed_target_candidate")
                        ids = original_ids[:512]
                        inp = scoring_input(row, False)
                        gamma = .75 if row["query_observation"].get("windows") or row["query_observation"].get("partial_peaks") else .25
                        def candidate_bits(values):
                            return dict(same_query_group=torch.tensor([groups[int(i)] == key[-1] for i in values]),
                                        external_heldout=torch.tensor([groups[int(i)] in heldout for i in values]),
                                        internal_group=torch.tensor([internal_group(groups[int(i)]) for i in values]))
                        full_bits = candidate_bits(original_ids)
                        full_excluded = full_bits["same_query_group"] | full_bits["external_heldout"] | full_bits["internal_group"]
                        bits = {k: v[:512] for k,v in full_bits.items()}
                        excluded = full_excluded[:512]
                        if fit:
                            need(torch.equal(row["negative_exclusion_mask"].cpu(),
                                             full_bits["same_query_group"] | full_bits["external_heldout"]),
                                 "cache_historical_mask_semantics")
                        traces, loss_traces, caps = {}, {}, {}
                        for cap in (5, 10):
                            corrections = []
                            for role, weight in (("no_aug", 1 - gamma), ("aug", gamma)):
                                model, scale, saved = models[(cap, seed, role)]
                                runtime.base.check_science(saved, data)
                                need(natural_forwards < 7040 and natural_forwards + loss_forwards < 8640,
                                     "natural_role_row_forward_budget")
                                tr = capture_role(model, scale, saved, inp, args.device); natural_forwards += 1
                                tr.update(shape_kind="natural512_inference", inference_role_weight=weight,
                                          actual_train_loss=None, actual_train_loss_status="not_training_query")
                                if fit and not appended_loss:
                                    tr["actual_train_loss"] = analytical_loss(
                                        inp["coarse_scores"], tr["delta"].detach(), tr["logit"].detach(),
                                        scale[0], natural_target, excluded, cap)
                                    tr["actual_train_loss_status"] = "fit_current_frozen_loss_reuses_natural512"
                                elif appended_loss:
                                    tr["actual_train_loss_status"] = "separate_loss_only_full513_shape"
                                    need(loss_forwards < 1600 and natural_forwards + loss_forwards < 8640,
                                         "loss_only_role_row_forward_budget")
                                    loss_inp = scoring_input(row, True)
                                    lt = capture_role(model, scale, saved, loss_inp, args.device); loss_forwards += 1
                                    lt.update(shape_kind="full513_loss_only", inference_role_weight=None,
                                              actual_train_loss_status="fit_current_frozen_loss_full513")
                                    lt["actual_train_loss"] = analytical_loss(
                                        loss_inp["coarse_scores"], lt["delta"].detach(), lt["logit"].detach(),
                                        scale[0], original_target, full_excluded, cap)
                                    loss_traces[(cap, role)] = lt
                                traces[(cap, role)] = tr
                                corrections.append(tr["delta"].detach() * scale[0])
                            cost = inp["coarse_scores"].to(corrections[0].device) + (1 - gamma) * corrections[0] + gamma * corrections[1]
                            caps[cap] = dict(natural_cost=cost.detach(), top10_positions=native_top10(ids, cost),
                                             dtypes=dict(coarse_dtype=str(inp["coarse_scores"].dtype),
                                                         correction_dtypes=[str(x.dtype) for x in corrections],
                                                         final_cost_dtype=str(cost.dtype)))
                        jac_positions = {p for cap in caps.values() for p in cap["top10_positions"]}
                        if natural_target >= 0:
                            jac_positions.add(natural_target)
                        old_other = selected_record.get("original_other_candidate_index")
                        if old_other is not None:
                            jac_positions.update((ids == old_other).nonzero().flatten().tolist())
                        role_payload, loss_payload = {}, {}
                        for cap in (5, 10):
                            role_payload[cap] = {}; loss_payload[cap] = {}
                            for role in ("no_aug", "aug"):
                                tr = traces[(cap, role)]
                                tr["input_jacobians"] = selected_jacobians(
                                    tr, list(jac_positions), tr["inference_role_weight"], tr["scale"][0], cap)
                                tr["trace_sha256_natural512"] = tensor_trace_hashes(tr)
                                role_payload[cap][role] = detached(tr)
                                if appended_loss:
                                    lt = loss_traces[(cap, role)]
                                    lp = sorted(jac_positions | {original_target})
                                    lj = selected_jacobians(lt, lp, 1., lt["scale"][0], cap)
                                    lj["weighted_dcost_dinput36"] = None
                                    lj["gradient_semantics"] = "loss_shape_local_jacobian_not_inference"
                                    lt["input_jacobians"] = lj
                                    lt["trace_sha256_loss513"] = tensor_trace_hashes(lt, natural_only=False)
                                    loss_payload[cap][role] = detached(lt)
                        check_reference(selected_record, pool, caps, role_payload, ids, w, pins)
                        shard = dict(schema=SCHEMA + "_tensor_row", seed=seed, cohort=cohort,
                                     population=selected_record["population"], query_identity=identity_dict(key),
                                     pool=pool, kind=row["kind"], fixed400_member=fixed400_member,
                                     original_bound_obstructed=selected_record.get("original_bound_obstructed"),
                                     cached_outside_pool_rank=row.get("outside_pool_rank"),
                                     mask_content_sha256=row.get("mask_content_sha256"),
                                     observation_semantics=row["observation_input"].get("semantics"),
                                     public_coverage=("explicit_window" if row["query_observation"].get("windows")
                                                      else "declared_partial" if row["query_observation"].get("partial_peaks")
                                                      else "unknown"),
                                     public_observation=dict(windows=row["query_observation"].get("windows"),
                                                             partial_peaks=row["query_observation"].get("partial_peaks")),
                                     structural_evidence=structural_evidence(row,selected_record,ids,caps,
                                                         data["smiles"],runtime.fixed.identity_source.identity),
                                     candidate_indices=ids, candidate_groups=[groups[int(i)] for i in ids],
                                     candidate_smiles=[data["smiles"][int(i)] for i in ids],
                                     coarse_cost=inp["coarse_scores"].cpu(), target_position=natural_target,
                                     target_candidate_id=selected_record.get("target_candidate_index",
                                                              int(ids[natural_target]) if natural_target>=0 else None),
                                     target_natural=natural_target>=0, target_input_status="present" if natural_target>=0 else "null_natural_miss",
                                     cache_loss_only_positive_present=len(original_ids)==513,
                                     diagnostic_loss_only_positive_used=appended_loss,
                                     loss_target_position=original_target if fit else None,
                                     loss_candidate_indices=original_ids if appended_loss else None,
                                     loss_candidate_groups=[groups[int(i)] for i in original_ids] if appended_loss else None,
                                     loss_candidate_smiles=[data["smiles"][int(i)] for i in original_ids] if appended_loss else None,
                                     loss_coarse_cost=row["coarse_scores"].cpu() if appended_loss else None,
                                     loss_strict_excluded=full_excluded if appended_loss else None,
                                     actual_train_query=fit, query_external_heldout=q_external,
                                     query_internal_group=q_internal, candidate_qualification_bits=bits,
                                     strict_excluded=excluded, gamma=gamma, original_other_candidate_index=old_other,
                                     actual_historical_exposure=None, actual_historical_exposure_status="unknown",
                                     caps=detached(caps), roles=role_payload,
                                     loss_roles=loss_payload if appended_loss else None)
                        filename = "tensor_shards/" + f"{shard_number:06d}.pt"
                        destination = out / filename; torch.save(shard, destination)
                        written += destination.stat().st_size; shard_number += 1
                        need(written <= w.get("max_artifact_bytes", 2 * 1024**3), "artifact_byte_budget")
                        receipt = dict(seed=seed, cohort=cohort, population=selected_record["population"],
                                       query_identity=identity_dict(key), pool=pool, tensor_file=filename,
                                       tensor_sha256=digest_file(destination), natural_role_row_forwards=4,
                                       loss_only_role_row_forwards=4 if appended_loss else 0,
                                       role_row_forwards=8 if appended_loss else 4,
                                       raw_candidates=512, target_position=natural_target,
                                       loss_candidates=513 if appended_loss else None,
                                       fixed400_member=fixed400_member, actual_train_query=fit,
                                       query_external_heldout=q_external, query_internal_group=q_internal,
                                       input_gradient_positions=sorted(jac_positions))
                        index.write(json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n"); index.flush()
                        seen.add((seed, cohort, key, pool))
                        need(time.monotonic() - started <= w.get("max_seconds", 3600), "wall_budget")
                        import resource
                        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        peak_bytes = int(peak) if sys.platform=="darwin" else int(peak)*1024
                        need(peak_bytes <= w.get("max_rss_bytes", 32*1024**3), "RSS_budget")
                        need(torch.cuda.max_memory_allocated(args.device) <= w["max_gpu_allocated_bytes"],
                             "GPU_allocated_budget")
                        del traces, loss_traces, role_payload, loss_payload, shard
                    del data
    need(len(seen) == len(records)*2 and natural_forwards == len(records)*8 and
         loss_forwards <= 1600 and natural_forwards+loss_forwards <= 8640, "complete_frozen_replay")
    for key, (model, scale, _) in models.items():
        need(module.tensor_hashes(model, scale) == before[key] and
             all(not p.requires_grad and p.grad is None for p in model.parameters()), "model_state_mutated")
    pins.finish()
    summary = dict(schema=SCHEMA, status="PASS", query_seed_pairs=len(records),
                   role_row_forwards=natural_forwards+loss_forwards,
                   natural_role_row_forwards=natural_forwards, loss_only_role_row_forwards=loss_forwards,
                   natural512_inference_role_row_forwards=natural_forwards,
                   full513_loss_role_row_forwards=loss_forwards,
                   model_loads=8, model_state_unchanged=True, optimizer_instances=0, optimizer_steps=0,
                   weight_updates=0, new_spectrum_forwards=0, forward_bank_calls=0,
                   external_api_calls=0, test_read=False, inputs_unchanged=True,
                   input_sha256=pins.checked, artifact_bytes=written, row_shards=shard_number,
                   measured_runtime=cuda_runtime_snapshot(torch), device=args.device,
                   cpu_threads=torch.get_num_threads(), gpu_name=torch.cuda.get_device_name(args.device),
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                   peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                   max_gpu_allocated_bytes=w["max_gpu_allocated_bytes"],
                   elapsed_seconds=time.monotonic()-started, protocol_sha256=args.protocol_sha256,
                   admission=admission)
    atomic_json(out/"summary.json", summary); (out/"EXIT_CODE").write_text("0\n")
    return summary

_REFERENCE_CACHE = {}
def check_reference(record, pool, caps, roles, ids, worker, pins):
    """只读冻结旧预测中选中行；缺参考的 train single/internal 明确登记而不伪造。"""
    import torch
    for cap in (5, 10):
        if record["population"] == "internal":
            caps[cap]["reference_status"] = "unavailable_registered_scope"
            continue
        ref_key = f"{cap}:{record['seed']}:{record['cohort']}:{pool}"
        info = worker.get("reference_predictions", {}).get(ref_key)
        if info is None:
            need(record["population"] == "internal" or
                 (record["population"] == "fixed_fit" and pool == "single"), "required_reference_missing")
            caps[cap]["reference_status"] = "unavailable_registered_scope"
            continue
        path = pins.verify(info["path"], info["sha256"])
        if ref_key not in _REFERENCE_CACHE:
            requested = {identity(r) for r in worker["_selected_records"]
                         if r["seed"] == record["seed"] and r["cohort"] == record["cohort"]}
            table = {}
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["query_identity"]["variant"] != "observed":
                        continue
                    key = identity(row)
                    if key in requested:
                        need(key not in table, "duplicate_reference_row")
                        table[key] = {k: row[k] for k in ("natural_candidate_indices", "natural_candidate_costs",
                                 "natural_candidate_coarse_costs", "natural_candidate_trace_sha256",
                                 "coarse_dtype", "final_cost_dtype", "correction_dtypes")}
            _REFERENCE_CACHE[ref_key] = table
        ref = _REFERENCE_CACHE[ref_key][identity(record)]
        need(ref["natural_candidate_indices"] == ids[:512].tolist(), "replay_candidate_order")
        need(ref["final_cost_dtype"] == caps[cap]["dtypes"]["final_cost_dtype"] and
             ref["coarse_dtype"] == caps[cap]["dtypes"]["coarse_dtype"] and
             ref["correction_dtypes"] == caps[cap]["dtypes"]["correction_dtypes"], "replay_native_dtypes")
        dtype = getattr(torch, ref["final_cost_dtype"].split(".")[-1])
        expected = torch.tensor(ref["natural_candidate_costs"], dtype=dtype)
        need(torch.equal(caps[cap]["natural_cost"].cpu(), expected), "replay_native_cost_mismatch")
        need(ref["natural_candidate_trace_sha256"] ==
             [roles[cap][role]["trace_sha256_natural512"] for role in ("no_aug", "aug")],
             "replay_role_trace_mismatch")
        caps[cap]["reference_status"] = "exact_native_cost_and_trace_match"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("identities", "replay"))
    parser.add_argument("--protocol", required=True); parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--bindings")
    parser.add_argument("--query-manifest"); parser.add_argument("--admission")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "identities":
        need(not args.execute and not args.bindings and not args.query_manifest, "metadata_only_arguments")
        result = identities_main(args)
    else:
        need(args.bindings and args.query_manifest, "replay_input_arguments")
        # Immutable selection is made available only for selective reference JSONL projection.
        try:
            result = run_replay(args)
        except Exception:
            if getattr(args, "_created_out", False):
                (Path(args.out)/"EXIT_CODE").write_text("1\n")
            raise
    print(json.dumps({k: result[k] for k in ("schema", "query_seed_pairs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""功能概述：Top10封存产物的零新NN科学解码/代数聚合器，在独审PASS和root冻结request后才打开科学产物。
本描述性副本保留9513封存源，复用nullable8d4和math71b定义；池外排名None保留unknown，loss标量按保存dtype重建原式；先核原NN0/1失败、独立闭包0/0和原规划明确授权，再解码封存shard。
validate_completed_provenance分开字节闭包、原失败和本次科学授权；validate_saved_shard只以保存tensor核shape/有限性/资格/mask/route/native成本与loss解析式，不调用模型。
非线性检查只沿原固定容差；CPU Ctanh解释不能冒充CUDA位值。原RSS缺口和模型状态source-bound边界保留。
loss_context保留非训练loss目标None；worker_facts仅以自然目标位置描述候选点，不创建实际loss。
实际fit要求有效整数loss目标。自然miss继续保留outside rank，不追加目标；其余数学、NN、
输入校验、输出、独审和执行边界与原冻结实现相同。
load_legacy复用20个允许predictions的完整observed成本；load_worker核shard字节SHA后消费冻结
worker输出，严格核自然512候选顺序、native成本位值、dtype和两role trace SHA，不做新前向。
worker_facts生成真实双role主桶、loss/资格/输入碰撞、匹配误差弱代理；build_references仅从原
固定400中fit资格样本的已有36输入预hash取参考，attach_nearest做同seed/cap/role/pool/kind
的out-of-query-group近邻。summarize和crosscounts分开fixed400/old87/C10旧pair释放、internal、
external full及hash探针，并逐seed/cohort/source给known/unknown/not_applicable分母。
run_analysis保留历史曝光unknown，核所有输入始末SHA，写本诊断目录下的新JSON/JSONL；
CLI缺--execute或精确独审凭证时失败，不加载真实tensor。无模型import、训练、干预选择、
API或谱forward；source/synthetic测试可直接调用纯函数，不构成真实执行授权。
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import time

import top10_barrier_math as bm

SCHEMA = "top10_barrier_analysis_v1_20261025"
IDENTITY = ("query_order", "variant", "source", "record_id", "smiles", "group")
ROLES = ("no_aug", "aug")
COHORTS = ("train", "original_val", "tune")
NATURAL = 512
EXECUTION_SCOPE = "CLOSED_OUTPUT_SAVED_TENSOR_SCIENTIFIC_ANALYSIS_ONLY"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(row):
    value = row.get("query_identity", row)
    bm.require(all(k in value for k in IDENTITY), "complete_query_identity")
    return tuple(value[k] for k in IDENTITY)


def key(seed, cap, pool, row):
    return (int(seed), int(cap), pool, identity(row))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def safe_path(path):
    p = Path(path).resolve()
    bm.require(not any(v in ("test", "sealed") or v.startswith("sealed-") for v in p.parts),
               "forbidden_scientific_path")
    bm.require(p.name not in ("research_report.md", "supplementary_report.md", "cohorts.pt"),
               "forbidden_payload")
    return p


class Pins:
    """白名单来自request的逐文件SHA；后续shard只接受已pin index给出的相对路径和SHA。"""
    def __init__(self):
        self.checked = {}

    def verify(self, path, digest):
        p = safe_path(path)
        bm.require(type(digest) is str and len(digest) == 64, "expected_sha_required")
        actual = file_sha(p)
        bm.require(actual == digest, "input_sha_mismatch:" + str(p))
        bm.require(str(p) not in self.checked or self.checked[str(p)] == actual, "input_changed")
        self.checked[str(p)] = actual
        return p

    def json(self, ref):
        p = self.verify(ref["path"], ref["sha256"])
        return json.loads(p.read_text())

    def finish(self):
        for path, digest in self.checked.items():
            bm.require(file_sha(path) == digest, "input_changed_at_finish:" + path)


def admit(request_path, request_sha256, admission_path, admission_sha256, execute):
    bm.require(execute, "explicit_execute_required")
    pins = Pins()
    admission = pins.json(dict(path=admission_path, sha256=admission_sha256))
    own = Path(__file__).resolve()
    math_path = own.with_name("top10_barrier_math.py")
    bm.require(admission.get("status") == "PASS" and admission.get("independent") is True and
               admission.get("execution_authorized") is True and
               admission.get("request_sha256") == request_sha256 and
               admission.get("execution_scope") == EXECUTION_SCOPE, "independent_PASS_required")
    for p in (own, math_path):
        bm.require(admission.get("reviewed_source_sha256", {}).get(str(p)) == file_sha(p),
                   "review_exact_source_required:" + str(p))
    request = pins.json(dict(path=request_path, sha256=request_sha256))
    bm.require(request.get("schema") == SCHEMA + "_request" and
               request.get("execution_authorized") is True and
               request.get("training") is False and request.get("test_read") is False and
               request.get("execution_scope") == EXECUTION_SCOPE and
               type(request.get("new_NN_budget")) is int and request["new_NN_budget"] == 0,
               "frozen_analysis_request")
    for name in (own.name, "top10_barrier_math.py", "prepare_closed_output_analysis_request_nullable_rank.py"):
        source=own.with_name(name)
        pins.verify(source,request["analysis_source_sha256"][str(source)])
    return request, admission, pins


def dtypes_from_prediction(row):
    return {k: row[k] for k in ("coarse_dtype", "correction_dtypes", "final_cost_dtype")}


def legacy_fact(row, spec, old):
    bm.require(row["query_identity"]["variant"] == "observed", "observed_only")
    cap, seed, pool, cohort = spec["cap"], spec["seed"], spec["pool"], spec["cohort"]
    ids, coarse, costs = (row[k] for k in
                          ("natural_candidate_indices", "natural_candidate_coarse_costs",
                           "natural_candidate_costs"))
    bm.require(len(ids) == len(coarse) == len(costs) == NATURAL, "legacy_natural512")
    dtypes = dtypes_from_prediction(row)
    scales = [v / cap for v in row["role_absolute_correction_bounds"]]
    target_id = row.get("target_candidate_index")
    diag = bm.diagnose_costs(ids, coarse, costs, target_id, row["gamma"], scales, cap, dtypes,
                            old_other_id=old.get("original_other_candidate_index") if old else None,
                            outside_rank=row.get("cached_outside_pool_rank"))
    bm.require(diag["rank"] == row["rank"] and diag["target_in_pool"] == row["target_in_pool"],
               "legacy_native_rank_recomputed")
    if old and target_id in ids and old.get("original_other_candidate_index") in ids:
        oldpos = ids.index(old["original_other_candidate_index"])
        bm.require(abs((coarse[ids.index(target_id)]-coarse[oldpos])-old["original_coarse_gap"]) < 2e-5,
                   "immutable_old_coarse_gap")
    return dict(seed=seed, cap=cap, pool=pool, cohort=cohort, query_identity=row["query_identity"],
                source=row["query_identity"]["source"], population_source="fixed400" if old else cohort,
                observation_semantics=row.get("observation_semantics"), public_coverage=row.get("public_coverage"),
                positive_membership="natural512_positive" if row["target_in_pool"] else "natural_miss_no_loss_append",
                fixed400_member=old is not None, original87=bool(old and old["original_bound_obstructed"]),
                old_manifest=old, gamma=row["gamma"], dtypes=dtypes, role_scale0=scales,
                cost_fingerprint=bm.cost_fingerprint(ids, coarse, costs, dtypes,
                                      row["natural_candidate_trace_sha256"]),
                cost_only_fingerprint=bm.cost_fingerprint(ids, coarse, costs, dtypes),
                trace_sha256=row["natural_candidate_trace_sha256"], diagnostic=diag,
                worker_probe=False, query_qualification="unknown" if cohort == "train" else "external",
                feature_evidence=None, loss_evidence=None, nn_evidence=None,
                structural_evidence=None, forward_matching_proxy=None,
                actual_historical_exposure=None, actual_historical_exposure_status="unknown",
                missing_measurement_mask_status="unknown", true_forward_RMSE_status="unknown")


def load_legacy(request, old_map, pins):
    specs = request["legacy_predictions"]
    bm.require(len(specs) == 20, "exact20_legacy_prediction_files")
    expected_specs = {(c, s, p, q) for c in (5, 10) for s in (193, 211)
                      for p in ("single", "dual") for q in ("original_val", "tune")}
    expected_specs |= {(c, s, "dual", "train") for c in (5, 10) for s in (193, 211)}
    bm.require({(z["cap"], z["seed"], z["pool"], z["cohort"]) for z in specs} == expected_specs,
               "legacy_matrix")
    result = {}
    for spec in specs:
        path = pins.verify(spec["path"], spec["sha256"])
        count = 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["query_identity"]["variant"] != "observed":
                    continue
                ikey = (spec["seed"], identity(row))
                old = old_map.get(ikey) if spec["cohort"] == "train" else None
                bm.require(spec["cohort"] != "train" or old is not None, "legacy_train_fixed400_only")
                k = key(spec["seed"], spec["cap"], spec["pool"], row)
                bm.require(k not in result, "duplicate_legacy_identity")
                result[k] = legacy_fact(row, spec, old)
                count += 1
        expected = 200 if spec["cohort"] == "train" else {"original_val": 320, "tune": 1024}[spec["cohort"]]
        bm.require(count == expected, "legacy_complete_observed_population")
    return result


def _list(value):
    return value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)


def _scalar(value):
    return value.detach().cpu().item() if hasattr(value, "detach") else value


def _cap(value, cap):
    return value[cap] if cap in value else value[str(cap)]



DECISION_SHA256 = "9754d2558ae890a898267d5c4e287aa0ce1d188c1555cbc10b6f3484ff177543"
DECISION_TEXT_SHA256 = "465394a644c78c7a588d1a3b237fa4a13c4aa4f3708bf4665d0f9886d39bf013"
CLOSURE_REVIEW_SHA256 = "57fa4c9a5dfc083a46c0d51f3e7743dede1a948558eadbcb3fde5306cf6b8d6e"
FIXED_RTOL, FIXED_ATOL = 2e-5, 2e-6


def exact_exit(obj, key, expected):
    bm.require(type(obj.get(key)) is int and obj[key] == expected, "exact_int_exit:" + key)


def validate_completed_provenance(request, pins):
    """只读封存闭包/授权元数据；原NN0/1、闭包0/0和本次分析授权分列。"""
    refs = request["completed_output_provenance"]
    bm.require(refs["original_decision"]["sha256"] == DECISION_SHA256 and
               refs["original_decision_text"]["sha256"] == DECISION_TEXT_SHA256 and
               refs["closure_review"]["sha256"] == CLOSURE_REVIEW_SHA256, "sole_original_decision_and_closure")
    decision = pins.json(refs["original_decision"])
    pins.verify(refs["original_decision_text"]["path"], refs["original_decision_text"]["sha256"])
    bm.require(decision["schema"] == "original_planning_scientific_decoding_decision_v1" and
               decision["status"] == "AUTHORIZED_CLOSED_SAVED_TENSOR_SCIENTIFIC_ANALYSIS" and
               decision["decision_owner"] == "original_planning_session" and
               decision["decision_text_path"] == refs["original_decision_text"]["path"] and
               decision["decision_text_sha256"] == DECISION_TEXT_SHA256 and
               decision["closure_independent_review_sha256"] == CLOSURE_REVIEW_SHA256,
               "original_decision_scope")
    for name in ("saved_tensor_scientific_decode_authorized", "saved_tensor_algebra_statistics_authorized"):
        bm.require(decision[name] is True, "decision_requires:" + name)
    for name in ("new_model_forward_authorized", "training_authorized", "test_read_authorized",
                 "external_API_authorized", "formal_system_changed", "additional_sample_authorized",
                 "new_definition_authorized"):
        bm.require(decision[name] is False, "decision_forbids:" + name)
    bm.require(type(decision["new_NN_budget"]) is int and decision["new_NN_budget"] == 0 and
               decision["query_seed_pairs"] == 826 and decision["pool_count"] == 2 and
               decision["source_NN_stage_status"] == "FAIL_SUPERVISION" and
               decision["source_RSS_gap_status"] == "UNKNOWN_NOT_CERTIFIED" and
               decision["model_state_assertion_scope"] == "SOURCE_BOUND_NOT_INDEPENDENTLY_REMEASURED",
               "decision_fixed_population_and_source_boundary")
    exact_exit(decision, "source_actual_inner_exit", 0); exact_exit(decision, "source_actual_external_exit", 1)
    source = pins.json(refs["source_execution"])
    closure_execution = pins.json(refs["closure_execution"])
    closure = pins.json(refs["closure_report"])
    review = pins.json(refs["closure_review"])
    sealed = pins.json(refs["sealed_manifest"])
    exact_exit(source, "actual_inner_exit", 0); exact_exit(source, "actual_external_exit", 1)
    bm.require(source["termination_reason"] == "rss_monitor_unavailable" and
               source.get("rss_missing_while_alive") is True, "source_NN_FAILURE_preserved")
    exact_exit(closure_execution, "actual_inner_exit", 0); exact_exit(closure_execution, "actual_external_exit", 0)
    bm.require(closure_execution["termination_reason"] is None, "closure_execution_termination")
    bm.require(closure["schema"] == "failed_supervision_output_closure_v1" and
               closure["status"] == "PASS_RAW_OUTPUT_CLOSURE" and
               closure["execution_scope"] == "FAILED_SUPERVISION_OUTPUT_CLOSURE_ONLY" and
               closure["original_stage_status"] == "FAIL_SUPERVISION" and
               closure["original_rss_gap_status"] == "unknown_not_certified" and
               closure["original_resource_envelope_recertified"] is False and
               closure["raw_shards_hashed"] == 1652 and closure["closed_output_files"] == 1661,
               "closed_byte_package_not_NN_success")
    exact_exit(closure, "original_actual_inner_exit", 0); exact_exit(closure, "original_actual_external_exit", 1)
    bm.require(review["status"] == "PASS_RAW_CLOSURE_REVIEW" and review["independent"] is True and
               review["execution_authorized"] is False and review["scientific_execution_authorized"] is False and
               review["source_stage_status"] == "FAIL_SUPERVISION_PRESERVED" and
               review["original_RSS_gap_status"] == "UNKNOWN_NOT_CERTIFIED" and
               review["original_resource_envelope_recertified"] is False, "independent_closure_scope")
    for name in ("closure_actual_inner_exit", "closure_actual_external_exit", "closure_actual_ssh_exit"):
        exact_exit(review, name, 0)
    exact_exit(review, "source_stage_actual_inner_exit", 0); exact_exit(review, "source_stage_actual_external_exit", 1)
    for name in ("closure_execution", "closure_report"):
        ref = refs[name]
        bm.require(review["evidence_sha256"].get(ref["path"]) == ref["sha256"], "reviewed_closure_ref:" + name)
    observed = review["closure_resource_observations"]
    bm.require(all(closure_execution[k] == observed[k] for k in
                   ("job_sha256", "review_sha256", "actual_inner_exit", "actual_external_exit",
                    "termination_reason")), "closure_execution_independent_binding")
    bm.require(source["job_sha256"] == closure["source_refs"]["job"]["sha256"] and
               source["review_sha256"] == closure["source_refs"]["replay_admission"]["sha256"] and
               request["query_manifest"]["sha256"] == closure["source_refs"]["query_manifest"]["sha256"] and
               refs["sealed_manifest"]["sha256"] == closure["source_refs"]["remote_manifest"]["sha256"],
               "source_and_manifest_closure_binding")
    files = sealed["files"]
    bm.require(len(files) == 1661 and
               sum(k.startswith("worker/tensor_shards/") for k in files) == 1652,
               "sealed_manifest_population")
    stage = safe_path(refs["source_stage_root"])
    bm.require(refs["source_execution"]["path"] == str(stage / "actual_execution.json") and
               refs["source_execution"]["sha256"] == files["actual_execution.json"]["sha256"] and
               len(request["worker_outputs"]) == 1, "sole_sealed_worker_stage")
    output = request["worker_outputs"][0]
    for name, relative in (("summary", "worker/summary.json"), ("index", "worker/rows.jsonl")):
        bm.require(output[name]["path"] == str(stage / relative) and
                   output[name]["sha256"] == files[relative]["sha256"], "sealed_worker_output:" + name)
    # 只投影历史训练排除groups，不访问contract中的其他cohort/结果字段。
    groups = pins.json(request["qualification_contract"])["training_excluded_groups"]
    binding = pins.json(request["qualification_bindings"])
    bm.require(type(groups) is list and len(groups) == len(set(groups)) and
               all(type(g) is str for g in groups), "heldout_group_metadata")
    group_sha = hashlib.sha256("\n".join(sorted(groups)).encode()).hexdigest()
    bm.require(group_sha == binding["heldout_groups_sha256"] and
               request["qualification_contract"]["sha256"] in closure["checked_input_sha256"].values() and
               request["qualification_bindings"]["sha256"] in closure["checked_input_sha256"].values(),
               "source_bound_heldout_metadata")
    request["_heldout_groups"] = frozenset(groups)
    request["_sealed_files"] = files
    return dict(original_decision=refs["original_decision"], closure_review=refs["closure_review"],
                source_NN_stage_status="FAIL_SUPERVISION", source_actual_inner_exit=0,
                source_actual_external_exit=1, source_RSS_gap_status="UNKNOWN_NOT_CERTIFIED",
                closure_actual_inner_exit=0, closure_actual_external_exit=0,
                original_resource_envelope_recertified=False,
                model_state_assertion_scope="SOURCE_BOUND_NOT_INDEPENDENTLY_REMEASURED",
                new_NN_calls=0, model_loads=0, scientific_scope=EXECUTION_SCOPE)


def internal_qualification(group):
    return int(hashlib.sha256(("ranker_internal_317" + str(group)).encode()).hexdigest()[:12], 16) % 5 == 0


def finite_present(value, path="shard"):
    """所有现存数值必须有限；None保留unknown，绝不nan_to_num或补字段。"""
    import torch
    if isinstance(value, torch.Tensor):
        bm.require(bool(torch.isfinite(value).all()), "nonfinite:" + path)
    elif type(value) in (float, int):
        bm.require(math.isfinite(value), "nonfinite:" + path)
    elif isinstance(value, dict):
        for k, v in value.items(): finite_present(v, path + "." + str(k))
    elif isinstance(value, (list, tuple)):
        for k, v in enumerate(value): finite_present(v, path + "." + str(k))


def check_tensor(value, shape, label, allowed_dtypes=None):
    import torch
    bm.require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape), "tensor_shape:" + label)
    if allowed_dtypes is not None:
        bm.require(value.dtype in allowed_dtypes, "tensor_dtype:" + label)
    bm.require(value.device.type == "cpu", "saved_tensor_CPU_only:" + label)


def tolerant(actual, expected, label):
    import torch
    bm.require(tuple(actual.shape) == tuple(expected.shape) and
               torch.allclose(actual.to(torch.float64), expected.to(torch.float64),
                              rtol=FIXED_RTOL, atol=FIXED_ATOL), "fixed_tolerance:" + label)


def validate_saved_loss(trace, coarse, strict, target, cap):
    """只验证保存的当前单role loss；沿已冻结容差核非线性与解析式，不反推历史梯度。"""
    import torch
    loss = trace["actual_train_loss"]; n = len(coarse)
    fields = ("actual_normalized_costs", "probability", "listwise_gradient_delta", "anchor_gradient_delta",
              "total_gradient_delta", "tanh_local_derivative", "listwise_gradient_logit",
              "anchor_gradient_logit", "total_gradient_logit")
    check_tensor(loss["allowed"], (n,), "loss.allowed", (torch.bool,))
    allowed = ~strict.clone(); allowed[target] = True
    bm.require(torch.equal(loss["allowed"], allowed) and int(allowed.sum()) >= 2 and
               type(loss["support_count"]) is int and loss["support_count"] == int(allowed.sum()),
               "actual_loss_exact_support")
    for name in fields: check_tensor(loss[name], (n,), "loss." + name, (torch.float32, torch.float64))
    expected_cost = coarse / trace["scale"][0] + trace["delta"]
    bm.require(loss["normalized_cost_dtype"] == str(loss["actual_normalized_costs"].dtype) ==
               str(expected_cost.dtype) and loss["probability_dtype"] == str(loss["probability"].dtype),
               "loss_actual_dtype")
    tolerant(loss["actual_normalized_costs"], expected_cost, "native_normalized_loss_cost")
    expected = bm.loss_probe(_list(coarse), _list(trace["delta"]), _scalar(trace["scale"][0]), target,
                            _list(allowed), _list(trace["logit"]), cap,
                            actual_costs=_list(loss["actual_normalized_costs"]))
    mappings = (("probability", "probabilities"), ("listwise_gradient_delta", "listwise_gradient_delta"),
                ("anchor_gradient_delta", "anchor_gradient_delta"), ("total_gradient_delta", "total_gradient_delta"),
                ("tanh_local_derivative", "tanh_derivative"), ("total_gradient_logit", "total_gradient_logit"))
    for name, other in mappings:
        tolerant(loss[name], torch.tensor(expected[other], dtype=torch.float64), name)
    tolerant(loss["listwise_gradient_logit"], loss["listwise_gradient_delta"] * loss["tanh_local_derivative"], "listwise_component_logit")
    tolerant(loss["anchor_gradient_logit"], loss["anchor_gradient_delta"] * loss["tanh_local_derivative"], "anchor_component_logit")
    for name in ("probability", "listwise_gradient_delta", "anchor_gradient_delta", "total_gradient_delta",
                 "listwise_gradient_logit", "anchor_gradient_logit", "total_gradient_logit"):
        bm.require(bool((loss[name][~allowed] == 0).all()), "masked_exact_zero:" + name)
    bm.require(bool((loss["probability"] >= 0).all()), "nonnegative_probability")
    # 原worker保存的是其原dtype、原运算次序的scalar。稳定FP64理想式在FP32大公共偏移下
    # 不与该scalar数值等价；这里仅用已保存cost/delta重建原式，容差仍完全沿用冻结值。
    saved_cost=loss["actual_normalized_costs"]
    native_listwise=torch.logsumexp(-saved_cost[allowed]/.5,0)+saved_cost[target]/.5
    native_anchor=.1*trace["delta"][allowed].square().mean()
    for name, number in (("listwise_loss",native_listwise),("anchor_loss",native_anchor),
                         ("total_loss",native_listwise+native_anchor)):
        bm.require(math.isclose(float(_scalar(loss[name])), float(_scalar(number)), rel_tol=FIXED_RTOL, abs_tol=FIXED_ATOL),
                   "saved_loss_scalar_native_dtype:" + name)


def saved_trace_hashes(trace):
    """保存tensor逐字段typed字节哈希；仅解码已有值，不重新计算网络输出。"""
    import numpy as np
    fields = dict(raw_X=trace["raw_X"], raw_O=trace["raw_O"],
                  centered_X=trace["input36"][:, :33], centered_O=trace["input36"][:, 33:],
                  logits=trace["logit"], delta=trace["delta"])
    arrays = [(k, v.detach().cpu().contiguous().numpy()) for k, v in sorted(fields.items())]
    result = []
    for i in range(len(trace["raw_X"])):
        h = hashlib.sha256()
        for k, arr in arrays:
            value = np.asarray(arr[i])
            h.update(k.encode()); h.update(str(value.dtype).encode()); h.update(str(value.shape).encode())
            h.update(value.tobytes())
        result.append(h.hexdigest())
    return result


def validate_saved_trace(trace, n, cap, coarse, weight, expected_positions):
    import torch
    floats = (torch.float32, torch.float64)
    for name, shape in (("raw_X", (n,33)), ("raw_O",(n,3)), ("input36",(n,36)),
                        ("logit",(n,)), ("delta",(n,)), ("scale",(33,)),
                        ("fixed_x_center",(33,)), ("fixed_o_center",(3,))):
        check_tensor(trace[name], shape, name, floats)
    bm.require(trace["shape_kind"] == ("natural512_inference" if n==512 else "full513_loss_only"), "saved_shape_kind")
    bm.require(trace["trace_sha256_natural512" if n==512 else "trace_sha256_loss513"] == saved_trace_hashes(trace), "saved_trace_typed_byte_hash")
    bm.require(bool((trace["scale"] > 0).all()) and bool((trace["delta"].abs() <= cap).all()), "scale_or_cap")
    x = ((trace["raw_X"]-trace["fixed_x_center"])/trace["scale"]).clamp(-20,20)
    o = (trace["raw_O"]-trace["fixed_o_center"]).clamp(-20,20)
    tolerant(trace["input36"], torch.cat((x,o),1), "saved_centered36")
    tolerant(trace["delta"], cap*torch.tanh(trace["logit"]/cap), "saved_Ctanh_analytic_only")
    jac = trace["input_jacobians"]; positions = jac["positions"]
    bm.require(type(positions) is list and positions == sorted(set(positions)) and
               all(type(p) is int and 0 <= p < n for p in positions) and
               set(positions) == set(expected_positions), "saved_jacobian_positions")
    for name in ("dlogit_dinput36", "ddelta_dinput36"):
        check_tensor(jac[name], (len(positions),36), name, floats)
    local = 1-torch.tanh(trace["logit"][positions]/cap).square()
    tolerant(jac["ddelta_dinput36"],jac["dlogit_dinput36"]*local[:,None], "saved_delta_Jacobian")
    if weight is None:
        bm.require(trace["inference_role_weight"] is None and jac["weighted_dcost_dinput36"] is None,
                   "loss_shape_is_not_inference")
    else:
        bm.require(trace["inference_role_weight"] == weight, "saved_role_weight")
        check_tensor(jac["weighted_dcost_dinput36"], (len(positions),36), "weighted_Jacobian", floats)
        tolerant(jac["weighted_dcost_dinput36"],jac["ddelta_dinput36"]*weight*trace["scale"][0],
                 "saved_weighted_cost_Jacobian")


def validate_saved_shard(shard, heldout, selected=None):
    """仅CPU保存tensor：检查真实shape/资格/成本闭包，不加载checkpoint或运行任何网络。"""
    import torch
    finite_present(shard)
    bm.require(shard["schema"] == "top10_frozen_worker_v1_tensor_row" and
               shard["seed"] in (193,211) and shard["cohort"] in COHORTS and
               shard["pool"] in ("single","dual") and type(shard["kind"]) is int and 0<=shard["kind"]<4,
               "saved_shard_identity")
    caps, roles = shard["caps"], shard["roles"]
    bm.require(set(map(int,caps)) == {5,10} and len(caps)==2 and
               set(map(int,roles)) == {5,10} and len(roles)==2, "exact_C5_C10_sets")
    ids, coarse = shard["candidate_indices"], shard["coarse_cost"]
    check_tensor(ids,(512,),"natural_ids",(torch.int32,torch.int64))
    check_tensor(coarse,(512,),"natural_coarse",(torch.float32,torch.float64))
    bm.require(len(ids.unique())==512 and bool(((ids>=0)&(ids<114982)).all()),"natural_id_values")
    groups = shard["candidate_groups"]; qgroup = identity(shard)[-1]
    bm.require(len(groups)==len(shard["candidate_smiles"])==512 and all(type(g)is str for g in groups),"candidate_metadata")
    q_external, q_internal = qgroup in heldout, internal_qualification(qgroup)
    actual = shard["cohort"]=="train" and not q_external and not q_internal
    bm.require(type(shard["actual_train_query"])is bool and shard["actual_train_query"]==actual and
               type(shard["query_external_heldout"])is bool and shard["query_external_heldout"]==q_external and
               type(shard["query_internal_group"])is bool and shard["query_internal_group"]==q_internal,"exact_query_eligibility")
    bits = dict(same_query_group=torch.tensor([g==qgroup for g in groups]),
                external_heldout=torch.tensor([g in heldout for g in groups]),
                internal_group=torch.tensor([internal_qualification(g) for g in groups]))
    bm.require(set(shard["candidate_qualification_bits"])==set(bits),"qualification_bits_set")
    for name, expected in bits.items():
        check_tensor(shard["candidate_qualification_bits"][name],(512,),name,(torch.bool,))
        bm.require(torch.equal(shard["candidate_qualification_bits"][name],expected),"qualification_bit:"+name)
    strict = bits["same_query_group"]|bits["external_heldout"]|bits["internal_group"]
    check_tensor(shard["strict_excluded"],(512,),"strict_excluded",(torch.bool,))
    bm.require(torch.equal(shard["strict_excluded"],strict),"strict_negative_mask")
    public = shard["public_observation"]
    gamma = .75 if public.get("windows") or public.get("partial_peaks") else .25
    bm.require(shard["gamma"] == gamma, "public_gamma_routing")
    target = shard["target_position"]
    bm.require(type(target)is int and -1<=target<512 and type(shard["target_natural"])is bool and
               shard["target_natural"]==(target>=0),"natural_target_contract")
    if target>=0:
        bm.require(groups[target]==qgroup,"natural_target_group")
        if shard.get("target_candidate_id") is not None:
            bm.require(int(ids[target])==shard["target_candidate_id"],"natural_target_ID")
    else:
        outside=shard["cached_outside_pool_rank"]
        bm.require(outside is None or (type(outside)is int and 512<outside<=114982),
                   "natural_miss_optional_outside_rank")
    if selected is not None:
        bm.require(identity(shard)==identity(selected) and shard["seed"]==selected["seed"] and
                   shard["cohort"]==selected["cohort"] and shard["population"]==selected["population"] and
                   shard["fixed400_member"]==selected["_fixed400_member"] and
                   shard["original_bound_obstructed"]==selected.get("original_bound_obstructed") and
                   shard.get("original_other_candidate_index")==selected.get("original_other_candidate_index"),"selected_saved_identity")
        if selected.get("target_candidate_index") is not None and actual:
            actual_id = int(ids[target]) if target>=0 else int(shard["loss_candidate_indices"][512])
            bm.require(actual_id==selected["target_candidate_index"],"selected_true_target_ID")
    appended = bool(shard.get("loss_roles"))
    bm.require(type(shard["diagnostic_loss_only_positive_used"])is bool and
               shard["diagnostic_loss_only_positive_used"]==appended,"loss_shape_flag")
    if appended:
        bm.require(actual and shard["fixed400_member"] is True and shard["pool"]=="single" and target==-1 and
                   shard["cache_loss_only_positive_present"] is True and shard["loss_target_position"]==512,
                   "fit_single_full513_only")
        bm.require(set(map(int,shard["loss_roles"]))=={5,10} and len(shard["loss_roles"])==2,"loss_caps")
        lids, lcoarse = shard["loss_candidate_indices"],shard["loss_coarse_cost"]
        check_tensor(lids,(513,),"loss_ids",(torch.int32,torch.int64))
        check_tensor(lcoarse,(513,),"loss_coarse",(torch.float32,torch.float64))
        bm.require(lids.dtype==ids.dtype and lcoarse.dtype==coarse.dtype and
                   torch.equal(lids[:512],ids) and torch.equal(lcoarse[:512],coarse) and len(lids.unique())==513 and
                   bool(((lids>=0)&(lids<114982)).all()),
                   "full513_natural_prefix")
        lg = shard["loss_candidate_groups"]
        bm.require(len(lg)==len(shard["loss_candidate_smiles"])==513 and lg[:512]==groups and lg[512]==qgroup,"loss_target_group")
        lstrict = torch.tensor([g==qgroup or g in heldout or internal_qualification(g) for g in lg])
        check_tensor(shard["loss_strict_excluded"],(513,),"loss_strict",(torch.bool,))
        bm.require(torch.equal(lstrict,shard["loss_strict_excluded"]),"loss_strict_mask")
    else:
        bm.require(not actual or (target>=0 and shard["loss_target_position"]==target), "fit_natural_target")
        bm.require(actual or shard["loss_target_position"] is None,"nontraining_loss_target_None")
        bm.require(all(shard.get(k) is None for k in
                       ("loss_candidate_indices","loss_coarse_cost","loss_strict_excluded")),"no_optional_loss_append")
    important = set()
    for cap in (5,10):
        obj, rr = _cap(caps,cap),_cap(roles,cap)
        bm.require(set(rr)==set(ROLES),"exact_two_roles")
        cost = obj["natural_cost"]
        check_tensor(cost,(512,),"native_cost",(torch.float32,torch.float64))
        order=sorted(range(512),key=lambda p:(float(cost[p]),int(ids[p])))[:10]
        bm.require(obj["top10_positions"]==order,"saved_top10_positions")
        important.update(order)
    if target>=0: important.add(target)
    old = shard.get("original_other_candidate_index")
    important.update(i for i,v in enumerate(_list(ids)) if v==old)
    for cap in (5,10):
        obj, rr = _cap(caps,cap),_cap(roles,cap)
        for role,w in zip(ROLES,(1-gamma,gamma)):
            tr=rr[role]
            validate_saved_trace(tr,512,cap,coarse,w,important)
            if actual and not appended: validate_saved_loss(tr,coarse,strict,target,cap)
            else: bm.require(tr["actual_train_loss"] is None,"no_actual_loss_on_inference_shape")
            if appended:
                lr=_cap(shard["loss_roles"],cap)
                bm.require(set(lr)==set(ROLES),"exact_loss_two_roles")
                validate_saved_trace(lr[role],513,cap,lcoarse,None,important|{512})
                bm.require(torch.equal(lr[role]["raw_X"][:512],tr["raw_X"]),"full513_rawcache_X_prefix")
                validate_saved_loss(lr[role],lcoarse,lstrict,512,cap)
        d=obj["dtypes"]
        actual_d=dict(coarse_dtype=str(coarse.dtype),
                      correction_dtypes=[str((rr[r]["delta"]*rr[r]["scale"][0]).dtype) for r in ROLES],
                      final_cost_dtype=str(obj["natural_cost"].dtype))
        bm.require(d==actual_d,"saved_actual_cost_dtypes")
        replay = [bm.native_cost_replay(b,a,z,_scalar(rr["no_aug"]["scale"][0]),
                   _scalar(rr["aug"]["scale"][0]),gamma,d["coarse_dtype"],d["correction_dtypes"])
                  for b,a,z in zip(_list(coarse),_list(rr["no_aug"]["delta"]),_list(rr["aug"]["delta"]))]
        bm.require(bm.native_array_digest(replay,d["final_cost_dtype"]) ==
                   bm.native_array_digest(_list(obj["natural_cost"]),d["final_cost_dtype"]),
                   "all512_native_cost_replay_including_target_miss")
        bm.diagnose_costs(_list(ids),_list(coarse),_list(obj["natural_cost"]),
            int(ids[target]) if target>=0 else None,gamma,[_scalar(rr[r]["scale"][0])for r in ROLES],
            cap,d,role_deltas=[_list(rr[r]["delta"])for r in ROLES],
            old_other_id=old,outside_rank=shard.get("cached_outside_pool_rank"))


def worker_rows(request, pins):
    """torch只在admission后的真实run调用本generator时导入，binary仅来自已pin的index。"""
    import torch
    total_inference, total_loss = 0, 0
    for entry in request["worker_outputs"]:
        summary = pins.json(entry["summary"])
        bm.require(summary["status"] == "PASS" and summary["model_state_unchanged"] is True and
                   summary["inputs_unchanged"] is True and summary["test_read"] is False,
                   "worker_summary_integrity")
        for name in ("optimizer_instances", "optimizer_steps", "weight_updates",
                     "new_spectrum_forwards", "forward_bank_calls", "external_api_calls"):
            bm.require(summary[name] == 0, "worker_forbidden_operation")
        bm.require(summary["natural512_inference_role_row_forwards"] == summary["natural_role_row_forwards"] and
                   summary["full513_loss_role_row_forwards"] == summary["loss_only_role_row_forwards"] and
                   summary["role_row_forwards"] == summary["natural_role_row_forwards"] + summary["loss_only_role_row_forwards"],
                   "worker_shape_counter_aliases")
        total_inference += summary["natural512_inference_role_row_forwards"]
        total_loss += summary["full513_loss_role_row_forwards"]
        bm.require(total_inference <= 7040 and total_loss <= 1600 and total_inference + total_loss <= 8640,
                   "global_double_shape_forward_budget")
        index_path = pins.verify(entry["index"]["path"], entry["index"]["sha256"])
        with index_path.open() as stream:
            for line in stream:
                receipt = json.loads(line)
                relative = Path(receipt["tensor_file"])
                bm.require(not relative.is_absolute() and ".." not in relative.parts, "shard_relative_path")
                path = pins.verify(index_path.parent / relative, receipt["tensor_sha256"])
                shard = torch.load(path, map_location="cpu", weights_only=False)
                bm.require(identity(shard) == identity(receipt) and
                           all(shard[z] == receipt[z] for z in ("seed", "cohort", "pool", "population")),
                           "shard_receipt_identity")
                relative_sealed = "worker/" + str(relative)
                sealed_ref = request["_sealed_files"][relative_sealed]
                bm.require(receipt["tensor_sha256"] == sealed_ref["sha256"] and path.stat().st_size == sealed_ref["bytes"], "closed_shard_SHA_and_bytes")
                selected = request["_selection_by_identity"][(shard["seed"],shard["cohort"],identity(shard))]
                validate_saved_shard(shard, request["_heldout_groups"], selected)
                yield shard, receipt


def _native_arrays(shard, cap):
    ids = _list(shard["candidate_indices"])[:NATURAL]
    coarse = _list(shard["coarse_cost"])[:NATURAL]
    obj = _cap(shard["caps"], cap)
    costs = _list(obj["natural_cost"])
    roles = _cap(shard["roles"], cap)
    dtypes = obj.get("dtypes") or dict(coarse_dtype=str(shard["coarse_cost"].dtype),
        correction_dtypes=[str((roles[r]["delta"] * roles[r]["scale"][0]).dtype) for r in ROLES],
        final_cost_dtype=str(obj["natural_cost"].dtype))
    return ids, coarse, costs, roles, dtypes


class _MaxItem:
    def __init__(self, sortkey, record):
        self.sortkey, self.record = sortkey, record
    def __lt__(self, other):
        return self.sortkey > other.sortkey


def loss_context(shard, cap, role):
    """自然512推理与fit-only单池full513实际loss严格分读，绝不由追加正例重写推理目标。"""
    source = _cap(shard["loss_roles"], cap)[role] if shard.get("loss_roles") else _cap(shard["roles"], cap)[role]
    ids = _list(shard["loss_candidate_indices"]) if shard.get("loss_roles") else _list(shard["candidate_indices"])
    target = shard.get("loss_target_position", shard["target_position"])
    if shard["actual_train_query"]:
        bm.require(type(target) is int and 0 <= target < len(ids), "actual_loss_target_position_required")
    else:
        bm.require(target is None or (type(target) is int and -1 <= target < len(ids)),
                   "nontraining_nullable_loss_target")
    if shard.get("loss_roles"):
        bm.require(shard["actual_train_query"] and shard["pool"] == "single" and
                   len(ids) == 513 and target == 512 and shard["target_position"] == -1,
                   "separate_full513_fit_single_only")
    return source, ids, target


def build_references(request, old_map, pins):
    """只从固定400且实际fit的已重放support取预hash参考，不新增行、不按失败选样。"""
    references, dual_negatives, seen, counts = {}, {}, set(), collections.Counter()
    limit = request.get("reference_records_per_seed_cap_role_pool", 4096)
    bm.require(type(limit) is int and 0 < limit <= 4096, "reference_limit")
    for shard, _ in worker_rows(request, pins):
        seed, cohort, pool = shard["seed"], shard["cohort"], shard["pool"]
        qid = identity(shard); skey = (seed, cohort, pool, qid)
        bm.require(skey not in seen, "duplicate_worker_row"); seen.add(skey)
        for cap in (5, 10):
            ids, coarse, costs, roles, dtypes = _native_arrays(shard, cap)
            target_pos = shard["target_position"]
            target_id = _list(shard["candidate_indices"])[target_pos] if target_pos >= 0 else None
            if pool == "dual" and target_id in ids:
                order = sorted(range(len(ids)), key=lambda p: (costs[p], ids[p]))
                dual_negatives[(seed, cap, qid)] = [ids[p] for p in order if ids[p] != target_id][:10]
        is_fixed = (seed, qid) in old_map
        if not is_fixed or not shard["actual_train_query"]:
            continue
        bm.require(not shard["query_internal_group"] and not shard["query_external_heldout"],
                   "actual_fit_qualification")
        all_ids = _list(shard["candidate_indices"])
        for cap in (5, 10):
            roles = _cap(shard["roles"], cap)
            for role in ROLES:
                trace, loss_ids, loss_target = loss_context(shard, cap, role)
                loss = trace["actual_train_loss"]
                bm.require(loss is not None, "fit_loss_required")
                allowed = _list(loss["allowed"])
                x36, raw33, rawO3 = map(_list, (trace["input36"], trace["raw_X"], trace["raw_O"]))
                kind = shard.get("kind")
                bm.require(type(kind) is int, "worker_kind_required")
                groupkey = (seed, cap, role, pool)
                heap = references.setdefault(groupkey, [])
                for pos, allowed_here in enumerate(allowed):
                    if not allowed_here:
                        continue
                    payload = ["top10-barrier-training-input-reference-v1", seed, cap, role,
                               pool, list(qid), loss_ids[pos]]
                    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                                       separators=(",", ":")).encode()).hexdigest()
                    sortkey = (digest, qid, loss_ids[pos])
                    if len(heap) >= limit and sortkey >= heap[0].sortkey:
                        continue
                    record = dict(reference_hash=digest, seed=seed, cap=cap, role=role, pool=pool,
                                  kind=kind, query_group=qid[-1], candidate_index=loss_ids[pos],
                                  training_label="target" if pos == loss_target else "eligible_negative",
                                  input36_dtype=str(trace["input36"].dtype),
                                  input36_hash=bm.vector_hash(x36[pos], str(trace["input36"].dtype)),
                                  raw33_hash=bm.vector_hash(raw33[pos], str(trace["raw_X"].dtype)),
                                  rawO3_hash=bm.vector_hash(rawO3[pos], str(trace["raw_O"].dtype)),
                                  input_shape=len(loss_ids),
                                  centered36=x36[pos], raw33=raw33[pos], rawO3=rawO3[pos])
                    item = _MaxItem(sortkey, record)
                    if len(heap) < limit:
                        heapq.heappush(heap, item)
                    else:
                        heapq.heapreplace(heap, item)
                    counts[groupkey] += 1
    bm.require(len(seen) <= 1760, "worker_row_budget")
    result = {k: [item.record for item in sorted(v, key=lambda t: t.sortkey)] for k, v in references.items()}
    bm.require(sum(map(len, result.values())) <= 65536, "reference_vector_budget")
    return result, dual_negatives, seen


def critical_membership_table(shard, loss_ids, loss, critical, loss_target):
    natural_ids = _list(shard["candidate_indices"])
    rows = []
    for cid in critical:
        natural_pos = natural_ids.index(cid) if cid in natural_ids else None
        loss_pos = loss_ids.index(cid) if cid in loss_ids else None
        bits = {k:bool(_list(v)[natural_pos]) for k,v in shard["candidate_qualification_bits"].items()} if natural_pos is not None else None
        allowed = bool(_list(loss["allowed"])[loss_pos]) if loss is not None and loss_pos is not None else None
        if loss is None: reason="not_training_query"
        elif loss_pos is None: reason="absent_from_actual_loss_row"
        elif loss_pos==loss_target: reason="positive_override"
        elif allowed: reason="eligible_negative"
        else:
            reason="same_query_group" if bits and bits["same_query_group"] else "external_heldout" if bits and bits["external_heldout"] else "internal_group" if bits and bits["internal_group"] else "excluded_missing_bit_evidence"
        rows.append(dict(candidate_index=cid,in_natural=natural_pos is not None,
                         in_loss=loss_pos is not None if loss is not None else None,
                         allowed=allowed,exclusion_bits=bits,exclusive_reason=reason,
                         exclusive_reason_priority=["positive_override","same_query_group","external_heldout","internal_group"]))
    return rows


def worker_facts(shard, old_map, legacy, dual_negatives):
    """一个worker query/pool生成两cap事实；保留所有限定条件，不把梯度当历史优化记录。"""
    import numpy as np
    seed, pool, cohort = shard["seed"], shard["pool"], shard["cohort"]
    qid = identity(shard); old = old_map.get((seed, qid))
    actual = shard["actual_train_query"]
    external, internal = shard["query_external_heldout"], shard["query_internal_group"]
    bm.require(not actual or (not external and not internal and cohort == "train"), "actual_query_qualification")
    qualification = "external" if external or cohort != "train" else "internal" if internal else "fit"
    target_pos = shard["target_position"]
    all_ids = _list(shard["candidate_indices"])
    target_id = all_ids[target_pos] if target_pos >= 0 else None
    bm.require(bool(shard["target_natural"]) == (0 <= target_pos < NATURAL), "natural_target_flag")
    raw_bits = {k: _list(v) for k, v in shard["candidate_qualification_bits"].items()}
    for cap in (5, 10):
        ids, coarse, costs, roles, dtypes = _native_arrays(shard, cap)
        scales = [_scalar(roles[r]["scale"][0]) for r in ROLES]
        deltas = [_list(roles[r]["delta"])[:NATURAL] for r in ROLES]
        diag = bm.diagnose_costs(ids, coarse, costs, target_id, shard["gamma"], scales, cap,
             dtypes, role_deltas=deltas, old_other_id=old.get("original_other_candidate_index") if old else None,
             outside_rank=shard.get("cached_outside_pool_rank"))
        trace_sha = [roles[r]["trace_sha256_natural512"] for r in ROLES]
        fp = bm.cost_fingerprint(ids, coarse, costs, dtypes, trace_sha)
        k = key(seed, cap, pool, shard)
        previous = legacy.get(k)
        if previous is not None:
            bm.require(previous["cost_fingerprint"] == fp, "worker_vs_legacy_cost_order_dtype_trace")
            bm.require(previous["diagnostic"]["rank"] == diag["rank"], "worker_legacy_rank")
        else:
            bm.require(cohort == "train" and (pool == "single" or internal), "unexpected_unreferenced_worker_scope")
        features, loss_info, queries = dict(input36_dtype_by_role={r: str(roles[r]["input36"].dtype) for r in ROLES}), {}, []
        inputs = {r: (roles[r]["input36"].detach().cpu().numpy() if hasattr(roles[r]["input36"], "detach")
                      else np.asarray(roles[r]["input36"])) for r in ROLES}
        bm.require(all(a.shape == (NATURAL, 36) and a.dtype in (np.float32, np.float64)
                       for a in inputs.values()), "actual_input36_dtype_shape")
        input_bytes = {r: a.view(np.uint8).reshape(NATURAL, -1) for r, a in inputs.items()}
        if shard["target_natural"]:
            equal = np.ones(NATURAL, dtype=bool)
            for role in ROLES:
                equal &= (input_bytes[role] == input_bytes[role][target_pos]).all(axis=1)
            ahead = [p for p in range(NATURAL) if p != target_pos and
                     costs[p] <= bm.native_threshold(costs[target_pos], dtypes["final_cost_dtype"])]
            collision_ids = [ids[p] for p in ahead if equal[p]]
            features.update(both_role_full36_equal_ahead_count=len(collision_ids),
                            both_role_full36_equal_ahead_ids=collision_ids,
                            representation_collision_top10_blocked=len(collision_ids) >= 10)
        else:
            features.update(both_role_full36_equal_ahead_count=None,
                            both_role_full36_equal_ahead_ids=None, representation_collision_top10_blocked=None)
        if diag["pairs"] is not None:
            pair_positions = {name: ids.index(v["candidate_index"]) for name, v in diag["pairs"].items()
                              if v is not None}
            for name, pos in pair_positions.items():
                flags = {r: bool(np.array_equal(input_bytes[r][target_pos], input_bytes[r][pos])) for r in ROLES}
                features[name] = dict(full36_equal_by_role=flags, both_roles_equal=all(flags.values()),
                    raw33_RMS={r: bm.rms_distance(_list(roles[r]["raw_X"])[target_pos],
                                                 _list(roles[r]["raw_X"])[pos]) for r in ROLES},
                    rawO3_RMS={r: bm.rms_distance(_list(roles[r]["raw_O"])[target_pos],
                                                _list(roles[r]["raw_O"])[pos]) for r in ROLES})
        critical = dual_negatives.get((seed, cap, qid), [])
        for role in ROLES:
            tr = roles[role]
            loss_trace, loss_ids, loss_target = loss_context(shard, cap, role)
            loss = loss_trace["actual_train_loss"]
            bm.require((loss is not None) == bool(actual), "actual_loss_scope")
            info = dict(status=loss_trace["actual_train_loss_status"], actual_historical_exposure="unknown",
                        target_probability=None, anchor_opposes_target=None,
                        critical_dual_negatives_in_loss_support=None, critical_dual_probability_mass=None)
            if loss is not None:
                allowed = _list(loss["allowed"]); p = _list(loss["probability"])
                coarse_loss = _list(shard["loss_coarse_cost"]) if shard.get("loss_roles") else _list(shard["coarse_cost"])
                verified = bm.loss_probe(coarse_loss, _list(loss_trace["delta"]), scales[ROLES.index(role)],
                    loss_target, allowed, _list(loss_trace["logit"]), cap,
                    actual_costs=_list(loss["actual_normalized_costs"]))
                for actual_name, math_name in (("probability", "probabilities"),
                    ("listwise_gradient_delta", "listwise_gradient_delta"),
                    ("anchor_gradient_delta", "anchor_gradient_delta"),
                    ("total_gradient_logit", "total_gradient_logit")):
                    bm.require(len(_list(loss[actual_name])) == len(verified[math_name]) and
                        all(math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-6)
                        for a, b in zip(_list(loss[actual_name]), verified[math_name])),
                        "actual_loss_formula_mismatch:" + actual_name)
                gl, ga = _list(loss["listwise_gradient_delta"]), _list(loss["anchor_gradient_delta"])
                indices = {cid: pos for pos, cid in enumerate(loss_ids)}
                support = [indices[cid] for cid in critical if cid in indices and allowed[indices[cid]]]
                info["actual_loss_formula_verified"] = True
                info.update(listwise_loss=float(_scalar(loss["listwise_loss"])),
                            anchor_loss=float(_scalar(loss["anchor_loss"])), total_loss=float(_scalar(loss["total_loss"])),
                            support_count=loss["support_count"],
                            current_gradient_scope=loss.get("gradient_scope"),
                            scalar_source="saved_current_frozen_single_role_loss",
                            target_total_gradient_delta=_list(loss["total_gradient_delta"])[loss_target],
                            target_tanh_local_derivative=_list(loss["tanh_local_derivative"])[loss_target],
                            target_listwise_gradient_logit=_list(loss["listwise_gradient_logit"])[loss_target],
                            target_anchor_gradient_logit=_list(loss["anchor_gradient_logit"])[loss_target])
                info.update(target_probability=p[loss_target], anchor_opposes_target=gl[loss_target] * ga[loss_target] < 0,
                            target_listwise_gradient=gl[loss_target], target_anchor_gradient=ga[loss_target],
                            target_total_logit_gradient=_list(loss["total_gradient_logit"])[loss_target],
                            actual_loss_shape=len(loss_ids),
                            critical_dual_negatives_in_loss_support=len(support),
                            critical_dual_probability_mass=math.fsum(p[pos] for pos in support),
                            target_in_natural512=bool(shard["target_natural"]),
                            loss_only_positive_used=bool(shard["diagnostic_loss_only_positive_used"]),
                            critical_dual_qualification=[dict(candidate_index=cid, in_loss_row=cid in indices,
                                in_support=cid in indices and allowed[indices[cid]],
                                exclusion_bits={name: vals[all_ids.index(cid)] for name, vals in raw_bits.items()}
                                    if cid in all_ids else None) for cid in critical])
            target_delta = _list(loss_trace["delta"])[loss_target] if loss is not None else (
                _list(tr["delta"])[target_pos] if target_pos >= 0 else None)
            info["target_saturated95"] = abs(target_delta) / cap >= .95 if target_delta is not None else None
            descriptor_target = loss_target if loss_target is not None else target_pos
            points = {"target": loss_ids[descriptor_target] if descriptor_target >= 0 else None,
                      "immutable_old_other": old.get("original_other_candidate_index") if old else None}
            if diag["pairs"]:
                points.update({name: value["candidate_index"] for name, value in diag["pairs"].items() if value})
            info["candidate_points"] = {}
            for name, cid in points.items():
                pos = loss_ids.index(cid) if cid in loss_ids else None
                known = loss is not None and pos is not None
                info["candidate_points"][name] = dict(candidate_index=cid, in_loss_row=pos is not None,
                    support=bool(_list(loss["allowed"])[pos]) if known else None,
                    probability=_list(loss["probability"])[pos] if known else None,
                    listwise_gradient_delta=_list(loss["listwise_gradient_delta"])[pos] if known else None,
                    anchor_gradient_delta=_list(loss["anchor_gradient_delta"])[pos] if known else None,
                    total_gradient_delta=_list(loss["total_gradient_delta"])[pos] if known else None,
                     tanh_local_derivative=_list(loss["tanh_local_derivative"])[pos] if known else None,
                     listwise_gradient_logit=_list(loss["listwise_gradient_logit"])[pos] if known else None,
                     anchor_gradient_logit=_list(loss["anchor_gradient_logit"])[pos] if known else None,
                    total_gradient_logit=_list(loss["total_gradient_logit"])[pos] if known else None,
                    status="known" if known else "not_training_query" if not actual else "absent_from_actual_loss_row")
            if loss is not None:
                for point in info["candidate_points"].values():
                    glp, gap = point["listwise_gradient_delta"], point["anchor_gradient_delta"]
                    point["absolute_anchor_to_listwise_delta_ratio"] = abs(gap/glp) if glp not in (None,0) else None
                    point["ratio_undefined_reason"] = ("not_in_actual_loss" if glp is None else
                        "zero_listwise_gradient" if glp==0 else None)
                info["target_absolute_anchor_to_listwise_delta_ratio"] = info["candidate_points"]["target"]["absolute_anchor_to_listwise_delta_ratio"]
                info["target_ratio_undefined_reason"] = info["candidate_points"]["target"]["ratio_undefined_reason"]
            info["critical_dual_membership_table"] = critical_membership_table(shard,loss_ids,loss,critical,loss_target)
            loss_info[role] = info
            jac_positions = tr["input_jacobians"]["positions"]
            jac = tr["input_jacobians"].get("weighted_dcost_dinput36")
            jac_values = _list(jac) if jac is not None else []
            features[role + "_input_gradient"] = [
                dict(candidate_index=all_ids[pos],
                     centered33_RMS=math.sqrt(math.fsum(x*x for x in value[:33])/33),
                     centeredO3_RMS=math.sqrt(math.fsum(x*x for x in value[33:])/3))
                for pos, value in zip(jac_positions, jac_values)]
            rawx, rawo = _list(tr["raw_X"]), _list(tr["raw_O"])
            features[role + "_important_candidate_hashes"] = [
                dict(candidate_index=all_ids[pos],
                     raw33=bm.vector_hash(rawx[pos], str(tr["raw_X"].dtype)),
                     rawO3=bm.vector_hash(rawo[pos], str(tr["raw_O"].dtype)),
                     full36=bm.vector_hash(inputs[role][pos].tolist(), str(tr["input36"].dtype)))
                for pos in jac_positions]
            for pos in jac_positions:
                queries.append(dict(seed=seed, cap=cap, role=role, pool=pool, kind=shard["kind"],
                                    query_group=qid[-1], candidate_index=all_ids[pos],
                                    is_target=(pos == target_pos), input36_dtype=str(tr["input36"].dtype),
                                    input36_hash=bm.vector_hash(inputs[role][pos].tolist(), str(tr["input36"].dtype)),
                                    actual_training_label=("target" if loss_ids.index(all_ids[pos]) == loss_target else "eligible_negative")
                                        if loss is not None and all_ids[pos] in loss_ids and _list(loss["allowed"])[loss_ids.index(all_ids[pos])] else None,
                                    centered36=inputs[role][pos].tolist(), raw33=rawx[pos], rawO3=rawo[pos]))
        proxy = None
        if target_pos >= 0:
            proxy = {role: dict(target_kde020=_list(roles[role]["raw_X"])[target_pos][2],
                                target_observed_nearest_mean=_list(roles[role]["raw_X"])[target_pos][9],
                                target_observation_positive_nll=_list(roles[role]["raw_O"])[target_pos][0],
                                boundary="observation_matching_proxy_not_aligned_forward_error") for role in ROLES}
        supplied = shard.get("structural_evidence")
        structure = None
        if supplied is not None:
            bm.require(supplied["schema"] == "top10_structural_evidence_v1" and
                       supplied["fingerprint"] == dict(algorithm="Morgan", radius=2, nBits=2048, useChirality=False)
                       and type(supplied["rdkit_version"]) is str and bool(supplied["rdkit_version"]),
                       "frozen_structure_fingerprint_contract")
            pairs = _cap(supplied["caps"], cap)
            bm.require(set(pairs) == {"immutable_old_other","current_negative1","current_negative10"},
                       "structure_pair_set")
            for name, value in pairs.items():
                bm.require(value["status"] in ("known","unknown","not_applicable"), "structure_status")
                if value["status"] == "known":
                    bm.require(bm.finite(value["tanimoto"]) and 0 <= value["tanimoto"] <= 1 and
                               type(value["same_connectivity"]) is bool, "structure_known_values")
                if diag["pairs"] and diag["pairs"].get(name) and value["status"] == "known":
                    bm.require(value["other_candidate_id"] == diag["pairs"][name]["candidate_index"] and
                               value["target_candidate_id"] == target_id, "structure_pair_identity")
            structure = dict(schema=supplied["schema"], rdkit_version=supplied["rdkit_version"],
                fingerprint=supplied["fingerprint"], target_smiles_source=supplied["target_smiles_source"],
                pairs=pairs)
        fact = dict(seed=seed, cap=cap, pool=pool, cohort=cohort, query_identity=shard["query_identity"],
                    source=qid[2], population_source=shard["population"], fixed400_member=old is not None,
                    original87=bool(old and old["original_bound_obstructed"]), old_manifest=old,
                    gamma=shard["gamma"], dtypes=dtypes, role_scale0=scales,
                    cost_fingerprint=fp, trace_sha256=trace_sha, diagnostic=diag,
                    worker_probe=True, query_qualification=qualification, feature_evidence=features,
                    loss_evidence=loss_info, nn_evidence=None, structural_evidence=structure,
                    forward_matching_proxy=proxy, observation_semantics=shard.get("observation_semantics"),
                    public_coverage=shard.get("public_coverage"),
                    positive_membership="natural512_positive" if shard["target_natural"] else
                        "loss_only_appended" if shard["diagnostic_loss_only_positive_used"] else "natural_miss_no_loss_append",
                    current_representation_boundary="Exact collision constrains only current frozen O3, centers/scales, full36 and same selected head.",
                    actual_historical_exposure=None,
                    actual_historical_exposure_status="unknown", missing_measurement_mask_status="unknown",
                    true_forward_RMSE_status="unknown")
        yield k, fact, queries


_REFERENCE_ARRAY_CACHE = {}


def attach_nearest(fact, queries, references):
    """numpy逐query精确距离；参考预hash冻结，预制矩阵避免每个候选重建参考矩阵。"""
    import numpy as np
    output = []
    for query in queries:
        groupkey = tuple(query[k] for k in ("seed", "cap", "role", "pool"))
        cachekey = (id(references), groupkey, query["kind"])
        if cachekey not in _REFERENCE_ARRAY_CACHE:
            rows = sorted((r for r in references.get(groupkey, []) if r["kind"] == query["kind"]),
                          key=lambda r: (r["reference_hash"], r["candidate_index"], r["query_group"]))
            matrix = np.asarray([r["centered36"] for r in rows], dtype=np.float64)
            hashes = collections.defaultdict(list)
            for index, row in enumerate(rows):
                hashes[(row["input36_dtype"], row["input36_hash"])].append(index)
            _REFERENCE_ARRAY_CACHE[cachekey] = (rows, matrix, hashes)
        rows, matrix, hashes = _REFERENCE_ARRAY_CACHE[cachekey]
        allowed = np.asarray([r["query_group"] != query["query_group"] for r in rows], dtype=bool)
        if not bool(allowed.any()):
            output.append(dict(candidate_index=query["candidate_index"], role=query["role"],
                               is_target=query.get("is_target", False), status="unknown", reason="no_out_of_group_sampled_reference"))
            continue
        diff = matrix - np.asarray(query["centered36"], dtype=np.float64)
        squared = np.einsum("ij,ij->i", diff, diff) / 36
        squared[~allowed] = np.inf
        at = int(np.argmin(squared))
        other = rows[at]
        exact_indices = [i for i in hashes.get((query["input36_dtype"], query["input36_hash"]), [])
                         if rows[i]["query_group"] != query["query_group"]]
        opposite = sum(rows[i]["training_label"] != query["actual_training_label"] for i in exact_indices) if query.get("actual_training_label") else None
        output.append(dict(candidate_index=query["candidate_index"], role=query["role"],
                           is_target=query.get("is_target", False), status="known",
                           reference_count=int(allowed.sum()), centered36_RMS=math.sqrt(float(squared[at])),
                           reference_hash=other["reference_hash"], reference_candidate_index=other["candidate_index"],
                           reference_input_shape=other.get("input_shape"),
                           current_frozen_input_exact_reference_count=len(exact_indices),
                           opposite_actual_loss_label_reference_count=opposite,
                           exact_collision_boundary="Current frozen O3/center/scale/full36 only; opposite labels show local pressure, not infeasibility of redesigned features or retraining O.",
                           raw33_RMS=bm.rms_distance(query["raw33"], other["raw33"]),
                           rawO3_RMS=bm.rms_distance(query["rawO3"], other["rawO3"]),
                           centered33_RMS=bm.rms_distance(query["centered36"][:33], other["centered36"][:33]),
                           centeredO3_RMS=bm.rms_distance(query["centered36"][33:], other["centered36"][33:])))
    fact["nn_evidence"] = output


def nearest_state(row, target_only=False):
    if target_only and not row["diagnostic"]["target_in_pool"]:
        return "not_applicable"
    records = row.get("nn_evidence")
    if records is None: return None
    records = [v for v in records if not target_only or v.get("is_target")]
    return "known" if any(v.get("status")=="known" for v in records) else None


def structural_state(row):
    pairs=(row.get("structural_evidence") or {}).get("pairs")
    if not pairs:return None
    statuses=[p["status"] for p in pairs.values()]
    if any(s=="known" for s in statuses):return "known"
    if all(s=="not_applicable" for s in statuses):return "not_applicable"
    return None


def crosscounts(rows):
    """关键交叉只数预固定成员；代理可用/不可用和阳性独立列，不把unknown归阴性。"""
    names = ("old_pair_released_but_actual_top10_failed", "released_failed_S1_fail_static_blocked",
        "released_failed_S2_fail_static_open", "released_failed_P0_target_absent",
        "S2_exact_collision_unknown", "S2_exact_collision_known", "S2_any_exact_collision_constraint",
        "S2_collision_at_least10", "S2_loss_evidence_unknown", "S2_actual_loss_not_applicable",
        "S2_actual_loss_known", "S2_anchor_opposes_target_any_role", "S2_target_saturated95_any_role",
        "S2_critical_dual_loss_support_known", "S2_observation_matching_proxy_available",
        "S2_measurement_mask_unknown", "S2_true_forward_error_unknown",
        "S2_crossquery_opposite_loss_label_exact_constraint", "S2_crossquery_collision_reference_unknown")
    counts = collections.Counter({name: 0 for name in names})
    for row in rows:
        diag = row["diagnostic"]
        old = row.get("old_manifest")
        released = bool(old and old["original_coarse_gap"] <= diag["pairwise_bound"])
        failed = diag["rank"] is not None and diag["rank"] > 10
        if not released or not failed:
            continue
        counts["old_pair_released_but_actual_top10_failed"] += 1
        counts["released_failed_" + diag["bucket"]] += 1
        if diag["bucket"] != "S2_fail_static_open":
            continue
        feature = row.get("feature_evidence")
        if feature is None:
            counts["S2_exact_collision_unknown"] += 1
        else:
            value = feature.get("both_role_full36_equal_ahead_count")
            if value is None:
                counts["S2_exact_collision_unknown"] += 1
            else:
                counts["S2_exact_collision_known"] += 1
                counts["S2_any_exact_collision_constraint"] += int(value > 0)
                counts["S2_collision_at_least10"] += int(value >= 10)
        loss = row.get("loss_evidence")
        if loss is None:
            counts["S2_loss_evidence_unknown"] += 1
        elif row["query_qualification"] != "fit":
            counts["S2_actual_loss_not_applicable"] += 1
        else:
            counts["S2_actual_loss_known"] += 1
            counts["S2_anchor_opposes_target_any_role"] += int(any(
                loss[r]["anchor_opposes_target"] is True for r in ROLES))
            counts["S2_target_saturated95_any_role"] += int(any(
                loss[r]["target_saturated95"] is True for r in ROLES))
            counts["S2_critical_dual_loss_support_known"] += int(all(
                loss[r]["critical_dual_negatives_in_loss_support"] is not None for r in ROLES))
        nearest = row.get("nn_evidence")
        if nearest_state(row) is None:
            counts["S2_crossquery_collision_reference_unknown"] += 1
        else:
            counts["S2_crossquery_opposite_loss_label_exact_constraint"] += int(any(
                v.get("opposite_actual_loss_label_reference_count", 0) is not None and
                v.get("opposite_actual_loss_label_reference_count", 0) > 0 for v in nearest))
        counts["S2_observation_matching_proxy_available"] += int(row.get("forward_matching_proxy") is not None)
        counts["S2_measurement_mask_unknown"] += int(row["missing_measurement_mask_status"] == "unknown")
        counts["S2_true_forward_error_unknown"] += int(row["true_forward_RMSE_status"] == "unknown")
    return dict(counts)


def evidence_distributions(rows):
    """弱代理只给连续分布/可用分母；不由结果发明阈值或把代理当真实前向误差。"""
    result = {}
    for role in ROLES:
        for field in ("target_probability", "target_listwise_gradient", "target_anchor_gradient",
                      "target_total_logit_gradient", "critical_dual_negatives_in_loss_support",
                      "critical_dual_probability_mass"):
            result[role + ".loss." + field] = bm.distribution([
                r["loss_evidence"][role].get(field) if r.get("loss_evidence") else None for r in rows])
        for field in ("centered36_RMS", "raw33_RMS", "rawO3_RMS", "centered33_RMS", "centeredO3_RMS"):
            values = []
            for row in rows:
                target = [v for v in (row.get("nn_evidence") or []) if v["role"] == role and v.get("is_target")]
                values.append(target[0].get(field) if target and target[0].get("status") == "known" else None)
            result[role + ".target_NN." + field] = bm.distribution(values)
        for field in ("target_kde020", "target_observed_nearest_mean", "target_observation_positive_nll"):
            result[role + ".matching_proxy." + field] = bm.distribution([
                r["forward_matching_proxy"][role][field] if r.get("forward_matching_proxy") else None for r in rows])
        for pair in ("immutable_old_other", "current_negative1", "current_negative10"):
            for block in ("raw33_RMS", "rawO3_RMS"):
                result[role + "." + pair + "." + block] = bm.distribution([
                    r["feature_evidence"][pair][block][role]
                    if r.get("feature_evidence") and pair in r["feature_evidence"] else None for r in rows])
    for pair in ("immutable_old_other","current_negative1","current_negative10"):
        values, states = [], []
        for row in rows:
            obj = (row.get("structural_evidence") or {}).get("pairs", {}).get(pair)
            values.append(obj["tanimoto"] if obj and obj["status"] == "known" else None)
            states.append("not_applicable" if obj and obj["status"] == "not_applicable"
                          else obj if obj and obj["status"] == "known" else None)
        result["structure."+pair+".Morgan_Tanimoto"] = bm.distribution(values)
        result["structure."+pair+".availability"] = bm.availability(states)
    return result


def summarize(rows):
    buckets = collections.Counter(r["diagnostic"]["bucket"] for r in rows)
    ranks = [r["diagnostic"]["rank"] for r in rows]
    metrics = {name: (math.fsum(1/v for v in ranks)/len(ranks) if name == "MRR" else
                      sum(v <= int(name[3:]) for v in ranks)/len(ranks)) if ranks and all(v is not None for v in ranks) else None
               for name in ("MRR", "top1", "top5", "top10")}
    query_hashes = {hashlib.sha256(canonical(identity(r)).encode()).hexdigest() for r in rows}
    groups = {identity(r)[-1] for r in rows}
    return dict(query_seed_pairs=len(rows), unique_query_identities=len(query_hashes),
                unique_connectivity_groups=len(groups), _query_hashes=sorted(query_hashes), _groups=sorted(groups),
                metrics=metrics, buckets=dict(buckets), core_crosscounts=crosscounts(rows),
                weak_proxy_distributions=evidence_distributions(rows),
                rank=bm.distribution([r["diagnostic"]["rank"] for r in rows]),
                static_unbeatable=bm.distribution([r["diagnostic"].get("static_unbeatable_count") for r in rows]),
                feature_evidence_availability=bm.availability([r.get("feature_evidence") for r in rows]),
                loss_evidence_availability=bm.availability([
                    "not_applicable" if r["query_qualification"] in ("internal", "external")
                    else r.get("loss_evidence") for r in rows]),
                nn_availability=bm.availability([nearest_state(r) for r in rows]),
                 natural_target_nn_availability=bm.availability([nearest_state(r,True) for r in rows]),
                structural_availability=bm.availability([structural_state(r) for r in rows]),
                actual_historical_exposure=dict(known=0, unknown=len(rows), not_applicable=0),
                interpretation="Necessary bounds, exact input constraints and weak proxies remain separate.")


def population_summaries(facts, released_keys):
    groups = collections.defaultdict(list)
    for row in facts:
        labels = []
        ikey = (row["seed"], identity(row))
        if row["fixed400_member"]:
            labels.append("fixed400")
            if row["original87"]:
                labels.append("fixed400_old87")
                if ikey in released_keys:
                    labels.append("fixed400_old87_C10_oldpair_released")
        if row["population_source"] == "internal":
            labels.append("internal_hash_probe")
        if row["cohort"] in ("original_val", "tune"):
            labels.append("external_observed_full")
            if row["worker_probe"]:
                labels.append("external_hash_probe")
        for label in labels:
            # 每seed先算；完整合并只报告query-seed计数，不冒称独立query。
            group = (label, row["cap"], row["seed"], row["pool"], row["cohort"], row["query_qualification"])
            groups[group].append(row)
            groups[group + ("source", row["source"])].append(row)
            for axis in ("observation_semantics", "gamma", "public_coverage", "positive_membership"):
                value = row.get(axis)
                groups[group + (axis, canonical(value))].append(row)
    return [{"group": list(k), **summarize(v)} for k, v in sorted(groups.items())]


def compact_population_rows(parts):
    """现有六轴无附加分层summary的直接展平，不定义新分组、阈值或统计量。"""
    rows=[]
    for part in parts:
        if len(part["group"])!=6: continue
        row=dict(zip(("population","cap","seed","pool","cohort","qualification"),part["group"]))
        row.update(denominator_query_seed_pairs=part["query_seed_pairs"],unique_query_identities=part["unique_query_identities"],
                   unique_connectivity_groups=part["unique_connectivity_groups"],**part["metrics"])
        for bucket in ("P0_target_absent","S0_actual_top10","S1_fail_static_blocked","S2_fail_static_open"):
            row["bucket_"+bucket]=part["buckets"].get(bucket,0)
        row.update(part["core_crosscounts"])
        for field in ("feature_evidence_availability","loss_evidence_availability","nn_availability","natural_target_nn_availability","structural_availability","actual_historical_exposure"):
            for status in ("known","unknown","not_applicable"):
                row[field+"_"+status]=part[field][status]
        rows.append(row)
    return rows


def equal_seed_summaries(parts):
    """仅对同一统计人口的两seed指标等权；计数求和，缺seed不补零也不伪造均值。"""
    grouped = collections.defaultdict(dict)
    for part in parts:
        g = part["group"]
        grouped[tuple(g[:2] + g[3:])][g[2]] = part
    output = []
    for group, seeds in sorted(grouped.items()):
        complete = set(seeds) == {193, 211}
        names = ("MRR", "top1", "top5", "top10")
        means = {name: math.fsum(seeds[s]["metrics"][name] for s in (193, 211))/2
                 if complete and all(seeds[s]["metrics"][name] is not None for s in (193, 211))
                 else None for name in names}
        counts = collections.Counter()
        for part in seeds.values():
            counts.update(part["core_crosscounts"])
        output.append(dict(group=list(group), seeds_available=sorted(seeds),
                           query_seed_pairs=sum(p["query_seed_pairs"] for p in seeds.values()),
                           unique_query_identities=len(set().union(*(set(p["_query_hashes"]) for p in seeds.values()))),
                           unique_connectivity_groups=len(set().union(*(set(p["_groups"]) for p in seeds.values()))),
                           equal_two_seed_mean=means, summed_core_crosscounts=dict(counts)))
    return output


def run_analysis(request, admission, pins, out):
    started = time.monotonic()
    provenance = validate_completed_provenance(request, pins)
    _REFERENCE_ARRAY_CACHE.clear()
    old_obj = pins.json(request["fixed_manifest"])
    bm.require(len(old_obj["records"]) == 400, "fixed400_manifest")
    old_map = {(r["seed"], identity(r)): r for r in old_obj["records"]}
    bm.require(len(old_map) == 400 and sum(bool(r["original_bound_obstructed"])
               for r in old_map.values()) == 87, "fixed87_manifest")
    selection = pins.json(request["query_manifest"])
    selected = {(r["seed"], r["cohort"], identity(r)) for r in selection["records"]}
    bm.require(len(selected) == len(selection["records"]) == request["expected_query_seed_pairs"] == 826, "probe_manifest")
    request["_selection_by_identity"] = {(r["seed"],r["cohort"],identity(r)):{**r, "_fixed400_member":(r["seed"],identity(r)) in old_map} for r in selection["records"]}
    facts = load_legacy(request, old_map, pins)
    references, dual_negatives, seen = build_references(request, old_map, pins)
    expected = {(s, c, p, i) for s, c, i in selected for p in ("single", "dual")}
    bm.require(seen == expected, "complete_presampled_worker_manifest")
    for shard, _ in worker_rows(request, pins):
        for k, fact, queries in worker_facts(shard, old_map, facts, dual_negatives):
            attach_nearest(fact, queries, references)
            facts[k] = fact
        bm.require(time.monotonic() - started <= request.get("max_seconds", 3600), "analysis_wall_budget")
    values = [facts[k] for k in sorted(facts)]
    released = {(r["seed"], identity(r)) for r in values if r["fixed400_member"] and r["original87"]
                and r["cap"] == 10 and r["pool"] == "dual" and
                r["old_manifest"]["original_coarse_gap"] <= r["diagnostic"]["pairwise_bound"]}
    if "expected_C10_released_old87" in request:
        bm.require(len(released) == request["expected_C10_released_old87"], "frozen_C10_release_count")
    paired = []
    for k, row in sorted(facts.items()):
        if row["cap"] != 5:
            continue
        other = facts.get((k[0], 10, k[2], k[3]))
        bm.require(other is not None, "paired_caps_required")
        paired.append(dict(seed=row["seed"], pool=row["pool"], cohort=row["cohort"],
                           query_identity=row["query_identity"],
                           **bm.paired_cap_difference(row["diagnostic"], other["diagnostic"])))
    root = safe_path(request["write_root"]); output = Path(out).resolve()
    bm.require(output.is_relative_to(root) and not output.exists(), "new_output_in_diagnostic_root")
    pins.finish()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "diagnostics.jsonl").open("x") as stream:
        for row in values:
            stream.write(canonical(row) + "\n")
    with (output / "paired_cap_changes.jsonl").open("x") as stream:
        for row in paired:
            stream.write(canonical(row) + "\n")
    ref_summary = [{"stratum": list(k), "count": len(v),
                    "ordered_reference_hashes_sha256": hashlib.sha256(
                        "\n".join(r["reference_hash"] for r in v).encode()).hexdigest()}
                   for k, v in sorted(references.items())]
    summaries = population_summaries(values, released)
    compact = compact_population_rows(summaries)
    (output / "compact_population_table.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    with (output / "compact_population_table.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(compact[0]) if compact else [])
        writer.writeheader(); writer.writerows(compact)
    result = dict(schema=SCHEMA, status="PASS", execution_scope=EXECUTION_SCOPE, closed_saved_tensor_provenance=provenance,
                  new_NN_calls=0, model_loads=0, source_NN_stage_status="FAIL_SUPERVISION",
                  original_resource_envelope_recertified=False, fixed400=400, original87=87,
                  C10_released_old87=len(released), summaries=summaries,
                  equal_seed_summaries=equal_seed_summaries(summaries),
                  training_reference=ref_summary, input_sha256=pins.checked,
                  independent_admission=admission, natural512_inference_role_row_budget=7040,
                  appended_fit_single_full513_loss_role_row_budget=1600, total_role_row_forward_budget=8640,
                  model_calls=0, new_spectrum_forward_calls=0, forward_bank_calls=0,
                  optimizer_steps=0, weight_updates=0, external_api_calls=0, test_read=False,
                  selection_performed=False, promotion=False, independent_blind_test=False,
                  actual_historical_exposure="unknown", inputs_unchanged=True,
                  elapsed_seconds=time.monotonic() - started)
    for part in summaries:
        part.pop("_query_hashes", None); part.pop("_groups", None)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    worker_bytes = 0
    for entry in request["worker_outputs"]:
        worker_meta = json.loads(Path(entry["summary"]["path"]).read_text())
        worker_bytes += worker_meta["artifact_bytes"]
        worker_bytes += Path(entry["index"]["path"]).stat().st_size + Path(entry["summary"]["path"]).stat().st_size
    bm.require(worker_bytes + sum(p.stat().st_size for p in output.iterdir() if p.is_file()) + 2 <=
               request.get("max_artifact_bytes", 4 * 1024**3), "global_new_artifact_budget")
    pins.finish()
    (output / "EXIT_CODE").write_text("0\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("request", "request-sha256", "admission", "admission-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    request, admission, pins = admit(args.request, args.request_sha256, args.admission,
                                     args.admission_sha256, args.execute)
    result = run_analysis(request, admission, pins, args.out)
    print(json.dumps(dict(status=result["status"], model_calls=0), separators=(",", ":")))


if __name__ == "__main__":
    main()

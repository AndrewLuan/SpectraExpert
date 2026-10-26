"""功能概述：Top10屏障诊断的纯数学层，只处理调用方已冻结的数值，不打开科学文件。
native_*逐运算重放原float32/float64成本、EPS阈值及排名；diagnose_costs生成固定旧对手、
当前负例1/10、双角色box极值必要桶和实际仍需击败数。loss_probe按单role真实listwise
与anchor计算末态概率和δ/logit导数，不重放历史优化。vector_hash/two_role_collision/
nearest_reference核完整36输入碰撞和冻结参考近邻；distribution/availability负责空值分母。
extract_frozen_diagnostics.py可调用本模块；analyze_diagnostics.py在来源核验后消费其结果。
所有函数返回新字典/列表，不修改输入、不调用模型/网络/API、不训练；不完整或非法输入
抛ValueError，调用方不能把缺证据补0。必要上界开放不是可学习证明，资格也不是实际曝光。
"""
from __future__ import annotations
import hashlib
import json
import math
import struct

EPS = 1e-7
DTYPES = ("torch.float32", "torch.float64")
ROLES = ("no_aug", "aug")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def cast_native(value, dtype):
    require(dtype in DTYPES and finite(value), "native_dtype_or_value")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0] if dtype == DTYPES[0] else float(value)
    except OverflowError as exc:
        raise ValueError("native_overflow") from exc
    require(math.isfinite(result), "native_nonfinite")
    return result


def promoted_dtype(coarse_dtype, correction_dtypes):
    require(coarse_dtype in DTYPES and len(correction_dtypes) == 2 and
            all(d in DTYPES for d in correction_dtypes), "native_input_dtypes")
    return DTYPES[1] if DTYPES[1] in [coarse_dtype, *correction_dtypes] else DTYPES[0]


def native_threshold(target, dtype):
    return cast_native(cast_native(target, dtype) + cast_native(EPS, dtype), dtype)


def native_cost_replay(coarse, delta0, delta1, scale0, scale1, gamma,
                       coarse_dtype, correction_dtypes):
    require(gamma in (.25, .75) and all(finite(x) and x > 0 for x in (scale0, scale1)),
            "route_or_scales")
    final = promoted_dtype(coarse_dtype, correction_dtypes)
    a = cast_native(delta0 * scale0, correction_dtypes[0])
    b = cast_native(delta1 * scale1, correction_dtypes[1])
    a = cast_native((1 - gamma) * a, correction_dtypes[0])
    b = cast_native(gamma * b, correction_dtypes[1])
    first = DTYPES[1] if DTYPES[1] in (coarse_dtype, correction_dtypes[0]) else DTYPES[0]
    return cast_native(cast_native(cast_native(coarse, coarse_dtype) + a, first) + b, final)


def native_rank(costs, target_position, dtype):
    require(type(target_position) is int and 0 <= target_position < len(costs), "target_position")
    require(all(finite(c) and cast_native(c, dtype) == c for c in costs), "native_cost_bytes")
    threshold = native_threshold(costs[target_position], dtype)
    return sum(c <= threshold for c in costs)


def native_array_digest(values, dtype):
    require(dtype in DTYPES, "native_digest_dtype")
    h = hashlib.sha256(json.dumps([dtype, len(values)], separators=(",", ":")).encode())
    pack = "<f" if dtype == DTYPES[0] else "<d"
    for value in values:
        require(cast_native(value, dtype) == value, "native_digest_value_not_representable")
        h.update(struct.pack(pack, value))
    return h.hexdigest()


def vector_hash(values, dtype="torch.float32"):
    """哈希一条typed数值向量；调用方另核role/kind，禁止把量化近似相等当精确相等。"""
    return native_array_digest(values, dtype)


def cost_fingerprint(candidate_ids, coarse, costs, dtypes, trace_sha256=None):
    require(len(candidate_ids) == len(coarse) == len(costs), "fingerprint_lengths")
    payload = dict(candidate_ids=list(candidate_ids),
                   coarse=native_array_digest(coarse, dtypes["coarse_dtype"]),
                   costs=native_array_digest(costs, dtypes["final_cost_dtype"]),
                   dtypes=dtypes)
    if trace_sha256 is not None:
        require(len(trace_sha256) == 2 and all(len(v) == len(costs) for v in trace_sha256),
                "trace_sha_shape")
        payload["trace_sha256"] = trace_sha256
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def diagnose_costs(candidate_ids, coarse, costs, target_id, gamma, role_scale0, cap,
                   dtypes, role_deltas=None, old_other_id=None, outside_rank=None):
    """natural候选成本→数学诊断；role_deltas可缺，但不得据此伪造角色贡献。"""
    n = len(candidate_ids)
    require(n >= 11 and n == len(coarse) == len(costs) and len(set(candidate_ids)) == n,
            "candidate_shape")
    require(all(type(i) is int for i in candidate_ids) and cap in (5, 10) and
            gamma in (.25, .75), "candidate_or_route")
    require(len(role_scale0) == 2 and all(finite(s) and s > 0 for s in role_scale0), "role_scale0")
    require(dtypes["final_cost_dtype"] == promoted_dtype(dtypes["coarse_dtype"],
                                                       dtypes["correction_dtypes"]), "final_dtype")
    require(all(cast_native(v, dtypes["coarse_dtype"]) == v for v in coarse), "coarse_native")
    require(all(cast_native(v, dtypes["final_cost_dtype"]) == v for v in costs), "cost_native")
    weights = (1 - gamma, gamma)
    absolute = [cap * s for s in role_scale0]
    total_absolute = math.fsum(w * a for w, a in zip(weights, absolute))
    order = sorted(range(n), key=lambda k: (costs[k], candidate_ids[k]))
    base = dict(top10_candidate_ids=[candidate_ids[k] for k in order[:10]],
                role_absolute_bounds=absolute, combined_absolute_bound=total_absolute,
                pairwise_bound=2 * total_absolute, actual_exposure="unknown")
    if target_id not in candidate_ids:
        require(outside_rank is None or (type(outside_rank) is int and outside_rank > n),
                "outside_rank")
        return dict(base, target_in_pool=False, rank=outside_rank, bucket="P0_target_absent",
                    negative_counted=None, still_need_to_beat=None,
                    static_unbeatable_count=None, real_static_unbeatable_count=None,
                    pairs=None, required_fields_unknown=[])
    target = candidate_ids.index(target_id)
    rank = native_rank(costs, target, dtypes["final_cost_dtype"])
    negatives = [k for k in order if k != target]
    if role_deltas is not None:
        require(len(role_deltas) == 2 and all(len(v) == n for v in role_deltas), "role_delta_shape")
        require(all(finite(d) and abs(d) <= cap for row in role_deltas for d in row), "role_delta_cap")
        replay = [native_cost_replay(coarse[k], role_deltas[0][k], role_deltas[1][k],
                                    *role_scale0, gamma, dtypes["coarse_dtype"],
                                    dtypes["correction_dtypes"]) for k in range(n)]
        require(replay == list(costs), "role_delta_native_cost_mismatch")
    tmin = native_cost_replay(coarse[target], -cap, -cap, *role_scale0, gamma,
                             dtypes["coarse_dtype"], dtypes["correction_dtypes"])
    jmax = {j: native_cost_replay(coarse[j], cap, cap, *role_scale0, gamma,
                                dtypes["coarse_dtype"], dtypes["correction_dtypes"])
            for j in negatives}
    blocked_ids = [candidate_ids[j] for j in negatives
                   if jmax[j] <= native_threshold(tmin, dtypes["final_cost_dtype"])]
    real_tmin = coarse[target] - total_absolute
    real_blocked = sum(coarse[j] + total_absolute <= real_tmin + EPS for j in negatives)
    bucket = "S0_actual_top10" if rank <= 10 else (
        "S1_fail_static_blocked" if len(blocked_ids) >= 10 else "S2_fail_static_open")

    def pair(j):
        gb = float(coarse[target]) - float(coarse[j])
        native_margin = float(costs[target]) - float(costs[j])
        result = dict(candidate_index=candidate_ids[j], coarse_gap=gb,
                      native_margin=native_margin, pairwise_bound=2 * total_absolute,
                      necessary_bound_released=gb <= 2 * total_absolute,
                      role_gains=None, combined_gain=None, real_margin=None,
                      native_minus_real_margin=None)
        if role_deltas is not None:
            gains = [s * (d[j] - d[target]) for s, d in zip(role_scale0, role_deltas)]
            gain = math.fsum(w * g for w, g in zip(weights, gains))
            result.update(role_gains=gains, combined_gain=gain, real_margin=gb - gain,
                          native_minus_real_margin=native_margin - (gb - gain))
        return result

    pairs = dict(current_negative1=pair(negatives[0]), current_negative10=pair(negatives[9]),
                 immutable_old_other=(pair(candidate_ids.index(old_other_id))
                    if old_other_id in candidate_ids and old_other_id != target_id else None))
    return dict(base, target_in_pool=True, rank=rank, bucket=bucket,
                negative_counted=rank - 1, still_need_to_beat=max(0, rank - 10),
                native_target_minimum=tmin, static_unbeatable_count=len(blocked_ids),
                static_unbeatable_candidate_ids=blocked_ids,
                real_static_unbeatable_count=real_blocked,
                native_real_static_boundary_differs=(len(blocked_ids) != real_blocked),
                pairs=pairs, required_fields_unknown=[],
                top10_threshold_equivalence=(costs[negatives[9]] >
                                            native_threshold(costs[target], dtypes["final_cost_dtype"])))


def loss_probe(coarse, delta, scale0, target_position, allowed_mask, logits, cap,
               eligible=True, temperature=.5, anchor=.1, actual_costs=None):
    """单role末态loss解析导数；eligible=False时所有训练量为null，非假装零梯度。"""
    if not eligible:
        return dict(status="not_training_query", probabilities=None, listwise_loss=None,
                    anchor_loss=None, listwise_gradient_delta=None, anchor_gradient_delta=None,
                    total_gradient_delta=None, total_gradient_logit=None,
                    tanh_derivative=None, support=None)
    n = len(delta)
    require(n == len(coarse) == len(logits) == len(allowed_mask) and n >= 2, "loss_shape")
    require(all(type(v) is bool for v in allowed_mask) and type(target_position) is int and
            0 <= target_position < n and allowed_mask[target_position], "loss_support")
    require(sum(allowed_mask) >= 2 and finite(scale0) and scale0 > 0 and cap in (5, 10) and
            temperature == .5 and anchor == .1, "loss_registered_constants")
    require(all(finite(v) for v in [*coarse, *delta, *logits]), "loss_finite")
    require(all(abs(v) <= cap for v in delta), "loss_cap")
    costs = ([b / scale0 + d for b, d in zip(coarse, delta)] if actual_costs is None else list(actual_costs))
    require(len(costs) == n and all(finite(c) for c in costs), "loss_actual_costs")
    support = [i for i, allowed in enumerate(allowed_mask) if allowed]
    logweights = [-costs[i] / temperature for i in support]
    maxlog = max(logweights)
    denom = math.fsum(math.exp(v - maxlog) for v in logweights)
    logz = maxlog + math.log(denom)
    p = [0.] * n
    for i, logweight in zip(support, logweights):
        p[i] = math.exp(logweight - maxlog) / denom
    gl = [((int(i == target_position) - p[i]) / temperature if allowed_mask[i] else 0.)
          for i in range(n)]
    ga = [(2 * anchor * delta[i] / len(support) if allowed_mask[i] else 0.) for i in range(n)]
    gt = [a + b for a, b in zip(gl, ga)]
    derivative = [1 - math.tanh(z / cap) ** 2 for z in logits]
    gz = [g * q for g, q in zip(gt, derivative)]
    return dict(status="current_frozen_training_objective", probabilities=p,
                probability_evaluation="stable_FP64_from_actual_normalized_costs" if actual_costs is not None else "FP64_formula_no_native_bit_claim",
                listwise_loss=logz + costs[target_position] / temperature,
                anchor_loss=anchor * math.fsum(delta[i] ** 2 for i in support) / len(support),
                listwise_gradient_delta=gl, anchor_gradient_delta=ga, total_gradient_delta=gt,
                total_gradient_logit=gz, tanh_derivative=derivative,
                support=list(allowed_mask), allowed_count=len(support),
                anchor_opposes_listwise=[(a * b < 0) if allowed_mask[i] else None
                                         for i, (a, b) in enumerate(zip(gl, ga))],
                abs_anchor_over_abs_listwise=[abs(b / a) if a else None for a, b in zip(gl, ga)],
                ratio_unknown_reason=["zero_listwise_gradient" if not a else None for a in gl],
                boundary="Frozen current local drive; historical actual exposure and gradients remain unknown.")


def two_role_collision(role_inputs, target_id, other_id):
    """role_inputs={role:{candidate_id:{values:[36],dtype,kind}}}；缺任一role则unknown。"""
    flags = {}
    for role in ROLES:
        entries = role_inputs.get(role, {})
        t = entries.get(target_id, entries.get(str(target_id)))
        j = entries.get(other_id, entries.get(str(other_id)))
        if t is None or j is None:
            flags[role] = None
            continue
        require(len(t["values"]) == len(j["values"]) == 36, "full36_required")
        flags[role] = (t["kind"] == j["kind"] and t["dtype"] == j["dtype"] and
                       vector_hash(t["values"], t["dtype"]) == vector_hash(j["values"], j["dtype"]))
    combined = None if any(v is None for v in flags.values()) else all(flags.values())
    return dict(per_role=flags, both_roles_full36_equal=combined,
                boundary="Both-role equality constrains current frozen O3/centers/scales/full36 and head only; it does not prove impossibility after changing representation.")


def rms_distance(left, right):
    require(len(left) == len(right) and len(left) > 0 and
            all(finite(v) for v in [*left, *right]), "distance_vector")
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def nearest_reference(query, references):
    """冻结sampled参考的精确搜索；参数role/cap/kind、query_group显式过滤，无学习阈值。"""
    eligible = [r for r in references if all(r[k] == query[k] for k in ("cap", "role", "kind", *[z for z in ("seed", "pool") if z in query]))
                and r["query_group"] != query["query_group"]]
    if not eligible:
        return dict(status="unknown", reason="no_eligible_out_of_group_sampled_reference",
                    reference_count=0, nearest=None)
    ranked = sorted(((rms_distance(query["centered36"], r["centered36"]), r["reference_hash"],
                      r["candidate_index"], r) for r in eligible),
                    key=lambda z: z[:3])
    distance, _, _, row = ranked[0]
    out = dict(status="known", reference_count=len(eligible), centered36_RMS=distance,
               nearest={k: row[k] for k in ("reference_hash", "candidate_index", "query_group")},
               raw33_RMS=None, rawO3_RMS=None,
               centered33_RMS=rms_distance(query["centered36"][:33], row["centered36"][:33]),
               centeredO3_RMS=rms_distance(query["centered36"][33:], row["centered36"][33:]))
    if query.get("raw33") is not None and row.get("raw33") is not None:
        out["raw33_RMS"] = rms_distance(query["raw33"], row["raw33"])
    if query.get("rawO3") is not None and row.get("rawO3") is not None:
        out["rawO3_RMS"] = rms_distance(query["rawO3"], row["rawO3"])
    return out


def distribution(values):
    known = [v for v in values if v is not None]
    require(all(finite(v) for v in known), "distribution_finite")
    ordered = sorted(known)
    def quantile(p):
        if not ordered:
            return None
        at = (len(ordered) - 1) * p
        i = int(at)
        return ordered[i] + (ordered[min(i + 1, len(ordered) - 1)] - ordered[i]) * (at - i)
    return dict(available=len(known), unknown=len(values) - len(known),
                mean=math.fsum(known) / len(known) if known else None,
                quantiles={label: quantile(p) for label, p in
                    (("min", 0), ("p10", .1), ("p25", .25), ("median", .5),
                     ("p75", .75), ("p90", .9), ("p95", .95), ("max", 1))})


def availability(values):
    counts = dict(known=0, unknown=0, not_applicable=0)
    for value in values:
        if value == "not_applicable":
            counts["not_applicable"] += 1
        elif value is None:
            counts["unknown"] += 1
        else:
            counts["known"] += 1
    return dict(total=len(values), **counts)


def paired_cap_difference(left, right):
    """同身份同池C5/C10诊断的集合差；不把当前对手移换说成旧对手被击败。"""
    a, b = left["top10_candidate_ids"], right["top10_candidate_ids"]
    require(len(a) == len(b) == 10 and len(set(a)) == len(set(b)) == 10, "paired_top10")
    sa, sb = set(a), set(b)
    return dict(C5_only=[i for i in a if i not in sb], C10_only=[i for i in b if i not in sa],
                common=[i for i in a if i in sb], overlap_count=len(sa & sb),
                rank_C10_minus_C5=(right["rank"] - left["rank"]
                    if right.get("rank") is not None and left.get("rank") is not None else None),
                actual_bucket_C5=left["bucket"], actual_bucket_C10=right["bucket"])

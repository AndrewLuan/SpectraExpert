"""功能概述：为新研究轮构建结构隔离、可追溯的训练与验收侧表。

build_cohorts 保留旧训练2048前缀和旧原验证/tune，按原哈希扩展训练支持。
独立验收采用正向资格白名单；祖先训练/选择曝光、历史验收和资格不明均排除。
main 仅读原缓存及冻结身份，新目录记录输入始末SHA、来源统计和实际退出码。
本模块不训练、不评分、不读取历史测试排名；供R/O/F训练和封存入口读取合同。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_THEORY = ("QME14S", "FG26", "QM9-NMR")
TRAIN_SALT = "global_train_317"
VERIFY_SALT = "research_independent_verify_20261022_317"


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def sha(path):
    out = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            out.update(block)
    return out.hexdigest()


def structural_split(group):
    if not isinstance(group, str) or not re.fullmatch(r"[0-9a-f]{64}", group):
        raise ValueError("invalid_connectivity_group")
    bucket = int(group[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def row_key(row):
    return tuple(row[k] for k in ("domain", "source", "record_id", "smiles"))


def unique_structures(rows, salt):
    chosen = {}
    for row in sorted(rows, key=lambda r: digest((salt, r["source"], r["record_id"], r["smiles"]))):
        chosen.setdefault(row["group"], row)
    return sorted(chosen.values(), key=lambda r: digest((salt, r["group"])))


def review_training_rows(rows, labels, artifact, apply_review):
    """审核只接收原始train原子侧表；不重复审核已派生的旧cohort annotations。"""
    train = [r for r in rows if r["domain"] == "experimental" and r["split"] == "train"]
    reviewed, audit = apply_review(labels, train, artifact, "domain")
    mapping = {row_key(r): r for r in reviewed}
    return [mapping[row_key(r)] if r["domain"] == "experimental" and r["split"] == "train" else r for r in rows], audit


def build_cohorts(rows, previous, candidate_groups, fixed_test, history,
                  train_target=8192, verify_target=1024):
    """纯选择逻辑：不接受排名、误差、来源权重或测试谱值。"""
    if train_target < len(previous["train"]) or verify_target < 1:
        raise ValueError("invalid_cohort_target")
    if history.get("schema") != "research_history_v1":
        raise ValueError("history_schema_required")
    exposed = set(history["exposed_groups"])
    uncertain = set(history["uncertain_groups"])
    cleared = set(history["cleared_independent_groups"])
    if cleared & (exposed | uncertain):
        raise ValueError("independent_eligibility_conflict")
    for row in rows:
        if row["split"] != structural_split(row["group"]):
            raise ValueError("stored_structural_split_conflict")
    old_eval = [r for name in ("original_val", "tune", "verify") for r in previous[name]]
    forbidden = {r["group"] for r in old_eval + list(fixed_test)}
    forbidden.update(history["evaluation_groups"])
    historical_protected = set(forbidden)
    forbidden.update(r["group"] for r in rows if r["split"] != "train")
    forbidden.update(g for g in candidate_groups if structural_split(g) != "train")
    retained = [r for r in rows if r["domain"] == "experimental" and r["nucleus"] == "1H"
                and r["split"] == "train" and r["group"] not in forbidden
                and float(r.get("review_spectral_weight", 0)) > 0]
    ordered = unique_structures(retained, TRAIN_SALT)
    target = min(train_target, len(ordered))
    train = ordered[:target]
    old = previous["train"]
    if [row_key(r) for r in train[:len(old)]] != [row_key(r) for r in old]:
        raise ValueError("old_2048_training_prefix_not_preserved")
    if len(train) < len(old):
        raise ValueError("insufficient_training_control_support")
    verification_pool = [r for r in rows if r["domain"] == "experimental" and r["nucleus"] == "1H"
                         and r["split"] != "train" and r["group"] in cleared
                         and r["group"] not in exposed | uncertain | historical_protected
                         and r["group"] in candidate_groups
                         and float(r.get("review_spectral_weight", 0)) > 0]
    verify = unique_structures(verification_pool, VERIFY_SALT)[:verify_target]
    verify_groups = {r["group"] for r in verify}
    if verify_groups & {r["group"] for r in train}:
        raise ValueError("new_verify_training_overlap")
    forbidden.update(verify_groups)
    cohorts = {"train": train, "train2048": list(old), "original_val": previous["original_val"],
               "tune": previous["tune"], "verify": verify}
    evaluation_names = ("original_val", "tune", "verify")
    for name in evaluation_names:
        if {r["group"] for r in train} & {r["group"] for r in cohorts[name]}:
            raise ValueError("training_evaluation_structure_overlap:" + name)
    for left, right in [("original_val", "tune"), ("original_val", "verify"), ("tune", "verify")]:
        if {r["group"] for r in cohorts[left]} & {r["group"] for r in cohorts[right]}:
            raise ValueError("evaluation_structure_overlap:" + left + ":" + right)
    theory_indices = {source: [i for i, r in enumerate(rows) if r["domain"] == "theory" and r["nucleus"] == "1H"
                              and r["source"] == source and r["split"] == "train"
                              and r["group"] not in forbidden] for source in REQUIRED_THEORY}
    if any(not indices for indices in theory_indices.values()):
        raise ValueError("required_theory_source_empty")
    return cohorts, {
        "schema": "research_cohort_contract_v1", "train_target": train_target,
        "training_eligible_unique_structures": len(ordered),
        "old_training_prefix_preserved": len(old),
        "selected_training_groups": [r["group"] for r in train],
        "forbidden_groups": sorted(forbidden),
        "training_excluded_groups": sorted(forbidden),
        "eligible_training_groups": sorted((set(candidate_groups) | {r["group"] for r in retained}) - forbidden),
        "forbidden_partition_buckets": list(range(80, 100)),
        "theory_requires_native_split_train": True,
        "theory_indices_by_source": theory_indices,
        "validation_keys": {name: [list(row_key(r)) for r in cohorts[name]] for name in evaluation_names},
        "historical_verify_excluded_groups": sorted({r["group"] for r in previous["verify"]}),
        "independent_verify": {"target": verify_target, "queries": len(verify),
            "state": "available" if verify else "unavailable",
            "qualified_unique_support": len(unique_structures(verification_pool, VERIFY_SALT)),
            "qualification": "positive historical clearance and no ancestor training/selection exposure; unknown excluded",
            "old_verify_reused": False},
        "training_sampling": "original global_train_317 record/group hashes; no error/source weighting; prefix retained",
        "selection": "fixed original320+tune1024 only; no old test result input",
        "external_api_calls": 0,
    }


def main():
    p = argparse.ArgumentParser()
    for name in ("base", "prepared", "previous-cohorts", "bank", "review", "history", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--train-target", type=int, default=8192)
    p.add_argument("--verify-target", type=int, default=1024)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    receipt = {"args": vars(args), "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
               "external_api_calls": 0, "state": "running", "hostname": socket.gethostname()}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        import torch
        from .data import identity
        from .h_condinfer import condition_key
        from .h_observation_study import stamp_queries
        from .h_reviewed_supervision import apply_review, load_artifact

        torch.set_num_threads(1)
        inputs = [Path(args.prepared) / "cache.pt", Path(args.prepared) / "manifest.json",
                  Path(args.previous_cohorts) / "cohorts.pt", Path(args.previous_cohorts) / "manifest.json",
                  Path(args.bank), Path(args.review) / "artifact.pt", Path(args.review) / "manifest.json", Path(args.history),
                  Path(args.prepared) / "metadata.pt", Path(args.base) / "hydrogen_analysis/labels_topology/annotations.pt"]
        inputs += [Path(__file__)] + [Path(__file__).parent / name for name in
                  ("data.py", "h_condinfer.py", "h_metadata.py", "h_reviewed_supervision.py", "h_observation_study.py")]
        before = {str(path): sha(path) for path in inputs}
        prepared_manifest = json.loads(inputs[1].read_text())
        if before[str(inputs[0])] != prepared_manifest["cache_sha256"] or before[str(inputs[8])] != prepared_manifest["metadata_sha256"]:
            raise ValueError("prepared_manifest_cache_or_metadata_sha_mismatch")
        history = json.loads(Path(args.history).read_text())
        for path, expected in history.get("evidence_inputs_sha256", {}).items():
            if sha(path) != expected:
                raise ValueError("history_evidence_sha_changed:" + path)
            before[path] = expected
        cache = torch.load(inputs[0], map_location="cpu", weights_only=False, mmap=True)
        previous_artifact = torch.load(inputs[2], map_location="cpu", weights_only=False, mmap=True)
        bank = torch.load(args.bank, map_location="cpu", weights_only=False, mmap=True)
        previous = previous_artifact["cohorts"]
        if len(previous["train"]) != 2048 or len(previous["original_val"]) != 320 or len(previous["tune"]) != 1024:
            raise ValueError("historical_cohort_count_drift")
        candidates = [bank["smiles"][i] for i in bank["single_indices"]]
        if len(candidates) != 114982 or len(bank["test"]) != 1000:
            raise ValueError("fixed_candidate_or_test_count_drift")
        old_manifest = json.loads(inputs[3].read_text())
        candidate_sha = digest("\n".join(candidates))
        test_sha = stamp_queries(bank["test"])
        if candidate_sha != old_manifest["candidate_order_sha256"] or test_sha != old_manifest["frozen_test_query_sha256"]:
            raise ValueError("fixed_candidate_or_test_identity_drift")
        group_map = {r["smiles"]: r["group"] for r in cache["rows"]}
        candidate_groups = {group_map[s] if s in group_map else identity(s)[1] for s in candidates}
        review, _ = load_artifact(args.review)
        labels = torch.load(inputs[9], map_location="cpu", weights_only=False, mmap=True)
        new_rows, review_audit = review_training_rows(cache["rows"], labels, review, apply_review)
        cohorts, contract = build_cohorts(new_rows, previous, candidate_groups, bank["test"], history,
                                         args.train_target, args.verify_target)
        if stamp_queries(cohorts["train"][:2048]) != old_manifest["cohorts"]["train"]["query_sha256"]:
            raise ValueError("training_prefix_spectrum_or_observation_changed")
        for name in ("original_val", "tune"):
            if stamp_queries(cohorts[name]) != old_manifest["cohorts"][name]["query_sha256"]:
                raise ValueError("validation_identity_or_spectrum_changed:" + name)
        contract.update({"candidate_count": len(candidates), "candidate_order_sha256": candidate_sha,
            "frozen_test_query_sha256": test_sha, "fixed_test_counts": {"normal": 1000, "crop": 200, "dropout": 171},
            "cohort_query_sha256": {name: stamp_queries(rr) for name, rr in cohorts.items()},
            "cohort_counts": {name: {"records": len(rr), "connectivities": len({r["group"] for r in rr}),
                "sources": dict(collections.Counter(r["source"] for r in rr))} for name, rr in cohorts.items()},
            "prepared_path": str(Path(args.prepared)), "prepared_cache_sha256": before[str(inputs[0])],
            "history_sha256": before[str(Path(args.history))], "review_audit": review_audit,
            "theory_source_counts": {source: {"eligible_records": len(ii), "eligible_connectivities": len({new_rows[i]["group"] for i in ii}),
                "measure": dict(collections.Counter(new_rows[i]["measure"] for i in ii)),
                "coverage": dict(collections.Counter(new_rows[i]["observation"]["kind"] for i in ii)),
                "assignment_trust": "not inferred from complete_atoms; requires reviewed site artifact"}
                for source, ii in contract["theory_indices_by_source"].items()},
            "input_sha256": before, "source_sha256": {str(Path(__file__)): sha(__file__)}})
        known_conditions = set(bank["predictions"])
        contract["condition_coverage"] = {}
        for name, rr in cohorts.items():
            keys = {condition_key(r, previous_artifact["metadata"], previous_artifact["key_mode"]) for r in rr}
            contract["condition_coverage"][name] = {"unique_conditions": len(keys),
                "condition_keys_sha256": digest(sorted(keys)),
                "missing_base_bank_conditions": [list(k) for k in sorted(keys - known_conditions)],
                "metadata_source": str(inputs[2]), "unknown_conditions_not_imputed": True}
        after = {path: sha(path) for path in before}
        if before != after:
            raise ValueError("cohort_input_changed_during_build")
        saved = {**previous_artifact, "cohorts": cohorts, "annotations": labels, "bank_path": args.bank,
                 "research_contract": str(out / "cohort_contract.json"),
                 "training_excluded_groups": contract["training_excluded_groups"],
                 "eligible_training_groups": contract["eligible_training_groups"]}
        torch.save(saved, out / "cohorts.pt")
        contract["cohorts_sha256"] = sha(out / "cohorts.pt")
        (out / "cohort_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        (out / "manifest.json").write_text(json.dumps({"cohorts": contract["cohort_counts"],
            "candidate_order_sha256": candidate_sha, "candidate_count": len(candidates),
            "frozen_test_query_sha256": test_sha, "contract_sha256": sha(out / "cohort_contract.json"),
            "cohorts_sha256": contract["cohorts_sha256"], "independent_verify": contract["independent_verify"],
            "input_sha256": before, "input_sha_start_end_equal": True, "external_api_calls": 0}, indent=2) + "\n")
        receipt.update(state="complete", ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time() - started,
                       actual_exit_code=0, input_sha_start_end_equal=True)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        (out / "EXIT_CODE").write_text("0\n")
        print(json.dumps({"state": "complete", "cohorts": contract["cohort_counts"], "verify": contract["independent_verify"]}), flush=True)
    except Exception as exc:
        receipt.update(state="failed", ended_utc=datetime.now(timezone.utc).isoformat(), error=repr(exc), actual_exit_code=1)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        (out / "EXIT_CODE").write_text("1\n")
        raise


if __name__ == "__main__":
    main()

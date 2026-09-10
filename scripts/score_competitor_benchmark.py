#!/usr/bin/env python3
"""功能概述：用 SpectraExpert 当前结构指标统一评分竞争方法在冻结 Raman 200 上的输出。
主要职责：load_export 验证统一数据导出；score_result 校验每个 query 的预测与候选身份，
按检索 Top-K 或生成 beam 计算 canonical/group 命中；score_structure 对各方法 top-1 统一计算
有效率、Morgan Tanimoto、legacy GED 与 chemical GED；main 合并多个结果并生成 JSON/Markdown 表。
代码库关系：输入兼容 repro/unified 的 Vib2Mol、VibraCLIP adapter JSONL；结构指标唯一复用
spectraexpert.eval.metrics，identity 口径与 run_closed_llm_direct.py 的冻结 group_hash 映射一致。
输入输出与副作用：只读 benchmark-dir 与 --result 文件；默认拒绝覆盖 output-dir。评分不会用
真值重排候选；GED 在隔离 worker 中限时运行，超时保留 null 并单列样本数。逐样本和汇总结果
通过临时文件原子写入，便于远端中断后判断完整性。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectraexpert.eval.metrics import graph_edit_distance, tanimoto_similarity


TOP_K = (1, 5, 10, 25)


def digest(path: Path) -> str:
    """流式返回文件 SHA-256。"""

    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """读取非空 JSONL 行。"""

    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical(smiles: object) -> str | None:
    """解析并返回含键阶的 canonical SMILES；无效输入返回 None。"""

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        molecule = Chem.MolFromSmiles(smiles)
    except (TypeError, ValueError):
        return None
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None


def group_identity(smiles: object) -> str | None:
    """按冻结协议取最大片段、忽略立体与同位素，供 group_hash 查表。"""

    value = canonical(smiles)
    if value is None:
        return None
    molecule = Chem.MolFromSmiles(value)
    assert molecule is not None
    fragments = Chem.GetMolFrags(molecule, asMols=True)
    largest = max(fragments, key=lambda item: (item.GetNumHeavyAtoms(), item.GetNumAtoms()))
    Chem.RemoveStereochemistry(largest)
    for atom in largest.GetAtoms():
        atom.SetIsotope(0)
    return Chem.MolToSmiles(largest, canonical=True)


def load_export(root: Path) -> tuple[dict, list[dict], list[dict]]:
    """验证导出协议与 JSONL 哈希、数量和 ID 唯一性。"""

    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "spectraexpert_competitor_raman_v1":
        raise ValueError("unsupported benchmark schema")
    for name in ("queries.jsonl", "pool.jsonl"):
        if digest(root / name) != protocol["outputs_sha256"][name]:
            raise ValueError(f"benchmark export hash mismatch: {name}")
    queries, pool = read_jsonl(root / "queries.jsonl"), read_jsonl(root / "pool.jsonl")
    if len(queries) != protocol["query_count"] or len(pool) != protocol["candidate_count"]:
        raise ValueError("benchmark count mismatch")
    if [row["query_id"] for row in queries] != protocol["query_ids"]:
        raise ValueError("query order mismatch")
    if [row["candidate_id"] for row in pool] != protocol["candidate_ids"]:
        raise ValueError("candidate order mismatch")
    return protocol, queries, pool


def score_structure(ground_truth: str, predicted_raw: object, ged_timeout: float) -> dict:
    """为一个 top-1 预测计算全部统一结构指标。"""

    predicted = canonical(predicted_raw)
    truth = canonical(ground_truth)
    if truth is None:
        raise ValueError(f"invalid frozen ground truth: {ground_truth!r}")
    if predicted is None:
        return {"valid": False, "predicted_canonical": None, "canonical_exact": False,
                "tanimoto": None, "legacy_ged_v1": None, "chemical_ged_v2": None}
    return {
        "valid": True,
        "predicted_canonical": predicted,
        "canonical_exact": predicted == truth,
        "tanimoto": tanimoto_similarity(truth, predicted),
        "legacy_ged_v1": graph_edit_distance(
            truth, predicted, timeout=ged_timeout, version="legacy_ged_v1"
        ),
        "chemical_ged_v2": graph_edit_distance(
            truth, predicted, timeout=ged_timeout, version="chemical_ged_v2"
        ),
    }


def summarize(rows: list[dict], arm: str) -> dict:
    """聚合统一结构指标与 arm 对应的 Top-K 命中。"""

    summary = {"n": len(rows), "arm": arm}
    for name in ("valid", "canonical_exact", "group_exact"):
        summary[name + "_rate"] = float(np.mean([bool(row[name]) for row in rows]))
    for name in ("tanimoto", "legacy_ged_v1", "chemical_ged_v2"):
        observed = [row[name] for row in rows if row[name] is not None]
        summary[name] = {
            "n": len(observed),
            "mean": float(np.mean(observed)) if observed else None,
            "median": float(np.median(observed)) if observed else None,
        }
    for limit in TOP_K:
        summary[f"canonical_hit_at_{limit}"] = float(
            np.mean([row[f"canonical_hit_at_{limit}"] for row in rows])
        )
        summary[f"group_hit_at_{limit}"] = float(
            np.mean([row[f"group_hit_at_{limit}"] for row in rows])
        )
    return summary


def score_result(
    tag: str,
    result_path: Path,
    queries: list[dict],
    pool: list[dict],
    ged_timeout: float,
    workers: int,
) -> tuple[list[dict], dict]:
    """校验并评分单个 adapter 输出，返回逐样本与汇总。"""

    raw_rows = read_jsonl(result_path)
    expected_ids = [query["query_id"] for query in queries]
    if [row.get("query_id") for row in raw_rows] != expected_ids:
        raise ValueError(f"{tag}: result query order/count mismatch")
    if any(row.get("skipped") for row in raw_rows):
        raise ValueError(f"{tag}: formal benchmark cannot contain skipped queries")
    arms = {row.get("arm") for row in raw_rows}
    if len(arms) != 1 or next(iter(arms)) not in {"retrieval", "generation"}:
        raise ValueError(f"{tag}: invalid or mixed arm")
    arm = next(iter(arms))
    candidate_by_id = {row["candidate_id"]: row for row in pool}
    group_by_identity: dict[str, str] = {}
    for candidate in pool:
        identity = group_identity(candidate["smiles"])
        if identity is None:
            raise ValueError(f"invalid frozen candidate: {candidate['candidate_id']}")
        previous = group_by_identity.setdefault(identity, candidate["group_hash"])
        if previous != candidate["group_hash"]:
            raise ValueError(f"ambiguous group identity: {identity}")

    prepared = []
    for query, raw in zip(queries, raw_rows):
        if arm == "retrieval":
            ranked = raw.get("ranked") or []
            if len(ranked) < max(TOP_K):
                raise ValueError(f"{tag}/{query['query_id']}: fewer than {max(TOP_K)} ranked candidates")
            if len({item["candidate_id"] for item in ranked}) != len(ranked):
                raise ValueError(f"{tag}/{query['query_id']}: duplicate ranked candidate")
            predictions = []
            groups = []
            for item in ranked:
                candidate = candidate_by_id.get(item["candidate_id"])
                if candidate is None:
                    raise ValueError(f"{tag}: unknown candidate {item['candidate_id']}")
                if item.get("candidate_smiles") != candidate["smiles"]:
                    raise ValueError(f"{tag}: candidate SMILES mismatch {item['candidate_id']}")
                predictions.append(candidate["smiles"])
                groups.append(candidate["group_hash"])
        else:
            predictions = list(raw.get("beam_smiles") or [])
            greedy = raw.get("greedy_smiles")
            if not predictions:
                predictions = [greedy]
            groups = [group_by_identity.get(group_identity(value)) for value in predictions]

        truth = canonical(query["target_smiles"])
        assert truth is not None
        canonical_predictions = [canonical(value) for value in predictions]
        top1 = predictions[0] if predictions else None
        prepared.append(
            {
                "query_id": query["query_id"],
                "mol_id": query["mol_id"],
                "ground_truth_smiles": query["target_smiles"],
                "group_hash": query["group_hash"],
                "top1_raw": top1,
                "group_exact": bool(groups and groups[0] == query["group_hash"]),
                **{
                    f"canonical_hit_at_{limit}": truth in canonical_predictions[:limit]
                    for limit in TOP_K
                },
                **{
                    f"group_hit_at_{limit}": query["group_hash"] in groups[:limit]
                    for limit in TOP_K
                },
            }
        )

    def calculate(item: dict) -> dict:
        structure = score_structure(item["ground_truth_smiles"], item["top1_raw"], ged_timeout)
        return item | structure

    with ThreadPoolExecutor(max_workers=workers) as executor:
        scored = list(executor.map(calculate, prepared))
    summary = summarize(scored, arm)
    summary.update(
        tag=tag,
        method=raw_rows[0].get("method"),
        modality=raw_rows[0].get("modality"),
        primary_prediction=("ranked[0]" if arm == "retrieval" else "beam_smiles[0]"),
        result_path=str(result_path),
        result_sha256=digest(result_path),
    )
    return scored, summary


def write_json(path: Path, value: object) -> None:
    """原子写入人类可读 JSON。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def markdown_table(summaries: list[dict]) -> str:
    """生成审稿回复可直接复用的紧凑结果表。"""

    header = (
        "| 方法 | arm | Valid | Group@1 | Canonical@1 | Group@5 | Group@10 | "
        "Morgan Tanimoto | chemical GED (n) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for row in summaries:
        chemical = row["chemical_ged_v2"]
        lines.append(
            f"| {row['tag']} | {row['arm']} | {100*row['valid_rate']:.1f}% | "
            f"{100*row['group_hit_at_1']:.1f}% | {100*row['canonical_hit_at_1']:.1f}% | "
            f"{100*row['group_hit_at_5']:.1f}% | {100*row['group_hit_at_10']:.1f}% | "
            f"{row['tanimoto']['mean']:.4f} | "
            f"{chemical['mean']:.4f} ({chemical['n']}) |"
        )
    return header + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--result",
        required=True,
        action="append",
        help="TAG=/absolute/path/to/adapter.jsonl，可重复",
    )
    parser.add_argument("--ged-timeout", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol, queries, pool = load_export(args.benchmark_dir)
    parsed = []
    for item in args.result:
        tag, separator, path = item.partition("=")
        if not separator or not tag or not path:
            raise ValueError(f"invalid --result: {item!r}")
        parsed.append((tag, Path(path)))
    args.output_dir.mkdir(parents=True)
    summaries = []
    for tag, path in parsed:
        scored, summary = score_result(tag, path, queries, pool, args.ged_timeout, args.workers)
        write_json(args.output_dir / f"{tag}.json", scored)
        summaries.append(summary)
        write_json(args.output_dir / "summary.partial.json", summaries)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    report = {
        "schema_version": "spectraexpert_competitor_score_v1",
        "benchmark_protocol_sha256": digest(args.benchmark_dir / "protocol.json"),
        "metric_source_sha256": digest(Path(__file__).resolve().parents[1] / "spectraexpert/eval/metrics.py"),
        "scorer_sha256": digest(Path(__file__)),
        "ged_timeout_seconds": args.ged_timeout,
        "ged_workers": args.workers,
        "metric_definitions": {
            "identity": "frozen group_hash; largest fragment, no stereochemistry/isotopes",
            "canonical_exact": "RDKit canonical SMILES equality",
            "tanimoto": "Morgan radius=2, 2048 bits, bond types, no chirality",
            "chemical_ged_v2": "atomic-number nodes + discrete bond types; completed bounded search",
            "legacy_ged_v1": "historical first optimize_graph_edit_distance upper bound",
            "missing_ged": "timeout/invalid retained as null and excluded from mean; n reported",
        },
        "summaries": summaries,
    }
    write_json(args.output_dir / "summary.json", report)
    (args.output_dir / "table.md").write_text(markdown_table(summaries), encoding="utf-8")
    (args.output_dir / "summary.partial.json").unlink()


if __name__ == "__main__":
    main()

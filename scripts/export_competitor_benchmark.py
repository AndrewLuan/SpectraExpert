#!/usr/bin/env python3
"""功能概述：把冻结的 Raman 10k/200 闭库数据导出为竞争方法共用的轻量 JSONL 协议。
主要职责：digest 流式计算文件哈希；load_role 校验 manifest、rows、Parquet 的 mol_id 与
group_hash 顺序；export_benchmark 写 200 条实验 Raman query、10,000 条候选结构和审计清单。
代码库关系：输入由 repair_closed_raman_exact.py 生成，输出直接供 JJ 上
repro/unified/adapters/vib2mol_adapter.py 与 vibraclip_adapter.py 使用；本文件不执行模型推理，
也不改变各方法自己的插值和归一化规则。
输入输出与副作用：只读指定 data-root；默认拒绝覆盖 output-dir。query 仅含原始 500--4000
cm^-1 Raman 强度、真值结构和冻结身份（真值只供推理后评分），不导出分子式或质量，避免先验泄漏；
pool 只含固定候选 ID、SMILES、mol_id 与 group_hash。写入过程先落临时文件再原子替换。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    """流式返回文件 SHA-256。"""

    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """以 UTF-8 JSONL 原子写入，避免中断留下貌似完整的文件。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_role(root: Path, role: str, manifest: dict) -> tuple[pd.DataFrame, np.ndarray | None]:
    """校验并按 rows.json 的冻结顺序返回 metadata 与可选光谱矩阵。"""

    folder = root / role
    required = ["smiles.parquet", "rows.json"]
    if role == "experimental_test":
        required += ["spectra.parquet", "wavenumber.parquet"]
    for name in required:
        expected = manifest["outputs_sha256"][role][name]
        actual = digest(folder / name)
        if actual != expected:
            raise ValueError(f"source hash mismatch: {role}/{name}: {actual} != {expected}")

    frozen_rows = json.loads((folder / "rows.json").read_text(encoding="utf-8"))
    ids = [int(row["mol_id"]) for row in frozen_rows]
    metadata = pd.read_parquet(folder / "smiles.parquet")
    if len(ids) != len(set(ids)) or metadata["mol_id"].duplicated().any():
        raise ValueError(f"duplicate mol_id in {role}")
    if set(ids) != set(metadata["mol_id"]):
        raise ValueError(f"metadata mol_id mismatch in {role}")
    metadata = metadata.set_index("mol_id").loc[ids].reset_index()
    if metadata["group_hash"].tolist() != [row["group_hash"] for row in frozen_rows]:
        raise ValueError(f"frozen group order mismatch in {role}")

    if role != "experimental_test":
        return metadata, None
    axis = pd.read_parquet(folder / "wavenumber.parquet")["wavenumber"].to_numpy(np.float32)
    if not np.array_equal(axis, np.arange(500, 4001, dtype=np.float32)):
        raise ValueError("expected frozen Raman axis 500..4000 cm^-1 at 1 cm^-1")
    spectra = pd.read_parquet(folder / "spectra.parquet").set_index("mol_id")
    columns = [f"w_{int(value)}" for value in axis]
    matrix = spectra.loc[ids, columns].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError("experimental spectra contain non-finite or empty rows")
    return metadata, matrix


def export_benchmark(data_root: Path, output_dir: Path, limit: int | None = None) -> dict:
    """导出统一 query/pool，并返回包含来源与输出哈希的协议清单。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest_path = data_root / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "raman_closed_library_sample_v1":
        raise ValueError("requires raman_closed_library_sample_v1")
    candidates, _ = load_role(data_root, "theory_train_candidate", manifest)
    queries, spectra = load_role(data_root, "experimental_test", manifest)
    assert spectra is not None
    if candidates["group_hash"].tolist() != manifest["candidate_groups"]:
        raise ValueError("candidate order does not match sample manifest")
    if queries["group_hash"].tolist() != manifest["test_groups"]:
        raise ValueError("query order does not match sample manifest")
    if len(candidates) != 10_000 or len(queries) != 200:
        raise ValueError("expected exact frozen 10,000 candidates and 200 queries")
    candidate_by_id = candidates.set_index("mol_id")
    for row in queries.itertuples():
        if candidate_by_id.loc[row.mol_id, "group_hash"] != row.group_hash:
            raise ValueError(f"query identity missing from candidate pool: mol_id={row.mol_id}")
        if "/exp/raman/" not in row.source_path:
            raise ValueError(f"query is not experimental Raman: mol_id={row.mol_id}")

    count = len(queries) if limit is None else min(int(limit), len(queries))
    if count <= 0:
        raise ValueError("limit must be positive")
    axis = np.arange(500, 4001, dtype=np.float32).tolist()
    query_rows = []
    for index, (row, spectrum) in enumerate(zip(queries.iloc[:count].itertuples(), spectra[:count])):
        query_rows.append(
            {
                "query_id": f"q{index:04d}",
                "mol_id": int(row.mol_id),
                "target_smiles": row.smiles,
                "group_hash": row.group_hash,
                "source": row.source,
                "source_detail": row.condition,
                "raman": {"wn": axis, "intensity": spectrum.astype(float).tolist()},
            }
        )
    pool_rows = [
        {
            "candidate_id": f"m{int(row.mol_id):05d}",
            "mol_id": int(row.mol_id),
            "smiles": row.smiles,
            "group_hash": row.group_hash,
        }
        for row in candidates.itertuples()
    ]

    output_dir.mkdir(parents=True)
    queries_path = output_dir / "queries.jsonl"
    pool_path = output_dir / "pool.jsonl"
    write_jsonl(queries_path, query_rows)
    write_jsonl(pool_path, pool_rows)
    protocol = {
        "schema_version": "spectraexpert_competitor_raman_v1",
        "data_root": str(data_root),
        "sample_manifest_sha256": digest(manifest_path),
        "source_sha256": manifest["outputs_sha256"],
        "query_count": count,
        "candidate_count": len(pool_rows),
        "query_ids": [row["query_id"] for row in query_rows],
        "query_mol_ids": [row["mol_id"] for row in query_rows],
        "candidate_ids": [row["candidate_id"] for row in pool_rows],
        "modality": "experimental Raman",
        "axis": {"start_cm-1": 500, "end_cm-1": 4000, "step_cm-1": 1, "points": 3501},
        "model_input_policy": "same raw query JSONL; each official adapter applies its documented preprocessing",
        "leakage_policy": "target SMILES/group are scoring-only; no target formula or mass exported",
        "outputs_sha256": {"queries.jsonl": digest(queries_path), "pool.jsonl": digest(pool_path)},
    }
    temporary = output_dir / "protocol.json.tmp"
    temporary.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output_dir / "protocol.json")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="仅用于 smoke；正式评测省略")
    args = parser.parse_args()
    print(json.dumps(export_benchmark(args.data_root, args.output_dir, args.limit), indent=2))


if __name__ == "__main__":
    main()

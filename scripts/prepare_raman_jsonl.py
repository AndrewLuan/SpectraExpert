#!/usr/bin/env python3
"""
功能概述：只把两个明确获授权的 Raman JSONL 转为统一 500..4000/3501 点的标准三件套 Parquet。
主要职责：``prepare_raman_jsonl`` 校验解析后的输入绝对路径、逐行解析 JSON、规范化 SMILES、验证
x/y 长度/有限性/严格单调和覆盖边界，再用 ``numpy.interp`` 线性插值；同时写 source hash、原始
坐标范围、插值参数、split manifest 和 preparation metadata。``main`` 提供无目录扫描的显式 CLI。
代码库关系：canonical identity 与 scaffold manifest 复用 ``structure_dataset``；输出的
``spectra.parquet``、``smiles.parquet``、``wavenumber.parquet`` 由结构感知 Dataset 和后续消融
共同消费。生产输入范围只限 ``/data/SpecGPT/data/theory_data_tiny`` 下的两个授权文件；仅 Python
测试调用可注入 ``allowed_input_paths``，CLI 不提供该参数，记录数不硬编码为 400。
输入输出与副作用：只逐行读取两个显式文件，不遍历其父目录；默认拒绝任何已存在的 output_dir。
全部记录先通过校验后才在同级 staging 目录写文件，成功后一次目录 rename 发布，失败清理 staging，
因此非法 JSON/SMILES/坐标或 I/O 失败不会留下可被误用的 partial 数据集；重复 identity 作为不同
来源行保留，并以不同 ``mol_id`` 参与同一 scaffold group。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spectraexpert.data.structure_dataset import canonicalize_smiles, scaffold_split


AUTHORIZED_INPUT_PATHS = (
    Path("/data/SpecGPT/data/theory_data_tiny/Raman_QME14S.jsonl"),
    Path("/data/SpecGPT/data/theory_data_tiny/Raman_fg26.jsonl"),
)
AUTHORIZED_BASENAMES = frozenset(path.name for path in AUTHORIZED_INPUT_PATHS)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_json_lines(path: Path) -> Iterator[tuple[int, Mapping[str, object]]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"blank JSONL line in {path.name}:{line_number}")
            try:
                decoded = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSON in {path.name}:{line_number}") from error
            if not isinstance(decoded, dict):
                raise ValueError(f"JSONL record must be an object in {path.name}:{line_number}")
            yield line_number, decoded


def _one_present(record: Mapping[str, object], keys: Sequence[str], label: str) -> object:
    present = [key for key in keys if key in record]
    if len(present) != 1:
        raise ValueError(f"record requires exactly one {label} field from {list(keys)}")
    return record[present[0]]


def _extract_record(
    record: Mapping[str, object],
    *,
    basename: str,
    line_number: int,
) -> tuple[str, np.ndarray, np.ndarray]:
    smiles = _one_present(record, ("smiles", "SMILES", "smiles_canonical"), "SMILES")
    spectrum_payload = record.get("raman_spectrum")
    coordinate_source: Mapping[str, object] = (
        spectrum_payload if isinstance(spectrum_payload, dict) else record
    )
    intensity = _one_present(
        coordinate_source, ("intensity", "intensities", "y", "Raman"), "intensity"
    )
    axis_keys = ("wavenumber", "wavenumbers", "x")
    present_axis = [key for key in axis_keys if key in coordinate_source]
    if len(present_axis) > 1:
        raise ValueError(f"record has ambiguous wavenumber fields in {basename}:{line_number}")
    try:
        y = np.asarray(intensity, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"intensity must be numeric in {basename}:{line_number}") from error
    if present_axis:
        try:
            x = np.asarray(coordinate_source[present_axis[0]], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"wavenumber must be numeric in {basename}:{line_number}") from error
    elif basename == "Raman_QME14S.jsonl" and y.ndim == 1 and y.size == 3500:
        x = np.linspace(500.0, 4000.0, 3500, dtype=np.float64)
    elif basename == "Raman_fg26.jsonl" and y.ndim == 1 and y.size == 4000:
        x = np.arange(1.0, 4001.0, dtype=np.float64)
    else:
        raise ValueError(f"record requires a wavenumber field in {basename}:{line_number}")
    return str(smiles) if isinstance(smiles, str) else smiles, x, y


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _validate_inputs(
    inputs: Sequence[str | Path],
    *,
    allowed_input_paths: Sequence[str | Path] | None = None,
) -> list[Path]:
    paths = [Path(value) for value in inputs]
    allowed = [
        Path(value)
        for value in (
            AUTHORIZED_INPUT_PATHS
            if allowed_input_paths is None
            else allowed_input_paths
        )
    ]
    if (
        len(paths) != 2
        or len(allowed) != 2
        or {path.name for path in paths} != AUTHORIZED_BASENAMES
        or {path.name for path in allowed} != AUTHORIZED_BASENAMES
    ):
        expected = ", ".join(sorted(AUTHORIZED_BASENAMES))
        raise ValueError(f"inputs must be exactly authorized Raman JSONL files: {expected}")
    allowed_by_name = {
        path.name: _absolute_without_resolving(path) for path in allowed
    }
    if len(allowed_by_name) != 2:
        raise ValueError("allowed_input_paths must name two distinct authorized files")
    resolved_paths: list[Path] = []
    for path in paths:
        absolute = _absolute_without_resolving(path)
        allowed_absolute = allowed_by_name[path.name]
        if absolute != allowed_absolute:
            raise ValueError(
                "input is not an authorized resolved input path: "
                f"{absolute}"
            )
        try:
            resolved = absolute.resolve(strict=True)
            allowed_resolved = allowed_absolute.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"authorized Raman JSONL input does not exist: {absolute}"
            ) from error
        if absolute != resolved or allowed_absolute != allowed_resolved:
            raise ValueError(f"authorized Raman JSONL path must not use a symlink: {absolute}")
        if resolved != allowed_resolved or not resolved.is_file():
            raise ValueError(
                "input is not an authorized resolved input path: "
                f"{resolved}"
            )
        resolved_paths.append(resolved)
    if len(set(resolved_paths)) != 2:
        raise ValueError("authorized Raman JSONL inputs must be two distinct files")
    return resolved_paths


def _column_names(axis: np.ndarray) -> list[str]:
    names = [f"w_{float(value):g}" for value in axis]
    if len(names) != len(set(names)):
        raise ValueError("output wavenumber values do not produce unique column names")
    return names


def _write_outputs(
    staging: Path,
    *,
    axis: np.ndarray,
    spectra: np.ndarray,
    smiles_rows: list[dict[str, object]],
    split_seed: int,
    metadata: dict[str, object],
) -> None:
    columns = _column_names(axis)
    spectra_schema = pa.schema(
        [pa.field("mol_id", pa.int64(), nullable=False)]
        + [pa.field(column, pa.float32(), nullable=False) for column in columns]
    )
    spectra_arrays = [pa.array(np.arange(len(spectra), dtype=np.int64), type=pa.int64())]
    spectra_arrays.extend(
        pa.array(spectra[:, index], type=pa.float32()) for index in range(spectra.shape[1])
    )
    pq.write_table(
        pa.Table.from_arrays(spectra_arrays, schema=spectra_schema),
        staging / "spectra.parquet",
        compression="zstd",
        version="2.6",
    )

    smiles_schema = pa.schema(
        [
            pa.field("mol_id", pa.int64(), nullable=False),
            pa.field("smiles_canonical", pa.string(), nullable=False),
            pa.field("source_basename", pa.string(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
            pa.field("source_sha256", pa.string(), nullable=False),
            pa.field("source_line", pa.int64(), nullable=False),
            pa.field("original_start", pa.float64(), nullable=False),
            pa.field("original_stop", pa.float64(), nullable=False),
            pa.field("original_points", pa.int64(), nullable=False),
            pa.field("interpolation", pa.string(), nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(smiles_rows, schema=smiles_schema),
        staging / "smiles.parquet",
        compression="zstd",
        version="2.6",
    )
    axis_schema = pa.schema([pa.field("wavenumber", pa.float32(), nullable=False)])
    pq.write_table(
        pa.Table.from_arrays([pa.array(axis, type=pa.float32())], schema=axis_schema),
        staging / "wavenumber.parquet",
        compression="zstd",
        version="2.6",
    )
    scaffold_split(
        [row["smiles_canonical"] for row in smiles_rows],
        mol_ids=[int(row["mol_id"]) for row in smiles_rows],
        seed=split_seed,
        manifest_path=staging / "split_manifest.json",
    )
    (staging / "preparation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in staging.iterdir():
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())


def prepare_raman_jsonl(
    inputs: Sequence[str | Path],
    output_dir: str | Path,
    start: float = 500.0,
    stop: float = 4000.0,
    points: int = 3501,
    split_seed: int = 0,
    *,
    allowed_input_paths: Sequence[str | Path] | None = None,
) -> dict[str, object]:
    """转换恰好两个白名单 JSONL；返回记录数、唯一 identity 数和输出轴摘要。"""

    paths = _validate_inputs(inputs, allowed_input_paths=allowed_input_paths)
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"output directory already exists: {output_path}")
    if not np.isfinite([start, stop]).all() or start >= stop:
        raise ValueError("start and stop must be finite with start < stop")
    if isinstance(points, bool) or int(points) != points or points < 2:
        raise ValueError("points must be an integer of at least two")
    points = int(points)
    target_axis = np.linspace(start, stop, points, dtype=np.float64)

    source_hashes = {path: _file_sha256(path) for path in paths}
    interpolated_rows: list[np.ndarray] = []
    smiles_rows: list[dict[str, object]] = []
    canonical_identities: set[str] = set()
    source_counts: dict[str, int] = {path.name: 0 for path in paths}
    for path in paths:
        for line_number, record in _iter_json_lines(path):
            raw_smiles, x, y = _extract_record(
                record, basename=path.name, line_number=line_number
            )
            if x.ndim != 1 or y.ndim != 1:
                raise ValueError(f"x/y must be one-dimensional in {path.name}:{line_number}")
            if x.size != y.size:
                raise ValueError(f"x/y must have equal length in {path.name}:{line_number}")
            if x.size < 2:
                raise ValueError(f"x/y require at least two points in {path.name}:{line_number}")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"non-finite x/y in {path.name}:{line_number}")
            if not np.all(np.diff(x) > 0):
                raise ValueError(
                    f"x must be strictly increasing without duplicates in {path.name}:{line_number}"
                )
            if x[0] > start or x[-1] < stop:
                raise ValueError(
                    f"source axis does not cover [{start}, {stop}] in {path.name}:{line_number}"
                )
            canonical = canonicalize_smiles(raw_smiles)
            canonical_identities.add(canonical)
            mol_id = len(smiles_rows)
            interpolated = np.interp(target_axis, x, y).astype(np.float32)
            if not np.isfinite(interpolated).all():
                raise ValueError(f"non-finite interpolated spectrum in {path.name}:{line_number}")
            interpolated_rows.append(interpolated)
            smiles_rows.append(
                {
                    "mol_id": mol_id,
                    "smiles_canonical": canonical,
                    "source_basename": path.name,
                    "source_path": str(path.resolve()),
                    "source_sha256": source_hashes[path],
                    "source_line": line_number,
                    "original_start": float(x[0]),
                    "original_stop": float(x[-1]),
                    "original_points": int(x.size),
                    "interpolation": "linear",
                }
            )
            source_counts[path.name] += 1
    if not smiles_rows:
        raise ValueError("authorized Raman JSONL inputs contain no records")

    spectra = np.stack(interpolated_rows).astype(np.float32, copy=False)
    source_metadata = [
        {
            "basename": path.name,
            "path": str(path.resolve()),
            "sha256": source_hashes[path],
            "records": source_counts[path.name],
        }
        for path in paths
    ]
    metadata: dict[str, object] = {
        "schema_version": "raman_jsonl_preparation_v1",
        "record_count": len(smiles_rows),
        "unique_canonical_smiles": len(canonical_identities),
        "sources": source_metadata,
        "interpolation": {
            "method": "numpy.interp",
            "start": float(start),
            "stop": float(stop),
            "points": points,
        },
        "split_seed": int(split_seed),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    )
    try:
        _write_outputs(
            staging,
            axis=target_axis.astype(np.float32),
            spectra=spectra,
            smiles_rows=smiles_rows,
            split_seed=int(split_seed),
            metadata=metadata,
        )
        if output_path.exists():
            raise FileExistsError(f"output directory already exists: {output_path}")
        os.rename(staging, output_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "records": len(smiles_rows),
        "unique_canonical_smiles": len(canonical_identities),
        "start": float(start),
        "stop": float(stop),
        "points": points,
        "output_dir": str(output_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare authorized Raman JSONL files")
    parser.add_argument("--input", dest="inputs", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", type=float, default=500.0)
    parser.add_argument("--stop", type=float, default=4000.0)
    parser.add_argument("--points", type=int, default=3501)
    parser.add_argument("--split-seed", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = prepare_raman_jsonl(
        args.inputs,
        args.output_dir,
        start=args.start,
        stop=args.stop,
        points=args.points,
        split_seed=args.split_seed,
    )
    print(
        f"prepared {summary['records']} unique Raman records at "
        f"{summary['output_dir']} ({summary['points']} points)"
    )


if __name__ == "__main__":
    main()

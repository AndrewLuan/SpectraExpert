"""
功能概述：验证两个获授权 Raman JSONL 的逐行转换、统一插值、来源审计与原子化输出行为。
主要职责：覆盖 QME14S 3500→3501、fg26 1..499 截除、固定白名单、非单调/重复坐标、
x/y 长度、NaN/Inf、canonical 重复、PyArrow dtype、已有输出拒绝和失败不留半成品。
代码库关系：直接调用 ``scripts.prepare_raman_jsonl.prepare_raman_jsonl``；输出三件套供
``StructureAwareSpectralDataset`` 使用，split manifest 供后续全部消融复用。
输入输出与副作用：只创建名称与获授权文件相同的临时微型 JSONL，不访问本地或服务器大数据；
转换成功写小型 Parquet/JSON，验证失败时输出目录必须不存在。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.prepare_raman_jsonl import prepare_raman_jsonl


def _record(smiles: str, x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    return {"smiles": smiles, "wavenumber": x.tolist(), "intensity": y.tolist()}


def _write_lines(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, allow_nan=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _valid_input_paths(tmp_path: Path) -> tuple[Path, Path]:
    qme_path = tmp_path / "Raman_QME14S.jsonl"
    fg_path = tmp_path / "Raman_fg26.jsonl"
    qme_x = np.linspace(500.0, 4000.0, 3500)
    fg_x = np.arange(1.0, 4001.0)
    _write_lines(qme_path, [_record("CC", qme_x, qme_x)])
    _write_lines(fg_path, [_record("O", fg_x, 2.0 * fg_x)])
    return qme_path, fg_path


def test_prepare_interpolates_both_authorized_sources_to_one_axis(tmp_path):
    qme_path, fg_path = _valid_input_paths(tmp_path)
    output_dir = tmp_path / "prepared"

    summary = prepare_raman_jsonl(
        inputs=[qme_path, fg_path],
        output_dir=output_dir,
        start=500.0,
        stop=4000.0,
        points=3501,
        split_seed=0,
        allowed_input_paths=[qme_path, fg_path],
    )

    axis = pd.read_parquet(output_dir / "wavenumber.parquet")["wavenumber"]
    spectra = pd.read_parquet(output_dir / "spectra.parquet")
    smiles = pd.read_parquet(output_dir / "smiles.parquet")
    assert summary["records"] == 2
    assert len(axis) == 3501
    assert axis.iloc[0] == pytest.approx(500.0)
    assert axis.iloc[-1] == pytest.approx(4000.0)
    assert spectra.shape == (2, 3502)
    assert spectra.columns[0] == "mol_id"
    assert spectra.iloc[0, 1] == pytest.approx(500.0)
    assert spectra.iloc[0, -1] == pytest.approx(4000.0)
    assert spectra.iloc[1, 1] == pytest.approx(1000.0)
    assert spectra.iloc[1, -1] == pytest.approx(8000.0)
    assert smiles["smiles_canonical"].tolist() == ["CC", "O"]
    assert smiles["smiles_canonical"].is_unique
    assert smiles["source_basename"].tolist() == [
        "Raman_QME14S.jsonl", "Raman_fg26.jsonl"
    ]
    assert all(smiles["source_sha256"].str.fullmatch(r"[0-9a-f]{64}"))
    assert smiles["interpolation"].tolist() == ["linear", "linear"]
    assert (output_dir / "split_manifest.json").is_file()
    manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in ("spectra.parquet", "smiles.parquet", "wavenumber.parquet")
    }


def test_prepare_writes_explicit_arrow_dtypes_and_audit_metadata(tmp_path):
    qme_path, fg_path = _valid_input_paths(tmp_path)
    output_dir = tmp_path / "prepared"

    prepare_raman_jsonl(
        [qme_path, fg_path], output_dir,
        allowed_input_paths=[qme_path, fg_path],
    )

    spectra_schema = pq.read_schema(output_dir / "spectra.parquet")
    smiles_schema = pq.read_schema(output_dir / "smiles.parquet")
    axis_schema = pq.read_schema(output_dir / "wavenumber.parquet")
    assert spectra_schema.field("mol_id").type == pa.int64()
    assert all(field.type == pa.float32() for field in list(spectra_schema)[1:])
    assert axis_schema.field("wavenumber").type == pa.float32()
    assert smiles_schema.field("mol_id").type == pa.int64()
    metadata = json.loads(
        (output_dir / "preparation_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["interpolation"] == {
        "method": "numpy.interp",
        "points": 3501,
        "start": 500.0,
        "stop": 4000.0,
    }
    assert {source["basename"] for source in metadata["sources"]} == {
        "Raman_QME14S.jsonl", "Raman_fg26.jsonl"
    }


def test_prepare_rejects_any_non_authorized_or_missing_input_basename(tmp_path):
    qme_path, fg_path = _valid_input_paths(tmp_path)
    unauthorized = tmp_path / "other.jsonl"
    unauthorized.write_text(qme_path.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="authorized Raman JSONL"):
        prepare_raman_jsonl(
            [qme_path, unauthorized], tmp_path / "out1",
            allowed_input_paths=[qme_path, fg_path],
        )
    with pytest.raises(ValueError, match="exactly.*Raman_QME14S.*Raman_fg26"):
        prepare_raman_jsonl(
            [fg_path], tmp_path / "out2",
            allowed_input_paths=[qme_path, fg_path],
        )


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        ([500.0, 499.0, 4000.0], [1.0, 2.0, 3.0], "strictly increasing"),
        ([500.0, 500.0, 4000.0], [1.0, 2.0, 3.0], "strictly increasing"),
        ([500.0, 4000.0], [1.0], "equal length"),
        ([500.0, float("nan"), 4000.0], [1.0, 2.0, 3.0], "non-finite"),
        ([500.0, 2000.0, 4000.0], [1.0, float("inf"), 3.0], "non-finite"),
    ],
)
def test_prepare_rejects_invalid_coordinates_or_intensities(tmp_path, x, y, message):
    qme_path, fg_path = _valid_input_paths(tmp_path)
    _write_lines(qme_path, [{"smiles": "CC", "wavenumber": x, "intensity": y}])
    output_dir = tmp_path / "prepared"

    with pytest.raises(ValueError, match=message):
        prepare_raman_jsonl(
            [qme_path, fg_path], output_dir,
            allowed_input_paths=[qme_path, fg_path],
        )

    assert not output_dir.exists()


def test_prepare_keeps_duplicate_canonical_smiles_as_distinct_observations(tmp_path):
    qme_path, fg_path = _valid_input_paths(tmp_path)
    fg_x = np.arange(1.0, 4001.0)
    _write_lines(fg_path, [_record("CC", fg_x, fg_x)])
    output_dir = tmp_path / "prepared"

    summary = prepare_raman_jsonl(
        [qme_path, fg_path], output_dir,
        allowed_input_paths=[qme_path, fg_path],
    )

    smiles = pd.read_parquet(output_dir / "smiles.parquet")
    manifest = json.loads(
        (output_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert summary["records"] == 2
    assert summary["unique_canonical_smiles"] == 1
    assert smiles["mol_id"].tolist() == [0, 1]
    assert smiles["smiles_canonical"].tolist() == ["CC", "CC"]
    owner = {
        mol_id: split
        for split in ("train", "validation", "test")
        for mol_id in manifest[split]
    }
    assert owner[0] == owner[1]


def test_default_whitelist_rejects_same_basenames_outside_authorized_directory(tmp_path):
    qme_path, fg_path = _valid_input_paths(tmp_path)

    with pytest.raises(ValueError, match="authorized resolved input path"):
        prepare_raman_jsonl([qme_path, fg_path], tmp_path / "prepared")


def test_prepare_accepts_explicitly_injected_test_paths(tmp_path):
    qme_path, fg_path = _valid_input_paths(tmp_path)

    summary = prepare_raman_jsonl(
        [qme_path, fg_path],
        tmp_path / "prepared",
        allowed_input_paths=[qme_path, fg_path],
    )

    assert summary["records"] == 2


def test_prepare_rejects_symlink_alias_even_with_injected_target_whitelist(tmp_path):
    qme_path, fg_path = _valid_input_paths(tmp_path)
    alias_dir = tmp_path / "aliases"
    alias_dir.mkdir()
    qme_alias = alias_dir / "Raman_QME14S.jsonl"
    qme_alias.symlink_to(qme_path)

    with pytest.raises(ValueError, match="symlink|authorized resolved input path"):
        prepare_raman_jsonl(
            [qme_alias, fg_path],
            tmp_path / "prepared",
            allowed_input_paths=[qme_path, fg_path],
        )


def test_prepare_refuses_to_overwrite_existing_output(tmp_path):
    qme_path, fg_path = _valid_input_paths(tmp_path)
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output directory already exists"):
        prepare_raman_jsonl(
            [qme_path, fg_path], output_dir,
            allowed_input_paths=[qme_path, fg_path],
        )

    assert marker.read_text(encoding="utf-8") == "do not replace"

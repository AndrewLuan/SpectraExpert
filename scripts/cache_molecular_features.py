#!/usr/bin/env python3
"""
功能概述：把显式 SMILES Parquet 批量转换为版本化、可审计且可原子发布的分子图特征缓存。
主要职责：``build_feature_cache`` 先校验全部 mol_id/SMILES 并计算 canonical source hash，再按批
调用 Task 2 ``featurize_smiles``，写固定 PyArrow nested schema；``main`` 仅接受调用方明确给出的
``--smiles-path`` 与 ``--output``，不会搜索输入目录或推断数据文件。
代码库关系：缓存 schema/hash 由 ``spectraexpert.data.structure_dataset`` 唯一定义，节点、边、
Morgan fingerprint 与 descriptors 唯一来自 ``molecular_graph``/``structure_features``；生成文件
由 StructureAwareSpectralDataset、图索引构建和 ``load_feature_cache`` 共同消费。
输入输出与副作用：读取单个显式 Parquet，输出一个 metadata 完整的 feature Parquet；目标已存在、
重复 ID、非法 SMILES 或写入失败时 fail fast。写入只发生在同目录临时文件，成功 fsync 后原子替换，
失败会关闭 writer 并删除临时文件，不覆盖已有缓存。
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from spectraexpert.data.molecular_graph import featurize_smiles
from spectraexpert.data.structure_dataset import (
    canonical_smiles_sha256,
    canonicalize_smiles,
    feature_cache_schema,
    feature_config_sha256,
)
from spectraexpert.data.structure_features import StructureFeatureConfig


def _input_columns(parquet: pq.ParquetFile) -> tuple[str, str]:
    names = parquet.schema_arrow.names
    if "mol_id" not in names:
        raise ValueError("smiles input is missing required mol_id column")
    if "smiles_canonical" in names:
        return "mol_id", "smiles_canonical"
    if "smiles" in names:
        return "mol_id", "smiles"
    raise ValueError("smiles input requires smiles_canonical or smiles column")


def _iter_input_rows(
    parquet: pq.ParquetFile,
    columns: tuple[str, str],
    batch_size: int,
) -> Iterator[tuple[object, object]]:
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        mol_ids = batch.column(0).to_pylist()
        smiles = batch.column(1).to_pylist()
        yield from zip(mol_ids, smiles)


def _validated_canonical_records(
    parquet: pq.ParquetFile,
    columns: tuple[str, str],
    batch_size: int,
) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    seen: set[int] = set()
    for raw_id, raw_smiles in _iter_input_rows(parquet, columns, batch_size):
        if raw_id is None or isinstance(raw_id, bool):
            raise ValueError("mol_id must be a non-null integer")
        try:
            mol_id = int(raw_id)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"mol_id must be integer-valued: {raw_id!r}") from error
        try:
            if float(raw_id) != mol_id or not np.isfinite(float(raw_id)):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"mol_id must be finite integer-valued: {raw_id!r}") from error
        if mol_id in seen:
            raise ValueError(f"duplicate mol_id in smiles input: {mol_id}")
        seen.add(mol_id)
        records.append((mol_id, canonicalize_smiles(raw_smiles, mol_id=mol_id)))
    if not records:
        raise ValueError("smiles input is empty")
    return records


def _graph_row(mol_id: int, canonical: str, config: StructureFeatureConfig) -> dict[str, object]:
    graph = featurize_smiles(canonical, mol_id, config)
    fingerprint = np.packbits(
        graph.fingerprint.detach().cpu().numpy().astype(np.uint8), bitorder="little"
    ).tobytes()
    return {
        "mol_id": mol_id,
        "smiles_canonical": graph.canonical_smiles,
        "node_features": graph.node_features.to(torch.int16).tolist(),
        "edge_src": graph.edge_index[0].to(torch.int32).tolist(),
        "edge_dst": graph.edge_index[1].to(torch.int32).tolist(),
        "edge_features": graph.edge_features.to(torch.int8).tolist(),
        "fingerprint": fingerprint,
        "descriptors": graph.descriptors.detach().cpu().numpy().astype(np.float32).tolist(),
    }
def build_feature_cache(
    smiles_path: str | Path,
    output: str | Path,
    *,
    config: StructureFeatureConfig = StructureFeatureConfig(),
    batch_size: int = 1024,
) -> dict[str, object]:
    """校验并批量写缓存；返回 record count、source hash 与 config hash。"""

    input_path = Path(smiles_path)
    output_path = Path(output)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not input_path.is_file():
        raise FileNotFoundError(f"smiles input does not exist: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"feature cache already exists: {output_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("smiles input and feature cache output must differ")

    try:
        parquet = pq.ParquetFile(input_path)
    except Exception as error:
        raise ValueError(f"unable to read smiles Parquet: {input_path}") from error
    columns = _input_columns(parquet)
    records = _validated_canonical_records(parquet, columns, batch_size)
    source_hash = canonical_smiles_sha256(
        [record[0] for record in records], [record[1] for record in records]
    )
    schema = feature_cache_schema(config, source_smiles_sha256=source_hash)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary_name,
            schema=schema,
            compression="zstd",
            version="2.6",
        )
        for start in range(0, len(records), batch_size):
            rows = [
                _graph_row(mol_id, canonical, config)
                for mol_id, canonical in records[start : start + batch_size]
            ]
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        writer.close()
        writer = None
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        if output_path.exists():
            raise FileExistsError(f"feature cache already exists: {output_path}")
        os.replace(temporary_name, output_path)
    except BaseException:
        if writer is not None:
            writer.close()
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    return {
        "record_count": len(records),
        "source_smiles_sha256": source_hash,
        "feature_config_sha256": feature_config_sha256(config),
        "feature_config": config.to_metadata(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build versioned molecular feature cache")
    parser.add_argument("--smiles-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=2048)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = StructureFeatureConfig(
        morgan_radius=args.morgan_radius,
        morgan_bits=args.morgan_bits,
    )
    metadata = build_feature_cache(
        args.smiles_path,
        args.output,
        config=config,
        batch_size=args.batch_size,
    )
    print(
        f"cached {metadata['record_count']} molecules to {args.output} "
        f"(feature_config_sha256={metadata['feature_config_sha256']})"
    )


if __name__ == "__main__":
    main()

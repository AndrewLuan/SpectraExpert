"""
功能概述：验证版本化分子特征缓存的 Arrow schema、逐批写入、完整恢复和兼容性校验。
主要职责：覆盖 node/edge/fingerprint/descriptor round-trip、Task 2 metadata/hash、Morgan 参数
不匹配、损坏文件、重复 ID、非法 SMILES、已有输出拒绝，以及失败时不遗留目标或临时文件。
代码库关系：通过 ``scripts.cache_molecular_features`` 构建缓存，并用
``spectraexpert.data.structure_dataset.load_feature_cache`` 恢复 Task 2 ``MoleculeGraph``。
输入输出与副作用：只在 pytest 临时目录写数条 SMILES 和小型 Parquet，不访问真实分子数据；
错误路径必须 fail fast，且原子写入不能覆盖已有文件或暴露 partial cache。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from scripts.cache_molecular_features import build_feature_cache
from spectraexpert.data.molecular_graph import featurize_smiles
from spectraexpert.data.structure_dataset import (
    StructureAwareSpectralDataset,
    load_feature_cache,
)
from spectraexpert.data.structure_features import StructureFeatureConfig


def _write_smiles(path: Path, rows: list[tuple[int, str]]) -> None:
    pd.DataFrame(rows, columns=["mol_id", "smiles_canonical"]).to_parquet(
        path, index=False
    )


def test_feature_cache_round_trip_preserves_every_graph_tensor(tmp_path):
    smiles_path = tmp_path / "smiles.parquet"
    output = tmp_path / "features.parquet"
    rows = [(7, "CCO"), (8, "O"), (9, "c1ccccc1")]
    _write_smiles(smiles_path, rows)

    metadata = build_feature_cache(smiles_path, output, batch_size=2)
    loaded = load_feature_cache(output, expected_config=StructureFeatureConfig())

    assert metadata["record_count"] == 3
    assert metadata["feature_config_sha256"]
    for mol_id, smiles in rows:
        expected = featurize_smiles(smiles, mol_id)
        actual = loaded[mol_id]
        assert actual.canonical_smiles == expected.canonical_smiles
        assert actual.metadata == expected.metadata
        for field in (
            "node_features", "edge_index", "edge_features", "fingerprint", "descriptors"
        ):
            assert torch.equal(getattr(actual, field), getattr(expected, field))


def test_dataset_loads_only_requested_graphs_from_matching_cache(tmp_path):
    pd.DataFrame({"wavenumber": [500.0, 501.0, 502.0, 503.0, 504.0]}).to_parquet(
        tmp_path / "wavenumber.parquet", index=False
    )
    pd.DataFrame(
        {
            "mol_id": [2, 1],
            "w_500": [2.0, 1.0],
            "w_501": [3.0, 2.0],
            "w_502": [4.0, 3.0],
            "w_503": [5.0, 4.0],
            "w_504": [6.0, 5.0],
        }
    ).to_parquet(tmp_path / "spectra.parquet", index=False)
    smiles_path = tmp_path / "smiles.parquet"
    _write_smiles(smiles_path, [(1, "CC"), (2, "O")])
    cache = tmp_path / "features.parquet"
    build_feature_cache(smiles_path, cache)

    dataset = StructureAwareSpectralDataset.from_directory(
        tmp_path, signal_length=5, mol_ids=[2], feature_cache_path=cache
    )

    assert len(dataset) == 1
    assert dataset[0]["mol_id"] == 2
    assert dataset[0]["graph"].canonical_smiles == "O"


def test_feature_cache_uses_exact_nested_arrow_schema_and_metadata(tmp_path):
    smiles_path = tmp_path / "smiles.parquet"
    output = tmp_path / "features.parquet"
    _write_smiles(smiles_path, [(1, "CC")])

    build_feature_cache(smiles_path, output)

    schema = pq.read_schema(output)
    assert schema.names == [
        "mol_id", "smiles_canonical", "node_features", "edge_src", "edge_dst",
        "edge_features", "fingerprint", "descriptors",
    ]
    assert schema.field("mol_id").type == pa.int64()
    assert schema.field("smiles_canonical").type == pa.string()
    assert schema.field("node_features").type == pa.list_(pa.list_(pa.int16()))
    assert schema.field("edge_src").type == pa.list_(pa.int32())
    assert schema.field("edge_dst").type == pa.list_(pa.int32())
    assert schema.field("edge_features").type == pa.list_(pa.list_(pa.int8()))
    assert schema.field("fingerprint").type == pa.binary()
    assert schema.field("descriptors").type == pa.list_(pa.float32())
    metadata = schema.metadata
    assert metadata is not None
    assert b"feature_config" in metadata
    assert b"feature_config_sha256" in metadata
    assert b"source_smiles_sha256" in metadata


def test_feature_cache_rejects_morgan_config_mismatch(tmp_path):
    smiles_path = tmp_path / "smiles.parquet"
    output = tmp_path / "features.parquet"
    _write_smiles(smiles_path, [(1, "CC")])
    build_feature_cache(smiles_path, output)

    with pytest.raises(ValueError, match="feature metadata mismatch.*morgan_radius"):
        load_feature_cache(
            output,
            expected_config=StructureFeatureConfig(morgan_radius=3),
        )


def test_feature_cache_rejects_unrecognized_config_fields(tmp_path):
    smiles_path = tmp_path / "smiles.parquet"
    output = tmp_path / "features.parquet"
    _write_smiles(smiles_path, [(1, "CC")])
    build_feature_cache(smiles_path, output)
    table = pq.read_table(output)
    metadata = dict(table.schema.metadata or {})
    feature_config = json.loads(metadata[b"feature_config"])
    feature_config["unknown_future_field"] = True
    encoded = json.dumps(
        feature_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    metadata[b"feature_config"] = encoded
    metadata[b"feature_config_sha256"] = hashlib.sha256(encoded).hexdigest().encode("ascii")
    pq.write_table(table.replace_schema_metadata(metadata), output)

    with pytest.raises(ValueError, match="feature metadata mismatch"):
        load_feature_cache(output, expected_config=StructureFeatureConfig())


def test_feature_cache_rejects_corrupt_parquet(tmp_path):
    output = tmp_path / "features.parquet"
    output.write_bytes(b"not parquet")

    with pytest.raises(ValueError, match="corrupt feature cache"):
        load_feature_cache(output, expected_config=StructureFeatureConfig())


def test_feature_cache_subset_load_still_rejects_duplicate_unselected_id(tmp_path):
    smiles_path = tmp_path / "smiles.parquet"
    output = tmp_path / "features.parquet"
    _write_smiles(smiles_path, [(1, "CC"), (2, "O")])
    build_feature_cache(smiles_path, output)
    table = pq.read_table(output)
    duplicated = pa.concat_tables([table, table.slice(1, 1)])
    pq.write_table(duplicated, output)

    with pytest.raises(ValueError, match="corrupt feature cache: duplicate mol_id 2"):
        load_feature_cache(
            output,
            expected_config=StructureFeatureConfig(),
            required_mol_ids=[1],
        )


def test_feature_cache_subset_load_still_validates_unselected_rows(tmp_path):
    smiles_path = tmp_path / "smiles.parquet"
    output = tmp_path / "features.parquet"
    _write_smiles(smiles_path, [(1, "CC"), (2, "O")])
    build_feature_cache(smiles_path, output)
    table = pq.read_table(output)
    rows = table.to_pylist()
    rows[1]["fingerprint"] = rows[1]["fingerprint"][:-1]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), output)

    with pytest.raises(ValueError, match="corrupt feature cache: fingerprint length"):
        load_feature_cache(
            output,
            expected_config=StructureFeatureConfig(),
            required_mol_ids=[1],
        )


@pytest.mark.parametrize("mutation", ["delete", "add", "replace_canonical"])
def test_feature_cache_recomputes_source_hash_from_all_rows(tmp_path, mutation):
    smiles_path = tmp_path / "smiles.parquet"
    output = tmp_path / "features.parquet"
    _write_smiles(smiles_path, [(1, "CC"), (2, "O")])
    build_feature_cache(smiles_path, output)
    table = pq.read_table(output)
    rows = table.to_pylist()
    if mutation == "delete":
        rows = rows[:1]
    elif mutation == "add":
        added = dict(rows[1])
        added["mol_id"] = 3
        rows.append(added)
    else:
        rows[1]["smiles_canonical"] = "N"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), output)

    with pytest.raises(ValueError, match="source_smiles_sha256"):
        load_feature_cache(
            output,
            expected_config=StructureFeatureConfig(),
            required_mol_ids=[1],
        )


@pytest.mark.parametrize(
    ("rows", "message"),
    [([(1, "CC"), (1, "O")], "duplicate mol_id"), ([(1, "not-a-smiles")], "invalid SMILES")],
)
def test_feature_cache_input_validation_is_atomic(tmp_path, rows, message):
    smiles_path = tmp_path / "smiles.parquet"
    output = tmp_path / "features.parquet"
    _write_smiles(smiles_path, rows)

    with pytest.raises(ValueError, match=message):
        build_feature_cache(smiles_path, output, batch_size=1)

    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_feature_cache_refuses_overwrite_and_preserves_existing_file(tmp_path):
    smiles_path = tmp_path / "smiles.parquet"
    output = tmp_path / "features.parquet"
    _write_smiles(smiles_path, [(1, "CC")])
    output.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="already exists"):
        build_feature_cache(smiles_path, output)

    assert output.read_bytes() == b"keep"

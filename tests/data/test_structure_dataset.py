"""
功能概述：验证结构感知光谱数据集只按 mol_id 对齐三类数据，并在训练前拒绝任何歧义或坏数据。
主要职责：覆盖乱序一对一 join、重复/缺失/额外 ID、非法 SMILES、非有限和长度错误光谱、
canonical multi-positive identity、graph collate 顺序、spawn DataLoader pickle，以及可复用 scaffold manifest。
代码库关系：直接消费 ``spectraexpert.data.structure_dataset`` 与 Task 2 的图 dataclass；临时
Parquet helper 固定生成五点微型数据，后续 Task 7 可依赖这里锁定的 batch 字段和 split 契约。
输入输出与副作用：测试只在 pytest 临时目录写小型 Parquet/JSON，不读取 QM9S、QME14S 或 fg26；
预期错误必须在 Dataset/DataLoader 构造或 manifest 校验阶段以 ``ValueError`` 明确暴露。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from spectraexpert.data.structure_dataset import (
    StructureAwareSpectralDataset,
    scaffold_split,
    structure_aware_collate,
)


SMILES = {1: "CC", 2: "O", 3: "OCC", 4: "c1ccccc1"}


def write_tiny_parquets(
    root: Path,
    spectrum_ids: list[int],
    smiles_ids: list[int],
    *,
    smiles_overrides: dict[int, str] | None = None,
    spectrum_overrides: dict[int, list[float]] | None = None,
    axis: list[float] | None = None,
) -> None:
    """写一组完全由调用方 ID 决定、不会依赖真实数据目录的五点 fixture。"""

    root.mkdir(parents=True, exist_ok=True)
    axis = axis if axis is not None else [500.0, 501.0, 502.0, 503.0, 504.0]
    pd.DataFrame({"wavenumber": axis}).to_parquet(
        root / "wavenumber.parquet", index=False
    )
    rows = []
    for mol_id in spectrum_ids:
        values = (
            spectrum_overrides[mol_id]
            if spectrum_overrides and mol_id in spectrum_overrides
            else [float(mol_id + offset) for offset in range(5)]
        )
        rows.append({"mol_id": mol_id, **{f"w_{i}": value for i, value in enumerate(values)}})
    pd.DataFrame(rows).to_parquet(root / "spectra.parquet", index=False)
    smiles_values = {**SMILES, **(smiles_overrides or {})}
    pd.DataFrame(
        {"mol_id": smiles_ids, "smiles_canonical": [smiles_values[i] for i in smiles_ids]}
    ).to_parquet(root / "smiles.parquet", index=False)


def test_dataset_joins_spectra_and_smiles_by_mol_id(tmp_path):
    write_tiny_parquets(tmp_path, spectrum_ids=[2, 1], smiles_ids=[1, 2])

    dataset = StructureAwareSpectralDataset.from_directory(tmp_path, signal_length=5)

    assert [dataset[index]["mol_id"] for index in range(2)] == [2, 1]
    for item in dataset:
        assert item["graph"].canonical_smiles == {1: "CC", 2: "O"}[item["mol_id"]]
        assert item["spectrum_view1"][0].item() == item["mol_id"]


@pytest.mark.parametrize(
    ("spectrum_ids", "smiles_ids"),
    [([1, 1], [1, 2]), ([1, 2], [1, 1])],
)
def test_duplicate_mol_id_fails_before_training(tmp_path, spectrum_ids, smiles_ids):
    write_tiny_parquets(tmp_path, spectrum_ids=spectrum_ids, smiles_ids=smiles_ids)

    with pytest.raises(ValueError, match="duplicate mol_id"):
        StructureAwareSpectralDataset.from_directory(tmp_path, signal_length=5)


@pytest.mark.parametrize(
    ("spectrum_ids", "smiles_ids", "message"),
    [([1], [1, 2], "extra mol_id"), ([1, 2], [1], "missing mol_id")],
)
def test_missing_or_extra_join_ids_fail_fast(
    tmp_path, spectrum_ids, smiles_ids, message
):
    write_tiny_parquets(tmp_path, spectrum_ids=spectrum_ids, smiles_ids=smiles_ids)

    with pytest.raises(ValueError, match=message):
        StructureAwareSpectralDataset.from_directory(tmp_path, signal_length=5)


def test_invalid_smiles_fails_before_training(tmp_path):
    write_tiny_parquets(
        tmp_path,
        spectrum_ids=[1, 2],
        smiles_ids=[1, 2],
        smiles_overrides={2: "not-a-smiles"},
    )

    with pytest.raises(ValueError, match="invalid SMILES.*mol_id=2"):
        StructureAwareSpectralDataset.from_directory(tmp_path, signal_length=5)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_spectrum_fails_before_training(tmp_path, bad_value):
    write_tiny_parquets(
        tmp_path,
        spectrum_ids=[1],
        smiles_ids=[1],
        spectrum_overrides={1: [1.0, 2.0, bad_value, 4.0, 5.0]},
    )

    with pytest.raises(ValueError, match="non-finite spectrum.*mol_id=1"):
        StructureAwareSpectralDataset.from_directory(tmp_path, signal_length=5)


def test_axis_and_signal_length_mismatch_fail_fast(tmp_path):
    write_tiny_parquets(tmp_path, spectrum_ids=[1], smiles_ids=[1])

    with pytest.raises(ValueError, match="signal length mismatch"):
        StructureAwareSpectralDataset.from_directory(tmp_path, signal_length=6)

    pd.DataFrame({"wavenumber": [500.0, 501.0, 502.0, 503.0]}).to_parquet(
        tmp_path / "wavenumber.parquet", index=False
    )
    with pytest.raises(ValueError, match="wavenumber.*intensity"):
        StructureAwareSpectralDataset.from_directory(tmp_path)


def test_canonical_duplicates_are_multi_positive_identities(tmp_path):
    write_tiny_parquets(
        tmp_path,
        spectrum_ids=[1, 3],
        smiles_ids=[3, 1],
        smiles_overrides={1: "CCO", 3: "OCC"},
    )

    dataset = StructureAwareSpectralDataset.from_directory(tmp_path, signal_length=5)

    assert dataset[0]["graph"].canonical_smiles == "CCO"
    assert dataset[1]["graph"].canonical_smiles == "CCO"
    assert dataset[0]["identity_id"].item() == dataset[1]["identity_id"].item()


def test_collate_preserves_graph_and_target_order(tmp_path):
    write_tiny_parquets(tmp_path, spectrum_ids=[2, 1], smiles_ids=[1, 2])
    dataset = StructureAwareSpectralDataset.from_directory(tmp_path, signal_length=5)

    batch = structure_aware_collate([dataset[0], dataset[1]])

    assert batch["spectrum_view1"].shape == (2, 5)
    assert batch["spectrum_view2"].shape == (2, 5)
    assert batch["descriptor_targets"].shape == (2, 15)
    assert batch["element_presence"].shape == (2, 4)
    assert batch["fingerprint"].shape == (2, 2048)
    assert batch["mol_id"].tolist() == [2, 1]
    assert batch["graph"].mol_ids.tolist() == [2, 1]
    assert batch["graph"].canonical_smiles == ("O", "CC")
    assert batch["identity_id"].tolist() == [1, 0]
    torch.testing.assert_close(
        batch["descriptor_targets"], batch["graph"].descriptors
    )
    assert torch.equal(batch["fingerprint"], batch["graph"].fingerprints)


def test_dataset_and_collate_are_spawn_dataloader_picklable(tmp_path):
    write_tiny_parquets(tmp_path, spectrum_ids=[1, 2], smiles_ids=[1, 2])
    dataset = StructureAwareSpectralDataset.from_directory(tmp_path, signal_length=5)
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,
        multiprocessing_context="spawn",
        collate_fn=structure_aware_collate,
    )

    batch = next(iter(loader))

    assert batch["mol_id"].tolist() == [1, 2]
    assert batch["graph"].mol_ids.tolist() == [1, 2]


def test_scaffold_split_is_deterministic_and_keeps_groups_together(tmp_path):
    mol_ids = [11, 12, 13, 14, 15, 16]
    smiles = [
        "c1ccccc1",
        "Cc1ccccc1",
        "C1CCCCC1",
        "CCO",
        "OCC",
        "CCN",
    ]
    manifest_path = tmp_path / "split_manifest.json"

    first = scaffold_split(
        smiles, mol_ids=mol_ids, fractions=(0.5, 0.25, 0.25), seed=7,
        manifest_path=manifest_path,
    )
    second = scaffold_split(
        list(reversed(smiles)), mol_ids=list(reversed(mol_ids)),
        fractions=(0.5, 0.25, 0.25), seed=7, manifest_path=manifest_path,
    )

    assert first == second
    owner = {
        mol_id: split
        for split in ("train", "validation", "test")
        for mol_id in first[split]
    }
    assert owner[11] == owner[12]
    assert owner[14] == owner[15]
    assert set(owner) == set(mol_ids)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "scaffold_split_v1"
    assert persisted["seed"] == 7
    assert len(persisted["source_smiles_sha256"]) == 64


def test_existing_scaffold_manifest_rejects_different_source(tmp_path):
    path = tmp_path / "split_manifest.json"
    scaffold_split(["CC", "O"], mol_ids=[1, 2], seed=0, manifest_path=path)

    with pytest.raises(ValueError, match="split manifest mismatch"):
        scaffold_split(["CC", "N"], mol_ids=[1, 2], seed=0, manifest_path=path)


def test_existing_scaffold_manifest_rejects_tampered_group_assignment(tmp_path):
    path = tmp_path / "split_manifest.json"
    smiles = ["c1ccccc1", "Cc1ccccc1", "CCO", "CCN"]
    mol_ids = [1, 2, 3, 4]
    original = scaffold_split(
        smiles, mol_ids=mol_ids, fractions=(0.5, 0.25, 0.25), seed=3,
        manifest_path=path,
    )
    tampered = dict(original)
    source_split = next(name for name in ("train", "validation", "test") if 1 in original[name])
    target_split = next(name for name in ("train", "validation", "test") if name != source_split)
    tampered[source_split] = [value for value in original[source_split] if value != 1]
    tampered[target_split] = [*original[target_split], 1]
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="split manifest mismatch"):
        scaffold_split(
            smiles, mol_ids=mol_ids, fractions=(0.5, 0.25, 0.25), seed=3,
            manifest_path=path,
        )

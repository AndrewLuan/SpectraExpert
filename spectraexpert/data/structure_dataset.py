"""
功能概述：提供按 ``mol_id`` 严格对齐的光谱—分子图 Dataset、批处理、scaffold split 与缓存恢复。
主要职责：``StructureAwareSpectralDataset`` 校验三件套 Parquet 并只图化所选 split；
``structure_aware_collate`` 保持光谱、图、identity 和辅助 targets 的输入顺序，元素 presence 固定按
C/N/O/F/H 五维从 Task 2 descriptor 派生；``scaffold_split``
按 Bemis–Murcko scaffold（无环分子按 canonical identity）产生可复用 manifest；
``load_feature_cache`` 校验 Task 2 schema/config/hash 后逐行恢复 ``MoleculeGraph``。
代码库关系：图特征唯一复用 ``molecular_graph.featurize_smiles/collate_graphs`` 和
``StructureFeatureConfig``；光谱增强复用 ``utils.augmentations``；后续训练、索引和缓存 CLI
通过本模块共享字段、hash 和 Arrow schema，禁止按 spectra/smiles 行序隐式配对。
输入输出与副作用：读取显式三件套或缓存文件；重复/缺失/额外 ID、非法 SMILES、非有限值、
shape/schema/hash 不一致均 fail fast。manifest 只在显式路径不存在时以临时文件原子写入；
Dataset 对象只保存 NumPy/tensor/dataclass 等可 pickle 状态，可供 spawn DataLoader 使用。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset

from spectraexpert.data.molecular_graph import (
    MoleculeGraph,
    _canonicalize_heavy_molecule,
    collate_graphs,
    featurize_smiles,
)
from spectraexpert.data.structure_features import (
    DESCRIPTOR_NAMES,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    DescriptorNormalizer,
    StructureFeatureConfig,
)
from spectraexpert.utils.augmentations import create_positive_pair


SPLIT_SCHEMA_VERSION = "scaffold_split_v1"
FEATURE_CACHE_METADATA_KEY = b"feature_config"
FEATURE_CONFIG_HASH_KEY = b"feature_config_sha256"
SOURCE_SMILES_HASH_KEY = b"source_smiles_sha256"

FEATURE_CACHE_SCHEMA = pa.schema(
    [
        pa.field("mol_id", pa.int64(), nullable=False),
        pa.field("smiles_canonical", pa.string(), nullable=False),
        pa.field("node_features", pa.list_(pa.list_(pa.int16())), nullable=False),
        pa.field("edge_src", pa.list_(pa.int32()), nullable=False),
        pa.field("edge_dst", pa.list_(pa.int32()), nullable=False),
        pa.field("edge_features", pa.list_(pa.list_(pa.int8())), nullable=False),
        pa.field("fingerprint", pa.binary(), nullable=False),
        pa.field("descriptors", pa.list_(pa.float32()), nullable=False),
    ]
)


def _stable_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def feature_config_sha256(config: StructureFeatureConfig) -> str:
    """返回 Task 2 完整 feature config 的稳定 SHA-256。"""

    return hashlib.sha256(_stable_json(config.to_metadata())).hexdigest()


def canonicalize_smiles(smiles: object, *, mol_id: int | None = None) -> str:
    """按 Task 2 的重原子/isomeric 规则规范化 SMILES，并附带可定位的 mol_id 错误。"""

    label = "" if mol_id is None else f" for mol_id={mol_id}"
    try:
        canonical, _ = _canonicalize_heavy_molecule(smiles)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid SMILES{label}: {smiles!r}") from error
    return canonical


def canonical_smiles_sha256(
    mol_ids: Sequence[int], canonical_smiles: Sequence[str]
) -> str:
    """按 mol_id 排序后 hash identity 清单，使同一数据的行序变化不改变审计值。"""

    if len(mol_ids) != len(canonical_smiles):
        raise ValueError("mol_id and canonical SMILES must have equal length")
    records = sorted(
        ((int(mol_id), str(smiles)) for mol_id, smiles in zip(mol_ids, canonical_smiles)),
        key=lambda item: item[0],
    )
    return hashlib.sha256(_stable_json(records)).hexdigest()


def feature_cache_schema(
    config: StructureFeatureConfig,
    *,
    source_smiles_sha256: str,
) -> pa.Schema:
    """把完整 Task 2 config 及两个兼容性 hash 附加到固定 Arrow schema。"""

    metadata = {
        FEATURE_CACHE_METADATA_KEY: _stable_json(config.to_metadata()),
        FEATURE_CONFIG_HASH_KEY: feature_config_sha256(config).encode("ascii"),
        SOURCE_SMILES_HASH_KEY: source_smiles_sha256.encode("ascii"),
    }
    return FEATURE_CACHE_SCHEMA.with_metadata(metadata)


def _validate_unique_ids(frame: pd.DataFrame, *, source: str) -> None:
    if "mol_id" not in frame.columns:
        raise ValueError(f"{source} is missing required mol_id column")
    if frame["mol_id"].isna().any():
        raise ValueError(f"{source} contains null mol_id")
    duplicates = frame.loc[frame["mol_id"].duplicated(keep=False), "mol_id"]
    if not duplicates.empty:
        values = sorted({int(value) for value in duplicates.tolist()})
        raise ValueError(f"duplicate mol_id in {source}: {values}")


def _int64_ids(values: Iterable[object], *, source: str) -> np.ndarray:
    try:
        numeric = pd.to_numeric(pd.Series(list(values)), errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} mol_id must be integer-valued") from error
    array = numeric.to_numpy()
    if not np.isfinite(array.astype(np.float64)).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{source} mol_id must be finite integer-valued")
    return array.astype(np.int64)


def load_feature_cache(
    path: str | Path,
    *,
    expected_config: StructureFeatureConfig = StructureFeatureConfig(),
    expected_source_smiles_sha256: str | None = None,
    required_mol_ids: Iterable[int] | None = None,
) -> dict[int, MoleculeGraph]:
    """校验并恢复 feature cache；损坏、维度错误或 metadata 不兼容均不返回部分结果。"""

    cache_path = Path(path)
    try:
        parquet = pq.ParquetFile(cache_path)
        schema = parquet.schema_arrow
    except Exception as error:
        raise ValueError(f"corrupt feature cache: {cache_path}") from error

    if not schema.remove_metadata().equals(FEATURE_CACHE_SCHEMA):
        raise ValueError("corrupt feature cache: Arrow schema mismatch")
    metadata = schema.metadata or {}
    try:
        cached_config = json.loads(metadata[FEATURE_CACHE_METADATA_KEY].decode("utf-8"))
        cached_config_hash = metadata[FEATURE_CONFIG_HASH_KEY].decode("ascii")
        cached_source_hash = metadata[SOURCE_SMILES_HASH_KEY].decode("ascii")
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("corrupt feature cache: missing or invalid metadata") from error

    expected_metadata = expected_config.to_metadata()
    unexpected_fields = sorted(set(cached_config) ^ set(expected_metadata))
    if unexpected_fields:
        raise ValueError(
            "feature metadata mismatch: unexpected/missing fields "
            f"{unexpected_fields}"
        )
    for field, expected_value in expected_metadata.items():
        actual_value = cached_config.get(field)
        if actual_value != expected_value:
            raise ValueError(
                "feature metadata mismatch: "
                f"{field} expected {expected_value!r}, got {actual_value!r}"
            )
    actual_config_hash = hashlib.sha256(_stable_json(cached_config)).hexdigest()
    if cached_config_hash != actual_config_hash:
        raise ValueError("corrupt feature cache: feature_config_sha256 mismatch")
    required = None if required_mol_ids is None else {int(value) for value in required_mol_ids}
    graphs: dict[int, MoleculeGraph] = {}
    seen_cache_ids: set[int] = set()
    cache_mol_ids: list[int] = []
    cache_canonical_smiles: list[str] = []
    try:
        batches = parquet.iter_batches(batch_size=1024)
        for batch in batches:
            for row in batch.to_pylist():
                mol_id = int(row["mol_id"])
                if mol_id in seen_cache_ids:
                    raise ValueError(f"corrupt feature cache: duplicate mol_id {mol_id}")
                seen_cache_ids.add(mol_id)
                canonical = canonicalize_smiles(row["smiles_canonical"], mol_id=mol_id)
                if canonical != row["smiles_canonical"]:
                    raise ValueError(
                        f"corrupt feature cache: non-canonical SMILES for mol_id {mol_id}"
                    )
                cache_mol_ids.append(mol_id)
                cache_canonical_smiles.append(canonical)
                node_features = torch.tensor(row["node_features"], dtype=torch.long)
                edge_src = torch.tensor(row["edge_src"], dtype=torch.long)
                edge_dst = torch.tensor(row["edge_dst"], dtype=torch.long)
                edge_features = torch.tensor(
                    row["edge_features"], dtype=torch.long
                ).reshape(-1, len(EDGE_FEATURE_NAMES))
                descriptors = torch.tensor(row["descriptors"], dtype=torch.float32)
                packed = np.frombuffer(row["fingerprint"], dtype=np.uint8)
                expected_bytes = (expected_config.morgan_bits + 7) // 8
                if packed.size != expected_bytes:
                    raise ValueError(
                        f"corrupt feature cache: fingerprint length for mol_id {mol_id}"
                    )
                fingerprint = torch.from_numpy(
                    np.unpackbits(packed, bitorder="little")[: expected_config.morgan_bits].copy()
                ).bool()
                if node_features.ndim != 2 or node_features.shape[1] != len(NODE_FEATURE_NAMES):
                    raise ValueError(
                        f"corrupt feature cache: node feature shape for mol_id {mol_id}"
                    )
                if edge_src.ndim != 1 or edge_dst.ndim != 1 or edge_src.shape != edge_dst.shape:
                    raise ValueError(
                        f"corrupt feature cache: edge index shape for mol_id {mol_id}"
                    )
                if edge_features.shape != (edge_src.numel(), len(EDGE_FEATURE_NAMES)):
                    raise ValueError(
                        f"corrupt feature cache: edge feature shape for mol_id {mol_id}"
                    )
                if descriptors.shape != (len(DESCRIPTOR_NAMES),) or not torch.isfinite(descriptors).all():
                    raise ValueError(
                        f"corrupt feature cache: descriptor shape/value for mol_id {mol_id}"
                    )
                edge_index = torch.stack((edge_src, edge_dst), dim=0)
                if edge_index.numel() and (
                    edge_index.min().item() < 0
                    or edge_index.max().item() >= node_features.shape[0]
                ):
                    raise ValueError(
                        f"corrupt feature cache: edge node index for mol_id {mol_id}"
                    )
                graph = MoleculeGraph(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_features=edge_features,
                    fingerprint=fingerprint,
                    descriptors=descriptors,
                    mol_id=mol_id,
                    canonical_smiles=canonical,
                    metadata=expected_metadata,
                )
                if required is None or mol_id in required:
                    graphs[mol_id] = graph
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"corrupt feature cache: {cache_path}") from error

    actual_source_hash = canonical_smiles_sha256(
        cache_mol_ids, cache_canonical_smiles
    )
    if actual_source_hash != cached_source_hash:
        raise ValueError("corrupt feature cache: source_smiles_sha256 mismatch")
    if (
        expected_source_smiles_sha256 is not None
        and actual_source_hash != expected_source_smiles_sha256
    ):
        raise ValueError("feature metadata mismatch: source_smiles_sha256")
    if required is not None:
        missing = sorted(required - set(graphs))
        if missing:
            raise ValueError(f"feature cache missing mol_id: {missing}")
    return graphs


class StructureAwareSpectralDataset(Dataset):
    """已严格对齐的光谱、分子图、multi-positive identity 与辅助监督 targets。"""

    def __init__(
        self,
        spectra: np.ndarray,
        wavenumber_axis: np.ndarray,
        mol_ids: np.ndarray,
        canonical_smiles: Sequence[str],
        graphs: Sequence[MoleculeGraph],
        identity_ids: np.ndarray,
        *,
        normalize: bool = False,
        descriptor_normalizer: DescriptorNormalizer | None = None,
        feature_config: StructureFeatureConfig = StructureFeatureConfig(),
        source_smiles_sha256: str,
    ) -> None:
        super().__init__()
        self.spectra = np.asarray(spectra, dtype=np.float32)
        self.wavenumber_axis = np.asarray(wavenumber_axis, dtype=np.float32)
        self.mol_ids = np.asarray(mol_ids, dtype=np.int64)
        self.canonical_smiles = tuple(canonical_smiles)
        self.graphs = tuple(graphs)
        self.identity_ids = np.asarray(identity_ids, dtype=np.int64)
        self.normalize = bool(normalize)
        self.descriptor_normalizer = descriptor_normalizer
        self.feature_config = feature_config
        self.source_smiles_sha256 = source_smiles_sha256

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        signal_length: int | None = None,
        mol_ids: Iterable[int] | None = None,
        feature_cache_path: str | Path | None = None,
        feature_config: StructureFeatureConfig = StructureFeatureConfig(),
        normalize: bool = False,
        descriptor_normalizer: DescriptorNormalizer | None = None,
    ) -> "StructureAwareSpectralDataset":
        """读取三件套，先验证全集一对一 join，再只为 ``mol_ids`` 所选 split 建图。"""

        root = Path(directory)
        paths = {
            name: root / f"{name}.parquet"
            for name in ("spectra", "smiles", "wavenumber")
        }
        missing_files = [str(path) for path in paths.values() if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f"missing dataset file(s): {missing_files}")

        spectra_frame = pd.read_parquet(paths["spectra"])
        smiles_frame = pd.read_parquet(paths["smiles"])
        axis_frame = pd.read_parquet(paths["wavenumber"])
        if list(axis_frame.columns) != ["wavenumber"] or axis_frame.empty:
            raise ValueError("wavenumber.parquet must contain one non-empty wavenumber column")
        axis = axis_frame["wavenumber"].to_numpy(dtype=np.float32)
        if not np.isfinite(axis).all():
            raise ValueError("wavenumber axis contains non-finite values")
        if len(axis) > 1 and not np.all(np.diff(axis) > 0):
            raise ValueError("wavenumber axis must be strictly increasing")
        if signal_length is not None and len(axis) != int(signal_length):
            raise ValueError(
                f"signal length mismatch: expected {signal_length}, got {len(axis)}"
            )

        _validate_unique_ids(spectra_frame, source="spectra.parquet")
        _validate_unique_ids(smiles_frame, source="smiles.parquet")
        spectra_frame = spectra_frame.copy()
        smiles_frame = smiles_frame.copy()
        spectra_frame["mol_id"] = _int64_ids(
            spectra_frame["mol_id"], source="spectra.parquet"
        )
        smiles_frame["mol_id"] = _int64_ids(
            smiles_frame["mol_id"], source="smiles.parquet"
        )
        intensity_columns = [column for column in spectra_frame.columns if column != "mol_id"]
        if len(intensity_columns) != len(axis):
            raise ValueError(
                "wavenumber/intensity length mismatch: "
                f"axis has {len(axis)}, spectra have {len(intensity_columns)} columns"
            )
        smiles_column = (
            "smiles_canonical"
            if "smiles_canonical" in smiles_frame.columns
            else "smiles" if "smiles" in smiles_frame.columns else None
        )
        if smiles_column is None:
            raise ValueError("smiles.parquet requires smiles_canonical or smiles column")

        spectrum_id_set = set(spectra_frame["mol_id"].tolist())
        smiles_id_set = set(smiles_frame["mol_id"].tolist())
        extra = sorted(smiles_id_set - spectrum_id_set)
        missing = sorted(spectrum_id_set - smiles_id_set)
        if extra:
            raise ValueError(f"extra mol_id in smiles.parquet: {extra}")
        if missing:
            raise ValueError(f"missing mol_id in smiles.parquet: {missing}")
        try:
            joined = spectra_frame.merge(
                smiles_frame[["mol_id", smiles_column]],
                on="mol_id",
                how="left",
                sort=False,
                validate="one_to_one",
            )
        except pd.errors.MergeError as error:
            raise ValueError(f"mol_id one-to-one join failed: {error}") from error

        try:
            intensity = joined[intensity_columns].to_numpy(dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("spectra intensity columns must be numeric") from error
        bad_rows = np.flatnonzero(~np.isfinite(intensity).all(axis=1))
        if bad_rows.size:
            bad_id = int(joined.iloc[int(bad_rows[0])]["mol_id"])
            raise ValueError(f"non-finite spectrum for mol_id={bad_id}")

        all_canonical = [
            canonicalize_smiles(value, mol_id=int(identifier))
            for identifier, value in zip(joined["mol_id"], joined[smiles_column])
        ]
        all_mol_ids = joined["mol_id"].to_numpy(dtype=np.int64)
        source_hash = canonical_smiles_sha256(all_mol_ids.tolist(), all_canonical)
        vocabulary = {
            canonical: identity
            for identity, canonical in enumerate(sorted(set(all_canonical)))
        }

        if mol_ids is None:
            selected_indices = np.arange(len(joined), dtype=np.int64)
        else:
            requested = [int(value) for value in mol_ids]
            if len(requested) != len(set(requested)):
                raise ValueError("duplicate mol_id in requested split")
            unknown = sorted(set(requested) - spectrum_id_set)
            if unknown:
                raise ValueError(f"requested split contains unknown mol_id: {unknown}")
            requested_set = set(requested)
            selected_indices = np.asarray(
                [index for index, value in enumerate(all_mol_ids) if int(value) in requested_set],
                dtype=np.int64,
            )
        if selected_indices.size == 0:
            raise ValueError("selected split is empty")

        selected_ids = all_mol_ids[selected_indices]
        selected_canonical = [all_canonical[index] for index in selected_indices]
        if feature_cache_path is None:
            graph_by_id = {
                int(identifier): featurize_smiles(canonical, int(identifier), feature_config)
                for identifier, canonical in zip(selected_ids, selected_canonical)
            }
        else:
            graph_by_id = load_feature_cache(
                feature_cache_path,
                expected_config=feature_config,
                expected_source_smiles_sha256=source_hash,
                required_mol_ids=selected_ids.tolist(),
            )
        graphs = []
        for identifier, canonical in zip(selected_ids, selected_canonical):
            graph = graph_by_id[int(identifier)]
            if graph.canonical_smiles != canonical:
                raise ValueError(
                    f"feature metadata mismatch: canonical SMILES for mol_id={int(identifier)}"
                )
            graphs.append(graph)

        return cls(
            spectra=intensity[selected_indices],
            wavenumber_axis=axis,
            mol_ids=selected_ids,
            canonical_smiles=selected_canonical,
            graphs=graphs,
            identity_ids=np.asarray(
                [vocabulary[canonical] for canonical in selected_canonical], dtype=np.int64
            ),
            normalize=normalize,
            descriptor_normalizer=descriptor_normalizer,
            feature_config=feature_config,
            source_smiles_sha256=source_hash,
        )

    def __len__(self) -> int:
        return len(self.mol_ids)

    @staticmethod
    def _normalize_spectrum(values: np.ndarray) -> np.ndarray:
        std = float(values.std())
        if std > 1e-10:
            return ((values - values.mean()) / std).astype(np.float32)
        return (values - values.mean()).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        spectrum = self.spectra[index].copy()
        view1, view2 = create_positive_pair(spectrum, self.wavenumber_axis)
        if self.normalize:
            view1 = self._normalize_spectrum(view1)
            view2 = self._normalize_spectrum(view2)
        graph = self.graphs[index]
        descriptor_targets = graph.descriptors
        if self.descriptor_normalizer is not None:
            descriptor_targets = self.descriptor_normalizer.transform(descriptor_targets)
        return {
            "spectrum_view1": torch.from_numpy(np.asarray(view1, dtype=np.float32)),
            "spectrum_view2": torch.from_numpy(np.asarray(view2, dtype=np.float32)),
            "graph": graph,
            "descriptor_targets": descriptor_targets.clone().float(),
            "element_presence": (graph.descriptors[:5] > 0).float(),
            "fingerprint": graph.fingerprint.clone(),
            "identity_id": torch.tensor(self.identity_ids[index], dtype=torch.long),
            "mol_id": int(self.mol_ids[index]),
        }


def structure_aware_collate(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """按输入顺序拼 batch，graph、mol_id、identity 和所有 targets 保持逐行对应。"""

    rows = list(items)
    if not rows:
        raise ValueError("structure_aware_collate requires at least one item")
    return {
        "spectrum_view1": torch.stack([row["spectrum_view1"] for row in rows]),
        "spectrum_view2": torch.stack([row["spectrum_view2"] for row in rows]),
        "graph": collate_graphs([row["graph"] for row in rows]),
        "descriptor_targets": torch.stack([row["descriptor_targets"] for row in rows]),
        "element_presence": torch.stack([row["element_presence"] for row in rows]),
        "fingerprint": torch.stack([row["fingerprint"] for row in rows]),
        "identity_id": torch.stack([row["identity_id"] for row in rows]),
        "mol_id": torch.tensor([row["mol_id"] for row in rows], dtype=torch.long),
    }


def _manifest_payload(
    canonical: Sequence[str],
    mol_ids: Sequence[int],
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, object]:
    groups: dict[str, list[int]] = {}
    for identifier, identity in zip(mol_ids, canonical):
        molecule = Chem.MolFromSmiles(identity)
        if molecule is None:
            raise ValueError(f"invalid SMILES for mol_id={identifier}: {identity!r}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
        group_key = f"scaffold:{scaffold}" if scaffold else f"identity:{identity}"
        groups.setdefault(group_key, []).append(int(identifier))

    targets = np.asarray(fractions, dtype=np.float64) * len(mol_ids)
    counts = np.zeros(3, dtype=np.int64)
    split_names = ("train", "validation", "test")
    result: dict[str, list[int]] = {name: [] for name in split_names}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(f"{seed}:{item[0]}".encode("utf-8")).hexdigest(),
        ),
    )
    for _, group_ids in ordered_groups:
        deficits = targets - counts
        split_index = max(range(3), key=lambda index: (deficits[index], -index))
        result[split_names[split_index]].extend(group_ids)
        counts[split_index] += len(group_ids)
    for split in split_names:
        result[split].sort()
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": int(seed),
        "fractions": [float(value) for value in fractions],
        "source_smiles_sha256": canonical_smiles_sha256(mol_ids, canonical),
        **result,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def scaffold_split(
    smiles: Sequence[str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    *,
    mol_ids: Sequence[int] | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    """按 scaffold 分组划分；已有 manifest 仅在 source/config 完全一致时复用。"""

    if len(smiles) == 0:
        raise ValueError("scaffold split requires at least one SMILES")
    if len(fractions) != 3 or any(value < 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must contain three non-negative values summing to one")
    identifiers = list(range(len(smiles))) if mol_ids is None else [int(value) for value in mol_ids]
    if len(identifiers) != len(smiles):
        raise ValueError("mol_id and SMILES must have equal length")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate mol_id in scaffold split")
    canonical = [
        canonicalize_smiles(value, mol_id=identifier)
        for identifier, value in zip(identifiers, smiles)
    ]
    expected = _manifest_payload(canonical, identifiers, fractions, int(seed))
    if manifest_path is None:
        return expected

    path = Path(manifest_path)
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"corrupt split manifest: {path}") from error
        compatibility_fields = (
            "schema_version", "seed", "fractions", "source_smiles_sha256"
        )
        differences = [
            field for field in compatibility_fields if current.get(field) != expected[field]
        ]
        all_ids = current.get("train", []) + current.get("validation", []) + current.get("test", [])
        assignment_mismatch = any(
            sorted(current.get(split, [])) != expected[split]
            for split in ("train", "validation", "test")
        )
        if (
            differences
            or assignment_mismatch
            or sorted(all_ids) != sorted(identifiers)
            or len(all_ids) != len(set(all_ids))
        ):
            raise ValueError(
                "split manifest mismatch: "
                + ", ".join(differences or ["mol_id assignments"])
            )
        return current
    _atomic_write_json(path, expected)
    return expected


__all__ = [
    "FEATURE_CACHE_SCHEMA",
    "StructureAwareSpectralDataset",
    "canonical_smiles_sha256",
    "canonicalize_smiles",
    "feature_cache_schema",
    "feature_config_sha256",
    "load_feature_cache",
    "scaffold_split",
    "structure_aware_collate",
]

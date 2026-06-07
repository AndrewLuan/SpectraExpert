#!/usr/bin/env python3
"""
build_benchmark.py — 光谱指纹基准数据集构建（仅 QM9S Raman）

从 QM9S 清洗后的数据构建统一的基准数据集。
采用 batch-wise pyarrow 写入以控制内存使用。

输出:
  - data/processed/benchmark.parquet   — QM9S 基准数据集
  - data/processed/dataset_stats.json  — 数据集统计信息
  - data/processed/wavenumber.parquet  — 波数轴
  - data/logs/build_benchmark.log
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

WAVENUM_MIN = 500.0
WAVENUM_MAX = 4000.0
WAVENUM_STEP = 1.0
N_PIXELS = int((WAVENUM_MAX - WAVENUM_MIN) / WAVENUM_STEP) + 1

WAVENAME_COLS = [f"w_{int(WAVENUM_MIN + i * WAVENUM_STEP)}"
                 for i in range(N_PIXELS)]


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_qm9s_data(qm9s_dir: Path, logger: logging.Logger):
    """Load QM9S data as batched dicts for writing."""
    spectra_path = qm9s_dir / "spectra.parquet"
    smiles_path = qm9s_dir / "smiles.parquet"

    if not spectra_path.exists() or not smiles_path.exists():
        logger.error(f"QM9S processed files not found in {qm9s_dir}")
        logger.error("Run scripts/prepare_qm9s.py first")
        sys.exit(1)

    smiles_df = pd.read_parquet(smiles_path)
    wavenumber_df = pd.read_parquet(qm9s_dir / "wavenumber.parquet")
    wavenumber = wavenumber_df["wavenumber"].values

    logger.info(f"  Loading QM9S spectra ({spectra_path})...")
    pf = pq.ParquetFile(spectra_path)
    n_rows = pf.metadata.num_rows
    n_cols = len(pf.schema_arrow.names) - 1  # minus mol_id
    logger.info(f"  Spectra matrix: {n_rows} rows × {n_cols} cols")

    # Read in chunks
    reader = pf.iter_batches(batch_size=20000)
    all_batches = []

    for i, batch in enumerate(reader):
        df_batch = batch.to_pandas()
        intensity = df_batch[WAVENAME_COLS].values.astype(np.float32)
        mol_ids = df_batch["mol_id"].values.astype(np.int32)

        combined = smiles_df.iloc[
            smiles_df["mol_id"].isin(mol_ids)
        ].copy()

        all_batches.append({
            "mol_ids": mol_ids,
            "intensity": intensity,
            "smiles_batch": combined,
            "wavenumber": wavenumber,
        })

        if (i + 1) % 5 == 0:
            logger.info(f"    Read batch {i+1} ({len(mol_ids)} rows)")

    logger.info(f"  QM9S loaded: {n_rows} rows across {len(all_batches)} batches")
    return all_batches


def write_benchmark(qm9s_batches, output_path: Path, logger: logging.Logger):
    """Write QM9S benchmark dataset as Parquet in batches."""
    schema = pa.schema([
        pa.field("mol_id", pa.int32()),
        pa.field("smiles", pa.string()),
        pa.field("source", pa.string()),
        pa.field("split", pa.string()),
        *[pa.field(c, pa.float32()) for c in WAVENAME_COLS],
    ])

    writer = None
    total_written = 0

    for i, batch_data in enumerate(qm9s_batches):
        mol_ids = batch_data["mol_ids"]
        intensity = batch_data["intensity"]
        smiles_batch = batch_data["smiles_batch"]
        n = len(mol_ids)

        col_data = [
            pa.array(mol_ids, type=pa.int32()),
            pa.array(smiles_batch["smiles_canonical"].values, type=pa.string()),
            pa.array(["qm9s"] * n, type=pa.string()),
            pa.array(["train"] * n, type=pa.string()),
        ]
        for ci in range(N_PIXELS):
            col_data.append(pa.array(intensity[:, ci], type=pa.float32()))

        table = pa.table(dict(zip(schema.names, col_data)), schema=schema)

        if writer is None:
            writer = pq.ParquetWriter(
                str(output_path), schema=schema,
                compression="zstd", version="2.6",
            )
        writer.write_table(table)
        total_written += n
        logger.info(f"  Written batch {i+1}: {n} rows (total: {total_written})")

    if writer:
        writer.close()

    return total_written


def main():
    parser = argparse.ArgumentParser(description="Build QM9S benchmark dataset")
    parser.add_argument("--qm9s-dir", type=str,
                        default=str(DATA_DIR / "qm9s"))
    parser.add_argument("--output-dir", type=str,
                        default=str(DATA_DIR / "processed"))
    parser.add_argument("--log-dir", type=str,
                        default=str(DATA_DIR / "logs"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir / "build_benchmark.log")
    logger.info("=== QM9S Benchmark Dataset Construction ===")

    # ---- Load QM9S ----
    logger.info("\n[Step 1] Loading QM9S training set...")
    qm9s = load_qm9s_data(Path(args.qm9s_dir), logger)

    if qm9s is None:
        logger.error("No data loaded!")
        sys.exit(1)

    # ---- Write ----
    logger.info("\n[Step 2] Writing benchmark dataset...")
    out_path = output_dir / "benchmark.parquet"
    total = write_benchmark(qm9s, out_path, logger)

    if total == 0:
        logger.error("No data written!")
        sys.exit(1)

    file_size = out_path.stat().st_size
    logger.info(f"  Total rows: {total}")
    logger.info(f"  File: {out_path} ({file_size / 1e9:.2f} GB)")

    # Save wavenumber axis
    wavenumber = np.arange(WAVENUM_MIN, WAVENUM_MAX + WAVENUM_STEP,
                           WAVENUM_STEP, dtype=np.float32)
    wavenumber_df = pd.DataFrame({"wavenumber": wavenumber})
    wavenumber_df.to_parquet(output_dir / "wavenumber.parquet", index=False)

    # ---- Statistics ----
    logger.info("\n[Step 3] Computing statistics...")
    pf = pq.ParquetFile(out_path)
    meta = pf.read(columns=["source", "split"]).to_pandas()
    n_train = (meta["split"] == "train").sum()

    smiles_meta = pf.read(columns=["smiles"]).to_pandas()
    n_unique = smiles_meta["smiles"].nunique()

    stats = {
        "total_samples": int(total),
        "train_samples": int(n_train),
        "unique_smiles": int(n_unique),
        "wavenumber_min": WAVENUM_MIN,
        "wavenumber_max": WAVENUM_MAX,
        "wavenumber_step": WAVENUM_STEP,
        "n_pixels": N_PIXELS,
        "file_size_gb": round(file_size / 1e9, 2),
    }

    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"QM9S Benchmark Dataset Summary:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()

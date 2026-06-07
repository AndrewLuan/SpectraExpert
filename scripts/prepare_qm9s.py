#!/usr/bin/env python3
"""
prepare_qm9s.py — QM9S Raman 光谱数据清洗与 SMILES 配对

流程:
  1. 读取 smiles.csv（idx, SMILES）
  2. 用 RDKit 校验并标准化 SMILES（去重、规范形式）
  3. 分块读取 raman_boraden.csv（8.2 GB），提取光谱数据
  4. 验证光谱有效性（非负、有信号）
  5. 输出为 Parquet 格式

输出:
  - data/qm9s/smiles.parquet     — 校验后的 SMILES 元数据
  - data/qm9s/spectra.parquet    — 强度矩阵 (float32, N×3501)
  - data/qm9s/wavenumber.parquet — 波数轴 (500-4000 cm⁻¹)
  - data/logs/qm9s_preparation.log
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings for invalid SMILES
RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QM9S_DIR = Path("/home/pleiades/projects/QM9S")

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


def validate_smiles(smiles_series: pd.Series) -> pd.DataFrame:
    """用 RDKit 校验 SMILES，返回带 canonical 形式的数据框"""
    results = []
    valid_count = 0
    invalid_count = 0

    for idx, smi in enumerate(smiles_series):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append({"original_idx": idx, "smiles": smi,
                            "smiles_canonical": None, "valid": False})
            invalid_count += 1
        else:
            try:
                canon = Chem.MolToSmiles(mol, canonical=True)
                # 进一步确保：移除不再需要的显式 H、立体化学等
                canon = Chem.MolToSmiles(Chem.MolFromSmiles(canon), canonical=True)
                results.append({"original_idx": idx, "smiles": smi,
                                "smiles_canonical": canon, "valid": True})
                valid_count += 1
            except Exception:
                results.append({"original_idx": idx, "smiles": smi,
                                "smiles_canonical": None, "valid": False})
                invalid_count += 1

        if (idx + 1) % 20000 == 0:
            logging.info(f"  SMILES validated: {idx+1}/{len(smiles_series)} "
                         f"(valid: {valid_count}, invalid: {invalid_count})")

    df = pd.DataFrame(results)
    logging.info(f"SMILES validation done: total={len(df)}, "
                 f"valid={valid_count}, invalid={invalid_count}")
    return df


def load_raman_spectra(raman_path: Path, chunk_size: int = 5000,
                       dtype=np.float32) -> tuple:
    """
    分块读取 raman_boraden.csv（8.2 GB）。

    文件格式：
      - 第 1 行 header: ",500.0,501.0,...,4000.0"（3502 列，首列为空）
      - 数据行: "idx,val1,val2,...,val3501"

    策略：用 pandas chunked reader 逐块读取，跳过第一列（索引列），
          累积成 float32 numpy 矩阵。

    返回 (wavenumber_array, spectra_matrix)
    """
    logging.info(f"Loading Raman spectra from {raman_path}")
    logging.info(f"  File size: {raman_path.stat().st_size / 1e9:.2f} GB")

    # 1. 读取 header 获取波数轴
    header = pd.read_csv(raman_path, nrows=0)
    wavenumber = np.array([float(c) for c in header.columns[1:]], dtype=np.float32)
    n_pixels = len(wavenumber)
    logging.info(f"  Wavenumber range: {wavenumber[0]:.1f} - {wavenumber[-1]:.1f} cm⁻¹, "
                 f"{n_pixels} pixels")

    # 2. 统计总行数
    import subprocess
    result = subprocess.run(["wc", "-l", str(raman_path)],
                            capture_output=True, text=True)
    total_lines = int(result.stdout.strip().split()[0])
    n_rows = total_lines - 1  # 减去 header
    logging.info(f"  Total data rows: {n_rows}")

    # 3. 分块读取 — 用 pandas chunks, 跳过第一列（usecols=range(1,3502)）
    #    注意 header 第一列是空字符串，从第二列开始是波数值
    n_cols = n_pixels + 1  # 3502 = index + 3501 intensities
    usecols = list(range(1, n_cols))  # 跳过 col 0 (index 列)
    col_names = [f"w{i}" for i in range(n_pixels)]

    all_chunks = []
    start_time = time.time()
    total_rows_read = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(raman_path, chunksize=chunk_size,
                    usecols=usecols, header=0,
                    dtype=np.float32, low_memory=False)
    ):
        # chunk: 跳过第一列后，只剩 3501 列（强度值）
        chunk.columns = col_names  # 重命名以免列名是浮点数
        values = chunk.values.astype(dtype)
        all_chunks.append(values)
        total_rows_read += len(chunk)

        if (chunk_idx + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = total_rows_read / elapsed
            logging.info(f"  Chunk {chunk_idx+1}: {total_rows_read}/{n_rows} rows "
                         f"({rate:.0f} rows/s)")

    # 合并
    logging.info(f"  Merging {len(all_chunks)} chunks ({total_rows_read} rows)...")
    spectra = np.vstack(all_chunks)
    assert spectra.shape[0] == n_rows, \
        f"Row count mismatch: expected {n_rows}, got {spectra.shape[0]}"
    assert spectra.shape[1] == n_pixels, \
        f"Column count mismatch: expected {n_pixels}, got {spectra.shape[1]}"

    elapsed = time.time() - start_time
    logging.info(f"  Spectra loaded: {spectra.shape} in {elapsed:.1f}s "
                 f"({total_rows_read/elapsed:.0f} rows/s)")
    return wavenumber, spectra


def validate_spectra(spectra: np.ndarray) -> np.ndarray:
    """
    验证光谱有效性。
    返回有效行的 boolean mask。
    """
    logging.info("Validating spectra...")
    n_total = spectra.shape[0]

    # 检查 NaN
    has_nan = np.isnan(spectra).any(axis=1)
    n_nan = has_nan.sum()
    if n_nan > 0:
        logging.warning(f"  {n_nan} spectra contain NaN values")

    # 检查负值（允许少量数值噪音）
    min_vals = spectra.min(axis=1)
    has_negative = min_vals < -1e-6
    n_neg = has_negative.sum()
    if n_neg > 0:
        logging.warning(f"  {n_neg} spectra have negative values < -1e-6")

    # 检查全零/无效光谱
    max_vals = spectra.max(axis=1)
    is_flat = max_vals < 1e-10
    n_flat = is_flat.sum()
    if n_flat > 0:
        logging.warning(f"  {n_flat} spectra are all-zero or flat")

    # 综合有效 mask
    valid_mask = ~has_nan & ~is_flat
    n_valid = valid_mask.sum()
    logging.info(f"  Valid spectra: {n_valid}/{n_total} "
                 f"({n_total - n_valid} filtered)")
    return valid_mask


def main():
    parser = argparse.ArgumentParser(description="QM9S data preparation")
    parser.add_argument("--qm9s-dir", type=str, default=str(QM9S_DIR),
                        help="QM9S data directory")
    parser.add_argument("--output-dir", type=str,
                        default=str(DATA_DIR / "qm9s"),
                        help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Rows per chunk during CSV reading")
    parser.add_argument("--log-dir", type=str,
                        default=str(DATA_DIR / "logs"))
    args = parser.parse_args()

    qm9s_dir = Path(args.qm9s_dir)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir / "qm9s_preparation.log")
    logger.info("=== QM9S Data Preparation ===")
    logger.info(f"Input:  {qm9s_dir}")
    logger.info(f"Output: {output_dir}")

    # ---- Step 1: Load SMILES ----
    smiles_path = qm9s_dir / "smiles.csv"
    logger.info(f"\n[Step 1] Loading SMILES from {smiles_path}")
    smiles_df = pd.read_csv(smiles_path, header=None, names=["mol_id", "smiles"])
    logger.info(f"  Loaded {len(smiles_df)} SMILES entries")
    logger.info(f"  Unique SMILES: {smiles_df['smiles'].nunique()}")
    assert len(smiles_df) == 129817, f"Expected 129817 SMILES, got {len(smiles_df)}"

    # ---- Step 2: Validate SMILES ----
    logger.info(f"\n[Step 2] Validating SMILES with RDKit...")
    validation_df = validate_smiles(smiles_df["smiles"])

    # Add mol_id
    validation_df["mol_id"] = smiles_df["mol_id"].values

    # Save validated SMILES
    valid_mask = validation_df["valid"].values
    n_valid_smiles = valid_mask.sum()
    n_invalid_smiles = (~valid_mask).sum()

    smiles_out = validation_df[["mol_id", "smiles", "smiles_canonical", "valid"]]
    smiles_out.to_parquet(output_dir / "smiles.parquet", index=False)
    logger.info(f"  Saved {len(smiles_out)} SMILES records to "
                f"{output_dir / 'smiles.parquet'}")
    logger.info(f"  Valid: {n_valid_smiles}, Invalid: {n_invalid_smiles}")

    # ---- Step 3: Load Raman Spectra ----
    logger.info(f"\n[Step 3] Loading Raman spectra...")
    raman_path = qm9s_dir / "raman_boraden.csv"
    wavenumber, spectra = load_raman_spectra(raman_path, args.chunk_size)

    # Save wavenumber axis
    wavenumber_df = pd.DataFrame({"wavenumber": wavenumber})
    wavenumber_df.to_parquet(output_dir / "wavenumber.parquet", index=False)
    logger.info(f"  Saved wavenumber axis: {len(wavenumber)} points")

    # ---- Step 4: Validate Spectra ----
    logger.info(f"\n[Step 4] Validating spectra...")
    spec_valid_mask = validate_spectra(spectra)

    # ---- Step 5: Combine & Save ----
    logger.info(f"\n[Step 5] Combining SMILES + Spectra and saving...")
    final_mask = valid_mask & spec_valid_mask
    n_final = final_mask.sum()
    n_filtered = len(final_mask) - n_final
    logger.info(f"  Total valid pairs: {n_final}/{len(final_mask)}")
    logger.info(f"  Filtered out: {n_filtered} entries")

    if n_final == 0:
        logger.error("No valid data pairs remaining! Check data quality.")
        sys.exit(1)

    # Apply mask
    spectra_valid = spectra[final_mask]
    mol_ids_valid = smiles_df.loc[final_mask, "mol_id"].values.astype(np.int32)
    smiles_valid = validation_df.loc[final_mask, "smiles_canonical"].values
    smiles_orig_valid = validation_df.loc[final_mask, "smiles"].values

    # Free large array
    del spectra

    # Write spectra as Parquet with pyarrow (more memory-efficient writes)
    import pyarrow as pa
    import pyarrow.parquet as pq

    wavename_cols = [f"w_{int(w)}" for w in wavenumber]

    schema = pa.schema([
        pa.field("mol_id", pa.int32()),
        *[pa.field(c, pa.float32()) for c in wavename_cols],
    ])

    BATCH = 20000
    writer = None
    for start in range(0, n_final, BATCH):
        end = min(start + BATCH, n_final)
        col_data = [pa.array(mol_ids_valid[start:end])]
        for ci in range(len(wavenumber)):
            col_data.append(pa.array(spectra_valid[start:end, ci], type=pa.float32()))

        names = ["mol_id"] + wavename_cols
        table = pa.table(dict(zip(names, col_data)), schema=schema)

        if writer is None:
            writer = pq.ParquetWriter(
                str(output_dir / "spectra.parquet"),
                schema=schema,
                compression="zstd",
                version="2.6",
            )
        writer.write_table(table)
        logger.info(f"  Written spectra batch {start//BATCH + 1}: rows {start}-{end}")

    if writer:
        writer.close()

    del spectra_valid
    logger.info(f"  Saved spectra → {output_dir / 'spectra.parquet'}")

    # Save SMILES (small, use pandas)
    smiles_out = pd.DataFrame({
        "mol_id": mol_ids_valid,
        "smiles": smiles_orig_valid,
        "smiles_canonical": smiles_valid,
    })
    smiles_out.to_parquet(output_dir / "smiles.parquet", index=False,
                          compression="zstd")
    logger.info(f"  Saved {len(smiles_out)} SMILES records → "
                f"{output_dir / 'smiles.parquet'}")

    # Save wavenumber axis
    wavenumber_df = pd.DataFrame({"wavenumber": wavenumber})
    wavenumber_df.to_parquet(output_dir / "wavenumber.parquet", index=False)
    logger.info(f"  Saved wavenumber axis → {output_dir / 'wavenumber.parquet'}")

    # ---- Summary ----
    logger.info(f"\n{'='*50}")
    logger.info(f"QM9S Preparation Summary:")
    logger.info(f"  Total SMILES entries: {len(smiles_df)}")
    logger.info(f"  Valid SMILES: {n_valid_smiles}")
    logger.info(f"  Invalid SMILES: {n_invalid_smiles}")
    logger.info(f"  Valid spectra: {spec_valid_mask.sum()}")
    logger.info(f"  Final valid pairs: {n_final}")
    logger.info(f"  Filtered out: {n_filtered}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
validate_dataset.py — 光谱基准数据集验证

检查数据质量，生成统计报告和可视化图表。

用法:
  python validate_dataset.py                     # 验证 benchmark.parquet
  python validate_dataset.py --mode qm9s         # 只验证 QM9S
  python validate_dataset.py --mode hitran       # 只验证 HITRAN
  python validate_dataset.py --mode visualize    # 生成样本可视化
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
N_PIXELS = 3501
WAVENUM_MIN = 500
WAVENUM_MAX = 4000


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[
                            logging.FileHandler(log_path, mode="w"),
                            logging.StreamHandler(sys.stdout),
                        ])
    return logging.getLogger(__name__)


def load_benchmark(benchmark_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """加载融合数据集（只加载元数据列，跳过光谱列）"""
    logger.info(f"Loading benchmark dataset from {benchmark_path}")

    # First check file size
    size_gb = benchmark_path.stat().st_size / 1e9
    logger.info(f"  File size: {size_gb:.2f} GB")

    # Load only metadata columns (not 3501 w_ columns)
    # Use pyarrow to read just a subset
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(benchmark_path)
    all_names = pf.schema_arrow.names
    meta_cols = [c for c in all_names if not c.startswith("w_")]
    logger.info(f"  Metadata columns: {meta_cols}")
    logger.info(f"  Spectra columns: {len(all_names) - len(meta_cols)}")

    df = pf.read(columns=meta_cols).to_pandas()
    logger.info(f"  Loaded {len(df)} rows, {len(meta_cols)} metadata columns")
    return df


def compute_stats(df: pd.DataFrame, logger: logging.Logger) -> dict:
    """计算基本统计信息"""
    stats = {
        "total_rows": len(df),
        "columns": list(df.columns),
    }

    if "source" in df.columns:
        stats["source_distribution"] = df["source"].value_counts().to_dict()

    if "split" in df.columns:
        stats["split_distribution"] = df["split"].value_counts().to_dict()

    if "smiles" in df.columns:
        stats["unique_smiles"] = int(df["smiles"].nunique())
        # Check for NaN SMILES
        n_null = df["smiles"].isna().sum()
        stats["null_smiles"] = int(n_null)
        if n_null > 0:
            logger.warning(f"  Found {n_null} null SMILES entries!")

    if "functional_groups" in df.columns:
        # Count molecules with functional group annotations
        n_with_fg = df["functional_groups"].apply(
            lambda x: x not in ("[]", "{}", "", None) if pd.notna(x) else False
        ).sum()
        stats["functional_groups_annotated"] = int(n_with_fg)

    logger.info(f"  Stats computed: {json.dumps(stats, indent=2)}")
    return stats


def validate_spectra(benchmark_path: Path, df_meta: pd.DataFrame,
                     logger: logging.Logger, n_sample: int = 1000):
    """抽样验证光谱质量"""
    logger.info(f"\nValidating spectra (sampling {n_sample} rows)...")

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(benchmark_path)

    # Sample rows
    n_total = len(df_meta)
    sample_idx = np.random.choice(n_total, min(n_sample, n_total), replace=False)

    # Read spectra columns for sampled rows
    w_cols = [c for c in pf.schema_arrow.names if c.startswith("w_")]
    wavenumbers = np.array([float(c.split("_")[1]) for c in w_cols], dtype=np.float32)

    # Read sample rows
    df_sample = pf.read(columns=w_cols).to_pandas().iloc[sample_idx]
    spectra = df_sample.values.astype(np.float32)

    logger.info(f"  Sampled {len(spectra)} spectra, {spectra.shape[1]} pixels each")

    # Check for NaN/Inf
    n_nan = np.isnan(spectra).sum()
    n_inf = np.isinf(spectra).sum()
    logger.info(f"  NaN values: {n_nan}")
    logger.info(f"  Inf values: {n_inf}")

    # Statistics per spectrum
    max_vals = spectra.max(axis=1)
    min_vals = spectra.min(axis=1)
    mean_vals = spectra.mean(axis=1)
    std_vals = spectra.std(axis=1)

    logger.info(f"  Max intensity:  min={max_vals.min():.4f}, "
                f"max={max_vals.max():.4f}, "
                f"mean={max_vals.mean():.4f}")
    logger.info(f"  Min intensity:  min={min_vals.min():.6f}, "
                f"max={min_vals.max():.6f}")
    logger.info(f"  Mean intensity: min={mean_vals.min():.6f}, "
                f"max={mean_vals.max():.6f}, "
                f"mean={mean_vals.mean():.6f}")

    # Check for flat spectra
    flat_count = (max_vals < 1e-10).sum()
    logger.info(f"  Flat/all-zero spectra: {flat_count}/{len(spectra)}")

    # Check for negative values
    neg_count = (min_vals < -1e-6).sum()
    logger.info(f"  Spectra with significant negative values: {neg_count}")

    return spectra, wavenumbers, sample_idx


def visualize_density_heatmap(benchmark_path: Path, output_dir: Path,
                               logger: logging.Logger):
    """密度热力图：加载全部光谱数据，用高斯平滑后的 2D 直方图展示谱峰分布。

    chunk 方式读取避免 OOM，支持任意规模数据集。
    """
    logger.info("Loading all spectra for density heatmap (chunked)...")

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(benchmark_path)

    w_cols = [c for c in pf.schema_arrow.names if c.startswith("w_")]
    wavenumbers = np.array([float(c.split("_")[1]) for c in w_cols], dtype=np.float32)

    # 先读一小批确定强度范围
    n_total = len(pf.read(columns=[w_cols[0]], use_threads=False))
    logger.info(f"  Total rows: {n_total}, spectra columns: {len(w_cols)}")

    # 确定 y 轴范围（从第一批采样 10% 行或者前 5000 行估算）
    preview_size = min(n_total, 5000)
    preview_batch = pf.read(columns=w_cols, use_threads=False).to_pandas().values.astype(np.float32)[:preview_size]
    v_low, v_high = np.percentile(preview_batch, [0.05, 99.95])
    y_margin = 0.05 * (v_high - v_low)
    y_range = [v_low - y_margin, v_high + y_margin]

    # 精细 bin
    n_wn_bins = 300
    n_y_bins = 300

    # 分块累加直方图
    accum = None
    batch_size = 2000
    n_batches = (n_total + batch_size - 1) // batch_size

    for i, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=w_cols, use_threads=False)):
        chunk = batch.to_pandas().values.astype(np.float32)
        H_chunk, x_edges, y_edges = np.histogram2d(
            np.tile(wavenumbers, chunk.shape[0]),
            chunk.ravel(),
            bins=[n_wn_bins, n_y_bins],
            range=[[WAVENUM_MIN, WAVENUM_MAX], y_range],
        )
        if accum is None:
            accum = H_chunk
        else:
            accum += H_chunk
        if (i + 1) % max(1, n_batches // 5) == 0:
            logger.info(f"    ... {i + 1}/{n_batches} batches processed")

    # histogram2d 返回 (n_x_bins, n_y_bins) → 转置为 (n_y_bins, n_x_bins) 适配 imshow
    accum = accum.T
    H_smooth = gaussian_filter(accum.astype(np.float64), sigma=1.5)
    H_smooth = np.clip(H_smooth, 1, None)

    # ── 绘制 ──
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#f5f5f5")  # 浅灰背景

    extent = [WAVENUM_MIN, WAVENUM_MAX, *y_range]
    im = ax.imshow(H_smooth, aspect="auto", origin="lower", extent=extent,
                   cmap="viridis", norm=matplotlib.colors.LogNorm())
    fig.colorbar(im, ax=ax, label="Spectra count (smoothed, log)")

    # 分位线
    p05 = np.percentile(preview_batch, 5, axis=0)
    p50 = np.percentile(preview_batch, 50, axis=0)
    p95 = np.percentile(preview_batch, 95, axis=0)

    ax.plot(wavenumbers, p50, color="#1a3a5c", linewidth=1.6,
            label="Median", alpha=0.85)
    ax.plot(wavenumbers, p05, color="#2a7a9c", linewidth=1.2, linestyle="--",
            label="5th / 95th percentile", alpha=0.75)
    ax.plot(wavenumbers, p95, color="#2a7a9c", linewidth=1.2, linestyle="--", alpha=0.75)

    ax.set_xlim(WAVENUM_MIN, WAVENUM_MAX)
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Spectra Density Heatmap — Peak Distribution (All Data, N={})".format(n_total),
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper right")

    outpath = output_dir / "spectra_density_heatmap.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved density heatmap → {outpath}")


def visualize_samples(benchmark_path: Path, df_meta: pd.DataFrame,
                      spectra: np.ndarray, wavenumbers: np.ndarray,
                      sample_idx: np.ndarray, output_dir: Path,
                      logger: logging.Logger, n_show: int = 9):
    """生成光谱可视化"""
    logger.info(f"\nGenerating visualizations → {output_dir}/")

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(n_show, len(sample_idx))):
        ax = axes[i]
        idx = sample_idx[i]
        row = df_meta.iloc[idx]

        smi = row.get("smiles", "?")
        source = row.get("source", "?")
        split = row.get("split", "?")

        ax.plot(wavenumbers, spectra[i], linewidth=0.8, alpha=0.8)
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"[{source}/{split}] {smi[:50]}", fontsize=9)
        ax.set_xlim(WAVENUM_MIN, WAVENUM_MAX)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    outpath = output_dir / "sample_spectra.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved sample spectra → {outpath}")

    # Also plot mean spectrum per source
    if len(np.unique([df_meta.iloc[i].get("source", "?")
                      for i in sample_idx])) > 1:
        fig, ax = plt.subplots(figsize=(12, 5))

        for source_name in ["qm9s", "hitran"]:
            mask = np.array([
                df_meta.iloc[i].get("source", "") == source_name
                for i in sample_idx
            ])
            if mask.sum() == 0:
                continue
            mean_spec = spectra[mask].mean(axis=0)
            ax.plot(wavenumbers, mean_spec, label=f"{source_name} (n={mask.sum()})",
                    linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Mean Intensity")
        ax.set_title("Mean Spectrum by Source")
        ax.legend()
        ax.set_xlim(WAVENUM_MIN, WAVENUM_MAX)

        outpath = output_dir / "mean_spectrum_by_source.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved mean spectrum → {outpath}")

    # Histogram of max intensities
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(spectra.max(axis=1), bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Max Intensity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Peak Intensities")
    ax.set_yscale("log")

    outpath = output_dir / "intensity_distribution.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved intensity distribution → {outpath}")

    # Molecular size distribution (from SMILES length as proxy)
    if "smiles" in df_meta.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        smiles_len = df_meta["smiles"].dropna().str.len()
        ax.hist(smiles_len, bins=50, alpha=0.7, edgecolor="black")
        ax.set_xlabel("SMILES String Length")
        ax.set_ylabel("Count")
        ax.set_title("Molecular Complexity (SMILES Length Distribution)")

        outpath = output_dir / "smiles_length_distribution.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved SMILES length distribution → {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Validate fused benchmark dataset")
    parser.add_argument("--benchmark-path", type=str,
                        default=str(DATA_DIR / "processed" / "benchmark.parquet"))
    parser.add_argument("--output-dir", type=str,
                        default=str(DATA_DIR / "processed"))
    parser.add_argument("--log-dir", type=str,
                        default=str(DATA_DIR / "logs"))
    parser.add_argument("--n-sample", type=int, default=1000,
                        help="Number of sample spectra to validate")
    parser.add_argument("--mode", choices=["all", "qm9s", "hitran", "visualize"],
                        default="all")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir / "validate_dataset.log")
    logger.info("=== Dataset Validation ===")

    # Create visualization subdirectory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    benchmark_path = Path(args.benchmark_path)
    if not benchmark_path.exists():
        logger.error(f"Benchmark dataset not found at {benchmark_path}")
        logger.error("Run scripts/build_benchmark.py first")
        sys.exit(1)

    # Load metadata
    df_meta = load_benchmark(benchmark_path, logger)

    # Statistics
    stats = compute_stats(df_meta, logger)

    # Save stats as JSON
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved → {stats_path}")

    # Validate spectra
    spectra, wavenumbers, sample_idx = validate_spectra(
        benchmark_path, df_meta, logger, args.n_sample
    )

    # Visualize
    visualize_samples(benchmark_path, df_meta, spectra, wavenumbers,
                      sample_idx, viz_dir, logger)
    visualize_density_heatmap(benchmark_path, viz_dir, logger)

    logger.info(f"\n{'='*50}")
    logger.info(f"Validation complete!")
    logger.info(f"  Stats: {stats_path}")
    logger.info(f"  Visualizations: {viz_dir}/")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()

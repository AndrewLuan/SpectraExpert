#!/usr/bin/env python3
"""
Figure 1a — Input Raman spectrum with detected peak annotations.

Optimised for ~45 mm (1.8 inch) column width.
"""

import sys, math
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "proj"))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from adjustText import adjust_text

DATA_DIR = Path("/home/pleiades/projects/SpectrExpert/data/qm9s")
OUTPUT_PDF = "figure1a_spectrum.pdf"

# ── load ────────────────────────────────────────────────────────
spec_df  = pd.read_parquet(DATA_DIR / "spectra.parquet")
cols = [c for c in spec_df.columns if c != "mol_id"]
wn       = pd.read_parquet(DATA_DIR / "wavenumber.parquet")["wavenumber"].values.astype(np.float32)
sp_norm  = spec_df.iloc[23602][cols].values.astype(np.float32)
sp_norm /= sp_norm.max()

# ── peaks ───────────────────────────────────────────────────────
peaks, _ = find_peaks(sp_norm, height=0.05, prominence=0.02, distance=5, width=1)
peak_wn, peak_h = wn[peaks].astype(int), sp_norm[peaks]

# ── figure (宽 3, 高 1) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.0, 1.0))

# 1. 谱线加粗 (lw=1.0)，使用低饱和度蓝色 (#4A7BB0)
ax.plot(wn, sp_norm, color="#4A7BB0", lw=1.0)
ax.scatter(peak_wn, peak_h, c="#B2182B", s=8, edgecolors="white", linewidths=0.4, zorder=5, marker="v")

# ── 筛选强峰 ────────────────────────────────────────────────────
strong = peak_h > 0.20
s_wn, s_h = peak_wn[strong], peak_h[strong]

# ── 标签排版（使用 adjustText 自动避让） ──────────────────────────
texts = []
for p, h in zip(s_wn, s_h):
    # 峰位置数字缩小至 5.0
    t = ax.text(
        p, h + 0.05, f"{p}", 
        fontsize=3.0, color="#B2182B",
        ha="center", va="bottom"
    )
    texts.append(t)

# ── 布局与样式微调 ────────────────────────────────────────────────
ax.set_xlim(400, 4100)
ax.set_ylim(-0.05, 1.55)  # 稍微拔高一点点上边界，给重合标签弹开留出空间
ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=3)
ax.set_ylabel("Intensity", fontsize=3)

# 2. 减小坐标轴外框的粗细 (轴线和刻度线均设为 0.4)
for spine in ax.spines.values():
    spine.set_linewidth(0.4)
ax.tick_params(labelsize=3, width=0.4, length=2.5)

# 3. 删除了原本在此处的 "XX peaks" 标签代码

# 让 adjust_text 自动计算并推开标签
adjust_text(
    texts, 
    ax=ax,
    x=s_wn, y=s_h,              # 目标点的原始坐标
    force_text=(0.2, 0.6),      # 限制横向排斥，强化纵向排斥（防止横向跑太远）
    expand_text=(1.2, 1.4),     # 标签周围的保护性边界气泡大小
    arrowprops=dict(arrowstyle="-", color="#B2182B", lw=0.4) # 指引线样式
)

fig.tight_layout(pad=0.15)
fig.savefig(OUTPUT_PDF, dpi=400)
print(f"✅ {OUTPUT_PDF}")
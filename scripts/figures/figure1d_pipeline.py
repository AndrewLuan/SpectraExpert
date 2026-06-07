#!/usr/bin/env python3
"""
Figure 1d — Pipeline architecture schematic (3-stage text-only diagram).

Optimised for ~45 mm (1.8 inch) column width.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "proj"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_PDF = "figure1d_pipeline.pdf"

INCH = 1.8
fig, ax = plt.subplots(figsize=(INCH, INCH * 1.0))
ax.axis("off")

# ── three boxes centered vertically ─────────────────────────────
boxes = [
    ("Stage 1\nContrastive\nRetrieval",  "#4393C3"),
    ("Stage 2\nSpectral\nTransition",    "#5AAE61"),
    ("Stage 3\nGuided\nReasoning",       "#F4A582"),
]
bw, bh = 0.38, 0.26
gap = 0.06
total_w = 3 * bw + 2 * gap
start_x = (1 - total_w) / 2
cy = 0.35

for i, (label, color) in enumerate(boxes):
    x = start_x + i * (bw + gap)
    rect = mpatches.FancyBboxPatch(
        (x, cy), bw, bh, boxstyle="round,pad=0.06",
        facecolor=color, edgecolor="white", linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(x + bw/2, cy + bh/2, label, ha="center", va="center",
            fontsize=5.5, color="white", fontweight="bold", linespacing=1.2)

    if i < len(boxes) - 1:
        ax.annotate("", xy=(x + bw + gap*0.4, cy + bh/2),
                    xytext=(x + bw - gap*0.4, cy + bh/2),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))

# input / output
ax.text(0.5, 0.08, "Input: 3,501-D spectrum", ha="center", fontsize=5,
        color="#555", style="italic")
ax.text(0.5, 0.85, "Output: molecule (SMILES) + confidence", ha="center", fontsize=5,
        color="#555", style="italic")

fig.savefig(OUTPUT_PDF, dpi=400, bbox_inches="tight", pad_inches=0.05)
print(f"✅ {OUTPUT_PDF}  ({Path(OUTPUT_PDF).stat().st_size/1024:.0f} KB)")

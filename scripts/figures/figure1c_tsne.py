#!/usr/bin/env python3
"""
Figure 1c — t-SNE embedding neighbourhood around the query + top-10 candidates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "proj"))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

EMB_PATH = Path("spectra_expert_pkg/embeddings.npz")
DATA_DIR = Path("data/qm9s")
OUTPUT_PDF = "figure1c_tsne.pdf"

# ── load ────────────────────────────────────────────────────────
emb_data = np.load(EMB_PATH, allow_pickle=True)
db_embs = emb_data["embeddings"].astype(np.float32)
db_smiles = emb_data["smiles"]
db_mol_ids = emb_data["mol_ids"]

spec_df = pd.read_parquet(DATA_DIR / "spectra.parquet")
cols = [c for c in spec_df.columns if c != "mol_id"]
idx = 23602
sp = spec_df.iloc[idx][cols].values.astype(np.float32)

from proj.spectra_expert.analyze_spectra import SpectralEncoder
CKPT_PATH = "spectra_expert_pkg/checkpoints/spectral_simclr_qm9s_8gpu_best.pth"
encoder = SpectralEncoder(CKPT_PATH)
q_emb = encoder.encode_spectrum(sp)
indices, sims = SpectralEncoder.retrieve(q_emb, db_embs, topk=500)
indices = indices.ravel()

# ── t-SNE on a representative background ────────────────────────
rng = np.random.RandomState(42)
bg_idx = rng.choice(len(db_embs), 3000, replace=False)

q_db_idx = int(np.where(db_mol_ids == spec_df.iloc[idx]["mol_id"])[0][0])
cand_indices, _ = SpectralEncoder.retrieve(q_emb, db_embs, topk=10)
cand_indices = cand_indices.ravel()

tsne_data = np.concatenate([db_embs[bg_idx], q_emb[None, :], db_embs[cand_indices]])
tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=500, verbose=0)
tsne_2d = tsne.fit_transform(tsne_data.astype(np.float64))

n_bg = len(bg_idx)
bg_pt = tsne_2d[:n_bg]
q_pt  = tsne_2d[n_bg]
cand_pt = tsne_2d[n_bg+1:]

# ── narrow viewport around query + candidates ───────────────────
all_focus = np.vstack([q_pt[None, :], cand_pt])
margin = 3                                          # ← 改动③：由 1.5 缩小至 0.8
xmin, xmax = all_focus[:, 0].min() - margin, all_focus[:, 0].max() + margin
ymin, ymax = all_focus[:, 1].min() - margin, all_focus[:, 1].max() + margin

# ── plot ────────────────────────────────────────────────────────
INCH = 1.8
fig, ax = plt.subplots(figsize=(INCH *2 , INCH))     # ← 改动①：长宽比 3:1（高:宽）

ax.scatter(bg_pt[:, 0], bg_pt[:, 1],
           c="#2E2E2E", s=2, alpha=0.6)               # ← 改动②：灰色加深，透明度提高

for i, pt in enumerate(cand_pt):
    lab = f"#{i+1}" if i < 3 else ""
    ax.scatter(*pt, c="#F4A582", s=16, edgecolors="#CA0020", linewidths=0.4, zorder=6)
    if lab:
        ax.annotate(lab, pt, (pt[0]+0.08, pt[1]+0.08), fontsize=4.5, color="#CA0020",
                    arrowprops=dict(arrowstyle="-", color="#CA0020", lw=0.2))

ax.scatter(*q_pt, c="#0571B0", s=30, edgecolors="white", linewidths=0.6, zorder=10, marker="*")

for pt in cand_pt:
    ax.plot([q_pt[0], pt[0]], [q_pt[1], pt[1]], color="#CA0020", lw=0.2, alpha=0.3, linestyle="--")

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("t-SNE 1", fontsize=5)
ax.set_ylabel("t-SNE 2", fontsize=5)
ax.tick_params(labelsize=4)

from matplotlib.lines import Line2D
leg = ax.legend(
    [Line2D([0],[0], marker='*', color='w', markerfacecolor='#0571B0', markersize=5),
     Line2D([0],[0], marker='o', color='w', markerfacecolor='#F4A582', markersize=4)],
    ["Query", "Top-10"], loc="upper left", fontsize=4.5, framealpha=0.7)
leg.get_frame().set_linewidth(0.3)

fig.tight_layout(pad=0.3)
fig.savefig(OUTPUT_PDF, dpi=400)
print(f"✅ {OUTPUT_PDF}  ({Path(OUTPUT_PDF).stat().st_size/1024:.0f} KB)")
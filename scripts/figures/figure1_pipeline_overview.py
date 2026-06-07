#!/usr/bin/env python3
"""
Figure 1 for SpectraExpert paper — Pipeline Overview Illustration.

Generates a 3-panel PDF showing:
  Panel A: Input spectrum with detected peaks annotated
  Panel B: Top-10 candidate molecules with similarity scores
  Panel C: t-SNE embedding space with query + candidates highlighted
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "proj"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import Draw

from proj.spectra_expert.analyze_spectra import SpectralEncoder

# ── paths ─────────────────────────────────────────────────────────────
DATA_DIR = Path("data/qm9s")
EMB_PATH = Path("spectra_expert_pkg/embeddings.npz")
CKPT_PATH = Path("spectra_expert_pkg/checkpoints/spectral_simclr_qm9s_8gpu_best.pth")
OUTPUT_PDF = "figure1_pipeline_overview.pdf"

# ── load data ─────────────────────────────────────────────────────────
print("[1/6] Loading data ...")
spec_df = pd.read_parquet(DATA_DIR / "spectra.parquet")
smiles_df = pd.read_parquet(DATA_DIR / "smiles.parquet")
wn_arr = pd.read_parquet(DATA_DIR / "wavenumber.parquet")["wavenumber"].values.astype(np.float32)

spec_cols = [c for c in spec_df.columns if c != "mol_id"]
idx_query = 23602  # CC(=O)c1ccccc1
spectrum = spec_df.iloc[idx_query][spec_cols].values.astype(np.float32)

print(f"  Spectrum: {spectrum.shape}, wavenumber: {wn_arr[0]:.0f}-{wn_arr[-1]:.0f} cm⁻¹")
print(f"  True SMILES: {smiles_df.iloc[idx_query]['smiles_canonical']}")

# ── detect peaks ──────────────────────────────────────────────────────
print("[2/6] Detecting peaks ...")
# Normalise to 0-1 range for consistent peak finding
spec_norm = spectrum / spectrum.max()
peaks, props = find_peaks(
    spec_norm,
    height=0.05,
    prominence=0.02,
    distance=5,
    width=1,
)
peak_positions = wn_arr[peaks].astype(int)
peak_heights = spec_norm[peaks]
print(f"  Found {len(peak_positions)} peaks: {peak_positions[:10].tolist()}")

# ── encode and retrieve top-10 ────────────────────────────────────────
print("[3/6] Encoding & retrieving ...")
emb_data = np.load(EMB_PATH, allow_pickle=True)
db_embs = emb_data["embeddings"].astype(np.float32)
db_smiles = emb_data["smiles"]
db_mol_ids = emb_data["mol_ids"]

encoder = SpectralEncoder(str(CKPT_PATH))
query_emb = encoder.encode_spectrum(spectrum)
indices, sims = SpectralEncoder.retrieve(query_emb, db_embs, topk=10)

print(f"  Top-1: {db_smiles[indices[0]]} (sim={sims[0]:.4f})")

# ── t-SNE embedding space ─────────────────────────────────────────────
print("[4/6] Computing t-SNE (subsample + query + candidates) ...")
# Subsample 3000 random + query + 10 candidates
rng = np.random.RandomState(42)
subsample_idx = rng.choice(len(db_embs), 3000, replace=False)
highlight_idx = [int(np.where(db_mol_ids == db_mol_ids[i])[0][0]) for i in indices]

# Check if query embedding is already in the database
query_mol_id = spec_df.iloc[idx_query]["mol_id"]
query_db_idx = int(np.where(db_mol_ids == query_mol_id)[0][0])
query_in_db = query_db_idx in subsample_idx

tsne_idx = np.concatenate([
    [query_db_idx] if not query_in_db else [],
    highlight_idx,
    subsample_idx,
])
tsne_data = np.concatenate([
    query_emb.reshape(1, -1),
    db_embs[highlight_idx],
    db_embs[subsample_idx],
])

tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=500, verbose=0)
tsne_2d = tsne.fit_transform(tsne_data.astype(np.float64))

n_highlight = len(highlight_idx)
n_subsample = len(subsample_idx)

query_tsne = tsne_2d[0]
cand_tsne = tsne_2d[1:1+n_highlight]
bg_tsne = tsne_2d[1+n_highlight:]

# ── draw molecular structures via RDKit ───────────────────────────────
print("[5/6] Drawing molecular structures ...")
mols = []
legends = []
for i in indices:
    smi = db_smiles[i]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmiles("C")  # fallback
    mols.append(mol)
    rank = list(indices).index(i) + 1
    legends.append(f"#{rank}  sim={sims[list(indices).index(i)]:.3f}")

img_mols = Draw.MolsToGridImage(
    mols, molsPerRow=5, subImgSize=(180, 120),
    legends=legends, returnPNG=False,
)
# RDKit returns a PIL Image; save to a temp buffer for display
img_mols.save("_tmp_mols.png", dpi=(200, 200))

# ── create the figure ─────────────────────────────────────────────────
print("[6/6] Building figure layout ...")

# Use a professional style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

fig = plt.figure(figsize=(14, 10), constrained_layout=False)
gs = GridSpec(3, 2, figure=fig, width_ratios=[1.2, 1], height_ratios=[0.6, 0.6, 1],
              hspace=0.40, wspace=0.35, left=0.07, right=0.97, top=0.96, bottom=0.06)

# ── Panel A: Input spectrum with peak annotations ─────────────────────
ax_spec = fig.add_subplot(gs[0, :])
ax_spec.plot(wn_arr, spec_norm, color="#2166AC", linewidth=0.8, label="Raman spectrum")
# Annotate peaks
for p, h in zip(peak_positions, peak_heights):
    ax_spec.annotate(f"{p}", xy=(p, h), xytext=(p, h + 0.06),
                     fontsize=6, ha="center", color="#B2182B", fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color="#B2182B", lw=0.5))

ax_spec.scatter(peak_positions, peak_heights, color="#B2182B", s=20,
                edgecolors="white", linewidths=0.5, zorder=5, marker="v")
ax_spec.set_xlabel("Raman shift (cm⁻¹)", fontsize=10)
ax_spec.set_ylabel("Normalised intensity", fontsize=10)
ax_spec.set_title("A  Input Spectrum  —  Acetophenone (CC(=O)c1ccccc1)", fontsize=11, loc="left")
ax_spec.set_xlim(400, 4100)
ax_spec.set_ylim(-0.02, 1.05)
ax_spec.text(0.99, 0.95, f"{len(peak_positions)} peaks detected", transform=ax_spec.transAxes,
             fontsize=8, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", lw=0.5))

# ── Panel B: Top-10 candidate structures ──────────────────────────────
ax_mols = fig.add_subplot(gs[1, :])
ax_mols.axis("off")
ax_mols.set_title("B  Top-10 Retrieval Candidates  (similarity ↓)", fontsize=11, loc="left", pad=5)

# Load the pre-rendered molecule grid
from PIL import Image
pil_img = Image.open("_tmp_mols.png")
ax_mols.imshow(np.asarray(pil_img))
ax_mols.axis("off")

# Add a bracket label for the correct match
ax_mols.annotate("← Correct", xy=(0.55, 0.65), xycoords="axes fraction",
                 fontsize=9, color="#0571B0", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#0571B0", lw=1.5))

# ── Panel C: t-SNE embedding space ────────────────────────────────────
ax_tsne = fig.add_subplot(gs[2, 0])
ax_tsne.scatter(bg_tsne[:, 0], bg_tsne[:, 1], c="lightgray", s=3, alpha=0.5, label="Database (N=129,817)")
ax_tsne.scatter(cand_tsne[:, 0], cand_tsne[:, 1], c="#F4A582", s=50, edgecolors="#CA0020",
                linewidths=0.8, zorder=6, label="Top-10 candidates")
ax_tsne.scatter(query_tsne[0], query_tsne[1], c="#0571B0", s=120, edgecolors="white",
                linewidths=1.5, zorder=10, marker="*", label="Query spectrum")

# Draw lines from query to top candidates
for c in cand_tsne:
    ax_tsne.plot([query_tsne[0], c[0]], [query_tsne[1], c[1]],
                 color="#CA0020", lw=0.3, alpha=0.4, linestyle="--")

ax_tsne.set_title("C  Embedding Space (t-SNE)", fontsize=11, loc="left")
ax_tsne.set_xlabel("t-SNE 1", fontsize=9)
ax_tsne.set_ylabel("t-SNE 2", fontsize=9)
ax_tsne.legend(framealpha=0.7, markerscale=0.7, loc="lower left", fontsize=7)
ax_tsne.set_xticks([])
ax_tsne.set_yticks([])

# ── Panel D: Pipeline schematic ──────────────────────────────────────
ax_schema = fig.add_subplot(gs[2, 1])
ax_schema.axis("off")
# Draw a simplified pipeline schematic
stages = [
    ("Stage 1\nContrastive\nRetrieval", "#4393C3",
     "Encode →\ncosine\nretrieval"),
    ("Stage 2\nSpectral\nTransition", "#5AAE61",
     "Peak detect\n+\nNakamoto\nrules"),
    ("Stage 3\nGuided\nReasoning", "#F4A582",
     "LLM CoT\n+\nChecklist\nverify"),
]
box_w, box_h = 0.28, 0.30
arrow_len = 0.06
start_x, start_y = 0.04, 0.60

for i, (label, color, detail) in enumerate(stages):
    x = start_x + i * (box_w + arrow_len + 0.02)
    y = start_y
    rect = mpatches.FancyBboxPatch(
        (x, y), box_w, box_h, boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85,
    )
    ax_schema.add_patch(rect)
    ax_schema.text(x + box_w/2, y + box_h*0.70, label, ha="center", va="center",
                   fontsize=8, color="white", fontweight="bold", linespacing=1.2)
    ax_schema.text(x + box_w/2, y + box_h*0.30, detail, ha="center", va="center",
                   fontsize=6.5, color="white", alpha=0.9, linespacing=1.3)

    if i < len(stages) - 1:
        ax_schema.annotate("", xy=(x + box_w + 0.01, y + box_h/2),
                           xytext=(x + box_w - 0.01, y + box_h/2),
                           arrowprops=dict(arrowstyle="->", color="#333333", lw=2))
    # Icons (text labels instead of emoji for font compatibility)
    icon_labels = ["[Spectrum]", "[Molecule]", "[LLM]"]
    ax_schema.text(x + box_w/2, start_y - 0.10, icon_labels[i], ha="center",
                   fontsize=9, color="#555", style="italic")

# Input / Output boxes
inp_box = mpatches.FancyBboxPatch(
    (start_x, start_y - 0.25), box_w, 0.08, boxstyle="round,pad=0.04",
    facecolor="#FEE090", edgecolor="gray", linewidth=0.8,
)
ax_schema.add_patch(inp_box)
ax_schema.text(start_x + box_w/2, start_y - 0.21, "Input: 3501-dim spectrum",
               ha="center", fontsize=7, color="#333")

out_box = mpatches.FancyBboxPatch(
    (start_x + 2*(box_w+arrow_len+0.02), start_y - 0.40), box_w, 0.12,
    boxstyle="round,pad=0.04", facecolor="#D1E5F0", edgecolor="#4393C3", linewidth=1.2,
)
ax_schema.add_patch(out_box)
ax_schema.text(start_x + 2*(box_w+arrow_len+0.02) + box_w/2, start_y - 0.34,
               "Output:\nSMILES + Confidence", ha="center", fontsize=7, color="#333",
               fontweight="bold")

# Arrow from input to Stage 1, and Stage 3 to output
ax_schema.annotate("", xy=(start_x, start_y - 0.25), xytext=(start_x, start_y - 0.08),
                   arrowprops=dict(arrowstyle="->", color="#333", lw=1))
ax_schema.annotate("", xy=(start_x + 2*(box_w+arrow_len+0.02) + box_w/2, start_y - 0.40),
                   xytext=(start_x + 2*(box_w+arrow_len+0.02) + box_w/2, start_y - 0.08),
                   arrowprops=dict(arrowstyle="->", color="#333", lw=1))

ax_schema.set_title("D  Pipeline Architecture", fontsize=11, loc="left", pad=5)
ax_schema.set_xlim(0, 1)
ax_schema.set_ylim(0, 1)

# ── save ──────────────────────────────────────────────────────────────
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0.1)
print(f"\n✅ Figure saved to: {OUTPUT_PDF}")

# Cleanup
Path("_tmp_mols.png").unlink(missing_ok=True)
print("Done.")

#!/usr/bin/env python3
"""
Figure 1b — Top-5 retrieval candidates as RDKit molecular structures.
Outputs a vector SVG for print-quality rendering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "proj"))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image

from proj.spectra_expert.analyze_spectra import SpectralEncoder

EMB_PATH  = Path("spectra_expert_pkg/embeddings.npz")
CKPT_PATH = Path("spectra_expert_pkg/checkpoints/spectral_simclr_qm9s_8gpu_best.pth")
DATA_DIR  = Path("data/qm9s")
OUTPUT_SVG = "figure1b_candidates.svg"   # ← 矢量图

# ── load & retrieve ─────────────────────────────────────────────
spec_df = pd.read_parquet(DATA_DIR / "spectra.parquet")
cols = [c for c in spec_df.columns if c != "mol_id"]
idx  = 23602
sp   = spec_df.iloc[idx][cols].values.astype(np.float32)

emb_data   = np.load(EMB_PATH, allow_pickle=True)
db_embs    = emb_data["embeddings"].astype(np.float32)
db_smiles  = emb_data["smiles"]
db_mol_ids = emb_data["mol_ids"]

encoder = SpectralEncoder(str(CKPT_PATH))
q_emb   = encoder.encode_spectrum(sp)
indices, sims = SpectralEncoder.retrieve(q_emb, db_embs, topk=5)   # ← top5

# ── render each molecule to a high-res PNG via Cairo drawer ────
#    (SVG-per-mol is possible but embedding in matplotlib is simpler)
MOL_W, MOL_H = 400, 320   # pixels per cell — plenty of resolution

def mol_to_pil(mol, w=MOL_W, h=MOL_H):
    """Render an RDKit mol to a PIL Image via the Cairo-based SVG drawer."""
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg_bytes = drawer.GetDrawingText().encode()
    # Convert SVG → PNG at the same pixel size via cairosvg if available,
    # otherwise fall back to the raster PNG drawer.
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=w, output_height=h)
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except ImportError:
        drawer2 = rdMolDraw2D.MolDraw2DCairo(w, h)
        drawer2.DrawMolecule(mol)
        drawer2.FinishDrawing()
        return Image.open(io.BytesIO(drawer2.GetDrawingText())).convert("RGBA")

mols, labels, images = [], [], []
for rank, (ix, s) in enumerate(zip(indices, sims)):
    smi = db_smiles[ix]
    mol = Chem.MolFromSmiles(smi) or Chem.MolFromSmiles("C")
    mols.append(mol)
    labels.append(f"")
    images.append(mol_to_pil(mol))

# ── compose figure: 1 row × 5 columns, vector-output SVG ───────
N = len(mols)
fig_w = N * 1.4        # inches  (1.4 in per molecule)
fig_h = 1.4 + 0.25     # height  + space for label

fig, axes = plt.subplots(1, N, figsize=(fig_w, fig_h))
fig.subplots_adjust(left=0, right=1, bottom=0.12, top=1, wspace=0.04)

for ax, img, label in zip(axes, images, labels):
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")
    ax.set_title(
        label,
        fontsize=6.5,
        pad=3,
        fontfamily="DejaVu Sans",
    )

fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
print(f"✅ {OUTPUT_SVG}  ({Path(OUTPUT_SVG).stat().st_size/1024:.0f} KB)")
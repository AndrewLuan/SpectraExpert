#!/usr/bin/env python3
"""
Build a held-out test set from ChEMBL Raman spectra.

Reads ``data/chembl_raman_500_4000_3501d.csv``, filters by SMILES length
(shortest ~200 molecules), validates SMILES strings, and saves the result
as a Parquet file for downstream evaluation.

Usage::

    python build_test_set.py \\
        --input  ../../data/chembl_raman_500_4000_3501d.csv \\
        --output ../../data/processed/chembl_test_set.parquet \\
        --max-samples 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def build_test_set(
    csv_path: str,
    output_path: str,
    max_samples: int = 200,
    max_smiles_len: int | None = None,
) -> pd.DataFrame:
    """Load, filter, validate, and save a held-out test set.

    Parameters
    ----------
    csv_path : str
        Path to the ChEMBL Raman CSV (columns: SMILES + 3501 wavenumbers).
    output_path : str
        Where to save the result as Parquet.
    max_samples : int
        Target number of test molecules (actual count may be slightly lower
        after dropping invalid SMILES).
    max_smiles_len : int | None
        If given, cap SMILES length at this value (useful for reproducibility).

    Returns
    -------
    pd.DataFrame with columns: smiles, smiles_len, and 3501 spectral columns.
    """
    print(f"[build_test_set] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    smiles_col = df.columns[0]

    # --- sort by SMILES length ---
    df["smiles_len"] = df[smiles_col].astype(str).str.len()
    sorted_df = df.sort_values("smiles_len").reset_index(drop=True)

    if max_smiles_len is not None:
        test_df = sorted_df[sorted_df["smiles_len"] <= max_smiles_len].copy()
        print(f"  Capped at length ≤{max_smiles_len} → {len(test_df)} molecules")
    else:
        # Find the maximum length that gives at least max_samples
        cum = 0
        cut_len = None
        for length, count in sorted_df["smiles_len"].value_counts().sort_index().items():
            if cum + count >= max_samples:
                cut_len = int(length)
                break
            cum += count

        test_df = sorted_df[sorted_df["smiles_len"] <= cut_len].copy()
        print(f"  Length ≤{cut_len} → {len(test_df)} molecules (target {max_samples})")

    smiles_list = test_df[smiles_col].values
    spec_values = test_df.iloc[:, 1:].values.astype(np.float32)

    # --- validate SMILES ---
    valid_mask = np.array(
        [Chem.MolFromSmiles(s) is not None for s in smiles_list]
    )
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  Dropping {n_invalid} invalid SMILES:")
        for s in smiles_list[~valid_mask]:
            print(f"    {s}")

    smiles_valid = smiles_list[valid_mask]
    spec_valid = spec_values[valid_mask]

    print(f"  Valid molecules: {len(smiles_valid)}")

    # --- build output DataFrame ---
    wavenumber_cols = list(df.columns[1:])  # e.g. 500.0, 501.0, ...
    result = pd.DataFrame(spec_valid, columns=wavenumber_cols)
    result.insert(0, "smiles", smiles_valid)
    result["smiles_len"] = result["smiles"].str.len()

    # --- save ---
    output_path = str(output_path)
    result.to_parquet(output_path, index=False)
    print(f"  Saved: {output_path}  ({len(result)} rows × {len(result.columns)} cols)")

    # --- quick stats ---
    print(f"  SMILES length range: {result['smiles_len'].min()}–{result['smiles_len'].max()}")
    print(f"  Spectrum range: {spec_valid.min():.4f}–{spec_valid.max():.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build held-out test set from ChEMBL Raman spectra",
    )
    parser.add_argument(
        "--input", type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent.parent
            / "data" / "chembl_raman_500_4000_3501d.csv"
        ),
        help="Path to the ChEMBL Raman CSV file",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent.parent
            / "data" / "processed" / "chembl_test_set.parquet"
        ),
        help="Output Parquet path",
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Target number of test molecules",
    )
    parser.add_argument(
        "--max-smiles-len", type=int, default=None,
        help="Hard cap on SMILES length (overrides --max-samples)",
    )
    args = parser.parse_args()

    build_test_set(
        csv_path=args.input,
        output_path=args.output,
        max_samples=args.max_samples,
        max_smiles_len=args.max_smiles_len,
    )


if __name__ == "__main__":
    main()

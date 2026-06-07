#!/usr/bin/env python3
"""
SpectraExpert Pipeline — CLI entry point.

Usage
-----
    # Run on a specific QM9S index
    python run_pipeline.py --index 42

    # Run on a custom spectrum file (.npy)
    python run_pipeline.py --query_file my_spectrum.npy

    # Use Anthropic instead of OpenAI
    python run_pipeline.py --index 5 --provider anthropic --model claude-sonnet-4-6

    # Use dummy client (for testing, no API key needed)
    python run_pipeline.py --index 5 --provider dummy

    # Print summary
    python run_pipeline.py --index 0 --summary
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from proj.spectra_expert.pipeline import SpectraExpertPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="SpectraExpert — 3-stage molecular identification from Raman spectra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Query source
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--index", type=int, default=None,
        help="Index in the QM9S database to use as query",
    )
    query_group.add_argument(
        "--query_file", type=str, default=None,
        help="Path to a .npy file containing a 3501-dim spectrum",
    )
    query_group.add_argument(
        "--smiles", type=str, default=None,
        help="SMILES string to search for in the database and use its spectrum as query",
    )

    # LLM config
    parser.add_argument(
        "--provider", type=str, default="dummy",
        choices=["openai", "anthropic", "dummy"],
        help="LLM provider (default: dummy — no API key needed)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model name (e.g. gpt-4o, claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--api-key-env", type=str, default=None,
        help="Environment variable name for the API key",
    )

    # Pipeline config
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of candidates to retrieve (default: 10)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file",
    )

    # Output
    parser.add_argument(
        "--summary", action="store_true",
        help="Print a human-readable summary of the result",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save PipelineResult fields as a .npz file",
    )

    args = parser.parse_args()

    # ── build config ───────────────────────────────────────────────
    config_dict = {}
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
        config_dict = cfg.to_dict()

    # Override with CLI args
    config_dict.setdefault("pipeline", {})["top_k"] = args.top_k
    config_dict.setdefault("stage3", {}).setdefault("llm", {})["provider"] = args.provider
    if args.model:
        config_dict["stage3"]["llm"]["model"] = args.model
    if args.api_key_env:
        config_dict["stage3"]["llm"]["api_key_env"] = args.api_key_env

    # ── get query spectrum ─────────────────────────────────────────
    data_dir = PipelineConfig.defaults().get("stage1", "data_dir")

    if args.index is not None:
        import pandas as pd
        spectra_path = Path(data_dir) / "spectra.parquet"
        smiles_path = Path(data_dir) / "smiles.parquet"

        spec_df = pd.read_parquet(spectra_path)
        smiles_df = pd.read_parquet(smiles_path)

        idx = args.index
        spec_cols = [c for c in spec_df.columns if c != "mol_id"]
        spectrum = spec_df.iloc[idx][spec_cols].values.astype(np.float32)
        query_info = f"Index {idx} — SMILES: {smiles_df.iloc[idx]['smiles_canonical']}"

    elif args.query_file:
        spectrum = np.load(args.query_file).astype(np.float32)
        if spectrum.ndim == 2:
            spectrum = spectrum.flatten()
        query_info = f"File: {args.query_file}"

    elif args.smiles:
        import pandas as pd
        from rdkit import Chem

        smiles_path = Path(data_dir) / "smiles.parquet"
        spectra_path = Path(data_dir) / "spectra.parquet"

        smiles_df = pd.read_parquet(smiles_path)
        target = Chem.MolToSmiles(Chem.MolFromSmiles(args.smiles), canonical=True)

        match = smiles_df[smiles_df["smiles_canonical"] == target]
        if match.empty:
            print(f"SMILES '{args.smiles}' not found in database.")
            sys.exit(1)

        idx = match.index[0]
        spec_df = pd.read_parquet(spectra_path)
        spec_cols = [c for c in spec_df.columns if c != "mol_id"]
        spectrum = spec_df.iloc[idx][spec_cols].values.astype(np.float32)
        query_info = f"SMILES: {target} (index {idx})"

    print(f"\n[SpectraExpert] Query: {query_info}")
    print(f"[SpectraExpert] Spectrum shape: {spectrum.shape}")
    print(f"[SpectraExpert] LLM provider: {args.provider}\n")

    # ── run pipeline ──────────────────────────────────────────────
    pipeline = SpectraExpertPipeline(config=config_dict)
    result = pipeline.run(spectrum)

    # ── output ─────────────────────────────────────────────────────
    if args.summary:
        result.print_summary()

    if args.output:
        np.savez(args.output,
                 decision=result.stage3.decision,
                 confidence=result.stage3.confidence,
                 explanation=result.stage3.explanation,
                 modified_smiles=str(result.stage3.modified_smiles or ""),
                 n_peaks=len(result.stage1.detected_peaks),
                 n_candidates=len(result.stage1.candidates),
                 duration=result.duration_seconds,
                 )
        print(f"Results saved to: {args.output}")

    # Print top candidates
    print("\nTop candidates:")
    for c in result.stage1.candidates[:5]:
        print(f"  {c}")
    print(f"\nFinal decision: {result.stage3.decision}")
    print(f"Confidence: {result.stage3.confidence:.2f}")
    if result.stage3.explanation:
        print(f"Explanation: {result.stage3.explanation}")


if __name__ == "__main__":
    main()

# Paper artifact manifest

This repository is the public artifact for the SpectraExpert paper. It separates the material needed to inspect or reproduce the work from machine-local caches and restricted or externally hosted datasets.

## Included in the repository

- `spectraexpert/`: the retrieval, peak-analysis, LLM-reasoning, training, and evaluation implementation.
- `scripts/`: data preparation, frozen-split construction, experiment launchers, scoring, normalization, and figure-generation entry points.
- `tests/`: unit, contract, and smoke tests for the public implementation.
- `configs/`: credential-free example configuration. API secrets are read only from environment variables.
- `knowledge_base/`: the structured and human-readable Nakamoto group-frequency rules used by the reasoning stage.
- `paper/`: the LaTeX source, bibliography, publication figures, reproducibility notes, and compact frozen evidence snapshots.
- `outputs/`: the frozen multi-model requests, responses, recovery attempts, per-query metrics, aggregate reports, and provenance hashes used by the five-model Agent comparison.
- `output/pdf/SpectraExpert_camera_ready.pdf`: the released paper PDF.

The authoritative paper-facing aggregate is `paper/results/agent-multimodel-20260909/summary.json`. The broader evidence snapshot used to check the remaining tables is under `paper/results/jj-20260909/`. The `outputs/` records contain the frozen inference payloads and model responses but no API credentials, model checkpoints, or copy of the full raw spectral corpus.

## Externally hosted data

The raw training and evaluation spectra are not duplicated in Git because of dataset size and distribution constraints:

- QM9S: <https://doi.org/10.6084/m9.figshare.24235333.v3>
- Raman/IR spectra associated with the ChEMBL extension: <https://doi.org/10.1038/s41597-025-04184-1>

After obtaining the data from its authoritative source, use `scripts/prepare_qm9s.py`, `scripts/build_benchmark.py`, `scripts/prepare_raman_jsonl.py`, or `scripts/prepare_specgpt_theory.py`, depending on the experiment. The experiment notes under `docs/` record the frozen split, candidate-library, and scoring conventions used by the paper.

## Deliberately excluded

The repository ignores checkpoints, full raw spectra, intermediate parquet/CSV/NumPy files, temporary render directories, editor files, and machine-specific handoff notes. These files are either reproducible intermediates, too large for ordinary Git, restricted by their upstream distribution terms, or unnecessary for checking the reported numbers.

No API key is stored in the repository. Example configurations name environment variables only.

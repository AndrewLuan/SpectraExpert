# NMR retrieval experiments

This directory brings the experimental implementation onto the repository's `main` branch. `nmr_forward/` contains the forward models. `nmr_transfer/` contains the retrieval, observation and pool-ranking implementation. This directory contains the frozen-center, controlled-cap and saved-tensor diagnostic tools.

| Location | Purpose |
|---|---|
| `fixedcenter/` | Train-only role centers, ranking matrix, eligibility checks and summaries |
| `cap_controlled/` | C5/C10 plans, queue supervision, verification and synthetic tests |
| `cap_execution/train/` | Source collected from the actual controlled training stage |
| `cap_execution/native_cost_evaluation/` | Source collected from the native-cost evaluation stage |
| `cap_execution/mechanism/` | Source collected from the fixed-row mechanism stage |
| `diagnosis/` | Closed-output checks, diagnostic algebra, supervision and backend probes |

`source_provenance.json` records the SHA-256 of each imported source file and its origin. The original local experiment branch and all sealed bundles remain intact. The new Git commit is an import made after those runs. For past runs, source SHA-256 establishes code identity; the import does not claim that a past run executed this new commit.

## Environment

Use Python 3.11. The local check environment used Python 3.11.16 and PyTorch 2.14.0 on macOS ARM. The JJ environment used Python 3.11.16, PyTorch 2.14.0+cu130, CUDA 13.0 and an A800 GPU. The requirements files record those observed package versions. `igraph` was observed on JJ and is needed by the graph-model family.

- `requirements-cpu.txt`: CPU check environment.
- `requirements-cuda.txt`: JJ package versions, including the CUDA wheel identifier.
- `protocol_defaults.json`: scientific constants and execution limits, without dataset records or local credentials.

The archived drivers retain their source-path and source-SHA checks. To reproduce an archived stage, stage the byte-identical files beside its private frozen metadata and use its recorded layout. In particular, the historical fixed-center adapter expects the scientific checkout at `/home/user/IRfingerprints/SpectrExpert` on JJ. The controlled loader also checks the fixed-center file SHA. Preserve that contract when reproducing a past run. A new layout or implementation requires a new explicit request and source fingerprint.

The mechanism directory deliberately remains a sibling of `native_cost_evaluation/`; its evaluator verifies that relative import.

## Local checks

Run these commands from the repository root with the check environment already installed:

```sh
python -m compileall -q nmr_forward nmr_transfer research/nmr_retrieval
cd research/nmr_retrieval/diagnosis
python -m unittest test_top10_barrier_math test_probe_saved_jacobian_failure_mac_stage
python synthetic_probe_saved_jacobian_JJ_algebra.py
```

These commands check syntax, synthetic contracts and saved-value algebra. They do not load experimental checkpoints or start a training or retrieval run. CUDA checks are a separate, admitted stage on the recorded JJ device. The repository's existing `pytest.ini` discovers only `tests/`; the additional NMR tests need explicit paths.

## Running a scientific stage

1. Record the Git commit and the SHA-256 of every source, request and external input.
2. Prepare a new request with explicit input references, device, output directory and resource limits.
3. Obtain independent input and stage reviews for that exact request and job.
4. Dry-run the source-only transfer to JJ, then transfer the reviewed files. Keep datasets, checkpoints and outputs in their existing private locations.
5. Run the stage under its supervisor once. Keep the actual inner exit, external exit, RSS observations and output hashes.
6. Review the actual results independently before interpreting them or proposing a change.

The fixed research preselection is `fixedcenter_mixed_listwise_single`. Test results do not select a replacement, and these tools do not promote or deploy a system.

## Saved-tensor diagnosis

The diagnostic tools use the existing fixed query manifest, 512-candidate natural pools and separately saved 513-candidate fit-loss traces. The manifest SHA and closed evidence references are recorded in `source_provenance.json`; the manifest records themselves remain private.

The original NN stage's supervision failure and RSS gap remain in the evidence chain. The raw closure validates provenance and bytes. The Mac and JJ probes identified a backend reconstruction difference for one fixed saved input: the Mac ARM `tanh` result differed, while the JJ CPU/CUDA delta Jacobian and weighted chains reproduced the saved values bit for bit. The original tolerance is `rtol=2e-5`, `atol=2e-6`.

`analyze_diagnostics_closed_outputs_nullable_rank.py` is the archived analyzer whose local reconstruction stopped at that guard. It is retained because the probes and tests reference its exact definitions. Its presence is not a completed scientific aggregation. A corrected backend-context analyzer must be reviewed under a new source hash before execution. The saved network VJP and model-state assertions retain their recorded source-bound scope.

## Publication boundary

This import contains source, tests, package constraints, protocol constants and source/evidence hashes. Dataset records, candidate/query manifests, spectra, checkpoints, tensors, predictions, metric tables, reports, logs, credentials and existing paper edits are excluded from the commit. Scientific artifacts stay in their original local/JJ stores. Keep a private run receipt linking each future execution to its Git commit, source hashes, input hashes and actual outputs.

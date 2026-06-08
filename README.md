# SpectraExpert: A Retrieval-Transition-Reasoning Framework for Molecular Identification from Raman Spectra

**[Paper]** · **[Knowledge Base](knowledge_base/)** · **[Benchmark Results](results/BENCHMARK_REPORT.md)**

---

## Overview

SpectraExpert (Fingerprint Reasoner) identifies molecular structures from Raman spectra by decomposing the problem into three sequential stages that mirror the workflow of a human spectroscopist:

1. **Contrastive Retrieval** — A SpectrumTransformer encoder (SimCLR, 5M parameters) trained on 129,817 QM9S spectra retrieves the top-*K* structurally related candidates via cosine similarity in a learned embedding space.
2. **Spectral Analysis** — Detected peaks are matched against 181 empirical group-frequency rules digitized from Nakamoto's authoritative textbook, producing a structured peak–rule annotation.
3. **LLM-Guided Reasoning** — A frontier LLM performs 4-step chain-of-thought reasoning over the candidates and Nakamoto rule annotations, then issues a structured decision: **SELECT** a candidate, **MODIFY** it, or declare **UNCERTAIN**.

![Pipeline overview](paper/fig/pipeline.pdf)

---

## Repository Structure

```
spectraexpert/          # Core Python package
  train_contrastive.py  # Stage 1: SimCLR encoder training
  run_pipeline.py       # CLI entry point for the full pipeline
  data/                 # Dataset loading (QM9S, ChEMBL)
  models/               # SpectrumTransformer architecture
  utils/                # Augmentations, NT-Xent loss
  pipeline/             # 3-stage pipeline
    stage1_retrieval.py
    stage2_peak_analysis.py
    stage3_llm_reasoning.py
    knowledge/          # Nakamoto rule loader & peak matcher
    llm/                # LLM client backends (Anthropic, OpenAI, dummy)
    prompts/            # Stage 3 prompt templates
  eval/                 # Evaluation harness (all 5 modes)
scripts/                # Data preparation & figure generation
knowledge_base/         # 181 Nakamoto group-frequency rules (JSON + Markdown)
paper/fig/              # Paper figures
```

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Run the full pipeline on a single spectrum

```bash
# Use a QM9S spectrum by index (requires QM9S data)
python spectraexpert/run_pipeline.py --index 42 --provider openai --model gpt-5

# Use Anthropic
python spectraexpert/run_pipeline.py --index 42 --provider anthropic --model claude-sonnet-4-6

# Test without an API key (dummy LLM)
python spectraexpert/run_pipeline.py --index 42 --provider dummy --summary
```

### Train the contrastive encoder

```bash
python spectraexpert/train_contrastive.py \
    --data_dir data/qm9s \
    --batch_size 128 \
    --epochs 80
```

### Reproduce benchmark evaluation

```bash
# Encoder retrieval on full 211-sample ChEMBL test set
python spectraexpert/eval/evaluate.py --mode encoder --max-samples 211

# LLM Direct with Claude Sonnet 4.6 (50 hardest molecules)
python spectraexpert/eval/evaluate.py --mode llm-direct \
    --model claude-sonnet-4-6 --max-samples 50

# Full agentic pipeline with o3 (20 small molecules)
python spectraexpert/eval/evaluate.py --mode agentic \
    --model o3 --max-samples 20
```

---

## Benchmark Results

| Method | GED (mean) | Tanimoto | Parsed | N |
|--------|-----------|---------|--------|---|
| Encoder best@10 | 16.93 | 0.058 | 100% | 211 |
| KNN best@10 | 17.04 | 0.056 | 100% | 211 |
| LLM+KB O3 | **13.52** | **0.075** | 84% | 50 |
| Agentic Claude | 16.16 | 0.045 | 100% | 20 |
| Agentic DeepSeek | 15.73 | 0.063 | 85% | 20 |

Key finding: the agentic pipeline achieves **best-case GED 8.0** on small molecules, while Claude Sonnet 4.6 maintains **100% valid-SMILES output** across all evaluation modes.

---

## Knowledge Base

[`knowledge_base/`](knowledge_base/) contains 181 empirical group-frequency rules digitized from:

> Nakamoto, K. (2008). *Infrared and Raman Spectra of Inorganic and Coordination Compounds*. John Wiley & Sons.

Each rule includes a unique ID (e.g., `GFQ-042`), frequency range, confidence level, and functional-group description. The LLM cites these IDs explicitly in its chain-of-thought reasoning, making every prediction auditable.

---

## Data

- **QM9S** (training): 129,817 computed Raman spectra, molecules ≤9 heavy atoms. Download from [figshare](https://doi.org/10.6084/m9.figshare.24235333.v3).
- **ChEMBL Raman** (evaluation): held-out test set of ~211 pharmaceutically relevant molecules. From [Liang et al. 2025, *Scientific Data*](https://doi.org/10.1038/s41597-025-04184-1).

Pre-processed test set: `data/processed/chembl_test_set.parquet` (not included in this repo due to size; see `scripts/build_benchmark.py` to reproduce).

---

## LLM API Setup

Set your API key before running any LLM-based evaluation:

```bash
# OpenAI-compatible endpoint (GPT-5, o3, DeepSeek, Gemini via proxy)
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://your-proxy/v1   # optional

# Anthropic (Claude)
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Citation

```bibtex
@inproceedings{luan2026spectraexpert,
  title     = {SpectraExpert: A Retrieval-Transition-Reasoning Framework for
               Molecular Identification from Raman Spectra},
  author    = {Luan, Denghong and Lu, Zaifan},
  booktitle = {IEEE Conference},
  year      = {2026}
}
```

---

## License

See [LICENSE](LICENSE).

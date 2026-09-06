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

### 结构感知 Stage 1：从数据准备到统一评测

新版 Stage 1 在训练时对齐 Raman 光谱与二维分子图，在查询时仍然只接收光谱。下面的命令沿用
同一个 `split_manifest.json`，避免不同消融实验因重新划分数据而失去可比性。数据准备入口只接受
项目已授权的两份 Raman JSONL；路径白名单与来源 SHA-256 会写入输出元数据。

```bash
python scripts/prepare_raman_jsonl.py \
    --input /data/SpecGPT/data/theory_data_tiny/Raman_QME14S.jsonl \
    --input /data/SpecGPT/data/theory_data_tiny/Raman_fg26.jsonl \
    --output-dir data/raman_tiny \
    --split-seed 0

python -m spectraexpert.train_structure_aware \
    --training-mode spectrum_graph_v2 \
    --data-dir data/raman_tiny \
    --output-dir outputs/structure_v2 \
    --split-manifest data/raman_tiny/split_manifest.json \
    --epochs 50 --patience 8 --batch-size 32 \
    --embed-dim 128 --depth 3 --num-heads 4 \
    --projection-dim 64 --graph-hidden-dim 128 \
    --lambda-aug 0.25 --lambda-struct 0.5 \
    --lambda-aux 0.1 --lambda-fingerprint 0.0
```

候选图索引只需要候选 SMILES。先按 checkpoint 使用的固定 Morgan/图特征 schema 缓存结构特征，
再建立带 checkpoint SHA-256 与 schema metadata 的 graph index：

```bash
python scripts/cache_molecular_features.py \
    --smiles-path data/raman_tiny/smiles.parquet \
    --output outputs/structure_v2/candidate_features.parquet

python scripts/build_graph_index.py \
    --checkpoint outputs/structure_v2/best.pt \
    --smiles-path data/raman_tiny/smiles.parquet \
    --feature-cache outputs/structure_v2/candidate_features.parquet \
    --output outputs/structure_v2/graph_index.npz
```

统一评测报告同时保留历史复现用的 `legacy_ged_v1`、主指标 `chemical_ged_v2` 与
`chemical_nged_v2`。identity-included 用于检查同域精确检索能力；identity-excluded 则衡量排除真实
分子后候选的结构接近程度。两种设置都在 ranking 完成后才读取 ground-truth SMILES，模型输入审计字段
固定为 `query_inputs=["spectrum"]`。

```bash
python -m spectraexpert.eval.evaluate_structure_retrieval \
    --mode spectrum_graph_v2 \
    --query-data-dir data/raman_tiny \
    --candidate-index outputs/structure_v2/graph_index.npz \
    --checkpoint outputs/structure_v2/best.pt \
    --top-k 10 \
    --output outputs/structure_v2/eval_identity_included.json

python -m spectraexpert.eval.evaluate_structure_retrieval \
    --mode spectrum_graph_v2 \
    --query-data-dir data/raman_tiny \
    --candidate-index outputs/structure_v2/graph_index.npz \
    --checkpoint outputs/structure_v2/best.pt \
    --top-k 10 --exclude-identity \
    --baseline-result outputs/spectral_only_v1/eval_identity_excluded.json \
    --output outputs/structure_v2/eval_identity_excluded.json
```

固定消融 launcher 可用 `--dry-run` 审计完整的 A–D、seed 0/1/2 共十二条命令，再在确认数据和 GPU
资源后去掉该参数执行：

```bash
bash scripts/run_tiny_pilot.sh \
    --data-dir data/raman_tiny \
    --output-root outputs/tiny_pilot \
    --gpu 0 \
    --dry-run
```

**tiny pilot 仅用于验证数据、训练、索引与评测闭环是否可运行。** 其样本规模、候选覆盖与随机种子
不足以支持论文性能结论；最终结论必须来自冻结测试集、相同候选库、完整消融和预先规定的 paired
bootstrap 验收。

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

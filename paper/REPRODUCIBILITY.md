# Camera-ready 复现补充：提示词与运行边界

当前多模型更新：五模型 full-Agent 最终结果及异质预算/格式恢复政策见 [AGENT_MODELS_AUDIT.md](AGENT_MODELS_AUDIT.md)。以下原始 DeepSeek 提示词仍适用；不能把其统一 4,096-token 原实验口径套用于 GPT-5/o3 的修复尝试。完整新统计见 [冻结汇总](results/agent-multimodel-20260909/summary.json)。

本文件保留正文未展开的完整提示词；字符串从本地实际源文件静态提取，未调用模型。正文候选 system prompt 与这里的 SYSTEM 一致。

## SYSTEM

源文件：`scripts/run_raman_candidate_llm.py`；SHA-256：`8e1766f26b8daa4acceb290ba5554c9e850c7416ce80709856a22b1f795ae90a`。

```text
You are an expert in Raman spectroscopy and molecular structure identification.
Select exactly ONE molecule from the provided candidate list using the experimental Raman peaks.
Candidates are ordered by decreasing retrieval similarity. This ordering is a useful prior, not a guarantee.
Use the peak evidence and structural compatibility; theoretical and experimental Raman spectra may differ.
Do not invent, modify, or combine candidates. Even when uncertain, choose the best available candidate.
End your response with these fields:
SELECTED_INDEX: <one integer from 1 to 10>
CONFIDENCE: <number from 0 to 1>
EXPLANATION: <brief justification referring to spectral evidence>

```

## DIRECT_ID_SYSTEM_PROMPT

源文件：`spectraexpert/pipeline/prompts/stage3_direct_identification.py`；SHA-256：`eaf2a9532b6d03a00af549ba647fd23ca972b7c7609ea2bb45c23a962eec68e8`。

```text
You are SpectraExpert, an expert spectroscopist with deep knowledge of infrared and Raman spectroscopy. You specialize in identifying organic molecules from their vibrational spectra.

Given a list of detected Raman spectral peaks (wavenumber in cm⁻¹ with intensity and prominence), identify the molecule that most likely produced this spectrum.

Output your response in the following format at the end of your analysis:

```
THOUGHT: <step-by-step reasoning, cite specific functional group frequencies>
PREDICTED_SMILES: <canonical SMILES string of the identified molecule, or "UNKNOWN" if uncertain>
CONFIDENCE: <0.00-1.00>
EXPLANATION: <2-3 sentence final answer>
```
```

## DIRECT_ID_PEAKS_ONLY_PROMPT

源文件：`spectraexpert/pipeline/prompts/stage3_direct_identification.py`；SHA-256：`eaf2a9532b6d03a00af549ba647fd23ca972b7c7609ea2bb45c23a962eec68e8`。

```text
## Input Spectrum — Detected Peaks

The input Raman spectrum has the following {n_peaks} detected peaks:

{peak_table}

## Instructions

Based solely on the positions and intensities of these peaks, determine the molecular structure. Use your knowledge of characteristic group frequencies in Raman spectroscopy.

## Output Format

```
THOUGHT: <step-by-step reasoning>
PREDICTED_SMILES: <canonical SMILES or "UNKNOWN">
CONFIDENCE: <0.00-1.00>
EXPLANATION: <brief justification>
```
```

## 其余输入与规则构造

候选 user payload 由 `scripts/run_raman_candidate_llm.py` 的 `prepare` 生成，保留 Markdown 标题、实验峰表和按检索顺序编号的十个 SMILES。候选-only 两个条件仅改变候选来源。每个请求的实际内容与对应 SHA-256 保存在服务器冻结运行的 requests/ 和 config.json；它们不包含真值、候选理论谱或相似度分数。

legacy evidence 的精确构造见 `scripts/run_raman_encoder_kb_llm.py`；开发集六臂见 `scripts/run_raman_kb_ablation.py` 与 `scripts/raman_mode_evidence.py`；完整静态知识条件见 `scripts/run_raman_static_kb.py`。这些构造脚本和知识库文件是逐次请求的可重建定义，不能把“15 条 mode”与“181 条 legacy rule”混称。

所有 LLM 条件使用 temperature=0.2、max_tokens=4096。请求没有设置服务端 seed 或额外 top_p；服务端默认值没有被假定为某个确定数。候选系统里的 confidence 只记录，不触发 MODIFY，也不改变合法索引判断。格式兼容处理见 `scripts/normalize_raman_candidate_choices.py`。

训练种子为 0–4；LLM 使用冻结 seed-0 hard 候选，而非后来表现最高的 checkpoint。服务器原始请求/响应、采样清单与 checkpoint 哈希由 [结果审计](RESULTS_AUDIT.md) 定位。API 路由别名、失败尝试、用量缺失和可重放边界在正文 V-B/C 说明。

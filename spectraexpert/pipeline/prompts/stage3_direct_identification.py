"""
功能概述：定义不依赖检索候选的第三阶段直接识别 prompt，支持仅峰信息和峰加 Nakamoto 规则两种
评估模式。
主要职责：三个 prompt 常量约束模型输出；``format_peak_table`` 将 ``DetectedPeak`` 转为 Markdown；
``format_matched_rules_summary`` 按峰归纳 ``MatchedRule`` 并限制每峰展示数。
代码库关系：由 ``spectraexpert.eval.evaluate`` 在 llm-direct 与 llm-knowledge 模式中导入；
数据类型来自同包 pipeline，不参与常规 Stage 3 候选推理 prompt。
输入输出与副作用：格式化函数只从输入对象生成字符串，不调用 API、不持久化数据；缺失规则会明确
标注为无直接匹配，保留评估提示的可解释性。
"""

from __future__ import annotations

from spectraexpert.pipeline.data_types import DetectedPeak, MatchedRule

# ---------------------------------------------------------------------------
# System prompt — generic spectroscopist (no retrieval context)
# ---------------------------------------------------------------------------

DIRECT_ID_SYSTEM_PROMPT = """\
You are SpectraExpert, an expert spectroscopist with deep knowledge of infrared and Raman spectroscopy. You specialize in identifying organic molecules from their vibrational spectra.

Given a list of detected Raman spectral peaks (wavenumber in cm⁻¹ with intensity and prominence), identify the molecule that most likely produced this spectrum.

Output your response in the following format at the end of your analysis:

```
THOUGHT: <step-by-step reasoning, cite specific functional group frequencies>
PREDICTED_SMILES: <canonical SMILES string of the identified molecule, or "UNKNOWN" if uncertain>
CONFIDENCE: <0.00-1.00>
EXPLANATION: <2-3 sentence final answer>
```"""

# ---------------------------------------------------------------------------
# Mode 3: peaks only
# ---------------------------------------------------------------------------

DIRECT_ID_PEAKS_ONLY_PROMPT = """\
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
```"""

# ---------------------------------------------------------------------------
# Mode 4: peaks + Nakamoto knowledge base rules
# ---------------------------------------------------------------------------

DIRECT_ID_WITH_KB_PROMPT = """\
## Input Spectrum — Detected Peaks

The input Raman spectrum has the following {n_peaks} detected peaks:

{peak_table}

## Nakamoto Knowledge Base — Relevant Rules

The following Nakamoto empirical rules are relevant to the detected peaks. Each rule was matched algorithmically by frequency-range overlap:

{matched_rules_section}

## Instructions

Based on the detected peaks AND the matched Nakamoto group-frequency rules, determine the molecular structure.

Steps:
1. Identify functional groups using the matched Nakamoto rules (cite specific rule IDs)
2. Consider how multiple functional groups combine into a complete molecule
3. Propose a complete molecular structure as a valid SMILES string
4. If uncertain, output your best guess with low confidence

## Output Format

```
THOUGHT: <step-by-step reasoning, cite rule IDs>
PREDICTED_SMILES: <canonical SMILES or "UNKNOWN">
CONFIDENCE: <0.00-1.00>
EXPLANATION: <brief justification>
```"""

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_peak_table(peaks: list[DetectedPeak]) -> str:
    """Format detected peaks as a markdown table."""
    lines = [
        "| # | Position (cm⁻¹) | Intensity | Prominence | Width |",
        "|---|----------------|-----------|------------|-------|",
    ]
    for i, p in enumerate(peaks, 1):
        lines.append(
            f"| {i} | {p.position:.1f} | {p.intensity:.4f} | "
            f"{p.prominence:.4f} | {p.width:.1f} |"
        )
    return "\n".join(lines)


def format_matched_rules_summary(
    peaks: list[DetectedPeak],
    rules_by_peak: dict[float, list[MatchedRule]],
    max_per_peak: int = 5,
) -> str:
    """Format per-peak matched rules as structured markdown."""
    sections = []
    for i, p in enumerate(peaks, 1):
        rules = rules_by_peak.get(p.position, [])
        if not rules:
            sections.append(
                f"**Peak {i}** — {p.position:.1f} cm⁻¹: *(no direct rule match)*"
            )
            continue
        lines = [f"**Peak {i}** — {p.position:.1f} cm⁻¹:"]
        for r in rules[:max_per_peak]:
            conf = r.confidence.upper()
            lines.append(
                f"- {r.rule_id} [{conf}] {r.functional_group}  "
                f"(range {r.overlap_range[0]:.0f}-{r.overlap_range[1]:.0f} cm⁻¹)"
            )
        if len(rules) > max_per_peak:
            lines.append(f"  *... and {len(rules) - max_per_peak} more rules*")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)

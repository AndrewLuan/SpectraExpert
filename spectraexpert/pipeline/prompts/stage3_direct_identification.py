"""
Direct molecule identification prompts for Stage 3 evaluation (no retrieval).

Two variants:
- Mode 3 (llm-direct): only detected peaks → identify molecule
- Mode 4 (llm-knowledge): detected peaks + Nakamoto rules → identify molecule
"""

from __future__ import annotations

from proj.spectra_expert.pipeline.data_types import DetectedPeak, MatchedRule

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

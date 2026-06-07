"""
Main CoT reasoning prompt template for Stage 3 of the SpectraExpert pipeline.

This is the core prompt sent to the LLM.  It includes:
- Detected peak table
- Candidate molecules with SMILES and similarity
- Matched Nakamoto rules per peak
- 4-step reasoning instructions
- Structured output format
"""

STAGE3_REASONING_PROMPT = """\

## Input Spectrum Analysis

The input Raman spectrum has **{num_peaks} detected peaks**:

{peak_table}

## Candidate Molecules

**{num_candidates}** candidate molecules were retrieved from the spectral database by contrastive embedding search:

{candidate_table}

## Nakamoto Knowledge Base — Relevant Rules

The following Nakamoto empirical rules are relevant to the detected peaks. Each rule was matched algorithmically by frequency-range overlap:

{matched_rules_section}

---

## Instructions

Perform a detailed spectroscopic analysis following these steps:

### Step 1: Functional Group Identification
For each detected peak, identify which functional groups are indicated by Nakamoto rules. Cite specific rule IDs (e.g. GFQ-042). Note which peaks have definitive vs. tentative assignments.

### Step 2: Candidate Evaluation
For each candidate molecule (up to {max_candidates} candidates provided):
- Based on the candidate's SMILES string, what functional groups does it contain?
- Which detected peaks does this candidate explain well?
- Are there strong peaks that this candidate CANNOT explain (unexplained peaks)?
- What is your confidence (0.0-1.0) that this candidate matches the observed spectrum?

### Step 3: Consensus Assessment
- Is there a clear winner among candidates (highest score, fewest unexplained peaks)?
- If the top candidate's confidence is much higher than the runner-up, flag this as a strong match.
- If multiple candidates have similar scores, note the ambiguity.

### Step 4: Decision
Choose ONE of the following:
a) **SELECT** the best-matching candidate (if one candidate clearly fits)
b) **MODIFY** the closest candidate by changing its chemical structure to better match the spectrum (if no candidate is satisfactory but a close match exists)
c) **UNCERTAIN** (if no candidate is close and structural modification is not feasible)

### Output Format

Your response MUST end with these structured fields on separate lines at the very end of your response, after your complete chain-of-thought reasoning:

```
CANDIDATE_ANALYSIS:
[For each candidate evaluated, 1-2 sentences of analysis]

DECISION: SELECT | MODIFY | UNCERTAIN
SELECTED_INDEX: <rank number 1-{max_candidates}, or "null">
SELECTED_SMILES: <canonical SMILES of selected candidate, or "null">
MODIFIED_SMILES: <proposed new SMILES string, or "null">
CONFIDENCE: <0.00-1.00>
EXPLANATION: <2-3 sentence final answer identifying the molecule>
```
"""


def format_peak_table(peaks: list) -> str:
    """Format a list of DetectedPeak objects as a markdown table."""
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


def format_candidate_table(candidates: list) -> str:
    """Format a list of Candidate objects as a markdown table."""
    lines = [
        "| Rank | Similarity | SMILES |",
        "|------|------------|-------|",
    ]
    for c in candidates:
        lines.append(f"| {c.rank} | {c.similarity:.4f} | `{c.smiles}` |")
    return "\n".join(lines)


def format_matched_rules(
    peaks: list,
    rules_by_peak: dict[float, list],
    max_per_peak: int = 5,
) -> str:
    """Format per-peak matched rules as structured markdown."""
    sections = []
    for i, p in enumerate(peaks, 1):
        rules = rules_by_peak.get(p.position, [])
        if not rules:
            sections.append(f"**Peak {i}** — {p.position:.1f} cm⁻¹: *(no direct rule match)*")
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

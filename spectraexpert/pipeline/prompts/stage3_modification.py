"""
Structural modification prompt template.

Used when the main reasoning stage yields low confidence — asks the LLM to
propose a chemical modification to the closest candidate's SMILES.
"""

STAGE3_MODIFICATION_PROMPT = """\

## Structural Modification Request

The initial analysis could not confidently identify a matching molecule from the database. The closest candidate was:

**Rank {best_rank}**: `{best_smiles}`  (cosine similarity: {best_similarity:.4f})

### Spectral Evidence — Strongly Indicated Functional Groups

The following functional groups are STRONGLY indicated by the detected peaks (definitive/high-confidence Nakamoto rules):

{strong_peak_evidence}

### Spectral Evidence — Moderately Indicated Functional Groups

The following functional groups are MODERATELY indicated:

{moderate_peak_evidence}

### Unexplained Peaks

These detected peaks are NOT explained by the closest candidate:

{unexplained_peaks}

### Task

Starting from the closest candidate's SMILES string (`{best_smiles}`), propose chemical modifications that would make the molecule better match the observed spectrum.

Consider:
1. Adding or removing functional groups indicated by unmatched peaks
2. Changing bond orders (single ↔ double ↔ triple) based on frequency shifts
3. Modifying substituent positions
4. Adding/removing rings or heteroatoms

### Output Format

```
MODIFIED_SMILES: <proposed SMILES string>
MODIFICATION_RATIONALE: <brief explanation of the changes and why they improve the spectral match>
EXPECTED_SPECTRAL_CHANGES: <how the modification would affect the vibrational spectrum>
CONFIDENCE: <0.00-1.00>
```
"""  # noqa: E501


def format_peak_evidence(
    matched_rules: list,
    label: str = "Strongly Indicated",
    min_confidence_rank: int = 3,
) -> str:
    """Format matched rules grouped by functional group, filtering by confidence.

    Parameters
    ----------
    matched_rules : list of MatchedRule objects.
    label : section heading.
    min_confidence_rank : minimum rank (5=definitive, 3=high) to include.

    Returns
    -------
    str — formatted markdown block.
    """
    confidence_rank = {
        "definitive": 5,
        "experimentally_verified": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }
    filtered = [
        m for m in matched_rules
        if confidence_rank.get(m.confidence.lower(), 0) >= min_confidence_rank
    ]
    if not filtered:
        return f"*No {label.lower()} functional group matches found.*"

    # Deduplicate by functional group
    seen: set[str] = set()
    lines = []
    for m in filtered:
        if m.functional_group not in seen:
            seen.add(m.functional_group)
            lines.append(
                f"- **{m.functional_group}**  "
                f"(peak {m.matched_peak:.1f} cm⁻¹, {m.rule_id}, {m.confidence})"
            )
    return "\n".join(lines)


def format_unexplained_peaks(
    detected_peaks: list,
    rules_by_peak: dict[float, list],
) -> str:
    """Return peaks that have NO matched rules."""
    unexplained = [
        p for p in detected_peaks
        if p.position not in rules_by_peak or not rules_by_peak[p.position]
    ]
    if not unexplained:
        return "*All detected peaks have at least one matching rule.*"
    lines = []
    for p in unexplained:
        lines.append(f"- {p.position:.1f} cm⁻¹ (intensity={p.intensity:.4f})")
    return "\n".join(lines)

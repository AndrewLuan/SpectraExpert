"""
Stage 2: Peak Analysis — match detected peaks to Nakamoto empirical rules.

This stage is purely algorithmic (no LLM call).  It:
1. Uses the PeakMatcher to find Nakamoto group-frequency rules matching each peak
2. Produces a formatted "knowledge summary" for the Stage 3 LLM prompt
3. Tracks which peaks are explained by rules and which are not
"""

from __future__ import annotations

from proj.spectra_expert.pipeline.data_types import DetectedPeak, MatchedRule, Stage1Result, Stage2Result
from .knowledge.nakamoto_loader import NakamotoRuleBase
from .knowledge.peak_matcher import PeakMatcher


class Stage2PeakAnalysis:
    """Stage 2: match spectral peaks to Nakamoto empirical rules.

    Parameters
    ----------
    config : dict
        Pipeline configuration.  Relevant keys:

        - ``stage2.knowledge.rules_path`` — path to the Nakamoto JSON.
        - ``stage2.knowledge.matching_tolerance`` — cm⁻¹ tolerance.
        - ``stage2.knowledge.max_rules`` — max rules to include in prompt.
        - ``stage2.knowledge.prefer_categories`` — priority order.
    """

    def __init__(self, config: dict):
        kcfg = config.get("stage2", config).get("knowledge", config)

        rules_path = kcfg.get("rules_path", "")
        self._rule_base = NakamotoRuleBase(rules_path)

        tolerance = float(kcfg.get("matching_tolerance", 20.0))
        max_rules = int(kcfg.get("max_rules", 30))
        self._matcher = PeakMatcher(
            self._rule_base,
            tolerance=tolerance,
            max_rules=max_rules,
        )

        self._prefer_categories = kcfg.get(
            "prefer_categories",
            [
                "group_frequencies",
                "geometry_determination",
                "substituent_effects",
                "charge_and_oxidation_effects",
            ],
        )

    # ── public API ─────────────────────────────────────────────────

    def run(self, stage1_result: Stage1Result) -> Stage2Result:
        """Run Stage 2 analysis on the Stage 1 output.

        Parameters
        ----------
        stage1_result : Stage1Result from the retrieval stage.

        Returns
        -------
        Stage2Result with matched rules and formatted knowledge summary.
        """
        peaks = stage1_result.detected_peaks

        # ── match all peaks to group-frequency rules ───────────────
        flat_rules, rules_by_peak = self._matcher.match_peaks_batch(peaks)

        # ── also fetch rules from other preferred categories ───────
        for cat in self._prefer_categories:
            if cat == "group_frequencies":
                continue  # already matched above
            for rule in self._rule_base.get_rules_by_category(cat):
                # Check if any peak matches this rule
                for peak in peaks:
                    ranges = self._rule_base._parse_frequency_ranges(
                        rule.get("spectral_details", "") + " " + rule.get("rule", "")
                    )
                    if not ranges:
                        # Rules without explicit frequency ranges still get included
                        # as general knowledge context
                        flat_rules.append(self._matcher._to_matched(
                            rule, peak, (0, 0)
                        ))
                        break

        # ── compute coverage ───────────────────────────────────────
        n_peaks = len(peaks)
        n_explained = sum(
            1 for p in peaks if p.position in rules_by_peak and rules_by_peak[p.position]
        )
        coverage = n_explained / n_peaks if n_peaks > 0 else 0.0

        # ── format knowledge summary ───────────────────────────────
        summary = self._format_knowledge_summary(peaks, rules_by_peak, flat_rules)

        return Stage2Result(
            detected_peaks=peaks,
            matched_rules=flat_rules,
            rules_by_peak=rules_by_peak,
            relevant_rules_summary=summary,
            all_peaks_explained=coverage,
        )

    # ── formatting ─────────────────────────────────────────────────

    def _format_knowledge_summary(
        self,
        peaks: list[DetectedPeak],
        rules_by_peak: dict[float, list[MatchedRule]],
        all_rules: list[MatchedRule],
    ) -> str:
        """Build a structured text summary for the LLM prompt context."""
        lines = []
        lines.append(f"## Detected Peaks ({len(peaks)} total)")
        lines.append("")
        for i, p in enumerate(peaks, 1):
            lines.append(f"  {i:>2}. {p.position:>7.1f} cm⁻¹  "
                         f"(intensity={p.intensity:.4f}, "
                         f"prominence={p.prominence:.4f})")
        lines.append("")

        lines.append(f"## Matched Nakamoto Rules ({len(all_rules)} unique rules)")
        lines.append("")
        for p in peaks:
            matched = rules_by_peak.get(p.position, [])
            if matched:
                lines.append(f"**Peak {p.position:.1f} cm⁻¹** — {len(matched)} rules:")
                for r in matched[:5]:
                    lines.append(f"  - {r.rule_id} [{r.confidence.upper()}] "
                                 f"{r.functional_group}  "
                                 f"(range {r.overlap_range[0]:.0f}–"
                                 f"{r.overlap_range[1]:.0f} cm⁻¹)")
                if len(matched) > 5:
                    lines.append(f"    ... and {len(matched) - 5} more rules")
            else:
                lines.append(f"**Peak {p.position:.1f} cm⁻¹** — no direct rule match")
            lines.append("")

        # Add full rule details as reference
        lines.append("## Full Rule Details (for reference)")
        lines.append("")
        seen_ids: set[str] = set()
        for r in all_rules:
            if r.rule_id not in seen_ids:
                seen_ids.add(r.rule_id)
                lines.append(
                    f"- **{r.rule_id}** [{r.confidence.upper()}] "
                    f"{r.rule_text}"
                )

        return "\n".join(lines)

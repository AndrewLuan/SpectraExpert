"""
Peak-to-rule matcher — matches detected spectral peaks to Nakamoto
empirical rules based on frequency-range overlap.
"""

from __future__ import annotations

from typing import Any

from .nakamoto_loader import NakamotoRuleBase
from proj.spectra_expert.pipeline.data_types import DetectedPeak, MatchedRule

# ── confidence ordering (higher = more reliable) ────────────────────────

_CONFIDENCE_ORDER = {
    "definitive": 5,
    "experimentally_verified": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


def _confidence_rank(conf: str) -> int:
    return _CONFIDENCE_ORDER.get(conf.lower(), 0)


class PeakMatcher:
    """Matches detected spectral peaks to Nakamoto empirical rules.

    Parameters
    ----------
    rule_base : loaded NakamotoRuleBase instance.
    tolerance : cm⁻¹ tolerance when checking frequency-range overlap.
    max_rules : maximum number of distinct rules to return per query.
    """

    def __init__(
        self,
        rule_base: NakamotoRuleBase,
        tolerance: float = 20.0,
        max_rules: int = 30,
    ):
        self._rule_base = rule_base
        self._tolerance = tolerance
        self._max_rules = max_rules

    # ── public API ──────────────────────────────────────────────────────

    def match_peak(
        self, peak: DetectedPeak, rules: list[dict] | None = None
    ) -> list[MatchedRule]:
        """Find all rules whose frequency ranges overlap *peak.position*.

        Parameters
        ----------
        peak : the detected peak to match.
        rules : optional subset of rules to search (default: all group_freq).

        Returns
        -------
        list[MatchedRule] — sorted by confidence (highest first).
        """
        search_rules = rules if rules is not None else \
            self._rule_base.get_rules_by_category("group_frequencies")

        matched: list[MatchedRule] = []
        for rule in search_rules:
            ranges = self._rule_base._parse_frequency_ranges(
                rule.get("spectral_details", "") + " " + rule.get("rule", "")
            )
            for lo, hi in ranges:
                if (lo - self._tolerance) <= peak.position <= (hi + self._tolerance):
                    matched.append(self._to_matched(rule, peak, (lo, hi)))
                    break  # one match per rule is enough

        matched.sort(key=lambda m: _confidence_rank(m.confidence), reverse=True)
        return matched[:self._max_rules]

    def match_peaks_batch(
        self,
        peaks: list[DetectedPeak],
        rules: list[dict] | None = None,
    ) -> tuple[list[MatchedRule], dict[float, list[MatchedRule]]]:
        """Batch version — match all peaks against rules.

        Returns
        -------
        flat : deduplicated flat list of MatchedRule objects.
        by_peak : dict mapping peak position → list[MatchedRule].
        """
        search_rules = rules if rules is not None else \
            self._rule_base.get_rules_by_category("group_frequencies")

        flat: list[MatchedRule] = []
        by_peak: dict[float, list[MatchedRule]] = {}
        seen_ids: set[str] = set()

        for peak in peaks:
            peak_matches = self.match_peak(peak, search_rules)
            if peak_matches:
                by_peak[peak.position] = peak_matches
                for m in peak_matches:
                    if m.rule_id not in seen_ids:
                        flat.append(m)
                        seen_ids.add(m.rule_id)

        return flat, by_peak

    def infer_functional_group(self, rule: dict) -> str:
        """Try to extract a short functional-group label from a rule dict.

        Falls back to the rule category if no explicit group is found.
        """
        text = rule.get("spectral_details", "") + " " + rule.get("rule", "")
        # Common patterns:  "O-H stretching", "C=O stretch", "C-H bend", ...
        # Look for a line that starts the entry in the GFQ table
        for line in text.split("\n"):
            line = line.strip()
            # e.g. "O-H stretch", "C=O carbonates", "C-H bending"
            if line and not line.startswith("*") and not line.startswith("ν"):
                # Take the first 60 chars max
                candidate = line.strip("| ,;").strip()[:60]
                if len(candidate) > 3:
                    return candidate
        return rule.get("category", "unknown")

    # ── private ─────────────────────────────────────────────────────────

    def _to_matched(
        self,
        rule: dict,
        peak: DetectedPeak,
        overlap_range: tuple[float, float],
    ) -> MatchedRule:
        return MatchedRule(
            rule_id=rule["id"],
            rule_text=rule["rule"],
            category=rule["category"],
            matched_peak=peak.position,
            confidence=rule.get("confidence", "high"),
            functional_group=self.infer_functional_group(rule),
            overlap_range=overlap_range,
        )

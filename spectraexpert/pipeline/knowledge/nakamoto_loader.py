"""
Nakamoto Empirical Rules knowledge base loader.

Loads the 181 Nakamoto rules from the structured JSON dataset, indexes them
by category and (optionally) by frequency range for fast lookup.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class NakamotoRuleBase:
    """Loaded and indexed Nakamoto empirical rules.

    Provides search/query interfaces for rule lookup by category and
    frequency-range matching.

    Parameters
    ----------
    rules_path : path to ``nakamoto_empirical_rules_dataset.json``.
    """

    def __init__(self, rules_path: str | Path):
        self._rules_path = Path(rules_path)
        with open(self._rules_path, "r") as f:
            raw = json.load(f)

        self._meta: dict = raw.get("dataset_meta", {})
        self._rules: list[dict] = raw["rules"]

        # -- index by category --
        self._by_category: dict[str, list[dict]] = {}
        for r in self._rules:
            cat = r["category"]
            self._by_category.setdefault(cat, []).append(r)

        # -- index rule IDs for quick validation --
        self._known_ids: set[str] = {r["id"] for r in self._rules}

        # -- pre-parse frequency ranges for group_frequencies rules --
        self._freq_index: list[tuple[tuple[float, float], dict]] = []
        for r in self.get_rules_by_category("group_frequencies"):
            ranges = self._parse_frequency_ranges(r.get("spectral_details", "")
                                                  + " " + r.get("rule", ""))
            for lo, hi in ranges:
                self._freq_index.append(((lo, hi), r))

    # -- public API -------------------------------------------------------

    @property
    def total_rules(self) -> int:
        return len(self._rules)

    @property
    def categories(self) -> list[str]:
        return list(self._by_category.keys())

    @property
    def known_ids(self) -> set[str]:
        return self._known_ids

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    def get_all_rules(self) -> list[dict]:
        return list(self._rules)

    def get_rules_by_category(self, category: str) -> list[dict]:
        """Return all rules in a given category."""
        return list(self._by_category.get(category, []))

    def get_rule_by_id(self, rule_id: str) -> dict | None:
        """Return a single rule by its ID (e.g. 'GFQ-042'), or None."""
        for r in self._rules:
            if r["id"] == rule_id:
                return dict(r)
        return None

    def get_rules_by_frequency(
        self, freq_cm: float, tolerance: float = 20.0
    ) -> list[dict]:
        """Return all group-frequency rules whose range overlaps *freq_cm*.

        Overlap is tested as::

            lo - tolerance <= freq_cm <= hi + tolerance
        """
        matched: list[dict] = []
        seen_ids: set[str] = set()
        for (lo, hi), rule in self._freq_index:
            if (lo - tolerance) <= freq_cm <= (hi + tolerance):
                if rule["id"] not in seen_ids:
                    matched.append(rule)
                    seen_ids.add(rule["id"])
        return matched

    def format_for_prompt(self, rules: list[dict], max_rules: int = 30) -> str:
        """Format a list of rules as a markdown text block for LLM prompts.

        Parameters
        ----------
        rules : list of rule dicts.
        max_rules : maximum number to include (truncate if larger).

        Returns
        -------
        str — formatted text block.
        """
        lines = []
        for r in rules[:max_rules]:
            conf = r.get("confidence", "high").upper()
            details = r.get("spectral_details", "")
            note = r.get("notes", "")
            line = (
                f"- **{r['id']}** [{conf}] {r['rule']}  \n"
                f"  *Detail:* {details}"
            )
            if note:
                line += f"\n  *Note:* {note}"
            lines.append(line)
        return "\n\n".join(lines)

    def get_statistics(self) -> dict[str, int]:
        """Return per-category rule counts."""
        return {cat: len(rules) for cat, rules in self._by_category.items()}

    # -- private helpers --------------------------------------------------

    @staticmethod
    def _parse_frequency_ranges(text: str) -> list[tuple[float, float]]:
        """Extract (lo, hi) cm⁻¹ ranges from unstructured rule text.

        Handles patterns::

            ~3600-3700 cm
            3600-3700 cm
            ~1180 cm
            3336 cm⁻¹
        """
        ranges: list[tuple[float, float]] = []
        # Pattern 1: explicit ranges  ~3600-3700  or  3600-3700
        for match in re.finditer(
            r"(?:~|\b)(\d{3,5})\s*[-–to]+\s*(\d{3,5})\s*cm",
            text, re.IGNORECASE,
        ):
            lo, hi = float(match.group(1)), float(match.group(2))
            ranges.append((min(lo, hi), max(lo, hi)))

        # Pattern 2: single value  ~1180 cm  or  3336 cm⁻¹
        for match in re.finditer(
            r"(?:~|\b)(\d{3,5})\s*cm[⁻¹]?(?:\b|[,;.\s])",
            text,
        ):
            # avoid double-counting if this value was already part of a range
            val = float(match.group(1))
            already_covered = any(
                lo <= val <= hi for lo, hi in ranges
            )
            if not already_covered:
                ranges.append((val - 10, val + 10))  # ±10 cm window

        return ranges

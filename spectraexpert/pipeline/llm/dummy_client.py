"""
Dummy LLM client — returns deterministic responses for testing.

No API key or network access needed.
"""

from __future__ import annotations

from typing import Any

from .base import LLMClient


class DummyClient(LLMClient):
    """Deterministic mock LLM that returns configurable or canned text.

    Parameters
    ----------
    config : dict
        If ``config.get("dummy_response")`` is set, that string is returned.
        Otherwise a canned analysis is returned.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._response = config.get("dummy_response", None)

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        if self._response:
            return self._response

        # ── canned deterministic response for integration tests ──
        return (
            "CANDIDATE_ANALYSIS:\n"
            "Candidate #1 (C) has a single carbon atom with no functional groups. "
            "The detected peaks at 1350 and 1524 cm⁻¹ may correspond to C-C vibrations, "
            "but the match is weak. Candidate #2 (CC) is a simple alkane chain. "
            "No candidate provides a strong match to the observed spectrum.\n\n"
            "DECISION: MODIFY\n"
            "SELECTED_INDEX: null\n"
            "SELECTED_SMILES: null\n"
            "MODIFIED_SMILES: C1CCCCC1\n"
            "CONFIDENCE: 0.45\n"
            "EXPLANATION: The spectrum shows peaks consistent with C-H bending and "
            "C-C stretching. The closest candidate is methane (C), but the peak "
            "pattern suggests a cyclic alkane. Modified to cyclohexane (C1CCCCC1) "
            "as a better structural match.\n"
        )

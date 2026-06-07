"""Prompt templates package."""
from .system_prompt import SYSTEM_PROMPT
from .stage3_reasoning import (
    STAGE3_REASONING_PROMPT,
    format_peak_table,
    format_candidate_table,
    format_matched_rules,
)
from .stage3_modification import (
    STAGE3_MODIFICATION_PROMPT,
    format_peak_evidence,
    format_unexplained_peaks,
)

__all__ = [
    "SYSTEM_PROMPT",
    "STAGE3_REASONING_PROMPT",
    "STAGE3_MODIFICATION_PROMPT",
    "format_peak_table",
    "format_candidate_table",
    "format_matched_rules",
    "format_peak_evidence",
    "format_unexplained_peaks",
]

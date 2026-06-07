"""
SpectraExpert Pipeline — 3-stage molecular identification from Raman spectra.

Stage 1: Contrastive retrieval — encode spectrum → find top-K candidates.
Stage 2: Peak analysis — match spectral peaks to Nakamoto empirical rules.
Stage 3: LLM reasoning — evaluate candidates, select or modify structure.
"""

__version__ = "0.1.0"

from .pipeline import SpectraExpertPipeline
from .config import PipelineConfig
from .data_types import (
    DetectedPeak,
    MatchedRule,
    Candidate,
    CandidateAnalysis,
    Stage1Result,
    Stage2Result,
    Stage3Result,
    PipelineResult,
)

__all__ = [
    "SpectraExpertPipeline",
    "PipelineConfig",
    "DetectedPeak",
    "MatchedRule",
    "Candidate",
    "CandidateAnalysis",
    "Stage1Result",
    "Stage2Result",
    "Stage3Result",
    "PipelineResult",
]

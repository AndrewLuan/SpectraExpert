"""
SpectraExpert Pipeline — end-to-end orchestrator.

Chains Stage 1 (retrieval) → Stage 2 (peak analysis) → Stage 3 (LLM reasoning)
and produces a PipelineResult capturing the complete trace.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import PipelineConfig
from .data_types import PipelineResult
from .stage1_retrieval import Stage1Retrieval
from .stage2_peak_analysis import Stage2PeakAnalysis
from .stage3_llm_reasoning import Stage3LLMReasoning


class SpectraExpertPipeline:
    """End-to-end pipeline: spectral encoding → peak analysis → LLM reasoning.

    Parameters
    ----------
    config : dict, str (path to YAML), PipelineConfig, or None (defaults).
    """

    def __init__(
        self,
        config: Optional[dict | str | PipelineConfig] = None,
    ):
        # Resolve config
        if isinstance(config, PipelineConfig):
            self._config_obj = config
        elif isinstance(config, str):
            self._config_obj = PipelineConfig.from_yaml(config)
        elif isinstance(config, dict):
            self._config_obj = PipelineConfig.from_dict(config)
        else:
            self._config_obj = PipelineConfig.defaults()

        cfg_dict = self._config_obj.to_dict()

        # Print warnings
        for w in self._config_obj.warnings:
            print(f"[Pipeline] Config warning: {w}")

        # Initialise stages
        print("[Pipeline] Initialising Stage 1 (Retrieval)...")
        self.stage1 = Stage1Retrieval(cfg_dict)

        print("[Pipeline] Initialising Stage 2 (Peak Analysis)...")
        self.stage2 = Stage2PeakAnalysis(cfg_dict)

        print("[Pipeline] Initialising Stage 3 (LLM Reasoning)...")
        self.stage3 = Stage3LLMReasoning(cfg_dict)

        print(f"[Pipeline] Ready. LLM: {self.stage3._client.model_name}")

    # ── public API ─────────────────────────────────────────────────

    @property
    def config(self) -> dict:
        return self._config_obj.to_dict()

    def run(self, spectrum: np.ndarray) -> PipelineResult:
        """Run the full pipeline on a single spectrum.

        Parameters
        ----------
        spectrum : (3501,) float32 array — raw Raman spectrum.

        Returns
        -------
        PipelineResult containing all stage outputs.
        """
        t0 = time.time()

        # Stage 1
        s1_result = self.stage1.run(spectrum)
        print(f"  Stage 1: {len(s1_result.detected_peaks)} peaks, "
              f"{len(s1_result.candidates)} candidates")

        # Stage 2
        s2_result = self.stage2.run(s1_result)
        print(f"  Stage 2: {len(s2_result.matched_rules)} rules matched, "
              f"coverage={s2_result.all_peaks_explained:.2f}")

        # Stage 3
        s3_result = self.stage3.run(s1_result, s2_result)
        print(f"  Stage 3: decision={s3_result.decision}, "
              f"confidence={s3_result.confidence:.2f}")

        duration = time.time() - t0

        return PipelineResult(
            input_spectrum_hash=PipelineResult._hash(spectrum),
            stage1=s1_result,
            stage2=s2_result,
            stage3=s3_result,
            config=self.config,
            duration_seconds=duration,
        )

    def run_batch(
        self,
        spectra: list[np.ndarray],
    ) -> list[PipelineResult]:
        """Run the pipeline on multiple spectra (no parallelism in v1)."""
        return [self.run(s) for s in spectra]

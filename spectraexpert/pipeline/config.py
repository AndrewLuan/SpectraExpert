"""
Configuration for the SpectraExpert pipeline.

PipelineConfig provides a single source of truth for all parameters across
the 3 stages.  It can be initialised from a dict, from a YAML file, or
from built-in defaults.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any


# ── project root helper ──────────────────────────────────────────────
def _project_root() -> Path:
    """Return the absolute path of the SpectrExpert project root."""
    return Path(__file__).resolve().parent.parent.parent.parent


PROJECT_ROOT = _project_root()


# ── default configuration ─────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    # ── pipeline ──────────────────────────────────────────────────────
    "pipeline": {
        "top_k": 10,
        "device": None,                 # "cuda", "cpu", or None → auto
    },
    # ── stage 1: contrastive retrieval ────────────────────────────────
    "stage1": {
        "checkpoint": str(PROJECT_ROOT / "spectra_expert_pkg" / "checkpoints"
                          / "spectral_simclr_qm9s_8gpu_best.pth"),
        "embeddings_cache": str(PROJECT_ROOT / "spectra_expert_pkg"
                                / "embeddings.npz"),
        "data_dir": str(PROJECT_ROOT / "data" / "qm9s"),
        "normalize_spectrum": True,
    },
    # ── stage 2: peak analysis & knowledge matching ───────────────────
    "stage2": {
        "peak_detection": {
            "method": "find_peaks",
            "height_rel": 0.05,         # min height as fraction of max
            "prominence_rel": 0.02,     # min prominence as fraction of range
            "distance": 5,              # min pixels between peaks
            "width": 1,
        },
        "knowledge": {
            "rules_path": str(PROJECT_ROOT
                              / "nakamoto_empirical_rules_dataset.json"),
            "matching_tolerance": 20,   # cm⁻¹ tolerance when matching peaks
            "max_rules": 30,            # cap rules in LLM context
            "prefer_categories": [
                "group_frequencies",
                "geometry_determination",
                "substituent_effects",
                "charge_and_oxidation_effects",
            ],
        },
    },
    # ── stage 3: LLM reasoning ────────────────────────────────────────
    "stage3": {
        "llm": {
            "provider": "openai",       # openai | anthropic | dummy
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 4096,
            "api_key_env": "OPENAI_API_KEY",
            "endpoint": None,
            "timeout_seconds": 120,
        },
        "reasoning": {
            "min_confidence": 0.6,      # below this → try modification
            "max_candidates_in_prompt": 5,
            "include_peak_tables": True,
        },
    },
}


# ── validation rules ─────────────────────────────────────────────────

_VALID_PROVIDERS = {"openai", "anthropic", "dummy"}


def _validate(config: dict) -> list[str]:
    """Return a list of validation warnings/errors (empty = valid)."""
    warnings: list[str] = []
    provider = config.get("stage3", {}).get("llm", {}).get("provider", "")
    if provider not in _VALID_PROVIDERS:
        warnings.append(f"Unknown LLM provider '{provider}'; "
                        f"expected one of {_VALID_PROVIDERS}")
    ckpt = config.get("stage1", {}).get("checkpoint", "")
    if ckpt and not os.path.isfile(ckpt):
        warnings.append(f"Checkpoint not found: {ckpt}")
    rules = config.get("stage2", {}).get("knowledge", {}).get("rules_path", "")
    if rules and not os.path.isfile(rules):
        warnings.append(f"Rules file not found: {rules}")
    return warnings


# ── PipelineConfig ───────────────────────────────────────────────────

class PipelineConfig:
    """Immutable-ish configuration container for the pipeline.

    Access values via the ``data`` dict or via ``.get(key, default)`` with
    dot-separated keys.
    """

    def __init__(self, config: dict | None = None):
        self._data = deepcopy(DEFAULT_CONFIG)
        if config is not None:
            self._merge(self._data, config)

    # -- dict-like helpers -------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        """Retrieve a nested value via dot-path-style keys.

        Examples
        --------
        >>> cfg = PipelineConfig.defaults()
        >>> cfg.get("stage1", "checkpoint")      # str
        >>> cfg.get("stage2", "peak_detection", "height_rel")   # 0.05
        """
        val: Any = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    # -- constructors -----------------------------------------------------
    @classmethod
    def defaults(cls) -> PipelineConfig:
        """Factory: built-in defaults."""
        return cls()

    @classmethod
    def from_dict(cls, config: dict) -> PipelineConfig:
        """Factory: dict (can be a partial override of defaults)."""
        return cls(config)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> PipelineConfig:
        """Factory: YAML file path (merged with defaults)."""
        import yaml
        with open(path, "r") as fh:
            return cls(yaml.safe_load(fh))

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return deepcopy(self._data)

    def to_yaml(self, path: str | os.PathLike) -> None:
        import yaml
        with open(path, "w") as fh:
            yaml.dump(self._data, fh, default_flow_style=False)

    # -- validation -------------------------------------------------------
    @property
    def warnings(self) -> list[str]:
        return _validate(self._data)

    # -- private ----------------------------------------------------------
    @staticmethod
    def _merge(base: dict, overrides: dict) -> None:
        """Recursive (in-place) dict merge."""
        for key, val in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                PipelineConfig._merge(base[key], val)
            else:
                base[key] = deepcopy(val)

    def __repr__(self) -> str:
        return f"PipelineConfig({len(self._data)} sections)"

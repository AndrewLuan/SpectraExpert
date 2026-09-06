"""
功能概述：提供 SpectraExpert 三阶段 pipeline 的单一配置来源，支持默认值、dict 与 YAML 合并。
主要职责：``DEFAULT_CONFIG`` 声明各阶段参数，Task 8 默认显式使用 ``spectrum_graph_v2``、
``outputs/structure_aware`` 下的 v2 checkpoint/index 与默认禁用 rerank；``PipelineConfig`` 提供深合并、
读取、序列化和 warnings；``_validate`` 检查 LLM provider、Stage 1 mode、exact bool 归一化、重排字段
及外部文件是否存在。
代码库关系：由 pipeline/run/eval 构造并传入 ``Stage1Retrieval``；legacy ``embeddings_cache`` 仍保留，
但只有 ``retrieval_mode=spectral_only_v1`` 才消费，v2 缺 index/checkpoint 时不会回退。
输入输出与副作用：默认/dict 构造不读文件，warnings 只做存在性和字段检查；YAML 方法显式读写调用方
路径。warning 不代替 Stage1 的严格 runtime metadata 校验。
"""

from __future__ import annotations

import os
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


# ── project root helper ──────────────────────────────────────────────
def _project_root() -> Path:
    """Return the absolute path of the SpectrExpert project root."""
    return Path(__file__).resolve().parent.parent.parent


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
        "retrieval_mode": "spectrum_graph_v2",
        "checkpoint": str(PROJECT_ROOT / "outputs" / "structure_aware" / "best.pt"),
        "graph_index": str(PROJECT_ROOT / "outputs" / "structure_aware" / "graph_index.npz"),
        "embeddings_cache": str(PROJECT_ROOT / "spectra_expert_pkg"
                                / "embeddings.npz"),
        "data_dir": str(PROJECT_ROOT / "data" / "qm9s"),
        "normalize_spectrum": True,
        "rerank": {
            "enabled": False,
            "top_m": 100,
            "fingerprint_weight": 0.0,
            "descriptor_weight": 0.0,
        },
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
_VALID_RETRIEVAL_MODES = {"spectral_only_v1", "spectrum_graph_v2"}


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
    stage1 = config.get("stage1", {})
    if type(stage1.get("normalize_spectrum")) is not bool:
        warnings.append("stage1.normalize_spectrum must be bool")
    mode = stage1.get("retrieval_mode")
    if mode not in _VALID_RETRIEVAL_MODES:
        warnings.append(
            f"Unknown stage1 retrieval_mode '{mode}'; expected one of {_VALID_RETRIEVAL_MODES}"
        )
    elif mode == "spectrum_graph_v2":
        index_path = stage1.get("graph_index", "")
        if index_path and not os.path.isfile(index_path):
            warnings.append(f"Graph index not found: {index_path}")
    rerank = stage1.get("rerank", {})
    if not isinstance(rerank, dict):
        warnings.append("stage1.rerank must be a mapping")
    else:
        if type(rerank.get("enabled")) is not bool:
            warnings.append("stage1.rerank.enabled must be bool")
        top_m = rerank.get("top_m")
        if not isinstance(top_m, int) or isinstance(top_m, bool) or top_m <= 0:
            warnings.append("stage1.rerank.top_m must be a positive integer")
        for field in ("fingerprint_weight", "descriptor_weight"):
            value = rerank.get(field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                warnings.append(f"stage1.rerank.{field} must be nonnegative")
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

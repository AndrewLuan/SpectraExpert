#!/usr/bin/env python3
"""
功能概述：在留出分子集合上比较 KNN、编码器、直接 LLM、知识增强 LLM 与完整 agentic 管线，
生成可复核的检索和结构质量指标。
主要职责：``_PeakDetector`` 提供无需 encoder 的轻量峰检测；数据与客户端辅助函数准备样本；各
``evaluate_*`` 函数运行指定模式；聚合/报告函数落盘并展示指标；``main`` 组装 CLI 工作流。
代码库关系：使用 ``spectraexpert.pipeline``、``spectraexpert.eval.metrics`` 与直接识别 prompt；
该模块是评估消费者，不定义训练模型或知识库规则。
输入输出与副作用：读取测试集、光谱和可选 checkpoint，可能通过配置的兼容端点调用 LLM，并写入
评估 JSON/报告；导入模块不自动执行评估。

Supported modes:
  knn          — Pure K-nearest-neighbour spectral similarity (baseline 1)
  encoder      — Trained encoder retrieval (baseline 2)
  llm-direct   — LLM with detected peaks only (mode 3)
  llm-knowledge — LLM with detected peaks + Nakamoto rules (mode 4)
  agentic      — Full 3-stage pipeline (mode 5)

All LLM calls go through a unified OpenAI-compatible proxy endpoint.
Model names (passed via ``--model``)::

    gpt-5, o3, deepseek-v4-flash, claude-sonnet-4-6, gemini-2.5-flash

Usage::

    # KNN baseline
    python evaluate.py --mode knn --max-samples 5

    # Encoder baseline
    python evaluate.py --mode encoder --max-samples 5

    # LLM-direct with gpt-5
    python evaluate.py --mode llm-direct --model gpt-5 --max-samples 3

    # LLM + knowledge with Claude
    python evaluate.py --mode llm-knowledge --model claude-sonnet-4-6 --max-samples 3

    # Full agentic pipeline (dummy client for testing)
    python evaluate.py --mode agentic --provider dummy --max-samples 3

    # Full agentic pipeline with real LLM
    python evaluate.py --mode agentic --model gpt-5 --max-samples 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional
from scipy.signal import find_peaks

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Ensure project root is on sys.path
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

_PROXY_BASE_URL = "https://api.chatanywhere.tech/v1"

from spectraexpert.pipeline import PipelineConfig, SpectraExpertPipeline
from spectraexpert.pipeline.stage1_retrieval import Stage1Retrieval
from spectraexpert.pipeline.stage2_peak_analysis import Stage2PeakAnalysis
from spectraexpert.pipeline.data_types import (
    Candidate,
    DetectedPeak,
    Stage1Result,
    Stage2Result,
)
from spectraexpert.eval.metrics import (
    compute_retrieval_metrics,
    graph_edit_distance,
    tanimoto_similarity,
)
from spectraexpert.pipeline.llm.openai_client import OpenAIClient
from spectraexpert.pipeline.prompts.stage3_direct_identification import (
    DIRECT_ID_SYSTEM_PROMPT,
    DIRECT_ID_PEAKS_ONLY_PROMPT,
    DIRECT_ID_WITH_KB_PROMPT,
    format_peak_table,
    format_matched_rules_summary,
)


# =====================================================================
#  Peak detector — lightweight, no encoder needed
# =====================================================================

class _PeakDetector:
    """Minimal peak detection using scipy, no encoder/model required.

    Shares the same wavenumber axis and detection logic as Stage1Retrieval
    but does NOT load the SpectralEncoder or database embeddings.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or PipelineConfig.defaults().to_dict()
        pd_cfg = cfg.get("stage2", {}).get("peak_detection", {})

        # Wavenumber axis
        data_dir = Path(cfg.get("stage1", cfg).get("data_dir", ""))
        wn_path = data_dir / "wavenumber.parquet"
        if wn_path.is_file():
            self._wavenumber = pd.read_parquet(wn_path)["wavenumber"].values.astype(np.float32)
        else:
            self._wavenumber = np.linspace(500.0, 4000.0, 3501, dtype=np.float32)

        # Peak-detection parameters
        self._height_rel = float(pd_cfg.get("height_rel", 0.05))
        self._prominence_rel = float(pd_cfg.get("prominence_rel", 0.02))
        self._distance = int(pd_cfg.get("distance", 5))
        self._width = int(pd_cfg.get("width", 1))
        self._normalize = True

    def detect(self, spectrum: np.ndarray) -> list[DetectedPeak]:
        """Detect peaks in the spectrum (same logic as Stage1Retrieval._detect_peaks)."""
        spec = spectrum.astype(np.float64)
        if self._normalize:
            spec = (spec - spec.mean()) / (spec.std() + 1e-10)

        height = float(np.max(spec)) * self._height_rel
        prominence = float(np.ptp(spec)) * self._prominence_rel

        indices, properties = find_peaks(
            spec,
            height=max(height, 1e-6),
            prominence=max(prominence, 1e-6),
            distance=self._distance,
            width=self._width,
        )

        detected = []
        for idx in indices:
            wavenumber = float(self._wavenumber[idx])
            intensity = float(spectrum[idx])
            detected.append(DetectedPeak(
                position=wavenumber,
                intensity=intensity,
                height=float(properties["peak_heights"][np.where(indices == idx)[0][0]]) if "peak_heights" in properties else 0.0,
                prominence=float(properties["prominences"][np.where(indices == idx)[0][0]]) if "prominences" in properties else 0.0,
                width=float(properties["widths"][np.where(indices == idx)[0][0]]) if "widths" in properties else 1.0,
                left_base=int(properties.get("left_bases", [idx])[0]),
                right_base=int(properties.get("right_bases", [idx])[0]),
            ))

        detected.sort(key=lambda p: p.intensity, reverse=True)
        return detected


# =====================================================================
#  Globals
# =====================================================================

_QM9S_DB = None  # lazy load: {"spectra": (N,3501), "smiles": [str], "mol_ids": [int]}


def _load_qm9s() -> dict:
    """Lazy-load QM9S spectra and SMILES from Parquet."""
    global _QM9S_DB
    if _QM9S_DB is not None:
        return _QM9S_DB
    import pyarrow.parquet as pq

    data_dir = _PROJ_ROOT / "data" / "qm9s"
    print("  Loading QM9S database...")

    # Spectra
    spec_table = pq.read_table(str(data_dir / "spectra.parquet"))
    spec_cols = [c for c in spec_table.column_names if c != "mol_id"]
    spectra = spec_table.to_pandas()[spec_cols].values.astype(np.float32)
    mol_ids = spec_table.column("mol_id").to_numpy()

    # SMILES
    smiles_df = pd.read_parquet(str(data_dir / "smiles.parquet"))
    smiles = smiles_df["smiles_canonical"].values

    _QM9S_DB = {"spectra": spectra, "smiles": smiles, "mol_ids": mol_ids}
    print(f"  QM9S loaded: {len(smiles)} spectra, {spectra.shape[1]} dims")
    return _QM9S_DB


# =====================================================================
#  Helpers
# =====================================================================


def load_test_set(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"  Test set: {len(df)} molecules")
    return df


def get_spectrum(df: pd.DataFrame, idx: int) -> np.ndarray:
    spec_cols = [c for c in df.columns if c not in ("smiles", "smiles_len")]
    return df.iloc[idx][spec_cols].values.astype(np.float32)


def setup_stage1(config_dict: dict) -> Stage1Retrieval:
    return Stage1Retrieval(config_dict)


def soft_match(gt: str, cand: str) -> bool:
    gt_mol = Chem.MolFromSmiles(gt)
    cd_mol = Chem.MolFromSmiles(cand)
    if gt_mol is None or cd_mol is None:
        return gt == cand
    return Chem.MolToSmiles(gt_mol) == Chem.MolToSmiles(cd_mol)


def make_llm_client(model: str, temperature: float = 0.2) -> OpenAIClient:
    """Create an OpenAI-format LLM client pointed at the proxy."""
    cfg = {
        "stage3": {
            "llm": {
                "provider": "openai",
                "model": model,
                "temperature": temperature,
                "max_tokens": 4096,
                "api_key_env": "OPENAI_API_KEY",
                "endpoint": _PROXY_BASE_URL,
                "timeout_seconds": 120,
            }
        }
    }
    return OpenAIClient(cfg)


def parse_llm_smiles(text: str) -> Optional[str]:
    """Extract a valid SMILES string from LLM output.

    Looks for PREDICTED_SMILES: <...> field, then validates with RDKit.
    Falls back to searching for any SMILES-like pattern.
    """
    # First try structured field
    m = re.search(
        r"PREDICTED_SMILES\s*:\s*(\S+)", text, re.MULTILINE | re.IGNORECASE
    )
    if m:
        candidate = m.group(1).strip().rstrip(".")
        if candidate.upper() == "UNKNOWN":
            return None
        mol = Chem.MolFromSmiles(candidate)
        if mol is not None:
            try:
                return Chem.MolToSmiles(mol)
            except Exception:
                return candidate

    # Fallback: try SELECTED_SMILES field (from agentic pipeline)
    m = re.search(
        r"SELECTED_SMILES\s*:\s*(\S+)", text, re.MULTILINE | re.IGNORECASE
    )
    if m:
        candidate = m.group(1).strip().rstrip(".")
        mol = Chem.MolFromSmiles(candidate)
        if mol is not None:
            try:
                return Chem.MolToSmiles(mol)
            except Exception:
                return candidate

    # Fallback: try to find any SMILES-like string
    for token in re.findall(r"[A-Za-z0-9@+\-\[\]\(\)\\/=#$%]+", text):
        if len(token) > 3 and len(token) < 100:
            mol = Chem.MolFromSmiles(token)
            if mol is not None:
                try:
                    return Chem.MolToSmiles(mol)
                except Exception:
                    return token

    return None


# =====================================================================
#  Evaluation modes
# =====================================================================


# ---- Mode 1: Pure KNN ----

def evaluate_knn(
    test_set_path: str,
    output_path: str,
    max_samples: int | None = None,
    top_k: int = 10,
) -> dict:
    """Pure K-nearest-neighbour spectral similarity baseline.

    Normalises each spectrum to L2 unit norm, then cosine-similarity
    retrieval against the full QM9S database.
    """
    print("=" * 60)
    print("Mode 1: Pure KNN Baseline")
    print("=" * 60)

    print("\n[1/4] Loading test set...")
    df = load_test_set(test_set_path)
    if max_samples:
        df = df.iloc[:max_samples]

    print("\n[2/4] Loading QM9S database...")
    qm9s = _load_qm9s()
    db_spec = qm9s["spectra"]  # (N, 3501)
    db_smi = qm9s["smiles"]  # (N,)

    # L2-normalize DB
    db_norms = np.linalg.norm(db_spec, axis=1, keepdims=True)
    db_norms[db_norms == 0] = 1.0
    db_unit = db_spec / db_norms

    print(f"\n[3/4] Running KNN retrieval on {len(df)} queries...")
    per_sample = []
    t_start = time.time()

    for idx in range(len(df)):
        t_sample = time.time()
        gt = df.iloc[idx]["smiles"]
        spectrum = get_spectrum(df, idx)

        # L2-normalize query
        q_norm_val = np.linalg.norm(spectrum)
        q_unit = spectrum / q_norm_val if q_norm_val > 0 else spectrum

        # Cosine similarity
        sims = db_unit @ q_unit
        topk = min(top_k, len(sims))
        indices = np.argpartition(-sims, topk)[:topk]
        order = np.argsort(-sims[indices])
        indices = indices[order]

        cand_smiles = [str(db_smi[i]) for i in indices]
        cand_sims = [float(sims[i]) for i in indices]

        metrics = compute_retrieval_metrics(gt, cand_smiles, cand_sims)
        exact_match = any(soft_match(gt, c) for c in cand_smiles)

        per_sample.append({
            "query_index": int(idx),
            "ground_truth_smiles": gt,
            "smiles_len": int(df.iloc[idx]["smiles_len"]),
            "exact_match_in_candidates": exact_match,
            "time_seconds": round(time.time() - t_sample, 3),
            **metrics,
        })

        if (idx + 1) % 20 == 0 or idx == len(df) - 1:
            elapsed = time.time() - t_start
            print(
                f"  [{idx + 1:>4}/{len(df)}]  "
                f"elapsed={elapsed:.1f}s  "
                f"ETA={(elapsed / (idx + 1)) * (len(df) - idx - 1):.0f}s"
            )

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.1f}s")

    print("\n[4/4] Aggregating...")
    aggregate = compute_aggregate_metrics(per_sample)
    results = _save_results(output_path, "knn", df, per_sample, aggregate,
                            total_time, top_k=top_k, max_samples=max_samples)
    print_aggregate_report(aggregate)
    return results


# ---- Mode 2: Encoder ----

def evaluate_encoder(
    test_set_path: str,
    output_path: str,
    max_samples: int | None = None,
    top_k: int = 10,
) -> dict:
    """Run Stage-1-only encoder retrieval evaluation."""
    print("=" * 60)
    print("Mode 2: Encoder Retrieval Baseline")
    print("=" * 60)

    print("\n[1/4] Loading test set...")
    df = load_test_set(test_set_path)
    if max_samples:
        df = df.iloc[:max_samples]

    print("\n[2/4] Initialising Stage 1 retrieval...")
    config = PipelineConfig.defaults().to_dict()
    config["pipeline"]["top_k"] = top_k
    stage1 = setup_stage1(config)
    db_smiles = set(stage1._db_smiles)

    print(f"\n[3/4] Running encoder retrieval on {len(df)} queries...")
    per_sample = []
    t_start = time.time()

    for idx in range(len(df)):
        t_sample = time.time()
        gt = df.iloc[idx]["smiles"]
        spectrum = get_spectrum(df, idx)

        s1_result = stage1.run(spectrum)

        cand_smiles = [c.smiles for c in s1_result.candidates]
        cand_sims = [c.similarity for c in s1_result.candidates]

        metrics = compute_retrieval_metrics(gt, cand_smiles, cand_sims)
        exact_match = any(soft_match(gt, c) for c in cand_smiles)

        per_sample.append({
            "query_index": int(idx),
            "ground_truth_smiles": gt,
            "smiles_len": int(df.iloc[idx]["smiles_len"]),
            "exact_match_in_db": gt in db_smiles,
            "exact_match_in_candidates": exact_match,
            "n_peaks_detected": len(s1_result.detected_peaks),
            "time_seconds": round(time.time() - t_sample, 3),
            **metrics,
        })

        if (idx + 1) % 20 == 0 or idx == len(df) - 1:
            elapsed = time.time() - t_start
            print(
                f"  [{idx + 1:>4}/{len(df)}]  "
                f"elapsed={elapsed:.1f}s  "
                f"ETA={(elapsed / (idx + 1)) * (len(df) - idx - 1):.0f}s"
            )

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.1f}s")

    print("\n[4/4] Aggregating...")
    aggregate = compute_aggregate_metrics(per_sample)
    results = _save_results(output_path, "encoder", df, per_sample, aggregate,
                            total_time, top_k=top_k, max_samples=max_samples)
    print_aggregate_report(aggregate)
    return results


# ---- Mode 3 & 4: LLM Direct / LLM + Knowledge ----

def _run_llm_identification(
    mode: str,  # "llm-direct" or "llm-knowledge"
    test_set_path: str,
    output_path: str,
    model: str,
    max_samples: int | None = None,
    top_k: int = 10,
) -> dict:
    """Run LLM-based molecular identification.

    mode == "llm-direct":   only detected peaks
    mode == "llm-knowledge": detected peaks + Nakamoto rules
    """
    assert mode in ("llm-direct", "llm-knowledge")
    label = "LLM Direct" if mode == "llm-direct" else "LLM + Knowledge"
    print("=" * 60)
    print(f"Mode {label}")
    print("=" * 60)

    print("\n[1/4] Loading test set...")
    df = load_test_set(test_set_path)
    if max_samples:
        df = df.iloc[:max_samples]

    print(f"\n[2/4] Initialising peak detection"
          f"{' + rule matching' if mode == 'llm-knowledge' else ''}...")
    peak_detector = _PeakDetector()
    stage2 = Stage2PeakAnalysis(
        PipelineConfig.defaults().to_dict()
    ) if mode == "llm-knowledge" else None

    print(f"\n[3/4] Running LLM identification on {len(df)} queries (model={model})...")
    client = make_llm_client(model)
    print(f"  LLM: {client.model_name} (via proxy)")
    per_sample = []
    t_start = time.time()

    for idx in tqdm(range(len(df)), desc=f"LLM-{model}", unit="query"):
        t_sample = time.time()
        gt = df.iloc[idx]["smiles"]
        spectrum = get_spectrum(df, idx)

        # Peak detection (no encoder needed)
        peaks = peak_detector.detect(spectrum)

        # Build prompt
        if mode == "llm-direct":
            user_prompt = DIRECT_ID_PEAKS_ONLY_PROMPT.format(
                n_peaks=len(peaks),
                peak_table=format_peak_table(peaks),
            )
        else:
            rules_flat, rules_by_peak = stage2._matcher.match_peaks_batch(peaks)
            matched_rules_section = format_matched_rules_summary(peaks, rules_by_peak)
            user_prompt = DIRECT_ID_WITH_KB_PROMPT.format(
                n_peaks=len(peaks),
                peak_table=format_peak_table(peaks),
                matched_rules_section=matched_rules_section,
            )

        messages = [
            {"role": "system", "content": DIRECT_ID_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Call LLM
        try:
            llm_output = client.chat(messages)
        except Exception as e:
            print(f"  [WARN] LLM call failed for sample {idx}: {e}")
            llm_output = "PREDICTED_SMILES: UNKNOWN"

        # Parse SMILES from response
        predicted_smiles = parse_llm_smiles(llm_output)
        predicted = predicted_smiles or ""

        # Compute GED
        if predicted:
            ged = graph_edit_distance(gt, predicted, timeout=10.0)
            tani = tanimoto_similarity(gt, predicted)
        else:
            ged = None
            tani = None

        exact_match = bool(predicted and soft_match(gt, predicted))

        per_sample.append({
            "query_index": int(idx),
            "ground_truth_smiles": gt,
            "smiles_len": int(df.iloc[idx]["smiles_len"]),
            "predicted_smiles": predicted,
            "exact_match": exact_match,
            "n_peaks_detected": len(peaks),
            "ged": ged,
            "tanimoto": tani,
            "llm_output_length": len(llm_output),
            "time_seconds": round(time.time() - t_sample, 3),
        })

        if (idx + 1) % 5 == 0 or idx == len(df) - 1:
            elapsed = time.time() - t_start
            print(
                f"  [{idx + 1:>4}/{len(df)}]  "
                f"elapsed={elapsed:.1f}s  "
                f"ETA={(elapsed / (idx + 1)) * (len(df) - idx - 1):.0f}s  "
                f"GT={gt[:24]:24s}  Pred={predicted[:24]:24s}  "
                f"GED={str(ged or '?'):>4s}"
            )

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.1f}s  "
          f"({total_time / len(df):.1f}s/sample)")

    print("\n[4/4] Aggregating...")
    aggregate = compute_llm_aggregate(per_sample)
    results = _save_results(output_path, mode, df, per_sample, aggregate,
                            total_time, top_k=top_k, max_samples=max_samples,
                            model=model)
    print_llm_aggregate_report(aggregate)
    return results


# ---- Mode 5: Full pipeline ----

def evaluate_agentic(
    test_set_path: str,
    output_path: str,
    model: str,
    max_samples: int | None = None,
    top_k: int = 5,
) -> dict:
    """Full 3-stage agentic pipeline evaluation."""
    print("=" * 60)
    print("Mode 5: Full Agentic Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading test set...")
    df = load_test_set(test_set_path)
    if max_samples:
        df = df.iloc[:max_samples]

    print(f"\n[2/4] Initialising full pipeline (model={model})...")
    config_dict = PipelineConfig.defaults().to_dict()
    config_dict["pipeline"]["top_k"] = top_k
    config_dict["stage3"]["llm"]["provider"] = "openai"
    config_dict["stage3"]["llm"]["model"] = model
    config_dict["stage3"]["llm"]["endpoint"] = _PROXY_BASE_URL
    config_dict["stage3"]["llm"]["api_key_env"] = "OPENAI_API_KEY"

    pipeline = SpectraExpertPipeline(config=config_dict)

    print(f"\n[3/4] Running pipeline on {len(df)} queries...")
    per_sample = []
    t_start = time.time()

    for idx in range(len(df)):
        t_sample = time.time()
        gt = df.iloc[idx]["smiles"]
        spectrum = get_spectrum(df, idx)

        try:
            result = pipeline.run(spectrum)
            s3 = result.stage3

            # Extract final SMILES
            final_smiles = None
            if s3.decision == "candidate_selected" and s3.selected_candidate:
                final_smiles = s3.selected_candidate.candidate.smiles
            elif s3.decision == "structure_modified" and s3.modified_smiles:
                final_smiles = s3.modified_smiles
            elif s3.selected_candidate:
                final_smiles = s3.selected_candidate.candidate.smiles

            predicted = final_smiles or ""
            if predicted:
                ged = graph_edit_distance(gt, predicted, timeout=10.0)
                tani = tanimoto_similarity(gt, predicted)
            else:
                ged = None
                tani = None

            exact_match = bool(predicted and soft_match(gt, predicted))

            per_sample.append({
                "query_index": int(idx),
                "ground_truth_smiles": gt,
                "smiles_len": int(df.iloc[idx]["smiles_len"]),
                "predicted_smiles": predicted,
                "decision": s3.decision,
                "confidence": s3.confidence,
                "fallback_used": s3.fallback_used,
                "exact_match": exact_match,
                "n_peaks_detected": len(result.stage1.detected_peaks),
                "ged": ged,
                "tanimoto": tani,
                "candidate_count": len(result.stage1.candidates),
                "time_seconds": round(time.time() - t_sample, 3),
            })

            print(
                f"  [{idx + 1:>4}/{len(df)}]  "
                f"GT={gt[:20]:20s}  "
                f"Pred={predicted[:20] if predicted else '':20s}  "
                f"GED={str(ged or '?'):>4s}  "
                f"Dec={s3.decision[:8]:8s}  "
                f"Conf={s3.confidence:.2f}"
            )

        except Exception as e:
            print(f"  [ERROR] Pipeline failed for sample {idx}: {e}")
            per_sample.append({
                "query_index": int(idx),
                "ground_truth_smiles": gt,
                "error": str(e),
                "time_seconds": round(time.time() - t_sample, 3),
            })

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.1f}s  "
          f"({total_time / len(df):.1f}s/sample)")

    print("\n[4/4] Aggregating...")
    aggregate = compute_llm_aggregate(per_sample)
    results = _save_results(output_path, "agentic", df, per_sample, aggregate,
                            total_time, top_k=top_k, max_samples=max_samples,
                            model=model)
    print_llm_aggregate_report(aggregate)
    return results


# =====================================================================
#  Aggregate helpers
# =====================================================================


def compute_aggregate_metrics(per_sample: list[dict]) -> dict:
    """Compute aggregate stats for retrieval modes (knn, encoder)."""
    n = len(per_sample)

    top1_ged = [s["top1_ged"] for s in per_sample if s["top1_ged"] is not None]
    min_top5 = [s["min_ged_top5"] for s in per_sample
                if s["min_ged_top5"] is not None]
    min_top10 = [s["min_ged_top10"] for s in per_sample
                 if s["min_ged_top10"] is not None]
    min_all = [s["min_ged_all"] for s in per_sample
               if s["min_ged_all"] is not None]

    exact_match_count = sum(
        1 for s in per_sample if s["exact_match_in_candidates"]
    )
    exact_match_rate = exact_match_count / n if n > 0 else 0.0

    top1_tani = [s["top1_tanimoto"] for s in per_sample
                 if s["top1_tanimoto"] is not None]
    ranks = [s["rank_of_min_ged"] for s in per_sample
             if s["rank_of_min_ged"] > 0]

    recall_ged0_t1 = sum(
        1 for s in per_sample
        if s["top1_ged"] is not None and s["top1_ged"] == 0
    )
    recall_ged0_t5 = sum(
        1 for s in per_sample
        if s["min_ged_top5"] is not None and s["min_ged_top5"] == 0
    )
    recall_ged0_t10 = sum(
        1 for s in per_sample
        if s["min_ged_top10"] is not None and s["min_ged_top10"] == 0
    )
    recall_ged_le2_t1 = sum(
        1 for s in per_sample
        if s["top1_ged"] is not None and s["top1_ged"] <= 2
    )
    recall_ged_le2_t5 = sum(
        1 for s in per_sample
        if s["min_ged_top5"] is not None and s["min_ged_top5"] <= 2
    )
    recall_ged_le2_t10 = sum(
        1 for s in per_sample
        if s["min_ged_top10"] is not None and s["min_ged_top10"] <= 2
    )

    return {
        "num_queries": n,
        "num_valid_ged_top1": len(top1_ged),
        "exact_match_rate_topK": round(exact_match_rate, 4),
        "exact_match_count": int(exact_match_count),
        "top1_ged_mean": (round(float(np.mean(top1_ged)), 2)
                          if top1_ged else None),
        "top1_ged_median": (round(float(np.median(top1_ged)), 2)
                            if top1_ged else None),
        "top1_ged_min": (round(float(np.min(top1_ged)), 2)
                         if top1_ged else None),
        "top1_ged_max": (round(float(np.max(top1_ged)), 2)
                         if top1_ged else None),
        "min_ged_top5_mean": (round(float(np.mean(min_top5)), 2)
                              if min_top5 else None),
        "min_ged_top10_mean": (round(float(np.mean(min_top10)), 2)
                               if min_top10 else None),
        "min_ged_all_mean": (round(float(np.mean(min_all)), 2)
                             if min_all else None),
        "recall_at_ged0_top1": (round(recall_ged0_t1 / n, 4)
                                if n else 0.0),
        "recall_at_ged0_top5": (round(recall_ged0_t5 / n, 4)
                                if n else 0.0),
        "recall_at_ged0_top10": (round(recall_ged0_t10 / n, 4)
                                 if n else 0.0),
        "recall_at_ged_le2_top1": (round(recall_ged_le2_t1 / n, 4)
                                   if n else 0.0),
        "recall_at_ged_le2_top5": (round(recall_ged_le2_t5 / n, 4)
                                   if n else 0.0),
        "recall_at_ged_le2_top10": (round(recall_ged_le2_t10 / n, 4)
                                    if n else 0.0),
        "top1_tanimoto_mean": (round(float(np.mean(top1_tani)), 4)
                               if top1_tani else None),
        "mean_rank_of_best_ged": (round(float(np.mean(ranks)), 1)
                                  if ranks else None),
        "median_rank_of_best_ged": (round(float(np.median(ranks)), 1)
                                    if ranks else None),
    }


def compute_llm_aggregate(per_sample: list[dict]) -> dict:
    """Compute aggregate stats for LLM-based modes."""
    n = len(per_sample)
    valid = [s for s in per_sample if s.get("ged") is not None]
    ged_vals = [s["ged"] for s in valid if s["ged"] is not None]
    tani_vals = [s["tanimoto"] for s in valid if s["tanimoto"] is not None]
    exact_match_count = sum(1 for s in per_sample if s.get("exact_match"))
    exact_match_rate = exact_match_count / n if n else 0.0
    parsed_count = sum(1 for s in per_sample if s.get("predicted_smiles"))
    parsed_rate = parsed_count / n if n else 0.0

    return {
        "num_queries": n,
        "exact_match_rate": round(exact_match_rate, 4),
        "exact_match_count": int(exact_match_count),
        "smiles_parsed_rate": round(parsed_rate, 4),
        "smiles_parsed_count": int(parsed_count),
        "ged_mean": (round(float(np.mean(ged_vals)), 2) if ged_vals else None),
        "ged_median": (round(float(np.median(ged_vals)), 2)
                       if ged_vals else None),
        "ged_min": (round(float(np.min(ged_vals)), 2) if ged_vals else None),
        "ged_max": (round(float(np.max(ged_vals)), 2) if ged_vals else None),
        "ged_std": (round(float(np.std(ged_vals)), 2)
                    if len(ged_vals) > 1 else None),
        "tanimoto_mean": (round(float(np.mean(tani_vals)), 4)
                          if tani_vals else None),
        "tanimoto_median": (round(float(np.median(tani_vals)), 4)
                            if tani_vals else None),
    }


def _save_results(
    output_path: str, mode: str, df, per_sample, aggregate, total_time,
    top_k: int = 10, max_samples: int | None = None,
    model: str | None = None,
) -> dict:
    results = {
        "config": {
            "mode": mode,
            "model": model,
            "top_k": top_k,
            "max_samples": max_samples,
            "num_queries": (len(df) if max_samples is None
                            else min(len(df), max_samples)),
            "test_set_size": len(df),
            "total_time_seconds": round(total_time, 1),
        },
        "per_sample": per_sample,
        "aggregate": aggregate,
    }
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")
    return results


# =====================================================================
#  Reports
# =====================================================================


def print_aggregate_report(agg: dict) -> None:
    """Print aggregate report for retrieval modes."""
    print("\n" + "=" * 60)
    print("Aggregate Results")
    print("=" * 60)
    n = agg["num_queries"]
    print(f"\n  Queries evaluated: {n}")
    print(f"\n  ── Exact Match Rate ──")
    print(f"    Top-K exact match:          {agg['exact_match_rate_topK']:.2%}")
    print(f"\n  ── Graph Edit Distance ──")
    if agg["top1_ged_mean"] is not None:
        print(f"    Top-1 GED (mean):           {agg['top1_ged_mean']:.1f}")
        print(f"    Top-1 GED (median):         {agg['top1_ged_median']:.1f}")
        print(f"    Top-1 GED range:            "
              f"{agg['top1_ged_min']:.0f} – {agg['top1_ged_max']:.0f}")
    if agg["min_ged_top5_mean"] is not None:
        print(f"    Best GED in Top-5 (mean):   "
              f"{agg['min_ged_top5_mean']:.1f}")
    if agg["min_ged_top10_mean"] is not None:
        print(f"    Best GED in Top-10 (mean):  "
              f"{agg['min_ged_top10_mean']:.1f}")
    print(f"\n  ── Recall @ GED Thresholds ──")
    print(f"    Recall@GED=0   (Top-1):     "
          f"{agg['recall_at_ged0_top1']:.2%}")
    print(f"    Recall@GED=0   (Top-5):     "
          f"{agg['recall_at_ged0_top5']:.2%}")
    print(f"    Recall@GED=0   (Top-10):    "
          f"{agg['recall_at_ged0_top10']:.2%}")
    print(f"    Recall@GED≤2  (Top-1):     "
          f"{agg['recall_at_ged_le2_top1']:.2%}")
    print(f"    Recall@GED≤2  (Top-5):     "
          f"{agg['recall_at_ged_le2_top5']:.2%}")
    print(f"    Recall@GED≤2  (Top-10):    "
          f"{agg['recall_at_ged_le2_top10']:.2%}")
    print(f"\n  ── Tanimoto ──")
    if agg.get("top1_tanimoto_mean") is not None:
        print(f"    Top-1 Tanimoto (mean):      "
              f"{agg['top1_tanimoto_mean']:.3f}")
    print(f"\n  ── Best Rank ──")
    if agg.get("mean_rank_of_best_ged") is not None:
        print(f"    Mean rank of best GED:      "
              f"{agg['mean_rank_of_best_ged']:.1f}")
    print("=" * 60)


def print_llm_aggregate_report(agg: dict) -> None:
    """Print aggregate report for LLM-based modes."""
    print("\n" + "=" * 60)
    print("Aggregate Results")
    print("=" * 60)
    n = agg["num_queries"]
    print(f"\n  Queries evaluated: {n}")
    print(f"\n  ── SMILES Parsed ──")
    print(f"    Parsed:  {agg['smiles_parsed_count']}/{n}  "
          f"({agg['smiles_parsed_rate']:.1%})")
    print(f"\n  ── Exact Match ──")
    print(f"    Exact match:    {agg['exact_match_count']}/{n}  "
          f"({agg['exact_match_rate']:.1%})")
    print(f"\n  ── Graph Edit Distance ──")
    if agg["ged_mean"] is not None:
        print(f"    GED (mean ± std):           "
              f"{agg['ged_mean']:.1f} ± {agg.get('ged_std', 0):.1f}")
        print(f"    GED (median):               {agg['ged_median']:.1f}")
        print(f"    GED range:                  "
              f"{agg['ged_min']:.0f} – {agg['ged_max']:.0f}")
    print(f"\n  ── Tanimoto ──")
    if agg.get("tanimoto_mean") is not None:
        print(f"    Tanimoto (mean):            "
              f"{agg['tanimoto_mean']:.3f}")
        print(f"    Tanimoto (median):          "
              f"{agg['tanimoto_median']:.3f}")
    print("=" * 60)


# =====================================================================
#  CLI
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Multi-method evaluation of SpectraExpert on ChEMBL test set",
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["knn", "encoder", "llm-direct", "llm-knowledge", "agentic"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--test-set", type=str,
        default=str(
            _PROJ_ROOT / "data" / "processed" / "chembl_test_set.parquet"
        ),
        help="Path to test set Parquet",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results (auto-named if not given)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit to first N test samples",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Top-K candidates (for retrieval-based modes)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5",
        choices=[
            "gpt-5", "o3", "deepseek-v4-flash",
            "claude-sonnet-4-6", "gemini-2.5-flash",
        ],
        help="LLM model name (used by proxy endpoint)",
    )

    args = parser.parse_args()

    # Auto-generate output path
    model_tag = args.model.replace(".", "-").replace("/", "-")
    if args.output is None:
        fname = f"{args.mode}_{model_tag}.json"
        args.output = str(_PROJ_ROOT / "data" / "results" / fname)

    # Source environment
    subprocess.run(
        "source ~/.zshrc",
        shell=True,
        executable="/bin/zsh",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if args.mode == "knn":
        evaluate_knn(
            test_set_path=args.test_set,
            output_path=args.output,
            max_samples=args.max_samples,
            top_k=args.top_k,
        )
    elif args.mode == "encoder":
        evaluate_encoder(
            test_set_path=args.test_set,
            output_path=args.output,
            max_samples=args.max_samples,
            top_k=args.top_k,
        )
    elif args.mode in ("llm-direct", "llm-knowledge"):
        _run_llm_identification(
            mode=args.mode,
            test_set_path=args.test_set,
            output_path=args.output,
            model=args.model,
            max_samples=args.max_samples,
            top_k=args.top_k,
        )
    elif args.mode == "agentic":
        evaluate_agentic(
            test_set_path=args.test_set,
            output_path=args.output,
            model=args.model,
            max_samples=args.max_samples,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()

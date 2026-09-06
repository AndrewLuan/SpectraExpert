"""
功能概述：定义 SpectraExpert 三阶段推理之间传递的类型化 dataclass，并保存完整 pipeline trace。
主要职责：``DetectedPeak`` 表示谱峰；``Candidate`` 保存 Stage 1 候选及 Task 8 新增的显式
``retrieval_mode``/原始 ``embedding_similarity`` provenance；各 StageResult 和 PipelineResult 连接
检索、规则匹配与 LLM 推理，``PipelineResult._hash/print_summary`` 提供追踪与人类可读输出。
代码库关系：Stage 1 构造 Candidate/Stage1Result，Stage 2/3 和 prompt/eval 消费这些对象；新增字段
置于 Candidate 旧四个必填字段之后并带默认值，因此既有 positional/keyword 调用保持兼容。
输入输出与副作用：dataclass 主要保存内存状态；只有 ``print_summary`` 写标准输出，``_hash`` 对输入
数组字节计算短 SHA-256。v2 启用重排时 similarity 是最终分数，embedding_similarity 保留原 cosine。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Stage 1 — spectral representation and retrieval
# ---------------------------------------------------------------------------

@dataclass
class DetectedPeak:
    """A single peak detected in the input spectrum.

    Attributes
    ----------
    position : wavenumber in cm⁻¹ (e.g. 1350.5).
    intensity : raw intensity value at the peak apex.
    height : peak height above surrounding baseline (from find_peaks).
    prominence : peak prominence — most robust quality metric.
    width : peak full-width measured by find_peaks (in pixels).
    left_base, right_base : index bounds of the peak base.
    """
    position: float
    intensity: float
    height: float
    prominence: float
    width: float
    left_base: int
    right_base: int

    def __str__(self) -> str:
        return (f"{self.position:>7.1f} cm⁻¹  "
                f"(intensity={self.intensity:.4f}, "
                f"prominence={self.prominence:.4f})")


@dataclass
class Candidate:
    """One candidate molecule returned by the Stage 1 retrieval step.

    Attributes
    ----------
    rank : 1-based rank (1 = most similar).
    mol_id : numeric identifier in the QM9S database.
    smiles : canonical SMILES string.
    similarity : 当前排序使用的分数；legacy/无重排时等于 cosine，重排时为最终组合分数。
    retrieval_mode : 产生候选的显式索引模式，默认保持历史 ``spectral_only_v1``。
    embedding_similarity : 重排前 cosine；旧调用未提供时为 None。
    """
    rank: int
    mol_id: int
    smiles: str
    similarity: float
    retrieval_mode: str = "spectral_only_v1"
    embedding_similarity: float | None = None

    def __str__(self) -> str:
        return (f"#{self.rank}  SIM={self.similarity:.4f}  "
                f"mol_id={self.mol_id}  {self.smiles}")


@dataclass
class Stage1Result:
    """Output of Stage 1 (contrastive retrieval + peak detection).

    Attributes
    ----------
    query_embedding : (256,) L2-normalised embedding of the query spectrum.
    detected_peaks : list of peaks found in the input spectrum.
    candidates : top-K candidate molecules, ranked by similarity.
    """
    query_embedding: np.ndarray
    detected_peaks: list[DetectedPeak]
    candidates: list[Candidate]


# ---------------------------------------------------------------------------
# Stage 2 — Nakamoto rule matching
# ---------------------------------------------------------------------------

@dataclass
class MatchedRule:
    """A Nakamoto empirical rule that matches one or more detected peaks.

    Attributes
    ----------
    rule_id : e.g. "GFQ-042".
    rule_text : full text of the rule.
    category : rule category (e.g. "group_frequencies").
    matched_peak : wavenumber of the peak that triggered this match.
    confidence : rule confidence level ("definitive", "high", etc.).
    functional_group : short description of the inferred functional group.
    overlap_range : (min_cm, max_cm) from the rule's frequency range.
    """
    rule_id: str
    rule_text: str
    category: str
    matched_peak: float
    confidence: str
    functional_group: str
    overlap_range: tuple[float, float]

    def __str__(self) -> str:
        return (f"{self.rule_id:>10s}  |  peak {self.matched_peak:>7.1f}  |  "
                f"{self.functional_group}  [{self.overlap_range[0]:.0f}-"
                f"{self.overlap_range[1]:.0f} cm⁻¹]")


@dataclass
class Stage2Result:
    """Output of Stage 2 (peak-to-rule matching).

    Attributes
    ----------
    detected_peaks : same peaks as Stage 1 (for convenience).
    matched_rules : flat list of all MatchedRule objects found.
    rules_by_peak : dict mapping peak position → list of MatchedRule.
    relevant_rules_summary : formatted text block for LLM prompt context.
    all_peaks_explained : fraction of peaks that matched at least one rule.
    """
    detected_peaks: list[DetectedPeak]
    matched_rules: list[MatchedRule]
    rules_by_peak: dict[float, list[MatchedRule]]
    relevant_rules_summary: str
    all_peaks_explained: float


# ---------------------------------------------------------------------------
# Stage 3 — LLM reasoning
# ---------------------------------------------------------------------------

@dataclass
class CandidateAnalysis:
    """Analysis of a single candidate produced by the LLM.

    Attributes
    ----------
    candidate : the Candidate being analysed.
    peak_coverage : fraction of detected peaks explained by this candidate.
    llm_reasoning : the LLM's analysis text (2-3 sentences).
    score : the LLM-assigned confidence score for this candidate (0-1).
    """
    candidate: Candidate
    peak_coverage: float
    llm_reasoning: str
    score: float


@dataclass
class Stage3Result:
    """Output of Stage 3 (LLM reasoning).

    Attributes
    ----------
    decision : one of "candidate_selected", "structure_modified", "uncertain".
    selected_candidate : the highest-scoring CandidateAnalysis (if decision != "uncertain").
    modified_smiles : proposed SMILES string when the candidate was modified.
    confidence : overall confidence of the decision (0-1).
    reasoning_chain : full LLM response text (chain-of-thought trace).
    explanation : human-readable final answer (1-3 sentences).
    fallback_used : whether the structural-modification prompt was invoked.
    candidates_analyzed : analysis for each candidate the LLM evaluated.
    """
    decision: str
    selected_candidate: Optional[CandidateAnalysis]
    modified_smiles: Optional[str]
    confidence: float
    reasoning_chain: str
    explanation: str
    fallback_used: bool
    candidates_analyzed: list[CandidateAnalysis]


# ---------------------------------------------------------------------------
# Pipeline-level result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Complete output of the 3-stage SpectraExpert pipeline.

    Attributes
    ----------
    input_spectrum_hash : truncated SHA256 hex of the input (for tracking).
    stage1 : Stage1Result from the retrieval stage.
    stage2 : Stage2Result from the peak-analysis stage.
    stage3 : Stage3Result from the LLM reasoning stage.
    config : dict of the configuration used.
    duration_seconds : wall-clock time for the full pipeline.
    """
    input_spectrum_hash: str
    stage1: Stage1Result
    stage2: Stage2Result
    stage3: Stage3Result
    config: dict
    duration_seconds: float

    @staticmethod
    def _hash(spectrum: np.ndarray) -> str:
        return hashlib.sha256(spectrum.tobytes()).hexdigest()[:12]

    def print_summary(self) -> None:
        """Print a human-readable summary of the result."""
        print("=" * 60)
        print(f"SpectraExpert Pipeline Result  [{self.input_spectrum_hash}]")
        print(f"Duration: {self.duration_seconds:.2f}s")
        print("-" * 60)

        print(f"\nStage 1 — Retrieval ({len(self.stage1.candidates)} candidates):")
        for c in self.stage1.candidates[:3]:
            print(f"  {c}")
        if len(self.stage1.candidates) > 3:
            print(f"  ... and {len(self.stage1.candidates) - 3} more")

        print(f"\nStage 2 — Peak Analysis ({len(self.stage2.detected_peaks)} peaks):")
        for p in self.stage2.detected_peaks[:6]:
            matched = [r.rule_id for r in self.stage2.rules_by_peak.get(p.position, [])]
            print(f"  {p}  →  {', '.join(matched[:3])}")
        if len(self.stage2.detected_peaks) > 6:
            print(f"  ... and {len(self.stage2.detected_peaks) - 6} more peaks")

        print(f"\nStage 3 — LLM Reasoning:")
        print(f"  Decision: {self.stage3.decision}")
        print(f"  Confidence: {self.stage3.confidence:.2f}")
        if self.stage3.selected_candidate:
            cand = self.stage3.selected_candidate
            print(f"  Selected: {cand.candidate.smiles}  "
                  f"(score={cand.score:.2f})")
        if self.stage3.modified_smiles:
            print(f"  Modified SMILES: {self.stage3.modified_smiles}")
        print(f"\n  Explanation: {self.stage3.explanation}")
        print("=" * 60)

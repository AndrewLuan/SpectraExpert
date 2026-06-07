"""
Stage 3: LLM Reasoning — candidate evaluation, selection, and modification.

Builds structured prompts from Stage 1 & 2 results, calls the LLM, parses
the structured response, and (if needed) runs a modification fallback.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from proj.spectra_expert.pipeline.data_types import (
    Candidate,
    CandidateAnalysis,
    Stage1Result,
    Stage2Result,
    Stage3Result,
)
from .llm import create_llm_client, LLMClient
from .prompts import (
    SYSTEM_PROMPT,
    STAGE3_REASONING_PROMPT,
    STAGE3_MODIFICATION_PROMPT,
    format_peak_table,
    format_candidate_table,
    format_matched_rules,
    format_peak_evidence,
    format_unexplained_peaks,
)


class Stage3LLMReasoning:
    """Stage 3: use an LLM to reason about candidates, select the best
    match, or propose structural modifications.

    Parameters
    ----------
    config : dict
        Pipeline configuration.  Relevant keys under ``stage3``:

        - ``llm`` — provider, model, temperature, max_tokens, etc.
        - ``reasoning.min_confidence`` — threshold for modification fallback.
        - ``reasoning.max_candidates_in_prompt`` — max candidates in prompt.
    """

    def __init__(self, config: dict):
        self._client: LLMClient = create_llm_client(config)

        rcfg = config.get("stage3", config).get("reasoning", config)
        self._min_confidence = float(rcfg.get("min_confidence", 0.6))
        self._max_candidates = int(rcfg.get("max_candidates_in_prompt", 5))

        # Compile pattern for parsing structured fields once
        self._field_pat = re.compile(
            r"^(DECISION|SELECTED_INDEX|SELECTED_SMILES|MODIFIED_SMILES|"
            r"CONFIDENCE|EXPLANATION|CANDIDATE_ANALYSIS|MODIFICATION_RATIONALE|"
            r"EXPECTED_SPECTRAL_CHANGES)\s*:\s*(.+)$",
            re.MULTILINE | re.IGNORECASE,
        )

    # ── public API ─────────────────────────────────────────────────

    def run(
        self,
        stage1_result: Stage1Result,
        stage2_result: Stage2Result,
    ) -> Stage3Result:
        """Run Stage 3 LLM reasoning.

        Parameters
        ----------
        stage1_result : output of Stage 1.
        stage2_result : output of Stage 2.

        Returns
        -------
        Stage3Result with decision, analysis, and optional modification.
        """
        # 1. Build and send reasoning prompt
        messages = self._build_reasoning_prompt(stage1_result, stage2_result)
        llm_output = self._client.chat(messages)

        # 2. Parse response
        fields = self._parse_fields(llm_output)
        decision = fields.get("DECISION", "UNCERTAIN").strip().upper()
        confidence = self._parse_confidence(fields.get("CONFIDENCE", "0.0"))

        # 3. Build candidate analyses from LLM output
        candidates_analyzed = self._parse_candidate_analyses(
            llm_output, stage1_result.candidates,
        )

        # 4. Determine selected candidate
        selected_idx_str = fields.get("SELECTED_INDEX", "").strip()
        selected_candidate = None
        if selected_idx_str and selected_idx_str.lower() != "null":
            try:
                idx = int(selected_idx_str) - 1  # 1-based → 0-based
                if 0 <= idx < len(candidates_analyzed):
                    selected_candidate = candidates_analyzed[idx]
            except (ValueError, IndexError):
                pass

        modified_smiles = fields.get("MODIFIED_SMILES", "").strip()
        if modified_smiles.lower() == "null":
            modified_smiles = None

        explanation = fields.get("EXPLANATION", "").strip()

        fallback_used = False

        # 5. If confidence is low and we have candidates, try modification
        if (decision in ("UNCERTAIN", "MODIFY")
                or confidence < self._min_confidence):
            if stage1_result.candidates:
                mod_result = self._try_modification(
                    stage1_result, stage2_result, llm_output,
                )
                if mod_result:
                    fallback_used = True
                    decision = "structure_modified"
                    modified_smiles = mod_result.get("MODIFIED_SMILES", modified_smiles)
                    confidence = max(
                        confidence,
                        self._parse_confidence(mod_result.get("CONFIDENCE", "0.0")),
                    )
                    if mod_result.get("EXPLANATION"):
                        explanation = mod_result["EXPLANATION"]

        final_decision = self._map_decision(decision)

        return Stage3Result(
            decision=final_decision,
            selected_candidate=selected_candidate,
            modified_smiles=modified_smiles,
            confidence=min(confidence, 1.0),
            reasoning_chain=llm_output,
            explanation=explanation or "No explanation provided.",
            fallback_used=fallback_used,
            candidates_analyzed=candidates_analyzed,
        )

    # ── prompt building ────────────────────────────────────────────

    def _build_reasoning_prompt(
        self,
        s1: Stage1Result,
        s2: Stage2Result,
    ) -> list[dict[str, str]]:
        """Construct the main CoT reasoning prompt."""
        # Limit candidates
        candidates = s1.candidates[:self._max_candidates]

        peak_table = format_peak_table(s1.detected_peaks)
        candidate_table = format_candidate_table(candidates)
        matched_rules = format_matched_rules(s1.detected_peaks, s2.rules_by_peak)
        knowledge_summary = s2.relevant_rules_summary

        user_prompt = STAGE3_REASONING_PROMPT.format(
            num_peaks=len(s1.detected_peaks),
            peak_table=peak_table,
            num_candidates=len(candidates),
            candidate_table=candidate_table,
            matched_rules_section=matched_rules
            + "\n\n### Additional Knowledge Context\n\n"
            + knowledge_summary,
            max_candidates=self._max_candidates,
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _build_modification_prompt(
        self,
        s1: Stage1Result,
        s2: Stage2Result,
        initial_output: str,
    ) -> list[dict[str, str]]:
        """Build the structural modification prompt."""
        best = s1.candidates[0] if s1.candidates else None
        if best is None:
            # Fallback — no candidate to modify
            return []

        strong_evidence = format_peak_evidence(s2.matched_rules, "Strongly Indicated", 4)
        moderate_evidence = format_peak_evidence(s2.matched_rules, "Moderately Indicated", 3)
        unexplained = format_unexplained_peaks(s1.detected_peaks, s2.rules_by_peak)

        user_prompt = STAGE3_MODIFICATION_PROMPT.format(
            best_rank=best.rank,
            best_smiles=best.smiles,
            best_similarity=best.similarity,
            strong_peak_evidence=strong_evidence,
            moderate_peak_evidence=moderate_evidence,
            unexplained_peaks=unexplained,
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    # ── response parsing ───────────────────────────────────────────

    def _parse_fields(self, text: str) -> dict[str, str]:
        """Extract structured fields from LLM output via regex."""
        fields: dict[str, str] = {}
        for match in self._field_pat.finditer(text):
            key = match.group(1).strip().upper()
            value = match.group(2).strip()
            fields[key] = value
        return fields

    def _parse_confidence(self, raw: str) -> float:
        """Parse a confidence value from LLM output."""
        try:
            val = float(raw.strip())
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0

    def _parse_candidate_analyses(
        self,
        llm_output: str,
        candidates: list[Candidate],
    ) -> list[CandidateAnalysis]:
        """Attempt to extract per-candidate analysis from the LLM output.

        Falls back to creating basic analyses from the structured fields.
        """
        analyses: list[CandidateAnalysis] = []
        # Look for numbered candidate references in the CANDIDATE_ANALYSIS section
        analysis_section = ""
        in_section = False
        for line in llm_output.split("\n"):
            if "CANDIDATE_ANALYSIS" in line.upper():
                in_section = True
                continue
            if in_section and re.match(r"^(DECISION|SELECTED)", line, re.IGNORECASE):
                in_section = False
                continue
            if in_section:
                analysis_section += line + "\n"

        for i, c in enumerate(candidates):
            # Extract mention of this candidate from analysis section
            mention = f"Candidate #{c.rank}"
            reason = ""
            if mention in analysis_section:
                # Get the sentence containing this mention
                for sentence in analysis_section.replace("\n", " ").split(". "):
                    if mention in sentence:
                        reason = sentence.strip()
                        break
            if not reason:
                reason = f"Analyzed from spectral evidence."

            analyses.append(CandidateAnalysis(
                candidate=c,
                peak_coverage=0.0,  # LLM doesn't provide this directly
                llm_reasoning=reason,
                score=self._infer_candidate_score(c, analysis_section),
            ))

        return analyses

    def _infer_candidate_score(self, candidate: Candidate, text: str) -> float:
        """Try to infer the LLM's score for this candidate from the output text."""
        # Look for explicit score mentions
        for pattern in [
            rf"Candidate #{candidate.rank}[^.]*?(\d\.\d+)",
            rf"#{candidate.rank}[^.]*?(\d\.\d+)",
        ]:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        # Default: use the cosine similarity as a rough proxy
        return candidate.similarity

    def _try_modification(
        self,
        s1: Stage1Result,
        s2: Stage2Result,
        initial_output: str,
    ) -> Optional[dict[str, str]]:
        """Try the modification fallback prompt."""
        messages = self._build_modification_prompt(s1, s2, initial_output)
        if not messages:
            return None
        mod_output = self._client.chat(messages)
        return self._parse_fields(mod_output)

    @staticmethod
    def _map_decision(decision: str) -> str:
        """Normalise the decision string."""
        decision = decision.strip().upper()
        if decision in ("SELECT", "CANDIDATE_SELECTED"):
            return "candidate_selected"
        elif decision in ("MODIFY", "STRUCTURE_MODIFIED"):
            return "structure_modified"
        else:
            return "uncertain"

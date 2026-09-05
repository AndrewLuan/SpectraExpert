"""
功能概述：实现三阶段流程的第一阶段，把输入 Raman 光谱转换为 checkpoint 嵌入、检测谱峰并从
预计算数据库中检索候选分子。
主要职责：``Stage1Retrieval`` 初始化编码器、嵌入缓存及元数据；``run`` 编排检测/编码/检索；
``_detect_peaks`` 使用 scipy 配置生成 ``DetectedPeak``，``_retrieve`` 使用 legacy 余弦检索生成
有 rank 的 ``Candidate``。
代码库关系：依赖 ``spectraexpert.inference.SpectralEncoder``、数据类型及 Parquet/NPZ 数据；由
``SpectraExpertPipeline`` 和评估脚本调用，输出被 Stage 2 与 Stage 3 消费。
输入输出与副作用：构造时读取 checkpoint、缓存与可选 Parquet；无缓存时抛 FileNotFoundError，
运行只返回内存中的 ``Stage1Result``，不修改数据文件。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from spectraexpert.inference import SpectralEncoder
from spectraexpert.pipeline.data_types import Candidate, DetectedPeak, Stage1Result


class Stage1Retrieval:
    """Stage 1: encode input spectrum, detect peaks, retrieve candidates.

    Parameters
    ----------
    config : dict | PipelineConfig
        Pipeline configuration dictionary.  Relevant keys:

        - ``stage1.checkpoint`` — path to the .pth checkpoint.
        - ``stage1.embeddings_cache`` — path to pre-computed embeddings .npz.
        - ``stage1.data_dir`` — directory with spectra.parquet + smiles.parquet.
        - ``stage1.normalize_spectrum`` — bool.
        - ``stage2.peak_detection.*`` — peak-finding parameters.
        - ``pipeline.top_k`` — number of candidates to retrieve.
    """

    def __init__(self, config: dict):
        sc = config.get("stage1", config)
        pd_cfg = config.get("stage2", {}).get("peak_detection", {})
        top_k = config.get("pipeline", {}).get("top_k", 10)

        # ── encoder ────────────────────────────────────────────────
        ckpt = sc.get("checkpoint", "")
        self._encoder = SpectralEncoder(ckpt)

        # ── load pre-computed embeddings ───────────────────────────
        emb_path = sc.get("embeddings_cache", "")
        if emb_path and Path(emb_path).is_file():
            data = np.load(emb_path, allow_pickle=True)
            self._db_embeddings: np.ndarray = data["embeddings"].astype(np.float32)
            self._db_mol_ids: np.ndarray = data["mol_ids"]
            self._db_smiles: np.ndarray = data["smiles"]
        else:
            raise FileNotFoundError(f"Embeddings cache not found: {emb_path}")

        # ── load SMILES metadata for lookups ───────────────────────
        data_dir = Path(sc.get("data_dir", ""))
        smiles_path = data_dir / "smiles.parquet"
        if smiles_path.is_file():
            self._smiles_df = pd.read_parquet(smiles_path)
        else:
            # Fall back to the .npz embedded smiles (which are smile strings)
            self._smiles_df = pd.DataFrame({
                "mol_id": self._db_mol_ids,
                "smiles_canonical": self._db_smiles,
            })

        # ── load wavenumber axis ───────────────────────────────────
        wn_path = data_dir / "wavenumber.parquet"
        if wn_path.is_file():
            self._wavenumber = pd.read_parquet(wn_path)["wavenumber"].values.astype(np.float32)
        else:
            self._wavenumber = np.linspace(500.0, 4000.0, 3501, dtype=np.float32)

        # ── peak-detection parameters ──────────────────────────────
        self._height_rel = float(pd_cfg.get("height_rel", 0.05))
        self._prominence_rel = float(pd_cfg.get("prominence_rel", 0.02))
        self._distance = int(pd_cfg.get("distance", 5))
        self._width = int(pd_cfg.get("width", 1))
        self._top_k = top_k
        self._normalize = bool(sc.get("normalize_spectrum", True))

    # ── public API ─────────────────────────────────────────────────

    def run(self, spectrum: np.ndarray) -> Stage1Result:
        """Run Stage 1 on a single input spectrum.

        Parameters
        ----------
        spectrum : (3501,) float32 numpy array.

        Returns
        -------
        Stage1Result with detected peaks, query embedding, and candidates.
        """
        # 1. Detect peaks
        peaks = self._detect_peaks(spectrum)

        # 2. Encode
        query_emb = self._encoder.encode_spectrum(spectrum)  # (256,)

        # 3. Retrieve candidates
        candidates = self._retrieve(query_emb)

        return Stage1Result(
            query_embedding=query_emb,
            detected_peaks=peaks,
            candidates=candidates,
        )

    # ── peak detection ─────────────────────────────────────────────

    def _detect_peaks(self, spectrum: np.ndarray) -> list[DetectedPeak]:
        """Run scipy.signal.find_peaks with configured parameters.

        The spectrum is first normalised the same way as during training.
        """
        spec = spectrum.astype(np.float64)
        if self._normalize:
            spec = (spec - spec.mean()) / (spec.std() + 1e-10)

        # Convert relative parameters to absolute
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
            h = float(properties["peak_heights"][np.where(indices == idx)[0][0]]) if "peak_heights" in properties else 0.0
            detected.append(DetectedPeak(
                position=wavenumber,
                intensity=intensity,
                height=h,
                prominence=float(properties["prominences"][np.where(indices == idx)[0][0]]) if "prominences" in properties else 0.0,
                width=float(properties["widths"][np.where(indices == idx)[0][0]]) if "widths" in properties else 1.0,
                left_base=int(properties.get("left_bases", [idx])[0]),
                right_base=int(properties.get("right_bases", [idx])[0]),
            ))

        # Sort by intensity (strongest first)
        detected.sort(key=lambda p: p.intensity, reverse=True)
        return detected

    # ── retrieval ──────────────────────────────────────────────────

    def _retrieve(self, query_emb: np.ndarray) -> list[Candidate]:
        """Cosine-similarity retrieval against the full database.

        Uses the same method as ``SpectralEncoder.retrieve``.
        """
        indices, similarities = SpectralEncoder.retrieve(
            query_emb, self._db_embeddings, topk=self._top_k,
        )
        candidates = []
        for rank, (idx, sim) in enumerate(zip(indices, similarities), 1):
            mol_id = int(self._db_mol_ids[idx])
            smiles = str(self._db_smiles[idx])
            candidates.append(Candidate(
                rank=rank,
                mol_id=mol_id,
                smiles=smiles,
                similarity=float(sim),
            ))
        return candidates

"""
功能概述：实现三阶段流程的第一阶段，并用显式 ``retrieval_mode/checkpoint_format`` 隔离历史或
Task 7 versioned 光谱库检索与
``spectrum_graph_v2`` 纯光谱 query → 离线分子图候选检索，绝不静默 fallback 或混用产物。
主要职责：``Stage1Retrieval.__init__`` 按 mode 恢复对应 encoder/cache，v2 严格绑定 checkpoint
SHA、model_config、feature_config 与 GraphIndex；``run`` 校验一维有限且长度匹配的 spectrum，按
checkpoint ``train_config.normalize`` 要求 exact bool 且与配置一致，训练约定只归一化一次供 embedding
和 auxiliary heads 共用；``_detect_peaks`` 产生 DetectedPeak；
``_retrieve_v1/_retrieve_v2`` 生成含 mode 与原 embedding cosine provenance 的 Candidate。
代码库关系：legacy 路径使用 ``SpectralEncoder`` 和历史 embeddings NPZ；v2 路径使用
``StructureAwareSpectralEncoder``、``GraphIndex``、``RerankConfig``，输出继续供 Stage 2/3 与评估
模块消费。PipelineConfig 声明 mode、graph_index 和 rerank 字段，Candidate 保持旧构造兼容。
输入输出与副作用：构造时只读取显式 checkpoint/index/cache/wavenumber 文件；v2 查询签名只接收
spectrum，不接收 SMILES、formula、真实 descriptor/fingerprint。正辅助权重缺训练 loss/head 时在构造期
fail fast；运行只返回内存结果，不写文件；top-k 超过候选数时由索引安全截断。
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from spectraexpert.inference import (
    SpectralEncoder,
    StructureAwareSpectralEncoder,
    VersionedSpectralEncoder,
)
from spectraexpert.pipeline.data_types import Candidate, DetectedPeak, Stage1Result
from spectraexpert.retrieval import GraphIndex, RerankConfig


_RETRIEVAL_MODES = {"spectral_only_v1", "spectrum_graph_v2"}


class Stage1Retrieval:
    """把一条 spectrum 编码并从显式选择的 legacy/v2 候选库稳定检索。"""

    def __init__(self, config: dict):
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        if not isinstance(config, dict):
            raise TypeError("Stage1Retrieval config must be a dict or PipelineConfig")
        sc = config.get("stage1", config)
        if not isinstance(sc, dict):
            raise ValueError("stage1 config must be a mapping")
        pd_cfg = config.get("stage2", {}).get("peak_detection", {})
        pipeline_cfg = config.get("pipeline", {})
        self._mode = sc.get("retrieval_mode", "spectrum_graph_v2")
        if self._mode not in _RETRIEVAL_MODES:
            raise ValueError(f"stage1.retrieval_mode must be one of {sorted(_RETRIEVAL_MODES)}")
        self._top_k = pipeline_cfg.get("top_k", 10)
        if not isinstance(self._top_k, int) or isinstance(self._top_k, bool) or self._top_k <= 0:
            raise ValueError("pipeline.top_k must be a positive integer")
        normalize = sc.get("normalize_spectrum", True)
        if type(normalize) is not bool:
            raise ValueError("stage1.normalize_spectrum must be exact bool")
        self._normalize = normalize
        self._graph_index: GraphIndex | None = None
        self._rerank = RerankConfig()

        checkpoint = sc.get("checkpoint", "")
        device = pipeline_cfg.get("device")
        if self._mode == "spectrum_graph_v2":
            index_path = Path(sc.get("graph_index", ""))
            if not index_path.is_file():
                raise FileNotFoundError(f"Graph index not found: {index_path}")
            self._encoder = StructureAwareSpectralEncoder.from_checkpoint(checkpoint, device=device)
            if self._normalize is not self._encoder.normalize_spectrum:
                raise ValueError(
                    "stage1.normalize_spectrum must match checkpoint train_config.normalize"
                )
            self._graph_index = GraphIndex.load(
                index_path,
                expected_checkpoint_sha256=self._encoder.checkpoint_sha256,
                expected_feature_metadata=self._encoder.feature_metadata,
                expected_model_config=self._encoder.model_config,
                expected_descriptor_normalizer={
                    "mean": self._encoder.descriptor_mean.astype(float).tolist(),
                    "std": self._encoder.descriptor_std.astype(float).tolist(),
                },
                expected_loss_weights=self._encoder.loss_weights,
            )
            rerank = sc.get("rerank", {})
            if not isinstance(rerank, dict):
                raise ValueError("stage1.rerank must be a mapping")
            self._rerank = RerankConfig(
                enabled=rerank.get("enabled", False),
                top_m=rerank.get("top_m", 100),
                fingerprint_weight=rerank.get("fingerprint_weight", 0.0),
                descriptor_weight=rerank.get("descriptor_weight", 0.0),
            )
            capabilities = self._graph_index.metadata["capabilities"]
            if self._rerank.enabled and self._rerank.fingerprint_weight > 0 and not capabilities["fingerprint_rerank"]:
                raise ValueError(
                    "stage1 fingerprint rerank requires a trained fingerprint head "
                    "(lambda_fingerprint > 0)"
                )
            if self._rerank.enabled and self._rerank.descriptor_weight > 0 and not capabilities["descriptor_rerank"]:
                raise ValueError(
                    "stage1 descriptor rerank requires a trained descriptor head (lambda_aux > 0)"
                )
            expected_length = int(self._encoder.model_config["signal_length"])
        else:
            checkpoint_format = sc.get("checkpoint_format", "legacy")
            if checkpoint_format == "versioned":
                self._encoder = VersionedSpectralEncoder.from_checkpoint(
                    checkpoint, device=device
                )
                if self._normalize is not self._encoder.normalize_spectrum:
                    raise ValueError(
                        "stage1.normalize_spectrum must match checkpoint train_config.normalize"
                    )
            elif checkpoint_format == "legacy":
                self._encoder = SpectralEncoder(checkpoint, device=device)
            else:
                raise ValueError("stage1.checkpoint_format must be 'legacy' or 'versioned'")
            cache_path = Path(sc.get("embeddings_cache", ""))
            if not cache_path.is_file():
                raise FileNotFoundError(f"Embeddings cache not found: {cache_path}")
            try:
                with np.load(cache_path, allow_pickle=False) as archive:
                    if set(archive.files) != {"embeddings", "mol_ids", "smiles"}:
                        raise ValueError("legacy embeddings cache has missing or unexpected arrays")
                    self._db_embeddings = archive["embeddings"].astype(np.float32)
                    self._db_mol_ids = archive["mol_ids"].copy()
                    self._db_smiles = archive["smiles"].copy()
            except (OSError, ValueError) as error:
                raise ValueError(f"invalid legacy embeddings cache: {cache_path}") from error
            if (self._db_embeddings.ndim != 2 or self._db_mol_ids.shape != (len(self._db_embeddings),)
                    or self._db_smiles.shape != (len(self._db_embeddings),)
                    or self._db_mol_ids.dtype.kind not in "iu" or self._db_smiles.dtype.kind not in "US"
                    or not np.isfinite(self._db_embeddings).all()):
                raise ValueError("invalid legacy embeddings cache array shape, dtype, or value")
            expected_length = int(self._encoder.model.signal_length)

        data_dir = Path(sc.get("data_dir", ""))
        wavenumber_path = data_dir / "wavenumber.parquet"
        if wavenumber_path.is_file():
            frame = pd.read_parquet(wavenumber_path)
            if list(frame.columns) != ["wavenumber"]:
                raise ValueError("wavenumber.parquet must contain only wavenumber")
            self._wavenumber = frame["wavenumber"].to_numpy(dtype=np.float32)
        else:
            self._wavenumber = np.linspace(500.0, 4000.0, expected_length, dtype=np.float32)
        if (self._wavenumber.shape != (expected_length,) or not np.isfinite(self._wavenumber).all()
                or (expected_length > 1 and not np.all(np.diff(self._wavenumber) > 0))):
            raise ValueError("wavenumber axis must match checkpoint signal length and increase finitely")
        self._signal_length = expected_length

        self._height_rel = float(pd_cfg.get("height_rel", 0.05))
        self._prominence_rel = float(pd_cfg.get("prominence_rel", 0.02))
        self._distance = int(pd_cfg.get("distance", 5))
        self._width = int(pd_cfg.get("width", 1))

    @property
    def checkpoint_payload(self) -> Mapping[str, Any]:
        """暴露 versioned checkpoint 的已验证只读语义视图，供评测核对 provenance。"""

        payload = getattr(self._encoder, "payload", None)
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint payload is unavailable for legacy inference")
        return payload

    @property
    def candidate_smiles(self) -> np.ndarray:
        """返回当前模式实际检索的 canonical 候选 identity 数组。"""

        return self._graph_index.smiles if self._graph_index is not None else self._db_smiles

    @property
    def candidate_embeddings(self) -> np.ndarray:
        """返回与候选 identity 对齐的模型 embedding，供固定 pair cosine 审计。"""

        return self._graph_index.embeddings if self._graph_index is not None else self._db_embeddings

    @property
    def candidate_source_smiles_sha256(self) -> str:
        """返回 v2 索引记录的候选来源 identity hash。"""

        if self._graph_index is None:
            raise ValueError("candidate source hash is available only for graph index")
        return str(self._graph_index.metadata["source_smiles_sha256"])

    @staticmethod
    def _normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
        """复用 Task 2 dataset 语义：常量谱减均值，其他谱按总体标准差标准化。"""
        std = float(spectrum.std())
        if std > 1e-10:
            return ((spectrum - spectrum.mean()) / std).astype(np.float32)
        return (spectrum - spectrum.mean()).astype(np.float32)

    def _validate_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        try:
            array = np.asarray(spectrum, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("spectrum must be numeric") from error
        if array.ndim != 1:
            raise ValueError("spectrum must be a one-dimensional array")
        if array.shape != (self._signal_length,):
            raise ValueError(f"spectrum signal length mismatch: expected {self._signal_length}, got {array.shape[0]}")
        if not np.isfinite(array).all():
            raise ValueError("spectrum must contain only finite values")
        return np.ascontiguousarray(array)

    def run(self, spectrum: np.ndarray) -> Stage1Result:
        """只接收一条 spectrum，返回 query embedding、检测峰及稳定排序候选。"""
        raw = self._validate_spectrum(spectrum)
        query = self._normalize_spectrum(raw) if self._normalize else raw
        peaks = self._detect_peaks(raw, query)
        query_embedding = self._encoder.encode_spectrum(query)
        if self._mode == "spectrum_graph_v2":
            candidates = self._retrieve_v2(query, query_embedding)
        else:
            candidates = self._retrieve_v1(query_embedding)
        return Stage1Result(query_embedding=query_embedding, detected_peaks=peaks,
                            candidates=candidates)

    def _detect_peaks(self, spectrum: np.ndarray,
                      prepared_spectrum: np.ndarray) -> list[DetectedPeak]:
        """按配置检测峰；强度保留 raw spectrum，阈值复用已准备的模型 query。"""
        spec = prepared_spectrum.astype(np.float64)
        height = float(np.max(spec)) * self._height_rel
        prominence = float(np.ptp(spec)) * self._prominence_rel
        indices, properties = find_peaks(spec, height=max(height, 1e-6),
            prominence=max(prominence, 1e-6), distance=self._distance, width=self._width)
        detected = []
        for offset, idx in enumerate(indices):
            detected.append(DetectedPeak(
                position=float(self._wavenumber[idx]), intensity=float(spectrum[idx]),
                height=float(properties.get("peak_heights", np.zeros(len(indices)))[offset]),
                prominence=float(properties.get("prominences", np.zeros(len(indices)))[offset]),
                width=float(properties.get("widths", np.ones(len(indices)))[offset]),
                left_base=int(properties.get("left_bases", indices)[offset]),
                right_base=int(properties.get("right_bases", indices)[offset]),
            ))
        detected.sort(key=lambda peak: peak.intensity, reverse=True)
        return detected

    def _retrieve_v1(self, query_embedding: np.ndarray) -> list[Candidate]:
        indices, similarities = SpectralEncoder.retrieve(
            query_embedding, self._db_embeddings, topk=self._top_k)
        return [Candidate(rank=rank, mol_id=int(self._db_mol_ids[index]),
            smiles=str(self._db_smiles[index]), similarity=float(similarity),
            retrieval_mode="spectral_only_v1", embedding_similarity=float(similarity))
            for rank, (index, similarity) in enumerate(zip(indices, similarities), 1)]

    def _retrieve_v2(self, query_spectrum: np.ndarray,
                     query_embedding: np.ndarray) -> list[Candidate]:
        assert self._graph_index is not None
        predicted_descriptors = predicted_fingerprint = None
        if self._rerank.enabled and (
            self._rerank.fingerprint_weight > 0 or self._rerank.descriptor_weight > 0
        ):
            predicted_descriptors, predicted_fingerprint = self._encoder.predict_auxiliary(query_spectrum)
        result = self._graph_index.search(query_embedding, self._top_k, rerank=self._rerank,
            predicted_fingerprint=predicted_fingerprint,
            predicted_descriptors=predicted_descriptors)
        return [Candidate(rank=rank, mol_id=int(self._graph_index.mol_ids[index]),
            smiles=str(self._graph_index.smiles[index]), similarity=float(score),
            retrieval_mode="spectrum_graph_v2", embedding_similarity=float(embedding_similarity))
            for rank, (index, score, embedding_similarity) in enumerate(zip(
                result.indices, result.scores, result.embedding_similarities), 1)]

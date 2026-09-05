"""
功能概述：提供与旧版检索产物兼容的 ``SpectralEncoder`` 推理门面，将训练得到的
``SpectrumTransformer`` checkpoint 还原为单谱编码器，并实现无模型依赖的余弦近邻检索。
主要职责：构造器加载 ``model_state_dict`` 或裸 state dict；``encode_spectrum`` 将一条
有限的一维光谱转换为 float32 嵌入；静态方法 ``retrieve`` 验证二维数据库并返回稳定降序的
int64 索引和 float32 余弦分数。
代码库关系：依赖 ``spectraexpert.models.spectrum_encoder.SpectrumTransformer``；由
``pipeline.stage1_retrieval`` 调用，并兼容 ``train_contrastive.py`` 保存的 args 与
``SpectrumSimCLR`` 的 ``encoder.`` 参数前缀。
输入输出与副作用：checkpoint 只在构造时读取一次，模型被置为 eval 模式且编码不计算梯度；
维度错误、非有限数、无效 topk 或不兼容状态字典均显式抛出 ValueError/运行时加载错误。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spectraexpert.models.spectrum_encoder import SpectrumTransformer


_MODEL_ARGUMENTS = (
    "signal_length",
    "patch_size",
    "stride",
    "embed_dim",
    "depth",
    "num_heads",
    "mlp_ratio",
    "dropout",
    "projection_dim",
)


class SpectralEncoder:
    """从训练 checkpoint 恢复的单谱编码器及 legacy 余弦检索工具。"""

    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        state_dict, model_args = self._unpack_checkpoint(checkpoint)
        self.model = SpectrumTransformer(**model_args).to(self.device)
        self.model.load_state_dict(self._normalise_state_dict(state_dict))
        self.model.eval()

    @staticmethod
    def _unpack_checkpoint(checkpoint: Any) -> tuple[Mapping[str, Any], dict[str, Any]]:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Checkpoint must be a state-dict mapping")

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise ValueError("Checkpoint model_state_dict must be a mapping")

        raw_args = checkpoint.get("args", {})
        if isinstance(raw_args, Mapping):
            values = raw_args
        else:
            values = vars(raw_args) if hasattr(raw_args, "__dict__") else {}
        model_args = {name: values[name] for name in _MODEL_ARGUMENTS if name in values}
        return state_dict, model_args

    @staticmethod
    def _normalise_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(state_dict)
        for prefix in ("module.", "encoder."):
            if normalized and all(key.startswith(prefix) for key in normalized):
                normalized = {key.removeprefix(prefix): value for key, value in normalized.items()}
        return normalized

    def encode_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """编码一条有限的一维光谱，并返回一维 float32 NumPy 嵌入。"""
        array = np.asarray(spectrum, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError("spectrum must be a one-dimensional array")
        if not np.isfinite(array).all():
            raise ValueError("spectrum must contain only finite values")

        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = self.model(tensor).squeeze(0).cpu().numpy()
        return embedding.astype(np.float32, copy=False)

    @staticmethod
    def retrieve(
        query: np.ndarray,
        database: np.ndarray,
        topk: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回与 query 余弦相似度最高的稳定降序数据库邻居。"""
        query_array = np.asarray(query, dtype=np.float32)
        database_array = np.asarray(database, dtype=np.float32)
        if query_array.ndim != 1:
            raise ValueError("query must be a one-dimensional embedding")
        if database_array.ndim != 2:
            raise ValueError("database must be a two-dimensional embedding array")
        if database_array.shape[1] != query_array.shape[0]:
            raise ValueError("query and database embedding dimensions must match")
        if not isinstance(topk, (int, np.integer)) or isinstance(topk, bool) or topk <= 0:
            raise ValueError("topk must be a positive integer")
        if not np.isfinite(query_array).all() or not np.isfinite(database_array).all():
            raise ValueError("query and database must contain only finite values")

        query_norm = np.linalg.norm(query_array)
        database_norms = np.linalg.norm(database_array, axis=1, keepdims=True)
        safe_query = query_array / max(float(query_norm), np.finfo(np.float32).eps)
        safe_database = database_array / np.maximum(database_norms, np.finfo(np.float32).eps)
        scores = (safe_database @ safe_query).astype(np.float32, copy=False)
        indices = np.argsort(-scores, kind="stable")[: min(topk, len(database_array))]
        return indices.astype(np.int64, copy=False), scores[indices].astype(np.float32, copy=False)

"""
功能概述：提供与旧版检索产物兼容的 ``SpectralEncoder`` 推理门面，将训练得到的
``SpectrumTransformer`` checkpoint 还原为单谱编码器，并实现无模型依赖的余弦近邻检索。
主要职责：构造器加载 ``model_state_dict`` 或裸 state dict；裸字典仅在权重形状完整吻合历史
训练默认架构时恢复约定参数，否则要求保存 ``args``；``encode_spectrum`` 将一条有限的一维光谱
转换为 float32 嵌入；静态方法 ``retrieve`` 验证二维数据库并返回稳定降序的 int64 索引和
float32 余弦分数。
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

_LEGACY_BARE_MODEL_ARGS = {
    "signal_length": 3501,
    "patch_size": 16,
    "stride": 8,
    "embed_dim": 256,
    "depth": 6,
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "dropout": 0.1,
    "projection_dim": 256,
}


class SpectralEncoder:
    """从训练 checkpoint 恢复的单谱编码器及 legacy 余弦检索工具。"""

    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        state_dict, model_args = self._unpack_checkpoint(checkpoint)
        state_dict = self._normalise_state_dict(state_dict)
        if not model_args:
            model_args = self._infer_legacy_bare_model_args(state_dict)
        self.model = SpectrumTransformer(**model_args).to(self.device)
        self.model.load_state_dict(state_dict)
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

    @staticmethod
    def _infer_legacy_bare_model_args(state_dict: Mapping[str, Any]) -> dict[str, Any]:
        """只为形状可验证的历史默认训练产物恢复缺失的模型元数据。"""
        required_keys = {
            "pos_embed",
            "patch_embed.proj.weight",
            "projection.0.weight",
            "projection.2.weight",
        }
        if not required_keys.issubset(state_dict):
            raise ValueError(
                "cannot infer architecture from bare state dict; save checkpoint args"
            )

        patch_weight = state_dict["patch_embed.proj.weight"]
        pos_embed = state_dict["pos_embed"]
        first_projection = state_dict["projection.0.weight"]
        output_projection = state_dict["projection.2.weight"]
        if (
            patch_weight.ndim != 2
            or pos_embed.ndim != 3
            or first_projection.ndim != 2
            or output_projection.ndim != 2
        ):
            raise ValueError(
                "cannot infer architecture from malformed bare state dict; save checkpoint args"
            )

        embed_dim, patch_size = patch_weight.shape
        _, num_patches, positional_dim = pos_embed.shape
        projection_hidden, projection_input = first_projection.shape
        projection_dim, projection_input_last = output_projection.shape
        layer_indices = {
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith("encoder.layers.") and key.endswith(".self_attn.in_proj_weight")
        }
        depth = len(layer_indices)
        expected_indices = set(range(depth))
        first_attention = state_dict.get("encoder.layers.0.self_attn.in_proj_weight")
        first_ffn = state_dict.get("encoder.layers.0.linear1.weight")
        if (
            not layer_indices
            or layer_indices != expected_indices
            or not isinstance(first_attention, torch.Tensor)
            or not isinstance(first_ffn, torch.Tensor)
            or first_attention.ndim != 2
            or first_ffn.ndim != 2
            or first_attention.shape != (3 * embed_dim, embed_dim)
            or first_ffn.shape[1] != embed_dim
            or positional_dim != embed_dim
            or projection_hidden != embed_dim
            or projection_input != embed_dim
            or projection_input_last != embed_dim
            or first_ffn.shape[0] % embed_dim != 0
        ):
            raise ValueError(
                "cannot infer architecture from inconsistent bare state dict; save checkpoint args"
            )

        inferred_shapes = {
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "depth": depth,
            "mlp_ratio": first_ffn.shape[0] / embed_dim,
            "projection_dim": projection_dim,
            "num_patches": num_patches,
        }
        legacy_shapes = {
            "patch_size": _LEGACY_BARE_MODEL_ARGS["patch_size"],
            "embed_dim": _LEGACY_BARE_MODEL_ARGS["embed_dim"],
            "depth": _LEGACY_BARE_MODEL_ARGS["depth"],
            "mlp_ratio": _LEGACY_BARE_MODEL_ARGS["mlp_ratio"],
            "projection_dim": _LEGACY_BARE_MODEL_ARGS["projection_dim"],
            "num_patches": 436,
        }
        if inferred_shapes != legacy_shapes:
            raise ValueError(
                "cannot infer signal_length, stride, num_heads, and dropout from "
                "this bare state dict; save checkpoint args"
            )
        return dict(_LEGACY_BARE_MODEL_ARGS)

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

"""
功能概述：定义动态 batch 的光谱增强、精确跨模态、Morgan 结构关系与光谱辅助监督目标。
主要职责：``dynamic_nt_xent`` 支持短 batch 和 canonical multi-positive；
``multi_positive_cross_entropy`` 将相同 identity 的列归一化为每行 soft target；
``relational_structure_loss`` 以固定 Morgan Tanimoto 分布监督任意方阵相似度；
``StructureAwareObjective`` 聚合 DDP 全局 embedding/identity/fingerprint，并组合全部七项标量 loss。
代码库关系：消费 Task 4 ``MultimodalOutput`` 和 Task 2 生成的布尔 fingerprint，不复制 Morgan
schema；复用同目录 ``distributed`` 的 rank-major gather，由 Task 7 训练循环直接调用返回字典。
输入输出与副作用：输入 tensor 必须具有一致 batch/device，embedding loss 保留本 rank 梯度；
标签 gather 不保留梯度。非法温度、无正样本、shape/dtype 违约或任一非有限 loss 会立即失败。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from spectraexpert.utils.distributed import gather_identity, gather_with_local_grad

if TYPE_CHECKING:
    from spectraexpert.models.multimodal_retriever import MultimodalOutput


def _validate_temperature(value: float, name: str) -> float:
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"{name} temperature must be positive and finite")
    return temperature


def _scale_replicated_global_gradient(loss: torch.Tensor) -> torch.Tensor:
    """抵消 DDP 对各 rank 完整 global loss 参数梯度的再次平均，同时保持标量值不变。"""

    if not dist.is_available() or not dist.is_initialized():
        return loss
    world_size = dist.get_world_size()
    if world_size == 1:
        return loss
    detached = loss.detach()
    return detached + world_size * (loss - detached)


def _validate_embedding_pair(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValueError("embedding views must share a non-empty two-dimensional shape")
    if first.shape[0] == 0:
        raise ValueError("embedding batch must not be empty")
    if not first.is_floating_point() or not second.is_floating_point():
        raise TypeError("embeddings must use a floating dtype")
    if first.device != second.device or first.dtype != second.dtype:
        raise ValueError("embedding views must share dtype and device")


def _validate_identity_ids(
    values: torch.Tensor,
    expected_rows: int,
    device: torch.device,
    name: str,
) -> None:
    if values.ndim != 1 or values.shape[0] != expected_rows:
        raise ValueError(f"{name} must contain exactly one identity per row")
    if values.device != device:
        raise ValueError(f"{name} must be on the logits device")
    if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"{name} must use an integer dtype")


def _soft_target_cross_entropy(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    positive_counts = positive_mask.sum(dim=1)
    missing = torch.nonzero(positive_counts == 0, as_tuple=False).flatten()
    if missing.numel():
        raise ValueError(f"rows with no positive identity match: {missing.tolist()}")
    targets = positive_mask.to(dtype=logits.dtype) / positive_counts.to(
        dtype=logits.dtype
    ).unsqueeze(1)
    log_probabilities = F.log_softmax(logits, dim=1)
    selected = torch.where(positive_mask, log_probabilities, torch.zeros_like(log_probabilities))
    return -(targets * selected).sum(dim=1).mean()


def multi_positive_cross_entropy(
    logits: torch.Tensor,
    row_identity_ids: torch.Tensor,
    column_identity_ids: torch.Tensor,
) -> torch.Tensor:
    """以 canonical identity equality 构造每行归一化的 multi-positive cross entropy。"""

    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must be a non-empty two-dimensional tensor")
    if not logits.is_floating_point():
        raise TypeError("logits must use a floating dtype")
    _validate_identity_ids(row_identity_ids, logits.shape[0], logits.device, "row_identity_ids")
    _validate_identity_ids(
        column_identity_ids, logits.shape[1], logits.device, "column_identity_ids"
    )
    positive_mask = row_identity_ids[:, None] == column_identity_ids[None, :]
    return _soft_target_cross_entropy(logits, positive_mask)


def dynamic_nt_xent(
    first_view: torch.Tensor,
    second_view: torch.Tensor,
    temperature: float,
    identity_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算不依赖构造时 batch size 的双向光谱增强 NT-Xent。"""

    _validate_embedding_pair(first_view, second_view)
    scale = _validate_temperature(temperature, "NT-Xent")
    batch_size = first_view.shape[0]
    if identity_ids is None:
        identity_ids = torch.arange(batch_size, device=first_view.device)
    else:
        _validate_identity_ids(identity_ids, batch_size, first_view.device, "identity_ids")
    representations = torch.cat((first_view, second_view), dim=0)
    repeated_ids = torch.cat((identity_ids, identity_ids), dim=0)
    positive_mask = repeated_ids[:, None] == repeated_ids[None, :]
    diagonal = torch.eye(2 * batch_size, device=first_view.device, dtype=torch.bool)
    positive_mask = positive_mask & ~diagonal
    logits = representations @ representations.T / scale
    logits = logits.masked_fill(diagonal, -torch.inf)
    return _soft_target_cross_entropy(logits, positive_mask)


def _pairwise_tanimoto(fingerprints: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    bits = fingerprints.detach().to(dtype=dtype)
    intersection = bits @ bits.T
    counts = bits.sum(dim=1, keepdim=True)
    union = counts + counts.T - intersection
    return torch.where(
        union > 0,
        intersection / union.clamp_min(1),
        torch.ones_like(union),
    )


def relational_structure_loss(
    model_similarity: torch.Tensor,
    fingerprints: torch.Tensor,
    *,
    model_temperature: float = 0.1,
    structure_temperature: float = 0.1,
    exclude_diagonal: bool = False,
) -> torch.Tensor:
    """用固定 Tanimoto soft distribution 监督模型相似度的逐行分布。"""

    model_scale = _validate_temperature(model_temperature, "model")
    structure_scale = _validate_temperature(structure_temperature, "structure")
    if (
        model_similarity.ndim != 2
        or model_similarity.shape[0] == 0
        or model_similarity.shape[0] != model_similarity.shape[1]
    ):
        raise ValueError("model_similarity must be a non-empty square matrix")
    if not model_similarity.is_floating_point():
        raise TypeError("model_similarity must use a floating dtype")
    if (
        fingerprints.ndim != 2
        or fingerprints.shape[0] != model_similarity.shape[0]
        or fingerprints.dtype != torch.bool
    ):
        raise ValueError("fingerprints must be a boolean matrix aligned with model_similarity")
    if fingerprints.device != model_similarity.device:
        raise ValueError("fingerprints and model_similarity must share a device")
    batch_size = model_similarity.shape[0]
    if exclude_diagonal and batch_size == 1:
        return model_similarity.sum() * 0.0

    tanimoto = _pairwise_tanimoto(fingerprints, model_similarity.dtype)
    model_logits = model_similarity / model_scale
    structure_logits = tanimoto / structure_scale
    valid = torch.ones_like(model_logits, dtype=torch.bool)
    if exclude_diagonal:
        valid.fill_diagonal_(False)
        model_logits = model_logits.masked_fill(~valid, -torch.inf)
        structure_logits = structure_logits.masked_fill(~valid, -torch.inf)
    model_log_probabilities = F.log_softmax(model_logits, dim=1)
    structure_log_probabilities = F.log_softmax(structure_logits, dim=1)
    structure_probabilities = structure_log_probabilities.exp()
    terms = structure_probabilities * (
        structure_log_probabilities - model_log_probabilities
    )
    terms = torch.where(valid, terms, torch.zeros_like(terms))
    return terms.sum(dim=1).mean()


class StructureAwareObjective(nn.Module):
    """组合 DDP 全局结构感知训练目标，并返回可单独记录的各项标量。"""

    def __init__(
        self,
        *,
        temperature: float = 0.1,
        structure_temperature: float = 0.1,
        lambda_aug: float = 1.0,
        lambda_struct: float = 1.0,
        lambda_aux: float = 1.0,
        lambda_fingerprint: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = _validate_temperature(temperature, "contrastive")
        self.structure_temperature = _validate_temperature(
            structure_temperature, "structure"
        )
        weights = {
            "lambda_aug": lambda_aug,
            "lambda_struct": lambda_struct,
            "lambda_aux": lambda_aux,
            "lambda_fingerprint": lambda_fingerprint,
        }
        for name, value in weights.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be non-negative and finite")
            setattr(self, name, float(value))

    @staticmethod
    def _validate_local_inputs(
        output: MultimodalOutput,
        fingerprints: torch.Tensor,
        descriptor_targets: torch.Tensor,
        element_presence: torch.Tensor,
        identity_ids: torch.Tensor,
    ) -> None:
        first = output.spectrum_view1_embedding
        _validate_embedding_pair(first, output.spectrum_view2_embedding)
        _validate_embedding_pair(first, output.graph_embedding)
        batch_size = first.shape[0]
        _validate_identity_ids(identity_ids, batch_size, first.device, "identity_ids")
        if (
            fingerprints.ndim != 2
            or fingerprints.shape[0] != batch_size
            or fingerprints.dtype != torch.bool
            or fingerprints.device != first.device
        ):
            raise ValueError("fingerprints must be a boolean matrix on the embedding device")
        predictions = (
            ("descriptor", output.descriptor_prediction, descriptor_targets),
            ("element presence", output.element_presence_logits, element_presence),
        )
        for name, prediction, target in predictions:
            if prediction.ndim != 2 or target.shape != prediction.shape:
                raise ValueError(f"{name} target shape must match its prediction")
            if prediction.shape[0] != batch_size or prediction.device != first.device:
                raise ValueError(f"{name} prediction must align with the embedding batch/device")
            if target.device != first.device:
                raise ValueError(f"{name} target must be on the embedding device")
        if output.fingerprint_logits is not None:
            logits = output.fingerprint_logits
            if logits.shape != fingerprints.shape or logits.device != first.device:
                raise ValueError("fingerprint logits must match fingerprint targets")

    @staticmethod
    def _fingerprint_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        float_targets = targets.to(dtype=logits.dtype)
        positives = float_targets.sum(dim=0)
        negatives = float_targets.shape[0] - positives
        pos_weight = torch.where(
            (positives > 0) & (negatives > 0),
            negatives / positives.clamp_min(1),
            torch.ones_like(positives),
        )
        return F.binary_cross_entropy_with_logits(
            logits,
            float_targets,
            pos_weight=pos_weight,
        )

    def forward(
        self,
        output: MultimodalOutput,
        fingerprints: torch.BoolTensor,
        descriptor_targets: torch.Tensor,
        element_presence: torch.Tensor,
        identity_ids: torch.LongTensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_local_inputs(
            output,
            fingerprints,
            descriptor_targets,
            element_presence,
            identity_ids,
        )
        spectrum1 = gather_with_local_grad(output.spectrum_view1_embedding)
        spectrum2 = gather_with_local_grad(output.spectrum_view2_embedding)
        graph = gather_with_local_grad(output.graph_embedding)
        global_ids = gather_identity(identity_ids)
        global_fingerprints = gather_identity(fingerprints)

        cross = 0.25 * sum(
            (
                multi_positive_cross_entropy(
                    spectrum1 @ graph.T / self.temperature, global_ids, global_ids
                ),
                multi_positive_cross_entropy(
                    graph @ spectrum1.T / self.temperature, global_ids, global_ids
                ),
                multi_positive_cross_entropy(
                    spectrum2 @ graph.T / self.temperature, global_ids, global_ids
                ),
                multi_positive_cross_entropy(
                    graph @ spectrum2.T / self.temperature, global_ids, global_ids
                ),
            )
        )
        aug = dynamic_nt_xent(
            spectrum1,
            spectrum2,
            self.temperature,
            identity_ids=global_ids,
        )
        spectrum = F.normalize((spectrum1 + spectrum2) * 0.5, dim=1)
        structure_arguments = {
            "model_temperature": self.temperature,
            "structure_temperature": self.structure_temperature,
        }
        spectrum_graph = relational_structure_loss(
            spectrum @ graph.T,
            global_fingerprints,
            **structure_arguments,
        )
        spectrum_spectrum = relational_structure_loss(
            spectrum @ spectrum.T,
            global_fingerprints,
            exclude_diagonal=True,
            **structure_arguments,
        )
        graph_graph = relational_structure_loss(
            graph @ graph.T,
            global_fingerprints,
            exclude_diagonal=True,
            **structure_arguments,
        )
        cross = _scale_replicated_global_gradient(cross)
        aug = _scale_replicated_global_gradient(aug)
        struct = _scale_replicated_global_gradient(
            spectrum_graph + 0.5 * spectrum_spectrum + 0.5 * graph_graph
        )

        descriptor = F.smooth_l1_loss(
            output.descriptor_prediction,
            descriptor_targets.to(dtype=output.descriptor_prediction.dtype),
        )
        presence = F.binary_cross_entropy_with_logits(
            output.element_presence_logits,
            element_presence.to(dtype=output.element_presence_logits.dtype),
        )
        if output.fingerprint_logits is None:
            fingerprint = output.descriptor_prediction.sum() * 0.0
        else:
            fingerprint = self._fingerprint_loss(output.fingerprint_logits, fingerprints)

        losses = {
            "cross": cross,
            "aug": aug,
            "struct": struct,
            "descriptor": descriptor,
            "presence": presence,
            "fingerprint": fingerprint,
        }
        for name, loss in losses.items():
            if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"{name} loss is not finite")
        total = (
            cross
            + self.lambda_aug * aug
            + self.lambda_struct * struct
            + self.lambda_aux * (descriptor + presence)
            + self.lambda_fingerprint * fingerprint
        )
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError("total loss is not finite")
        return {"total": total, **losses}


__all__ = [
    "StructureAwareObjective",
    "dynamic_nt_xent",
    "multi_positive_cross_entropy",
    "relational_structure_loss",
]

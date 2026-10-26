# 功能概述：独立的交换性氢实验可见性消融，只允许N/O/S上H减少积分贡献，保留C-H。
# 主要职责：VisibilityNMRModel继承NMRForwardModel，复用一次图编码与全部原输出，新增
# proton_weights及exchangeable_mask；visibility_loss计算合格计数约束、完整积分锚定和轻先验。
# 代码库关系：依赖model.encode_context/decode_context，沿用x[...,0]=Z/100与氢槽mask，
# loss读取batch.target_w峰积分；不修改训练或评估样本集合。调用者仍须在全部记录上计算
# 使用proton_weights的加权W1，不能将loss中的监督资格当作删除样本或修改测试分母的规则。
# 输入输出与边界：可交换候选有效H质量为0.02+0.98sigmoid，默认约0.95；其他H质量1，
# padding0。N/O/S只是可交换候选，输出解释为预期报告质量，不是确认交换概率或坏数据判别。
# 计数监督要求所有正峰积分近整数、总积分介于固定H数与结构H总数；其他记录仍需参与W1。
# 完整积分记录的候选H被present anchor推向1；小抑制惩罚限制通过隐藏位点逃避位移误差。
# 此模块不读取文件/标签归属/外部服务，不输出实验谱线，不声称已解决缺失峰或错误结构。
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .model import NMRForwardModel
except ImportError:
    from model import NMRForwardModel


class VisibilityNMRModel(NMRForwardModel):
    """共享基础模型参数名，唯一新增参数在visibility_head下。"""

    def __init__(self, feature_dim=18, hidden_dim=128, depth=5, condition_dim=4,
                 aux_classes=0, tie_methyl=False):
        super().__init__(feature_dim=feature_dim, hidden_dim=hidden_dim, depth=depth,
                         condition_dim=condition_dim, aux_classes=aux_classes, tie_methyl=tie_methyl)
        self.visibility_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + condition_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.normal_(self.visibility_head[-1].weight, std=0.001)
        initial_probability = (0.95 - 0.02) / 0.98
        nn.init.constant_(self.visibility_head[-1].bias,
                          math.log(initial_probability / (1 - initial_probability)))

    def forward(self, batch):
        context = self.encode_context(batch)
        output = self.decode_context(batch, context)
        mask = output["proton_mask"]
        atomic_numbers = (batch["x"][..., 0] * 100).round().long()
        candidates = ((atomic_numbers == 7) | (atomic_numbers == 8) | (atomic_numbers == 16))
        exchangeable = candidates.unsqueeze(-1) & mask
        probability = 0.02 + 0.98 * self.visibility_head(context).sigmoid()
        output["proton_weights"] = torch.where(exchangeable, probability, torch.ones_like(probability)) * mask
        output["exchangeable_mask"] = exchangeable
        return output


def visibility_loss(output, batch, count_weight=0.2, present_weight=0.05,
                    prior_weight=0.01, integer_tolerance=1e-3):
    """返回加权loss、三个原始分量和计数；不包含应由训练入口计算的全样本W1。

    counts中的跳过原因互斥；eligible表示可用计数监督，不表示数据质量被确认。
    无合格记录时返回仍连接到proton_weights的零损失，支持正常反向传播。
    """
    mass = output["proton_weights"]
    mask = output["proton_mask"].bool()
    exchangeable = output["exchangeable_mask"].bool() & mask
    if mass.shape != mask.shape or mass.shape != exchangeable.shape or mass.ndim != 3:
        raise ValueError("Expected matching [B,N,4] visibility output tensors")
    target = batch["target_w"].to(mass.device)
    if target.ndim != 2 or target.shape[0] != mass.shape[0]:
        raise ValueError("Expected target_w shaped [B,number_of_peaks]")
    # 零padding合法；不能让负/NaN峰在筛选正质量后悄悄消失。
    finite = torch.isfinite(target).all(1)
    nonnegative = (target >= 0).all(1)
    positive = (target > 0).any(1)
    numeric_valid = finite & nonnegative & positive
    integer = (target - target.round()).abs().le(integer_tolerance).all(1)
    total = target.sum(1)
    structural = mask.sum((1, 2)).to(mass.dtype)
    fixed = (mask & ~exchangeable).sum((1, 2)).to(mass.dtype)
    range_valid = (total >= fixed - integer_tolerance) & (total <= structural + integer_tolerance)
    eligible = numeric_valid & integer & range_valid & (structural > 0)
    complete = eligible & (total - structural).abs().le(integer_tolerance)
    anchor_slots = exchangeable & complete[:, None, None]
    zero = mass.sum() * 0
    count_loss = F.huber_loss(mass.sum((1, 2))[eligible], total[eligible], delta=1.0) if eligible.any() else zero
    present_loss = -mass[anchor_slots].clamp_min(1e-8).log().mean() if anchor_slots.any() else zero
    prior_loss = (1 - mass[exchangeable]).mean() if exchangeable.any() else zero
    counts = {
        "records": len(mass), "eligible_count_records": int(eligible.sum()),
        "complete_integral_records": int(complete.sum()), "present_anchor_slots": int(anchor_slots.sum()),
        "exchangeable_slots": int(exchangeable.sum()), "fixed_slots": int((mask & ~exchangeable).sum()),
        "skipped_invalid_integrals": int((~numeric_valid).sum()),
        "skipped_noninteger_integrals": int((numeric_valid & ~integer).sum()),
        "skipped_outside_structural_range": int((numeric_valid & integer & (~range_valid | (structural <= 0))).sum()),
    }
    return {"loss": count_weight * count_loss + present_weight * present_loss + prior_weight * prior_loss,
            "count_loss": count_loss, "present_loss": present_loss, "prior_loss": prior_loss,
            "counts": counts}

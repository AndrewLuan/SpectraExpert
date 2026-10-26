# 功能概述：将纯二维重原子图及实验条件映射为每个重原子最多四个氢位点的化学位移。
# 主要职责：EdgeMessageLayer 执行按键类型区分的残差消息传递；NMRForwardModel
# 编码图、汇聚全局上下文并预测无序氢位点；可选辅助头预测峰组多重性及组内J参数；
# gaussian_density 将位移展宽为可微分布。
# encode_context提取共享节点/全局/条件表示，decode_context复用输出头；独立可见性
# 消融子类可调用两者避免重复图编码。拆分方法不改变参数名、state_dict或默认输出。
# 输入契约：batch 包含 x[B,N,18]、adj[B,N,N,5]、mask[B,N]、cond[B,4]、
# hcount[B,N]；adj 的五通道依次为单键、双键、三键、芳香键、其他键，mask 表示有效原子。
# 输出契约：shifts/proton_mask 均为 [B,N,4]；同碳 slot 是无序预测，训练时应采用
# 置换不变匹配，不能把 slot 编号当作物理立体化学标签。gaussian_density 的输出仅是
# 位移质量分布代理，不含 J 劈裂或真实线形，不应称为完整实验谱预测。
# aux_classes>0时额外输出mult_logits[B,N,4,C]与couplings[B,N,4,3]；后者为各
# 氢槽所属峰组最多三个正J参数（Hz），不是原子对耦合矩阵，不强加互易性，也不负责
# 自旋Hamiltonian模拟。辅助标签匹配和缺失值掩码由训练代码处理；无效氢槽同样需mask。
# aux_classes=0不创建辅助参数，保持原位移模型的state_dict和返回字段兼容。
# tie_methyl为默认关闭的独立物理消融：仅碳上3/4个H的有效槽共享均值预测，CH2
# 不绑定。这近似常温溶液中快速甲基转动，不适用于所有低温、受限转动或固态体系；
# 不改变proton_mask，无效槽不计入平均，也不声称所有甲基氢严格磁等价。
# 代码库关系：仅依赖 PyTorch，供 nmr_forward 训练入口使用；test_model.py 检查
# 掩码、原子重排等变性及梯度。无磁盘、网络或全局状态副作用，不负责 RDKit 特征化。
from __future__ import annotations

from typing import Mapping
import math

import torch
from torch import Tensor, nn


class EdgeMessageLayer(nn.Module):
    """按五种化学键分别传播邻居信息，避免构造 B*N*N*hidden 张量。"""

    def __init__(self, hidden_dim: int, edge_dim: int = 5) -> None:
        super().__init__()
        self.edge_dim = edge_dim
        self.messages = nn.Linear(hidden_dim, hidden_dim * edge_dim, bias=False)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, adj: Tensor, mask: Tensor) -> Tensor:
        b, n, width = h.shape
        messages = self.messages(h).reshape(b, n, self.edge_dim, width)
        messages = messages.permute(0, 2, 1, 3)
        incoming = torch.matmul(adj.permute(0, 3, 1, 2), messages).sum(dim=1)
        updated = self.norm(h + self.update(torch.cat((h, incoming), dim=-1)))
        return updated * mask.unsqueeze(-1)


class NMRForwardModel(nn.Module):
    """重原子 MPNN；四个独立输出 slot 可表示非等价的同碳氢。"""

    def __init__(
        self, feature_dim: int = 18, hidden_dim: int = 128,
        depth: int = 5, condition_dim: int = 4, aux_classes: int = 0,
        tie_methyl: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList(EdgeMessageLayer(hidden_dim) for _ in range(depth))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + condition_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 4),
        )
        # 位移不作硬范围截断；以约 4 ppm 初始化，允许负位移及强去屏蔽位移。
        nn.init.normal_(self.head[-1].weight, std=0.02)
        nn.init.constant_(self.head[-1].bias, 4.0)
        if aux_classes < 0:
            raise ValueError("aux_classes must be nonnegative")
        self.aux_classes = aux_classes
        self.tie_methyl = tie_methyl
        if aux_classes:
            self.mult_head = nn.Sequential(
                nn.Linear(2 * hidden_dim + condition_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 4 * aux_classes),
            )
            self.coupling_head = nn.Sequential(
                nn.Linear(2 * hidden_dim + condition_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 4 * 3),
            )
            nn.init.normal_(self.coupling_head[-1].weight, std=0.01)
            nn.init.constant_(self.coupling_head[-1].bias, math.log(math.expm1(7.0)))

    def encode_context(self, batch: Mapping[str, Tensor]) -> Tensor:
        """返回[B,N,2*hidden+condition]共享上下文，不改变参数或缓存。"""
        x, adj = batch["x"], batch["adj"]
        mask = batch["mask"].to(x.dtype)
        # 同时屏蔽边的两端，保证 padding 数值不参与计算。
        adj = adj * mask[:, :, None, None] * mask[:, None, :, None]
        h = self.encoder(x) * mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, adj, mask)
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        global_context = pooled[:, None, :].expand(-1, x.shape[1], -1)
        conditions = batch["cond"].to(x.dtype)[:, None, :].expand(-1, x.shape[1], -1)
        return torch.cat((h, global_context, conditions), dim=-1)

    def decode_context(self, batch: Mapping[str, Tensor], head_input: Tensor) -> dict[str, Tensor]:
        """将已经编码的上下文送入原位移/辅助头，保留可选甲基约束。"""
        x = batch["x"]
        shifts = self.head(head_input)
        slots = torch.arange(4, device=x.device)
        proton_mask = (slots < batch["hcount"].unsqueeze(-1)) & batch["mask"].bool().unsqueeze(-1)
        result = {"shifts": shifts, "proton_mask": proton_mask}
        if self.aux_classes:
            result["mult_logits"] = self.mult_head(head_input).reshape(*x.shape[:2], 4, self.aux_classes)
            result["couplings"] = nn.functional.softplus(self.coupling_head(head_input)).reshape(*x.shape[:2], 4, 3)
        if self.tie_methyl:
            # 原子序数编码x[...,0]=Z/100；仅真正的碳以及3H/4H有效节点参与。
            carbon = torch.isclose(x[..., 0], x.new_tensor(0.06), atol=1e-6, rtol=0)
            hydrogen_count = batch["hcount"]
            tied = carbon & ((hydrogen_count == 3) | (hydrogen_count == 4)) & batch["mask"].bool()
            for key in ("shifts", "mult_logits", "couplings"):
                if key not in result:
                    continue
                values = result[key]
                valid = proton_mask.to(values.dtype)
                replace = tied.unsqueeze(-1) & proton_mask
                if values.ndim == 4:
                    valid = valid.unsqueeze(-1)
                    replace = replace.unsqueeze(-1)
                mean = (values * valid).sum(dim=2, keepdim=True) / valid.sum(dim=2, keepdim=True).clamp_min(1)
                result[key] = torch.where(replace, mean, values)
        return result

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return self.decode_context(batch, self.encode_context(batch))


def gaussian_density(
    shifts: Tensor, proton_mask: Tensor, grid: Tensor,
    sigma: float | Tensor = 0.08,
) -> Tensor:
    """每个氢贡献相同离散质量；grid 应为严格递增、均匀的 ppm 网格。

    每个核在有限网格上单独归一化，边界核不丢失积分。远离网格的位移会集中
    到最近端点，故训练仍需位移或边界损失；这不是物理线宽或完整 NMR 模拟。
    sigma 可为正标量或形状匹配 shifts 的正张量；所有无效氢的梯度为零。
    """
    sigma_tensor = torch.as_tensor(sigma, device=shifts.device, dtype=shifts.dtype)
    if sigma_tensor.ndim:
        sigma_tensor = sigma_tensor.unsqueeze(-1)
    offsets = (grid.to(shifts)[None, None, None, :] - shifts.unsqueeze(-1))
    logits = -0.5 * (offsets / sigma_tensor.clamp_min(1e-6)).square()
    kernels = logits.softmax(dim=-1) * proton_mask.unsqueeze(-1).to(shifts.dtype)
    mass = kernels.sum(dim=(1, 2))
    count = proton_mask.sum(dim=(1, 2)).to(shifts.dtype).unsqueeze(-1)
    return mass / count.clamp_min(1)

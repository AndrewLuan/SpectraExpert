"""
功能概述：汇总 SpectraExpert 的训练期通用工具公开接口。
主要职责：导出 DDP rank-major gather、动态 NT-Xent、multi-positive cross entropy、Morgan
结构关系 loss 和组合 ``StructureAwareObjective``，同时保留子模块路径的向后兼容导入方式。
代码库关系：Task 5 的训练目标由 ``structure_losses`` 实现并依赖 ``distributed``；Task 7 可从
本包入口导入这些稳定接口。Morgan schema 继续由 ``spectraexpert.data.structure_features`` 独占。
输入输出与副作用：本文件只做符号重导出；导入时不初始化进程组、不分配训练 tensor，也不访问文件。
"""

from spectraexpert.utils.distributed import gather_identity, gather_with_local_grad
from spectraexpert.utils.structure_losses import (
    StructureAwareObjective,
    dynamic_nt_xent,
    multi_positive_cross_entropy,
    relational_structure_loss,
)

__all__ = [
    "StructureAwareObjective",
    "dynamic_nt_xent",
    "gather_identity",
    "gather_with_local_grad",
    "multi_positive_cross_entropy",
    "relational_structure_loss",
]

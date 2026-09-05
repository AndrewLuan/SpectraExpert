"""
功能概述：提供 Task 8 版本化图候选索引与重排配置的稳定公开导入入口。
主要职责：导出 ``GraphIndex``、``GraphSearchResult``、``RerankConfig`` 和离线
``build_graph_index``，确保 pipeline 与 CLI 使用相同索引和重排契约，具体实现位于 ``graph_index``。
代码库关系：上游依赖 Task 2 的结构特征 schema 与 Task 7 checkpoint metadata，下游由 Stage 1
纯光谱查询和离线索引构建脚本消费；本文件只负责符号组织。
输入输出与副作用：导入包本身不读取 checkpoint、NPZ 或分子数据，不创建设备 tensor。
"""

from spectraexpert.retrieval.graph_index import (
    GraphIndex,
    GraphSearchResult,
    RerankConfig,
    build_graph_index,
)

__all__ = ["GraphIndex", "GraphSearchResult", "RerankConfig", "build_graph_index"]

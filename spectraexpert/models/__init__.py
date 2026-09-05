"""
功能概述：提供光谱与分子图编码器的统一公开导入入口。
主要职责：导出 PatchEmbed1D、SpectrumTransformer、SpectrumSimCLR、GraphEncoder 及输出类型，
以及 StructureAwareRetriever/MultimodalOutput，方便训练、推理和 checkpoint 构造复用。
代码库关系：实现分别位于 spectrum_encoder、graph_encoder 和 multimodal_retriever；保持旧模块路径。
输入输出与副作用：仅导入类定义，不实例化模型、不读取数据或 checkpoint。
"""

from spectraexpert.models.graph_encoder import GraphEncoder, GraphEncoderOutput
from spectraexpert.models.multimodal_retriever import MultimodalOutput, StructureAwareRetriever
from spectraexpert.models.spectrum_encoder import PatchEmbed1D, SpectrumSimCLR, SpectrumTransformer

__all__ = [
    "GraphEncoder", "GraphEncoderOutput", "MultimodalOutput", "PatchEmbed1D",
    "SpectrumSimCLR", "SpectrumTransformer", "StructureAwareRetriever",
]

"""
功能概述：验证安装后的 ``spectraexpert`` 包不再依赖历史 ``proj.spectra_expert`` 命名空间，并为
旧版余弦检索 API 建立最小、可重复的行为基线。
主要职责：``test_public_modules_import_without_legacy_proj_namespace`` 覆盖训练、管线、评估和推理
公开模块；``test_retrieve_returns_descending_cosine_neighbors`` 验证 ``SpectralEncoder.retrieve`` 的
候选排序与分数单调性。
代码库关系：由 ``pytest.ini`` 收集，直接消费 ``spectraexpert.inference``；它保护
``train_contrastive``、pipeline 和 eval 的统一绝对包导入迁移。
输入输出与副作用：构造内存中的 float32 向量，不读取 checkpoint 或数据集；导入失败或检索排序偏离
契约时以 pytest assertion 失败，不修改项目文件。
"""

import importlib

import numpy as np


def test_public_modules_import_without_legacy_proj_namespace():
    for name in (
        "spectraexpert.train_contrastive",
        "spectraexpert.pipeline",
        "spectraexpert.eval.metrics",
        "spectraexpert.inference",
    ):
        importlib.import_module(name)


def test_retrieve_returns_descending_cosine_neighbors():
    from spectraexpert.inference import SpectralEncoder

    query = np.array([1.0, 0.0], dtype=np.float32)
    database = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]], dtype=np.float32
    )

    indices, scores = SpectralEncoder.retrieve(query, database, topk=2)

    assert indices.tolist() == [1, 2]
    assert scores[0] >= scores[1]

"""功能概述：列出整体 NMR 评分实际读取的数据与算法输入，统一冻结边界。

scoring_inputs_paths 覆盖四份既有评分/报告先验和 HMDB 原始谱表；actual_inputs 再加入
bank、误差宽度、冻结查询侧表、增量预测和关键评分源码。返回 canonical 绝对路径，
不读取内容、不写文件；调用方对每项计算 SHA 并在 locked 阶段逐项验证。
避免仅锁候选均值却允许改变报告先验、观测口径或算法的伪冻结。
"""
from pathlib import Path


CRITICAL_SOURCES=(
    'h_global_protocol.py','h_global_bank.py','h_global_scoring.py','h_global_features.py',
    'h_global_extract.py','h_global_ranker.py','h_latent_score.py','h_score.py','h_observed.py',
    'h_uncertainty.py','h_observation_finalize.py','h_condinfer.py','h_observation_study.py',
    'likelihood.py','evaluate.py','h_labels.py','h_conditions.py','h_metadata.py','h_eval.py','joint.py','prepare.py','data.py',
)


def scoring_inputs_paths(base):
    h=Path(base)/'hydrogen_analysis'
    return [h/'observation_comparison/frozen_selection.json',h/'routed_evaluation/site/count_prior.json',
        h/'routed_evaluation/frozen_selection.json',h/'observation_process/training_dropout_priors.json',
        Path('/data/SpecGPT/dataset/NMR_exprimental/experimental_finetune/data.jsonl')]


def actual_inputs(base,bank,widths,cohorts,extension=None):
    paths=[Path(bank),Path(widths),Path(cohorts)/'cohorts.pt',Path(cohorts)/'manifest.json']+scoring_inputs_paths(base)
    paths += [Path(__file__).with_name(name) for name in CRITICAL_SOURCES]
    if extension:paths.append(Path(extension))
    return sorted({path.resolve() for path in paths})

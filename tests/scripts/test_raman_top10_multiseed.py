"""功能概述：检验多种子结果统计函数的样本标准差与边界值。
主要职责：用小数组验证 mean、sample_sd、minimum、maximum；不训练、不访问真实数据。
代码库关系：直接调用 scripts.raman_top10_multiseed.stats，防止汇总口径静默改变。
"""
import pytest

from scripts.raman_top10_multiseed import stats


def test_stats_uses_sample_standard_deviation():
    result = stats([.4, .5, .6])
    assert {k: result[k] for k in ('mean', 'sample_sd', 'minimum', 'maximum')} == pytest.approx(
        dict(mean=.5, sample_sd=.1, minimum=.4, maximum=.6))
    assert result['mean_t_ci95'][0] < .5 < result['mean_t_ci95'][1]

"""功能概述：验证非 LLM 重排实验的科学边界和数值契约，使用小型合成数据。
主要职责：检查峰—结构证据、无证据保序、训练组隔离、分子组重复测量等权、线性排序
方向和配对缺失处理；不读取真实实验缓存、不训练 encoder、不调用 API。
代码库关系：直接调用 scripts.raman_nonllm_rerank 的纯函数；这些检查防止特征退化、
样本泄漏和 GED 超时被零填充，从而保护 develop/test 的统一评分协议。
"""
import numpy as np
import pytest

from scripts.raman_nonllm_rerank import features, check_split, fit_ranker, predict, paired

CANDIDATES=['CC', 'CCC', 'CCO', 'CCN', 'CC#N', 'CC(=O)C', 'COC', 'C=C', 'CCCl', 'CCBr']


def test_evidence_is_mode_specific_and_no_evidence_keeps_order():
    empty=features([],CANDIDATES)
    assert not empty[:,1:].any()
    assert np.argsort(-empty[:,1],kind='stable').tolist()==list(range(10))
    nitrile=features([2240],CANDIDATES)
    assert nitrile[4,1]>0
    assert nitrile[5,1]==0
    with pytest.raises(ValueError): features([],['CC']*10)


def test_experimental_group_isolation():
    check_split(['a'],['b'],['c'])
    for parts in [(['a'],['a'],['c']),(['a'],['b'],['b']),([],['b'],['c'])]:
        with pytest.raises(ValueError): check_split(*parts)


def test_pairwise_fit_direction_and_group_equal_repeats():
    rng=np.random.default_rng(9)
    x=rng.normal(size=(3,10,2))
    utility=2*x[:,:,1]-x[:,:,0]
    model=fit_ranker(x,utility,['a','b','c'],[0,1],.01)
    assert np.array_equal(predict(x,model),utility.argmax(1))
    duplicate=fit_ranker(np.concatenate([x,x[:1]]),np.concatenate([utility,utility[:1]]),
                         ['a','b','c','a'],[0,1],.01)
    np.testing.assert_allclose(model['scale'],duplicate['scale'],atol=1e-12)
    np.testing.assert_allclose(model['weights'],duplicate['weights'],atol=1e-12)


def test_paired_missing_ged_is_not_zero_and_ids_must_match():
    a=[dict(query_id=1,group_hash='g',exact=True,tanimoto=.5,chemical_ged_v2=None),
       dict(query_id=2,group_hash='h',exact=False,tanimoto=.3,chemical_ged_v2=2.)]
    b=[dict(query_id=1,group_hash='g',exact=False,tanimoto=.2,chemical_ged_v2=4.),
       dict(query_id=2,group_hash='h',exact=False,tanimoto=.2,chemical_ged_v2=3.)]
    stats=paired(a,b)
    assert stats['chemical_ged_v2']['n']==1
    assert stats['chemical_ged_v2']['improvement']==1
    assert stats['exact']['improved']==1 and stats['exact']['unchanged']==1
    with pytest.raises(ValueError): paired(a,list(reversed(b)))

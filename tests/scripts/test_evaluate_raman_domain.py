"""功能概述：校验域迁移报告的配对比较方向和缺失指标处理，避免 GED 符号反转或样本错配。
主要职责：合成严格配对向量验证 bootstrap 差值与区间；缺失项必须成对删除。
代码库关系：调用 evaluate_raman_domain.paired_bootstrap，不进行真实评测、GPU 或 API 请求。
输入输出与副作用：只创建内存数组，没有文件副作用。
"""
from scripts.evaluate_raman_domain import paired_bootstrap


def test_paired_ged_sign_and_nulls():
    result=paired_bootstrap([2,3,None,100],[3,4,5,None],larger_better=False)
    assert result['n']==2 and result['improvement']==1
    assert result['ci95']==[1.,1.]


def test_equal_exact_and_all_missing():
    same=paired_bootstrap([True,False],[True,False])
    assert same['improvement']==0 and same['ci95']==[0.,0.]
    assert paired_bootstrap([None],[1])['n']==0
    clustered=paired_bootstrap([1,1,1],[0,0,0],groups=['a','a','b'])
    assert clustered['ci95']==[1.,1.]

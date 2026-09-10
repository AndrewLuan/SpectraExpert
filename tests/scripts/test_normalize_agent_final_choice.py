"""功能概述：验证模型无关最终答案归一化不会根据正文讨论、冲突数字或候选外结构猜选。
主要职责：覆盖 Markdown、换行、SMILES、最终数字和矛盾字段；全部使用合成候选。
代码库关系：直接测试 normalize_agent_final_choice.extract，无网络、真实谱或真值输入。
"""
import pytest
from scripts.normalize_agent_final_choice import extract


@pytest.mark.parametrize('text,index',[
    ('**Selected Candidate:**\n2. CCN',1),
    ('Select: **2**',1),
    ('Selected Candidate: C(C)O',0),
    ('Analysis mentions candidate 2\n\n1',0),
    ('SELECTED_INDEX: 1\nCONFIDENCE: 0.7',0),
    ('Selected Candidate: 1 (CCN)',None),
    ('Selected Candidate: 1\nSELECTED_INDEX: 2',None),
    ('Candidate 2 might fit. Candidate 1 is also possible.',None),
    ('Select: 11',None),
    ('Selected Candidate: CCC',None),
    ('Candidate 2: **CCN**',1),
    ('Therefore the most likely candidate is:\n\n**1. CCO**',0),
    ('The selected candidate is: **2. CCN**',1),
    ('Candidate: **8**',None),
    ('Candidate: **CCN**',1),
    ('Conclusion\n2. CCO',None),
])
def test_final_fields(text,index):
    assert extract(text,['CCO','CCN'])[0]==index

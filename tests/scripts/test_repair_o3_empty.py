"""功能概述：离线检查 o3 修复仅替换空正文截断请求，保留其余响应与完整提示词。
主要职责：构造 200 条最小冻结产物，验证预算变更、复用数量和续跑哈希防护。
代码库关系：测试 repair_o3_empty.prepare 及 run_agent_multimodel.load，不调用网络。
输入输出与副作用：测试文件仅在 pytest 临时目录生成，不读取真实实验和凭据。
"""
import pytest
from scripts import run_agent_multimodel as base
from scripts.repair_o3_empty import prepare


def test_repair_retains_success_and_hash_protects_it(tmp_path):
    parent=tmp_path/'parent'; parent.mkdir()
    for sub in ('requests','responses'):(parent/'o3'/sub).mkdir(parents=True)
    for name in ('queries.json','tasks.json'):base.save(parent/name, [])
    for i in range(200):
        base.save(parent/'o3/requests'/f'{i:04d}.json',dict(model='o3',max_tokens=4096,messages=[{'content':'frozen'}]))
        base.save(parent/'o3/responses'/f'{i:04d}.json',dict(content='' if i==0 else 'SELECTED_INDEX: 1',finish_reason='length' if i==0 else 'stop'))
    protocol=dict(config=dict(concurrency=200,timeout_seconds=600,max_attempts=3,models=[dict(tag='o3',model='o3')]),
                  runner_sha256=base.digest(__import__('pathlib').Path(base.__file__)),files_sha256={})
    base.save(parent/'protocol.json',protocol)
    out=tmp_path/'child'
    assert prepare(parent,out,16384)==dict(reused=199,pending=1,max_tokens=16384)
    assert not (out/'o3/responses/0000.json').exists()
    assert base.read(out/'o3/requests/0000.json')['max_tokens']==16384
    assert base.read(out/'o3/requests/0001.json')['max_tokens']==4096
    assert base.read(out/'o3/requests/0000.json')['messages']==[{'content':'frozen'}]
    base.load(out)
    base.save(out/'o3/responses/0001.json',{'content':'changed'})
    with pytest.raises(ValueError,match='frozen input changed'):base.load(out)

"""功能概述：离线验证多模型 Agent 重放的并发、超时、续跑及错误留存契约。
主要职责：fixture 创建微型冻结请求；测试使用 httpx.MockTransport 模拟服务，
统计并发峰值、HTTP 重试与成功缓存，确认凭据回显被脱敏，不接触真实 API 或谱库。
代码库关系：直接调用 scripts.run_agent_multimodel 的配置验证和异步运行入口。
输入输出与副作用：所有结果写 pytest 临时目录，环境变量用 monkeypatch 自动恢复。
"""
import asyncio
import json

import httpx
import pytest

from scripts import run_agent_multimodel as runner


def config(concurrency=200, timeout=600):
    return dict(concurrency=concurrency, timeout_seconds=timeout, max_attempts=2,
                models=[dict(tag='demo', model='demo', base_url_env='TEST_AGENT_URL', api_key_env='TEST_AGENT_KEY')])


@pytest.mark.parametrize('concurrency,timeout', [(201,600), (0,600), (200,300), (200,float('inf'))])
def test_limits(concurrency, timeout):
    with pytest.raises(ValueError):
        runner.validate(config(concurrency, timeout))


def test_global_concurrency_resume_and_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_AGENT_URL', 'https://example.invalid/custom/v1')
    monkeypatch.setenv('TEST_AGENT_KEY', 'secret-value')
    for folder in ('requests', 'responses', 'attempts'):
        (tmp_path/'demo'/folder).mkdir(parents=True)
    for i in range(200):
        runner.save(tmp_path/'demo/requests'/f'{i:04d}.json', {'model':'demo', 'messages':[]})
    state = dict(active=0, peak=0, calls=0)

    async def handler(request):
        assert str(request.url) == 'https://example.invalid/custom/v1/chat/completions'
        assert all(v == 600 for v in request.extensions['timeout'].values())
        state['calls'] += 1
        state['active'] += 1
        state['peak'] = max(state['peak'], state['active'])
        await asyncio.sleep(.01)
        state['active'] -= 1
        return httpx.Response(200, json={'choices':[{'message':{'content':'SELECTED_INDEX: 1 secret-value'}}]})

    async def exercise():
        transport = httpx.MockTransport(handler)
        protocol = {'config':config()}
        assert await runner.run(tmp_path, protocol, transport) == 0
        assert state['peak'] == 200
        assert await runner.run(tmp_path, protocol, transport) == 0
        assert state['calls'] == 200
    asyncio.run(exercise())
    assert 'secret-value' not in ''.join(p.read_text() for p in tmp_path.rglob('*.json'))


def test_retry_and_permanent_error(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_AGENT_URL', 'https://example.invalid/v1')
    monkeypatch.setenv('TEST_AGENT_KEY', 'secret-value')
    for folder in ('requests', 'responses', 'attempts'):
        (tmp_path/'demo'/folder).mkdir(parents=True)
    for i in range(200):
        runner.save(tmp_path/'demo/requests'/f'{i:04d}.json', {'model':'demo', 'index':i})
        if i > 1:
            runner.save(tmp_path/'demo/responses'/f'{i:04d}.json', {'content':'cached'})
    calls = {0:0, 1:0}
    async def handler(request):
        i = json.loads(request.content)['index']
        calls[i] += 1
        if i == 1:
            return httpx.Response(400)
        if calls[i] == 1:
            raise httpx.ReadTimeout('secret-value')
        return httpx.Response(200, json={'choices':[{'message':{'content':'SELECTED_INDEX: 2'}}]})
    assert asyncio.run(runner.run(tmp_path, {'config':config(2)}, httpx.MockTransport(handler))) == 2
    assert calls == {0:2, 1:1}
    assert runner.read(tmp_path/'status.json')['api_errors'] == 1
    assert 'secret-value' not in ''.join(p.read_text() for p in tmp_path.rglob('*.json'))

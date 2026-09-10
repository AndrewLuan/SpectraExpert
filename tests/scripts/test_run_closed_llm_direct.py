"""功能概述：验证纯 LLM 闭库运行器的 API 并发、发送节流和断点续跑契约。
主要职责：FakeClient 模拟慢速成功响应；test_concurrency_and_resume 验证每 key 最多
五个在途请求及成功样本不重复调用；test_pacing 验证同 key 发送间隔；
test_small_batch_uses_all_keys 验证八条请求交错分给三个 key；解析测试防止正文误判。
代码库关系：直接导入 scripts.run_closed_llm_direct，使用 pytest 的临时目录与 monkeypatch；
不访问外部网络、不使用真实 key、不读取数据集，也不运行 GED 搜索。
"""
import asyncio
import json
import time
from collections import defaultdict

from scripts import run_closed_llm_direct as runner


class FakeClient:
    active = defaultdict(int)
    peak = defaultdict(int)
    sends = defaultdict(list)

    def __init__(self, **kwargs):
        self.key = kwargs['headers']['Authorization']

    async def post(self, path, json):
        assert path == '/chat/completions'
        self.active[self.key] += 1
        self.peak[self.key] = max(self.peak[self.key], self.active[self.key])
        self.sends[self.key].append(time.monotonic())
        await asyncio.sleep(0.03)
        self.active[self.key] -= 1

        class Response:
            status_code = 200

            def json(self):
                return {'choices': [{'message': {'content': 'PREDICTED_SMILES: CCO'},
                                     'finish_reason': 'stop'}]}
        return Response()

    async def aclose(self):
        pass


def setup_client(monkeypatch, tmp_path):
    FakeClient.active.clear()
    FakeClient.peak.clear()
    FakeClient.sends.clear()
    monkeypatch.setattr(runner.httpx, 'AsyncClient', FakeClient)
    (tmp_path / 'responses').mkdir()


def test_concurrency_and_resume(monkeypatch, tmp_path):
    setup_client(monkeypatch, tmp_path)
    keys = ['fake-a', 'fake-b', 'fake-c']
    requests = [{} for _ in range(41)]
    result = asyncio.run(runner.infer(keys, requests, tmp_path, interval=0))
    assert len(result) == 41
    assert list(FakeClient.peak.values()) == [5, 5, 5]
    assert len(list((tmp_path / 'responses').glob('*.json'))) == 41
    second = asyncio.run(runner.infer(keys, requests, tmp_path, interval=0))
    assert second == result
    assert sum(map(len, FakeClient.sends.values())) == 41
    # 已失败请求允许重试，但原始失败记录必须归档。
    path = tmp_path / 'responses/0000.json'
    path.write_text(json.dumps({'error': 'HTTPError', 'http_status': 429}))
    asyncio.run(runner.infer(keys, requests, tmp_path, interval=0))
    assert sum(map(len, FakeClient.sends.values())) == 42
    assert len(list((tmp_path / 'failed_attempts').glob('*.json'))) == 1


def test_pacing(monkeypatch, tmp_path):
    setup_client(monkeypatch, tmp_path)
    asyncio.run(runner.infer(['fake-a'], [{}] * 8, tmp_path, interval=0.02))
    times = FakeClient.sends['Bearer fake-a']
    assert all(b - a >= 0.019 for a, b in zip(times, times[1:]))


def test_small_batch_uses_all_keys(monkeypatch, tmp_path):
    setup_client(monkeypatch, tmp_path)
    asyncio.run(runner.infer(['fake-a', 'fake-b', 'fake-c'], [{}] * 8, tmp_path, interval=0))
    assert [len(FakeClient.sends['Bearer ' + k]) for k in ('fake-a', 'fake-b', 'fake-c')] == [3, 3, 2]
    assert max(FakeClient.peak.values()) <= 5


def test_strict_field_and_group_identity():
    assert runner.strict_parse('PREDICTED_SMILES: CCO') == 'CCO'
    assert runner.strict_parse('Example CCO without a final answer') is None
    assert runner.strict_parse('PREDICTED_SMILES: UNKNOWN') is None
    assert runner.group_identity('[13CH3][15NH2]') == runner.group_identity('CN')

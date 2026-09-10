"""功能概述：检查候选索引选择与完整 API 用量留档汇总的关键契约。
主要职责：拒绝越界、多次选择及非整数输出；合成重试响应验证 usage 累计与缺失计数；
模拟 HTTP 响应核对完整供应商字段留存，测试不发真实网络请求。
代码库关系：调用 run_raman_candidate_llm 的无网络函数，不读取真实密钥或实验数据。
"""
import json
import asyncio
import httpx
from scripts.run_raman_candidate_llm import parse_choice, usage_report, infer
from scripts.normalize_raman_candidate_choices import parse_markdown_choice


def test_choice_requires_one_valid_integer():
    assert parse_choice('Reason\nSELECTED_INDEX: 10\nCONFIDENCE: 0.4') == 9
    for text in ('SELECTED_INDEX: 11', 'SELECTED_INDEX: 0', 'SELECTED_INDEX: 1.5',
                 'SELECTED_INDEX: 1\nSELECTED_INDEX: 2', 'I select 1'):
        assert parse_choice(text) is None


def test_markdown_choice_keeps_unique_index_contract():
    assert parse_markdown_choice('**SELECTED_INDEX: 3**  \n**CONFIDENCE: 0.85**') == 2
    assert parse_markdown_choice('SELECTED_INDEX: `10`') == 9
    assert parse_markdown_choice('**SELECTED_INDEX: 2**\nSELECTED_INDEX: 3') is None
    assert parse_markdown_choice('**SELECTED_INDEX: 11**') is None


def test_usage_keeps_retries_and_missing(tmp_path):
    (tmp_path / 'attempts').mkdir()
    for i, body in enumerate(({'usage': {'prompt_tokens': 10, 'completion_tokens': 5,
                                       'completion_tokens_details': {'reasoning_tokens': 2}}},
                               {'usage': {'prompt_tokens': 11, 'completion_tokens': 6}}, {'error': 'rate limit'})):
        (tmp_path / 'attempts' / f'{i}.json').write_text(json.dumps(dict(request_index=0, key_slot=1, response_json=body)))
    usage_report(tmp_path, [dict(method='model')])
    bucket = json.loads((tmp_path / 'usage_summary.json').read_text())['buckets']['all']
    assert bucket['attempts'] == 3 and bucket['attempts_with_usage'] == 2
    assert bucket['numeric_usage_totals']['prompt_tokens'] == 21
    assert bucket['numeric_usage_totals']['completion_tokens_details.reasoning_tokens'] == 2


def test_inference_preserves_full_response(monkeypatch, tmp_path):
    body = dict(model='test-model', id='response-id', extra_vendor_field={'a': [1, 2]},
        usage={'prompt_tokens': 7, 'completion_tokens': 3},
        choices=[dict(finish_reason='stop', message=dict(content='SELECTED_INDEX: 2', reasoning_content='reason'))])
    class Client:
        def __init__(self, **kwargs):
            pass
        async def post(self, *args, **kwargs):
            return httpx.Response(200, json=body)
        async def aclose(self):
            pass
    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    for name in ('responses', 'attempts', 'requests'):
        (tmp_path / name).mkdir()
    asyncio.run(infer(['test-secret'], [{'model': 'test-model', 'messages': []}], tmp_path))
    records = list((tmp_path / 'attempts').glob('*.json'))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record['response_json'] == body
    assert json.loads(record['response_text']) == body
    assert record['status'] == 'received'
    assert 'test-secret' not in records[0].read_text()

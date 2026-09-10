"""功能概述：冻结最新 encoder 与原始余弦 kNN 的 Top10，让同一 LLM 根据实验峰表选择。
主要职责：prepare 校验候选、测试身份和旧 baseline 参数并生成交错的 400 个请求；
infer 统一管理三个密钥的并发/节流，逐次保存完整响应正文和 JSON（含 usage/推理）；
parse_choice 严格解析唯一索引；score 映射到候选完整结构并复用既有结构评分；
usage_report 汇总所有已报告 token，包括重试，不把缺失 usage 当零。
代码库关系：消费 raman_top10_experiment/final 的冻结排名，复用 run_closed_llm_direct
的峰检测、原指标与序列化。选择输出不生成新结构，不使用知识库或修改回退。
输入输出与副作用：仅向已授权 USTC API 发送实验峰表和候选 SMILES；密钥从 stdin 单行
JSON 读入内存，不写认证头。错误正文若回显密钥则替换并标记；其余完整响应保留。
输出新目录；恢复调用前校验所有输入/请求/源码哈希，跳过已有成功响应，历史重试永不覆盖。
--model 可选择用户指定别名；--reuse-from 在用户确认同模型别名时复用成功输出，保留每条
原始请求模型名，并把所有旧尝试连同其原请求路径导入，以连续核算用量。
"""
import argparse
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import httpx
import numpy as np
from scripts import run_closed_llm_direct as base
from scripts.repair_closed_raman_exact import canonical

SYSTEM = '''You are an expert in Raman spectroscopy and molecular structure identification.
Select exactly ONE molecule from the provided candidate list using the experimental Raman peaks.
Candidates are ordered by decreasing retrieval similarity. This ordering is a useful prior, not a guarantee.
Use the peak evidence and structural compatibility; theoretical and experimental Raman spectra may differ.
Do not invent, modify, or combine candidates. Even when uncertain, choose the best available candidate.
End your response with these fields:
SELECTED_INDEX: <one integer from 1 to 10>
CONFIDENCE: <number from 0 to 1>
EXPLANATION: <brief justification referring to spectral evidence>
'''


def parse_choice(text):
    matches = re.findall(r'^\s*(?:\*\*)?SELECTED_INDEX(?:\*\*)?\s*:\s*`?(\d+)`?\s*$', text, re.M | re.I)
    if len(matches) != 1:
        return None
    index = int(matches[0])
    return index - 1 if 1 <= index <= 10 else None


def prepare(args):
    out = args.out
    protocol = json.loads((args.run / 'protocol.json').read_text())
    selection = json.loads((args.run / 'final/selection.json').read_text())
    assert selection['protocol_sha256'] == base.digest(args.run / 'protocol.json')
    checkpoint = args.run / selection['selected_arm'] / 'best.pt'
    assert base.digest(checkpoint) == selection['summaries'][selection['selected_arm']]['checkpoint_sha256']
    root = Path(protocol['data_root'])
    queries, spectra, _ = base.load_queries(root, protocol['sample_manifest_sha256'])
    final = json.loads((args.run / 'final/per_query.json').read_text())
    assert final['queries'] == queries
    names = final['candidates']
    assert names == protocol['candidate_names'][:10000]
    assert len(names) == len(set(names)) == 10000
    old = json.loads((args.baseline / 'config.json').read_text())
    assert (old['model'], old['temperature'], old['max_tokens'], old['endpoint']) == (base.MODEL, .2, 4096, base.ENDPOINT)
    assert old['manifest_sha256'] == protocol['sample_manifest_sha256']
    detector = base._PeakDetector({'stage1': {'data_dir': str(root / 'experimental_test')}})
    assert np.array_equal(detector._wavenumber, np.arange(500, 4001))
    # 对所有 200 条核对原始 direct baseline 的峰提示，确保峰提取和输入谱未发生变化。
    for i, spectrum in enumerate(spectra):
        previous = json.loads((args.baseline / 'requests' / f'{i:04d}.json').read_text())
        assert previous['messages'] == base.make_messages(spectrum, detector)[0]
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('requests', 'responses', 'attempts'):
        (out / folder).mkdir()
    tasks = []
    for i, spectrum in enumerate(spectra):
        peaks = detector.detect(spectrum)
        table = base.format_peak_table(peaks)
        for method in ('model', 'raw_knn'):
            indices = final['methods'][method]['top10'][i]
            assert len(indices) == len(set(indices)) == 10
            candidates = [names[j] for j in indices]
            rank = final['methods'][method]['ranks'][i]
            assert (canonical(queries[i]['ground_truth_smiles']) in candidates) == (rank <= 10)
            candidate_text = '\n'.join(f'{j+1}. {s}' for j, s in enumerate(candidates))
            prompt = f'## Experimental Raman peaks ({len(peaks)} peaks)\n{table}\n\n## Ten candidate SMILES (retrieval order)\n{candidate_text}\n\nSelect exactly one candidate from this list.'
            request = dict(model=args.model, temperature=old['temperature'], max_tokens=old['max_tokens'],
                messages=[dict(role='system', content=SYSTEM), dict(role='user', content=prompt)])
            index = len(tasks)
            base.save(out / 'requests' / f'{index:04d}.json', request)
            tasks.append(dict(request_index=index, method=method, query_index=i, query_id=queries[i]['mol_id'],
                candidates=candidates, candidate_indices=indices, true_rank=rank, peak_count=len(peaks)))
    base.save(out / 'tasks.json', tasks)
    base.save(out / 'queries.json', queries)
    reused = []
    if args.reuse_from:
        parent = args.reuse_from
        assert json.loads((parent / 'tasks.json').read_text()) == tasks
        assert json.loads((parent / 'queries.json').read_text()) == queries
        for f in sorted((parent / 'attempts').glob('*.json')):
            record = json.loads(f.read_text())
            record['request_artifact'] = str((parent / 'requests' / f"{record['request_index']:04d}.json").resolve())
            record['imported_from'] = str(f.resolve())
            base.save(out / 'attempts' / f.name, record)
        for f in sorted((parent / 'responses').glob('*.json')):
            response = json.loads(f.read_text())
            if response.get('error') or 'content' not in response:
                continue
            previous = json.loads((parent / 'requests' / f.name).read_text())
            current = json.loads((out / 'requests' / f.name).read_text())
            assert {k:v for k,v in previous.items() if k != 'model'} == {k:v for k,v in current.items() if k != 'model'}
            assert previous['model'] in ('deepseek-v4-flash', 'deepseek-v4-flash-ascend')
            shutil.copy2(parent / 'requests' / f.name, out / 'requests' / f.name)
            shutil.copy2(f, out / 'responses' / f.name)
            reused.append(int(f.stem))
    repo = Path(__file__).resolve().parents[1]
    sources = ['scripts/run_raman_candidate_llm.py', 'scripts/run_closed_llm_direct.py',
               'spectraexpert/eval/evaluate.py', 'spectraexpert/eval/metrics.py']
    base.save(out / 'config.json', dict(schema='raman_candidate_selection_v1', model=args.model,
        reused_request_indices=reused, parent_run=str(args.reuse_from.resolve()) if args.reuse_from else None,
        alias_policy='User states flash and flash-ascend are the same model; original request and returned model names retained per call',
        endpoint=old['endpoint'], temperature=.2, max_tokens=4096, checkpoint_sha256=base.digest(checkpoint),
        retrieval_run=str(args.run.resolve()), raw_baseline=str(args.baseline.resolve()), data_root=str(root),
        manifest_sha256=protocol['sample_manifest_sha256'], selection='single index 1..10; no fallback on invalid index',
        input='experimental peak table + ranked candidate SMILES; no scores, candidate spectra, knowledge base, or truth',
        concurrency_per_key=5, interval_per_key=3.2, max_attempts=5,
        files_sha256={str(f.relative_to(out)): base.digest(f) for f in sorted(out.glob('requests/*.json'))}
            | {'tasks.json': base.digest(out / 'tasks.json'), 'queries.json': base.digest(out / 'queries.json')},
        source_sha256={f: base.digest(repo / f) for f in sources},
        source_results_sha256=base.digest(args.run / 'final/per_query.json'),
        raw_baseline_results_sha256=base.digest(args.baseline / 'results.json')))
    print(json.dumps(dict(prepared_requests=len(tasks), encoder_ceiling=sum(t['true_rank'] <= 10 for t in tasks if t['method'] == 'model'),
        knn_ceiling=sum(t['true_rank'] <= 10 for t in tasks if t['method'] == 'raw_knn'), reused=len(reused))), flush=True)


def load(out):
    config = json.loads((out / 'config.json').read_text())
    for name, sha in config['files_sha256'].items():
        assert base.digest(out / name) == sha, name
    repo = Path(__file__).resolve().parents[1]
    for name, sha in config['source_sha256'].items():
        assert base.digest(repo / name) == sha, name
    tasks = json.loads((out / 'tasks.json').read_text())
    requests = [json.loads((out / 'requests' / f'{i:04d}.json').read_text()) for i in range(len(tasks))]
    return config, tasks, requests


async def infer(keys, requests, out):
    queue = asyncio.Queue()
    for i, request in enumerate(requests):
        f = out / 'responses' / f'{i:04d}.json'
        if f.exists() and 'content' in json.loads(f.read_text()) and not json.loads(f.read_text()).get('error'):
            continue
        queue.put_nowait((i, request))
    locks = [asyncio.Lock() for _ in keys]
    next_send = [0.] * len(keys)
    async def worker(slot, client):
        while not queue.empty():
            i, request = queue.get_nowait()
            result = dict(query_index=i, key_slot=slot+1)
            for attempt in range(5):
                async with locks[slot]:
                    while next_send[slot] > time.monotonic():
                        await asyncio.sleep(next_send[slot] - time.monotonic())
                    next_send[slot] = time.monotonic() + 3.2
                started = time.time()
                record = dict(request_index=i, key_slot=slot+1, attempt_in_run=attempt+1, started_at=started,
                    request_artifact=str((out / 'requests' / f'{i:04d}.json').resolve()), request_model=request['model'])
                path = out / 'attempts' / f'{i:04d}_{time.time_ns()}.json'
                record['status'] = 'in_flight'
                base.save(path, record)
                retry = False; delay = 0.
                try:
                    resp = await client.post('/chat/completions', json=request)
                    body_text = resp.text
                    for key in keys:
                        body_text = body_text.replace(key, '[REDACTED_API_KEY]')
                    record.update(http_status=resp.status_code, response_text=body_text,
                        credential_redacted=body_text != resp.text,
                        response_headers={k: v for k, v in resp.headers.items() if k.lower() in
                            ('content-type', 'date', 'x-request-id', 'request-id', 'retry-after')})
                    try:
                        body = json.loads(body_text)
                        record['response_json'] = body
                    except ValueError:
                        body = None
                    if resp.status_code == 200 and isinstance(body, dict) and body.get('choices'):
                        choice = body['choices'][0]
                        result.update(content=choice['message'].get('content') or '',
                            reasoning_content=choice['message'].get('reasoning_content'), usage=body.get('usage'),
                            response_model=body.get('model'), response_id=body.get('id'), finish_reason=choice.get('finish_reason'))
                        result.pop('error', None)
                    else:
                        result['error'] = f'HTTP_{resp.status_code}' if resp.status_code != 200 else 'MalformedResponse'
                        retry = resp.status_code in (429, 500, 502, 503, 504)
                        if retry:
                            try:
                                delay = max(60., float(resp.headers.get('retry-after', '60'))) if resp.status_code == 429 else 2**(attempt+1)
                            except ValueError:
                                delay = 60.
                except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                    record['exception_type'] = type(exc).__name__
                    result['error'] = type(exc).__name__
                    retry = True; delay = 2**(attempt+1)
                record['elapsed_seconds'] = time.time() - started
                record['status'] = 'received' if 'http_status' in record else 'transport_error'
                base.save(path, record)
                result['last_attempt_file'] = str(path.relative_to(out))
                if 'content' in result or not retry or attempt == 4:
                    break
                next_send[slot] = max(next_send[slot], time.monotonic() + delay)
            base.save(out / 'responses' / f'{i:04d}.json', result)
            print(json.dumps(dict(request_index=i, status=result.get('error', 'ok'), key_slot=slot+1)), flush=True)
            queue.task_done()
    clients = [httpx.AsyncClient(base_url=base.ENDPOINT, headers={'Authorization': 'Bearer '+key},
        timeout=httpx.Timeout(180, connect=20), follow_redirects=False) for key in keys]
    try:
        await asyncio.gather(*(worker(slot, client) for _ in range(5) for slot, client in enumerate(clients)))
    finally:
        for client in clients:
            await client.aclose()


def usage_report(out, tasks):
    bins = defaultdict(lambda: dict(attempts=0, attempts_with_usage=0, numeric_usage_totals={}))
    def flatten(value, prefix=''):
        result = {}
        for key, item in value.items():
            name = prefix + key
            if isinstance(item, dict):
                result.update(flatten(item, name+'.'))
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                result[name] = item
        return result
    for f in sorted((out / 'attempts').glob('*.json')):
        r = json.loads(f.read_text()); body = r.get('response_json')
        usage = body.get('usage') if isinstance(body, dict) else None
        keys = ['all', 'method:'+tasks[r['request_index']]['method'], 'key_slot:'+str(r['key_slot'])]
        for key in keys:
            bucket = bins[key]; bucket['attempts'] += 1
            if isinstance(usage, dict):
                bucket['attempts_with_usage'] += 1
                for name, value in flatten(usage).items():
                    bucket['numeric_usage_totals'][name] = bucket['numeric_usage_totals'].get(name, 0) + value
    base.save(out / 'usage_summary.json', dict(buckets=dict(bins),
        note='Totals include every response reporting usage, including retries. Missing usage is unavailable, not zero. Interrupted parent requests without received responses have unknown usage, so totals are reported usage only. Nested token fields may overlap; do not add them together. No inferred pricing.'))


def score(out, config, tasks):
    queries, _, groups = base.load_queries(Path(config['data_root']), config['manifest_sha256'])
    methods = {}; selection_summary = {}
    for method in ('model', 'raw_knn'):
        responses = []; details = []
        for t in [t for t in tasks if t['method'] == method]:
            r = json.loads((out / 'responses' / f"{t['request_index']:04d}.json").read_text())
            choice = parse_choice(r.get('content', '')) if not r.get('error') else None
            smiles = t['candidates'][choice] if choice is not None else None
            resolved = dict(content='PREDICTED_SMILES: '+(smiles or 'UNKNOWN'))
            if r.get('error'):
                resolved['error'] = r['error']
            responses.append(resolved)
            correct = smiles == canonical(queries[t['query_index']]['ground_truth_smiles'])
            details.append(dict(t, selected_index=choice+1 if choice is not None else None,
                predicted_smiles=smiles, exact=correct, api_error=r.get('error')))
        folder = out / method; (folder / 'metrics').mkdir(parents=True, exist_ok=True)
        base.save(folder / 'selections.json', details)
        n = len(details); inside = sum(r['true_rank'] <= 10 for r in details)
        hits = sum(r['exact'] for r in details)
        selection_summary[method] = dict(n=n, valid_choices=sum(r['selected_index'] is not None for r in details),
            retrieval_top10_count=inside, final_exact_count=hits, final_exact_rate=hits/n,
            selection_accuracy_given_retrieved=hits/inside if inside else None,
            lost_retrieved=inside-hits, picked_rank1=sum(r['selected_index'] == 1 for r in details))
        methods[method] = (responses, folder)
    base.save(out / 'selection_summary.json', selection_summary)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(base.evaluate, queries, responses, groups, folder) for responses, folder in methods.values()]
        for future in futures:
            future.result()
    report = {m: json.loads((folder / 'results.json').read_text())['summary'] for m, (_, folder) in methods.items()}
    baseline = json.loads((Path(config['raw_baseline']) / 'results.json').read_text())
    assert base.digest(Path(config['raw_baseline']) / 'results.json') == config['raw_baseline_results_sha256']
    report['raw_llm'] = baseline['summary']
    base.save(out / 'comparison.json', report)
    print(json.dumps(selection_summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'run', 'score'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--run', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--model', default=base.MODEL, choices=('deepseek-v4-flash', 'deepseek-v4-flash-ascend'))
    p.add_argument('--reuse-from', type=Path)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(args); return
    config, tasks, requests = load(args.out)
    code = 1
    try:
        if args.mode == 'run':
            keys = json.loads(sys.stdin.readline())['keys']
            assert 1 <= len(keys) <= 3 and len(keys) == len(set(keys)) and all(keys)
            asyncio.run(infer(keys, requests, args.out)); keys.clear()
            (args.out / 'inference_finished_at').write_text(str(time.time()))
        usage_report(args.out, tasks)
        score(args.out, config, tasks)
        code = 0 if all(not json.loads(f.read_text()).get('error') for f in (args.out / 'responses').glob('*.json')) else 2
    finally:
        (args.out / f'exit_code_{time.time_ns()}').write_text(str(code))
        (args.out / 'exit_code').write_text(str(code))
    raise SystemExit(code)


if __name__ == '__main__':
    main()

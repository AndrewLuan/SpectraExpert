"""功能概述：在 JJ 上重放当前 Test-200 的冻结知识增强 Agent 请求，仅替换模型。
主要职责：prepare 校验源请求哈希并冻结多模型请求；run 用全局工作队列限制并发，
保存尝试、响应及断点状态；score 调用现有候选解析和结构评分；main 提供三阶段入口。
代码库关系：消费 run_raman_encoder_kb_llm 的 encoder_kb_llm_paired_v1 产物，
评分复用 normalize_raman_candidate_choices 与 run_closed_llm_direct，不重建知识库。
输入输出与副作用：配置只存 base_url_env/api_key_env 的环境变量名；运行时读取服务地址
和密钥。最多 200 个全局 worker，每次 HTTP 调用及连接/读写/连接池超时均大于 300 秒。
成功响应不重复请求；传输/429/5xx 有限重试，格式或化学错误不重问；原始响应脱敏落盘。
运行锁阻止两个进程重复调用同一结果目录；评分在推理完成后单独执行，失败保留总分母。
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import httpx


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(config):
    if type(config['concurrency']) is not int or not 1 <= config['concurrency'] <= 200:
        raise ValueError('concurrency must be an integer in 1..200')
    if not math.isfinite(config['timeout_seconds']) or config['timeout_seconds'] <= 300:
        raise ValueError('timeout_seconds must be finite and >300')
    if type(config['max_attempts']) is not int or not 1 <= config['max_attempts'] <= 5:
        raise ValueError('max_attempts must be in 1..5')
    tags = [m['tag'] for m in config['models']]
    if not tags or len(tags) != len(set(tags)):
        raise ValueError('models must have unique tags')
    import re
    for model in config['models']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', model['tag']):
            raise ValueError('unsafe model tag')
        if not model['model'] or model['model'].startswith('SET_'):
            raise ValueError('set exact API model IDs before prepare')
        if set(model.get('request_overrides', {})) - {'temperature', 'max_tokens', 'max_completion_tokens', 'reasoning_effort', 'top_p'}:
            raise ValueError('unsupported request override')


def prepare(config_path, out):
    config = read(config_path)
    validate(config)
    source = Path(config['source_run'])
    original = read(source / 'config.json')
    if original['schema'] != 'encoder_kb_llm_paired_v1':
        raise ValueError('requires frozen encoder+KB Agent source')
    tasks, queries = read(source / 'tasks.json'), read(source / 'queries.json')
    if len(tasks) != 200 or len(queries) != 200 or [t['query_index'] for t in tasks] != list(range(200)):
        raise ValueError('requires complete ordered Test-200')
    names = ['tasks.json', 'queries.json'] + [f"requests/{t['request_index']:04d}.json" for t in tasks]
    for name in names:
        if digest(source / name) != original['files_sha256'][name]:
            raise ValueError('source hash mismatch: ' + name)
    if any(t['method'] != 'model' or len(t['candidates']) != 10 for t in tasks):
        raise ValueError('requires encoder Top10 only')
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'tasks.json', tasks)
    save(out / 'queries.json', queries)
    for model in config['models']:
        folder = out / model['tag']
        for sub in ('requests', 'responses', 'attempts'):
            (folder / sub).mkdir(parents=True)
        for i, task in enumerate(tasks):
            request = copy.deepcopy(read(source / 'requests' / f"{task['request_index']:04d}.json"))
            request['model'] = model['model']
            for key, value in model.get('request_overrides', {}).items():
                if value is None:
                    request.pop(key, None)
                else:
                    request[key] = value
            save(folder / 'requests' / f'{i:04d}.json', request)
    save(out / 'protocol.json', dict(config=config, source_config=original,
        source_config_sha256=digest(source / 'config.json'),
        runner_sha256=digest(Path(__file__)),
        files_sha256={str(p.relative_to(out)): digest(p) for p in out.rglob('*.json')}))


def load(out):
    protocol = read(out / 'protocol.json')
    validate(protocol['config'])
    if digest(Path(__file__)) != protocol['runner_sha256']:
        raise ValueError('runner changed since preparation')
    for name, sha in protocol['files_sha256'].items():
        if digest(out / name) != sha:
            raise ValueError('frozen input changed: ' + name)
    return protocol


async def run(out, protocol, transport=None):
    config = protocol['config']
    queue = asyncio.Queue()
    clients = {}
    secrets = []
    endpoints = {}
    # 先验证所有模型配置，避免部分启动后才发现缺少凭据。
    for model in config['models']:
        url = os.environ[model['base_url_env']].rstrip('/')
        key = os.environ[model['api_key_env']]
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or not key:
            raise ValueError('requires HTTPS base URL without credentials/query and nonempty key')
        endpoints[model['tag']] = url
        secrets.append(key)
    route_file = out / 'endpoints.json'
    if route_file.exists() and read(route_file) != endpoints:
        raise ValueError('endpoint changed; prepare a new experiment directory')
    save(route_file, endpoints)
    try:
        for model in config['models']:
            clients[model['tag']] = httpx.AsyncClient(
                headers={'Authorization': 'Bearer ' + os.environ[model['api_key_env']]},
                timeout=httpx.Timeout(config['timeout_seconds']),
                limits=httpx.Limits(max_connections=config['concurrency'], max_keepalive_connections=config['concurrency']),
                follow_redirects=False, trust_env=False, transport=transport)
        for i in range(200):
            for model in config['models']:
                folder = out / model['tag']
                response = folder / 'responses' / f'{i:04d}.json'
                if response.exists() and not read(response).get('error') and isinstance(read(response).get('content'), str):
                    continue
                queue.put_nowait((model['tag'], i))

        async def worker():
            while not queue.empty():
                tag, i = queue.get_nowait()
                folder = out / tag
                request = read(folder / 'requests' / f'{i:04d}.json')
                for attempt in range(config['max_attempts']):
                    started = time.monotonic()
                    log = folder / 'attempts' / f'{i:04d}_{time.time_ns()}.json'
                    record = dict(query_index=i, model=request['model'], started_at=time.time(), state='in_flight')
                    save(log, record)
                    retry, delay = False, min(2 ** (attempt + 1), 30)
                    result = dict(query_index=i)
                    try:
                        async with asyncio.timeout(config['timeout_seconds']):
                            response = await clients[tag].post(endpoints[tag] + '/chat/completions', json=request)
                        raw = response.text
                        for key in secrets:
                            raw = raw.replace(key, '[REDACTED]')
                        record.update(http_status=response.status_code, response_text=raw)
                        if response.status_code != 200:
                            result['error'] = f'HTTP_{response.status_code}'
                            retry = response.status_code == 429 or 500 <= response.status_code < 600
                            try:
                                delay = max(delay, float(response.headers.get('retry-after', '0')))
                                if not math.isfinite(delay):
                                    delay = 60
                            except ValueError:
                                delay = 60
                        else:
                            body = json.loads(raw)
                            choice = body['choices'][0]
                            content = choice['message']['content']
                            if not isinstance(content, str):
                                raise ValueError('non-text content')
                            result.update(content=content, usage=body.get('usage'), response_model=body.get('model'),
                                finish_reason=choice.get('finish_reason'), response_id=body.get('id'))
                    except (httpx.HTTPError, TimeoutError) as exc:
                        result['error'] = type(exc).__name__
                        retry = True
                    except (ValueError, KeyError, IndexError, TypeError):
                        result['error'] = 'MalformedResponse'
                    record.update(state='finished', elapsed_seconds=time.monotonic()-started, error=result.get('error'))
                    save(log, record)
                    result['last_attempt_file'] = str(log.relative_to(out))
                    save(folder / 'responses' / f'{i:04d}.json', result)
                    if not retry or attempt + 1 == config['max_attempts']:
                        break
                    await asyncio.sleep(delay)
                print(json.dumps(dict(model=tag, query_index=i, status=result.get('error', 'ok'))), flush=True)
                queue.task_done()
        await asyncio.gather(*(worker() for _ in range(config['concurrency'])))
    finally:
        for client in clients.values():
            await client.aclose()
    errors = sum(bool(read(out/m['tag']/'responses'/f'{i:04d}.json').get('error')) for m in config['models'] for i in range(200))
    save(out / 'status.json', dict(state='complete' if not errors else 'inference_failed', api_errors=errors))
    return 2 if errors else 0


def score(out, protocol):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import run_closed_llm_direct as base
    from scripts.normalize_raman_candidate_choices import parse_markdown_choice
    source = protocol['source_config']
    queries, _, groups = base.load_queries(Path(source['data_root']), source['manifest_sha256'])
    if queries != read(out/'queries.json'):
        raise ValueError('query mismatch')
    tasks = read(out/'tasks.json')
    for model in protocol['config']['models']:
        folder = out/model['tag']
        predictions, selections = [], []
        for i, task in enumerate(tasks):
            path = folder/'responses'/f'{i:04d}.json'
            response = read(path) if path.exists() else {'error': 'MissingResponse'}
            choice = None if response.get('error') else parse_markdown_choice(response.get('content', ''))
            smiles = task['candidates'][choice] if choice is not None else None
            predictions.append(dict(content='PREDICTED_SMILES: ' + (smiles or 'UNKNOWN'), error=response.get('error')))
            selections.append(dict(query_index=i, selected_index=choice+1 if choice is not None else None, predicted_smiles=smiles))
        scored = folder/'scored'
        (scored/'metrics').mkdir(parents=True, exist_ok=True)
        save(scored/'selections.json', selections)
        base.evaluate(queries, predictions, groups, scored)
        save(scored/'provenance.json', {str(Path(module.__file__).resolve()): digest(Path(module.__file__)) for module in (base, sys.modules[parse_markdown_choice.__module__], sys.modules['spectraexpert.eval.metrics'])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'score'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not args.config:
            parser.error('prepare requires --config')
        prepare(args.config, args.out)
        return
    with (args.out/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = load(args.out)
        if args.mode == 'run':
            raise SystemExit(asyncio.run(run(args.out, protocol)))
        score(args.out, protocol)


if __name__ == '__main__':
    main()

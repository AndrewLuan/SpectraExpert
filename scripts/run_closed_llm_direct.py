#!/usr/bin/env python3
"""功能概述：在冻结 Raman 10k/200 的实验谱上复现论文 peaks-only LLM Direct。
主要职责：load_queries 校验冻结文件哈希并按 mol_id 连接，默认原版哈希，新版调用方须显式传入；make_messages 复用原峰检测
和 prompt；infer 用每 key 五个固定 worker 调用 USTC；evaluate 计算严格字段与原解析器
两种结果、完整 canonical exact、分子组命中、Tanimoto 和版本化 GED；main 持久化审计。
代码库关系：复用 eval.evaluate 的 _PeakDetector/parse_llm_smiles 和 eval.metrics，
候选结构只在本地评估组身份，不进入 LLM；不使用 encoder、知识库或理论谱作提示。
输入输出与副作用：仅读指定冻结目录，1–3 个已授权密钥从 stdin JSON 获取并只驻留内存；向指定 HTTPS
服务发送峰表。每条请求与响应独立落盘，错误只记录类型/状态码，永不记录认证信息。
默认输出目录必须不存在；--resume 核对全部请求后跳过成功响应、归档失败响应。
worker 按 key 交错启动，少量补跑也分摊到所有 key；每 key 发送间隔 3.2 秒（低于实测 20 RPM），429 按 Retry-After 共享冷却；最多五次尝试，
结构无效不重试。evaluate 使用四个线程调度隔离 GED worker；exit_code 区分失败。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import httpx
import numpy as np
import pandas as pd
from rdkit import Chem
from spectraexpert.eval.evaluate import _PeakDetector, parse_llm_smiles
from spectraexpert.eval.metrics import graph_edit_distance, tanimoto_similarity
from spectraexpert.pipeline.prompts.stage3_direct_identification import (
    DIRECT_ID_SYSTEM_PROMPT, DIRECT_ID_PEAKS_ONLY_PROMPT, format_peak_table,
)

MANIFEST_SHA = '9262856ad71db4802ca103261e91507f23969afab83cadf5f4ef7c64899a2ecb'
MODEL = 'deepseek-v4-flash'
ENDPOINT = 'https://api.llm.ustc.edu.cn/v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def group_identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True)
    largest = max(fragments, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()))
    Chem.RemoveStereochemistry(largest)
    for atom in largest.GetAtoms():
        atom.SetIsotope(0)
    return Chem.MolToSmiles(largest)


def strict_parse(text):
    match = re.search(r'^\s*(?:\*\*)?PREDICTED_SMILES(?:\*\*)?\s*:\s*(\S+)', text, re.M | re.I)
    if not match:
        return None
    value = match.group(1).strip('`\"\'').rstrip('.')
    mol = Chem.MolFromSmiles(value) if value.upper() != 'UNKNOWN' else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def load_queries(root, expected_manifest_sha=MANIFEST_SHA):
    """校验指定版本的候选身份与实验输入；默认绑定原版，新版调用方显式提供 hash。"""
    # 新版闭库调用方必须显式提供已验证的 manifest hash；旧入口仍锁定原版。
    assert digest(root / 'sample_manifest.json') == expected_manifest_sha
    manifest = json.loads((root / 'sample_manifest.json').read_text())
    assert manifest['schema_version'] == 'raman_closed_library_sample_v1'
    # 只校验使用到的实验文件与候选身份；不加载 10k 理论谱。
    for role, names in [('experimental_test', ['spectra.parquet', 'smiles.parquet', 'wavenumber.parquet', 'rows.json']),
                        ('theory_train_candidate', ['smiles.parquet', 'rows.json'])]:
        for name in names:
            assert digest(root / role / name) == manifest['outputs_sha256'][role][name], (role, name)
    candidate = pd.read_parquet(root / 'theory_train_candidate/smiles.parquet')
    truth = pd.read_parquet(root / 'experimental_test/smiles.parquet').set_index('mol_id')
    spectra = pd.read_parquet(root / 'experimental_test/spectra.parquet').set_index('mol_id')
    rows = json.loads((root / 'experimental_test/rows.json').read_text())
    ids = [r['mol_id'] for r in rows]
    assert len(candidate) == candidate.group_hash.nunique() == 10000
    assert len(ids) == len(set(ids)) == len(truth) == len(spectra) == 200
    assert truth.index.is_unique and spectra.index.is_unique
    assert set(ids) == set(truth.index) == set(spectra.index)
    assert [r['group_hash'] for r in rows] == manifest['test_groups']
    lookup = candidate.set_index('mol_id')
    groups = {}
    for r in candidate.itertuples():
        identity = group_identity(r.smiles)
        assert identity is not None
        if identity in groups:
            assert groups[identity] == r.group_hash, 'ambiguous normalized candidate identity'
        groups[identity] = r.group_hash
    queries = []
    for r in rows:
        mid = r['mol_id']
        gt = truth.loc[mid]
        assert gt.group_hash == r['group_hash'] == lookup.loc[mid].group_hash
        assert '/exp/raman/' in gt.source_path
        assert groups[group_identity(gt.smiles)] == gt.group_hash
        queries.append({'mol_id': int(mid), 'ground_truth_smiles': gt.smiles, 'group_hash': gt.group_hash})
    columns = [f'w_{i}' for i in range(500, 4001)]
    array = spectra.loc[ids, columns].to_numpy(dtype=np.float32)
    assert np.isfinite(array).all()
    return queries, array, groups


def make_messages(spectrum, detector):
    peaks = detector.detect(spectrum)
    return [{'role': 'system', 'content': DIRECT_ID_SYSTEM_PROMPT},
            {'role': 'user', 'content': DIRECT_ID_PEAKS_ONLY_PROMPT.format(
                n_peaks=len(peaks), peak_table=format_peak_table(peaks))}], len(peaks)


async def infer(keys, requests, out, interval=3.2):
    queue = asyncio.Queue()
    results = [None] * len(requests)
    for index, request in enumerate(requests):
        old = out / 'responses' / f'{index:04d}.json'
        if old.exists():
            previous = json.loads(old.read_text())
            if 'content' in previous and 'error' not in previous:
                results[index] = previous
                continue
            archive = out / 'failed_attempts'
            archive.mkdir(exist_ok=True)
            old.rename(archive / f'{index:04d}_{time.time_ns()}.json')
        queue.put_nowait((index, request))
    locks = [asyncio.Lock() for _ in keys]
    next_send = [0.0] * len(keys)
    # 固定每 key 五个协程，无 SDK 隐式重试或额外后台请求。
    async def worker(key_index, client):
        while not queue.empty():
            index, request = queue.get_nowait()
            started = time.monotonic()
            result = {'query_index': index, 'key_slot': key_index + 1}
            for attempt in range(5):
                try:
                    async with locks[key_index]:
                        while next_send[key_index] > time.monotonic():
                            await asyncio.sleep(next_send[key_index] - time.monotonic())
                        next_send[key_index] = time.monotonic() + interval
                    resp = await client.post('/chat/completions', json=request)
                    result['http_status'] = resp.status_code
                    if resp.status_code != 200:
                        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                            delay = max(60.0, float(resp.headers.get('retry-after', 60))) if resp.status_code == 429 else 2 ** (attempt + 1)
                            next_send[key_index] = max(next_send[key_index], time.monotonic() + delay)
                            print(json.dumps({'query_index': index, 'http_status': resp.status_code,
                                              'cooldown_seconds': delay, 'key_slot': key_index + 1}), flush=True)
                            continue
                        result['error'] = 'HTTPError'
                        break
                    body = resp.json()
                    choice = body['choices'][0]
                    result.update({'response_model': body.get('model'), 'response_id': body.get('id'),
                                   'usage': body.get('usage'), 'finish_reason': choice.get('finish_reason'),
                                   'content': choice['message'].get('content') or '',
                                   'reasoning_content': choice['message'].get('reasoning_content')})
                    break
                except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
                    result['error'] = type(exc).__name__
                    if attempt < 4:
                        await asyncio.sleep(2 ** (attempt + 1))
                    else:
                        break
            if 'content' in result:
                result.pop('error', None)
            result.update(attempts=attempt + 1, time_seconds=round(time.monotonic() - started, 3))
            save(out / 'responses' / f'{index:04d}.json', result)
            results[index] = result
            print(json.dumps({'completed': sum(r is not None for r in results), 'total': len(results),
                              'query_index': index, 'key_slot': key_index + 1,
                              'status': result.get('error', 'ok'), 'finish_reason': result.get('finish_reason')}), flush=True)
            queue.task_done()
    clients = [httpx.AsyncClient(base_url=ENDPOINT, headers={'Authorization': 'Bearer ' + k},
                                timeout=httpx.Timeout(180, connect=20), follow_redirects=False) for k in keys]
    try:
        await asyncio.gather(*(worker(i, client) for _ in range(5) for i, client in enumerate(clients)))
    finally:
        for client in clients:
            await client.aclose()
    return results


def evaluate(queries, responses, groups, out):
    def one(item):
        i, (query, response) = item
        content = response.get('content', '')
        strict = strict_parse(content)
        legacy = parse_llm_smiles(content)
        gt = Chem.MolToSmiles(Chem.MolFromSmiles(query['ground_truth_smiles']))
        row = dict(query, query_index=i, api_error=response.get('error'), strict_smiles=strict, legacy_smiles=legacy)
        for label, predicted in [('strict', strict), ('legacy', legacy)]:
            group = groups.get(group_identity(predicted)) if predicted else None
            row[label] = {'valid': bool(predicted), 'exact_match': bool(predicted and predicted == gt),
                          'predicted_group_hash': group, 'group_match': bool(group == query['group_hash']),
                          'tanimoto': tanimoto_similarity(gt, predicted) if predicted else None}
        # 论文 legacy GED 是第一个优化解；chemical GED 要求搜索完成，超时保留 null。
        for version in ('legacy_ged_v1', 'chemical_ged_v2'):
            value = graph_edit_distance(gt, legacy, timeout=10.0, version=version) if legacy else None
            row['legacy'][version] = value
            row['strict'][version] = (value if strict == legacy else
                graph_edit_distance(gt, strict, timeout=10.0, version=version) if strict else None)
        save(out / 'metrics' / f'{i:04d}.json', row)
        return row
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(one, enumerate(zip(queries, responses))):
            results.append(row)
            if len(results) % 10 == 0:
                print(f'evaluated={len(results)}/{len(queries)}', flush=True)
    summary = {'n_queries': len(queries), 'api_errors': sum(bool(r['api_error']) for r in results)}
    for label in ('strict', 'legacy'):
        values = [r[label] for r in results]
        summary[label] = {name + '_rate': sum(v[name] for v in values) / len(values)
                          for name in ('valid', 'exact_match', 'group_match')}
        for metric in ('tanimoto', 'legacy_ged_v1', 'chemical_ged_v2'):
            observed = [v[metric] for v in values if v[metric] is not None]
            summary[label][metric] = {'n': len(observed), 'mean': float(np.mean(observed)) if observed else None,
                                      'median': float(np.median(observed)) if observed else None}
    save(out / 'results.json', {'summary': summary, 'queries': results})
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    queries, spectra, groups = load_queries(args.data_root)
    detector = _PeakDetector({'stage1': {'data_dir': str(args.data_root / 'experimental_test')}})
    assert np.array_equal(detector._wavenumber, np.arange(500, 4001))
    requests = []
    for spectrum in spectra:
        messages, _ = make_messages(spectrum, detector)
        requests.append({'model': MODEL, 'temperature': 0.2, 'max_tokens': 4096, 'messages': messages})
    if args.prepare_only:
        print(json.dumps({'validated_queries': len(queries), 'model': MODEL,
                          'first_peak_count': len(detector.detect(spectra[0]))}))
        return
    keys = json.load(sys.stdin)['keys']
    assert 1 <= len(keys) <= 3 and len(keys) == len(set(keys)) and all(isinstance(k, str) and k for k in keys)
    if args.resume:
        previous_config = json.loads((args.output_dir / 'config.json').read_text())
        assert previous_config['manifest_sha256'] == MANIFEST_SHA
        assert previous_config['query_ids'] == [q['mol_id'] for q in queries]
        for i, request in enumerate(requests):
            assert json.loads((args.output_dir / 'requests' / f'{i:04d}.json').read_text()) == request
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    out = args.output_dir
    for name in ('requests', 'responses', 'metrics'):
        (out / name).mkdir(exist_ok=args.resume)
    source_root = Path(__file__).resolve().parents[1]
    sources = ['scripts/run_closed_llm_direct.py', 'spectraexpert/eval/evaluate.py',
               'spectraexpert/eval/metrics.py', 'spectraexpert/pipeline/prompts/stage3_direct_identification.py']
    config_path = out / (f'resume_config_{time.time_ns()}.json' if args.resume else 'config.json')
    save(config_path, {'mode': 'llm-direct', 'model': MODEL, 'endpoint': ENDPOINT,
         'temperature': 0.2, 'max_tokens': 4096, 'key_count': len(keys), 'concurrency_per_key': 5,
         'request_interval_per_key_seconds': 3.2, 'max_attempts': 5, 'evaluation_workers': 4,
         'data_root': str(args.data_root), 'manifest_sha256': MANIFEST_SHA, 'query_ids': [q['mol_id'] for q in queries],
         'source_sha256': {p: digest(source_root / p) for p in sources},
         'peak_detection': {'height_rel': 0.05, 'prominence_rel': 0.02, 'distance': 5, 'width': 1, 'zscore': True},
         'group_identity': 'largest fragment by heavy atoms then atoms, ignore stereochemistry and isotopes; lookup frozen candidate group_hash; validated on all 200 queries',
         'ged_timeout_seconds': 10, 'python': sys.version, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    (out / 'run.pid').write_text(str(os.getpid()))
    for i, request in enumerate(requests):
        if not args.resume:
            save(out / 'requests' / f'{i:04d}.json', request)
    code = 1
    try:
        responses = asyncio.run(infer(keys, requests, out))
        keys.clear()
        (out / 'inference_finished_at').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        evaluate(queries, responses, groups, out)
        code = 0 if all('error' not in r for r in responses) else 2
    finally:
        (out / 'exit_code').write_text(str(code))
        (out / 'finished_at').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    raise SystemExit(code)


if __name__ == '__main__':
    main()

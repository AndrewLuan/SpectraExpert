#!/usr/bin/env python3
"""功能概述：无需训练，测量实验 Raman 到理论 Raman 重合分子组的闭库检索。
主要职责：load_overlap 只读索引；extract 按 group_hash 流式筛选六个 Raman gzip；
score 在双方 coverage 的交集计算 cosine/Pearson；evaluate 先按候选分子组取最大分数，
再输出逐谱排名与分子等权统计；self_test 验证排名、缺失覆盖与常数谱处理；main 保存审计产物。
代码库关系：独立 NumPy baseline，消费 SpecGPT continuous_v1 的 3000 点 0..4000 网格，
沿用索引最大片段/忽略立体化学的身份定义，不依赖训练器或 LLM，也不修改上游划分。
输入输出与副作用：只读显式 Raman 文件和 SQLite；仅新建 output-dir，拒绝覆盖已有结果。
实验使用所有 split 的重合组，属于描述性 pilot，不是训练后 held-out 评估；原始预处理已
使用 robust_minmax 1/99 分位。比较限制 500..4000 cm-1，覆盖交集至少为该范围一半。
同组多条理论谱取最佳相似度，多条实验谱先组内平均；并列分数使用保守最差名次。
"""
import argparse
import gzip
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('/data/SpecGPT/dataset/processed_stage_2_v1/continuous')
INDEX = Path('/data/SpecGPT/dataset/data_index/specgpt_dataset_index.sqlite')
HASH = re.compile(rb'"group_hash"\s*:\s*"([0-9a-f]{32})"')
AXIS = np.linspace(0, 4000, 3000)
REGION = AXIS >= 500


def load_overlap(path):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as con:
        rows = con.execute('SELECT hex(group_hash),exp_raman_count,theory_raman_count '
                           'FROM molecule_groups WHERE exp_raman_count>0 AND theory_raman_count>0').fetchall()
    return {key.lower(): {'exp': e, 'theory': t} for key, e, t in rows}


def extract(root, domain, overlap):
    vectors, masks, records, audit = [], [], [], []
    for split in ('train', 'valid', 'test'):
        path = root / 'combined' / domain / 'raman' / f'{split}.jsonl.gz'
        digest = hashlib.sha256()
        counts = Counter()
        started = time.perf_counter()
        with gzip.open(path, 'rb') as handle:
            for line_no, raw in enumerate(handle, 1):
                digest.update(raw)
                counts['scanned'] += 1
                match = HASH.search(raw)
                if match is None:
                    raise ValueError(f'missing group_hash: {path}:{line_no}')
                group = match[1].decode()
                if group not in overlap:
                    continue
                counts['matched'] += 1
                row = json.loads(raw)
                spectrum = np.asarray(row['spectrum'], dtype=np.float64)
                coverage = row['coverage']
                if spectrum.shape != (3000,) or not np.isfinite(spectrum).all():
                    counts['invalid_spectrum'] += 1
                    continue
                if len(coverage) != 2 or not 0 <= coverage[0] < coverage[1] <= 3000:
                    counts['invalid_coverage'] += 1
                    continue
                mask = REGION & (np.arange(3000) >= coverage[0]) & (np.arange(3000) < coverage[1])
                if mask.sum() < REGION.sum() * 0.5:
                    counts['insufficient_coverage'] += 1
                    continue
                if np.linalg.norm(spectrum[mask]) <= 1e-12:
                    counts['zero_spectrum'] += 1
                    continue
                vectors.append(spectrum)
                masks.append(mask)
                records.append({k: row.get(k) for k in ('group_hash', 'smiles', 'source', 'condition', 'coverage')} |
                               {'split': split, 'line': line_no})
                counts['retained'] += 1
        entry = {'path': str(path), 'bytes': path.stat().st_size,
                 'decompressed_sha256': digest.hexdigest(), 'seconds': time.perf_counter()-started,
                 'counts': dict(counts)}
        audit.append(entry)
        print(json.dumps({'extraction': entry}), flush=True)
    return np.asarray(vectors), np.asarray(masks), records, audit


def score(query, qmask, candidates, cmasks, min_points):
    mask = cmasks & qmask
    n = mask.sum(axis=1)
    x = candidates * mask
    y = query * mask
    dot = np.einsum('ij,ij->i', x, y)
    xx = np.einsum('ij,ij->i', x, x)
    yy = np.einsum('ij,ij->i', y, y)
    valid = n >= min_points
    denom = np.sqrt(xx * yy)
    cosine = np.full(len(candidates), -np.inf)
    ok = valid & (denom > 1e-12)
    cosine[ok] = dot[ok] / denom[ok]
    sx, sy = x.sum(axis=1), y.sum(axis=1)
    safe_n = np.maximum(n, 1)
    centered = dot - sx * sy / safe_n
    denom = np.sqrt(np.maximum(0, xx-sx*sx/safe_n) * np.maximum(0, yy-sy*sy/safe_n))
    pearson = np.full(len(candidates), -np.inf)
    ok = valid & (denom > 1e-12)
    pearson[ok] = centered[ok] / denom[ok]
    return {'cosine': cosine, 'pearson': pearson}


def ranked_groups(scores, codes, groups, target):
    values = np.full(len(groups), -np.inf)
    np.maximum.at(values, codes, scores)
    finite = np.isfinite(values)
    target_i = groups.index(target) if target in groups else None
    rank = None
    if target_i is not None and finite[target_i]:
        rank = int(np.sum(values[finite] >= values[target_i]))
    order = np.argsort(-values, kind='stable')
    top = [{'group_hash': groups[i], 'score': float(values[i])} for i in order if finite[i]][:10]
    return {'rank': rank, 'eligible_groups': int(finite.sum()), 'top10': top}


def evaluate(q, qm, qr, c, cm, cr, overlap, out):
    groups = sorted({r['group_hash'] for r in cr})
    lookup = {g: i for i, g in enumerate(groups)}
    codes = np.array([lookup[r['group_hash']] for r in cr])
    results, latency = [], []
    started = time.perf_counter()
    with (out / 'per_query.jsonl').open('x') as handle:
        for i, row in enumerate(qr):
            tic = time.perf_counter()
            scores = score(q[i], qm[i], c, cm, int(np.ceil(REGION.sum()*0.5)))
            result = {'query': row, 'metrics': {name: ranked_groups(v, codes, groups, row['group_hash'])
                                               for name, v in scores.items()}}
            latency.append(time.perf_counter()-tic)
            handle.write(json.dumps(result, allow_nan=False)+'\n')
            results.append(result)
            if (i+1) % 100 == 0:
                print(json.dumps({'queries_completed': i+1, 'seconds': time.perf_counter()-started}), flush=True)
    summary = {}
    for metric in ('cosine', 'pearson'):
        by_group = {g: [] for g in overlap}
        ranks = []
        for result in results:
            rank = result['metrics'][metric]['rank']
            values = [float(rank is not None and rank <= k) for k in (1, 5, 10)] + [1/rank if rank else 0.]
            by_group[result['query']['group_hash']].append(values)
            if rank is not None:
                ranks.append(rank)
        macro = np.mean([np.mean(v, axis=0) if v else np.zeros(4) for v in by_group.values()], axis=0)
        summary[metric] = dict(zip(('group_macro_top1', 'group_macro_top5', 'group_macro_top10', 'group_macro_mrr'), macro.tolist()))
        summary[metric].update(target_rank_available_queries=len(ranks),
                               median_rank_available=float(np.median(ranks)) if ranks else None)
    return {'metrics': summary, 'candidate_groups': len(groups), 'candidate_spectra': len(cr),
            'query_groups': len({r['group_hash'] for r in qr}), 'query_spectra': len(qr),
            'indexed_overlap_groups_denominator': len(overlap),
            'missing_query_groups': sorted(set(overlap)-{r['group_hash'] for r in qr}),
            'missing_candidate_groups': sorted(set(overlap)-set(groups)),
            'retrieval_seconds': time.perf_counter()-started,
            'both_metrics_per_query_ms_median': float(np.median(latency)*1000),
            'both_metrics_per_query_ms_p95': float(np.percentile(latency, 95)*1000)}


def self_test():
    q = np.array([1., 2., 3., 4.])
    c = np.array([q, q[::-1], q*3, np.ones(4)])
    m = np.ones_like(c, dtype=bool)
    s = score(q, m[0], c, m, 4)
    assert np.isclose(s['cosine'][0], 1) and np.isclose(s['cosine'][2], 1)
    assert np.isclose(s['pearson'][1], -1) and not np.isfinite(s['pearson'][3])
    assert ranked_groups(np.array([1., .3, .8]), np.array([0,1,0]), ['a','b'], 'a')['rank'] == 1
    assert ranked_groups(np.array([1., 1.]), np.array([0,1]), ['a','b'], 'a')['rank'] == 2
    assert ranked_groups(np.array([1.]), np.array([0]), ['a'], 'missing')['rank'] is None
    m[:, 0] = False
    assert not np.isfinite(score(q, m[0], c, m, 4)['cosine']).any()
    print('self-test passed', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--index', type=Path, default=INDEX)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output_dir is None:
        parser.error('--output-dir required')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    overlap = load_overlap(args.index)
    manifests = {d: json.loads((args.root / 'manifests' / f'processed_manifest_{d}_raman.json').read_text()) for d in ('exp','theory')}
    for doc in manifests.values():
        assert doc['config']['grid_size'] == 3000
        assert doc['config']['modalities']['Raman']['axis_min'] == 0
        assert doc['config']['modalities']['Raman']['axis_max'] == 4000
    (out / 'protocol.json').write_text(json.dumps({'overlap': overlap, 'manifests': manifests,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'identity': 'index parent connectivity, largest fragment, no stereochemistry',
        'scope': 'all-split closed-library descriptive overlap pilot, no training',
        'range_cm-1': [500,4000], 'minimum_pair_coverage_fraction': 0.5,
        'score': 'cosine primary; Pearson diagnostic; pairwise coverage intersection',
        'group_scoring': 'maximum over theory spectra, mean over experimental queries per group',
        'ties': 'conservative worst rank', 'missing_groups': 'zero contribution, indexed denominator retained'}, indent=2))
    q, qm, qr, qa = extract(args.root, 'exp', overlap)
    c, cm, cr, ca = extract(args.root, 'theory', overlap)
    np.savez_compressed(out / 'overlap_arrays.npz', query=q, query_mask=qm, candidate=c, candidate_mask=cm)
    (out / 'records.json').write_text(json.dumps({'query': qr, 'candidate': cr}))
    summary = evaluate(q, qm, qr, c, cm, cr, overlap, out)
    summary.update(extraction=qa+ca, total_seconds=time.perf_counter()-started)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({'completed': summary}), flush=True)


if __name__ == '__main__':
    main()

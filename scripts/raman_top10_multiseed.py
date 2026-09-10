"""功能概述：对 Top10 baseline 与 hard 方案执行配对多随机种子复核。
主要职责：train 复用冻结数据和训练协议，以指定 seed 从头训练单个方案并按独立验证
Exact@10 保存 checkpoint；evaluate 用该 checkpoint 评估固定 200 条测试但不参与选择；
summarize 汇总 seed=0 既有结果与新增种子，报告逐种子差值、均值、标准差、t 区间、
胜出次数和理论来源分层；区间以种子为独立重复单位，不把 200 条查询当作独立训练重复。
代码库关系：依赖 raman_top10_experiment 的数据校验、视图、损失、编码及检索函数；
seed=0 checkpoint 来自原 run01，新增 seed 写入独立 --output，不改变任何旧结果。
输入输出与副作用：不调用 API，不重建或修改数据。每个 seed/arm 目录只创建一次，保存
配置、逐轮指标、最佳 checkpoint、验证与测试排名；已存在目录会报错，避免静默覆盖。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.raman_top10_experiment import (MODEL_CONFIG, MaskedSpectrumEncoder, canonical,
    contrastive, coverage_mask, digest, encode, load, load_queries, retrieval, save, view)


def train(base, output, seed, arm):
    protocol, d = load(base)
    directory = output / f'seed_{seed}' / arm
    directory.mkdir(parents=True, exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('formal multiseed training requires CUDA')
    model = MaskedSpectrumEncoder(**MODEL_CONFIG).cuda()
    initial_hash = hashlib.sha256(b''.join(
        value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed); extra_rng = np.random.default_rng(seed + 101)
    params = json.loads((base / 'domain_parameters.json').read_text())
    proxy = json.loads((base / 'proxy.json').read_text())
    residuals = json.loads((base / 'random_parameters.json').read_text())
    t, tm, e, em = (d[k] for k in ('theory', 'theory_mask', 'experiment', 'experiment_mask'))
    train_idx, val, ref_train, ref_ids = (d[k] for k in
                                           ('train', 'validation', 'theory_train', 'reference_ids'))
    groups = [protocol['experimental_metadata'][i]['group_hash'] for i in val]
    train_masks = em[train_idx]
    positive_refs, source_refs = {}, {}
    for i in ref_train:
        positive_refs.setdefault(int(ref_ids[i]), []).append(int(i))
        source_refs.setdefault(protocol['reference_sources'][i], []).append(int(i))
    real_by_source = {}
    for source, refs in source_refs.items():
        identities = set(ref_ids[refs]); buckets = {}
        for i in train_idx:
            if d['targets'][i] in identities:
                buckets.setdefault(int(d['targets'][i]), []).append(int(i))
        if buckets:
            real_by_source[source] = buckets
    source_names = sorted(real_by_source)
    config = dict(seed=seed, arm=arm, protocol_sha256=digest(base / 'protocol.json'),
        epochs=protocol['epochs'], steps=protocol['steps'], patience=protocol['patience'],
        batch_queries=64, extra_references=32, learning_rate=3e-4, temperature=.1,
        initial_sha256=initial_hash, selection=protocol['selection'])
    save(directory / 'config.json', config)
    best, bad, neighbors = None, 0, None; started = time.monotonic()
    for epoch in range(protocol['epochs']):
        model.train(); losses = []
        for _ in range(protocol['steps']):
            query, qm, refs, identities, chosen_exp = [], [], [], [], []
            for _ in range(32):
                idx = int(rng.choice(ref_train)); source = protocol['reference_sources'][idx]
                aug, mask = view(t[idx], tm[idx], source, arm, params, proxy, residuals,
                                 rng, train_masks)
                query.append(aug); qm.append(mask); refs.append(idx); identities.append(int(ref_ids[idx]))
            for _ in range(32):
                source = source_names[rng.integers(len(source_names))]
                buckets = real_by_source[source]; identity = int(rng.choice(list(buckets)))
                idx = int(rng.choice(buckets[identity]))
                candidates = [j for j in positive_refs[identity]
                              if protocol['reference_sources'][j] == source]
                ref = int(rng.choice(candidates))
                query.append(e[idx]); qm.append(em[idx]); refs.append(ref)
                identities.append(identity); chosen_exp.append(idx)
            extras = []
            for i in chosen_exp:
                if arm == 'hard' and neighbors is not None:
                    extras.append(int(extra_rng.choice(neighbors[i])))
                else:
                    eligible = ref_train[ref_ids[ref_train] != d['targets'][i]]
                    extras.append(int(extra_rng.choice(eligible)))
            refs = np.array(refs + extras)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                left = model(torch.as_tensor(np.stack(query), device='cuda'),
                             torch.as_tensor(np.stack(qm), device='cuda'))
                right = model(torch.as_tensor(t[refs], device='cuda'),
                              torch.as_tensor(tm[refs], device='cuda'))
                loss = contrastive(left, right, torch.tensor(identities, device='cuda'),
                                   torch.tensor(ref_ids[refs], device='cuda'))
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
        reference = encode(model, t, tm, 'cuda')
        query_embedding = encode(model, e[val], em[val], 'cuda')
        metrics, ranks, top = retrieval(query_embedding, reference[:protocol['candidate_count']],
                                        d['targets'][val], groups)
        key = (metrics['group_exact@10'], metrics['group_exact@1'], -float(np.mean(losses)))
        record = dict(epoch=epoch, loss=float(np.mean(losses)), validation=metrics,
                      seconds=time.monotonic() - started)
        with (directory / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if best is None or key > best:
            best, bad = key, 0
            torch.save(dict(schema='raman_top10_multiseed_v1', state_dict=model.state_dict(),
                model_config=MODEL_CONFIG, protocol_sha256=digest(base / 'protocol.json'), seed=seed,
                arm=arm, epoch=epoch, validation=metrics, initial_sha256=initial_hash),
                directory / 'best.pt')
            save(directory / 'validation.json', dict(metrics=metrics, ranks=ranks.tolist(),
                top10=top.tolist(), experiment_indices=val.tolist(), groups=groups))
        else:
            bad += 1
        if bad >= protocol['patience']:
            break
        if arm == 'hard':
            train_query = encode(model, e[train_idx], em[train_idx], 'cuda')
            scores = train_query @ reference[ref_train].T
            scores[ref_ids[ref_train][None, :] == d['targets'][train_idx, None]] = -np.inf
            nearest = np.argsort(-scores, axis=1, kind='stable')[:, :16]
            neighbors = {int(i): ref_train[row] for i, row in zip(train_idx, nearest)}
    save(directory / 'summary.json', dict(seed=seed, arm=arm, best_key=best, epochs=epoch + 1,
        seconds=time.monotonic() - started, initial_sha256=initial_hash,
        checkpoint_sha256=digest(directory / 'best.pt')))


def evaluate(base, directory):
    protocol, d = load(base)
    summary = json.loads((directory / 'summary.json').read_text())
    checkpoint = directory / 'best.pt'
    if digest(checkpoint) != summary['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['protocol_sha256'] != digest(base / 'protocol.json'):
        raise ValueError('checkpoint protocol mismatch')
    torch.set_num_threads(4)
    model = MaskedSpectrumEncoder(**MODEL_CONFIG).cuda(); model.load_state_dict(payload['state_dict'])
    root = Path(protocol['data_root'])
    queries, spectra, _ = load_queries(root, protocol['sample_manifest_sha256'])
    rows = {r['mol_id']: r for r in json.loads((root / 'experimental_test/rows.json').read_text())}
    masks = np.stack([coverage_mask(rows[q['mol_id']]['coverage']) for q in queries])
    names = protocol['candidate_names'][:10000]; lookup = {s: i for i, s in enumerate(names)}
    targets = np.array([lookup[canonical(q['ground_truth_smiles'])] for q in queries])
    reference = encode(model, d['theory'][:10000], d['theory_mask'][:10000], 'cuda')
    query_embedding = encode(model, spectra, masks, 'cuda')
    metrics, ranks, top = retrieval(query_embedding, reference, targets,
                                    [q['group_hash'] for q in queries])
    save(directory / 'test.json', dict(metrics=metrics, ranks=ranks.tolist(), top10=top.tolist(),
        query_ids=[q['mol_id'] for q in queries], checkpoint_sha256=summary['checkpoint_sha256'],
        selection_uses='validation only; test evaluated after checkpoint frozen'))
    print(json.dumps(metrics), flush=True)


def stats(values):
    values = np.asarray(values, dtype=float)
    sd = float(values.std(ddof=1))
    half = float(student_t.ppf(.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return dict(mean=float(values.mean()), sample_sd=sd, minimum=float(values.min()),
                maximum=float(values.max()), mean_t_ci95=[float(values.mean() - half),
                                                         float(values.mean() + half)])


def summarize(base, output, seeds):
    protocol, _ = load(base)
    records = []
    for seed in seeds:
        locations = ({a: base / a for a in ('baseline', 'hard')} if seed == 0 else
                     {a: output / f'seed_{seed}' / a for a in ('baseline', 'hard')})
        row = {'seed': seed, 'arms': {}}
        for arm, directory in locations.items():
            validation = json.loads((directory / 'validation.json').read_text())
            test = json.loads((directory / 'test.json').read_text())
            row['arms'][arm] = {'validation': validation['metrics'], 'test': test['metrics']}
        records.append(row)
    report = {'schema': 'raman_top10_multiseed_summary_v1', 'seeds': seeds, 'records': records,
              'aggregate': {}}
    for split, key in (('validation', 'group_exact@10'), ('test', 'exact@10')):
        baseline = [r['arms']['baseline'][split][key] for r in records]
        hard = [r['arms']['hard'][split][key] for r in records]
        delta = np.subtract(hard, baseline)
        report['aggregate'][split] = dict(baseline=stats(baseline), hard=stats(hard),
            hard_minus_baseline=stats(delta), hard_wins=int((delta > 0).sum()),
            ties=int((delta == 0).sum()), baseline_wins=int((delta < 0).sum()))
    queries, _, _ = load_queries(Path(protocol['data_root']), protocol['sample_manifest_sha256'])
    name_to_source = dict(zip(protocol['candidate_names'], protocol['candidate_sources']))
    sources = np.array([name_to_source[canonical(q['ground_truth_smiles'])] for q in queries])
    report['aggregate']['test_by_source'] = {}
    for source in sorted(set(sources)):
        mask = sources == source; result = {}
        for arm in ('baseline', 'hard'):
            values = []
            for seed, row in zip(seeds, records):
                directory = (base / arm if seed == 0 else output / f'seed_{seed}' / arm)
                ranks = np.array(json.loads((directory / 'test.json').read_text())['ranks'])
                values.append(float((ranks[mask] <= 10).mean()))
            result[arm] = stats(values)
        hard_values = []; baseline_values = []
        for seed in seeds:
            for arm, values in (('hard', hard_values), ('baseline', baseline_values)):
                directory = base / arm if seed == 0 else output / f'seed_{seed}' / arm
                ranks = np.array(json.loads((directory / 'test.json').read_text())['ranks'])
                values.append(float((ranks[mask] <= 10).mean()))
        result['hard_minus_baseline'] = stats(np.subtract(hard_values, baseline_values))
        result['n_queries'] = int(mask.sum())
        report['aggregate']['test_by_source'][source] = result
    destination = output / 'summary_v2.json'
    if destination.exists():
        raise FileExistsError(destination)
    save(destination, report); print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train', 'evaluate', 'summarize'))
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--arm', choices=('baseline', 'hard'))
    p.add_argument('--directory', type=Path)
    p.add_argument('--seeds', default='0,1,2,3,4')
    args = p.parse_args()
    if args.mode == 'train':
        train(args.base, args.output, args.seed, args.arm)
    elif args.mode == 'evaluate':
        evaluate(args.base, args.directory)
    else:
        summarize(args.base, args.output, [int(x) for x in args.seeds.split(',')])


if __name__ == '__main__':
    main()

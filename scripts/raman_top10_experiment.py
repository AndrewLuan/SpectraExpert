"""功能概述：统一 surrogate 分组，从头比较四种训练策略，以真实验证 Exact@10 冻结选择。
主要职责：prepare 校验来源哈希、排除受保护实验谱、重采样到原编码器轴并建立多理论参考；
retrieval 以完整结构计算名次与组等权指标；contrastive 支持多正样本和额外负样本；
view 生成经验/固定代理/随机代理视图；train 在相同初始化与预算下训练，困难负样本
仅由训练查询挖掘；finalize 先写选择，再对固定 200 测试查询评估唯一所选模型。
代码库关系：复用 raman_domain_experiment 的编码器、经验域拟合与掩码；复用
fit_raman_curve_proxy 的曲线变换，消费严格 surrogate 数据与修正后的固定候选库。
输入输出与副作用：只写新的 --out 目录，不覆盖旧实验，不调用 API。所有来源的
实验验证谱仅用于验证，实验测试谱仅在 finalize 读取；候选理论谱可用于训练的闭库
设定明确保留。四组都使用多理论正样本和等量额外参考，避免把计算量误当负样本增益。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.raman_domain_experiment import (AXIS, MODEL_CONFIG, MaskedSpectrumEncoder,
    encode, fit_domain, augment_domain, coverage_mask)
from scripts.fit_raman_curve_proxy import Dataset, transform, BOUNDS
from scripts.evaluate_closed_raman import digest, load_role
from scripts.repair_closed_raman_exact import canonical
from scripts.run_closed_llm_direct import load_queries, save
from spectraexpert.utils.augmentations import create_positive_pair

ARMS = ('baseline', 'fixed', 'random', 'hard')


def retrieval(query, reference, targets, groups):
    scores = query @ reference.T
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite retrieval scores')
    order = np.argsort(-scores, axis=1, kind='stable')
    ranks = np.argmax(order == np.asarray(targets)[:, None], axis=1) + 1
    group_names = sorted(set(groups))
    metrics = {'n': len(ranks), 'groups': len(group_names)}
    for k in (1, 5, 10):
        hit = ranks <= k
        metrics[f'exact@{k}'] = float(hit.mean())
        metrics[f'count@{k}'] = int(hit.sum())
        metrics[f'group_exact@{k}'] = float(np.mean([
            hit[np.asarray(groups) == g].mean() for g in group_names]))
    return metrics, ranks, order[:, :10]


def contrastive(left, right, left_ids, right_ids, temperature=.1):
    logits = left.float() @ right.float().T / temperature
    positives = left_ids[:, None] == right_ids[None, :]
    if not positives.any(1).all():
        raise ValueError('query has no positive reference')
    def loss(z, p):
        return (torch.logsumexp(z, 1) - torch.logsumexp(z.masked_fill(~p, -torch.inf), 1)).mean()
    supported = positives.any(0)
    return .5 * (loss(logits, positives) + loss(logits.T[supported], positives.T[supported]))


def prepare(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    for name, sha in manifest['files_sha256'].items():
        if digest(args.dataset / name) != sha:
            raise ValueError(f'surrogate hash mismatch: {name}')
    ds = Dataset(args.dataset)
    sample = json.loads((args.data_root / 'sample_manifest.json').read_text())
    assert manifest['protected_manifest_sha256'] == digest(args.data_root / 'sample_manifest.json')
    assert set(ds.split['groups']['protected_final_test']) == set(sample['test_groups'])
    rows, base = load_role(args.data_root, 'theory_train_candidate', sample)
    split = json.loads((args.data_root / 'training_split.json').read_text())
    names = [canonical(s) for s in rows['smiles']]
    assert len(names) == len(set(names)) == 10000
    lookup = {s: i for i, s in enumerate(names)}
    pool = list(base)
    masks = [coverage_mask(c) for c in rows['coverage']]
    sources = rows['source'].tolist()
    ref_ids = list(range(10000))
    train_base = [i for i, mid in enumerate(rows['mol_id']) if mid in set(split['train'])]
    assert len(train_base) == 9500
    withheld_names = {names[i] for i in range(10000) if i not in set(train_base)}
    train_refs = train_base.copy()
    def resample(domain, index):
        y = ds.spectra[domain][index]
        m = ds.masks[domain][index]
        return (np.interp(AXIS, ds.axis, y).astype(np.float32),
                np.interp(AXIS, ds.axis, m.astype(float), left=0, right=0) > .999)
    theory_indices = sorted({p['theory_index'] for p in ds.pairs})
    # 先为每个新结构建立唯一候选，再追加其它理论重复记录，避免候选位置被重复谱打断。
    first = {}
    for i in theory_indices:
        s = ds.records['theory'][i]['canonical_smiles']
        first.setdefault(s, i)
    reference_map = {}
    for s, i in sorted(first.items()):
        if s not in lookup:
            lookup[s] = len(names)
            names.append(s)
            y, m = resample('theory', i)
            reference_map[i] = len(pool)
            pool.append(y); masks.append(m); sources.append(ds.records['theory'][i]['source'])
            ref_ids.append(lookup[s]); train_refs.append(len(pool) - 1)
    candidate_count = len(names)
    for i in theory_indices:
        if i in reference_map:
            continue
        s = ds.records['theory'][i]['canonical_smiles']
        y, m = resample('theory', i)
        reference_map[i] = len(pool)
        pool.append(y); masks.append(m); sources.append(ds.records['theory'][i]['source'])
        ref_ids.append(lookup[s])
        if s not in withheld_names:
            train_refs.append(len(pool) - 1)
    observations = {}
    for pair in ds.pairs:
        observations.setdefault(pair['exp_index'], pair)
    exp, em, metadata, targets = [], [], [], []
    for i, pair in sorted(observations.items()):
        y, m = resample('exp', i)
        exp.append(y); em.append(m)
        metadata.append(dict(group_hash=pair['group_hash'], canonical_smiles=pair['smiles'],
            role=pair['role'], phase=pair['phase'], exp_index=i))
        targets.append(lookup[pair['smiles']])
    train = np.array([i for i, r in enumerate(metadata) if r['role'] == 'surrogate_train'])
    val = np.array([i for i, r in enumerate(metadata) if r['role'] == 'surrogate_validation'])
    pool, masks, exp, em, targets = map(np.array, (pool, masks, exp, em, targets))
    assert not set(targets[train]) & {lookup[n] for n in withheld_names}, 'experimental train targets withheld theory'
    proxy = json.loads((args.proxy / 'model.json').read_text())
    proxy_metrics = json.loads((args.proxy / 'metrics.json').read_text())
    # 代理必须绑定同一 surrogate 划分；参数只来自 surrogate_train。
    assert proxy['fitted_role'] == 'surrogate_train'
    assert proxy_metrics['audit']['split_sha256'] == digest(args.dataset / 'split.json')
    assert proxy_metrics['audit']['dataset_manifest_sha256'] == digest(args.dataset / 'manifest.json')
    assert proxy_metrics['audit']['protected_spectra_accessed'] == 0
    params, diagnostics = fit_domain(pool[targets[train]], masks[targets[train]], exp[train], em[train])
    by_source = {}
    experiment_lookup = {r['exp_index']: i for i, r in enumerate(metadata)}
    for source in sorted(set(sources)):
        pairs = [p for p in ds.pairs if p['role'] == 'surrogate_train' and p['source'] == source]
        ti = [reference_map[p['theory_index']] for p in pairs]
        ei = [experiment_lookup[p['exp_index']] for p in pairs]
        _, fitted = fit_domain(pool[ti], masks[ti], exp[ei], em[ei])
        selected = [d for d in fitted if d['reliable']]
        affine = np.array([[d['scale'], d['shift']] for d in selected])
        deviations = (affine - np.median(affine, axis=0)).tolist() if len(affine) >= 10 else [[0., 0.]]
        by_source[source] = {'deviations': deviations, 'reliable_fits': len(selected), 'train_pairs': len(pairs)}
        save(out / f'{source}_training_fits.json', fitted)
    save(out / 'domain_parameters.json', params)
    save(out / 'domain_diagnostics.json', diagnostics)
    save(out / 'proxy.json', proxy)
    save(out / 'random_parameters.json', by_source)
    np.savez_compressed(out / 'prepared.npz', theory=pool, theory_mask=masks, experiment=exp,
        experiment_mask=em, targets=targets, train=train, validation=val,
        theory_train=np.array(train_refs), reference_ids=np.array(ref_ids))
    protocol = dict(schema='raman_top10_v1', data_root=str(args.data_root.resolve()),
        sample_manifest_sha256=digest(args.data_root / 'sample_manifest.json'),
        surrogate_manifest_sha256=digest(args.dataset / 'manifest.json'),
        surrogate_split_sha256=digest(args.dataset / 'split.json'),
        proxy_model_sha256=digest(args.proxy / 'model.json'), candidate_names=names,
        candidate_sources=sources[:candidate_count], reference_sources=sources,
        candidate_mol_ids=rows['mol_id'].tolist(), experimental_metadata=metadata,
        train_groups=ds.split['groups']['surrogate_train'], validation_groups=ds.split['groups']['surrogate_validation'],
        excluded_test_groups=ds.split['groups']['protected_final_test'],
        candidate_count=candidate_count, final_candidate_count=10000,
        theory_reference_count=len(pool), theoretical_training_references=len(train_refs),
        train_spectra=len(train), validation_spectra=len(val), model_config=MODEL_CONFIG,
        epochs=args.epochs, steps=args.steps, seed=0, patience=8,
        selection='group_equal_exact@10, group_equal_exact@1, lower training loss; arm order breaks exact ties',
        training_policy='same seed initialization; 32 synthetic + 32 real queries, 64 positive + 32 extra references; train source balanced real pairs; 9500 original train records plus nonprotected supplemental theory; validation theory identities permitted; validation experiment withheld',
        augmentation_policy='baseline uses refitted legacy domain augmentation; fixed uses saved deterministic curve proxy; random/hard use 20% identity, 20% fixed, 60% source-centered joint residual affine and 0..12 extra sigma; all arms share base create_positive_pair',
        hashes={f: digest(out / f) for f in ('prepared.npz', 'proxy.json', 'domain_parameters.json', 'random_parameters.json')},
        source_sha256={f: digest(Path(__file__).parent / f) for f in ('raman_top10_experiment.py', 'raman_domain_experiment.py', 'fit_raman_curve_proxy.py')})
    save(out / 'protocol.json', protocol)
    print(json.dumps({k: protocol[k] for k in ('candidate_count', 'theory_reference_count', 'train_spectra', 'validation_spectra')}), flush=True)


def load(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    for name, sha in protocol['hashes'].items():
        if digest(out / name) != sha:
            raise ValueError(f'prepared file changed: {name}')
    for name, sha in protocol['source_sha256'].items():
        if digest(Path(__file__).parent / name) != sha:
            raise ValueError(f'protocol source changed: {name}')
    with np.load(out / 'prepared.npz', allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    return protocol, data


def view(x, mask, source, arm, params, proxy, residuals, rng, exp_masks):
    _, aug = create_positive_pair(np.where(mask, x, 0), AXIS)
    if arm == 'baseline':
        return augment_domain(aug, mask, params, rng, exp_masks)
    p = list(proxy['sources'][source]['params'])
    if arm in ('random', 'hard'):
        draw = rng.random()
        if draw < .2:
            p = [1., 0., 0.]
        elif draw >= .4:
            delta = residuals[source]['deviations'][rng.integers(len(residuals[source]['deviations']))]
            p[0] = np.clip(p[0] + .5 * delta[0], *BOUNDS[0])
            p[1] = np.clip(p[1] + .5 * delta[1], *BOUNDS[1])
            p[2] = rng.uniform(0, p[2])
    values, valid = transform(AXIS, aug, mask, p)
    y, m = values[0], valid[0]
    m &= exp_masks[rng.integers(len(exp_masks))]
    if m.sum() < 100:
        return np.where(mask, aug, 0).astype(np.float32), mask.copy()
    return np.where(m, y, 0).astype(np.float32), m


def train(args):
    protocol, d = load(args.out)
    directory = args.out / args.arm
    directory.mkdir(exist_ok=False)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('formal training requires remote CUDA')
    model = MaskedSpectrumEncoder(**MODEL_CONFIG).cuda()
    initial_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    extra_rng = np.random.default_rng(101)
    params = json.loads((args.out / 'domain_parameters.json').read_text())
    proxy = json.loads((args.out / 'proxy.json').read_text())
    residuals = json.loads((args.out / 'random_parameters.json').read_text())
    t, tm, e, em = (d[k] for k in ('theory', 'theory_mask', 'experiment', 'experiment_mask'))
    train_idx, val, ref_train, ref_ids = (d[k] for k in ('train', 'validation', 'theory_train', 'reference_ids'))
    train_masks = em[train_idx]
    groups = [protocol['experimental_metadata'][i]['group_hash'] for i in val]
    source_refs = {}
    positive_refs = {}
    for i in ref_train:
        positive_refs.setdefault(int(ref_ids[i]), []).append(int(i))
        source_refs.setdefault(protocol['reference_sources'][i], []).append(int(i))
    # 按来源 -> 结构 -> 测量采样；有多种理论来源的身份在各来源均可配对。
    real_by_source = {}
    for source, refs in source_refs.items():
        eligible = set(ref_ids[refs])
        buckets = {}
        for i in train_idx:
            if d['targets'][i] in eligible:
                buckets.setdefault(int(d['targets'][i]), []).append(int(i))
        if buckets:
            real_by_source[source] = buckets
    source_names = sorted(real_by_source)
    best, bad, neighbors = None, 0, None
    start = time.monotonic()
    save(directory / 'config.json', dict(arm=args.arm, initial_sha256=initial_hash,
        protocol_sha256=digest(args.out / 'protocol.json'), batch_queries=64, extra_references=32,
        lr=3e-4, temperature=.1, seed=0, epochs=protocol['epochs'], steps=protocol['steps']))
    for epoch in range(protocol['epochs']):
        model.train(); losses = []
        for step in range(protocol['steps']):
            query, qm, refs, identities, chosen_exp = [], [], [], [], []
            for _ in range(32):
                idx = int(rng.choice(ref_train))
                source = protocol['reference_sources'][idx]
                aug, mask = view(t[idx], tm[idx], source, args.arm, params, proxy, residuals, rng, train_masks)
                query.append(aug); qm.append(mask); refs.append(idx); identities.append(int(ref_ids[idx]))
            for _ in range(32):
                source = source_names[rng.integers(len(source_names))]
                buckets = real_by_source[source]
                identity = int(rng.choice(list(buckets)))
                idx = int(rng.choice(buckets[identity]))
                eligible = [j for j in positive_refs[identity] if protocol['reference_sources'][j] == source]
                ref = int(rng.choice(eligible))
                query.append(e[idx]); qm.append(em[idx]); refs.append(ref); identities.append(identity); chosen_exp.append(idx)
            extras = []
            for i in chosen_exp:
                if args.arm == 'hard' and neighbors is not None:
                    extras.append(int(extra_rng.choice(neighbors[i])))
                else:
                    eligible = ref_train[ref_ids[ref_train] != d['targets'][i]]
                    extras.append(int(extra_rng.choice(eligible)))
            refs = np.array(refs + extras)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                left = model(torch.as_tensor(np.stack(query), device='cuda'), torch.as_tensor(np.stack(qm), device='cuda'))
                right = model(torch.as_tensor(t[refs], device='cuda'), torch.as_tensor(tm[refs], device='cuda'))
                loss = contrastive(left, right, torch.tensor(identities, device='cuda'), torch.tensor(ref_ids[refs], device='cuda'))
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            losses.append(float(loss.detach()))
        reference = encode(model, t, tm, 'cuda')
        query = encode(model, e[val], em[val], 'cuda')
        metrics, ranks, top = retrieval(query, reference[:protocol['candidate_count']], d['targets'][val], groups)
        key = (metrics['group_exact@10'], metrics['group_exact@1'], -float(np.mean(losses)))
        record = dict(epoch=epoch, loss=float(np.mean(losses)), validation=metrics, seconds=time.monotonic()-start)
        with (directory / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if best is None or key > best:
            best, bad = key, 0
            torch.save(dict(schema='raman_top10_v1', state_dict=model.state_dict(), model_config=MODEL_CONFIG,
                protocol_sha256=digest(args.out / 'protocol.json'), arm=args.arm, epoch=epoch,
                validation=metrics, initial_sha256=initial_hash), directory / 'best.pt')
            save(directory / 'validation.json', dict(metrics=metrics, ranks=ranks.tolist(), top10=top.tolist(),
                experiment_indices=val.tolist(), groups=groups))
        else:
            bad += 1
        if bad >= protocol['patience']:
            break
        if args.arm == 'hard':
            train_query = encode(model, e[train_idx], em[train_idx], 'cuda')
            scores = train_query @ reference[ref_train].T
            scores[ref_ids[ref_train][None, :] == d['targets'][train_idx, None]] = -np.inf
            nearest = np.argsort(-scores, axis=1, kind='stable')[:, :16]
            neighbors = {int(i): ref_train[row] for i, row in zip(train_idx, nearest)}
    save(directory / 'summary.json', dict(arm=args.arm, best_key=best, epochs=epoch+1,
        seconds=time.monotonic()-start, initial_sha256=initial_hash, checkpoint_sha256=digest(directory / 'best.pt')))


def finalize(args):
    protocol, d = load(args.out)
    summaries = {arm: json.loads((args.out / arm / 'summary.json').read_text()) for arm in ARMS}
    assert len({s['initial_sha256'] for s in summaries.values()}) == 1
    for arm in ARMS:
        assert digest(args.out / arm / 'best.pt') == summaries[arm]['checkpoint_sha256']
    selected = max(ARMS, key=lambda a: tuple(summaries[a]['best_key']))
    final = args.out / 'final'
    final.mkdir(exist_ok=False)
    save(final / 'selection.json', dict(selected_arm=selected, summaries=summaries,
        protocol_sha256=digest(args.out / 'protocol.json'), uses='independent validation only'))
    torch.set_num_threads(4)
    payload = torch.load(args.out / selected / 'best.pt', map_location='cpu', weights_only=False)
    assert payload['protocol_sha256'] == digest(args.out / 'protocol.json')
    model = MaskedSpectrumEncoder(**MODEL_CONFIG).cuda()
    model.load_state_dict(payload['state_dict'])
    root = Path(protocol['data_root'])
    queries, spectra, _ = load_queries(root, protocol['sample_manifest_sha256'])
    rows = {r['mol_id']: r for r in json.loads((root / 'experimental_test/rows.json').read_text())}
    masks = np.stack([coverage_mask(rows[q['mol_id']]['coverage']) for q in queries])
    names = protocol['candidate_names'][:10000]
    lookup = {s: i for i, s in enumerate(names)}
    targets = np.array([lookup[canonical(q['ground_truth_smiles'])] for q in queries])
    reference = encode(model, d['theory'][:10000], d['theory_mask'][:10000], 'cuda')
    query = encode(model, spectra, masks, 'cuda')
    raw = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    metrics = {}; details = {}
    for method, q, r in (('model', query, reference), ('raw_knn', raw(spectra), raw(d['theory'][:10000]))):
        m, ranks, top = retrieval(q, r, targets, [x['group_hash'] for x in queries])
        metrics[method] = m
        details[method] = dict(ranks=ranks.tolist(), top10=top.tolist())
    np.savez_compressed(final / 'candidate_spectrum_index.npz', embeddings=reference,
        canonical_smiles=np.array(names), mol_ids=np.array(protocol['candidate_mol_ids']),
        checkpoint_sha256=np.array(summaries[selected]['checkpoint_sha256']),
        sample_manifest_sha256=np.array(protocol['sample_manifest_sha256']))
    save(final / 'per_query.json', dict(queries=queries, candidates=names, methods=details))
    save(final / 'summary.json', dict(selected_arm=selected, metrics=metrics))
    print(json.dumps(dict(selected_arm=selected, metrics=metrics)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'train', 'finalize'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--proxy', type=Path)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--steps', type=int, default=160)
    args = p.parse_args()
    {'prepare': prepare, 'train': train, 'finalize': finalize}[args.mode](args)


if __name__ == '__main__':
    main()

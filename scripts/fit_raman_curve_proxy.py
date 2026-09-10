#!/usr/bin/env python3
"""功能概述：为 build_raman_surrogate_pairs.py 的严格配对数据绘图并拟合极轻量曲线代理。
主要职责：Dataset 校验分组、权限及索引，仅通过允许的配对访问内存映射光谱；transform
实现 x_exp=a*x_theory+b 与额外 Gaussian 展宽；fit_source 在 train 上按分子组等权拟合
三个有效参数；summarize 输出组级指标及配对 bootstrap；select_examples 按来源/相态选择
原始余弦相似度中位附近样本；plot_examples 绘制固定样本；main 管理 preview/fit 两个入口。
代码库关系：只消费独立 surrogate 数据集，不读取上游大型目录或检索测试产物；依赖
NumPy、SciPy 和 Matplotlib。model.json 的 sources 参数可由 transform 直接用于新理论谱。
输入输出与副作用：输出目录必须不存在，避免覆盖实验结果；训练和验证沿用 split.json，
protected_final_test 只检查身份元数据，永不索引其光谱。缺失 coverage 不作为零信号。
缩放/平移/展宽是归一化连续曲线上的有效参数，不代表恢复激光、温度或仪器参数；不生成
新峰、不反卷积、不拟合验证谱、不迁移 donor 残差。逐对拟合被禁止，验证预测只依赖理论来源。
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize


ROLES = ('surrogate_train', 'surrogate_validation')
BOUNDS = ((0.94, 1.02), (-40.0, 40.0), (0.0, 12.0))
IDENTITY = (1.0, 0.0, 0.0)


def read_rows(path):
    with gzip.open(path, 'rt') as handle:
        return [json.loads(line) for line in handle]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.split = json.loads((self.root / 'split.json').read_text())
        self.molecules = read_rows(self.root / 'molecules.jsonl.gz')
        groups = self.split['groups']
        role_names = (*ROLES, 'protected_final_test')
        for i, role in enumerate(role_names):
            for other in role_names[i + 1:]:
                assert not set(groups[role]) & set(groups[other]), 'group leakage'
            assert set(self.split['roles'][role]) == {
                m['molecule_id'] for m in self.molecules if m['role'] == role}
            assert set(groups[role]) == {
                m['group_hash'] for m in self.molecules if m['role'] == role}
        self.records = {d: read_rows(self.root / f'{d}_records.jsonl.gz')
                        for d in ('theory', 'exp')}
        self.axis = np.load(self.root / 'wavenumber_cm-1.npy')
        assert np.all(np.diff(self.axis) > 0)
        self.spectra = {d: np.load(self.root / f'{d}_spectra.npy', mmap_mode='r')
                        for d in ('theory', 'exp')}
        self.masks = {d: np.load(self.root / f'{d}_coverage_masks.npy', mmap_mode='r')
                      for d in ('theory', 'exp')}
        self.pairs = []
        for m in self.molecules:
            role = m['role']
            assert role in role_names
            assert m['allowed_for_parameter_fit'] == (role == ROLES[0])
            assert m['allowed_for_model_selection'] == (role == ROLES[1])
            if role not in ROLES:
                continue
            for ti in m['theory_indices']:
                for ei in m['experimental_indices']:
                    tr, er = self.records['theory'][ti], self.records['exp'][ei]
                    for domain, index, record in (('theory', ti, tr), ('exp', ei, er)):
                        assert record['spectrum_index'] == index
                        assert record['role'] == role
                        assert record['molecule_id'] == m['molecule_id']
                        assert record['group_hash'] == m['group_hash']
                        assert record['canonical_smiles'] == m['canonical_smiles']
                    phase = ('LIQUID' if 'LIQUID' in er['condition'] else
                             'SOLID' if 'SOLID' in er['condition'] else 'UNKNOWN')
                    self.pairs.append(dict(molecule_id=m['molecule_id'], group_hash=m['group_hash'],
                                           smiles=m['canonical_smiles'], role=role, theory_index=ti,
                                           exp_index=ei, source=tr['source'], phase=phase,
                                           theory_record_id=tr['record_id'], exp_record_id=er['record_id']))
        for d in self.spectra:
            assert self.spectra[d].shape == self.masks[d].shape == (len(self.records[d]), len(self.axis))

    def arrays(self, pairs):
        assert pairs and all(p['role'] in ROLES for p in pairs)
        ti = [p['theory_index'] for p in pairs]
        ei = [p['exp_index'] for p in pairs]
        t, e = self.spectra['theory'][ti].copy(), self.spectra['exp'][ei].copy()
        tm, em = self.masks['theory'][ti].copy(), self.masks['exp'][ei].copy()
        assert np.isfinite(t).all() and np.isfinite(e).all()
        return t, e, tm, em


def transform(axis, theory, coverage, params):
    """返回变换后的二维曲线与有效掩码；支持单条或批量，不需实验谱或实验相态。"""
    a, b, sigma = map(float, params)
    if a <= 0 or sigma < 0:
        raise ValueError('scale must be positive; broadening must be nonnegative')
    theory, coverage = np.atleast_2d(theory), np.atleast_2d(coverage)
    query = (axis - b) / a
    spacing = float(axis[1] - axis[0])
    clean = np.where(coverage, theory, 0.0)
    if sigma > 1e-8:
        clean = gaussian_filter1d(clean, sigma / spacing, axis=1, mode='constant', truncate=4)
    values = np.stack([np.interp(query, axis, row, left=0, right=0) / a for row in clean])
    # 要求整个截断 Gaussian 核位于已知覆盖内，防止零填充污染边界。
    mask = np.stack([(np.interp(query - 4 * sigma, axis, row.astype(float), left=0, right=0) > .999)
                     & (np.interp(query + 4 * sigma, axis, row.astype(float), left=0, right=0) > .999)
                     for row in coverage])
    return values, mask


def fixed_support(axis, tm, em):
    """所有候选变换共用保守评价域；不允许通过移动覆盖区挑选容易的谱点。"""
    mask = tm & em & (axis >= 500)[None, :]
    for a in BOUNDS[0]:
        for b in BOUNDS[1]:
            _, valid = transform(axis, np.zeros_like(tm, dtype=float), tm, (a, b, BOUNDS[2][1]))
            mask &= valid
    assert np.all(mask.sum(axis=1) >= 500), 'insufficient common support'
    return mask


def cosine_rows(left, right, mask):
    l, r = np.where(mask, left, 0), np.where(mask, right, 0)
    denominator = np.sqrt((l * l).sum(axis=1) * (r * r).sum(axis=1))
    return (l * r).sum(axis=1) / np.maximum(denominator, 1e-15)


def group_weights(pairs):
    counts = Counter(p['group_hash'] for p in pairs)
    return np.array([1 / counts[p['group_hash']] / len(counts) for p in pairs])


def fit_source(ds, pairs):
    assert all(p['role'] == ROLES[0] for p in pairs)
    t, e, tm, em = ds.arrays(pairs)
    mask = fixed_support(ds.axis, tm, em)
    weights = group_weights(pairs)
    stride = 3

    def objective(params):
        predicted, _ = transform(ds.axis, t, tm, params)
        return 1 - float(weights @ cosine_rows(predicted[:, ::stride], e[:, ::stride], mask[:, ::stride]))

    # 先粗搜缩放与平移，避免从 identity 单点陷入错误的峰对应；候选只看 train。
    candidates = [IDENTITY]
    for a in np.linspace(*BOUNDS[0], 9):
        for b in (-40., -20., 0., 20., 40.):
            candidates.append((float(a), b, 4.))
    ranked = sorted((objective(p), tuple(p)) for p in candidates)
    best_score, best = ranked[0]
    attempts = []
    for _, start in ranked[:2]:
        result = minimize(objective, start, method='Powell', bounds=BOUNDS,
                          options={'maxiter': 40, 'maxfev': 1200, 'xtol': 1e-4, 'ftol': 1e-5})
        attempts.append(dict(success=bool(result.success), message=str(result.message),
                             evaluations=int(result.nfev), loss=float(result.fun)))
        if result.fun < best_score:
            best_score, best = float(result.fun), tuple(map(float, result.x))
    # 全分辨率复核，确保输出参数不比 identity 的训练目标更差。
    full_identity = float(weights @ cosine_rows(t, e, mask))
    pred, _ = transform(ds.axis, t, tm, best)
    full_fit = float(weights @ cosine_rows(pred, e, mask))
    if full_fit < full_identity:
        best, full_fit = IDENTITY, full_identity
    return dict(params=list(best), train_pairs=len(pairs), train_groups=len(set(p['group_hash'] for p in pairs)),
                train_cosine_before=full_identity, train_cosine_after=full_fit,
                optimizer_attempts=attempts, fit_stride=stride)


def evaluate(ds, pairs, sources):
    rows = []
    for source in sorted(set(p['source'] for p in pairs)):
        selected = [p for p in pairs if p['source'] == source]
        t, e, tm, em = ds.arrays(selected)
        mask = fixed_support(ds.axis, tm, em)
        pred, _ = transform(ds.axis, t, tm, sources.get(source, {'params': IDENTITY})['params'])
        before = cosine_rows(t, e, mask)
        after = cosine_rows(pred, e, mask)
        for i, p in enumerate(selected):
            rows.append(dict(p, cosine_before=float(before[i]), cosine_after=float(after[i]),
                             evaluated_points=int(mask[i].sum())))
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_hash']].append([row['cosine_before'], row['cosine_after']])
    means = np.array([np.mean(groups[g], axis=0) for g in sorted(groups)])
    delta = means[:, 1] - means[:, 0]
    rng = np.random.default_rng(0)
    boot = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(axis=1)
    return dict(pairs=len(rows), molecules=len(set(r['molecule_id'] for r in rows)), groups=len(groups),
                group_mean_cosine_before=float(means[:, 0].mean()),
                group_mean_cosine_after=float(means[:, 1].mean()), mean_delta=float(delta.mean()),
                delta_bootstrap_95ci=np.quantile(boot, [.025, .975]).tolist(),
                fraction_groups_improved=float(np.mean(delta > 0)),
                minimum_evaluated_points=min(r['evaluated_points'] for r in rows))


def select_examples(rows):
    chosen, used = [], set()
    for source in sorted(set(r['source'] for r in rows)):
        for phase in ('LIQUID', 'SOLID'):
            candidates = [r for r in rows if r['source'] == source and r['phase'] == phase]
            if not candidates:
                continue
            # 每个组只保留一对，避免重复测量多的分子主导中位数。
            unique = {}
            for row in sorted(candidates, key=lambda r: (r['molecule_id'], r['theory_index'], r['exp_index'])):
                unique.setdefault(row['group_hash'], row)
            median = float(np.median([r['cosine_before'] for r in unique.values()]))
            for row in sorted(unique.values(), key=lambda r: (abs(r['cosine_before'] - median), r['molecule_id'])):
                if row['group_hash'] not in used:
                    chosen.append(row)
                    used.add(row['group_hash'])
                    break
    return chosen


def plot_examples(ds, rows, path, sources=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    role = 'TRAIN' if rows[0]['role'] == ROLES[0] else 'VALIDATION'
    fig.suptitle(f'{role} | Raman theory vs SDBS experiment', fontsize=20, x=.055, ha='left', y=.975)
    fig.text(.055, .925, 'One median-similarity example per source / phase; selection uses ORIGINAL curves.\n'
             'Each curve is scaled to its covered maximum for display; no peak alignment in the original overlay.', fontsize=11, color='#475569', va='top')
    for ax, row in zip(axes.flat, rows):
        t, e, tm, em = ds.arrays([row])
        x = ds.axis
        for value, mask, label, color in ((t[0], tm[0], 'Theory (original)', '#2563eb'),
                                          (e[0], em[0], 'Experiment (SDBS)', '#dc5b20')):
            mask = mask & (x >= 500)
            value = value / max(float(value[mask].max()), 1e-12)
            ax.plot(x, np.where(mask, value, np.nan), label=label, color=color, lw=1.15, alpha=.88)
        if sources is not None:
            pred, pm = transform(x, t, tm, sources[row['source']]['params'])
            valid = pm[0] & (x >= 500)
            value = pred[0] / max(float(pred[0][valid].max()), 1e-12)
            ax.plot(x, np.where(valid, value, np.nan), color='#16836b', lw=1.3, label='Proxy (train-fitted)', alpha=.85)
        missing = (x >= 500) & ~(tm[0] & em[0])
        ax.fill_between(x, 0, 1.1, where=missing, color='#cbd5e1', alpha=.3)
        label = row['source'].replace('Raman_', '') + ' / ' + row['phase']
        scores = f"cosine = {row['cosine_before']:.3f}"
        if sources is not None:
            scores += f" -> {row['cosine_after']:.3f}"
        ax.set_title(f'{label}     {scores}\n{row["smiles"]}', loc='left', fontsize=11, pad=9, wrap=True)
        ax.set_xlim(500, 4000)
        ax.set_ylim(-.04, 1.1)
        ax.grid(alpha=.15)
        ax.text(.02, .97, row['molecule_id'], transform=ax.transAxes, va='top', fontsize=8, color='#64748b')
    for ax in list(axes.flat)[len(rows):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Raman shift (cm$^{-1}$)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Relative intensity (display max = 1)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .035))
    fig.text(.055, .014, 'Gray: at least one domain is unmeasured. Scores use identical conservative support for all transforms; protected test excluded.', fontsize=9, color='#475569')
    fig.subplots_adjust(top=.81, bottom=.14, left=.055, right=.98, hspace=.40, wspace=.15)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('preview', 'fit'), default='preview')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ds = Dataset(args.dataset)
    sources = {}
    if args.mode == 'fit':
        train = [p for p in ds.pairs if p['role'] == ROLES[0]]
        for source in sorted(set(p['source'] for p in train)):
            print('Fitting ' + source, flush=True)
            sources[source] = fit_source(ds, [p for p in train if p['source'] == source])
            print(json.dumps({source: sources[source]}), flush=True)
    summary = {}
    for role in ROLES:
        pairs = [p for p in ds.pairs if p['role'] == role]
        rows = evaluate(ds, pairs, sources)
        selected = select_examples(rows)
        save_json(args.output / f'{role}_pairs.json', rows)
        save_json(args.output / f'{role}_examples.json', selected)
        plot_examples(ds, selected, args.output / f'{role}_original.png')
        if args.mode == 'fit':
            plot_examples(ds, selected, args.output / f'{role}_proxy.png', sources)
        summary[role] = {'overall': summarize(rows), 'by_source': {}, 'by_phase': {}}
        for field, key in [('source', 'by_source'), ('phase', 'by_phase')]:
            for value in sorted(set(r[field] for r in rows)):
                summary[role][key][value] = summarize([r for r in rows if r[field] == value])
    summary['audit'] = dict(mode=args.mode, protected_spectra_accessed=0,
                            seconds=time.monotonic() - started, dataset=str(args.dataset),
                            split_sha256=sha256(args.dataset / 'split.json'),
                            dataset_manifest_sha256=sha256(args.dataset / 'manifest.json'),
                            script_sha256=sha256(Path(__file__)),
                            weighting='equal groups, equal pair combinations within each group',
                            evaluation='masked cosine on fixed support valid for ALL bounded transforms; x>=500',
                            interpretation='curve-shape agreement, NOT molecular retrieval accuracy or physical parameter recovery')
    save_json(args.output / 'metrics.json', summary)
    if args.mode == 'fit':
        save_json(args.output / 'model.json', dict(schema='raman_curve_proxy_v1', sources=sources,
                  parameter_names=['scale', 'shift_cm-1', 'extra_sigma_theory_cm-1'], bounds=BOUNDS,
                  fitted_role=ROLES[0], fitted_parameters=3 * len(sources),
                  note='Deterministic effective curve proxy; no validation conditioning or per-pair optimization.'))
        # 冻结完整模型参数后关闭部分变换，用于诊断平滑贡献；不重新优化，也不据此选模型。
        ablations = {}
        for name in ('warp_only', 'broadening_only'):
            reduced = {}
            for source, item in sources.items():
                a, b, sigma = item['params']
                reduced[source] = {'params': (a, b, 0.) if name == 'warp_only' else (1., 0., sigma)}
            ablations[name] = {}
            for role in ROLES:
                rows = evaluate(ds, [p for p in ds.pairs if p['role'] == role], reduced)
                ablations[name][role] = {'overall': summarize(rows), 'by_source': {
                    source: summarize([r for r in rows if r['source'] == source]) for source in sources}}
        save_json(args.output / 'ablations.json', ablations)
    print(json.dumps({'completed': str(args.output), 'metrics': summary}), flush=True)


if __name__ == '__main__':
    main()

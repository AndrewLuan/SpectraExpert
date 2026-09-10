"""功能概述：在独立验证大候选库比较原始曲线与固定代理，补充训练对照的配对统计。
主要职责：main 只加载已校验的训练准备产物；masked_scores 在每个查询/候选共同覆盖
内计算余弦，两种曲线使用相同支持域；comparison 统计 @10 进出与组重采样置信区间。
代码库关系：依赖 raman_top10_experiment 的冻结数据、fit_raman_curve_proxy 的变换；
不访问实验测试谱或更新训练参数。--out 指向实验目录，新增 diagnostic.json，拒绝覆盖。
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.raman_top10_experiment import load, AXIS
from scripts.fit_raman_curve_proxy import transform
from scripts.run_closed_llm_direct import save


def masked_scores(q, qm, r, rm):
    q = np.where(qm, q, 0).astype(np.float32)
    r = np.where(rm, r, 0).astype(np.float32)
    denominator = np.sqrt(np.maximum((q*q) @ rm.astype(np.float32).T *
                         (qm.astype(np.float32) @ (r*r).T), 0))
    return q @ r.T / np.maximum(denominator, 1e-12)


def comparison(a, b, groups):
    delta = np.array([np.mean((a[np.array(groups) == g] <= 10).astype(float) -
                             (b[np.array(groups) == g] <= 10).astype(float)) for g in sorted(set(groups))])
    rng = np.random.default_rng(0)
    boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
    return dict(group_mean_delta=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
        gained=int(((a <= 10) & (b > 10)).sum()), lost=int(((a > 10) & (b <= 10)).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    destination = args.out / 'proxy_retrieval_diagnostic.json'
    if destination.exists():
        raise FileExistsError(destination)
    p, d = load(args.out)
    proxy = json.loads((args.out / 'proxy.json').read_text())
    n = p['candidate_count']; val = d['validation']
    t = d['theory'][:n]; tm = d['theory_mask'][:n]
    corrected = np.empty_like(t); valid = np.empty_like(tm)
    for source in sorted(set(p['candidate_sources'])):
        indices = np.flatnonzero(np.array(p['candidate_sources']) == source)
        values, masks = transform(AXIS, t[indices], tm[indices], proxy['sources'][source]['params'])
        corrected[indices] = values; valid[indices] = masks
    common = tm & valid
    targets = d['targets'][val]
    groups = [p['experimental_metadata'][i]['group_hash'] for i in val]
    result = {'support': 'same pairwise intersection for raw and proxy; no query fitting', 'methods': {}}
    ranks = {}
    for name, values in (('raw', t), ('fixed_proxy', corrected)):
        score = masked_scores(d['experiment'][val], d['experiment_mask'][val], values, common)
        order = np.argsort(-score, axis=1, kind='stable')
        rank = np.argmax(order == targets[:, None], axis=1) + 1
        ranks[name] = rank
        result['methods'][name] = dict(ranks=rank.tolist(),
            exact={str(k): float((rank <= k).mean()) for k in (1, 5, 10)})
    result['proxy_minus_raw'] = comparison(ranks['fixed_proxy'], ranks['raw'], groups)
    save(destination, result)
    print(json.dumps({k: v if k != 'methods' else {a: b['exact'] for a, b in v.items()} for k, v in result.items()}))


if __name__ == '__main__':
    main()

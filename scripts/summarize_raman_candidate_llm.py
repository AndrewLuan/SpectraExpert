"""功能概述：汇总完成的检索＋LLM 选择评测，与同一 200 条 raw LLM 基线配对比较。
主要职责：main 核对查询身份，计算共同可评分样本均值、逐查询 bootstrap 区间、
相对检索首位的新增/丢失命中及实际 API 返回模型统计；不发请求、不修改候选或预测。
代码库关系：读取 run_raman_candidate_llm 的完整产物、冻结检索运行及已有 raw baseline；
调用 evaluate_raman_domain.paired_bootstrap。--normalized 读取格式兼容评分；分别创建
对应目录的 analysis.json，拒绝覆盖结果。
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_closed_llm_direct import digest, save
from scripts.evaluate_raman_domain import paired_bootstrap


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--normalized', action='store_true')
    args = p.parse_args()
    out = args.out
    metric_root = out / 'format_normalized' if args.normalized else out
    if (metric_root / 'analysis.json').exists():
        raise FileExistsError(metric_root / 'analysis.json')
    config = json.loads((out / 'config.json').read_text())
    results = {m: json.loads((metric_root / m / 'results.json').read_text()) for m in ('model', 'raw_knn')}
    baseline = Path(config['raw_baseline']) / 'results.json'
    assert digest(baseline) == config['raw_baseline_results_sha256']
    results['raw_llm'] = json.loads(baseline.read_text())
    for r in results.values():
        assert len(r['queries']) == 200
        for row, other in zip(r['queries'], results['raw_llm']['queries']):
            assert all(row[k] == other[k] for k in ('mol_id', 'ground_truth_smiles', 'group_hash'))
    report = dict(common_means={}, paired_vs_raw_llm={}, versus_retrieval_top1={})
    for metric in ('valid', 'exact_match', 'tanimoto', 'legacy_ged_v1', 'chemical_ged_v2'):
        arrays = {m: np.array([np.nan if row['strict'][metric] is None else row['strict'][metric]
                              for row in r['queries']], dtype=float) for m, r in results.items()}
        mask = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
        report['common_means'][metric] = dict(n=int(mask.sum()),
            means={m: float(a[mask].mean()) if mask.any() else None for m, a in arrays.items()})
        report['paired_vs_raw_llm'][metric] = {
            m: paired_bootstrap(a.tolist(), arrays['raw_llm'].tolist(), larger_better='ged' not in metric)
            for m, a in arrays.items() if m != 'raw_llm'}
    for m in ('model', 'raw_knn'):
        rows = json.loads((metric_root / m / 'selections.json').read_text())
        retrieval = np.array([r['true_rank'] == 1 for r in rows])
        llm = np.array([r['exact'] for r in rows])
        report['versus_retrieval_top1'][m] = dict(retrieval_exact=int(retrieval.sum()),
            llm_exact=int(llm.sum()), gained=int((llm & ~retrieval).sum()), lost=int((~llm & retrieval).sum()))
    returned = Counter(); requested = Counter(); statuses = Counter()
    for f in sorted((out / 'attempts').glob('*.json')):
        r = json.loads(f.read_text())
        statuses[str(r.get('http_status', r.get('status', 'unknown')))] += 1
        body = r.get('response_json')
        if isinstance(body, dict) and body.get('model'):
            returned[body['model']] += 1
        request = Path(r['request_artifact']) if r.get('request_artifact') else out / 'requests' / f"{r['request_index']:04d}.json"
        requested[json.loads(request.read_text())['model']] += 1
    report['api_attempts'] = dict(requested_models=dict(requested), returned_models=dict(returned), statuses=dict(statuses))
    save(metric_root / 'analysis.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

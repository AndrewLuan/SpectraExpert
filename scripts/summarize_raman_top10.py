"""功能概述：汇总已完成的 Top10 四组实验，给出验证配对差值及历史固定测试比较。
主要职责：main 验证已有最终选择存在，读取各组冻结排名，调用组 bootstrap 比较
新增/丢失命中；按理论来源与相态拆分最终结果，不据测试指标修改模型或重新选型。
代码库关系：消费 raman_top10_experiment 产物和旧 C 的 topk_exact/per_query.json；
复用 diagnose_raman_top10.comparison。只新增 comparison.json，拒绝覆盖已有报告。
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.raman_top10_experiment import load, ARMS
from scripts.diagnose_raman_top10 import comparison
from scripts.run_closed_llm_direct import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--old', type=Path, required=True)
    args = p.parse_args()
    destination = args.out / 'comparison.json'
    if destination.exists():
        raise FileExistsError(destination)
    protocol, _ = load(args.out)
    selected = json.loads((args.out / 'final/selection.json').read_text())['selected_arm']
    validations = {a: json.loads((args.out / a / 'validation.json').read_text()) for a in ARMS}
    baseline = validations['baseline']
    report = dict(selected_arm=selected, validation={}, validation_vs_baseline={})
    for a, result in validations.items():
        assert result['groups'] == baseline['groups']
        assert result['experiment_indices'] == baseline['experiment_indices']
        report['validation'][a] = result['metrics']
        report['validation_vs_baseline'][a] = comparison(np.array(result['ranks']),
            np.array(baseline['ranks']), result['groups'])
    final = json.loads((args.out / 'final/per_query.json').read_text())
    old = {r['query_id']: r for r in json.loads(args.old.read_text())}
    prior = []
    for q in final['queries']:
        row = old[q['mol_id']]
        assert row['ground_truth_smiles'] == q['ground_truth_smiles']
        prior.append(row['model']['true_rank'])
    ranks = np.array(final['methods']['model']['ranks'])
    groups = [q['group_hash'] for q in final['queries']]
    report['test_vs_old_C'] = comparison(ranks, np.array(prior), groups)
    report['test_vs_knn'] = comparison(ranks, np.array(final['methods']['raw_knn']['ranks']), groups)
    names = {s: i for i, s in enumerate(protocol['candidate_names'])}
    sources = [protocol['candidate_sources'][names[q['ground_truth_smiles']]] for q in final['queries']]
    root = Path(protocol['data_root'])
    rows = {r['mol_id']: r for r in json.loads((root / 'experimental_test/rows.json').read_text())}
    conditions = [rows[q['mol_id']].get('condition', '') for q in final['queries']]
    report['test_breakdown'] = {}
    for field, values in (('source', sources), ('condition', conditions)):
        report['test_breakdown'][field] = {}
        for value in sorted(set(values)):
            mask = np.array(values) == value
            report['test_breakdown'][field][value] = dict(n=int(mask.sum()),
                count10=int((ranks[mask] <= 10).sum()), rate10=float((ranks[mask] <= 10).mean()),
                old_count10=int((np.array(prior)[mask] <= 10).sum()))
    save(destination, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""功能概述：只读审计前向 NMR 峰表缓存，区分积分不一致与来源结构标准化问题。
主要职责：source_fragment_candidate 仅比较结构及来源，识别被删去完整片段的候选；
audit_cache 按来源统计积分差、交换性氢容许量与整数比例，并保存固定顺序例子；
main 读取 prepare.py 生成的可信本地 torch 缓存，拒绝覆盖既有审计报告。
关系与契约：依赖 RDKit 和 PyTorch；不读取模型预测、不据测试残差挑样本、不修改
cache、源 JSONL、标签或 split。输出 JSON 供人工数据核查及保留原评估的敏感性分析。
输入输出与边界：torch.load(weights_only=False) 仅用于本项目生成的可信缓存。
N/O/S 氢数与整数总积分比只是诊断线索，不能证明交换、积分倍率或数据错误。
完整盐候选证明来源结构曾包含额外片段，但不能单独证明报告谱包含这些片段信号。
"""
import argparse
import collections
import json
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def source_fragment_candidate(record):
    """仅凭来源 SMILES 比较，返回完整片段删除候选；完全不访问积分或预测。"""
    provenance = (record.get('source_locator') or {}).get('provenance') or {}
    source_smiles = provenance.get('smiles_std')
    if not source_smiles:
        return None
    source = Chem.MolFromSmiles(source_smiles)
    selected = Chem.MolFromSmiles(record.get('smiles') or '')
    if source is None or selected is None:
        return None
    if len(Chem.GetMolFrags(selected)) != 1:
        return None
    fragments = Chem.GetMolFrags(source, asMols=True)
    if len(fragments) < 2:
        return None
    identity = Chem.MolToSmiles(selected, isomericSmiles=False)
    fragment_ids = [Chem.MolToSmiles(m, isomericSmiles=False) for m in fragments]
    matches = [i for i, key in enumerate(fragment_ids) if key == identity]
    if not matches:
        return None
    retained = matches[0]
    dropped = [m for i, m in enumerate(fragments) if i != retained]
    dropped_smiles = [Chem.MolToSmiles(m, isomericSmiles=True) for m in dropped]
    return {
        'record_id': record.get('record_id'), 'source': record.get('source'),
        'split': record.get('split'), 'selected_smiles': record['smiles'],
        'source_smiles_std': source_smiles,
        'source_canonical_smiles': Chem.MolToSmiles(source, isomericSmiles=True),
        'source_fragment_count': len(fragments),
        'matching_fragment_indices': matches, 'dropped_fragments': dropped_smiles,
        'dropped_has_carbon': any(a.GetAtomicNum() == 6 for m in dropped for a in m.GetAtoms()),
        'dropped_has_hydrogen': any(a.GetTotalNumHs() > 0 or a.GetAtomicNum() == 1
                                    for m in dropped for a in m.GetAtoms()),
        'provenance': provenance,
        'interpretation': 'Exact nonstereo fragment match proves omitted source structure; '
                          'full-salt spectral assignment still requires source evidence.',
    }


def _example(record, exchangeable_h, source_h, categories):
    """保存审计用的小证据块，不输出图张量、预测或任何误差排序。"""
    return {
        'record_id': record['record_id'], 'source': record['source'], 'split': record['split'],
        'smiles': record['smiles'], 'integral': record['integral'], 'structure_H': record['protons'],
        'delta': round(record['integral'] - record['protons'], 6),
        'potentially_exchangeable_NOSH_H': exchangeable_h, 'source_structure_H': source_h,
        'source_smiles_std': (record.get('source_locator') or {}).get('provenance', {}).get('smiles_std'),
        'categories': categories, 'flags': record.get('flags', []),
        'locator': record.get('source_locator'),
        'peaks': [{'shift': a['shift_mid'], 'integral': a['integration_h'], 'raw': a.get('raw')}
                  for a in record.get('annotations', [])],
    }


def audit_cache(data, examples_per_category=3, fragment_examples=7):
    """遍历完整缓存，返回 JSON 可序列化审计；类别可重叠，例子按输入顺序选取。"""
    counts = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    fragment_counts = collections.defaultdict(collections.Counter)
    fragment_rows, organic_fragment_rows = [], []
    for record in data:
        source = str(record.get('source'))
        c = counts[source]
        c['total'] += 1
        if 'provenance_structure_differs' in record.get('flags', []):
            c['provenance_structure_differs'] += 1
        candidate = source_fragment_candidate(record)
        if candidate:
            f = fragment_counts[source]
            f['exact_fragment_candidates'] += 1
            f['split:' + record['split']] += 1
            f['dropped_has_carbon'] += int(candidate['dropped_has_carbon'])
            f['dropped_has_hydrogen'] += int(candidate['dropped_has_hydrogen'])
            if len(fragment_rows) < fragment_examples:
                fragment_rows.append(candidate)
            if candidate['dropped_has_carbon'] and len(organic_fragment_rows) < fragment_examples:
                organic_fragment_rows.append(candidate)
        integral, hydrogens = record['integral'], record['protons']
        delta = round(integral - hydrogens, 6)
        if abs(delta) <= .25:
            continue
        c['mismatch'] += 1
        c['delta:' + str(delta)] += 1
        c['split:' + record['split']] += 1
        mol = Chem.MolFromSmiles(record['smiles'])
        exchangeable = sum(a.GetTotalNumHs() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8, 16))
        source_h = None
        provenance = (record.get('source_locator') or {}).get('provenance') or {}
        source_smiles = provenance.get('smiles_std')
        if source_smiles:
            source_mol = Chem.MolFromSmiles(source_smiles)
            if source_mol is not None:
                source_h = sum(a.GetTotalNumHs() for a in source_mol.GetAtoms())
        categories = []
        if delta < 0 and -delta <= exchangeable + .25:
            categories.append('exchangeable_H_deficit_compatible')
        if delta < 0 and -delta > exchangeable + .25:
            categories.append('deficit_exceeds_exchangeable_H')
        if delta > 0:
            categories.append('integral_excess')
        if source_h is not None and source_h != hydrogens:
            categories.append('source_structure_H_changed')
            if abs(source_h - integral) <= .25:
                categories.append('integral_matches_source_structure_H')
        if hydrogens and integral / hydrogens >= 2 and abs(integral / hydrogens - round(integral / hydrogens)) < 1e-6:
            categories.append('integer_multiple_integral')
        if integral and hydrogens / integral >= 2 and abs(hydrogens / integral - round(hydrogens / integral)) < 1e-6:
            categories.append('integer_fraction_integral')
        for category in categories:
            c[category] += 1
            key = source + '|' + category
            if len(examples[key]) < examples_per_category:
                examples[key].append(_example(record, exchangeable, source_h, categories))
    return {
        'definition': 'delta = reported integral sum minus structure total H; '
                      'diagnostic categories overlap and are not adjudicated errors',
        'counts_by_source': dict(counts),
        'rates_by_source': {s: {'mismatch_rate': c['mismatch'] / c['total'],
                                'mismatch_count': c['mismatch'], 'total': c['total']}
                           for s, c in counts.items()},
        'examples': dict(examples),
        'source_fragment_audit': {
            'selection_rule': 'Single-fragment selected structure equals one complete source fragment '
                              'under nonstereo canonical SMILES, and source has multiple fragments. '
                              'Selection does not inspect integral, spectrum or model error.',
            'counts_by_source': dict(fragment_counts),
            'first_examples': fragment_rows, 'first_organic_dropped_examples': organic_fragment_rows,
            'automatic_spectral_repair_proven': False,
        },
        'policy': 'Read-only full-cache audit; no removal or changes to labels/splits. No model prediction used.',
        'interpretation_boundaries': [
            'N/O/S-bound H count is a plausibility allowance, not proof of exchange or omitted signal.',
            'Integer total ratios do not establish global integral scaling.',
            'Exact source-fragment matches prove structure standardization dropped fragments; '
            'they do not prove all source fragments contributed reported peaks.',
            'Retain original holdout metrics. Any source-restored analysis must be separately versioned '
            'and applied consistently without test-residual selection.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--examples-per-category', type=int, default=3)
    parser.add_argument('--fragment-examples', type=int, default=7)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite audit report: ' + str(output))
    cache = Path(args.cache)
    data = torch.load(cache, weights_only=False, map_location='cpu')
    report = audit_cache(data, args.examples_per_category, args.fragment_examples)
    report['cache'] = str(cache.resolve())
    report['cache_bytes'] = cache.stat().st_size
    report['cache_records'] = len(data)
    report['args'] = vars(args)
    with output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'rates_by_source': report['rates_by_source'],
                      'fragment_counts': report['source_fragment_audit']['counts_by_source'],
                      'output': str(output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""功能概述：审计实验¹H原子归属，并生成保留原谱的可追溯更正副本。

inspect_record检查数组/原子编号契约、普通甲基内部标签及H键几何；甲基快速转动
是明确假设，CH2和跨母节点图对称不自动合并，静态理论甲基分裂不判为实验错标。
H_bond_upper_bound使用元素共价半径给H键长诊断设宽松上限，避免将Se-H套入C-H阈值。
find_repairs只比较train内部同结构、同编号、同位移多重集的2018/2024记录；仅接受
唯一的两位置交换，且2024归属和更正后归属均消除甲基冲突。理论值不参与更正。
apply_repair核验补丁前值、交叉版本证据及谱多重集，原位移数值/缺失掩码不变。
apply_reviewed_repair只接受已逐例复核的原报告约束；固定其他位点，枚举最多8个待审H
的原值排列，结构确认醛氢与报告明确指出的等价位点，仅在唯一赋值时生成补丁。
copy_and_verify按核种/ID复制更正副本并独立逐行回读；其余行按bytes保留。
main依赖data.identity的既有划分，母节点契约与h_labels.source_groups一致，仅读取
原始JSONL、可选重算理论及逐例复核约束JSON；写新目录，拒绝覆盖。test行逐字节复制，
所有未更正行逐字节复制。audit.jsonl和site_policy.jsonl供质量复核及后续训练显式
接入，不会自动替换现有缓存、checkpoint、候选库或检索查询，也不调用外部服务。
源文件前后SHA、逐行不变性、核种/位移/缺失值及更正证据均登记到summary.json。
"""
import argparse
import collections
import copy
import hashlib
import json
import itertools
from pathlib import Path

import numpy as np
from rdkit import Chem

from .data import identity


RULE = 'same_spectrum_cross_version_two_index_swap'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def spectrum_multiset(record):
    return sorted(float(v) for v, m in zip(record['atom_targets'], record['atom_target_mask']) if m)


def H_bond_upper_bound(parent_atomic_number):
    """几何诊断用宽松元素阈值，不用于重排原子、推断键型或修改化学位移。"""
    table = Chem.GetPeriodicTable()
    return max(1.5, 1.3 * (table.GetRcovalent(1) + table.GetRcovalent(parent_atomic_number)))


def inspect_record(r, theory=None):
    """返回源索引分组及独立质量证据；不改变记录，不用理论偏差决定标签更正。"""
    result = dict(record_id=r['record_id'], source=r['source_ref'],
                  nucleus=r['nucleus'], source_database_id=r.get('source_database_id'),
                  source_record_key=r.get('source_record_key'),
                  conditions=r.get('experimental_conditions') or r.get('conditions') or {},
                  formal_issues=[], methyl_conflicts=[], groups=[], geometry_issues=[])
    try:
        smi, group, split = identity(r['smiles'])
        result.update(smiles=smi, group=group, split=split)
    except ValueError as exc:
        result.update(split='excluded', identity_issue=str(exc))
    if result['split'] == 'test':
        result['classification'] = 'held_out_not_label_audited'
        return result
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(r['smiles']))
        order = r['atom_order']['source_to_canonical']
        numbers = r['atom_order']['atomic_numbers']
        n = mol.GetNumAtoms()
        if any(len(r[k]) != n for k in ('atoms', 'coordinates', 'atom_targets', 'atom_target_mask')):
            raise ValueError('array_length_mismatch')
        if len(order) != n or sorted(order) != list(range(n)):
            raise ValueError('source_order_not_permutation')
        if numbers != [mol.GetAtomWithIdx(i).GetAtomicNum() for i in order]:
            raise ValueError('source_order_element_mismatch')
        if r['atoms'] != [mol.GetAtomWithIdx(i).GetSymbol() for i in order]:
            raise ValueError('source_symbol_mismatch')
        values = r['atom_targets']; mask = r['atom_target_mask']
        wanted = 1 if r['nucleus'] == '1H' else 6
        for i, (v, active) in enumerate(zip(values, mask)):
            if active not in (0, 1):
                raise ValueError('invalid_mask')
            if active and (v is None or not np.isfinite(float(v)) or numbers[i] != wanted):
                raise ValueError('invalid_label_or_nucleus')
            if not active and v is not None:
                raise ValueError('unmasked_target_present')
        listed = [(int(x['atom_index']), float(x['shift'])) for x in r['spectrum']]
        if sorted(listed) != [(i, float(v)) for i, (v, m) in enumerate(zip(values, mask)) if m]:
            raise ValueError('spectrum_array_disagreement')
        reverse = {j: i for i, j in enumerate(order)}
        parents = collections.defaultdict(list)
        xyz = np.asarray(r['coordinates'], dtype=float)
        if xyz.shape != (n, 3) or not np.isfinite(xyz).all():
            raise ValueError('invalid_coordinates')
        heavy = [i for i, z in enumerate(numbers) if z != 1]
        for i, node in enumerate(order):
            atom = mol.GetAtomWithIdx(node)
            if atom.GetAtomicNum() != 1:
                continue
            if atom.GetDegree() != 1:
                raise ValueError('nonterminal_H')
            parent = reverse[atom.GetNeighbors()[0].GetIdx()]
            parents[parent].append(i)
            distances = np.linalg.norm(xyz[heavy] - xyz[i], axis=1)
            closest = heavy[int(np.argmin(distances))]
            bond = float(np.linalg.norm(xyz[parent] - xyz[i]))
            upper = H_bond_upper_bound(numbers[parent])
            if bond > upper or (closest != parent and bond - float(distances.min()) > .1):
                result['geometry_issues'].append(dict(source_H=i, source_parent=parent,
                    bond_distance_A=bond, closest_heavy=closest,
                    closest_distance_A=float(distances.min()),
                    upper_distance_A=upper,
                    kind='long_H_bond' if bond > upper else 'close_contact_not_reassignment'))
        theory_values = None
        if theory is not None:
            # 重算原子索引以1为起点，与源数组对应；结构一致后才能比较。
            try:
                if identity(theory['smiles'])[0] != result.get('smiles'):
                    raise ValueError('theory_structure_mismatch')
                theory_values = {int(i)-1: float(v) for i, v in
                                 zip(theory['value']['atom_index'], theory['value']['x'])}
                if not all(np.isfinite(v) for v in theory_values.values()):
                    raise ValueError('nonfinite_theory')
            except (ValueError, KeyError, TypeError) as exc:
                result['theory_diagnostic_issue'] = str(exc)
                theory_values = None
        residuals = []
        for parent, hs in parents.items():
            atom = mol.GetAtomWithIdx(order[parent])
            ordinary_methyl = (atom.GetAtomicNum() == 6 and len(hs) == 3
                and atom.GetFormalCharge() == 0
                and all(mol.GetAtomWithIdx(order[i]).GetIsotope() in (0, 1) for i in hs))
            selected = [i for i in hs if mask[i]]
            xs = [float(values[i]) for i in selected]
            span = max(xs)-min(xs) if xs else None
            g = dict(source_parent=parent, element=atom.GetSymbol(),
                source_H=hs, observed_source_H=selected, reported_ppm=xs,
                full_parent_observed=len(selected) == len(hs),
                ordinary_methyl=ordinary_methyl, span_ppm=span,
                supervision='unordered_parent_values',
                equivalence='fast_methyl_rotation_assumption' if ordinary_methyl else 'not_inferred')
            if ordinary_methyl and len(xs) >= 2 and span > .05:
                result['methyl_conflicts'].append(dict(source_parent=parent,
                    source_H=selected, reported_ppm=xs, span_ppm=span,
                    severity='large' if span > .2 else 'small'))
            if theory_values is not None and xs and len(selected) == len(hs) and all(i in theory_values for i in hs):
                tx = [theory_values[i] for i in hs]
                residual = float(np.mean(xs)-np.mean(tx))
                g.update(theory_parent_mean_ppm=float(np.mean(tx)),
                         theory_static_span_ppm=max(tx)-min(tx),
                         experiment_minus_theory_parent_mean_ppm=residual)
                residuals.append(abs(residual))
            elif theory_values is not None and xs and len(selected) != len(hs):
                g['theory_diagnostic_skipped'] = 'partial_parent_observation'
            result['groups'].append(g)
        result.update(observed_H=sum(mask), structural_H=sum(len(h) for h in parents.values()),
                      observed_multiset=spectrum_multiset(r))
        if residuals:
            result['raw_theory_parent_mean_MAE_ppm'] = float(np.mean(residuals))
            result['theory_interpretation'] = 'diagnostic_only_not_assignment_or_domain_shift_truth'
        if result.get('identity_issue'):
            result['classification'] = 'excluded_structure'
        elif any(x['kind'] == 'long_H_bond' for x in result['geometry_issues']):
            result['classification'] = 'coordinate_contract_conflict'
        elif result['methyl_conflicts']:
            result['classification'] = 'assignment_or_observation_semantics_review'
        else:
            result['classification'] = 'no_detected_local_conflict'
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        result['formal_issues'].append(str(exc))
        result['classification'] = 'formal_contract_error'
    return result


def max_methyl_span(audit):
    return max((g['span_ppm'] for g in audit['groups']
                if g['ordinary_methyl'] and g['span_ppm'] is not None), default=0.)


def find_repairs(records, audits):
    """仅train跨版本、唯一两标签交换，明确共享来源证据等级；不读取理论。"""
    donors = collections.defaultdict(list)
    for r in records:
        a = audits[r['record_id']]
        if r['source_ref'] == 'nmrshiftdb2_2024' and a['split'] == 'train' and a['classification'] == 'no_detected_local_conflict':
            donors[(a['smiles'], tuple(spectrum_multiset(r)))].append(r)
    patches = []
    for r in records:
        a = audits[r['record_id']]
        if r['source_ref'] != 'nmrshiftdb2_2018' or a['split'] != 'train' or max_methyl_span(a) <= .2:
            continue
        proposals = []
        for donor in donors.get((a['smiles'], tuple(spectrum_multiset(r))), []):
            if any(r[k] != donor[k] for k in ('smiles', 'atoms', 'atom_order', 'atom_target_mask')):
                continue
            if any(r.get(k) != donor.get(k) for k in ('conditions', 'experimental_conditions',
                    'solvent', 'frequency', 'temperature', 'pH', 'ph', 'pressure')):
                continue
            indices = [i for i, (v, w) in enumerate(zip(r['atom_targets'], donor['atom_targets'])) if v != w]
            if len(indices) != 2 or not all(r['atom_target_mask'][i] for i in indices):
                continue
            i, j = indices
            if r['atom_targets'][i] != donor['atom_targets'][j] or r['atom_targets'][j] != donor['atom_targets'][i]:
                continue
            trial = copy.deepcopy(r)
            trial['atom_targets'][i], trial['atom_targets'][j] = trial['atom_targets'][j], trial['atom_targets'][i]
            trial['spectrum'] = [dict(atom_index=k, shift=v) for k, (v, m) in
                                 enumerate(zip(trial['atom_targets'], trial['atom_target_mask'])) if m]
            after = inspect_record(trial)
            if after['classification'] != 'no_detected_local_conflict' or max_methyl_span(after) > .05:
                continue
            proposals.append(dict(rule=RULE, record_id=r['record_id'], split='train',
                donor_record_id=donor['record_id'], donor_source=donor['source_ref'],
                source_database_id=donor.get('source_database_id'),
                source_indices=indices, before=[r['atom_targets'][k] for k in indices],
                after=[donor['atom_targets'][k] for k in indices],
                before_max_methyl_span_ppm=max_methyl_span(a),
                after_max_methyl_span_ppm=max_methyl_span(after),
                confidence='cross_version_supported_shared_database',
                independent_measurement_confirmation=False,
                theoretical_values_used=False, spectrum_multiset_unchanged=True))
        # 多个不同修复提案时保留疑点，不按理论或模型分数选择。
        unique = {(tuple(p['source_indices']), tuple(p['after'])) for p in proposals}
        if len(unique) == 1:
            patches.append(proposals[0])
    return patches


def apply_repair(record, patch, donor):
    """重新从记录构造唯一证据提案，拒绝伪造补丁、test修改和位移值替换。"""
    audits = {r['record_id']: inspect_record(r) for r in (record, donor)}
    expected = find_repairs([record, donor], audits)
    if len(expected) != 1 or expected[0] != patch:
        raise ValueError('repair_evidence_mismatch')
    updated = copy.deepcopy(record)
    for i, before, after in zip(patch['source_indices'], patch['before'], patch['after']):
        if updated['atom_targets'][i] != before:
            raise ValueError('repair_before_mismatch')
        updated['atom_targets'][i] = after
    updated['spectrum'] = [dict(atom_index=i, shift=v) for i, (v, m) in
                          enumerate(zip(updated['atom_targets'], updated['atom_target_mask'])) if m]
    updated['assignment_correction'] = patch
    if spectrum_multiset(record) != spectrum_multiset(updated):
        raise ValueError('repair_changed_spectrum')
    return updated


def apply_reviewed_repair(record, review):
    """仅在固定其余位点的有限原值排列中求唯一解；约束及证据等级原样保存。"""
    if (record['nucleus'] != '1H' or identity(record['smiles'])[2] != 'train'
            or record['record_id'] != review['record_id']
            or record.get('source_database_id') != review['source_database_id']):
        raise ValueError('review_source_or_partition_mismatch')
    if (review.get('confidence') != 'source_report_structurally_supported'
            or review.get('independent_measurement_confirmation') is not False
            or len(set(review.get('reviewed_by', []))) < 2
            or not review.get('evidence_uri', '').startswith('https://')):
        raise ValueError('review_evidence_incomplete')
    indices = review['source_indices']; values = record['atom_targets']
    if (not 2 <= len(indices) <= 8 or len(set(indices)) != len(indices)
            or any(not 0 <= i < len(values) or not record['atom_target_mask'][i] for i in indices)
            or [values[i] for i in indices] != review['before']):
        raise ValueError('review_before_or_indices_mismatch')
    mol = Chem.AddHs(Chem.MolFromSmiles(record['smiles']))
    order = record['atom_order']['source_to_canonical']
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True)
    parents = {}
    for i in indices:
        atom = mol.GetAtomWithIdx(order[i])
        if atom.GetAtomicNum() != 1 or atom.GetDegree() != 1:
            raise ValueError('review_requires_terminal_H')
        parents[i] = atom.GetNeighbors()[0]
    c = review['constraints']; fi = c['formyl_source_H']
    if fi not in parents:
        raise ValueError('review_formyl_not_in_scope')
    parent = parents[fi]
    if (parent.GetAtomicNum() != 6 or sum(a.GetAtomicNum() == 1 for a in parent.GetNeighbors()) != 1
            or not any(b.GetBondType() == Chem.BondType.DOUBLE
            and b.GetOtherAtom(parent).GetAtomicNum() == 8 for b in parent.GetBonds())):
        raise ValueError('review_formyl_structure_mismatch')
    groups = c['equal_source_H_groups']
    if not groups:
        raise ValueError('review_equivalence_evidence_missing')
    for group in groups:
        if len(group) < 2 or any(i not in parents for i in group):
            raise ValueError('review_equivalence_scope')
        if not all(parents[i].GetIsAromatic() and parents[i].GetAtomicNum() == 6 for i in group):
            raise ValueError('review_equivalence_requires_reported_aromatic_sites')
        if len({ranks[parents[i].GetIdx()] for i in group}) != 1:
            raise ValueError('review_equivalence_structure_mismatch')
    candidates = set(itertools.permutations(review['before']))
    solutions = []
    for perm in candidates:
        trial = dict(zip(indices, perm))
        if trial[fi] != c['reported_formyl_ppm']:
            continue
        if all(len({trial[i] for i in group}) == 1 for group in groups):
            solutions.append(perm)
    if len(solutions) != 1:
        raise ValueError('review_assignment_not_unique')
    updated = copy.deepcopy(record)
    after = list(solutions[0])
    for i, value in zip(indices, after):
        updated['atom_targets'][i] = value
    updated['spectrum'] = [dict(atom_index=i, shift=v) for i, (v, m) in
                          enumerate(zip(updated['atom_targets'], updated['atom_target_mask'])) if m]
    proof = dict(review, rule='reviewed_source_report_unique_local_permutation',
        after=after, split='train', enumerated_distinct_assignments=len(candidates),
        satisfying_assignments=1, other_source_atoms_frozen=True,
        changed_source_indices=[i for i,b,a in zip(indices,review['before'],after) if b != a],
        theoretical_values_used=False, spectrum_multiset_unchanged=True)
    updated['assignment_correction'] = proof
    if spectrum_multiset(record) != spectrum_multiset(updated):
        raise ValueError('review_changed_spectrum')
    return updated


def copy_and_verify(source, output, corrections, audits):
    """按核种和ID应用更正；其余行按bytes复制，再独立回读校验变更字段及谱。"""
    counts = collections.Counter()
    with open(source, 'rb') as f, open(output, 'xb') as dest:
        for line in f:
            r = json.loads(line)
            corrected = corrections.get(r['record_id']) if r.get('nucleus') == '1H' else None
            if corrected is None:
                dest.write(line); counts['unchanged'] += 1
                if r.get('nucleus') == '1H' and audits[r['record_id']]['split'] == 'test':
                    counts['held_out'] += 1
            else:
                if audits[r['record_id']]['split'] != 'train':
                    raise ValueError('nontrain_correction')
                dest.write((json.dumps(corrected, ensure_ascii=False)+'\n').encode('utf-8'))
                counts['changed'] += 1
    checked = collections.Counter()
    with open(source, 'rb') as f, open(output, 'rb') as dest:
        for before, after in itertools.zip_longest(f, dest):
            if before is None or after is None:
                raise ValueError('corrected_source_row_count')
            r = json.loads(before)
            key = r['record_id']
            changed = r.get('nucleus') == '1H' and key in corrections
            if not changed:
                if before != after:
                    raise ValueError('unmodified_row_not_byte_identical')
                checked['unchanged'] += 1
                continue
            u = json.loads(after)
            differing = {k for k in r.keys() | u.keys() if r.get(k) != u.get(k)}
            if differing != {'atom_targets', 'spectrum', 'assignment_correction'}:
                raise ValueError('unexpected_correction_fields')
            if spectrum_multiset(r) != spectrum_multiset(u):
                raise ValueError('corrected_source_spectrum_changed')
            if inspect_record(u)['classification'] != 'no_detected_local_conflict':
                raise ValueError('corrected_record_failed_reaudit')
            checked['changed'] += 1
    if any(counts[k] != checked[k] for k in ('unchanged', 'changed')):
        raise ValueError('corrected_source_verification_counts')
    return dict(counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--theory')
    p.add_argument('--reviewed-patches')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    source = Path(args.source); inputs = {str(source): sha(source)}
    try:
        records = [r for r in map(json.loads, source.open()) if r.get('nucleus') == '1H']
        theory = {}
        if args.theory:
            inputs[args.theory] = sha(args.theory)
            eligible = set()
            for r in records:
                try:
                    if identity(r['smiles'])[2] in ('train', 'val'):
                        eligible.add(r['record_id'])
                except ValueError:
                    pass
            theory = {r['record_id']: r for r in map(json.loads, open(args.theory)) if r['record_id'] in eligible}
        audits = {r['record_id']: inspect_record(r, theory.get(r['record_id'])) for r in records}
        if len(audits) != len(records):
            raise ValueError('nonunique_source_record_id')
        patches = find_repairs(records, audits)
        lookup = {r['record_id']: r for r in records}
        corrections = {x['record_id']: apply_repair(lookup[x['record_id']], x,
                       lookup[x['donor_record_id']]) for x in patches}
        if args.reviewed_patches:
            inputs[args.reviewed_patches] = sha(args.reviewed_patches)
            reviewed = json.loads(Path(args.reviewed_patches).read_text())
            if reviewed['source_sha256'] != inputs[str(source)]:
                raise ValueError('review_source_sha_mismatch')
            for review in reviewed['patches']:
                key = review['record_id']
                if key in corrections:
                    raise ValueError('conflicting_correction_sources')
                corrections[key] = apply_reviewed_repair(lookup[key], review)
                patches.append(corrections[key]['assignment_correction'])
        with (out/'audit.jsonl').open('x') as f, (out/'site_policy.jsonl').open('x') as policy:
            for r in records:
                a = audits[r['record_id']]
                f.write(json.dumps(a)+'\n')
                if a['split'] not in ('train', 'val'):
                    continue
                current = inspect_record(corrections[r['record_id']]) if r['record_id'] in corrections else a
                keep = current['classification'] == 'no_detected_local_conflict'
                policy.write(json.dumps(dict(record_id=r['record_id'], source=r['source_ref'],
                    smiles=a.get('smiles'), split=a['split'], keep_spectral_supervision=True,
                    keep_atom_parent_supervision=keep,
                    quality=('cross_version_corrected' if corrections[r['record_id']]['assignment_correction']['rule'] == RULE
                        else 'source_report_corrected') if r['record_id'] in corrections else current['classification'],
                    within_parent_order='unordered', index_basis='source_array',
                    shift_values_imputed=False,
                    source_atom_groups=current['groups'] if keep else []))+'\n')
        output_counts = copy_and_verify(source, out/'atom_assigned.corrected.jsonl', corrections, audits)
        for path, digest in inputs.items():
            if sha(path) != digest:
                raise ValueError('source_changed_during_audit')
        buckets = collections.defaultdict(collections.Counter)
        spans = collections.defaultdict(list)
        for a in audits.values():
            key = a['source']+'|'+a['split']
            buckets[key][a['classification']] += 1
            if a['split'] not in ('train', 'val'):
                continue
            buckets[key]['methyl_conflicting_records'] += int(bool(a['methyl_conflicts']))
            buckets[key]['methyl_large_conflicting_records'] += int(any(x['span_ppm'] > .2 for x in a['methyl_conflicts']))
            buckets[key]['records_with_conditions'] += int(bool(a['conditions']))
            buckets[key]['records_with_theory'] += int('raw_theory_parent_mean_MAE_ppm' in a)
            for g in a['groups']:
                if g['ordinary_methyl'] and len(g['reported_ppm']) >= 2:
                    spans[key].append(g['span_ppm'])
        report = dict(status='complete', source_1H_records=len(records),
            partitions={k:dict(v) for k,v in buckets.items()},
            methyl_group_summary={k:dict(groups=len(v), over_005=sum(x>.05 for x in v),
                over_02=sum(x>.2 for x in v), median=float(np.median(v))) for k,v in spans.items()},
            automatic_correction_rule=RULE, correction_rules=sorted({x['rule'] for x in patches}),
            corrections=patches, corrected_records=output_counts.get('changed', 0),
            unchanged_source_lines_copied_verbatim=output_counts.get('unchanged', 0),
            held_out_1H_lines_copied_verbatim=output_counts.get('held_out', 0),
            output_independently_reread_verified=True,
            test_labels_used_for_audit_or_correction=False,
            input_sha256=inputs, source_inputs_unchanged=True,
            output_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file()},
            boundaries=dict(theory_is_diagnostic_only=True,
                conditionless_discrepancies='unresolved_not_confirmed_domain_shift',
                methyl_rule='ordinary fast rotation assumption, not universal labeling accuracy',
                graph_symmetry='no automatic chemical equivalence across parents',
                corrections='new derivative only; frozen caches/candidates/tests unchanged',
                training='policy sidecar written; no training or automatic cache replacement'))
        (out/'summary.json').write_text(json.dumps(report, indent=2))
        (out/'EXIT_CODE').write_text('0\n')
        print(json.dumps(report), flush=True)
    except Exception:
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

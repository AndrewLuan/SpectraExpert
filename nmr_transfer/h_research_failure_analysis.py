"""功能概述：只读汇总本轮验证集的召回、排序和条件覆盖失败，不参与方案选择。

输入是已完成 results 的真实 summary/manifest/recall 表及可选前向诊断。所有文件
均须明确 SHA，读取始末复核。只接受 original_val/tune；不提供 test/verify 入口。
逐 seed 保留精确失败计数；两 seed 先逐 query 平均 Top1/5/10 和倒数排名。固定
R/O/F 十二对、physical/local 与2048/8192两对只用于描述。固定哈希抽样不按误差
大小挑故事。正观测最近残差是 model-matched 支持距离，不是原子 assignment MAE，
也不直接证明溶剂、缺峰或 domain shift 的因果。未知频窗和未知池成员不补全。
"""
import argparse
import collections
import itertools
import json
import math
from pathlib import Path

import numpy as np

from . import h_research_results as results


SCHEMA = 'nmr_research_failure_analysis_20261022'
COHORTS = ('original_val', 'tune')
SAMPLE_SALT = 'failure_analysis_20261022_317'
SAMPLE_LIMIT = 3
SEMANTICS = ('assigned_labels', 'integral', 'line_positions', 'quantile_support')
META_FIELDS = ('semantics', 'coverage', 'missingness', 'solvent_status', 'solvent_family',
               'solvent_label', 'pH_known', 'pH_used_by_model', 'frequency_known', 'residual_bin')
FAILURES = ('recall_failure', 'ranking_failure_gt10', 'unattributed_gt10', 'top1', 'rank2_5', 'rank6_10')


def require(condition, message):
    if not condition:
        raise ValueError('failure_analysis_' + message)


def validation_path(path):
    for parts in (Path(path).parts, Path(path).resolve().parts):
        require(not any(part in ('test', 'verify', 'sealed') or part.startswith(('test_', 'verify_', 'sealed_'))
                        for part in parts), 'nonvalidation_artifact_path_forbidden')


def family_name(r, o, f):
    return f'factor_R{r}O{o}F{f}'


def contrast_plan():
    """每轴固定另两轴的四组，不根据验证表现选配对或方向。"""
    plan = []
    for axis in range(3):
        for fixed in itertools.product((0, 1), repeat=2):
            before = list(fixed); before.insert(axis, 0)
            after = before.copy(); after[axis] = 1
            left, right = family_name(*before), family_name(*after)
            plan.append(dict(id=f'{"ROF"[axis]}:{left}->{right}', axis='ROF'[axis], before=left, after=right))
    plan += [dict(id='F_representation:physical->local', axis='F_representation',
                  before='factor_R0O0F1', after='forward_local_slots'),
             dict(id='training_budget:2048->8192', axis='training_budget',
                  before='matched_budget_2048', after='factor_R0O0F0')]
    return plan


CORE_CI = tuple(p['id'] for p in contrast_plan() if p['before'] == 'factor_R0O0F0' or p['axis'] in
                ('F_representation', 'training_budget'))


def sample_key(row):
    return results.json_digest([SAMPLE_SALT, *results.query_key(row)])


def sample_identity(row):
    keep = ('cohort', 'source', 'record_id', 'smiles', 'group', 'variant', 'query_order', 'query_uid',
            'rank', 'target_in_coarse_pool', 'ranks_by_seed', 'membership_by_seed', 'before_ranks', 'after_ranks', 'delta')
    return {key: row[key] for key in keep if key in row}


def samples(rows):
    return [sample_identity(row) for row in sorted(rows, key=sample_key)[:SAMPLE_LIMIT]]


def failure_category(row):
    member = results.recall_evidence(row)
    if member is False:
        return 'recall_failure'
    if row['rank'] > 10:
        return 'ranking_failure_gt10' if member is True else 'unattributed_gt10'
    return 'top1' if row['rank'] == 1 else 'rank2_5' if row['rank'] <= 5 else 'rank6_10'


def residual_bin(value):
    if value is None:
        return 'unknown'
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid_positive_residual')
    for upper, label in ((.1, '<=0.1ppm'), (.25, '(0.1,0.25]ppm'), (.5, '(0.25,0.5]ppm'), (1., '(0.5,1]ppm')):
        if value <= upper:
            return label
    return '>1ppm'


def metadata(diagnostic=None, *, residual_bound=False):
    """仅取诊断中的显式观测/条件字段；缺诊断不凭variant或source猜测。"""
    if diagnostic is None:
        return dict.fromkeys(META_FIELDS, 'unknown') | dict(residual_mean_ppm=None)
    windows, partial = diagnostic.get('windows_ppm'), diagnostic.get('partial_peaks')
    require(isinstance(windows, list) and type(partial) is bool, 'invalid_observation_metadata')
    for window in windows:
        require(isinstance(window, (list, tuple)) and len(window) == 2 and
                all(type(v) in (int, float) and math.isfinite(v) for v in window) and window[0] < window[1],
                'invalid_explicit_window')
    semantics = diagnostic.get('semantics')
    require(semantics in SEMANTICS, 'unknown_observation_semantics')
    expected_coverage = 'explicit_window' if windows else 'declared_partial' if partial else 'unknown'
    require(diagnostic.get('coverage') == expected_coverage, 'inconsistent_coverage_metadata')
    output = dict(semantics=semantics, coverage=expected_coverage,
        missingness='window_and_partial' if windows and partial else 'explicit_window' if windows else
                    'explicit_partial' if partial else 'unknown_coverage',
        solvent_status=diagnostic.get('solvent_status', 'unknown'),
        solvent_family=diagnostic.get('solvent_family', 'unknown'),
        solvent_label=diagnostic.get('solvent') or 'unknown')
    require(output['solvent_status'] in ('reported', 'unknown'), 'invalid_solvent_status')
    for key, value_key in (('pH_known', 'pH'), ('frequency_known', 'frequency_MHz')):
        known, value = diagnostic.get(key), diagnostic.get(value_key)
        require(type(known) is bool, 'invalid_condition_knownness')
        require((known and type(value) in (int, float) and math.isfinite(value)) or
                (not known and value is None), 'inconsistent_condition_knownness')
        require(not known or key != 'frequency_known' or value > 0, 'invalid_MHz')
        output[key] = 'known' if known else 'unknown'
    used = diagnostic.get('pH_used_by_model')
    require(type(used) is bool and (not used or diagnostic['pH_known']), 'invalid_pH_used')
    output['pH_used_by_model'] = str(used).lower()
    residual = diagnostic.get('residuals', {})
    value = residual.get('per_query_weighted', {}).get('mean_ppm') if residual_bound else None
    if residual_bound:
        require(residual.get('atom_assignment_metric') is False, 'assignment_residual_not_supported')
    output.update(residual_mean_ppm=value, residual_bin=residual_bin(value))
    return output


def single_bucket(rows):
    counts = collections.Counter(failure_category(row) for row in rows)
    decomposition = {}
    for k in (1, 5, 10):
        failed = [row for row in rows if row['rank'] > k]
        recall = sum(results.recall_evidence(row) is False for row in failed)
        ranking = sum(results.recall_evidence(row) is True for row in failed)
        unknown = len(failed)-recall-ranking
        decomposition[str(k)] = dict(total_failures=len(failed), recall_failures=recall,
            recalled_ranking_failures=ranking, unknown_membership_failures=unknown,
            recall_fraction_of_failures=recall/len(failed) if failed else None)
    return dict(metrics=results.metrics(rows), failure_counts={key: counts[key] for key in FAILURES},
                Topk_failure_decomposition=decomposition,
                samples={key: samples([r for r in rows if failure_category(r) == key]) for key in FAILURES})


def grouped(rows, bucket):
    """variant各自完整分层，避免把正常查询和人工压力当成独立同分布混合总体。"""
    output = {}
    for variant in results.VARIANTS:
        part = [row for row in rows if row['variant'] == variant]
        strata = {}
        for field in ('source', 'cohort', *META_FIELDS):
            getter = (lambda row, key=field: row[key]) if field in ('source', 'cohort') else (lambda row, key=field: row['analysis_metadata'][key])
            strata[field] = {str(value): bucket([row for row in part if getter(row) == value])
                             for value in sorted({getter(row) for row in part}, key=str)}
        output[variant] = dict(overall=bucket(part), strata=strata)
    return output


def mean_query_rows(by_seed):
    keys, left, right = results._aligned(by_seed[193], by_seed[211])
    output = []
    for key in keys:
        a, b = left[key], right[key]
        ma, mb = a['analysis_metadata'], b['analysis_metadata']
        merged = {}
        for name in META_FIELDS:
            if name == 'residual_bin':
                continue
            require(ma[name] == mb[name] or 'unknown' in (ma[name], mb[name]), 'two_seed_query_metadata_mismatch')
            merged[name] = ma[name] if ma[name] == mb[name] else 'unknown'
        values = [ma['residual_mean_ppm'], mb['residual_mean_ppm']]
        mean_residual = float(np.mean(values)) if all(v is not None for v in values) else None
        meta = dict(merged, residual_mean_ppm=mean_residual, residual_bin=residual_bin(mean_residual))
        row = dict(a, analysis_metadata=meta,
            ranks_by_seed={'193': a['rank'], '211': b['rank']},
            membership_by_seed={'193': results.recall_evidence(a), '211': results.recall_evidence(b)},
            metric_mean=np.mean([results.metric_vector(a['rank']), results.metric_vector(b['rank'])], axis=0).tolist(),
            failure_pattern=failure_category(a)+'|'+failure_category(b))
        # 两seed平均不能保留一个seed的rank作为平均排名。
        row.pop('rank'); row.pop('target_in_coarse_pool', None)
        output.append(row)
    return output


def mean_bucket(rows):
    metrics = dict(queries=len(rows))
    metrics.update(dict(zip(results.METRICS, np.mean([row['metric_mean'] for row in rows], axis=0).tolist())) if rows else {})
    members = [list(row['membership_by_seed'].values()) for row in rows]
    known = bool(rows) and all(value is not None for pair in members for value in pair)
    metrics['recall512'] = float(np.mean(members)) if known else None
    conditional = []
    for seed in ('193', '211'):
        recalled = [1./row['ranks_by_seed'][seed] for row in rows if row['membership_by_seed'][seed] is True]
        conditional.append(float(np.mean(recalled)) if recalled else None)
    metrics['in_pool_MRR'] = float(np.mean(conditional)) if known and all(v is not None for v in conditional) else None
    metrics['final_MRR'] = metrics.get('MRR')
    patterns = collections.Counter(row['failure_pattern'] for row in rows)
    membership = collections.Counter('unknown' if any(v is None for v in pair) else
        'both_in' if all(pair) else 'both_out' if not any(pair) else 'mixed' for pair in members)
    return dict(metrics=metrics, failure_pattern_counts=dict(patterns), membership_pattern_counts=dict(membership),
        samples={key: samples([r for r in rows if r['failure_pattern'] == key]) for key in sorted(patterns)})


def effect_bucket(effects):
    delta = {name: float(np.mean([row['delta'][name] for row in effects])) if effects else None for name in results.METRICS}
    groups = {name: [row for row in effects if (row['delta']['MRR'] > 0 if name == 'improved' else
              row['delta']['MRR'] < 0 if name == 'regressed' else row['delta']['MRR'] == 0)] for name in ('improved', 'regressed', 'tied')}
    return dict(queries=len(effects), two_seed_query_metric_mean_delta=delta,
        direction_counts={name: len(part) for name, part in groups.items()}, samples={name: samples(part) for name, part in groups.items()})


def compare_families(before, after, *, ci=False):
    """先形成每query两个seed指标差的均值；MRR变化不改称位移误差变化。"""
    effects, seed_counts = [], {}
    indexed_seeds = {}
    for seed in results.SEEDS:
        keys, left, right = results._aligned(after[seed], before[seed])
        indexed_seeds[seed] = (keys, left, right)
        differences = [1./left[key]['rank']-1./right[key]['rank'] for key in keys]
        seed_counts[str(seed)] = dict(metric_delta={name: float(np.mean([
            results.metric_vector(left[k]['rank'])[i]-results.metric_vector(right[k]['rank'])[i] for k in keys])) if keys else None
            for i, name in enumerate(results.METRICS)},
            improved=sum(d > 0 for d in differences), regressed=sum(d < 0 for d in differences),
            tied=sum(d == 0 for d in differences), membership_transitions=dict(collections.Counter(
                str(results.recall_evidence(right[key]))+'->'+str(results.recall_evidence(left[key])) for key in keys)))
    require(indexed_seeds[193][0] == indexed_seeds[211][0], 'contrast_two_seed_query_mismatch')
    after_metadata = {results.query_key(r): r['analysis_metadata'] for r in mean_query_rows(after)}
    before_metadata = {results.query_key(r): r['analysis_metadata'] for r in mean_query_rows(before)}
    for key in indexed_seeds[193][0]:
        rows = [(indexed_seeds[seed][1][key], indexed_seeds[seed][2][key]) for seed in results.SEEDS]
        value = np.mean([results.metric_vector(a['rank'])-results.metric_vector(b['rank']) for a, b in rows], axis=0)
        for name in META_FIELDS:
            if name != 'residual_bin':
                a, b = after_metadata[key][name], before_metadata[key][name]
                require(a == b or 'unknown' in (a, b), 'contrast_query_metadata_mismatch')
        row = dict({name: rows[0][0][name] for name in
                    ('cohort', 'source', 'record_id', 'smiles', 'group', 'variant', 'query_order', 'query_uid')},
            analysis_metadata=after_metadata[key], delta=dict(zip(results.METRICS, value.tolist())),
            before_ranks={str(seed): rows[i][1]['rank'] for i, seed in enumerate(results.SEEDS)},
            after_ranks={str(seed): rows[i][0]['rank'] for i, seed in enumerate(results.SEEDS)},
            before_membership={str(seed): results.recall_evidence(rows[i][1]) for i, seed in enumerate(results.SEEDS)},
            after_membership={str(seed): results.recall_evidence(rows[i][0]) for i, seed in enumerate(results.SEEDS)})
        effects.append(row)
    report = dict(effect_bucket(effects), per_seed=seed_counts)
    report['strata'] = {}
    for field in ('source', 'cohort', *META_FIELDS):
        getter = (lambda row, key=field: row[key]) if field in ('source', 'cohort') else (lambda row, key=field: row['analysis_metadata'][key])
        report['strata'][field] = {str(value): effect_bucket([row for row in effects if getter(row) == value])
                                  for value in sorted({getter(row) for row in effects}, key=str)}
    if ci and effects:
        report['paired_cluster_CI'] = results.paired_cluster_bootstrap(after, before, repeats=5000, seed=317)
    return report, effects


class Snapshot:
    def __init__(self):
        self.inputs = {}

    def pin(self, path, expected=None):
        path = str(Path(path).resolve()); digest = results.sha(path)
        require(expected is None or digest == expected, 'input_SHA_mismatch:' + path)
        require(path not in self.inputs or self.inputs[path] == digest, 'input_changed_between_reads:' + path)
        self.inputs[path] = digest
        return path

    def json(self, path, expected=None):
        return results.read(self.pin(path, expected))

    def rows(self, path, expected):
        validation_path(path)
        self.pin(path, expected)
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    def complete(self, path, expected=None):
        require(Path(self.pin(path, expected)).read_text().strip() == '0', 'actual_EXIT_not_zero')

    def finish(self):
        for path, digest in self.inputs.items():
            require(results.sha(path) == digest, 'input_changed_during_read:' + path)


def load_results(spec, snapshot):
    """先核验证集计划，再读取逐query事实表；不加载bank或原始完整cohort pickle。"""
    require(spec.get('schema') == SCHEMA and spec.get('scope') == 'train_val_only', 'validation_only_manifest_required')
    for key in ('results_manifest', 'results_summary', 'recall_queries'):
        validation_path(spec[key])
    manifest = snapshot.json(spec['results_manifest'], spec['results_manifest_sha256'])
    results._scope_guard(manifest, 'train_val_only', None)
    require(set(manifest.get('expected_query_identity_sha256', {})) == set(COHORTS), 'fixed_validation_query_identities_required')
    for family in manifest['families']:
        for entry in family.get('entries', []):
            require(entry['cohort'] in COHORTS, 'nonvalidation_entry')
            for path in entry['predictions']:
                validation_path(path)
    summary = snapshot.json(spec['results_summary'], spec['results_summary_sha256'])
    require(summary.get('schema') == results.SCHEMA and summary.get('scope') == 'train_val_only', 'nonvalidation_results')
    require(summary['result_plan_sha256'] == results.result_plan_sha256(manifest), 'results_plan_mismatch')
    summary_inputs = summary['inputs_sha256']
    require(summary_inputs.get(str(Path(spec['results_manifest']).resolve())) == spec['results_manifest_sha256'], 'manifest_not_results_bound')
    for path, digest in summary_inputs.items():
        snapshot.pin(path, digest)
    snapshot.complete(spec.get('results_exit_code', Path(spec['results_summary']).parent/'EXIT_CODE'), spec.get('results_exit_code_sha256'))
    table = summary['recall_table']
    require(str(Path(table['path']).resolve()) == str(Path(spec['recall_queries']).resolve()) and
            table['sha256'] == spec['recall_queries_sha256'], 'recall_table_not_summary_bound')
    rows = snapshot.rows(spec['recall_queries'], spec['recall_queries_sha256'])
    require(type(table['rows']) is int and table['rows'] == len(rows), 'recall_table_count_mismatch')
    complete = {name for name, family in summary['families'].items() if family['state'] == 'complete'}
    require(type(summary['complete_families']) is int and summary['complete_families'] == len(complete), 'complete_count_mismatch')
    registered = snapshot.json(manifest['preregistered_arms'], manifest['preregistered_arms_sha256'])
    require(len(registered['families']) == 11 and set(summary['families']) == {f['name'] for f in registered['families']}, 'family_registry_mismatch')
    require(type(summary['all_families_complete']) is bool and summary['all_families_complete'] == (len(complete) == 11) and
            summary['selection_provisional'] is (len(complete) != 11), 'provisional_completeness_mismatch')
    declared = {family['name'] for family in manifest['families']}
    require(complete <= declared and {row['family'] for row in rows} == complete, 'complete_family_table_mismatch')
    by_family = {name: {seed: [] for seed in results.SEEDS} for name in complete}
    seen = set()
    for row in rows:
        require(row['cohort'] in COHORTS and type(row['seed']) is int and row['seed'] in results.SEEDS, 'nonvalidation_row_or_seed')
        require(row['source'] in manifest['expected_sources'] and row['variant'] in results.VARIANTS, 'invalid_source_or_variant')
        require(row['group'] == results.connectivity(row['smiles']), 'connectivity_mismatch')
        require(type(row['query_order']) is int and row['query_order'] >= 0, 'invalid_query_order')
        require(row['query_uid'] == results.json_digest(list(results.query_key(row))), 'query_UID_mismatch')
        results.metric_vector(row['rank']); member = results.recall_evidence(row)
        require(row['recall_status'] == ('known' if member is not None else 'unknown'), 'recall_status_mismatch')
        require(row['final_reciprocal_rank'] == 1./row['rank'] and
                row['in_pool_reciprocal_rank'] == (1./row['rank'] if member is True else None), 'reciprocal_rank_mismatch')
        key = (row['family'], row['seed'], results.query_key(row))
        require(key not in seen, 'duplicate_query_row'); seen.add(key)
        by_family[row['family']][row['seed']].append(row)
    for name, by_seed in by_family.items():
        results._aligned(by_seed[193], by_seed[211])
        for seed, items in by_seed.items():
            for cohort in COHORTS:
                part = [row for row in items if row['cohort'] == cohort]
                require({v: sum(r['variant'] == v for r in part) for v in results.VARIANTS} == manifest['expected_counts'][cohort], 'cohort_count_mismatch')
                expected = manifest.get('expected_query_identity_sha256', {}).get(cohort)
                require(expected is None or results.query_identity_sha256(part) == expected, 'query_identity_changed')
            results.assert_summary(summary['families'][name]['by_seed'][str(seed)], results.summaries(items))
            require(set(results.summaries(items)['by_source']) == set(manifest['expected_sources']), 'five_source_population_missing')
        mean = results._mean_summaries({s: results.summaries(by_seed[s]) for s in results.SEEDS}, manifest['expected_sources'])
        results.assert_summary(summary['families'][name]['mean'], mean)
        for field in ('recall512', 'in_pool_MRR', 'final_MRR'):
            require(summary['families'][name]['mean']['overall'].get(field) == mean['overall'][field], 'R_metric_mismatch')
    if by_family:
        first = next(iter(by_family.values()))
        for values in by_family.values():
            for seed in results.SEEDS:
                results._aligned(values[seed], first[seed])
    return manifest, summary, by_family


def diagnostic_digests(rows):
    rows = sorted(rows, key=lambda row: (row['query_order'], row['variant']))
    identities = [{key: row[key] for key in ('query_order', 'variant', 'source', 'record_id', 'smiles', 'group')} for row in rows]
    masks = [(tuple(row[key] for key in ('query_order', 'variant', 'source', 'record_id', 'smiles')),
              row['mask_content_sha256']) for row in rows]
    return dict(identity_content_sha256=results.json_digest(identities), variant_mask_content_sha256=results.json_digest(masks))


def load_diagnostics(entries, manifest, by_family, snapshot):
    output = {}
    for entry in entries:
        family, seed, cohort = entry['family'], entry['seed'], entry['cohort']
        require(cohort in COHORTS and family in by_family and type(seed) is int and seed in results.SEEDS, 'invalid_diagnostic_owner')
        for name in ('summary', 'protocol', 'per_query', 'exit_code'):
            validation_path(entry[name])
        summary = snapshot.json(entry['summary'], entry['summary_sha256'])
        require(summary.get('schema') == 'research_forward_diagnostics_v1' and summary['cohort'] == cohort and
                summary['candidate_order_sha256'] == manifest['candidate_order_sha256'], 'diagnostic_population_mismatch')
        require(summary['inputs_start_sha256'] == summary['inputs_end_sha256'], 'diagnostic_input_changed')
        protocol = snapshot.json(entry['protocol'], entry['protocol_sha256'])
        require(protocol['inputs_sha256'] == summary['inputs_start_sha256'] and protocol['args']['cohort'] == cohort,
                'diagnostic_protocol_mismatch')
        # 诊断完成产物本身及其SHA必核；历史bank只引用它已记录的始末指纹，不重新反序列化。
        snapshot.complete(entry['exit_code'], entry['exit_code_sha256'])
        require(entry['per_query_sha256'] == summary['per_query_sha256'], 'diagnostic_rows_not_summary_bound')
        rows = snapshot.rows(entry['per_query'], entry['per_query_sha256'])
        expected = {results.query_key(row): row for row in by_family[family][seed] if row['cohort'] == cohort}
        digests = diagnostic_digests(rows)
        require(all(summary.get(key) == value for key, value in digests.items()), 'diagnostic_identity_or_mask_digest_mismatch')
        owners = [e for f in manifest['families'] if f['name'] == family for e in f['entries']
                  if e['seed'] == seed and e['cohort'] == cohort]
        require(len(owners) == 1, 'diagnostic_retrieval_owner_missing')
        retrieval_args = None
        if owners[0]['adapter'] == 'research':
            retrieval = results.read(owners[0]['protocol']); retrieval_args = retrieval['args']
            cache_summary = str((Path(retrieval_args['cache']).parent/'summary.json').resolve())
            validation_path(cache_summary)
            require(cache_summary in retrieval['inputs_sha256'], 'retrieval_cache_summary_unbound')
            cache = snapshot.json(cache_summary, retrieval['inputs_sha256'][cache_summary])
            require(cache['cohort'] == cohort and cache['candidate_order_sha256'] == manifest['candidate_order_sha256'] and
                    cache['checkpoint_sha256'] == summary['checkpoint_sha256'] and
                    all(cache.get(key) == value for key, value in digests.items()), 'retrieval_diagnostic_observation_or_bank_mismatch')
            require(all(path not in summary['inputs_start_sha256'] or summary['inputs_start_sha256'][path] == digest
                        for path, digest in cache['source_inputs_sha256'].items()), 'retrieval_diagnostic_shared_input_mismatch')
            actual = protocol['args']
            required = [str(Path(actual['bank']).resolve()), str((Path(actual['cohorts'])/'cohorts.pt').resolve())]
            if actual.get('extension') is not None:
                required.append(str(Path(actual['extension']).resolve()))
            for path in required:
                digest = summary['inputs_start_sha256'].get(path)
                require(digest is not None and cache['source_inputs_sha256'].get(path) == digest,
                        'retrieval_diagnostic_required_bank_or_cohort_input_mismatch')
        actual = {}
        residual_bound = entry.get('forward_checkpoint_sha256') == summary['checkpoint_sha256']
        require(not entry.get('forward_checkpoint_sha256') or residual_bound, 'diagnostic_checkpoint_mismatch')
        if residual_bound:
            binding = entry.get('checkpoint_binding')
            require(binding is not None, 'residual_checkpoint_binding_required')
            checkpoint = str(Path(binding['path']).resolve())
            require(retrieval_args is not None, 'residual_requires_research_checkpoint_owner')
            require(checkpoint in {str(Path(retrieval_args[k]).resolve()) for k in ('no_aug_checkpoint', 'aug_checkpoint')}, 'residual_binding_owner_mismatch')
            snapshot.pin(checkpoint, binding['sha256'])
            import torch
            proof = torch.load(checkpoint, map_location='cpu', weights_only=False)
            require(type(proof['args'].get('seed')) is int and proof['args']['seed'] == seed and
                    proof['cache_protocol']['checkpoint_sha256'] == summary['checkpoint_sha256'] and
                    proof['cache_protocol']['candidate_order_sha256'] == manifest['candidate_order_sha256'],
                    'residual_checkpoint_binding_mismatch')
        for row in rows:
            require(row.get('cohort', cohort) == cohort, 'nonvalidation_diagnostic_row')
            row = dict(row, cohort=cohort)
            key = results.query_key(row)
            require(key in expected and key not in actual, 'diagnostic_query_mismatch_or_duplicate')
            require(row['query_order'] == expected[key]['query_order'] and row['group'] == expected[key]['group'], 'diagnostic_identity_mismatch')
            actual[key] = metadata(row, residual_bound=residual_bound)
        require(actual.keys() == expected.keys(), 'diagnostic_query_population_incomplete')
        for key, value in actual.items():
            owner = (family, seed, key)
            require(owner not in output, 'duplicate_diagnostic_owner')
            output[owner] = value
    return output


def analyze(by_family, *, diagnostics=None, core_ci=False):
    diagnostics = diagnostics or {}
    prepared, families = {}, {}
    for family, by_seed in sorted(by_family.items()):
        prepared[family] = {seed: [dict(row, analysis_metadata=diagnostics.get((family, seed, results.query_key(row)), metadata()))
                                 for row in rows] for seed, rows in by_seed.items()}
        combined = mean_query_rows(prepared[family])
        families[family] = dict(by_seed={str(seed): grouped(rows, single_bucket) for seed, rows in prepared[family].items()},
            two_seed_query_metric_mean=grouped(combined, mean_bucket))
    comparisons, query_effects = {}, []
    for plan in contrast_plan():
        if plan['before'] not in prepared or plan['after'] not in prepared:
            comparisons[plan['id']] = dict(plan, state='pending', missing_families=[name for name in (plan['before'], plan['after']) if name not in prepared])
            continue
        variants = {}
        for variant in results.VARIANTS:
            before = {s: [r for r in prepared[plan['before']][s] if r['variant'] == variant] for s in results.SEEDS}
            after = {s: [r for r in prepared[plan['after']][s] if r['variant'] == variant] for s in results.SEEDS}
            report, effects = compare_families(before, after, ci=core_ci and plan['id'] in CORE_CI)
            variants[variant] = report
            for effect in effects:
                query_effects.append(dict(contrast=plan['id'], **{key: value for key, value in effect.items() if key not in ('analysis_metadata', 'family', 'seed')}))
        comparisons[plan['id']] = dict(plan, state='complete', variants=variants)
    return dict(families=families, comparisons=comparisons, paired_query_rows=query_effects,
        selection_performed=False, sample_salt=SAMPLE_SALT, sample_limit_per_category=SAMPLE_LIMIT,
        averaging='per_query_Top1_Top5_Top10_and_reciprocal_rank_mean_before_aggregation',
        residual_scope='model-matched positive-support nearest residual; not assignment MAE or causal domain-shift evidence',
        coverage_scope='explicit windows and partial_peaks only; unknown coverage remains unknown',
        comparison_strata_metadata='after-family two-seed metadata; known conflicts with before-family metadata rejected',
        diagnostic_history_inputs='completed diagnostic start/end fingerprints referenced; historical banks are not reloaded or rehashed',
        CI_scope=dict(enabled=bool(core_ci), fixed_pairs=list(CORE_CI), repetitions=5000, bootstrap_seed=317,
                      selection_use=False, metric_names=list(results.METRICS)))


def run(manifest_path):
    snapshot = Snapshot()
    spec = snapshot.json(manifest_path)
    snapshot.pin(__file__); snapshot.pin(results.__file__)
    manifest, summary, by_family = load_results(spec, snapshot)
    diagnostics = load_diagnostics(spec.get('diagnostics', []), manifest, by_family, snapshot)
    require(type(spec.get('core_ci', False)) is bool, 'invalid_CI_flag')
    report = analyze(by_family, diagnostics=diagnostics, core_ci=spec.get('core_ci', False))
    snapshot.finish()
    return dict(schema=SCHEMA, scope='train_val_only', candidate_count=114982,
        candidate_order_sha256=manifest['candidate_order_sha256'], expected_sources=manifest['expected_sources'],
        available_complete_families=sorted(by_family), all_families_complete=summary['all_families_complete'],
        selection_provisional=summary['selection_provisional'], **report,
        inputs_start_sha256=snapshot.inputs, inputs_end_sha256=dict(snapshot.inputs), external_api_calls=0)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args.manifest)
        facts = report.pop('paired_query_rows')
        path = out/'paired_query_effects.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in facts))
        report['paired_query_table'] = dict(path=str(path.resolve()), sha256=results.sha(path), rows=len(facts))
        (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        (out/'EXIT_CODE').write_text('0\n')
    except Exception:
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

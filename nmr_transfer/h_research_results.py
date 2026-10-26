"""功能概述：汇总本轮冻结口径的检索结果，并计算连接结构聚类的配对置信区间。

本模块只在CPU读取已完成预测、真实summary、EXIT及其SHA，不训练、不调用网络。
validation按original_val320+tune1024先合并查询，再对193/211两seed取指标均值；
仅完整双seed家族参与晋升。晋升同时检查normal MRR、Top1、两类缺测MRR和
五来源宏MRR。统计从逐查询rank重算，不能只信任summary或目录中的seed名称。
新结果核protocol及实际CPU checkpoint的seed、模型角色和summary权重；旧结果通过明确legacy证据适配。
Recall@512只接受逐query的明确池成员证据；未知不补零，也不从最终rank反推召回。

paired_cluster_bootstrap以非立体连接结构为抽样单位。同结构多谱整组保留；两seed
平均先在相同query上平均Top1/5/10及倒数排名，再重采样，绝不先平均rank。
CLI默认仅train/val；test/verify须本轮freeze锁定结果计划和本模块SHA，且关闭选型。
result_plan_sha256排除运行后才产生的输出SHA值，但保留全部路径、角色和科学配置；
实际输出SHA另在读取始末核验。完整/partial/failed分开记录，未知EXIT不是完成。
"""
import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem


SCHEMA = 'nmr_research_results_20261022'
METRICS = ('top1', 'top5', 'top10', 'MRR')
VARIANTS = ('observed', 'frequency_crop', 'peak_dropout')
SEEDS = (193, 211)
VAL_COUNTS = {'original_val': 320, 'tune': 1024}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def result_plan_sha256(manifest):
    """输出内容SHA不可在运行前预知；其余manifest全部是需要预封存的计划。"""
    plan = copy.deepcopy(manifest)
    for family in plan['families']:
        for entry in family.get('entries', []):
            entry.pop('inputs_sha256', None)
    return json_digest(plan)


def pointer(value, path):
    for name in path:
        value = value[name]
    return value


@lru_cache(maxsize=200000)
def connectivity(smiles):
    """与data.identity的非立体连接结构SHA口径一致，不导入GPU/图训练依赖。"""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('research_results_invalid_smiles')
    molecule = Chem.RemoveHs(molecule)
    if '.' in Chem.MolToSmiles(molecule):
        raise ValueError('research_results_multicomponent_smiles')
    return hashlib.sha256(Chem.MolToSmiles(molecule, isomericSmiles=False).encode()).hexdigest()


def query_key(row):
    return (row['cohort'], row['source'], json.dumps(row['record_id'], sort_keys=True),
            row['smiles'], row['variant'])


def indexed(rows):
    output = {}
    for row in rows:
        key = query_key(row)
        if key in output:
            raise ValueError('research_results_duplicate_query_identity')
        output[key] = row
    return output


def query_identity_sha256(rows):
    """只封存查询身份及变体，不含rank、Top10或预测值。"""
    return json_digest([list(key) for key in sorted(indexed(rows))])


def metric_vector(rank):
    if isinstance(rank, bool) or not isinstance(rank, (int, float)) or not math.isfinite(rank) or int(rank) != rank or not 1 <= rank <= 114982:
        raise ValueError('research_results_invalid_rank')
    return np.asarray([rank <= 1, rank <= 5, rank <= 10, 1. / rank], dtype=float)


def recall_evidence(row):
    """池成员标记和512池大小均明确时才可辨识召回；最终排名不能代替该证据。"""
    member = row.get('target_in_coarse_pool')
    size = row.get('evaluation_pool_size')
    if member is not None and type(member) is not bool:
        raise ValueError('research_results_invalid_pool_membership')
    if size is not None and (type(size) is not int or size != 512):
        raise ValueError('research_results_invalid_evaluation_pool_size')
    if member is None or size is None:
        return None
    if member != (row['rank'] <= size):
        raise ValueError('research_results_pool_membership_rank_inconsistent')
    return member


def recall_metrics(rows):
    evidence = [recall_evidence(row) for row in rows]
    known = sum(value is not None for value in evidence)
    included = [row for row, value in zip(rows, evidence) if value is True]
    complete = bool(rows) and known == len(rows)
    return dict(recall512=len(included)/len(rows) if complete else None,
        in_pool_MRR=float(np.mean([1./row['rank'] for row in included])) if complete and included else None,
        recall_diagnostics=dict(state='supported' if complete else 'partial' if known else 'unsupported',
            known_queries=known, known_in_pool_queries=len(included), unknown_queries=len(rows)-known,
            pool_size=512, in_pool_MRR_definition='conditional_mean_reciprocal_final_rank_among_recalled_queries'))


def metrics(rows):
    if not rows:
        return dict(queries=0, final_MRR=None, **recall_metrics(rows))
    values = np.asarray([metric_vector(row['rank']) for row in rows])
    output = dict(queries=len(rows), **dict(zip(METRICS, values.mean(0).tolist())))
    return dict(output, final_MRR=output['MRR'], **recall_metrics(rows))


def recall_query_row(family, seed, row):
    """导出事实表，不补正例、不修改排名；唯一身份中包含cohort及压力变体。"""
    member = recall_evidence(row)
    return dict(family=family, seed=seed, cohort=row['cohort'], source=row['source'],
        record_id=row['record_id'], smiles=row['smiles'], group=row['group'],
        variant=row['variant'], query_order=row.get('query_order'),
        query_uid=json_digest(list(query_key(row))), rank=row['rank'],
        evaluation_pool_size=row.get('evaluation_pool_size'), target_in_coarse_pool=member,
        recall_status='known' if member is not None else 'unknown',
        in_pool_reciprocal_rank=1./row['rank'] if member is True else None,
        final_reciprocal_rank=1./row['rank'])


def summaries(rows):
    normal = [row for row in rows if row['variant'] == 'observed']
    return dict(overall=metrics(normal),
        by_source={source: metrics([r for r in normal if r['source'] == source])
                   for source in sorted({r['source'] for r in normal})},
        stress={variant: metrics([r for r in rows if r['variant'] == variant])
                for variant in VARIANTS[1:]})


def assert_summary(actual, computed):
    """逐个normal/source/stress四指标核验真实产物，不信任手写汇总。"""
    for key in ('overall', 'by_source', 'stress'):
        if key not in actual:
            raise ValueError('research_results_summary_missing:' + key)
    if set(actual['by_source']) != set(computed['by_source']):
        raise ValueError('research_results_summary_source_mismatch')
    pairs = [(actual['overall'], computed['overall'])]
    pairs += [(actual[k][name], value) for k in ('by_source', 'stress')
              for name, value in computed[k].items()]
    for saved, expected in pairs:
        if isinstance(saved.get('queries'), bool) or not isinstance(saved.get('queries'), int) or saved.get('queries') != expected['queries']:
            raise ValueError('research_results_summary_query_count_mismatch')
        for name in METRICS if expected['queries'] else ():
            value = saved.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or not math.isclose(value, expected[name], abs_tol=1e-12, rel_tol=1e-12)):
                raise ValueError('research_results_summary_metric_mismatch:' + name)


def _aligned(current, previous):
    left, right = indexed(current), indexed(previous)
    if left.keys() != right.keys():
        raise ValueError('research_results_paired_query_mismatch')
    keys = sorted(left)
    for key in keys:
        if left[key]['group'] != connectivity(left[key]['smiles']) or right[key]['group'] != connectivity(right[key]['smiles']) or left[key]['group'] != right[key]['group']:
            raise ValueError('research_results_paired_connectivity_mismatch')
        if ('query_order' in left[key] and 'query_order' in right[key] and
                left[key]['query_order'] != right[key]['query_order']):
            raise ValueError('research_results_paired_query_order_mismatch')
    return keys, left, right


def paired_cluster_bootstrap(current_by_seed, previous_by_seed, *, repeats=5000, seed=317):
    """返回各seed及逐query指标先平均后的两seed差值；同一cluster draw覆盖全部指标。"""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 5000:
        raise ValueError('research_results_bootstrap_requires_at_least_5000')
    if set(current_by_seed) != set(previous_by_seed) or not set(current_by_seed) <= set(SEEDS) or not current_by_seed or any(type(s) is not int for s in current_by_seed):
        raise ValueError('research_results_bootstrap_seed_mismatch')
    differences, clusters, first_keys = [], None, None
    for model_seed in sorted(current_by_seed):
        keys, left, right = _aligned(current_by_seed[model_seed], previous_by_seed[model_seed])
        if not keys:
            raise ValueError('research_results_bootstrap_empty')
        groups = [left[key]['group'] for key in keys]
        if first_keys is not None and (keys != first_keys or groups != clusters):
            raise ValueError('research_results_two_seed_query_mismatch')
        first_keys, clusters = keys, groups
        differences.append(np.asarray([metric_vector(left[key]['rank']) - metric_vector(right[key]['rank']) for key in keys]))
    unique = sorted(set(clusters)); lookup = {group: i for i, group in enumerate(unique)}
    ids = np.asarray([lookup[group] for group in clusters])
    counts = np.bincount(ids, minlength=len(unique))
    matrix = np.stack(differences + [np.mean(differences, axis=0)])
    sums = np.stack([np.stack([np.bincount(ids, weights=part[:, i], minlength=len(unique))
                              for i in range(len(METRICS))], 1) for part in matrix])
    draws = np.empty((len(matrix), repeats, len(METRICS)))
    random = np.random.default_rng(seed)
    for start in range(0, repeats, 256):
        chosen = random.integers(0, len(unique), (min(256, repeats-start), len(unique)))
        denominator = counts[chosen].sum(1)
        draws[:, start:start+len(chosen), :] = sums[:, chosen, :].sum(2) / denominator[None, :, None]
    result = {}
    labels = [str(s) for s in sorted(current_by_seed)] + ['two_seed_query_mean' if len(differences) == 2 else 'query_mean']
    for i, label in enumerate(labels):
        result[label] = {name: dict(difference=float(matrix[i, :, j].mean()),
            cluster_95_percentile_interval=np.quantile(draws[i, :, j], [.025, .975]).tolist())
            for j, name in enumerate(METRICS)}
    return dict(queries=len(first_keys), connectivity_clusters=len(unique), repeats=repeats, bootstrap_seed=seed,
                sampling_unit='nonstereochemical_connectivity', averaging='per_query_metric_before_cluster_resampling', metrics=result)


def _entry_paths(entry):
    paths = list(entry['predictions']) + [entry['summary'], entry['exit_code']]
    if entry['adapter'] == 'research':
        paths.append(entry['protocol'])
        if Path(entry['protocol']).is_file():
            args = read(entry['protocol'])['args']
            paths.extend(args[name] for name in ('no_aug_checkpoint', 'aug_checkpoint'))
    elif entry['adapter'] == 'legacy':
        for evidence in entry['seed_evidence']:
            paths.append(evidence['checkpoint'])
            if evidence.get('config'):
                paths.append(evidence['config'])
    else:
        raise ValueError('research_results_unknown_adapter')
    paths = [str(Path(path).resolve()) for path in paths]
    if len(set(map(str, entry['predictions']))) != len(entry['predictions']) or not entry['predictions']:
        raise ValueError('research_results_invalid_prediction_files')
    return sorted(set(paths))


@lru_cache(maxsize=128)
def _checkpoint_args(checkpoint, digest):
    """只缓存实际权重内的小型配置；参数张量在函数返回后释放。digest是缓存身份。"""
    import torch
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    args = payload.get('args')
    if not isinstance(args, dict):
        raise ValueError('research_results_checkpoint_args_missing')
    return dict(args)


def _seed_proof(entry, summary):
    expected = entry['seed']
    if entry['adapter'] == 'research':
        protocol = read(entry['protocol'])
        args = protocol['args']
        if type(args.get('seed')) is not int or args.get('seed') != expected or args.get('stage') != 'evaluate':
            raise ValueError('research_results_protocol_seed_or_stage_mismatch')
        pair = [(role, str(Path(args[role+'_checkpoint']).resolve())) for role in ('no_aug', 'aug')]
        checkpoints = {path for _, path in pair}
        if checkpoints != set(summary['checkpoints']) or len(checkpoints) != 2:
            raise ValueError('research_results_protocol_checkpoint_mismatch')
        for role, path in pair:
            digest = sha(path)
            if protocol['inputs_sha256'].get(path) != summary['checkpoints'][path] or digest != summary['checkpoints'][path]:
                raise ValueError('research_results_checkpoint_SHA_mismatch')
            config_args = _checkpoint_args(path, digest)
            if (type(config_args.get('seed')) is not int or config_args.get('seed') != expected or
                    config_args.get('augment') is not (role == 'aug')):
                raise ValueError('research_results_checkpoint_seed_or_role_mismatch')
    else:
        evidence = entry['seed_evidence']
        if len(evidence) != 2 or {item['role'] for item in evidence} != {'no_aug', 'aug'}:
            raise ValueError('research_results_legacy_requires_ordered_pair_evidence')
        for item in evidence:
            role = item['role']; checkpoint = str(Path(item['checkpoint']).resolve())
            digest = sha(checkpoint)
            if (str(Path(summary[role+'_checkpoint']).resolve()) != checkpoint or
                    summary[role+'_sha256'] != digest):
                raise ValueError('research_results_legacy_checkpoint_mismatch')
            args = _checkpoint_args(checkpoint, digest)
            if item.get('config'):
                config = read(item['config'])
                if (config.get('checkpoint_sha256') != digest or
                        str(Path(config['checkpoint']).resolve()) != checkpoint):
                    raise ValueError('research_results_legacy_config_unbound')
                if any(config['args'].get(field) != args.get(field) for field in ('seed', 'augment')):
                    raise ValueError('research_results_legacy_config_actual_checkpoint_mismatch')
            if type(args.get('seed')) is not int or args.get('seed') != expected or args.get('augment') is not (role == 'aug'):
                raise ValueError('research_results_legacy_seed_or_role_mismatch')


def load_entry(entry, manifest):
    if not Path(entry['exit_code']).is_file():
        return dict(state='pending', reason='actual_exit_missing'), None, {}
    exit_text = Path(entry['exit_code']).read_text().strip()
    if exit_text != '0':
        known = exit_text.lstrip('-').isdigit() and str(int(exit_text)) == exit_text
        return dict(state='failed' if known else 'pending', reason='actual_nonzero_exit' if known else 'unknown_exit',
                    actual_exit_code=int(exit_text) if known else None), None, {str(Path(entry['exit_code']).resolve()): sha(entry['exit_code'])}
    paths = _entry_paths(entry)
    if any(not Path(path).is_file() for path in paths):
        return dict(state='pending', reason='required_artifact_missing'), None, {}
    pinned = entry['inputs_sha256']
    actual = {path: sha(path) for path in paths}
    if any(pinned.get(path) != digest for path, digest in actual.items()):
        raise ValueError('research_results_input_SHA_mismatch')
    summary = read(entry['summary'])
    if summary.get('cohort') != entry['cohort']:
        raise ValueError('research_results_summary_cohort_mismatch')
    if summary['candidate_order_sha256'] != manifest['candidate_order_sha256']:
        raise ValueError('research_results_candidate_order_mismatch')
    _seed_proof(entry, summary)
    rows = []
    for path in entry['predictions']:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if entry['adapter'] == 'legacy':
                row.setdefault('variant', 'observed')
            if (row.get('variant') not in VARIANTS or row.get('source') not in manifest['expected_sources'] or
                    row.get('cohort', entry['cohort']) != entry['cohort'] or row.get('seed', entry['seed']) != entry['seed']):
                raise ValueError('research_results_prediction_metadata_mismatch')
            rank = row.get('rank')
            if isinstance(rank, bool) or not isinstance(rank, (int, float)) or not math.isfinite(rank) or int(rank) != rank or not 1 <= rank <= manifest['candidate_count']:
                raise ValueError('research_results_invalid_rank')
            group = connectivity(row['smiles'])
            if row.get('group', group) != group:
                raise ValueError('research_results_prediction_connectivity_mismatch')
            rows.append(dict(row, cohort=entry['cohort'], rank=int(rank), group=group))
    indexed(rows)
    expected_identity = manifest.get('expected_query_identity_sha256', {}).get(entry['cohort'])
    if expected_identity is not None and query_identity_sha256(rows) != expected_identity:
        raise ValueError('research_results_frozen_query_identity_changed')
    counts = {variant: sum(row['variant'] == variant for row in rows) for variant in VARIANTS}
    if counts != manifest['expected_counts'][entry['cohort']]:
        raise ValueError('research_results_prediction_count_mismatch')
    computed = summaries(rows)
    assert_summary(pointer(summary, entry.get('summary_metrics_pointer', ['metrics'])), computed)
    return dict(state='complete', actual_exit_code=0, counts=counts, metrics=computed), rows, actual


def promotion_guards(candidate, incumbent):
    return dict(normal_MRR=candidate['overall']['MRR'] > incumbent['overall']['MRR'],
        normal_top1=candidate['overall']['top1'] >= incumbent['overall']['top1'],
        crop_MRR=candidate['stress']['frequency_crop']['MRR'] >= incumbent['stress']['frequency_crop']['MRR'],
        dropout_MRR=candidate['stress']['peak_dropout']['MRR'] >= incumbent['stress']['peak_dropout']['MRR'],
        five_source_macro_MRR=candidate['five_source_macro_MRR'] >= incumbent['five_source_macro_MRR'])


def _mean_summaries(per_seed, sources):
    def mean_metrics(parts):
        if len({part['queries'] for part in parts}) != 1:
            raise ValueError('research_results_seed_metric_count_mismatch')
        output = (dict(queries=parts[0]['queries'], **{name: float(np.mean([part[name] for part in parts])) for name in METRICS})
                  if parts[0]['queries'] else {'queries': 0})
        output['final_MRR'] = output.get('MRR')
        for name in ('recall512', 'in_pool_MRR'):
            output[name] = (float(np.mean([part[name] for part in parts]))
                            if all(part.get(name) is not None for part in parts) else None)
        states = [part.get('recall_diagnostics', {}).get('state', 'unsupported') for part in parts]
        output['recall_diagnostics'] = dict(
            state='supported' if all(s == 'supported' for s in states) else
                  'unsupported' if all(s == 'unsupported' for s in states) else 'partial',
            averaging='arithmetic_mean_of_per_seed_metrics',
            by_seed={str(seed): part.get('recall_diagnostics', dict(state='unsupported'))
                     for seed, part in zip(SEEDS, parts)})
        return output
    values = [per_seed[seed] for seed in SEEDS]
    result = dict(overall=mean_metrics([v['overall'] for v in values]),
        by_source={source: mean_metrics([v['by_source'].get(source, {'queries': 0}) for v in values]) for source in sources},
        stress={variant: mean_metrics([v['stress'][variant] for v in values]) for variant in VARIANTS[1:]})
    result['five_source_macro_MRR'] = (float(np.mean([result['by_source'][source]['MRR'] for source in sources]))
                                      if all(result['by_source'][source]['queries'] for source in sources) else None)
    return result


def _scope_guard(manifest, scope, frozen_selection):
    if manifest.get('schema') != SCHEMA or manifest.get('scope') != scope:
        raise ValueError('research_results_manifest_schema_or_scope')
    if set(manifest['seeds']) != set(SEEDS) or len(manifest['seeds']) != 2 or any(type(s) is not int for s in manifest['seeds']):
        raise ValueError('research_results_requires_two_distinct_seeds')
    if len(manifest['expected_sources']) != 5 or len(set(manifest['expected_sources'])) != 5 or any(not isinstance(s, str) or not s for s in manifest['expected_sources']) or manifest['candidate_count'] != 114982:
        raise ValueError('research_results_population_contract')
    if scope == 'train_val_only':
        if set(manifest['expected_counts']) != set(VAL_COUNTS):
            raise ValueError('research_results_validation_cohort_contract')
        if any(manifest['expected_counts'][cohort]['observed'] != count for cohort, count in VAL_COUNTS.items()):
            raise ValueError('research_results_validation_normal_count')
    elif scope == 'frozen_evaluation':
        if not frozen_selection:
            raise ValueError('research_results_frozen_selection_required')
        frozen = read(frozen_selection)
        if (frozen.get('schema') != 'nmr_research_20261022' or not frozen.get('frozen_utc') or
                frozen.get('selection_scope') != 'train_val_only' or
                frozen.get('results_plan_sha256') != result_plan_sha256(manifest)):
            raise ValueError('research_results_invalid_freeze')
        try:
            timestamp = datetime.fromisoformat(frozen['frozen_utc'].replace('Z', '+00:00'))
            if timestamp.tzinfo is None or timestamp > datetime.now(timezone.utc):
                raise ValueError('invalid freeze time')
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError('research_results_invalid_freeze_time') from exc
        if frozen['evaluation_inputs_sha256'].get(str(Path(__file__).resolve())) != sha(__file__):
            raise ValueError('research_results_source_not_frozen')
        if set(manifest.get('expected_query_identity_sha256', {})) != set(manifest['expected_counts']):
            raise ValueError('research_results_frozen_query_identity_required')
        if set(manifest['expected_counts']) not in ({'test'}, {'verify'}):
            raise ValueError('research_results_frozen_cohort_contract')
        if 'test' in manifest['expected_counts'] and manifest['expected_counts']['test'] != dict(observed=1000, frequency_crop=200, peak_dropout=171):
            raise ValueError('research_results_fixed_test_counts')
        if 'verify' in manifest['expected_counts'] and frozen.get('verify_eligibility') != 'eligible':
            raise ValueError('research_results_verify_unavailable')
    else:
        raise ValueError('research_results_unknown_scope')
    for counts in manifest['expected_counts'].values():
        if set(counts) != set(VARIANTS) or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts.values()):
            raise ValueError('research_results_invalid_expected_counts')
    if any(sum(c[v] for c in manifest['expected_counts'].values()) == 0 for v in VARIANTS):
        raise ValueError('research_results_missing_stress_population')


def aggregate_manifest(path, *, scope='train_val_only', frozen_selection=None, repeats=5000, bootstrap_seed=317):
    """完整性先于数值晋升。返回报告；只读，运行中任何输入变化均拒绝完成。"""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 5000 or bootstrap_seed != 317:
        raise ValueError('research_results_fixed_bootstrap_contract')
    manifest_start_sha = sha(path)
    frozen_start_sha = sha(frozen_selection) if frozen_selection else None
    manifest = read(path)
    _scope_guard(manifest, scope, frozen_selection)
    arms_path = Path(manifest['preregistered_arms']).resolve()
    if sha(arms_path) != manifest['preregistered_arms_sha256']:
        raise ValueError('research_results_preregistered_arms_changed')
    registered = read(arms_path)
    frozen = read(frozen_selection) if scope == 'frozen_evaluation' else None
    if frozen and frozen['evaluation_inputs_sha256'].get(str(arms_path)) != manifest['preregistered_arms_sha256']:
        raise ValueError('research_results_preregistered_arms_not_frozen')
    names = [arm['name'] for arm in registered['families']]
    if len(names) != len(set(names)) or len(names) != 11 or set(registered['seeds']) != set(SEEDS):
        raise ValueError('research_results_preregistered_family_contract')
    declared = {family['name']: family for family in manifest['families']}
    if len(declared) != len(manifest['families']) or set(declared) - set(names):
        raise ValueError('research_results_unknown_or_duplicate_family')
    incumbent_name = manifest.get('incumbent', 'incumbent_frozen')
    if incumbent_name != 'incumbent_frozen' or incumbent_name not in names:
        raise ValueError('research_results_incumbent_contract')
    inputs = {str(Path(path).resolve()): manifest_start_sha, str(arms_path): manifest['preregistered_arms_sha256'], str(Path(__file__).resolve()): sha(__file__)}
    if frozen_selection:
        inputs[str(Path(frozen_selection).resolve())] = frozen_start_sha
    families, raw, recall_rows = {}, {}, []
    seen_prediction_paths = set()
    for name in names:
        family = declared.get(name, {})
        entries = family.get('entries', [])
        identities = [(entry['seed'], entry['cohort']) for entry in entries]
        required = {(seed, cohort) for seed in SEEDS for cohort in manifest['expected_counts']}
        if len(set(identities)) != len(identities) or set(identities) - required or any(type(entry['seed']) is not int for entry in entries):
            raise ValueError('research_results_duplicate_seed_or_cohort:' + name)
        statuses, loaded = {}, {}
        for entry in entries:
            files = {str(Path(p).resolve()) for p in entry['predictions']}
            if files & seen_prediction_paths:
                raise ValueError('research_results_seed_cohort_or_family_reuses_predictions:' + name)
            seen_prediction_paths |= files
            status, rows, actual = load_entry(entry, manifest)
            if frozen and rows is not None:
                if entry['adapter'] == 'research' and str(Path(read(entry['protocol'])['args'].get('frozen_selection', '')).resolve()) != str(Path(frozen_selection).resolve()):
                    raise ValueError('research_results_protocol_frozen_selection_mismatch')
                locked_paths = ([read(entry['protocol'])['args'][key] for key in ('no_aug_checkpoint', 'aug_checkpoint')]
                                if entry['adapter'] == 'research' else
                                [p for evidence in entry['seed_evidence'] for p in (evidence['checkpoint'], evidence.get('config')) if p])
                if any(frozen['evaluation_inputs_sha256'].get(str(Path(p).resolve())) != actual[str(Path(p).resolve())] for p in locked_paths):
                    raise ValueError('research_results_weight_or_legacy_config_not_frozen')
                if any(p in frozen['evaluation_inputs_sha256'] and frozen['evaluation_inputs_sha256'][p] != digest for p, digest in actual.items()):
                    raise ValueError('research_results_frozen_input_changed')
            key = str(entry['seed']) + '/' + entry['cohort']
            statuses[key] = status
            if any(input_path in inputs and inputs[input_path] != digest for input_path, digest in actual.items()):
                raise ValueError('research_results_shared_input_changed')
            inputs.update(actual)
            if rows is not None:
                loaded[(entry['seed'], entry['cohort'])] = rows
        if set(loaded) != required:
            families[name] = dict(state='failed' if any(s['state'] == 'failed' for s in statuses.values()) else 'pending', entries=statuses)
            continue
        cost = family.get('inference_cost')
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise ValueError('research_results_missing_fixed_inference_cost:' + name)
        per_seed_rows = {seed: sum((loaded[(seed, cohort)] for cohort in sorted(manifest['expected_counts'])), []) for seed in SEEDS}
        _aligned(per_seed_rows[193], per_seed_rows[211])
        by_seed = {seed: summaries(rows) for seed, rows in per_seed_rows.items()}
        if any(set(value['by_source']) != set(manifest['expected_sources']) for value in by_seed.values()):
            raise ValueError('research_results_five_source_population_missing')
        families[name] = dict(state='complete', entries=statuses, by_seed={str(s): by_seed[s] for s in SEEDS},
            by_cohort={cohort: _mean_summaries({s: summaries(loaded[(s, cohort)]) for s in SEEDS}, manifest['expected_sources'])
                       for cohort in manifest['expected_counts']},
            mean=_mean_summaries(by_seed, manifest['expected_sources']), inference_cost=cost)
        raw[name] = per_seed_rows
        recall_rows.extend(recall_query_row(name, seed, row) for seed in SEEDS for row in per_seed_rows[seed])
    incumbent = families[incumbent_name]
    eligible = []
    if incumbent['state'] == 'complete':
        for name, family in families.items():
            if family['state'] != 'complete':
                continue
            for seed in SEEDS:
                _aligned(raw[name][seed], raw[incumbent_name][seed])
            family['paired_vs_incumbent'] = {variant: paired_cluster_bootstrap(
                {s: [r for r in raw[name][s] if r['variant'] == variant] for s in SEEDS},
                {s: [r for r in raw[incumbent_name][s] if r['variant'] == variant] for s in SEEDS},
                repeats=repeats, seed=bootstrap_seed) for variant in VARIANTS}
            if scope == 'train_val_only' and name != incumbent_name:
                family['promotion_guards'] = promotion_guards(family['mean'], incumbent['mean'])
                family['eligible_for_promotion'] = all(family['promotion_guards'].values())
                if family['eligible_for_promotion']:
                    eligible.append(name)
    selected = (min(eligible, key=lambda name: (-families[name]['mean']['overall']['MRR'],
                families[name]['inference_cost'], names.index(name))) if eligible else
                incumbent_name if incumbent['state'] == 'complete' else None) if scope == 'train_val_only' else None
    for input_path, digest in inputs.items():
        if sha(input_path) != digest:
            raise ValueError('research_results_input_changed_during_read:' + input_path)
    all_complete = all(v['state'] == 'complete' for v in families.values())
    return dict(schema=SCHEMA, scope=scope, result_plan_sha256=result_plan_sha256(manifest),
        families=families, incumbent=incumbent_name, selected_family=selected,
        selection_disabled=scope != 'train_val_only', selection_provisional=not all_complete,
        selection_rule='strict normal MRR improvement and four nondegradation guards; highest eligible MRR then lower fixed inference cost',
        complete_families=sum(v['state'] == 'complete' for v in families.values()),
        all_families_complete=all_complete, recall_query_rows=recall_rows,
        pending_families=[n for n, v in families.items() if v['state'] == 'pending'],
        failed_families=[n for n, v in families.items() if v['state'] == 'failed'],
        inputs_sha256=inputs, external_api_calls=0, bootstrap_repetitions=repeats, bootstrap_seed=bootstrap_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scope', choices=('train_val_only', 'frozen_evaluation'), default='train_val_only')
    parser.add_argument('--frozen-selection')
    parser.add_argument('--repeats', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=317)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    try:
        result = aggregate_manifest(args.manifest, scope=args.scope, frozen_selection=args.frozen_selection,
                                    repeats=args.repeats, bootstrap_seed=args.bootstrap_seed)
        recall_rows = result.pop('recall_query_rows')
        recall_path = out / 'recall_queries.jsonl'
        recall_path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in recall_rows))
        result['recall_table'] = dict(path=str(recall_path.resolve()), sha256=sha(recall_path), rows=len(recall_rows),
            membership_evidence='explicit_bool_target_in_coarse_pool_and_explicit_evaluation_pool_size_512',
            incomplete_family_rows='excluded')
        temporary = out / 'summary.tmp'
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(out / 'summary.json')
        (out / 'EXIT_CODE').write_text('0\n')
    except Exception:
        (out / 'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

"""功能概述：在冻结候选预测上独立诊断实验正观测的前向位移残差。

本模块只选真结构的已生成分位数，不执行 GNN 或候选排序。原 bank 分位数为
ppm/10，candidate_quantile_sigma 已是 ppm。prepare_observation 保留谱线、人工
标签、积分和未知覆盖的不同口径。正向最近残差不是原子 MAE；局部类别来自
最近预测分位成分，不是真实 assignment。反向最近距离和 W1 仅描述可见支持
分布，未知覆盖时不将它们解释为缺峰惩罚。宽度覆盖也是最近成分区间覆盖，
不是混合预测分布的校准证明。未知条件不从来源名称、结构或位移推断。

CLI 默认 original_val，可运行 tune；verify/test 必须绑定本轮新冻结配置与
全部实际输入 SHA。test 另核固定 1000+200+171 身份与缺测 mask。输出独占
目录、逐查询 JSONL、分层统计、始末输入 SHA 与 EXIT_CODE；不改旧数据，
不训练、不重排、不联网。main 由外层进程收集真实操作系统退出码。
"""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from .evaluate import missing_queries
from .h_condinfer import condition_key, scoring_inputs
from .h_conditions import parse_ph
from .h_global_bank import load_bank
from .h_global_field_capacity import frequency_mhz
from .h_global_protocol import scoring_inputs_paths
from .h_labels import make_key
from .h_metadata import MISSING, metadata_view
from .h_observation_study import stamp_queries
from .h_research_contract import atomic_json, input_fingerprints
from .h_research_observation import prepare_observation
from .h_score import attach_observation
from .h_uncertainty import CLASSES


CANDIDATE_COUNT = 114982
QUERY_COUNTS = {'original_val': 320, 'tune': 1024, 'verify': 1024, 'test': 1000}
METRIC_VERSION = 'positive_support_nearest_ppm_v1'
CRITICAL_SOURCES = (
    'h_research_forward_diagnostics.py', 'h_research_contract.py', 'h_research_observation.py',
    'h_research_recall.py', 'h_global_bank.py', 'h_global_protocol.py', 'h_condinfer.py',
    'h_conditions.py', 'h_metadata.py', 'h_score.py', 'h_labels.py', 'h_observed.py',
    'h_global_field_capacity.py', 'h_observation_study.py', 'h_uncertainty.py', 'evaluate.py', 'data.py', 'prepare.py',
)


def content_sha(value):
    """与冻结查询侧表相同的数组 JSON 内容散列；不修改输入。"""
    def convert(item):
        if hasattr(item, 'tolist'):
            return convert(item.tolist())
        if isinstance(item, dict):
            return {key: convert(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(val) for val in item]
        return item
    return hashlib.sha256(json.dumps(convert(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def guard_locked(selection_path, inputs, config, cohort):
    """先于 bank/查询解析核封存；旧 extract 配置不能隐式授权新诊断。"""
    if cohort not in ('verify', 'test'):
        return
    if not selection_path:
        raise ValueError('forward_diagnostics_new_frozen_selection_required')
    frozen = json.loads(Path(selection_path).read_text())
    if (frozen.get('schema') != 'nmr_research_20261022'
            or frozen.get('selection_scope') != 'train_val_only' or not frozen.get('frozen_utc')):
        raise ValueError('forward_diagnostics_invalid_freeze')
    if config not in frozen.get('allowed_forward_diagnostics_configs', []):
        raise ValueError('forward_diagnostics_config_not_frozen')
    for path, digest in inputs.items():
        if frozen.get('evaluation_inputs_sha256', {}).get(path) != digest:
            raise ValueError('forward_diagnostics_frozen_input_changed:' + path)
    if cohort == 'verify' and frozen.get('verify_eligibility') != 'eligible':
        raise ValueError('forward_diagnostics_independent_verify_not_eligible')


def _present(value):
    return value is not None and str(value).strip().lower() not in MISSING


def condition_evidence(row, meta, raw_record=None):
    """只解析实际字段，取首个可解析值；无有效值时保留首个声明的原值。

    frequency 顺序为 raw 顶层/experimental_conditions/conditions，再按相同顺序
    查 query、metadata，最后查 metadata.solvent_recovery.experimental_conditions。
    metadata 顶层优先 raw_frequency 再 frequency。pH 保留既有 metadata.raw_pH
    优先级，随后采用上述来源顺序，各来源先 pH 后 ph。原 pH 可用不代表模型
    使用了它；场强必须带 MHz 单位，不以数值大小或来源推断。
    """
    meta = meta or {}
    raw_record = raw_record or {}
    solvent = meta.get('solvent')
    solvent_field = 'metadata.solvent' if _present(solvent) else None
    condition_sources = [
        ('raw.experimental_conditions', raw_record.get('experimental_conditions') or {}),
        ('raw.conditions', raw_record.get('conditions') or {}),
        ('query.experimental_conditions', row.get('experimental_conditions') or {}),
        ('query.conditions', row.get('conditions') or {}),
        ('metadata.experimental_conditions', meta.get('experimental_conditions') or {}),
        ('metadata.conditions', meta.get('conditions') or {}),
        ('metadata.solvent_recovery.experimental_conditions',
         (meta.get('solvent_recovery') or {}).get('experimental_conditions') or {}),
    ]
    actual_sources = [('raw', raw_record), *condition_sources[:2],
                      ('query', row), *condition_sources[2:4],
                      ('metadata', meta), *condition_sources[4:]]
    if solvent_field is None:
        for prefix, value in [('raw.solvent', raw_record.get('solvent')),
                              ('query.solvent', row.get('solvent'))] + [
                                  (prefix + '.solvent', values.get('solvent')) for prefix, values in condition_sources]:
            if _present(value):
                solvent, solvent_field = value, prefix
                break
    solvent_reported = solvent_field is not None
    def first_parsed(candidates, parser):
        first_field, first_value = None, None
        for field, value in candidates:
            if not _present(value):
                continue
            if first_field is None:
                first_field, first_value = field, value
            parsed = parser(value)
            if parsed is not None:
                return parsed, field, value
        return None, first_field, first_value
    ph_candidates = [('metadata.raw_pH', meta.get('raw_pH'))] + [
        (prefix + '.' + key, values.get(key)) for prefix, values in actual_sources for key in ('pH', 'ph')]
    # 既有 parse_ph 的 value-or-empty 会吞掉数值 0；先转字符串保留实际 pH=0。
    ph, ph_field, ph_value = first_parsed(ph_candidates, lambda value: parse_ph(str(value)))
    frequency_candidates = [(prefix + '.' + key, values.get(key)) for prefix, values in actual_sources
                            for key in (('raw_frequency', 'frequency') if prefix == 'metadata' else ('frequency',))]
    frequency, frequency_field, frequency_raw = first_parsed(
        frequency_candidates, lambda value: frequency_mhz({'experimental_conditions': {'frequency': value}}))
    return dict(solvent=str(solvent) if solvent_reported else None,
                solvent_field=solvent_field,
                solvent_status='reported' if solvent_reported else 'unknown',
                solvent_family=(meta.get('family', 'reported_unclassified') if solvent_field == 'metadata.solvent'
                                else 'reported_unclassified') if solvent_reported else 'unknown',
                solvent_recognized=bool(solvent_reported and meta.get('family') not in (None, 'unknown', 'other')),
                pH_raw=None if ph_field is None else ph_value, pH=ph, pH_known=ph is not None,
                pH_field=ph_field, pH_used_by_model=bool(meta.get('pH_used', False)),
                frequency_raw=frequency_raw, frequency_MHz=frequency,
                frequency_known=frequency is not None, frequency_field=frequency_field)


def weighted_stats(values, weights=None):
    """非负残差的经验分布；分位数使用加权 ECDF 左逆，不对相邻样本插值。"""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if (values.shape != weights.shape or not np.isfinite(values).all() or (values < 0).any()
            or not np.isfinite(weights).all() or (weights < 0).any()):
        raise ValueError('forward_diagnostics_invalid_residuals')
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    if not len(values):
        return dict(positions=0, mean_ppm=None, median_ppm=None, P90_ppm=None, P95_ppm=None,
                    fraction_gt_0_5ppm=None, fraction_gt_1ppm=None)
    order = np.argsort(values, kind='stable')
    cdf = np.cumsum(weights[order]) / weights.sum()
    quantile = lambda q: float(values[order[min(int(np.searchsorted(cdf, q, side='left')), len(values)-1)]])
    return dict(positions=len(values), mean_ppm=float(np.average(values, weights=weights)),
                median_ppm=quantile(.5), P90_ppm=quantile(.9), P95_ppm=quantile(.95),
                fraction_gt_0_5ppm=float(np.average(values > .5, weights=weights)),
                fraction_gt_1ppm=float(np.average(values > 1., weights=weights)))


def support_w1(xs, weights, predictions):
    """观测归一化报告质量与等分位预测质量之间的精确一维 W1，单位 ppm。"""
    xs, weights, predictions = map(lambda x: np.asarray(x, dtype=float), (xs, weights, predictions))
    if not len(predictions):
        return None
    weights = weights / weights.sum()
    points = np.unique(np.concatenate((xs, predictions)))
    if len(points) < 2:
        return 0.
    observed = (weights[:, None] * (xs[:, None] <= points[:-1])).sum(0)
    predicted = (predictions[:, None] <= points[:-1]).mean(0)
    return float((np.abs(observed-predicted) * np.diff(points)).sum())


def residual_diagnostics(observation, predicted_quantiles, tags=None, sigma_ppm=None):
    """正观测→所有预测中心最近距离；显式窗只限制反向/W1的预测可见中心。"""
    predicted = np.asarray(predicted_quantiles, dtype=float).reshape(-1) * 10.
    xs = np.asarray(observation['xs'], dtype=float)
    weights = np.asarray(observation['weights'], dtype=float)
    if (not len(predicted) or not np.isfinite(predicted).all() or not len(xs)
            or weights.shape != xs.shape or not np.isfinite(xs).all()
            or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('forward_diagnostics_invalid_support')
    weights = weights / weights.sum()
    tags = None if tags is None else np.asarray(tags)
    if tags is not None and (tags.shape != predicted.shape or not np.issubdtype(tags.dtype, np.integer)
                             or (tags < 0).any() or (tags >= len(CLASSES)).any()):
        raise ValueError('forward_diagnostics_invalid_class_labels')
    sigma = None if sigma_ppm is None else np.asarray(sigma_ppm, dtype=float)
    if sigma is not None and (sigma.shape != predicted.shape or not np.isfinite(sigma).all() or (sigma <= 0).any()):
        raise ValueError('forward_diagnostics_invalid_ppm_sigma')
    distance = np.abs(xs[:, None] - predicted[None, :])
    nearest = distance.argmin(1)
    residual = distance[np.arange(len(xs)), nearest]
    matched_classes, exchangeable, matched_sigma, tie_counts = [], [], [], []
    for i, error in enumerate(residual):
        ties = np.flatnonzero(np.isclose(distance[i], error, atol=1e-8, rtol=0))
        tie_counts.append(len(ties))
        classes = set(tags[ties].tolist()) if tags is not None else set()
        matched_classes.append(CLASSES[next(iter(classes))] if len(classes) == 1
                               else 'nearest_class_ambiguous' if classes else 'unavailable')
        states = {'exchangeable' if int(c) in (5, 6, 7)
                  else 'nonexchangeable' if int(c) < 5 else 'unclassified' for c in classes}
        exchangeable.append(next(iter(states)) if len(states) == 1 else 'ambiguous' if states else 'unavailable')
        matched_sigma.append(float(sigma[ties[0]]) if sigma is not None
                             and np.allclose(sigma[ties], sigma[ties[0]], atol=1e-8, rtol=0) else None)
    visible = np.ones(len(predicted), dtype=bool)
    if observation['windows']:
        visible[:] = False
        for low, high in observation['windows']:
            visible |= (predicted >= low) & (predicted <= high)
    visible_predicted = predicted[visible]
    reverse = (np.abs(visible_predicted[:, None]-xs[None, :]).min(1)
               if len(visible_predicted) else np.array([]))
    return dict(observed_ppm=xs.tolist(), normalized_report_weights=weights.tolist(),
                nearest_prediction_ppm=predicted[nearest].tolist(),
                signed_residual_ppm=(xs-predicted[nearest]).tolist(), absolute_residual_ppm=residual.tolist(),
                model_matched_local_class=matched_classes, model_matched_exchangeability=exchangeable,
                nearest_tie_quantiles=tie_counts, nearest_sigma_ppm=matched_sigma,
                per_query_weighted=weighted_stats(residual, weights),
                per_query_position_equal=weighted_stats(residual),
                predicted_quantiles=len(predicted), visible_predicted_quantiles=int(visible.sum()),
                descriptive_support_distances=dict(
                    reverse_nearest_mean_ppm=float(reverse.mean()) if len(reverse) else None,
                    W1_ppm=support_w1(xs, weights, visible_predicted),
                    prediction_scope='centers inside explicit windows' if observation['windows'] else 'all centers; coverage unknown or partial',
                    mass_scope='normalized reported observation weights vs equal quantile weights; mass semantics may differ',
                    atom_assignment_metric=False, missing_peak_penalty=False),
                forward_prediction_scope='all quantile centers; positive residual may cross a window boundary',
                atom_assignment_metric=False)


def _aggregate(rows, selector=None):
    errors, weights, sigmas, width_weights, standardized, means, reverse, w1 = [], [], [], [], [], [], [], []
    contributing, width_queries = 0, 0
    for row in rows:
        r = row['residuals']
        keep = np.ones(len(r['absolute_residual_ppm']), dtype=bool) if selector is None else selector(row)
        if not np.any(keep):
            continue
        contributing += 1
        e = np.asarray(r['absolute_residual_ppm'])[keep]
        w = np.asarray(r['normalized_report_weights'])[keep]
        w = w / w.sum()
        errors.extend(e.tolist()); weights.extend(w.tolist()); means.append(float(np.sum(e*w)))
        selected_sigma = [s for s, yes in zip(r['nearest_sigma_ppm'], keep) if yes]
        valid = np.array([s is not None for s in selected_sigma])
        if valid.any():
            width_queries += 1
            sw = w[valid] / w[valid].sum()
            s = np.asarray([v for v in selected_sigma if v is not None])
            sigmas.extend(s.tolist()); standardized.extend((e[valid]/s).tolist()); width_weights.extend(sw.tolist())
        if selector is None:
            support = r['descriptive_support_distances']
            if support['reverse_nearest_mean_ppm'] is not None:
                reverse.append(support['reverse_nearest_mean_ppm'])
            if support['W1_ppm'] is not None:
                w1.append(support['W1_ppm'])
    width = dict(queries=width_queries, positions=len(sigmas),
                 interval_scope='nearest matched quantile component; tied unequal sigmas excluded; not mixture calibration',
                 mean_sigma_ppm=float(np.average(sigmas, weights=width_weights)) if sigmas else None,
                 mean_absolute_standardized_residual=float(np.average(standardized, weights=width_weights)) if sigmas else None)
    for z in (1., 1.96, 2.):
        width[f'fraction_within_{z:g}sigma'] = (float(np.average(np.asarray(standardized) <= z, weights=width_weights))
                                              if sigmas else None)
    return dict(queries=contributing, unique_observed_positions=len(errors),
                query_equal_weighted_positions=weighted_stats(errors, weights),
                position_micro_equal=weighted_stats(errors),
                query_mean_absolute_residual_ppm=float(np.mean(means)) if means else None,
                width_coverage=width,
                descriptive_query_mean_reverse_nearest_ppm=float(np.mean(reverse)) if reverse else None,
                descriptive_query_mean_W1_ppm=float(np.mean(w1)) if w1 else None)


def summarize(rows):
    """每查询先归一化报告质量再合并；不混合不同谱表的原始积分单位。"""
    normal = [row for row in rows if row['variant'] == 'observed']
    strata = {}
    fields = ('source', 'semantics', 'coverage', 'solvent_status', 'solvent_family',
              'pH_known', 'pH', 'pH_used_by_model', 'frequency_known', 'frequency_MHz', 'variant')
    for field in fields:
        part = rows if field == 'variant' else normal
        values = sorted({str(row[field]) for row in part})
        strata[field] = {value: _aggregate([r for r in part if str(r[field]) == value]) for value in values}
    for field in ('model_matched_local_class', 'model_matched_exchangeability'):
        classes = sorted({v for row in normal for v in row['residuals'][field]})
        strata[field] = {value: _aggregate(normal, lambda r, v=value: np.asarray(r['residuals'][field]) == v)
                         for value in classes}
    return dict(overall=_aggregate(normal), strata=strata,
                stress={variant: _aggregate([r for r in rows if r['variant'] == variant])
                        for variant in sorted({r['variant'] for r in rows} - {'observed'})},
                normal_records=len(normal), total_records=len(rows),
                metric_definition=dict(
                    residual='observed ppm minus nearest predicted quantile center; positive observations only',
                    quantiles='weighted empirical CDF left inverse',
                    query_equal='each contributing query has total weight one; within-query weights preserve stated observation semantics',
                    position_micro='each unique prepared reported position has weight one',
                    local_class='model-matched nearest prediction class, not ground-truth atom assignment',
                    raw_integral_mass_pooled_across_queries=False, atom_MAE_reported=False,
                    source_is_stratum_only=True, overall_and_nonvariant_strata='observed only',
                    width_is_component_interval_diagnostic=True))


def required_inputs(args):
    paths = [args.bank, args.extension, Path(args.cohorts)/'cohorts.pt', Path(args.cohorts)/'manifest.json',
             Path(args.cohorts)/'cohort_contract.json',
             args.fixed_identities] + scoring_inputs_paths(args.base)
    paths += [Path(__file__).with_name(name) for name in CRITICAL_SOURCES]
    return paths


def diagnose(bank, payload, raw, args, fixed):
    indices = [int(i) for i in bank['single_indices']]
    smiles = [bank['smiles'][i] for i in indices]
    candidate_sha = hashlib.sha256('\n'.join(smiles).encode()).hexdigest()
    if (len(smiles) != CANDIDATE_COUNT or len(set(smiles)) != len(smiles)
            or candidate_sha != fixed['candidate_order_sha256']):
        raise ValueError('forward_diagnostics_fixed_candidate_library_changed')
    queries = bank['test'] if args.cohort == 'test' else payload['cohorts'][args.cohort]
    if len(queries) != QUERY_COUNTS[args.cohort]:
        raise ValueError('forward_diagnostics_fixed_query_count_changed')
    keys = [make_key(row) for row in queries]
    if len(set(keys)) != len(keys):
        raise ValueError('forward_diagnostics_duplicate_query_identity')
    if args.cohort != 'test' and stamp_queries(queries) != fixed['cohort_query_sha256'][args.cohort]:
        raise ValueError('forward_diagnostics_fixed_validation_query_changed')
    if args.cohort == 'test' and (not args.augment or stamp_queries(queries) != fixed['frozen_test_query_sha256']):
        raise ValueError('forward_diagnostics_fixed_test_query_changed')
    metadata = metadata_view(payload['metadata'], payload['key_mode'])
    lookup = {s: i for s, i in zip(smiles, indices)}
    rows = []
    for order, row in enumerate(queries):
        key = condition_key(row, payload['metadata'], payload['key_mode'])
        if key not in bank['predictions'] or row['smiles'] not in lookup:
            raise ValueError('forward_diagnostics_truth_prediction_unavailable')
        target = lookup[row['smiles']]
        # target 只用于诊断读取真结构预测；本模块不产生部署分数或候选排名。
        prediction = bank['predictions'][key][target].cpu().numpy()
        if prediction.shape != (128,):
            raise ValueError('forward_diagnostics_expected_128_quantiles')
        classes = bank.get('candidate_quantile_classes', {}).get(key)
        classes = None if classes is None else classes[target].cpu().numpy()
        sigma = bank.get('candidate_quantile_sigma', {}).get(key)
        sigma = None if sigma is None else sigma[target].cpu().numpy()
        evidence = condition_evidence(row, metadata.get(make_key(row)), raw.get(row['record_id']))
        query = attach_observation(row, payload['annotations'].get(make_key(row), {}))
        variants = [('observed', query)]
        if args.augment and (args.cohort != 'test' or order < 200):
            variants += missing_queries(query, '1H')
        for variant, observed in variants:
            obs = prepare_observation(observed, raw)
            r = residual_diagnostics(obs, prediction, classes, sigma)
            frequency = evidence['frequency_MHz']
            r['absolute_residual_Hz'] = ([v*frequency for v in r['absolute_residual_ppm']]
                                         if frequency is not None else None)
            rows.append(dict(query_order=order, variant=variant, source=row['source'],
                record_id=row['record_id'], smiles=row['smiles'], group=row['group'],
                semantics=obs['semantics'], coverage=obs['coverage'], partial_peaks=obs['partial_peaks'],
                windows_ppm=obs['windows'], reported_count=obs['reported_count'],
                raw_report_weight_sum=float(sum(obs['raw_weights'])),
                **evidence, residuals=r,
                mask_content_sha256=content_sha({name: observed.get(name) for name in
                    ('q', 'observation', 'measure', 'observed_xs', 'observed_weights')})))
    rows.sort(key=lambda r: (r['query_order'], r['variant']))
    identities = [{k: row[k] for k in ('query_order', 'variant', 'source', 'record_id', 'smiles', 'group')} for row in rows]
    masks = [(tuple(row[k] for k in ('query_order', 'variant', 'source', 'record_id', 'smiles')),
              row['mask_content_sha256']) for row in rows]
    counts = dict(collections.Counter(r['variant'] for r in rows))
    if args.cohort == 'test' and (counts != fixed['variants']
            or content_sha(identities) != fixed['identity_content_sha256']
            or content_sha(masks) != fixed['variant_mask_content_sha256']):
        raise ValueError('forward_diagnostics_fixed_test_identity_or_mask_changed')
    return rows, dict(candidate_order_sha256=candidate_sha, query_sha256=stamp_queries(queries),
                      identity_content_sha256=content_sha(identities), variant_mask_content_sha256=content_sha(masks),
                      variants=counts, base_queries=len(queries))


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('base', 'bank', 'cohorts', 'fixed-identities', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--extension')
    parser.add_argument('--cohort', choices=tuple(QUERY_COUNTS), default='original_val')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--frozen-selection')
    args = parser.parse_args(argv)
    config = dict(metric_version=METRIC_VERSION, bank=str(Path(args.bank).resolve()),
                  extension=None if args.extension is None else str(Path(args.extension).resolve()),
                  cohort=args.cohort, augment=args.augment, line_resolution_ppm=.08,
                  prediction_units='ppm/10', candidate_quantile_sigma_units='ppm')
    pinned = input_fingerprints(required_inputs(args))
    selection_sha = input_fingerprints([args.frozen_selection]) if args.frozen_selection else {}
    guard_locked(args.frozen_selection, pinned, config, args.cohort)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    status = lambda stage, **kw: atomic_json(out/'status.json', dict(stage=stage, **kw))
    started, tick = datetime.datetime.now(datetime.timezone.utc).isoformat(), time.monotonic()
    try:
        torch.set_num_threads(4)
        status('loading')
        bank = load_bank(args.bank, args.extension)
        checkpoint = Path(bank['checkpoint_path'])
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parent.parent / checkpoint
        extra = input_fingerprints([checkpoint])
        if extra[str(checkpoint.resolve())] != bank['checkpoint_sha256']:
            raise ValueError('forward_diagnostics_checkpoint_changed')
        pinned.update(extra)
        guard_locked(args.frozen_selection, pinned, config, args.cohort)
        atomic_json(out/'protocol.json', dict(args=vars(args), config=config, inputs_sha256=pinned,
                                             frozen_selection_sha256=selection_sha,
                                             started_utc=started, hostname=socket.gethostname()))
        payload = torch.load(Path(args.cohorts)/'cohorts.pt', map_location='cpu', weights_only=False)
        raw = scoring_inputs(Path(args.base))[-1]
        fixed = json.loads(Path(args.fixed_identities).read_text())
        contract = json.loads((Path(args.cohorts)/'cohort_contract.json').read_text())
        if (contract['cohorts_sha256'] != pinned[str((Path(args.cohorts)/'cohorts.pt').resolve())]
                or contract['candidate_order_sha256'] != fixed['candidate_order_sha256']):
            raise ValueError('forward_diagnostics_cohort_contract_changed')
        fixed = dict(fixed, cohort_query_sha256=contract['cohort_query_sha256'])
        rows, identity = diagnose(bank, payload, raw, args, fixed)
        with (out/'per_query.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        summary = dict(schema='research_forward_diagnostics_v1', cohort=args.cohort,
            checkpoint_sha256=bank['checkpoint_sha256'], **identity, **summarize(rows),
            inputs_start_sha256=pinned, metric_version=METRIC_VERSION,
            frozen_selection_sha256=selection_sha,
            started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hostname=socket.gethostname(), elapsed_seconds=time.monotonic()-tick,
            model_forward_executed=False, retrieval_ranking_executed=False, external_api_calls=0)
        if input_fingerprints(pinned) != pinned:
            raise ValueError('forward_diagnostics_input_changed_during_run')
        if input_fingerprints(selection_sha) != selection_sha:
            raise ValueError('forward_diagnostics_frozen_selection_changed_during_run')
        summary['inputs_end_sha256'] = dict(pinned)
        summary['per_query_sha256'] = input_fingerprints([out/'per_query.jsonl'])[str((out/'per_query.jsonl').resolve())]
        atomic_json(out/'summary.json', summary)
        status('complete', records=len(rows)); (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed', error=repr(exc)); (out/'EXIT_CODE').write_text('1\n'); raise


if __name__ == '__main__':
    main()

"""功能概述：复用同一 no_aug/aug pair 的 blend 结果，评估公开观测标签路由。

explicit_missing 仅读取 query_observation 的 windows/partial_peaks，缺字段为假；
route_predictions 对未声明缺测查询选择 gamma=0，对显式缺测选择预声明 gamma。
source、ID、variant、结构与正确候选只核验身份和计算分层指标，不选择分支。
validate_origin/validate_predictions 核验缓存、权重、源码、完整库排名和预测身份；
沿用 h_global_ranker_blend.validate_pair，拒绝验证补正例与不同 coarse/candidate。
缓存必须已保存 query_observation；本轮 train/val 缓存均具备，缺失时明确失败，
不会从 variant 推断观测状态，也不会重新附加原始谱表而恢复删峰。

main 的 train_val 仅 original_val/tune，输出五个共同 gamma 的原谱与压力预测。
locked 仅 verify/test；metadata_route_configs 精确冻结有序 pair、policy 与 gamma，
且既有 blend 必须由同一 frozen selection 产生，并包含 gamma=0 和所选 gamma。
冻结格式为 [{no_aug_checkpoint:绝对路径,aug_checkpoint:绝对路径,
policy:'explicit_observation',gamma:数值}]。源码、缓存来源与权重沿用现有 SHA 门禁，
所有实际读取文件运行始末再核验 SHA。输出目录新建，成功/失败写 EXIT_CODE。
本模块不运行模型前向、训练、GPU、网络或外部 API，不修改既有输入与结果。
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from .h_global_ranker import metrics
from .h_global_ranker_blend import GAMMAS, locked_inputs, validate_pair
from .prepare import sha


POLICY = 'explicit_observation'


def explicit_missing(observation):
    """严格只读两个公开观测字段；无字段默认为未声明缺测。"""
    windows = observation.get('windows')
    partial = observation.get('partial_peaks', False)
    if windows is not None:
        if not isinstance(windows, (list, tuple)):
            raise ValueError('metadata_route_invalid_windows')
        for window in windows:
            if (not isinstance(window, (list, tuple)) or len(window) != 2
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) for x in window)
                    or window[0] >= window[1]):
                raise ValueError('metadata_route_invalid_windows')
    if not isinstance(partial, bool):
        raise ValueError('metadata_route_invalid_partial_peaks')
    return bool(windows) or partial


def row_key(row):
    """查询次序和 variant 仅构成身份键，不参与路由判定。"""
    return row['query_order'], row['variant']


def frozen_routes(frozen, no_aug_path, aug_path):
    """只返回冻结列表精确匹配有序 pair/policy 的共同 gamma。"""
    pair = str(Path(no_aug_path).resolve()), str(Path(aug_path).resolve())
    selected, seen = [], set()
    for config in frozen.get('metadata_route_configs', []):
        if set(config) != {'no_aug_checkpoint', 'aug_checkpoint', 'policy', 'gamma'}:
            raise ValueError('metadata_route_invalid_frozen_config')
        gamma = config['gamma']
        if (config['policy'] != POLICY or isinstance(gamma, bool)
                or not isinstance(gamma, (int, float)) or gamma not in GAMMAS):
            raise ValueError('metadata_route_unregistered_policy_or_gamma')
        key = (str(Path(config['no_aug_checkpoint']).resolve()),
               str(Path(config['aug_checkpoint']).resolve()), config['policy'], float(gamma))
        if key in seen:
            raise ValueError('metadata_route_duplicate_frozen_config')
        seen.add(key)
        if key[:2] == pair:
            selected.append(float(gamma))
    if not selected:
        raise ValueError('metadata_route_pair_not_frozen')
    return selected


def validate_origin(summary, manifest, no_aug_path, aug_path, cache_sha, stage,
                    selection_sha=None):
    """既有 blend 必须与当前缓存、权重、算法和封存阶段完全同源。"""
    expected = dict(cohort=manifest['cohort'], cache_sha256=cache_sha,
        no_aug_checkpoint=str(Path(no_aug_path).resolve()),
        aug_checkpoint=str(Path(aug_path).resolve()),
        no_aug_sha256=sha(no_aug_path), aug_sha256=sha(aug_path),
        coarse_config=manifest['coarse_config'],
        candidate_order_sha256=manifest['candidate_order_sha256'],
        queries=manifest['queries'], source_sha256=sha(Path(__file__).with_name('h_global_ranker_blend.py')))
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError('metadata_route_blend_origin_mismatch:' + field)
    if (summary.get('device') != 'cpu' or summary.get('external_api_calls') != 0
            or not isinstance(summary.get('gammas'), list)
            or len(summary['gammas']) != len(set(summary['gammas']))):
        raise ValueError('metadata_route_invalid_blend_summary')
    scope = 'train_val_only' if stage == 'train_val' else 'frozen_blend_configs'
    if summary.get('selection_scope') != scope:
        raise ValueError('metadata_route_blend_scope_mismatch')
    if stage == 'locked' and summary.get('frozen_selection_sha256') != selection_sha:
        raise ValueError('metadata_route_blend_selection_mismatch')


def validate_predictions(rows, data):
    """拒绝重复/缺失查询、身份漂移、候选注入和名单外排名漂移。"""
    cached = {row_key(row): row for row in data['rows']}
    indexed = {}
    count = len(data['smiles'])
    for row in rows:
        key = row_key(row)
        if key in indexed or key not in cached:
            raise ValueError('metadata_route_duplicate_or_unknown_prediction')
        original = cached[key]
        for field in ('source', 'record_id', 'smiles', 'group', 'baseline_rank', 'coarse_rank'):
            if row[field] != original[field]:
                raise ValueError('metadata_route_prediction_identity_mismatch:' + field)
        rank = row['rank']
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= count:
            raise ValueError('metadata_route_invalid_full_library_rank')
        if not original['target_in_coarse_pool'] and rank != original['coarse_rank']:
            raise ValueError('metadata_route_outside_pool_rank_changed')
        if not original.get('rerank_eligible', True) and rank != original['coarse_rank']:
            raise ValueError('metadata_route_ineligible_rank_changed')
        allowed = set(original['candidate_indices'].tolist()) | set(original['coarse_top10'])
        allowed_smiles = {data['smiles'][i] for i in allowed}
        top = row['top10']
        if (not isinstance(top, list) or len(top) != min(10, count)
                or len(set(top)) != len(top) or any(s not in allowed_smiles for s in top)):
            raise ValueError('metadata_route_invalid_top10')
        if (not original.get('rerank_eligible', True)
                and top != [data['smiles'][i] for i in original['coarse_top10']]):
            raise ValueError('metadata_route_ineligible_top10_changed')
        indexed[key] = row
    if indexed.keys() != cached.keys():
        raise ValueError('metadata_route_missing_prediction')
    return indexed


def route_predictions(data, predictions, gammas):
    """对齐已有固定 gamma 预测；仅由公开观测字段选择 0 或共同 gamma。"""
    gammas = list(dict.fromkeys(float(gamma) for gamma in gammas))
    if not gammas or any(gamma not in GAMMAS for gamma in gammas):
        raise ValueError('metadata_route_unregistered_gamma')
    needed = set(gammas) | {0.}
    branches = {str(gamma): validate_predictions(predictions[str(gamma)], data) for gamma in needed}
    flags = {}
    for row in data['rows']:
        if not isinstance(row.get('query_observation'), dict):
            raise ValueError('metadata_route_cached_observation_missing')
        flags[row_key(row)] = explicit_missing(row['query_observation'])
    outputs = {str(gamma): [] for gamma in gammas}
    for gamma in gammas:
        for row in data['rows']:
            key = row_key(row)
            selected = gamma if flags[key] else 0.
            result = dict(branches[str(selected)][key])
            result['route_gamma'] = selected
            result['explicit_missing'] = flags[key]
            outputs[str(gamma)].append(result)
    counts = Counter('explicit_missing' if value else 'undeclared_missing' for value in flags.values())
    return outputs, {gamma: metrics(rows) for gamma, rows in outputs.items()}, dict(counts)


def snapshot_inputs(paths, expected=None):
    """记录所有实际文件 SHA，并核验缓存来源中已有的冻结 SHA。"""
    checks = {str(Path(path).resolve()): sha(path) for path in paths}
    for path, digest in (expected or {}).items():
        canonical = str(Path(path).resolve())
        actual = sha(path)
        if actual != digest:
            raise ValueError('metadata_route_source_input_changed:' + path)
        checks[canonical] = actual
    return checks


def verify_unchanged(checks):
    """完成后重新散列全部输入；变化时保留失败现场，不能宣告完成。"""
    for path, digest in checks.items():
        if sha(path) != digest:
            raise ValueError('metadata_route_input_changed_during_run:' + path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['train_val', 'locked'], required=True)
    for name in ('no-aug-checkpoint', 'aug-checkpoint', 'cache', 'blend-dir', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--frozen-selection')
    args = parser.parse_args()
    out, blend = Path(args.out), Path(args.blend_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(4)
        sources = [Path(__file__), Path(__file__).with_name('h_global_ranker_blend.py'),
                   Path(__file__).with_name('h_global_ranker.py'),
                   Path(__file__).with_name('h_global_features.py'),
                   Path(__file__).with_name('prepare.py')]
        paths = sources + [Path(args.cache), Path(args.no_aug_checkpoint), Path(args.aug_checkpoint),
                           blend / 'summary.json', blend / 'EXIT_CODE']
        initial = snapshot_inputs(paths)
        if (blend / 'EXIT_CODE').read_text().strip() != '0':
            raise ValueError('metadata_route_blend_not_complete')
        no_aug = torch.load(args.no_aug_checkpoint, map_location='cpu', weights_only=False)
        aug = torch.load(args.aug_checkpoint, map_location='cpu', weights_only=False)
        data = torch.load(args.cache, map_location='cpu', weights_only=False, mmap=True)
        manifest = data['manifest']
        validate_pair(no_aug, aug, data)
        selection_sha = None
        if args.stage == 'train_val':
            if manifest['cohort'] not in ('original_val', 'tune'):
                raise ValueError('metadata_route_train_val_access_violation')
            gammas = list(GAMMAS)
        else:
            if manifest['cohort'] not in ('verify', 'test') or not args.frozen_selection:
                raise ValueError('metadata_route_locked_access_violation')
            selection_sha = sha(args.frozen_selection)
            frozen = json.loads(Path(args.frozen_selection).read_text())
            gammas = frozen_routes(frozen, args.no_aug_checkpoint, args.aug_checkpoint)
            initial.update(locked_inputs(frozen, args.no_aug_checkpoint, args.aug_checkpoint, manifest))
            for source in sources:
                path = str(source.resolve())
                if frozen['evaluation_inputs_sha256'].get(path) != initial[path]:
                    raise ValueError('metadata_route_source_not_frozen:' + path)
            initial[str(Path(args.frozen_selection).resolve())] = selection_sha
        summary = json.loads((blend / 'summary.json').read_text())
        validate_origin(summary, manifest, args.no_aug_checkpoint, args.aug_checkpoint,
                        initial[str(Path(args.cache).resolve())], args.stage, selection_sha)
        predictions = {}
        for gamma in sorted(set(gammas) | {0.}):
            if gamma not in summary['gammas']:
                raise ValueError('metadata_route_required_blend_gamma_missing')
            rows = []
            for name in ('predictions.jsonl', 'missing_predictions.jsonl'):
                path = blend / ('gamma_' + str(gamma)) / name
                initial.update(snapshot_inputs([path]))
                rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
            predictions[str(gamma)] = rows
        initial.update(snapshot_inputs([], manifest['source_inputs_sha256']))
        outputs, reports, counts = route_predictions(data, predictions, gammas)
        for gamma, rows in outputs.items():
            folder = out / ('gamma_' + gamma)
            folder.mkdir()
            for name, subset in (('predictions.jsonl', [r for r in rows if r['variant'] == 'observed']),
                                 ('missing_predictions.jsonl', [r for r in rows if r['variant'] != 'observed'])):
                (folder / name).write_text(''.join(json.dumps(row) + '\n' for row in subset))
            (folder / 'summary.json').write_text(json.dumps(reports[gamma], indent=2))
        verify_unchanged(initial)
        result = dict(policy=POLICY, predicate_fields=['windows', 'partial_peaks'],
            gammas=gammas, metrics=reports, routing_counts=counts, queries=manifest['queries'],
            cohort=manifest['cohort'], no_aug_checkpoint=str(Path(args.no_aug_checkpoint).resolve()),
            aug_checkpoint=str(Path(args.aug_checkpoint).resolve()), blend_dir=str(blend.resolve()),
            coarse_config=manifest['coarse_config'], candidate_order_sha256=manifest['candidate_order_sha256'],
            inputs_sha256=initial, frozen_selection_sha256=selection_sha,
            selection_scope='train_val_only' if args.stage == 'train_val' else 'frozen_metadata_route_configs',
            device='cpu', model_forward_calls=0, external_api_calls=0)
        (out / 'summary.json').write_text(json.dumps(result, indent=2))
        (out / 'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        (out / 'failure.json').write_text(json.dumps(dict(error=repr(exc))))
        (out / 'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

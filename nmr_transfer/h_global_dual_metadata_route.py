"""功能概述：用公开观测状态，在同一固定 pair 的两个已声明 blend 之间路由。

route_predictions 对 windows/partial_peaks 为真的查询使用 missing_gamma，其余
使用 normal_gamma。未知观测默认走 normal 分支，不把未知解释成完整谱，也不按
source、ID、variant、SMILES 或正确候选决定分支。所有排名直接复用既有 blend，
沿用 h_global_metadata_route 的完整库/身份/名单外排名、origin 与始末 SHA 验证，
以及 h_global_ranker_blend 的 pair/coarse/candidate/无补正例门禁，不运行模型。

train_val 只接纳 original_val/tune，两个 gamma 必须由调用方明确传入；本模块不
搜索或选择参数。locked 的参数必须逐项匹配 dual_metadata_route_configs 中的
有序 pair/policy/normal_gamma/missing_gamma，实际新模块、旧路由 helper 与全部
读取来源均须冻结源码 SHA，且 blend 必须来自同一 frozen selection。每个 out
只有一个组合，直接写原谱/压力预测、完整 metrics、输入 SHA 与真实 EXIT_CODE。
本文件独立于已 pinned route 与 critical 源码；不访问网络、外部 API 或 GPU，
不重建观测或恢复删峰，不修改已有数据、权重、blend 和实验产物。
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from .h_global_metadata_route import (POLICY, explicit_missing, row_key,
    snapshot_inputs, validate_origin, validate_predictions, verify_unchanged)
from .h_global_ranker import metrics
from .h_global_ranker_blend import GAMMAS, locked_inputs, validate_pair
from .prepare import sha


def validate_gamma(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in GAMMAS:
        raise ValueError('dual_metadata_route_unregistered_gamma')
    return float(value)


def frozen_route(frozen, no_aug_path, aug_path, normal_gamma, missing_gamma):
    """精确核验调用方已声明的组合；不从封存结果挑选任何 gamma。"""
    requested = (str(Path(no_aug_path).resolve()), str(Path(aug_path).resolve()),
                 POLICY, validate_gamma(normal_gamma), validate_gamma(missing_gamma))
    seen = set()
    required = {'no_aug_checkpoint', 'aug_checkpoint', 'policy', 'normal_gamma', 'missing_gamma'}
    for config in frozen.get('dual_metadata_route_configs', []):
        if set(config) != required or config['policy'] != POLICY:
            raise ValueError('dual_metadata_route_invalid_frozen_config')
        paths = tuple(str(Path(config[key]).resolve()) for key in ('no_aug_checkpoint', 'aug_checkpoint'))
        if any(not Path(config[key]).is_absolute() or str(Path(config[key]).resolve()) != config[key]
               for key in ('no_aug_checkpoint', 'aug_checkpoint')):
            raise ValueError('dual_metadata_route_noncanonical_frozen_pair')
        key = paths + (config['policy'], validate_gamma(config['normal_gamma']), validate_gamma(config['missing_gamma']))
        if key in seen:
            raise ValueError('dual_metadata_route_duplicate_frozen_config')
        seen.add(key)
    if requested not in seen:
        raise ValueError('dual_metadata_route_exact_pair_and_gammas_not_frozen')
    return requested[-2:]


def route_predictions(data, predictions, normal_gamma, missing_gamma):
    """只用两个公开观测字段选分支；身份字段仅供既有验证器核验。"""
    normal_gamma, missing_gamma = validate_gamma(normal_gamma), validate_gamma(missing_gamma)
    branches = {gamma: validate_predictions(predictions[str(gamma)], data)
                for gamma in {normal_gamma, missing_gamma}}
    rows, flags = [], []
    for original in data['rows']:
        observation = original.get('query_observation')
        if observation is None:
            observation = {}
        if not isinstance(observation, dict):
            raise ValueError('dual_metadata_route_invalid_cached_observation')
        missing = explicit_missing(observation)
        selected = missing_gamma if missing else normal_gamma
        result = dict(branches[selected][row_key(original)])
        result.update(route_gamma=selected, explicit_missing=missing)
        rows.append(result)
        flags.append(missing)
    counts = Counter('explicit_missing' if flag else 'undeclared_missing' for flag in flags)
    return rows, metrics(rows), dict(counts)


def sources():
    here = Path(__file__)
    return [here, here.with_name('h_global_metadata_route.py'), here.with_name('h_global_ranker_blend.py'),
            here.with_name('h_global_ranker.py'), here.with_name('h_global_features.py'), here.with_name('prepare.py')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['train_val', 'locked'], required=True)
    for name in ('no-aug-checkpoint', 'aug-checkpoint', 'cache', 'blend-dir', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--normal-gamma', type=float, required=True)
    parser.add_argument('--missing-gamma', type=float, required=True)
    parser.add_argument('--frozen-selection')
    args = parser.parse_args()
    out, blend = Path(args.out), Path(args.blend_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(4)
        normal_gamma, missing_gamma = validate_gamma(args.normal_gamma), validate_gamma(args.missing_gamma)
        source_paths = sources()
        paths = source_paths + [Path(args.cache), Path(args.no_aug_checkpoint), Path(args.aug_checkpoint),
                                blend / 'summary.json', blend / 'EXIT_CODE']
        initial = snapshot_inputs(paths)
        if (blend / 'EXIT_CODE').read_text().strip() != '0':
            raise ValueError('dual_metadata_route_blend_not_complete')
        frozen, selection_sha = None, None
        if args.stage == 'locked':
            if not args.frozen_selection:
                raise ValueError('dual_metadata_route_locked_selection_required')
            selection_sha = sha(args.frozen_selection)
            frozen = json.loads(Path(args.frozen_selection).read_text())
            if frozen.get('selection_scope') != 'train_val_only' or not frozen.get('frozen_utc'):
                raise ValueError('dual_metadata_route_finalized_selection_required')
            frozen_route(frozen, args.no_aug_checkpoint, args.aug_checkpoint, normal_gamma, missing_gamma)
            for source in source_paths:
                path = str(source.resolve())
                if frozen['evaluation_inputs_sha256'].get(path) != initial[path]:
                    raise ValueError('dual_metadata_route_source_not_frozen:' + path)
            initial[str(Path(args.frozen_selection).resolve())] = selection_sha
        elif args.frozen_selection:
            raise ValueError('dual_metadata_route_train_val_frozen_selection_not_allowed')
        no_aug = torch.load(args.no_aug_checkpoint, map_location='cpu', weights_only=False)
        aug = torch.load(args.aug_checkpoint, map_location='cpu', weights_only=False)
        data = torch.load(args.cache, map_location='cpu', weights_only=False, mmap=True)
        manifest = data['manifest']
        validate_pair(no_aug, aug, data)
        if args.stage == 'train_val':
            if manifest['cohort'] not in ('original_val', 'tune'):
                raise ValueError('dual_metadata_route_train_val_access_violation')
        else:
            if manifest['cohort'] not in ('verify', 'test'):
                raise ValueError('dual_metadata_route_locked_access_violation')
            initial.update(locked_inputs(frozen, args.no_aug_checkpoint, args.aug_checkpoint, manifest))
        summary = json.loads((blend / 'summary.json').read_text())
        validate_origin(summary, manifest, args.no_aug_checkpoint, args.aug_checkpoint,
                        initial[str(Path(args.cache).resolve())], args.stage, selection_sha)
        predictions = {}
        for gamma in sorted({normal_gamma, missing_gamma}):
            if gamma not in summary['gammas']:
                raise ValueError('dual_metadata_route_required_blend_gamma_missing')
            rows = []
            for name in ('predictions.jsonl', 'missing_predictions.jsonl'):
                path = blend / ('gamma_' + str(gamma)) / name
                initial.update(snapshot_inputs([path]))
                rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
            predictions[str(gamma)] = rows
        initial.update(snapshot_inputs([], manifest['source_inputs_sha256']))
        rows, reports, counts = route_predictions(data, predictions, normal_gamma, missing_gamma)
        for name, subset in (('predictions.jsonl', [row for row in rows if row['variant'] == 'observed']),
                             ('missing_predictions.jsonl', [row for row in rows if row['variant'] != 'observed'])):
            (out / name).write_text(''.join(json.dumps(row) + '\n' for row in subset))
        verify_unchanged(initial)
        result = dict(policy=POLICY, predicate_fields=['windows', 'partial_peaks'],
            normal_gamma=normal_gamma, missing_gamma=missing_gamma, metrics=reports,
            routing_counts=counts, queries=manifest['queries'], cohort=manifest['cohort'],
            no_aug_checkpoint=str(Path(args.no_aug_checkpoint).resolve()),
            aug_checkpoint=str(Path(args.aug_checkpoint).resolve()),
            no_aug_sha256=initial[str(Path(args.no_aug_checkpoint).resolve())],
            aug_sha256=initial[str(Path(args.aug_checkpoint).resolve())],
            cache_sha256=initial[str(Path(args.cache).resolve())], blend_dir=str(blend.resolve()),
            coarse_config=manifest['coarse_config'], candidate_order_sha256=manifest['candidate_order_sha256'],
            inputs_sha256=initial, source_sha256=initial[str(Path(__file__).resolve())],
            frozen_selection_sha256=selection_sha,
            selection_scope='train_val_only' if args.stage == 'train_val' else 'frozen_dual_metadata_route_configs',
            device='cpu', model_forward_calls=0, external_api_calls=0)
        (out / 'summary.json').write_text(json.dumps(result, indent=2))
        (out / 'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        (out / 'failure.json').write_text(json.dumps(dict(error=repr(exc))))
        (out / 'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

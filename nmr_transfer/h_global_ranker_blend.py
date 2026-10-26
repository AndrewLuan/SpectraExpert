"""功能概述：在相同粗排缓存上混合 no_aug/aug 重排残差，保留完整候选库。

GAMMAS 预声明五个全查询共用权重；validate_pair 核对两个 checkpoint 的预测、
粗排、特征、候选次序与共同训练缓存，拒绝验证补正例。combined_residual 将各支
saved alpha、delta 和自身 scale[0] 换回原评分单位，再作凸组合。evaluate_blend
分别按各自 scale 规范化并只在 CPU 前向，端点直接复用原 rank_one 单模型路径；
中间权重仍用 rank_one，名单外排名、tie 回退与训练时协议一致。source/ID/variant/
正确结构只用于身份与指标，不选择权重或模型。没有观测 metadata 硬路由。

main 的 train_val 只接受 original_val/tune，输出五权重的原谱/压力预测与指标。
locked 只接受 verify/test，要求两权重、源码和缓存来源 SHA 已冻结；权重精确读取
blend_configs=[{no_aug_checkpoint:绝对路径, aug_checkpoint:绝对路径, gamma:数值}]，
不能在 locked 搜索或新增 pair。原缓存/核心模块/旧结果只读，输出目录必须新建，
失败保留现场并写 EXIT_CODE；不训练、不使用 GPU、不访问网络或外部 API。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .h_global_features import FEATURE_NAMES
from .h_global_ranker import ResidualRanker, metrics, normalized_rows, pack, rank_one
from .prepare import sha


GAMMAS = (0., .25, .5, .75, 1.)
PAIR_FIELDS = ('coarse_config', 'pool_size', 'feature_code_sha256',
               'feature_names', 'candidate_order_sha256')


def validate_pair(no_aug, aug, data):
    """核验共同协议与查询唯一性；不根据查询身份选择混合权重。"""
    manifest = data['manifest']
    if no_aug['args'].get('augment') is not False or aug['args'].get('augment') is not True:
        raise ValueError('blend_requires_no_aug_and_aug_checkpoints')
    if no_aug['train_sha256'] != aug['train_sha256']:
        raise ValueError('blend_training_cache_mismatch')
    for field in PAIR_FIELDS:
        if no_aug[field] != aug[field] or no_aug[field] != manifest[field]:
            raise ValueError('blend_protocol_mismatch:' + field)
    if (no_aug['prediction_checkpoint_sha256'] != aug['prediction_checkpoint_sha256']
            or no_aug['prediction_checkpoint_sha256'] != manifest['checkpoint_sha256']):
        raise ValueError('blend_prediction_checkpoint_mismatch')
    if manifest['feature_names'] != FEATURE_NAMES:
        raise ValueError('blend_feature_schema_mismatch')
    order = hashlib.sha256('\n'.join(data['smiles']).encode()).hexdigest()
    if order != manifest['candidate_order_sha256'] or len(data['smiles']) != manifest['candidate_count']:
        raise ValueError('blend_candidate_order_mismatch')
    for saved in (no_aug, aug):
        scale = saved['scale']
        if (scale.ndim != 1 or len(scale) != len(FEATURE_NAMES)
                or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('blend_invalid_scale')
        if not math.isfinite(saved['alpha']) or saved['alpha'] < 0:
            raise ValueError('blend_invalid_saved_alpha')
    rows = data['rows']
    identities = [(r['query_order'], r['variant']) for r in rows]
    if not rows or len(identities) != len(set(identities)) or len(rows) != manifest['queries']:
        raise ValueError('blend_duplicate_or_missing_cached_query')
    if manifest.get('appended_training_positives', 0):
        raise ValueError('blend_evaluation_appended_positive')
    for row in rows:
        indices = row['candidate_indices']
        size = len(indices)
        position = row['target_position']
        if (position >= 0) != bool(row['target_in_coarse_pool']):
            raise ValueError('blend_evaluation_positive_injection')
        if position >= size:
            raise ValueError('blend_target_position_out_of_pool')
        if (size == 0 or indices.ndim != 1 or indices.unique().numel() != size
                or (indices < 0).any() or (indices >= len(data['smiles'])).any()):
            raise ValueError('blend_invalid_candidate_indices')
        if (row['features'].shape != (size, len(FEATURE_NAMES))
                or row['coarse_scores'].shape != (size,)
                or not torch.isfinite(row['features']).all()
                or not torch.isfinite(row['coarse_scores']).all()):
            raise ValueError('blend_invalid_cached_features')


def frozen_gammas(frozen, no_aug_path, aug_path):
    """只返回冻结列表中精确匹配该有序 checkpoint pair 的预声明权重。"""
    left, right = str(Path(no_aug_path).resolve()), str(Path(aug_path).resolve())
    selected, seen = [], set()
    for config in frozen.get('blend_configs', []):
        if set(config) != {'no_aug_checkpoint', 'aug_checkpoint', 'gamma'}:
            raise ValueError('blend_invalid_frozen_config')
        gamma = config['gamma']
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or gamma not in GAMMAS:
            raise ValueError('blend_unregistered_frozen_gamma')
        pair = (str(Path(config['no_aug_checkpoint']).resolve()),
                str(Path(config['aug_checkpoint']).resolve()), float(gamma))
        if pair in seen:
            raise ValueError('blend_duplicate_frozen_config')
        seen.add(pair)
        if pair[:2] == (left, right):
            selected.append(float(gamma))
    if not selected:
        raise ValueError('blend_checkpoint_pair_not_frozen')
    return selected


def combined_residual(no_delta, aug_delta, no_aug, aug, gamma):
    """两个分支先各自换回原 coarse score 单位，再以全局 gamma 混合。"""
    no_raw = no_aug['alpha'] * no_delta * float(no_aug['scale'][0])
    aug_raw = aug['alpha'] * aug_delta * float(aug['scale'][0])
    return (1 - gamma) * no_raw + gamma * aug_raw


@torch.inference_mode()
def evaluate_blend(no_aug, aug, data, gammas=GAMMAS):
    """对同一缓存计算固定权重；返回逐行预测和原谱/压力分层指标。"""
    validate_pair(no_aug, aug, data)
    gammas = list(dict.fromkeys(float(g) for g in gammas))
    if not gammas or any(g not in GAMMAS for g in gammas):
        raise ValueError('blend_unregistered_gamma')
    device = torch.device('cpu')
    models = []
    for saved in (no_aug, aug):
        model = ResidualRanker(saved['columns'], saved['architecture'], saved['typed']).to(device)
        model.load_state_dict(saved['state_dict'])
        models.append(model.eval())
    no_rows = normalized_rows(data['rows'], no_aug['scale'])
    aug_rows = normalized_rows(data['rows'], aug['scale'])
    outputs = {str(g): [] for g in gammas}
    for start in range(0, len(no_rows), 64):
        left, right = no_rows[start:start + 64], aug_rows[start:start + 64]
        nz, _, nk = pack(left, device)
        az, _, ak = pack(right, device)
        no_base, no_delta = models[0](nz, nk)
        aug_base, aug_delta = models[1](az, ak)
        for j, row in enumerate(data['rows'][start:start + 64]):
            for gamma in gammas:
                if gamma == 0:
                    rank, top = rank_one(left[j], no_base[j], no_delta[j], no_aug['alpha'])
                elif gamma == 1:
                    rank, top = rank_one(right[j], aug_base[j], aug_delta[j], aug['alpha'])
                else:
                    delta = combined_residual(no_delta[j], aug_delta[j], no_aug, aug, gamma)
                    rank, top = rank_one(dict(row, score_scale=1.), row['coarse_scores'], delta, 1.)
                outputs[str(gamma)].append(dict(source=row['source'], record_id=row['record_id'],
                    smiles=row['smiles'], group=row['group'], variant=row['variant'], rank=rank,
                    top10_indices=top, query_order=row['query_order'],
                    baseline_rank=row['baseline_rank'], coarse_rank=row['coarse_rank']))
    return outputs, {g: metrics(rows) for g, rows in outputs.items()}


def locked_inputs(frozen, no_aug_path, aug_path, manifest):
    """在读封存缓存后、任何模型计算前核对两权重与实际算法/缓存来源。"""
    if frozen.get('selection_scope') != 'train_val_only':
        raise ValueError('blend_invalid_frozen_selection')
    pinned = frozen['evaluation_inputs_sha256']
    checks = {}
    for path in (no_aug_path, aug_path):
        path = Path(path).resolve()
        digest = frozen['ranker_checkpoints_sha256'].get(str(path))
        if not digest or sha(path) != digest:
            raise ValueError('blend_checkpoint_not_frozen:' + str(path))
        checks[str(path)] = digest
    required_sources = (Path(__file__).resolve(), Path(__file__).with_name('h_global_ranker.py').resolve(),
                        Path(__file__).with_name('h_global_features.py').resolve())
    required = {str(p): pinned.get(str(p)) for p in required_sources}
    required.update(manifest['source_inputs_sha256'])
    for path, digest in required.items():
        if not digest or pinned.get(path) != digest or sha(path) != digest:
            raise ValueError('blend_source_or_cache_input_not_frozen:' + path)
        checks[path] = digest
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['train_val', 'locked'], required=True)
    for name in ('no-aug-checkpoint', 'aug-checkpoint', 'cache', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--frozen-selection')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(4)
        cache_sha = sha(args.cache)
        no_sha, aug_sha = sha(args.no_aug_checkpoint), sha(args.aug_checkpoint)
        no_aug = torch.load(args.no_aug_checkpoint, map_location='cpu', weights_only=False)
        aug = torch.load(args.aug_checkpoint, map_location='cpu', weights_only=False)
        data = torch.load(args.cache, map_location='cpu', weights_only=False)
        manifest = data['manifest']
        checks, selection_sha = {}, None
        if args.stage == 'train_val':
            if manifest['cohort'] not in ('original_val', 'tune'):
                raise ValueError('blend_train_val_access_violation')
            gammas = list(GAMMAS)
        else:
            if manifest['cohort'] not in ('verify', 'test') or not args.frozen_selection:
                raise ValueError('blend_locked_access_violation')
            selection_sha = sha(args.frozen_selection)
            frozen = json.loads(Path(args.frozen_selection).read_text())
            gammas = frozen_gammas(frozen, args.no_aug_checkpoint, args.aug_checkpoint)
            checks = locked_inputs(frozen, args.no_aug_checkpoint, args.aug_checkpoint, manifest)
        results, reports = evaluate_blend(no_aug, aug, data, gammas)
        for gamma, rows in results.items():
            if len(rows) != manifest['queries']:
                raise ValueError('blend_output_count_changed')
            folder = out / ('gamma_' + gamma)
            folder.mkdir()
            for row in rows:
                row['top10'] = [data['smiles'][i] for i in row.pop('top10_indices')]
            for filename, subset in (
                    ('predictions.jsonl', [r for r in rows if r['variant'] == 'observed']),
                    ('missing_predictions.jsonl', [r for r in rows if r['variant'] != 'observed'])):
                (folder / filename).write_text(''.join(json.dumps(r) + '\n' for r in subset))
            (folder / 'summary.json').write_text(json.dumps(reports[gamma], indent=2))
        for path, digest in checks.items():
            if sha(path) != digest:
                raise ValueError('blend_frozen_input_changed_during_run:' + path)
        if (sha(args.cache) != cache_sha or sha(args.no_aug_checkpoint) != no_sha
                or sha(args.aug_checkpoint) != aug_sha):
            raise ValueError('blend_cache_or_checkpoint_changed_during_run')
        if selection_sha and sha(args.frozen_selection) != selection_sha:
            raise ValueError('blend_selection_changed_during_run')
        summary = dict(metrics=reports, gammas=gammas, queries=manifest['queries'],
            cohort=manifest['cohort'], no_aug_checkpoint=str(Path(args.no_aug_checkpoint).resolve()),
            aug_checkpoint=str(Path(args.aug_checkpoint).resolve()), no_aug_sha256=no_sha,
            aug_sha256=aug_sha, cache_sha256=cache_sha, source_sha256=sha(__file__),
            frozen_selection_sha256=selection_sha, coarse_config=manifest['coarse_config'],
            candidate_order_sha256=manifest['candidate_order_sha256'],
            selection_scope='train_val_only' if args.stage == 'train_val' else 'frozen_blend_configs',
            device='cpu', external_api_calls=0)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        (out / 'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        (out / 'failure.json').write_text(json.dumps(dict(error=repr(exc))))
        (out / 'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

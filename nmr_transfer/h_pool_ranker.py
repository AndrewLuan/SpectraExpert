"""功能概述：在冻结观测模型下比较候选池训练与排序损失，保留旧轮封存代码。
主要职责：validate_caches核对两池的查询、观测和科学输入；training_schedule生成各臂
相同的结构/变体曝光与独立平衡池序列；pool_loss实现困难16负例或全合格候选列表损失。
fit继承旧R0O1F0权重与scale，只更新network；load_checkpoint/validate_pair兼容旧基准
及新模型；evaluate只评价自然512候选，输出逐查询候选成本供同池诊断。
输入输出：CLI fit接收single/dual train缓存和旧初始checkpoint；evaluate接收缓存及
正常/增强模型对。producer EXIT、文件SHA、负例排除和冻结清单均失败即停；写入新输出
目录的初始/最佳/末轮权重、历史、协议、状态和真实内部退出码，不修改旧缓存或checkpoint。
"""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .h_research_ranker import (ResearchRanker, internal_group, read_feature_cache,
    select_rows, result_row, validation)
from .h_research_contract import atomic_json, input_fingerprints, rank_shortlist, summarize_predictions

SCHEMA = 'nmr_pool_ranker_20261023'
OLD_SCHEMA = 'nmr_research_ranker_20261022'
IDENTITY = ('schema', 'candidate_order_sha256', 'checkpoint_sha256', 'pool_size')
PAIR_IDENTITY = ('query_sha256', 'identity_content_sha256', 'variant_mask_content_sha256',
                 'negative_excluded_structure_sha256', 'training_subset')
BUDGET = ('epochs', 'samples', 'batch_size', 'anchor', 'lr', 'training_pool', 'loss')


def source_paths():
    names = ('h_pool_ranker.py', 'h_research_ranker.py', 'h_research_observation.py',
             'h_research_contract.py', 'h_global_ranker.py', 'h_global_features.py')
    return [Path(__file__).with_name(n) for n in names]


def row_key(row):
    return tuple(row[k] for k in ('query_order', 'variant', 'source', 'record_id', 'smiles', 'group'))


def validate_row(row, training):
    n = int(row['evaluation_pool_size'])
    indices = row['candidate_indices']
    target = int(row['target_position'])
    if n != 512 or len(indices.unique()) != len(indices) or len(indices) not in (n, n+1):
        raise ValueError('pool_invalid_natural_candidates')
    if not training and (len(indices) != n or target >= n):
        raise ValueError('pool_evaluation_positive_injection')
    if training:
        mask = row['negative_exclusion_mask']
        if mask.dtype != torch.bool or len(mask) != len(indices) or not 0 <= target < len(indices):
            raise ValueError('pool_invalid_training_target_or_mask')
        if len(indices) == n and not row['target_in_coarse_pool']:
            raise ValueError('pool_training_presence_mismatch')
        if len(indices) == n+1 and (target != n or row['target_in_coarse_pool']):
            raise ValueError('pool_invalid_appended_positive')
    elif bool(target >= 0) != bool(row['target_in_coarse_pool']):
        raise ValueError('pool_target_presence_mismatch')
    for field in ('features', 'coarse_scores', 'predicted_ppm', 'tags', 'H', 'K'):
        if len(row[field]) != len(indices):
            raise ValueError('pool_candidate_tensor_length:' + field)


def same_value(a, b):
    """递归比较缓存中的真实观测内容，不只依赖声明的摘要。"""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    return a == b


def validate_caches(single, dual):
    a, b = single['manifest'], dual['manifest']
    if a['cohort'] != 'train' or b['cohort'] != 'train':
        raise ValueError('pool_requires_training_caches')
    if a['recall'] != 'single' or b['recall'] != 'dual':
        raise ValueError('pool_cache_roles')
    for key in IDENTITY + PAIR_IDENTITY:
        if a.get(key) != b.get(key):
            raise ValueError('pool_cache_identity:' + key)
    if a['source_inputs_sha256'] != b['source_inputs_sha256']:
        raise ValueError('pool_shared_science_mismatch')
    if a.get('candidate_count') != 114982 or a.get('pool_size') != 512:
        raise ValueError('pool_candidate_universe_changed')
    if not a.get('negative_excluded_structure_sha256'):
        raise ValueError('pool_missing_negative_exclusion_provenance')
    ar, br = single['rows'], dual['rows']
    if len(ar) != len(br) or len({row_key(r) for r in ar}) != len(ar):
        raise ValueError('pool_training_query_count_or_duplicate')
    lookup = {row_key(r): r for r in br}
    if len(lookup) != len(br) or set(lookup) != {row_key(r) for r in ar}:
        raise ValueError('pool_training_query_identity')
    for row in ar:
        other = lookup[row_key(row)]
        if row['mask_content_sha256'] != other['mask_content_sha256']:
            raise ValueError('pool_observation_mask_changed')
        for field in ('kind', 'observation_input', 'query_observation'):
            if not same_value(row[field], other[field]):
                raise ValueError('pool_observation_content_changed:' + field)
        validate_row(row, True); validate_row(other, True)
        if int(row['candidate_indices'][row['target_position']]) != int(other['candidate_indices'][other['target_position']]):
            raise ValueError('pool_target_candidate_mismatch')
        exclusions = dict(zip(row['candidate_indices'].tolist(), row['negative_exclusion_mask'].tolist()))
        for idx, excluded in zip(other['candidate_indices'].tolist(), other['negative_exclusion_mask'].tolist()):
            if idx in exclusions and exclusions[idx] != excluded:
                raise ValueError('pool_shared_negative_exclusion_mismatch')
    return lookup


def training_schedule(names, groups, seed, epoch, samples):
    """池选择使用独立随机流，不改变结构或变体曝光；偶数draws严格各半。"""
    rng = np.random.default_rng(seed+2000+epoch)
    vrng = np.random.default_rng(seed+4000+epoch)
    sequence = rng.integers(0, len(names), size=samples)
    pools = np.arange(samples) % 2
    np.random.default_rng(seed+6000+epoch).shuffle(pools)
    return [(groups[names[int(index)]][int(vrng.integers(len(groups[names[int(index)]])))],
             int(pools[pos])) for pos, index in enumerate(sequence)]


def pool_loss(row, delta, scale, anchor, mode):
    target = int(row['target_position'])
    if not 0 <= target < len(delta):
        raise ValueError('pool_training_positive_missing')
    costs = row['coarse_scores'].to(delta.device)/scale[0]+delta
    excluded = row['negative_exclusion_mask'].to(delta.device).clone()
    excluded[target] = True
    negatives = costs[~excluded]
    if not len(negatives):
        raise ValueError('pool_no_eligible_negatives')
    allowed = ~excluded
    allowed[target] = True
    if mode == 'pairwise16':
        hard = negatives.topk(min(16, len(negatives)), largest=False).values
        loss = F.softplus((costs[target]-hard)/.5).mean()
    elif mode == 'listwise':
        loss = torch.logsumexp(-costs[allowed]/.5, 0)+costs[target]/.5
    else:
        raise ValueError('pool_unknown_loss')
    return loss+anchor*delta[allowed].square().mean()


def load_checkpoint(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('schema') not in (SCHEMA, OLD_SCHEMA) or not saved.get('observation'):
        raise ValueError('pool_requires_observation_checkpoint')
    model = ResearchRanker(True, saved['initial_sigmas']).to(device)
    model.load_state_dict(saved['state_dict'], strict=True)
    for parameter in model.observation_model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, saved['scale'].to(device), saved


def check_science(saved, data):
    for key in IDENTITY:
        if data['manifest'].get(key) != saved['cache_protocol'].get(key):
            raise ValueError('pool_model_cache_identity:' + key)
    if data['manifest']['source_inputs_sha256'] != saved['source_inputs_sha256']:
        raise ValueError('pool_model_science_mismatch')
    if data['manifest'].get('recall') not in ('single', 'dual'):
        raise ValueError('pool_unknown_recall')
    pins = saved['ranker_source_sha256']
    if input_fingerprints(pins) != pins:
        raise ValueError('pool_model_source_changed')


def save_checkpoint(path, model, scale, args, data, epoch, alpha, exposure_sha, pool_sha, inputs):
    torch.save(dict(schema=SCHEMA,
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        scale=scale.detach().cpu(), observation=True,
        initial_sigmas=model.observation_model.initial_sigmas.detach().cpu().tolist(),
        epoch=epoch, alpha=alpha, args=vars(args), exposure_sha256=exposure_sha,
        pool_exposure_sha256=pool_sha, train_sha256=inputs[str(Path(args.train).resolve())],
        dual_train_sha256=inputs[str(Path(args.dual_train).resolve())],
        initialization_sha256=inputs[str(Path(args.initial_checkpoint).resolve())],
        cache_protocol={k:data['manifest'][k] for k in IDENTITY},
        source_inputs_sha256=data['manifest']['source_inputs_sha256'],
        ranker_source_sha256=input_fingerprints(source_paths())), path)


def fit(args, out, device, status, pins):
    single, _ = read_feature_cache(args.train)
    dual, _ = read_feature_cache(args.dual_train)
    dual_lookup = validate_caches(single, dual)
    rows = select_rows(single, args.augment)
    train = [r for r in rows if not internal_group(r['group'])]
    valid = [r for r in rows if internal_group(r['group'])]
    if not train or not valid:
        raise ValueError('pool_empty_internal_split')
    groups = collections.defaultdict(list)
    for row in train:
        groups[row['group']].append(row)
    names = sorted(groups)
    torch.manual_seed(args.seed)
    model, scale, initial = load_checkpoint(args.initial_checkpoint, device)
    check_science(initial, single)
    if initial['args']['seed'] != args.seed or initial['args']['augment'] is not args.augment:
        raise ValueError('pool_initial_checkpoint_role')
    if initial['cache_protocol'].get('recall') != 'single':
        raise ValueError('pool_initial_checkpoint_not_R0')
    if initial['train_sha256'] != pins[str(Path(args.train).resolve())]:
        raise ValueError('pool_initial_training_cache_changed')
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=args.lr, weight_decay=.001)
    initial_o = {k:v.detach().cpu().clone() for k,v in model.observation_model.state_dict().items()}
    exposure, pool_exposure = hashlib.sha256(), hashlib.sha256()
    alphas = [0., .25, .5, 1.]
    best, chosen = -float('inf'), None
    started = time.monotonic()
    for epoch in range(args.epochs+1):
        total, steps = 0., 0
        if epoch:
            model.train()
            model.observation_model.eval()
            schedule = training_schedule(names, groups, args.seed, epoch, args.samples)
            for begin in range(0, len(schedule), args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for row, pool in schedule[begin:begin+args.batch_size]:
                    exposure.update(json.dumps(row_key(row), separators=(',', ':')).encode()+b'\n')
                    use_dual = args.training_pool == 'mixed' and pool == 1
                    pool_exposure.update(bytes([int(use_dual)]))
                    selected = dual_lookup[row_key(row)] if use_dual else row
                    delta = model.row_delta(selected, scale, device)
                    losses.append(pool_loss(selected, delta, scale, args.anchor, args.loss))
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise ValueError('pool_nonfinite_loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.network.parameters(), 2.)
                optimizer.step()
                total += float(loss.detach()); steps += 1
        if epoch == 0 or epoch % 8 == 0 or epoch == args.epochs:
            _, reports = validation(model, valid, scale, device, alphas)
            alpha = max(alphas, key=lambda a:(reports[str(a)]['overall']['MRR'], -a))
            signal = reports[str(alpha)]['overall']['MRR']
            if epoch == 0:
                save_checkpoint(out/'initial.pt', model, scale, args, single, epoch,
                                initial['alpha'], exposure.hexdigest(), pool_exposure.hexdigest(), pins)
            if signal > best+1e-5:
                best = signal
                chosen = dict(epoch=epoch, alpha=alpha, MRR=signal)
                save_checkpoint(out/'best.pt', model, scale, args, single, epoch, alpha,
                                exposure.hexdigest(), pool_exposure.hexdigest(), pins)
            record = dict(epoch=epoch, training_loss=total/max(steps,1), internal=reports,
                          best=chosen, seconds=time.monotonic()-started)
            with (out/'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            status('fit', **record)
            print(json.dumps({k:record[k] for k in ('epoch','best','seconds')}), flush=True)
    save_checkpoint(out/'last.pt', model, scale, args, single, args.epochs, chosen['alpha'],
                    exposure.hexdigest(), pool_exposure.hexdigest(), pins)
    changes = {k:float((v.detach().cpu()-initial_o[k]).abs().max())
               for k,v in model.observation_model.state_dict().items()}
    if any(changes.values()):
        raise ValueError('pool_observation_parameters_changed')
    atomic_json(out/'summary.json', dict(best=chosen, training_groups=len(groups),
        internal_groups=len({r['group'] for r in valid}), exposure_sha256=exposure.hexdigest(),
        pool_exposure_sha256=pool_exposure.hexdigest(), observation_parameter_max_changes=changes,
        samples_per_epoch=args.samples, epochs=args.epochs, training_pool=args.training_pool,
        loss=args.loss, input_sha256=pins, elapsed_seconds=time.monotonic()-started,
        external_api_calls=0))


def validate_pair(models, data, seed):
    if len(models) != 2:
        raise ValueError('pool_requires_two_models')
    first, second = (m[2] for m in models)
    for saved, augment in ((first, False), (second, True)):
        check_science(saved, data)
        if saved['args']['seed'] != seed or saved['args']['augment'] is not augment:
            raise ValueError('pool_model_role_mismatch')
    if first['schema'] != second['schema'] or first['train_sha256'] != second['train_sha256']:
        raise ValueError('pool_pair_training_mismatch')
    fields = BUDGET if first['schema'] == SCHEMA else ('epochs','samples','batch_size','anchor')
    for key in fields:
        if first['args'][key] != second['args'][key]:
            raise ValueError('pool_pair_budget_mismatch:' + key)
    if first['schema'] == SCHEMA and first['dual_train_sha256'] != second['dual_train_sha256']:
        raise ValueError('pool_pair_dual_training_mismatch')


def check_frozen(args, data, cache_sha):
    cohort = data['manifest']['cohort']
    if cohort not in ('verify','test'):
        return
    if not args.frozen_selection:
        raise ValueError('pool_frozen_selection_required')
    frozen = json.loads(Path(args.frozen_selection).read_text())
    if (frozen.get('schema') != SCHEMA or not frozen.get('frozen_utc')
            or frozen.get('selection_scope') != 'train_val_only'):
        raise ValueError('pool_invalid_frozen_selection')
    config = dict(no_aug_checkpoint=str(Path(args.no_aug_checkpoint).resolve()),
                  aug_checkpoint=str(Path(args.aug_checkpoint).resolve()), seed=args.seed,
                  cohort=cohort, recall=data['manifest']['recall'], cache_sha256=cache_sha)
    if config not in frozen['allowed_ranker_configs']:
        raise ValueError('pool_config_not_frozen')
    if cohort == 'verify' and frozen.get('verify_eligibility') != 'eligible':
        raise ValueError('pool_independent_acceptance_unavailable')
    pinned = frozen['evaluation_inputs_sha256']
    actual = input_fingerprints(source_paths()+[args.no_aug_checkpoint,args.aug_checkpoint,args.cache])
    actual.update(data['manifest']['source_inputs_sha256'])
    if any(pinned.get(path) != sha for path,sha in actual.items()):
        raise ValueError('pool_frozen_input_mismatch')


@torch.inference_mode()
def evaluate(args, out, device, status, pins):
    data, cache_sha = read_feature_cache(args.cache)
    cohort = data['manifest']['cohort']
    if cohort not in ('original_val','tune','verify','test'):
        raise ValueError('pool_evaluation_wrong_cohort')
    models = [load_checkpoint(path, device) for path in (args.no_aug_checkpoint,args.aug_checkpoint)]
    validate_pair(models, data, args.seed)
    check_frozen(args, data, cache_sha)
    rows, durations = [], []
    started = time.monotonic()
    for i,row in enumerate(data['rows']):
        validate_row(row, False)
        begin = time.monotonic()
        correction = [saved['alpha']*model.row_delta(row, scale, device)*scale[0]
                      for model,scale,saved in models]
        obs = row['query_observation']
        gamma = .75 if obs.get('windows') or obs.get('partial_peaks') else .25
        costs = row['coarse_scores'].to(device)+(1-gamma)*correction[0]+gamma*correction[1]
        rank, top = rank_shortlist(row,costs)
        durations.append(time.monotonic()-begin)
        n = row['evaluation_pool_size']
        rows.append(dict(result_row(row,rank,top), gamma=gamma,
            mask_content_sha256=row['mask_content_sha256'],
            natural_candidate_indices=row['candidate_indices'][:n].tolist(),
            natural_candidate_costs=costs[:n].cpu().tolist()))
        if (i+1)%100 == 0:
            status('evaluating', records=i+1,total=len(data['rows']))
    with (out/'predictions.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    atomic_json(out/'summary.json', dict(metrics=summarize_predictions(rows), cohort=cohort,
        recall=data['manifest']['recall'], query_sha256=data['manifest']['query_sha256'],
        candidate_order_sha256=data['manifest']['candidate_order_sha256'],
        elapsed_seconds=time.monotonic()-started, reranking_seconds_mean=float(np.mean(durations)),
        reranking_seconds_p95=float(np.quantile(durations,.95)), input_sha256=pins,
        checkpoints=input_fingerprints([args.no_aug_checkpoint,args.aug_checkpoint]),external_api_calls=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('fit','evaluate'), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--train'); p.add_argument('--dual-train'); p.add_argument('--initial-checkpoint')
    p.add_argument('--cache'); p.add_argument('--no-aug-checkpoint'); p.add_argument('--aug-checkpoint')
    p.add_argument('--frozen-selection')
    p.add_argument('--training-pool', choices=('original','mixed'), default='original')
    p.add_argument('--loss', choices=('pairwise16','listwise'), default='pairwise16')
    p.add_argument('--seed',type=int,default=211); p.add_argument('--augment',action='store_true')
    p.add_argument('--epochs',type=int,default=64); p.add_argument('--samples',type=int,default=8192)
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--anchor',type=float,default=.1)
    p.add_argument('--lr',type=float,default=.001); p.add_argument('--device',default='cuda')
    args = p.parse_args()
    if min(args.epochs,args.samples,args.batch_size) < 1 or args.samples%2 or args.lr <= 0 or args.anchor < 0:
        raise ValueError('pool_invalid_budget')
    paths = source_paths()
    caches = [args.train,args.dual_train] if args.stage == 'fit' else [args.cache]
    paths += caches
    models = [args.initial_checkpoint] if args.stage == 'fit' else [args.no_aug_checkpoint,args.aug_checkpoint]
    paths += models
    for checkpoint in models:
        parent = Path(checkpoint).parent
        if (parent/'EXIT_CODE').read_text().strip() != '0':
            raise ValueError('pool_checkpoint_producer_not_exit0')
        paths += [parent/'summary.json', parent/'EXIT_CODE']
    if args.frozen_selection:
        paths.append(args.frozen_selection)
    for cache in caches:
        paths += [Path(cache).parent/'summary.json',Path(cache).parent/'EXIT_CODE']
    pins = input_fingerprints(paths)
    out = Path(args.out); out.mkdir(parents=True,exist_ok=False)
    status = lambda stage,**kw:atomic_json(out/'status.json',dict(stage=stage,**kw))
    try:
        torch.set_num_threads(4)
        atomic_json(out/'protocol.json',dict(args=vars(args),inputs_sha256=pins))
        if args.stage == 'fit':
            fit(args,out,torch.device(args.device),status,pins)
        else:
            evaluate(args,out,torch.device(args.device),status,pins)
        if input_fingerprints(pins) != pins:
            raise ValueError('pool_input_changed_during_run')
        status('complete'); (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc)); (out/'EXIT_CODE').write_text('1\n'); raise


if __name__ == '__main__':
    main()

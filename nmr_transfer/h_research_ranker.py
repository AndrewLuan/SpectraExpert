"""功能概述：以相同容量联合训练新轮观测模型和候选重排器。

ResearchRanker在原33维特征后追加三个O特征；O关闭时填零但保留相同网络容量。
costs对同一查询的全部候选调用TrainableObservationModel，未知频窗和公开缺峰标志
由缓存契约传入，不使用来源/ID作网络输入。fit只接收train缓存，结构哈希隔离内部
验证，统一draws/seed曝光，采用难负例排序、评分锚与观测参数正则；保存真实历史和
初始/最佳/末轮权重。evaluate使用固定正常.25/显式缺测.75混合两个训练臂，名单外
遵守完整排序契约。CLI在verify/test核对新轮冻结模型与输入，不改旧源和结果。
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

from .h_global_features import FEATURE_NAMES
from .h_global_ranker import fit_scale, row_center
from .h_research_contract import atomic_json, input_fingerprints, rank_shortlist, summarize_predictions
from .h_research_observation import O_FEATURE_NAMES, TrainableObservationModel


class ResearchRanker(nn.Module):
    def __init__(self, observation=False, initial_sigmas=None):
        super().__init__()
        self.observation_enabled = bool(observation)
        self.observation_model = TrainableObservationModel(initial_sigmas)
        self.network = nn.Sequential(nn.Linear(len(FEATURE_NAMES)+len(O_FEATURE_NAMES), 32),
                                     nn.SiLU(), nn.Linear(32, 4, bias=False))
        nn.init.zeros_(self.network[-1].weight)
        if not self.observation_enabled:
            for parameter in self.observation_model.parameters():
                parameter.requires_grad_(False)

    def row_delta(self, row, scale, device):
        x = row['features'].to(device)
        center = row_center(row).to(device)
        z = ((x-center)/scale).clamp(-20, 20)
        if self.observation_enabled:
            sigma = row.get('per_quantile_sigma')
            extra = self.observation_model(row['predicted_ppm'].to(device), row['tags'].to(device).long(),
                row['observation_input'], H=row['H'].to(device), K=row['K'].to(device),
                per_quantile_sigma=None if sigma is None else sigma.to(device))
            # 正例补入不得参与中心估计；detach让可学习密度保留相对差异的梯度。
            extra_center = extra[:row['evaluation_pool_size']].median(0).values.detach()
            extra = (extra-extra_center).clamp(-20, 20)
        else:
            extra = x.new_zeros((len(x), len(O_FEATURE_NAMES)))
        logits = self.network(torch.cat((z, extra), 1))[:, int(row['kind'])]
        return 5*torch.tanh(logits/5)


def internal_group(group):
    return int(hashlib.sha256(('ranker_internal_317'+str(group)).encode()).hexdigest()[:12], 16) % 5 == 0


def select_rows(data, augment):
    if data['manifest'].get('schema') != 'nmr_research_20261022':
        raise ValueError('research_ranker_cache_schema')
    return [r for r in data['rows'] if augment or r['variant'] == 'observed']


def read_feature_cache(path, known_sha=None):
    """producer实际EXIT和输出SHA均通过才消费cache，不能把半成品当完整输入。"""
    path = Path(path)
    if (path.parent/'EXIT_CODE').read_text().strip() != '0':
        raise ValueError('research_feature_producer_not_exit0')
    summary = json.loads((path.parent/'summary.json').read_text())
    actual = known_sha or input_fingerprints([path])[str(path.resolve())]
    if actual != summary['feature_cache_sha256']:
        raise ValueError('research_feature_cache_SHA_mismatch')
    data = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if any(summary.get(key) != value for key, value in data['manifest'].items()):
        raise ValueError('research_feature_manifest_summary_mismatch')
    return data, actual


def ranking_loss(row, delta, scale, anchor):
    """已隔离结构既不作为负例，也不接受评分锚的梯度；训练正例单独启用。"""
    target = int(row['target_position'])
    if target < 0:
        raise ValueError('research_training_positive_missing')
    costs = row['coarse_scores'].to(delta.device)/scale[0]+delta
    exclusion = row['negative_exclusion_mask'].to(delta.device).clone()
    exclusion[target] = True
    negatives = costs[~exclusion]
    if not len(negatives):
        raise ValueError('research_no_eligible_training_negatives')
    hard = negatives.topk(min(16, len(negatives)), largest=False).values
    allowed = ~exclusion
    allowed[target] = True
    return F.softplus((costs[target]-hard)/.5).mean()+anchor*delta[allowed].square().mean()


def predict_row(model, row, scale, device, alpha):
    delta = model.row_delta(row, scale, device)
    costs = row['coarse_scores'].to(device)+float(alpha)*delta*scale[0]
    rank, top = rank_shortlist(row, costs)
    return rank, top, delta


def result_row(row, rank, top):
    return dict(source=row['source'], record_id=row['record_id'], smiles=row['smiles'], group=row['group'],
                query_order=row['query_order'], variant=row['variant'], rank=rank, top10_indices=top,
                target_in_coarse_pool=row['target_in_coarse_pool'], coarse_rank=row['coarse_rank'],
                baseline_rank=row['baseline_rank'], evaluation_pool_size=row['evaluation_pool_size'])


@torch.inference_mode()
def validation(model, rows, scale, device, alphas):
    model.eval()
    outputs = {str(a): [] for a in alphas}
    for row in rows:
        delta = model.row_delta(row, scale, device)
        base = row['coarse_scores'].to(device)
        for alpha in alphas:
            rank, top = rank_shortlist(row, base+alpha*delta*scale[0])
            outputs[str(alpha)].append(result_row(row, rank, top))
    return outputs, {key: summarize_predictions(value) for key, value in outputs.items()}


def save_checkpoint(path, model, scale, args, data, epoch, alpha, training_sha, exposure_sha):
    torch.save(dict(schema='nmr_research_ranker_20261022', state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        scale=scale.detach().cpu(), observation=args.observation,
        initial_sigmas=model.observation_model.initial_sigmas.detach().cpu().tolist(),
        epoch=epoch, alpha=alpha, args=vars(args), train_sha256=training_sha,
        exposure_sha256=exposure_sha, feature_names=FEATURE_NAMES+list(O_FEATURE_NAMES),
        cache_protocol={k: data['manifest'][k] for k in ('schema', 'recall', 'candidate_order_sha256', 'checkpoint_sha256', 'pool_size')},
        source_inputs_sha256=data['manifest']['source_inputs_sha256'],
        ranker_source_sha256=input_fingerprints([__file__, Path(__file__).with_name('h_research_observation.py'),
                                                Path(__file__).with_name('h_research_contract.py')])), path)


def fit(args, out, device, status):
    data, train_sha = read_feature_cache(args.train, getattr(args, 'execution_cache_sha256', None))
    if data['manifest']['cohort'] != 'train':
        raise ValueError('research_ranker_requires_train')
    rows = select_rows(data, args.augment)
    train = [r for r in rows if not internal_group(r['group'])]
    valid = [r for r in rows if internal_group(r['group'])]
    if not train or not valid:
        raise ValueError('research_empty_internal_split')
    groups = collections.defaultdict(list)
    for row in train:
        groups[row['group']].append(row)
    names = sorted(groups)
    scale = fit_scale(train).to(device)
    widths = (np.asarray(data['manifest']['local_widths_ppm'])*.2)**.5
    torch.manual_seed(args.seed)
    model = ResearchRanker(args.observation, widths.tolist()).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=1e-3)
    alphas = [0., .25, .5, 1.]
    exposure = hashlib.sha256()
    best, chosen = -float('inf'), None
    start = time.monotonic()
    initial_observation = {k: v.detach().cpu().clone() for k, v in model.observation_model.state_dict().items()}
    for epoch in range(args.epochs+1):
        total, steps = 0., 0
        if epoch:
            model.train()
            rng = np.random.default_rng(args.seed+2000+epoch)
            vrng = np.random.default_rng(args.seed+4000+epoch)
            sequence = rng.integers(0, len(names), size=args.samples)
            exposure.update(sequence.tobytes())
            for begin in range(0, len(sequence), args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                selected = sequence[begin:begin+args.batch_size]
                losses = []
                for index in selected:
                    options = groups[names[int(index)]]
                    row = options[int(vrng.integers(len(options)))]
                    exposure.update((row['group']+'\0'+row['variant']+'\n').encode())
                    delta = model.row_delta(row, scale, device)
                    losses.append(ranking_loss(row, delta, scale, args.anchor))
                loss = torch.stack(losses).mean()
                if args.observation:
                    loss = loss+args.observation_regularization*model.observation_model.regularization()
                if not torch.isfinite(loss):
                    raise ValueError('research_ranker_nonfinite_loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
                total += float(loss.detach()); steps += 1
        if epoch == 0 or epoch % 8 == 0 or epoch == args.epochs:
            _, reports = validation(model, valid, scale, device, alphas)
            alpha = max(alphas, key=lambda a: (reports[str(a)]['overall']['MRR'], -a))
            signal = reports[str(alpha)]['overall']['MRR']
            if epoch == 0:
                save_checkpoint(out/'initial.pt', model, scale, args, data, epoch, 0., train_sha, exposure.hexdigest())
            if signal > best+1e-5:
                best = signal; chosen = dict(epoch=epoch, alpha=alpha, MRR=signal)
                save_checkpoint(out/'best.pt', model, scale, args, data, epoch, alpha, train_sha, exposure.hexdigest())
            record = dict(epoch=epoch, training_loss=total/max(steps, 1), internal=reports, best=chosen,
                          seconds=time.monotonic()-start)
            with (out/'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            status('fit', **record); print(json.dumps({k: record[k] for k in ('epoch', 'best', 'seconds')}), flush=True)
    save_checkpoint(out/'last.pt', model, scale, args, data, args.epochs, chosen['alpha'], train_sha, exposure.hexdigest())
    changes = {k: float((v.detach().cpu()-initial_observation[k]).abs().max())
               for k, v in model.observation_model.state_dict().items()}
    atomic_json(out/'summary.json', dict(best=chosen, training_groups=len(groups), internal_groups=len({r['group'] for r in valid}),
        exposure_sha256=exposure.hexdigest(), samples_per_epoch=args.samples, epochs=args.epochs,
        observation_parameter_max_changes=changes, training_cache_sha256=train_sha,
        elapsed_seconds=time.monotonic()-start, external_api_calls=0))


def load_model(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('schema') != 'nmr_research_ranker_20261022':
        raise ValueError('research_ranker_checkpoint_schema')
    model = ResearchRanker(saved['observation'], saved['initial_sigmas']).to(device)
    model.load_state_dict(saved['state_dict'], strict=True); model.eval()
    return model, saved['scale'].to(device), saved


def validate_model_pair(models, data, expected_seed):
    """绑定固定混合的角色、seed、预算和完整共享科学输入，拒绝换序或混配。"""
    if len(models) != 2:
        raise ValueError('research_ranker_pair_requires_two_models')
    first, second = (value[2] for value in models)
    for saved, augmented in ((first, False), (second, True)):
        if saved['args']['augment'] is not augmented:
            raise ValueError('research_ranker_pair_role_mismatch')
        if saved['args']['seed'] != expected_seed:
            raise ValueError('research_ranker_pair_seed_mismatch')
        if any(data['manifest'][key] != value for key, value in saved['cache_protocol'].items()):
            raise ValueError('research_ranker_cache_identity_mismatch')
        if data['manifest']['source_inputs_sha256'] != saved['source_inputs_sha256']:
            raise ValueError('research_ranker_shared_science_mismatch')
        if input_fingerprints(saved['ranker_source_sha256']) != saved['ranker_source_sha256']:
            raise ValueError('research_ranker_source_changed_since_training')
    if first['train_sha256'] != second['train_sha256'] or first['observation'] != second['observation']:
        raise ValueError('research_ranker_pair_training_mismatch')
    for field in ('epochs', 'samples', 'batch_size', 'anchor', 'observation_regularization'):
        if first['args'][field] != second['args'][field]:
            raise ValueError('research_ranker_pair_budget_mismatch:' + field)


@torch.inference_mode()
def evaluate(args, out, device, status):
    data, _ = read_feature_cache(args.cache, getattr(args, 'execution_cache_sha256', None))
    cohort = data['manifest']['cohort']
    if cohort not in ('original_val', 'tune', 'verify', 'test'):
        raise ValueError('research_eval_wrong_cohort')
    models = [load_model(path, device) for path in (args.no_aug_checkpoint, args.aug_checkpoint)]
    validate_model_pair(models, data, args.seed)
    if cohort in ('verify', 'test'):
        if not args.frozen_selection:
            raise ValueError('research_ranker_selection_required')
        frozen = json.loads(Path(args.frozen_selection).read_text())
        if (frozen.get('schema') != 'nmr_research_20261022' or not frozen.get('frozen_utc')
                or frozen.get('selection_scope') != 'train_val_only'):
            raise ValueError('research_ranker_invalid_selection')
        config = dict(no_aug_checkpoint=str(Path(args.no_aug_checkpoint).resolve()),
                      aug_checkpoint=str(Path(args.aug_checkpoint).resolve()), seed=args.seed, cohort=cohort)
        if config not in frozen['allowed_ranker_configs']:
            raise ValueError('research_ranker_pair_not_frozen')
        if cohort == 'verify' and frozen.get('verify_eligibility') != 'eligible':
            raise ValueError('research_independent_verify_not_eligible')
        pinned = frozen['evaluation_inputs_sha256']
        for path, digest in input_fingerprints([args.no_aug_checkpoint, args.aug_checkpoint, __file__]).items():
            if pinned.get(path) != digest:
                raise ValueError('research_ranker_frozen_model_changed:' + path)
        if any(pinned.get(path) != digest for path, digest in data['manifest']['source_inputs_sha256'].items()):
            raise ValueError('research_ranker_unfrozen_cache_input')
    rows = []
    started = time.monotonic()
    durations = []
    for i, row in enumerate(data['rows']):
        row_started = time.monotonic()
        correction = []
        for model, scale, saved in models:
            correction.append(saved['alpha']*model.row_delta(row, scale, device)*scale[0])
        observation = row['query_observation']
        gamma = .75 if observation.get('windows') or observation.get('partial_peaks') else .25
        costs = row['coarse_scores'].to(device)+(1-gamma)*correction[0]+gamma*correction[1]
        rank, top = rank_shortlist(row, costs)
        durations.append(time.monotonic()-row_started)
        rows.append(dict(result_row(row, rank, top), gamma=gamma))
        if (i+1) % 100 == 0:
            status('evaluating', records=i+1, total=len(data['rows']))
    with (out/'predictions.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    atomic_json(out/'summary.json', dict(metrics=summarize_predictions(rows), cohort=cohort,
        query_sha256=data['manifest']['query_sha256'], candidate_order_sha256=data['manifest']['candidate_order_sha256'],
        elapsed_seconds=time.monotonic()-started, reranking_seconds_mean=float(np.mean(durations)),
        reranking_seconds_p95=float(np.quantile(durations, .95)),
        checkpoints=input_fingerprints([args.no_aug_checkpoint, args.aug_checkpoint]), external_api_calls=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('fit', 'evaluate'), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--train'); parser.add_argument('--cache')
    parser.add_argument('--no-aug-checkpoint'); parser.add_argument('--aug-checkpoint')
    parser.add_argument('--frozen-selection')
    parser.add_argument('--observation', action='store_true'); parser.add_argument('--augment', action='store_true')
    parser.add_argument('--seed', type=int, default=211)
    parser.add_argument('--epochs', type=int, default=64); parser.add_argument('--samples', type=int, default=8192)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--anchor', type=float, default=.1)
    parser.add_argument('--observation-regularization', type=float, default=.01)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 1 or args.batch_size < 1:
        raise ValueError('research_invalid_training_budget')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    status = lambda stage, **kw: atomic_json(out/'status.json', dict(stage=stage, **kw))
    paths = [__file__, Path(__file__).with_name('h_research_observation.py'),
             Path(__file__).with_name('h_research_contract.py')]
    paths += [args.train] if args.stage == 'fit' else [args.cache, args.no_aug_checkpoint, args.aug_checkpoint]
    cache_path = Path(args.train if args.stage == 'fit' else args.cache)
    paths += [cache_path.parent/'summary.json', cache_path.parent/'EXIT_CODE']
    pinned = input_fingerprints(paths)
    args.execution_cache_sha256 = pinned[str(cache_path.resolve())]
    try:
        torch.set_num_threads(4)
        atomic_json(out/'protocol.json', dict(args=vars(args), inputs_sha256=pinned))
        if args.stage == 'fit':
            fit(args, out, torch.device(args.device), status)
        else:
            evaluate(args, out, torch.device(args.device), status)
        if input_fingerprints(pinned) != pinned:
            raise ValueError('research_ranker_input_changed_during_run')
        status('complete'); (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed', error=repr(exc)); (out/'EXIT_CODE').write_text('1\n'); raise


if __name__ == '__main__':
    main()

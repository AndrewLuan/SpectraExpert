"""功能概述：为新轮R/O/F消融生成同一完整候选库上的观测和短名单缓存。

ResearchScorer精确复用上轮已选报告粗排，并提供独立的正观测召回通道；extract
以固定512容量生成单/双通道名单，缓存旧33匹配特征、预测支持及明确的观测语义，
供h_research_ranker联合训练观测模型。真结构只在候选及特征生成后用于监督和指标。
train允许在名单末尾附加正例，evaluation_pool_size严格阻止它进入评测；非连续
名单的池外名次采用h_research_contract定义。verify/test必须绑定新轮冻结输入。
main只写独占输出目录，核验始末SHA和真实EXIT，不覆盖旧bank/谱表/缓存，无外部API。
"""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import identity
from .evaluate import missing_queries
from .h_condinfer import condition_key, scoring_inputs
from .h_conditions import SOLVENT_DIM
from .h_global_bank import load_bank
from .h_global_extract import Scorer
from .h_global_features import FEATURE_NAMES, candidate_features
from .h_global_observation_study import artifact_inputs, corrections
from .h_global_protocol import actual_inputs
from .h_global_scoring import configs
from .h_labels import make_key
from .h_observation_study import stamp_queries
from .h_score import attach_observation
from .h_research_contract import atomic_json, check_locked, input_fingerprints, pool_background_rank
from .h_research_recall import combine_shortlists, directional_kernel_scores
from .h_research_observation import prepare_observation, training_line_variants, training_environment_variant


def content_sha(value):
    def convert(item):
        if hasattr(item, 'tolist'):
            return convert(item.tolist())
        if isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(convert(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def materialize_training_variant(query, variant, raw):
    """同一合成观测同时进入粗排、旧匹配特征和新O输入，原始raw保持只读。"""
    support = variant['observation']
    xs = np.asarray(support['xs'], dtype=float)
    weights = np.asarray(support['raw_weights'], dtype=float)
    cdf = np.cumsum(weights)/weights.sum()
    q = xs[np.searchsorted(cdf, (np.arange(128)+.5)/128).clip(max=len(xs)-1)]
    state = dict(query['observation'], windows=support['windows'], partial_peaks=support['partial_peaks'])
    if support['partial_peaks']:
        state['kind'] = 'partial_window' if support['windows'] else 'unknown'
    observed = dict(query, q=q, observed_xs=xs.tolist(), observed_weights=weights.tolist(), observation=state)
    rewritten_raw = raw
    if support['semantics'] == 'line_positions':
        original = raw.get(query['record_id'], {})
        record = dict(original, value=dict(original.get('value', {}), x=xs.tolist(), y=np.ones(len(xs)).tolist()))
        rewritten_raw = collections.ChainMap({query['record_id']: record}, raw)
    return observed, rewritten_raw


class ResearchScorer(Scorer):
    def __init__(self, bank, payload, widths, inputs, device, table, models, observation_config):
        choice = next(c for c in configs() if c['name'] == 'atom_count_1.0')
        super().__init__(bank, payload, widths, inputs, choice, device)
        self.environments = table['K'].to(device)
        self.models = models
        self.observation_config = observation_config
        if observation_config['capacity'] != 0:
            raise ValueError('research_incumbent_expected_zero_capacity')

    def score(self, query):
        coarse, baseline = super().score(query)
        delta, _ = corrections(self, query, self.environments, self.models, {})
        return coarse + self.observation_config['report'] * delta, baseline


def required_inputs(args):
    paths = actual_inputs(args.base, args.bank, args.widths, args.cohorts, args.extension)
    paths += artifact_inputs(args.artifact) + [Path(args.feature_choice)]
    paths += [args.fixed_identities, args.environment_mapping]
    paths += [Path(__file__).with_name(name) for name in (
        'h_research_extract.py', 'h_research_contract.py', 'h_research_recall.py',
        'h_research_observation.py', 'h_global_reporting.py', 'h_global_observation_study.py')]
    return paths


@torch.inference_mode()
def extract(args, out, inputs_sha, status):
    device = torch.device(args.device)
    bank = load_bank(args.bank, args.extension)
    payload = torch.load(Path(args.cohorts)/'cohorts.pt', map_location='cpu', weights_only=False)
    queries = bank['test'] if args.cohort == 'test' else payload['cohorts'][args.cohort]
    if not queries:
        raise ValueError('research_empty_cohort')
    if args.limit:
        queries = queries[:args.limit]
    train = args.cohort in ('train', 'train2048')
    indices = bank['single_indices']
    smiles = [bank['smiles'][i] for i in indices]
    if len(smiles) != 114982 or len(set(smiles)) != len(smiles):
        raise ValueError('research_candidate_library_changed')
    candidate_sha = hashlib.sha256('\n'.join(smiles).encode()).hexdigest()
    report_manifest = json.loads((Path(args.artifact)/'manifest.json').read_text())
    identities = json.loads(Path(args.fixed_identities).read_text())
    if identities['candidate_order_sha256'] != candidate_sha:
        raise ValueError('research_frozen_candidate_order_changed')
    if args.cohort == 'test' and (len(queries) != 1000 or not args.augment
            or stamp_queries(queries) != identities['frozen_test_query_sha256']):
        raise ValueError('research_frozen_test_query_changed')
    if report_manifest['candidate_order_sha256'] != candidate_sha:
        raise ValueError('research_reporting_candidate_order_changed')
    table = torch.load(Path(args.artifact)/'table.pt', map_location='cpu', weights_only=False)
    if not torch.allclose(table['H'], bank['physics'][indices, 0].cpu()*40, atol=.001, rtol=0):
        raise ValueError('research_candidate_H_changed')
    model_info = json.loads((Path(args.artifact)/'models.json').read_text())
    feature_choice = json.loads(Path(args.feature_choice).read_text())
    if feature_choice['selection_scope'] != 'train_val_only':
        raise ValueError('research_incumbent_choice_scope')
    widths = json.loads(Path(args.widths).read_text())['selected_ppm']
    scoring = scoring_inputs(Path(args.base))
    scorer = ResearchScorer(bank, payload, widths, scoring, device, table, model_info,
                            feature_choice['observation_config'])
    lookup = {s: i for i, s in enumerate(smiles)}
    candidate_groups = [identity(s)[1] for s in smiles] if train else None
    if train and not payload.get('training_excluded_groups'):
        raise ValueError('research_training_exclusion_contract_required')
    excluded_groups = set(payload.get('training_excluded_groups', []))
    if train and any(row['group'] in excluded_groups for row in queries):
        raise ValueError('research_training_query_overlaps_heldout_structure')
    rows = []
    raw, prior = scoring[-1], scoring[1]
    environment_mappings = json.loads(Path(args.environment_mapping).read_text())
    augmentation_audit = collections.Counter()
    start = time.monotonic()
    grouped = {}
    for order, row in enumerate(queries):
        key = condition_key(row, payload['metadata'], payload['key_mode'])
        grouped.setdefault(key, []).append((order, row))
    for key in sorted(grouped):
        if key not in bank['predictions']:
            raise ValueError('research_bank_missing_query_condition:' + str(key))
        group = grouped[key]
        kernel_started = time.monotonic()
        scorer.group(key, any(r['source'] == 'HMDB' for _, r in group))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        kernel_seconds = time.monotonic()-kernel_started
        for order, row in group:
            query = attach_observation(row, payload['annotations'].get(make_key(row), {}))
            query['_condition_known'] = bool(key[4+SOLVENT_DIM-1] > 0)
            variants = [('observed', query, raw, None)]
            if args.augment and (args.cohort != 'test' or order < 200):
                variants += [(name, value, raw, None) for name, value in missing_queries(query, '1H')]
            if train and args.augment:
                mapping_key = json.dumps(make_key(row), ensure_ascii=False, separators=(',', ':'))
                seed = int(hashlib.sha256(('observation_train_317'+mapping_key).encode()).hexdigest()[:8], 16)
                for name, result in (
                    ('fine_lines', training_line_variants(query, raw, seed)),
                    ('environment_dropout', training_environment_variant(query, raw,
                        environment_mappings.get('entries', {}).get(mapping_key), seed))):
                    augmentation_audit[name+':'+str(result['audit'].get('skipped') or 'generated')] += 1
                    for variant in result['variants']:
                        augmented, augmented_raw = materialize_training_variant(query, variant, raw)
                        variants.append((variant['name'], augmented, augmented_raw, variant['observation']))
            for variant, observed, current_raw, prepared_observation in variants:
                inference_started = time.monotonic()
                scorer.inputs = (*scoring[:-1], current_raw)
                coarse, baseline = scorer.score(observed)
                observation = prepared_observation or prepare_observation(observed, current_raw)
                if args.recall == 'dual':
                    positive = directional_kernel_scores(scorer.variable, observation)
                    pool = combine_shortlists(coarse, positive, pool_size=512, per_channel=256,
                                              expected_candidates=114982)
                else:
                    pool = torch.argsort(coarse, stable=True)[:512]
                evaluation_size = len(pool)
                if evaluation_size != 512 or len(pool.unique()) != 512:
                    raise ValueError('research_shortlist_not_exactly_512')
                # 候选集合在读取target前完成。训练补入正例不改变evaluation_size。
                target = lookup[row['smiles']]
                present = bool((pool == target).any())
                outside = pool_background_rank(coarse, pool, target)
                if train and not present:
                    pool = torch.cat((pool, pool.new_tensor([target])))
                features, old_support = candidate_features(
                    scorer.kernels['pred'][pool], scorer.tags[pool], scorer.counts[pool], coarse[pool],
                    observed, prior, current_raw)
                positions = (pool == target).nonzero().reshape(-1)
                record = dict(query_order=order, source=row['source'], record_id=row['record_id'],
                    smiles=row['smiles'], group=row['group'], variant=variant,
                    candidate_indices=pool.cpu(), evaluation_pool_size=evaluation_size,
                    features=features.cpu(), coarse_scores=coarse[pool].cpu(),
                    predicted_ppm=(scorer.kernels['pred'][pool]*10).cpu(), tags=scorer.tags[pool].cpu(),
                    H=scorer.counts[pool].cpu(), K=table['K'][pool.cpu()],
                    structural_trust=table['known'][pool.cpu()], observation_input=observation,
                    kind=old_support['kind'], positions=old_support['positions'], reported=old_support['reported'],
                    query_observation=observed['observation'],
                    mask_content_sha256=content_sha({field: observed.get(field) for field in
                        ('q', 'observation', 'measure', 'observed_xs', 'observed_weights')}),
                    target_position=int(positions[0]) if len(positions) else -1,
                    target_in_coarse_pool=present, outside_pool_rank=outside,
                    coarse_rank=int((coarse <= coarse[target]+1e-7).sum()),
                    baseline_rank=int((baseline <= baseline[target]+1e-7).sum()),
                    coarse_top10=torch.argsort(coarse, stable=True)[:10].cpu().tolist())
                record['matching_seconds'] = time.monotonic()-inference_started
                record['condition_kernel_seconds_per_base_query'] = kernel_seconds/len(group)
                if key in bank.get('candidate_quantile_sigma', {}):
                    record['per_quantile_sigma'] = bank['candidate_quantile_sigma'][key][indices][pool.cpu()].cpu()
                if train:
                    record['negative_exclusion_mask'] = torch.tensor(
                        [candidate_groups[i] == row['group'] or candidate_groups[i] in excluded_groups
                         for i in pool.cpu().tolist()])
                rows.append(record)
                if len(rows) % 25 == 0:
                    status('extracting', records=len(rows), base_queries=len(queries), seconds=time.monotonic()-start)
        scorer.kernels = scorer.tags = scorer.variable = None
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    rows.sort(key=lambda r: (r['query_order'], r['variant']))
    identity_sha = content_sha([{key: row[key] for key in
        ('query_order', 'variant', 'source', 'record_id', 'smiles', 'group')} for row in rows])
    mask_sha = content_sha([(tuple(row[key] for key in ('query_order', 'variant', 'source', 'record_id', 'smiles')),
                             row['mask_content_sha256']) for row in rows])
    variants_count = dict(collections.Counter(row['variant'] for row in rows))
    if args.cohort == 'test':
        if (variants_count != identities['variants'] or identity_sha != identities['identity_content_sha256']
                or mask_sha != identities['variant_mask_content_sha256']):
            raise ValueError('research_fixed_test_identity_or_mask_changed')
    manifest = dict(schema='nmr_research_20261022', cohort='train' if train else args.cohort,
        training_subset=args.cohort if train else None, feature_names=FEATURE_NAMES,
        query_sha256=stamp_queries(queries), base_queries=len(queries), records=len(rows), pool_size=512,
        recall=args.recall, candidate_count=len(smiles), candidate_order_sha256=candidate_sha,
        checkpoint_sha256=bank['checkpoint_sha256'], source_inputs_sha256=inputs_sha,
        appended_training_positives=sum(not r['target_in_coarse_pool'] for r in rows) if train else 0,
        variants=variants_count, identity_content_sha256=identity_sha, variant_mask_content_sha256=mask_sha,
        training_augmentation_audit=dict(augmentation_audit),
        extraction_seconds=time.monotonic()-start,
        matching_seconds_mean=float(np.mean([r['matching_seconds'] for r in rows])),
        matching_seconds_p95=float(np.quantile([r['matching_seconds'] for r in rows], .95)),
        recall512=float(np.mean([r['target_in_coarse_pool'] for r in rows])),
        shortlist_ranking='pool first; outside ordered by original coarse; pessimistic tied ranks',
        local_widths_ppm=widths, external_api_calls=0)
    if train:
        manifest['negative_excluded_structure_count'] = len(excluded_groups)
        manifest['negative_excluded_structure_sha256'] = hashlib.sha256(
            '\n'.join(sorted(excluded_groups)).encode()).hexdigest()
    torch.save(dict(rows=rows, smiles=smiles, manifest=manifest), out/'features.pt')
    manifest['feature_cache_sha256'] = input_fingerprints([out/'features.pt'])[str((out/'features.pt').resolve())]
    atomic_json(out/'summary.json', manifest)


def main():
    parser = argparse.ArgumentParser()
    for name in ('base', 'bank', 'widths', 'cohorts', 'artifact', 'feature-choice', 'cohort', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--fixed-identities', required=True)
    parser.add_argument('--environment-mapping', required=True)
    parser.add_argument('--extension')
    parser.add_argument('--recall', choices=('single', 'dual'), default='single')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--frozen-selection')
    args = parser.parse_args()
    if args.cohort not in ('train', 'train2048', 'original_val', 'tune', 'verify', 'test'):
        raise ValueError('research_invalid_extract_cohort')
    if args.cohort in ('test', 'verify') and args.limit:
        raise ValueError('research_cannot_limit_locked_cohort')
    pinned = input_fingerprints(required_inputs(args))
    config = dict(bank=str(Path(args.bank).resolve()), extension=args.extension, recall=args.recall,
                  augment=args.augment, pool_size=512, cohort=args.cohort)
    check_locked(args.frozen_selection, pinned, config, args.cohort)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    status = lambda stage, **kw: atomic_json(out/'status.json', dict(stage=stage, **kw))
    try:
        torch.set_num_threads(4)
        atomic_json(out/'protocol.json', dict(inputs_sha256=pinned, config=config))
        extract(args, out, pinned, status)
        if input_fingerprints(pinned) != pinned:
            raise ValueError('research_extract_input_changed_during_run')
        status('complete'); (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed', error=repr(exc)); (out/'EXIT_CODE').write_text('1\n'); raise


if __name__ == '__main__':
    main()

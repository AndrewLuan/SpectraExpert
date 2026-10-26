"""功能概述：执行新F的理论继续训练+实验适配配对实验，原数据/旧源码只读。

ResearchSpectra显式删除prepared中的旧embedding，只使用公开measure决定峰等权/核质量
口径；完整原子归属不升级为完整测量窗。谱级/归属权重和完整分布证据来自必填audit，
缺峰保留单向观测/显式窗口语义。forward_objective将模型component_sigma放入Gaussian
观测NLL，故sigma确实受监督；可信位点无序损失与冻结教师保持项分别记录。

select_training_rows强制train连接结构白名单和所有保留结构排除；验证仅取显式key，
禁止借理论重放暴露val/verify/test。make_exposure在每步保持实验:理论=1:1，理论三源
轮转等额，实验按连接结构均权；同seed所有physical/local共享同一身份曝光序列。
audit须给inputs_sha256、selected_training_groups、forbidden_groups、validation_keys；
quality_defaults按domain|source声明可信度，row_weights可逐key覆盖，未知归属默认0。

main使用load_checkpoint_model与真实末层GINE，保存初始/逐epoch/best/last及真实EXIT。
默认pilot为2epoch、每域4096次曝光、每域batch32、GINE lr2e-5/head lr1e-4；pilot不
计算val目标并固定保留最后epoch。完整训练显式--stage full才用声明val归档选择。
改变预算须由父流程在训练准入前固定，不读取任何test指标。
固定组来自既有Preserved实现；新增部分是末层GINE继续训练、source均衡与可学习误差。
无网络、无外部API；本文件不会同步文件、创建额外会话或自动启动其他研究job。
"""
import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_batch

from .data import identity, graph
from .h_conditions import local_environment
from .h_labels import make_key
from .h_metadata import metadata_view, RECOVERED_MODE
from .h_replaytrain import ReplaySpectra
from .h_reviewed_supervision import load_artifact, apply_review
from .h_site_quality import apply_quality
from .h_objective import site_loss
from .h_observation_study import uniform_peaks
from .h_research_forward import (load_checkpoint_model, strip_embedding_caches,
    configure_forward_training)
from .observation import observation_loss, observed_prediction as condition_prediction, KINDS
from .prepare import sha
from .train import collate, seed_worker


THEORY_SOURCES = ('QME14S', 'FG26', 'QM9-NMR')
LINE_MEASURES = {'peak_equal', 'peak_equal_intensity_unused', 'peak_height_not_integral'}
MASS_MEASURES = {'atom_equal', 'observed_assigned_atom_equal_partial_possible', 'integral_weighted'}


def row_key(row):
    return json.dumps(list(make_key(row)), separators=(',', ':'))


def exposure_key(row):
    """cache索引区分QME同key的不同观测；不改变任何记录的q或源身份。"""
    return json.dumps([*make_key(row), row.get('research_cache_row_index')], separators=(',', ':'))


def validate_weight(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('research_forward_invalid_' + name)
    return value


def select_training_rows(rows, audit):
    selected = set(audit['selected_training_groups'])
    forbidden = set(audit['forbidden_groups'])
    if not selected or len(selected) != len(audit['selected_training_groups']) or len(selected) > 8192:
        raise ValueError('research_forward_invalid_selected_groups')
    if selected & forbidden:
        raise ValueError('research_forward_selected_forbidden_overlap')
    validation = audit['validation_keys']
    if isinstance(validation, dict):
        if not {'original_val', 'tune'} <= set(validation) or set(validation) - {'original_val', 'tune', 'verify'}:
            raise ValueError('research_forward_unknown_validation_cohort')
        validation = validation['original_val'] + validation['tune']
    validation_keys = {tuple(key) for key in validation}
    if not validation_keys or len(validation_keys) != len(validation):
        raise ValueError('research_forward_invalid_validation_keys')
    defaults = audit.get('quality_defaults', {})
    overrides = audit.get('row_weights', {})
    exp, theory, val = [], [], []
    seen_experimental, selected_seen, validation_seen = set(), set(), set()
    theory_seen = {source: set() for source in THEORY_SOURCES}
    by_source_indices = audit.get('theory_indices_by_source')
    if by_source_indices is None or set(by_source_indices) != set(THEORY_SOURCES):
        raise ValueError('research_forward_audited_theory_indices_required')
    allowed_theory = {source: set(indices) for source, indices in by_source_indices.items()}
    if any(len(values) != len(by_source_indices[source]) for source, values in allowed_theory.items()):
        raise ValueError('research_forward_duplicate_theory_indices')
    allowed_indices = set.union(*allowed_theory.values())
    if any(type(index) is not int or index < 0 or index >= len(rows) for index in allowed_indices):
        raise ValueError('research_forward_theory_index_out_of_bounds')
    theory_key_counts = collections.Counter(make_key(rows[index]) for index in allowed_indices)
    for row_index, row in enumerate(rows):
        if row['nucleus'] != '1H':
            continue
        key = make_key(row)
        if row['domain'] == 'experimental' and (key in validation_keys or row['group'] in selected):
            if key in seen_experimental:
                raise ValueError('research_forward_duplicate_experimental_row_key')
            seen_experimental.add(key)
        if key in validation_keys:
            if row['domain'] != 'experimental' or row['split'] != 'val' or row['group'] not in forbidden:
                raise ValueError('research_forward_validation_boundary')
            val.append(dict(row))
            validation_seen.add(key)
            continue
        if row['split'] != 'train' or row['group'] in forbidden:
            continue
        if row['domain'] == 'experimental':
            if row['group'] not in selected:
                continue
            selected_seen.add(row['group'])
        elif row['domain'] == 'theory':
            if row['source'] not in THEORY_SOURCES or row_index not in allowed_theory[row['source']]:
                continue
        else:
            raise ValueError('research_forward_unknown_domain')
        _, computed_group, computed_split = identity(row['smiles'])
        if computed_group != row['group'] or computed_split != 'train':
            raise ValueError('research_forward_structure_split_mismatch')
        policy = dict(defaults.get(row['domain'] + '|' + row['source'], {}))
        policy.update(overrides.get(row_key(row), {}))
        if type(policy.get('full_distribution', False)) is not bool:
            raise ValueError('research_forward_full_distribution_must_be_boolean')
        rr = dict(row, research_cache_row_index=row_index,
            research_duplicate_annotation_key=(row['domain'] == 'theory' and theory_key_counts[key] > 1),
            research_spectral_weight=min(validate_weight(
            policy.get('spectral_weight', 1.), 'spectral_weight'),
            validate_weight(row.get('review_spectral_weight', 1.), 'review_spectral_weight')),
            research_assignment_weight=validate_weight(policy.get('assignment_weight', 0.), 'assignment_weight'),
            research_full_distribution=bool(policy.get('full_distribution', False)))
        if rr['research_full_distribution'] and not policy.get('full_distribution_evidence'):
            raise ValueError('research_forward_full_distribution_requires_evidence')
        if rr['research_full_distribution'] and row.get('observation', {}).get('kind') not in ('complete_atoms', 'window'):
            raise ValueError('research_forward_full_distribution_without_original_coverage_evidence')
        rr['research_assignment_evidence'] = policy.get('assignment_evidence')
        if rr['research_assignment_weight'] > 0 and not rr['research_assignment_evidence']:
            raise ValueError('research_forward_assignment_requires_evidence')
        if rr['research_duplicate_annotation_key'] and rr['research_assignment_weight'] > 0:
            raise ValueError('research_forward_ambiguous_theory_assignment_key')
        (exp if row['domain'] == 'experimental' else theory).append(rr)
        if row['domain'] == 'theory':
            theory_seen[row['source']].add(row_index)
    if selected_seen != selected or validation_seen != validation_keys:
        raise ValueError('research_forward_audit_rows_missing')
    if set(r['source'] for r in theory) != set(THEORY_SOURCES):
        raise ValueError('research_forward_three_theory_sources_required')
    if theory_seen != allowed_theory:
        raise ValueError('research_forward_theory_index_scope_mismatch')
    if not exp or not theory or not val:
        raise ValueError('research_forward_empty_partition')
    if {r['group'] for r in exp + theory} & {r['group'] for r in val}:
        raise ValueError('research_forward_train_validation_overlap')
    return exp, theory, val


def trusted_annotations(rows, annotations):
    """在无序合并之前过滤逐H可信mask；一个坏母节点不会关闭其他明确可信组。"""
    result, row_weights, node_weights = {}, {}, {}
    for row in rows:
        key = make_key(row)
        meta = annotations.get(key, {})
        safe = dict(meta, groups={})
        trusted = meta.get('trusted_site_mask', meta.get('site_trust', {}))
        trust_by_node = {}
        blocked = row.get('research_disable_assignment', False) or meta.get('trusted_site_H') == 0
        for parent, values in meta.get('groups', {}).items():
            entry = trusted.get(parent, trusted.get(str(parent))) if isinstance(trusted, dict) else None
            if entry is None:
                strength = row.get('research_assignment_weight', 0.) if row.get('research_assignment_evidence') else 0.
                entry = [strength] * len(values)
            elif isinstance(entry, (bool, int, float)):
                entry = [entry] * len(values)
            if len(entry) != len(values):
                raise ValueError('research_forward_trusted_site_mask_shape_mismatch')
            weights = [validate_weight(value, 'site_trust') for value in entry]
            selected = [(float(value), weight) for value, weight in zip(values, weights) if weight > 0 and not blocked]
            if selected:
                safe['groups'][parent] = [value for value, _ in selected]
                trust_by_node[int(parent)] = sum(weight for _, weight in selected) / len(selected)
        result[key] = safe
        if row.get('research_duplicate_annotation_key') and trust_by_node:
            raise ValueError('research_forward_ambiguous_theory_assignment_key')
        row_weights[key] = sum(trust_by_node.values()) / len(trust_by_node) if trust_by_node else 0.
        node_weights[key] = trust_by_node
    return result, row_weights, node_weights


class ResearchSpectra(ReplaySpectra):
    def __init__(self, rows, graphs, nucleus, metadata, *, annotations, **kwargs):
        safe, self.assignment_weights, self.assignment_node_weights = trusted_annotations(rows, annotations)
        super().__init__(rows, graphs, nucleus, metadata, annotations=safe, **kwargs)

    def __getitem__(self, index):
        row = self.rows[index]
        graph = strip_embedding_caches(super().__getitem__(index), clone=False)
        measure = row['measure']
        if measure not in LINE_MEASURES | MASS_MEASURES:
            raise ValueError('research_forward_unknown_measure:' + str(measure))
        # 原子完整性只供可信位点/分布监督，不作为仪器完整观测窗。
        if int(graph.coverage_kind.item()) == KINDS['complete_atoms']:
            graph.coverage_kind.fill_(KINDS['unknown'])
        observation = row.get('observation', {})
        if (observation.get('partial_peaks') or observation.get('kind') in ('window', 'partial_window')):
            # ConditionsSpectra的HMDB原始簇侧表不能把衍生部分谱遗漏的峰重新补回。
            graph.q = torch.as_tensor(np.asarray(row['q']).copy() / 10, dtype=torch.float32).reshape(1, -1)
            graph.supervised_q = graph.q.clone()
        if graph.q.shape != (1, 128) or not torch.isfinite(graph.q).all():
            raise ValueError('research_forward_invalid_quantile_target')
        graph.research_line_measure = torch.tensor([measure in LINE_MEASURES])
        graph.research_spectral_weight = torch.tensor([row.get('research_spectral_weight', 1.)], dtype=torch.float32)
        graph.research_assignment_weight = torch.tensor([self.assignment_weights[make_key(row)]], dtype=torch.float32)
        weights = self.assignment_node_weights[make_key(row)]
        # Replay的酰胺/理论轨道可能把多个母节点映射到同一proxy；只汇入保留下来的可信组。
        for owner in graph.group_owner.unique().tolist():
            members = [parent for parent, strength in weights.items() if int(graph.group_owner[parent]) == owner]
            if members:
                graph.group_site_weight[owner] *= sum(weights[parent] for parent in members) / len(members)
        graph.research_full_distribution = torch.tensor([row.get('research_full_distribution', False)])
        return graph


def grouped_site_rows(details, batch):
    owners = batch.group_owner + batch.ptr[batch.batch]
    proxy = SimpleNamespace(x=batch.x, site_targets=batch.group_targets, site_mask=batch.group_mask,
        site_expected=batch.group_expected, site_weights=batch.group_site_weight,
        batch=batch.batch, num_graphs=batch.num_graphs)
    return site_loss(dict(details, owners=owners[details['owners']]), proxy)


def gaussian_observed_nll(details, batch):
    """只对已观测位置算NLL；显式窗口按Gaussian质量条件化，未知不猜测频窗。"""
    values, groups = details['values'], details['groups']
    if not len(values):
        raise ValueError('research_forward_empty_H_distribution')
    sigmas = details['component_sigma']
    weights = details['weights'].clone()
    for graph_id in range(batch.num_graphs):
        take = groups == graph_id
        if not take.any():
            raise ValueError('research_forward_graph_without_H')
        if bool(batch.research_line_measure[graph_id]):
            # 相同预测位置只贡献一个峰质量，观测峰高不进入此计算。
            _, inverse, counts = values[take].detach().unique(return_inverse=True, return_counts=True)
            weights[take] = 1. / counts[inverse].to(weights.dtype)
    mu, _ = to_dense_batch(values, groups, batch_size=batch.num_graphs)
    width, _ = to_dense_batch(sigmas, groups, fill_value=1., batch_size=batch.num_graphs)
    mass, _ = to_dense_batch(weights, groups, batch_size=batch.num_graphs)
    mass = mass / mass.sum(1, keepdim=True)
    logmass = mass.clamp_min(1e-30).log().masked_fill(mass <= 0, -torch.inf)
    residual = (batch.q[:, :, None] - mu[:, None, :]) / width[:, None, :]
    density = -.5 * residual.square() - width[:, None, :].log() - .5 * math.log(2 * math.pi)
    nll = -torch.logsumexp(density + logmass[:, None, :], -1).mean(1)
    windows = batch.windows.reshape(-1, 8, 2)
    mask = batch.window_mask.reshape(-1, 8)

    def amount(bounds):
        low = (bounds[:, :1] - mu) / (width * math.sqrt(2))
        high = (bounds[:, 1:] - mu) / (width * math.sqrt(2))
        return ((high.erf() - low.erf()) * .5 * mass).sum(1)

    order = windows[:, :, 0].masked_fill(~mask, torch.inf).argsort(1)
    ordered = windows.gather(1, order[:, :, None].expand(-1, -1, 2))
    valid = mask.gather(1, order)
    current, current_valid = torch.zeros_like(ordered[:, 0]), torch.zeros_like(valid[:, 0])
    union = torch.zeros_like(nll)
    for position in range(8):
        item, active = ordered[:, position], valid[:, position]
        extend = active & current_valid & (item[:, 0] <= current[:, 1])
        union = union + amount(current) * (current_valid & ~extend)
        current = torch.stack([torch.where(extend, current[:, 0], item[:, 0]),
                               torch.where(extend, torch.maximum(current[:, 1], item[:, 1]), item[:, 1])], 1)
        current_valid = active
    union = union + amount(current) * current_valid
    conditional = torch.where(union >= 1e-8, nll + union.clamp_min(1e-12).log(), nll + 20.)
    return torch.where(mask.any(1), conditional, nll)


def forward_objective(details, batch, teacher_quantiles=None, anchor_weight=.1, sigma_weight=.005):
    predicted = details['quantiles']
    line = batch.research_line_measure.reshape(-1)
    observed_prediction = torch.where(line[:, None], uniform_peaks(predicted), predicted)
    presence = observation_loss(observed_prediction, batch)
    sites, eligible = grouped_site_rows(details, batch)
    sites = torch.where(eligible, sites, torch.zeros_like(sites))
    assignment = batch.research_assignment_weight.reshape(-1).to(predicted.dtype)
    spectral = batch.research_spectral_weight.reshape(-1).to(predicted.dtype)
    if not torch.all((assignment >= 0) & (assignment <= 1) & (spectral >= 0) & (spectral <= 1)):
        raise ValueError('research_forward_invalid_batch_trust')
    full = batch.research_full_distribution.reshape(-1).to(predicted.dtype)
    conditioned, _, _ = condition_prediction(observed_prediction, batch.windows, batch.window_mask)
    shape_prediction = torch.where(batch.window_mask.reshape(-1, 8).any(1)[:, None], conditioned, observed_prediction)
    shape = (shape_prediction - batch.supervised_q).abs().mean(1) * full * spectral
    nll = gaussian_observed_nll(details, batch)
    anchor = (predicted - teacher_quantiles.detach()).abs().mean(1) if teacher_quantiles is not None else predicted.sum(1) * 0
    per_graph = spectral * (presence + sigma_weight * nll) + .75 * assignment * sites + .5 * shape + anchor_weight * anchor
    return dict(loss=per_graph.mean(), per_graph=per_graph, presence=(presence * spectral).mean(),
        site=(sites * assignment).mean(), shape=shape.mean(), nll=(nll * spectral).mean(), anchor=anchor.mean())


def make_exposure(exp, theory, samples, seed, epoch):
    """固定身份序列，不读取误差；理论三源总次数差至多1，域曝光严格相等。"""
    if samples <= 0:
        raise ValueError('research_forward_positive_exposure_required')
    rng = np.random.default_rng(seed + 100003 * epoch)
    def buckets(rows, source=None):
        result = collections.defaultdict(list)
        for index, row in enumerate(rows):
            if source is None or row['source'] == source:
                result[row['group']].append(index)
        return [(key, sorted(indices, key=lambda index: exposure_key(rows[index]))) for key, indices in sorted(result.items())]
    experimental = buckets(exp)
    by_source = {source: buckets(theory, source) for source in THEORY_SOURCES}
    if not experimental or any(not values for values in by_source.values()):
        raise ValueError('research_forward_missing_source_bucket')
    sequence_exp, sequence_theory = [], []
    source_order = list(rng.permutation(THEORY_SOURCES))
    for position in range(samples):
        for target, collection in ((sequence_exp, experimental),
                (sequence_theory, by_source[source_order[position % 3]])):
            _, indices = collection[int(rng.integers(len(collection)))]
            target.append(indices[int(rng.integers(len(indices)))])
    return sequence_exp, sequence_theory


def frozen_core_check(initial, current, prefixes):
    if initial.keys() != current.keys():
        raise ValueError('research_forward_state_keys_changed')
    count = 0
    for name, tensor in initial.items():
        if not name.startswith(prefixes):
            if not torch.equal(tensor, current[name].detach().cpu()):
                raise ValueError('research_forward_frozen_tensor_changed:' + name)
            count += 1
    return count


@torch.inference_mode()
def validate_forward(model, rows, graphs, metadata, labels, device, batch_size=64):
    model.eval()
    by_source = {}
    for source in sorted({row['source'] for row in rows}):
        selected = [dict(row, research_assignment_weight=0., research_spectral_weight=1.,
                         research_full_distribution=False, research_disable_assignment=True) for row in rows if row['source'] == source]
        loader = DataLoader(ResearchSpectra(selected, graphs, '1H', metadata, annotations=labels,
            grouped=True), batch_size=batch_size, collate_fn=collate)
        total, count = 0., 0
        for batch in loader:
            batch = batch.to(device)
            parts = forward_objective(model(batch, details=True), batch, anchor_weight=0.)
            total += float(parts['loss']) * batch.num_graphs
            count += batch.num_graphs
        by_source[source] = dict(records=count, spectral_objective=total / count)
    return dict(by_source=by_source,
        macro_objective=float(np.mean([row['spectral_objective'] for row in by_source.values()])),
        selection='explicit validation keys only; spectral objective, no assignment-based selection')


def main(device=None):
    parser = argparse.ArgumentParser()
    for name in ('prepared', 'metadata', 'labels', 'init', 'audit', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--mode', choices=('physical', 'local'), required=True)
    parser.add_argument('--stage', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--seed', type=int, choices=(193, 211), required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--domain-batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr-gine', type=float, default=2e-5)
    parser.add_argument('--lr-head', type=float, default=1e-4)
    parser.add_argument('--anchor-weight', type=float, default=.1)
    parser.add_argument('--sigma-weight', type=float, default=.005)
    parser.add_argument('--review-artifact')
    parser.add_argument('--theory-sidecar')
    parser.add_argument('--feature-counts')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def status(stage, **fields):
        temporary = out / 'status.tmp'
        temporary.write_text(json.dumps(dict(stage=stage, seconds=time.monotonic() - started, **fields)))
        temporary.replace(out / 'status.json')
    try:
        if args.epochs < 1 or args.samples <= 0 or args.domain_batch <= 0 or args.samples % args.domain_batch:
            raise ValueError('research_forward_complete_equal_domain_batches_required')
        if bool(args.theory_sidecar) != bool(args.feature_counts):
            raise ValueError('research_forward_sidecar_and_feature_counts_required_together')
        if (any(not math.isfinite(value) or value <= 0 for value in (args.lr_gine, args.lr_head, args.sigma_weight))
                or not math.isfinite(args.anchor_weight) or args.anchor_weight < 0 or args.workers < 0):
            raise ValueError('research_forward_invalid_optimizer_configuration')
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('high')
        device = torch.device('cuda' if device is None else device)
        audit = json.loads(Path(args.audit).read_text())
        if audit.get('review_audit') and not args.review_artifact:
            raise ValueError('research_forward_reviewed_cohort_requires_review_artifact')
        prepared = Path(args.prepared)
        required = [prepared / 'cache.pt', prepared / 'manifest.json', Path(args.metadata), Path(args.labels), Path(args.init)]
        if args.theory_sidecar:
            required.extend([Path(args.theory_sidecar), Path(args.feature_counts)])
        if args.review_artifact:
            required.extend([Path(args.review_artifact) / 'manifest.json', Path(args.review_artifact) / 'artifact.pt'])
        expected = audit.get('inputs_sha256', audit.get('input_sha256'))
        if not isinstance(expected, dict) or (audit.get('inputs_sha256') is not None and
                audit.get('input_sha256') is not None and audit['inputs_sha256'] != audit['input_sha256']):
            raise ValueError('research_forward_invalid_input_SHA_contract')
        inputs = {}
        status('checking_inputs')
        for path in required:
            key = str(path.resolve())
            digest = sha(path)
            if expected.get(key) != digest:
                raise ValueError('research_forward_input_not_audited:' + key)
            inputs[key] = digest
        inputs[str(Path(args.audit).resolve())] = sha(args.audit)
        saved = torch.load(args.init, map_location='cpu', weights_only=False)
        registry = Path(saved['args']['equivalence_registry_path']).resolve()
        if expected.get(str(registry)) != sha(registry):
            raise ValueError('research_forward_registry_not_audited')
        inputs[str(registry)] = sha(registry)
        dependencies = ('h_research_forward_train.py', 'h_research_forward.py', 'h_preserved_group_model.py',
            'h_equivalence_model.py', 'h_solvent_message.py', 'h_deep_context.py', 'h_response.py',
            'h_condmodel.py', 'h_conditions.py', 'h_acyl.py', 'emission_model.py', 'model.py', 'data.py',
            'h_replaytrain.py', 'h_objective.py', 'observation.py', 'h_reviewed_supervision.py',
            'h_site_quality.py', 'h_symmetry_supervision.py', 'h_metadata.py', 'train.py',
            'h_amide_audit.py', 'h_observation_study.py', 'h_labels.py', 'h_hydrogen_order.py',
            'h_observed.py', 'h_count_sidecars.py', 'prepare.py')
        source_sha256 = {str(Path(__file__).with_name(name).resolve()): sha(Path(__file__).with_name(name))
                        for name in dependencies}
        inputs.update(source_sha256)
        cache = torch.load(prepared / 'cache.pt', map_location='cpu', weights_only=False, mmap=True)
        manifest = json.loads((prepared / 'manifest.json').read_text())
        if saved.get('prepared_sha256') != manifest['cache_sha256']:
            raise ValueError('research_forward_prepared_checkpoint_mismatch')
        rows = list(cache['rows'])
        labels = copy.deepcopy(torch.load(args.labels, map_location='cpu', weights_only=False))
        metadata = metadata_view(torch.load(args.metadata, map_location='cpu', weights_only=False), RECOVERED_MODE)
        review_audit = None
        if args.review_artifact:
            artifact, _ = load_artifact(args.review_artifact)
            exp_all = [row for row in rows if row['domain'] == 'experimental' and row['split'] == 'train']
            corrected, review_audit = apply_review(labels, exp_all, artifact, 'domain')
            corrected = {make_key(row): row for row in corrected}
            rows = [corrected.get(make_key(row), row) for row in rows]
        exp_all = [row for row in rows if row['domain'] == 'experimental' and row['split'] == 'train']
        quality_audit = apply_quality(labels, exp_all, 'methyl_record_mask')
        added = 0
        if args.theory_sidecar:
            from .h_count_sidecars import augment_theory_rows
            theory = [row for row in rows if row['domain'] == 'theory' and row['split'] == 'train']
            expanded, added = augment_theory_rows(theory,
                torch.load(args.theory_sidecar, map_location='cpu', weights_only=False),
                torch.load(args.feature_counts, map_location='cpu', weights_only=False))
            existing = {make_key(row) for row in rows}
            rows.extend(row for row in expanded if make_key(row) not in existing)
        exp, theory, val = select_training_rows(rows, audit)
        graphs = dict(cache['graphs'])
        for smiles in sorted({row['smiles'] for row in exp + theory + val} - set(graphs)):
            item = graph(smiles)
            item.local_environment = torch.tensor(local_environment(smiles))
            graphs[smiles] = item
        model = load_checkpoint_model(args.init, device=device, equivalence_mode=args.mode, init_seed=args.seed, trainable=True)
        teacher = load_checkpoint_model(args.init, device=device, equivalence_mode=args.mode, init_seed=args.seed)
        teacher.eval()
        if any(not torch.equal(tensor, teacher.state_dict()[name]) for name, tensor in model.state_dict().items()):
            raise ValueError('research_forward_teacher_initialization_mismatch')
        initial = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        prefixes, names = configure_forward_training(model)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        parameter_groups = []
        for prefix, lr in ((prefixes[:2], args.lr_gine), (prefixes[2:], args.lr_head)):
            group = [parameter for name, parameter in model.named_parameters() if name.startswith(prefix) and parameter.requires_grad]
            parameter_groups.append(dict(params=group, lr=lr))
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
        datasets = [ResearchSpectra(partition, graphs, '1H', metadata, annotations=labels,
            grouped=True, theory_orbits=True) for partition in (exp, theory)]
        config = dict(args=vars(args), model_args=model.research_config, audit_sha256=sha(args.audit),
            init_sha256=sha(args.init), input_sha256=inputs, train_prefixes=prefixes,
            trainable_parameter_names=names, trainable_parameters=sum(parameter.numel() for parameter in parameters),
            experimental_records=len(exp), experimental_unique_groups=len({row['group'] for row in exp}),
            theory_sources=dict(collections.Counter(row['source'] for row in theory)), added_sidecar_rows=added,
            val_records=len(val), val_key_sha256=hashlib.sha256(json.dumps(sorted(map(row_key, val))).encode()).hexdigest(),
            assignment_review=review_audit, methyl_quality=quality_audit, source_sha256=source_sha256,
            exposure='equal domain samples; equal three-source rotation; group-uniform within domain/source',
            selection=('fixed final pilot epoch; no validation objective computed' if args.stage == 'pilot'
                       else 'explicit val spectral objective incl epoch0; downstream retrieval chooses only train/val'))
        (out / 'config.json').write_text(json.dumps(config, indent=2))
        exposure = hashlib.sha256()
        seen_experimental_groups, seen_theory_groups = set(), set()
        actual_theory_sources = collections.Counter()
        best, best_epoch = float('inf'), None
        gradient_seen = dict(gine=False, bond=False, sigma=False)
        def checkpoint(name, epoch, metrics):
            temporary = out / (name + '.tmp')
            torch.save(dict(state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                args=model.research_config, epoch=epoch, validation=metrics,
                baseline_sha256=saved.get('baseline_sha256'), prepared_sha256=saved['prepared_sha256'],
                research_training=dict(scope='last_original_GINE_and_group_head', mode=args.mode,
                    exposure_sha256=exposure.hexdigest(), gradient_seen=gradient_seen)), temporary)
            temporary.replace(out / name)
        for epoch in range(args.epochs + 1):
            totals, steps = np.zeros(6), 0
            training_seconds = 0.
            if epoch:
                sequence = make_exposure(exp, theory, args.samples, args.seed, epoch)
                for domain, indices in zip(('experimental', 'theory'), sequence):
                    exposure.update(json.dumps([domain, epoch, [exposure_key((exp if domain == 'experimental' else theory)[i])
                        for i in indices]], separators=(',', ':')).encode())
                seen_experimental_groups.update(exp[i]['group'] for i in sequence[0])
                seen_theory_groups.update(theory[i]['group'] for i in sequence[1])
                actual_theory_sources.update(theory[i]['source'] for i in sequence[1])
                loaders = [DataLoader(dataset, batch_size=args.domain_batch, sampler=indices,
                    num_workers=args.workers, collate_fn=collate, worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(args.seed + epoch))
                    for dataset, indices in zip(datasets, sequence)]
                model.train()
                status('training', epoch=epoch)
                epoch_started = time.monotonic()
                for left, right in zip(*loaders):
                    optimizer.zero_grad(set_to_none=True)
                    batch_parts = []
                    for batch in (left, right):
                        batch = batch.to(device)
                        details = model(batch, details=True)
                        with torch.no_grad():
                            reference = teacher(batch)
                        batch_parts.append(forward_objective(details, batch, reference, args.anchor_weight, args.sigma_weight))
                    loss = .5 * (batch_parts[0]['loss'] + batch_parts[1]['loss'])
                    if not torch.isfinite(loss):
                        raise ValueError('research_forward_nonfinite_loss')
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.abs().sum()) > 0:
                            if name.startswith(prefixes[0]): gradient_seen['gine'] = True
                            if name.startswith(prefixes[1]): gradient_seen['bond'] = True
                            if name.startswith('research_sigma_head.'): gradient_seen['sigma'] = True
                    torch.nn.utils.clip_grad_norm_(parameters, 2.)
                    optimizer.step()
                    totals += [.5 * sum(float(part[key].detach()) for part in batch_parts)
                               for key in ('loss', 'presence', 'site', 'shape', 'nll', 'anchor')]
                    steps += 1
                    if steps % 25 == 0:
                        status('training', epoch=epoch, steps=steps, losses=(totals / steps).tolist())
                if steps != args.samples // args.domain_batch:
                    raise ValueError('research_forward_domain_exposure_mismatch')
                training_seconds = time.monotonic() - epoch_started
            status('pilot_train_only_checkpoint' if args.stage == 'pilot' else 'validation', epoch=epoch)
            metrics = (dict(pilot_train_only=True, validation_objectives_computed=False,
                selection='fixed final pilot epoch; throughput and train loss only') if args.stage == 'pilot'
                else validate_forward(model, val, graphs, metadata, labels, device))
            if epoch == 0:
                checkpoint('initial.pt', epoch, metrics)
            if args.stage == 'pilot' or metrics['macro_objective'] < best:
                best, best_epoch = (None if args.stage == 'pilot' else metrics['macro_objective']), epoch
                checkpoint('best.pt', epoch, metrics)
            checkpoint('epoch_' + str(epoch) + '.pt', epoch, metrics)
            checkpoint('last.pt', epoch, metrics)
            row = dict(epoch=epoch, steps=steps, losses=(totals / max(1, steps)).tolist(),
                validation=metrics, gradient_seen=gradient_seen, seconds=time.monotonic() - started,
                training_seconds=training_seconds,
                graphs_per_training_second=(2 * args.samples / training_seconds if training_seconds else None),
                unique_experimental_groups_seen=len(seen_experimental_groups),
                unique_theory_groups_seen=len(seen_theory_groups), actual_theory_source_exposures=dict(actual_theory_sources),
                exposure_sha256=exposure.hexdigest())
            with (out / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        if not all(gradient_seen.values()):
            raise ValueError('research_forward_GINE_or_sigma_never_received_gradient')
        frozen = frozen_core_check(initial, model.state_dict(), prefixes)
        for path, digest in inputs.items():
            if sha(path) != digest:
                raise ValueError('research_forward_input_changed:' + path)
        (out / 'summary.json').write_text(json.dumps(dict(best_epoch=best_epoch, best_macro=best,
            frozen_core_exact=True, frozen_tensors_checked=frozen, gradient_seen=gradient_seen,
            exposure_sha256=exposure.hexdigest(), seconds=time.monotonic() - started,
            unique_experimental_groups_seen=len(seen_experimental_groups),
            unique_theory_groups_seen=len(seen_theory_groups), actual_theory_source_exposures=dict(actual_theory_sources),
            checkpoint_sha256=sha(out / 'best.pt'), external_api_calls=0, test_used=False), indent=2))
        status('complete')
        (out / 'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed', error=repr(exc))
        (out / 'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

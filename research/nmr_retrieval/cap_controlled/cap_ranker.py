"""功能概述：本轮C5/C10受控续训与推理共享的有界残差模型，不修改历史科学模块。
configure_runtime显式加载已绑定的科学checkout和固定中心模块；CapRanker保留历史
state_dict键及33+O3固定中心，只将两处共同使用的row_delta变为C*tanh(logits/C)。
strict_negative_mask核历史external/samegroup mask后，增加internal候选排除；
listwise_loss仅在训练中单独允许target，不向推理池注入候选。tensor_hashes/frozen_digest
用于初始化继承和冻结审计。load_checkpoint只接受本轮新schema、固定final64/alpha1及
producer/中心文件闭包；score_pair沿用既有gamma与两个role的成本组合，标签不参与。
输入为train缓存行或调用方已切为自然512的无身份行；输出为残差、成本或加载三元组。
本文件不启动训练、不选择checkpoint、不读取test、不调用外部服务。train_cap_ranker和
evaluate_cap_ranker通过此模块共享cap数学与加载契约，原固定中心源保持只读。
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

SCHEMA = 'nmr_cap_continuation_20261024'
FIXED_SHA256 = '3b8a6510198e40a63da542705987960f5a75feb023445510dc06ffb80ce19d20'
BUDGET = dict(epochs=64, samples=8192, batch_size=32, lr=.001, weight_decay=.001,
              anchor=.1, temperature=.5, clip_grad_norm=2., draws_per_fit=524288,
              optimizer_steps=16384)
MASK_POLICY = 'external_heldout_OR_ranker_internal_OR_same_query_group; target_loss_only'
CENTER_NAMES = {'fixed_x_center', 'fixed_o_center'}
_runtime = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprints(paths):
    return {str(Path(p).resolve()): sha256(p) for p in sorted(set(map(str, paths)))}


def configure_runtime(source, fixed_module):
    """返回显式runtime；固定模块及所有科学源由调用入口纳入始末SHA闭包。"""
    global _runtime
    source, path = Path(source).resolve(), Path(fixed_module).resolve()
    require((source / 'nmr_transfer/h_pool_ranker.py').is_file(), 'scientific_checkout_missing')
    require(sha256(path) == FIXED_SHA256, 'fixed_module_SHA_mismatch')
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('_cap_fixed_center_runtime', path)
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    require(Path(fixed.base.__file__).resolve().parent.parent == source, 'wrong_scientific_checkout')
    paths = sorted(set([str(Path(__file__).resolve()), str(path)] +
                       [str(Path(p).resolve()) for p in fixed.source_paths()]))
    _runtime = SimpleNamespace(fixed=fixed, base=fixed.base, source=source, fixed_module=path,
        read_feature_cache=fixed.read_feature_cache, select_rows=fixed.select_rows,
        internal_group=fixed.internal_group, source_paths=lambda: [Path(p) for p in paths])
    return _runtime


def runtime_or_default(runtime=None):
    result = runtime if runtime is not None else _runtime
    require(result is not None, 'configure_runtime_required')
    return result


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(json.dumps([str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def tensor_hashes(model, scale=None):
    result = {k: tensor_hash(v) for k, v in model.state_dict().items()}
    if scale is not None:
        result['__scale__'] = tensor_hash(scale)
    return result


def frozen_digest(model, scale):
    hashes = tensor_hashes(model, scale)
    frozen = {k: v for k, v in hashes.items() if not k.startswith('network.')}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class CapRanker(nn.Module):
    def __init__(self, initial_sigmas, xcenter, ocenter, cap, runtime=None):
        super().__init__()
        require(type(cap) in (int, float) and cap in (5, 10), 'registered_cap_required')
        native = runtime_or_default(runtime).fixed.FixedCenterRanker(initial_sigmas, xcenter, ocenter)
        self.observation_model = native.observation_model
        self.network = native.network
        self.register_buffer('fixed_x_center', native.fixed_x_center.detach().clone())
        self.register_buffer('fixed_o_center', native.fixed_o_center.detach().clone())
        self.delta_cap = int(cap)
        for p in self.observation_model.parameters():
            p.requires_grad_(False)

    def row_delta(self, row, scale, device):
        x = row['features'].to(device)
        z = ((x - self.fixed_x_center) / scale).clamp(-20, 20)
        sigma = row.get('per_quantile_sigma')
        extra = self.observation_model(row['predicted_ppm'].to(device), row['tags'].to(device).long(),
            row['observation_input'], H=row['H'].to(device), K=row['K'].to(device),
            per_quantile_sigma=None if sigma is None else sigma.to(device))
        extra = (extra - self.fixed_o_center).clamp(-20, 20)
        logits = self.network(torch.cat((z, extra), 1))[:, int(row['kind'])]
        return self.delta_cap * torch.tanh(logits / self.delta_cap)


def strict_negative_mask(row, qualification, runtime=None):
    """返回训练用排除mask；不改缓存，不把target当负例，不改变自然候选集合。"""
    rt = runtime_or_default(runtime)
    indices = row['candidate_indices'].cpu()
    mask = row['negative_exclusion_mask'].cpu()
    require(indices.dtype in (torch.int32, torch.int64) and indices.ndim == 1 and
            len(indices) in (512, 513) and len(indices.unique()) == len(indices), 'strict_candidate_indices')
    require(mask.dtype == torch.bool and mask.shape == indices.shape, 'strict_negative_mask_shape')
    groups, heldout = qualification['groups'], qualification['heldout']
    require(bool(((indices >= 0) & (indices < len(groups))).all()), 'strict_candidate_index_range')
    group = row['group']
    require(group not in heldout and not rt.internal_group(group), 'fit_query_not_eligible')
    expected = torch.tensor([groups[int(i)] == group or groups[int(i)] in heldout for i in indices])
    require(torch.equal(mask, expected), 'historical_negative_mask_semantics_mismatch')
    strict = mask | qualification['internal'][indices].cpu()
    target = row['target_position']
    require(type(target) is int and 0 <= target < len(indices), 'training_target_position')
    require(groups[int(indices[target])] == group, 'training_target_group_mismatch')
    strict = strict.clone()
    strict[target] = True
    require((~strict).any(), 'no_eligible_training_negatives')
    return strict


def listwise_loss(row, delta, scale, excluded, anchor=.1):
    require(excluded.dtype == torch.bool and excluded.shape == delta.shape, 'loss_mask_shape')
    target = int(row['target_position'])
    excluded = excluded.to(delta.device).clone()
    excluded[target] = True
    require((~excluded).any(), 'no_eligible_training_negatives')
    allowed = ~excluded
    allowed[target] = True
    costs = row['coarse_scores'].to(delta.device) / scale[0] + delta
    return (torch.logsumexp(-costs[allowed] / .5, 0) + costs[target] / .5 +
            anchor * delta[allowed].square().mean())


def validate_saved(saved, expected_cap=None, require_final=True):
    require(saved.get('schema') == SCHEMA and saved.get('observation') is True, 'cap_checkpoint_schema')
    cap = saved.get('delta_cap')
    require(type(cap) is int and cap in (5, 10), 'cap_checkpoint_cap')
    require(expected_cap is None or cap == expected_cap, 'cap_checkpoint_expected_cap')
    require(saved.get('budget') == BUDGET and saved.get('negative_mask_policy') == MASK_POLICY,
            'cap_checkpoint_training_contract')
    require(saved.get('seed') in (193, 211) and saved.get('role') in ('no_aug', 'aug'), 'cap_checkpoint_role')
    args = saved.get('args', {})
    require(args.get('seed') == saved['seed'] and args.get('augment') is (saved['role'] == 'aug') and
            args.get('cap') == cap and args.get('training_pool') == 'mixed' and args.get('loss') == 'listwise',
            'cap_checkpoint_args_role')
    require(saved.get('alpha') == 1 and type(saved.get('alpha')) is not bool, 'cap_fixed_alpha')
    if require_final:
        require(type(saved.get('epoch')) is int and saved['epoch'] == 64 and
                saved.get('draws') == BUDGET['draws_per_fit'] and
                saved.get('optimizer_steps') == BUDGET['optimizer_steps'], 'cap_final_full_budget')
    require(saved.get('strict_negative_qualification_sha256') and saved.get('parent_checkpoint_sha256') and
            saved.get('center_estimate_sha256') and saved.get('center_rows_sha256'), 'cap_provenance_required')


def load_checkpoint(path, device, expected_cap=None, require_final=True, runtime=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    validate_saved(saved, expected_cap, require_final)
    rt = runtime_or_default(runtime)
    sources = saved.get('ranker_source_sha256', {})
    require(all(str(Path(p).resolve()) in sources for p in rt.source_paths()) and
            fingerprints(sources) == sources, 'cap_source_binding')
    state = saved['state_dict']
    require(all(torch.isfinite(v).all() for v in state.values()), 'cap_nonfinite_state')
    scale = saved['scale']
    require(scale.shape == (33,) and torch.isfinite(scale).all() and (scale > 0).all(), 'cap_scale')
    model = CapRanker(saved['initial_sigmas'], state['fixed_x_center'], state['fixed_o_center'],
                      saved['delta_cap'], rt).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    parent = Path(path).resolve().parent
    center_path, rows_path = parent / 'centers.json', parent / 'center_rows.jsonl'
    require(sha256(center_path) == saved['center_estimate_sha256'] and
            sha256(rows_path) == saved['center_rows_sha256'], 'cap_center_artifact_SHA')
    estimate = json.loads(center_path.read_text())
    require(estimate['row_audit_sha256'] == saved['center_rows_sha256'] and
            torch.equal(model.fixed_x_center.cpu(), torch.tensor(estimate['center33'], dtype=torch.float32)) and
            torch.equal(model.fixed_o_center.cpu(), torch.tensor(estimate['O_center3'], dtype=torch.float32)),
            'cap_center_buffer_binding')
    require(frozen_digest(model, scale) == saved['frozen_tensor_sha256'], 'cap_frozen_tensor_binding')
    if require_final:
        producer = json.loads((parent / 'summary.json').read_text())
        require((parent / 'EXIT_CODE').read_text().strip() == '0', 'cap_producer_not_exit0')
        validate_saved(producer, expected_cap, True)
        require(producer['final_checkpoint_sha256'] == sha256(path) and
                all(producer.get(k) == v for k, v in saved.items()
                    if k not in ('state_dict', 'scale', 'initial_sigmas')) and
                producer['source_inputs_start_sha256'] == saved['input_sha256'] == producer['source_inputs_end_sha256'],
                'cap_producer_checkpoint_binding')
    return model, scale.to(device), saved


def score_pair(models, natural_row, device):
    require(len(models) == 2, 'score_two_roles_required')
    require(all(len(natural_row[k]) == 512 for k in ('features', 'coarse_scores', 'predicted_ppm', 'tags', 'H', 'K')),
            'score_natural512_required')
    require(not {'source', 'group', 'record_id', 'target_position', 'negative_exclusion_mask'} & natural_row.keys(),
            'score_label_or_identity_input')
    require(models[0][2]['delta_cap'] == models[1][2]['delta_cap'], 'score_cap_pair_mismatch')
    obs = natural_row['query_observation']
    require(set(obs) <= {'windows', 'partial_peaks'}, 'score_public_observation_only')
    gamma = .75 if obs.get('windows') or obs.get('partial_peaks') else .25
    correction = [saved['alpha'] * model.row_delta(natural_row, scale, device) * scale[0]
                  for model, scale, saved in models]
    costs = natural_row['coarse_scores'].to(device) + (1 - gamma) * correction[0] + gamma * correction[1]
    return costs, gamma

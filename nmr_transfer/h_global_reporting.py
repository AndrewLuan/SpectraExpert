"""功能概述：以结构 H 数和已核验等价环境数建模原子标签的报告数量。

structure_summary / reporting_table 从 canonical 分子图与冻结 registry 校验 H 质量，
导出总 H、物理环境 K、组大小及非交换 H/K。groups 的 complete 字段仅表示母节点
是否整组归入，不能解释为实验测量完整。未知 registry 项保留 K 未知，不由预测位移
小数或观测峰数生成物理环境。此侧表不修改 bank.physics、候选顺序或谱表。

training_records 只接受 experimental train 行，从已审核原子标签读取实际报告数，
跳过被暂缓的谱、空归属和显式频窗。fit_reporting 以每个来源、每个连接结构均权的
EM 拟合 P(k|H,K)=w_atom P_old(k|H)+sum_r w_r Binomial(k|K,r)，并向旧原子报告
过程收缩。代表氢和逐 H 标签是潜在报告过程，不根据实验位移硬分组或造出缺峰。

reporting_nll / reporting_delta 在候选自己的 H/K 上评分，k 只取查询报告标签数。
未知 K 精确调用旧 count_nll 回退；显式窗口要求调用方使用原窗内观测路径。完整性
标志、真结构索引、母节点归属不进入候选分数。模块不主动选择设备，无文件/网络副作用，供整体
train/val 消融独立调用；根研究流程负责冻结拟合产物后再开放核验与测试。
"""
import collections
import math

import numpy as np
import torch
from rdkit import Chem

from .h_labels import make_key
from .h_observed import count_nll


REPRESENTATIVE_RATES = (.1, .25, .5, .75, 1.)
EXCHANGEABLE_ATOMIC_NUMBERS = (7, 8, 16)


def structure_summary(smiles, entry=None):
    """校验已有结构分区并返回 H/K；不执行新的等价性推断。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('reporting_invalid_smiles')
    full = Chem.AddHs(mol)
    expected = {}
    exchangeable = {}
    for atom in full.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        parent = atom.GetIdx()
        expected[parent] = sum(a.GetAtomicNum() == 1 for a in atom.GetNeighbors())
        exchangeable[parent] = atom.GetAtomicNum() in EXCHANGEABLE_ATOMIC_NUMBERS
    h = sum(a.GetAtomicNum() == 1 for a in full.GetAtoms())
    h_fixed = sum(n for i, n in expected.items() if not exchangeable[i])
    result = dict(H=h, K=None, group_sizes=[], nonexchange_H=h_fixed,
                  nonexchange_K=None, nonexchange_group_sizes=[], known=False,
                  partition_evidence='unknown')
    if not entry or not entry.get('groups'):
        return result
    if Chem.MolToSmiles(mol, isomericSmiles=True) != smiles:
        raise ValueError('reporting_registry_requires_canonical_smiles')
    if h != sum(expected.values()):
        raise ValueError('reporting_unattached_H_not_supported_by_registry')
    totals = collections.Counter()
    sizes, fixed = [], []
    for group in entry['groups']:
        parents = {}
        for raw_parent, raw_count in group['parents'].items():
            parent, count = int(raw_parent), int(raw_count)
            if (parent not in expected or count <= 0 or float(raw_count) != count
                    or float(raw_parent) != parent or parent in parents):
                raise ValueError('reporting_invalid_group_parent')
            parents[parent] = count
        mass = sum(parents.values())
        if mass <= 0 or float(group['mass']) != mass:
            raise ValueError('reporting_group_mass_mismatch')
        fixed_mass = sum(n for i, n in parents.items() if not exchangeable[i])
        if fixed_mass not in (0, mass):
            raise ValueError('reporting_mixed_exchangeability_group')
        totals.update(parents)
        sizes.append(mass)
        if fixed_mass:
            fixed.append(mass)
    if any(totals[i] != n for i, n in expected.items()) or set(totals) - set(expected):
        raise ValueError('reporting_H_partition_mismatch')
    k = len(sizes)
    if entry.get('counts', {}).get('1H', k) != k:
        raise ValueError('reporting_registry_count_mismatch')
    return dict(H=h, K=k, group_sizes=sizes, nonexchange_H=h_fixed,
                nonexchange_K=len(fixed), nonexchange_group_sizes=fixed, known=True,
                partition_evidence=entry.get('partition_evidence', 'existing_registry_partition'))


def reporting_table(smiles, registry, expected_H=None):
    """按原候选顺序返回 CPU 侧表；K=-1 表示未知，不冒充零环境。"""
    rows = [structure_summary(s, registry.get(s)) for s in smiles]
    counts = torch.tensor([r['H'] for r in rows], dtype=torch.float32)
    if expected_H is not None:
        expected = torch.as_tensor(expected_H, dtype=torch.float32).cpu()
        if expected.shape != counts.shape or not torch.allclose(counts, expected, atol=.001, rtol=0):
            raise ValueError('reporting_candidate_H_drift')
    return dict(H=counts, K=torch.tensor([r['K'] if r['known'] else -1 for r in rows], dtype=torch.float32),
                nonexchange_H=torch.tensor([r['nonexchange_H'] for r in rows], dtype=torch.float32),
                nonexchange_K=torch.tensor([r['nonexchange_K'] if r['known'] else -1 for r in rows], dtype=torch.float32),
                known=torch.tensor([r['known'] for r in rows]),
                group_sizes=[r['group_sizes'] for r in rows],
                nonexchange_group_sizes=[r['nonexchange_group_sizes'] for r in rows],
                partition_evidence=[r['partition_evidence'] for r in rows])


def training_records(rows, annotations, registry):
    """从已审核 train 原子归属提取报告数；任何非 train 输入立即失败。"""
    if any(r.get('split') != 'train' or r.get('domain') != 'experimental' for r in rows):
        raise ValueError('reporting_fit_requires_experimental_train_only')
    result = []
    for row in rows:
        if row.get('measure') != 'observed_assigned_atom_equal_partial_possible':
            continue
        if row.get('review_spectral_weight', 1.) <= 0 or row['observation']['windows']:
            continue
        labels = annotations.get(make_key(row), {})
        count = sum(map(len, labels.get('groups', {}).values()))
        if count <= 0:
            continue
        info = structure_summary(row['smiles'], registry.get(row['smiles']))
        result.append(dict(split='train', source=row['source'], group=row['group'],
                           H=info['H'], K=info['K'], observed_count=count,
                           representative_count_feasible=info['known'] and count <= info['K'],
                           all_H_labels=count == info['H']))
    return result


def _binomial_log_prob(n, k, rates):
    n = n[:, None]
    k = torch.as_tensor(k, device=n.device, dtype=n.dtype)
    if k.ndim:
        k = k[:, None]
    coefficient = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma((n - k).clamp_min(0) + 1)
    values = []
    for rate in rates:
        if not 0 < rate <= 1:
            raise ValueError('reporting_invalid_representative_rate')
        if rate == 1:
            value = torch.where(n == k, torch.zeros_like(n), torch.full_like(n, -torch.inf))
        else:
            value = coefficient + k * math.log(rate) + (n - k) * math.log1p(-rate)
            value = value.masked_fill(k > n, -torch.inf)
        values.append(value[:, 0])
    return torch.stack(values, 1)


def _atomic_log_prob(n, k, entry):
    """EM 的向量 k 版本；正式未知 K 回退直接调用原 count_nll。"""
    k = torch.as_tensor(k, device=n.device, dtype=n.dtype)
    rates = entry['rates']
    weights = torch.as_tensor(entry['weights'], device=n.device, dtype=n.dtype)
    if (weights.shape != (len(rates),) or not torch.isfinite(weights).all() or (weights < 0).any()
            or weights.sum() <= 0 or any(not math.isfinite(r) or r <= 0 for r in rates)
            or (entry.get('concentration') is not None and entry['concentration'] <= 0)):
        raise ValueError('reporting_invalid_atomic_prior')
    weights = weights / weights.sum()
    columns = []
    for rate in rates:
        if rate <= 1:
            if entry.get('concentration') is None or rate == 1:
                value = _binomial_log_prob(n, k, [rate])[:, 0]
            else:
                concentration = entry['concentration']; a = rate * concentration; b = (1 - rate) * concentration
                coefficient = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma((n - k).clamp_min(0) + 1)
                value = (coefficient + torch.lgamma(k + a) + torch.lgamma((n - k).clamp_min(0) + b)
                         - torch.lgamma(n + concentration) - math.lgamma(a) - math.lgamma(b) + math.lgamma(concentration))
                value = value.masked_fill(k > n, -torch.inf)
        else:
            mu = (n * rate).clamp_min(1e-8)
            value = k * mu.log() - mu - torch.lgamma(k + 1)
        columns.append(value)
    return torch.logsumexp(torch.stack(columns, 1) + weights.log()[None, :], 1)


def fit_reporting(records, atomic_priors, prior_strength=20., iterations=100, representative_rates=REPRESENTATIVE_RATES):
    """拟合来源共享的 H/等价组报告混合；记录只来自 train，结构内重复均分权重。"""
    if any(r.get('split') != 'train' for r in records):
        raise ValueError('reporting_fit_requires_train_only')
    if prior_strength <= 0 or iterations < 1:
        raise ValueError('reporting_invalid_fit_settings')
    rates = tuple(float(r) for r in representative_rates)
    if not rates or any(not 0 < r <= 1 for r in rates):
        raise ValueError('reporting_invalid_representative_rate')
    result = {}
    for source in sorted({r['source'] for r in records}):
        if source not in atomic_priors:
            raise ValueError('reporting_missing_atomic_prior:' + source)
        all_rows = [r for r in records if r['source'] == source]
        known = [r for r in all_rows if r.get('K') is not None and r['K'] > 0]
        base = np.array([.9] + [.1 / len(rates)] * len(rates), dtype=float)
        weights = base.copy()
        if known:
            if any(r['H'] <= 0 or r['K'] > r['H'] or r['observed_count'] <= 0
                   or any(float(r[k]) != round(float(r[k])) for k in ('H', 'K', 'observed_count')) for r in known):
                raise ValueError('reporting_invalid_training_count')
            h = torch.tensor([r['H'] for r in known], dtype=torch.float64)
            k = torch.tensor([r['K'] for r in known], dtype=torch.float64)
            observed = torch.tensor([r['observed_count'] for r in known], dtype=torch.float64)
            components = torch.cat([_atomic_log_prob(h, observed, atomic_priors[source])[:, None],
                                    _binomial_log_prob(k, observed, rates)], 1).numpy()
            group_sizes = collections.Counter(r['group'] for r in known)
            sample_weight = np.array([1 / group_sizes[r['group']] for r in known])
            for _ in range(iterations):
                logits = components + np.log(weights)[None, :]
                largest = logits.max(1, keepdims=True)
                if not np.isfinite(largest).all():
                    raise ValueError('reporting_training_count_outside_model_support')
                responsibility = np.exp(logits - largest)
                responsibility /= responsibility.sum(1, keepdims=True)
                weights = ((responsibility * sample_weight[:, None]).sum(0) + prior_strength * base)
                weights /= sample_weight.sum() + prior_strength
        else:
            weights = np.array([1.] + [0.] * len(rates))
        result[source] = dict(atomic_prior=atomic_priors[source], representative_rates=list(rates),
                              mixture_weights=weights.tolist(), training_records=len(all_rows),
                              known_K_records=len(known), known_K_structures=len({r['group'] for r in known}),
                              prior_strength_connectivities=prior_strength,
                              interpretation='mixture of reported H labels and one representative per physical environment; acquisition coverage remains unknown')
    return result


def reporting_nll(counts, environments, observed_count, model, observation=None):
    """返回每报告标签的负对数概率；K 未知时逐位保留旧 H 计数结果。"""
    if observation and observation.get('windows'):
        raise ValueError('reporting_explicit_window_requires_visible_layout')
    if observed_count < 1 or float(observed_count) != round(float(observed_count)):
        raise ValueError('reporting_requires_positive_integer_labels')
    k = int(observed_count)
    n = counts.round()
    if (counts.ndim != 1 or environments.shape != counts.shape or not torch.isfinite(counts).all()
            or (n < 0).any() or ((counts - n).abs() > .001).any()):
        raise ValueError('reporting_invalid_candidate_counts')
    environments = environments.to(device=counts.device, dtype=counts.dtype)
    original = count_nll(counts, k, model['atomic_prior'])
    known = torch.isfinite(environments) & (environments > 0)
    if (environments[known] > n[known]).any() or not torch.equal(environments[known], environments[known].round()):
        raise ValueError('reporting_invalid_candidate_environment_count')
    if not known.any():
        return original
    weights = torch.tensor(model['mixture_weights'], device=counts.device, dtype=counts.dtype)
    if weights.shape != (1 + len(model['representative_rates']),) or (weights < 0).any() or not torch.isclose(weights.sum(), weights.new_tensor(1.)):
        raise ValueError('reporting_invalid_mixture_weights')
    if weights[0] == 1:
        return original
    safe_environments = torch.where(known, environments, torch.zeros_like(environments))
    representative = _binomial_log_prob(safe_environments, k, model['representative_rates'])
    components = torch.cat([-k * original[:, None], representative], 1)
    mixed = -torch.logsumexp(components + weights.log()[None, :], 1) / k
    return torch.where(known, mixed, original)


def reporting_delta(counts, environments, observed_count, model, observation=None):
    """返回替换旧 count 项所需的差值，供原评分权重做独立消融。"""
    return (reporting_nll(counts, environments, observed_count, model, observation)
            - count_nll(counts, int(observed_count), model['atomic_prior']))

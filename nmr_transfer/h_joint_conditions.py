"""功能概述：把条件位移学习与羰基CH的报告状态分开，供配对研究训练和检索共用。

ReportingHead只接收结构类别、查询溶剂/pH及已知缺峰过程，输出整谱报告概率；
不接收分子ID、目标位移或测试身份。JointSpectra在ReplaySpectra上附加训练观测字段，
保留原q/窗口/原子mask；额外整峰删除只用于独立的训练辅助似然。
state_nll对完整/未报告羰基CH两个整谱状态求边缘似然，梯度可回传到保留的峰位。
joint_objective只替换合格HMDB记录原来的谱NLL；物理发射及谱线计数始终保留。
explicit窗口不启用此报告模型；空剩余谱和混合醛/甲酰胺结构回退完整状态。
build_pairs仅在train内寻找同结构、不同明确溶剂且可信原子或整数积分口径一致的配对；
不把无积分HMDB峰表强行对齐。pair_loss比较完整谱分位差以及可信NMe2的均值/间距差，
不指定等价甲基的原子方向。由h_joint_train/h_joint_workflow调用，无网络或源文件写入。
"""
import collections
import math

import numpy as np
import torch
from torch import nn

from .data import quantiles
from .emission_model import weighted_quantiles
from .evaluate import missing_queries
from .h_acyl import acyl_environment
from .h_amide_audit import amide_groups
from .h_conditions import CONDITION_DIM
from .h_formyl_reporting import candidate_families, counterfactual_quantiles
from .h_labels import make_key
from .h_replaytrain import ReplaySpectra
from .h_reviewed_supervision import reviewed_objective
from .h_uncertainty import quantile_classes


class ReportingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(CONDITION_DIM + 3, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.constant_(self.layers[-1].bias, math.log(9.))

    def forward(self, families, conditions, dropout=False):
        code = conditions.new_tensor([[f == 'formamide', f == 'aldehyde'] for f in families])
        process = torch.as_tensor(dropout, device=conditions.device, dtype=conditions.dtype)
        process = process.expand(len(families)).reshape(-1, 1)
        p = self.layers(torch.cat([code, conditions, process], 1))[:, 0].sigmoid()
        return torch.where(code.any(1), p, torch.ones_like(p))


class JointSpectra(ReplaySpectra):
    def __getitem__(self, index):
        g = super().__getitem__(index)
        row = self.rows[index]
        family = candidate_families([row['smiles']])[0]
        meta = self.metadata.get(make_key(row), {})
        eligible = (row['domain'] == 'experimental' and row['source'] == 'HMDB'
                    and family != 'other' and not row['observation']['windows']
                    and row.get('review_spectral_weight', 1.) > 0)
        lines = meta.get('line_centers', np.unique(row['q']))
        g.reporting_eligible = torch.tensor([eligible])
        g.reporting_k = torch.tensor([max(1, len(lines))], dtype=torch.float32)
        g.reporting_dropout_q = g.q.clone()
        g.reporting_dropout_k = g.reporting_k.clone()
        g.reporting_dropout_valid = torch.tensor([False])
        if eligible:
            for name, altered in missing_queries(dict(row, q=quantiles(lines)), '1H'):
                if name != 'peak_dropout':
                    continue
                g.reporting_dropout_q = torch.tensor(altered['q'] / 10, dtype=torch.float32)[None, :]
                g.reporting_dropout_k = torch.tensor([len(np.unique(altered['q']))], dtype=torch.float32)
                g.reporting_dropout_valid = torch.tensor([True])
        return g


def spectrum_nll(observed, predicted, sigma=.025):
    residual = (observed[:, :, None] - predicted[:, None, :]) / sigma
    return (-torch.logsumexp(-.5 * residual.square(), dim=2)
            + math.log(predicted.shape[1])).mean(1)


def state_nll(full, missing, probability, effective):
    p = probability.clamp(1e-6, 1 - 1e-6)
    k = effective.clamp_min(1)
    mixed = -torch.logaddexp(p.log() - k * full, torch.log1p(-p) - k * missing) / k
    return torch.where((probability >= 1) | (full == missing), full, mixed)


def joint_objective(model, physical, observed_details, batch, enabled):
    parts = reviewed_objective(observed_details, batch)
    if not enabled:
        return dict(parts, report_aux=parts['loss'] * 0)
    families = candidate_families(batch.structure_smiles)
    # training_details只替换水相观测分位点，原物理成分仍完整；类别必须使用同一可见质量。
    visible = physical
    if batch.line_water.any():
        fixed_mass = torch.zeros(batch.num_graphs, device=batch.x.device).index_add(
            0, physical['groups'], physical['weights'] * (~physical['exchangeable']).float())
        suppress = batch.line_water.reshape(-1) & (fixed_mass > 0)
        keep = ~(physical['exchangeable'] & suppress[physical['groups']])
        visible = {k: v[keep] for k, v in physical.items() if k != 'quantiles'}
    tags, checked = quantile_classes(visible, batch)
    if not torch.equal(checked, observed_details['quantiles']):
        raise ValueError('joint_reporting_tags_not_same_forward')
    full = observed_details['quantiles']
    missing, active = counterfactual_quantiles(full, tags, families)
    allowed = active & batch.reporting_eligible.reshape(-1)
    if not allowed.any():
        return dict(parts, report_aux=parts['loss'] * 0)
    conditions = batch.condition_vector.reshape(-1, CONDITION_DIM)
    p = model.reporting_head(families, conditions)
    fn = spectrum_nll(batch.q, full)
    mn = spectrum_nll(batch.q, missing)
    mixed = state_nll(fn, mn, p, batch.reporting_k.reshape(-1))
    # 使用与旧NLL相同的逐图分母，其他图和被审核隔离的图保持原有权重。
    delta = torch.where(allowed, mixed - fn, torch.zeros_like(fn)).mean()
    valid = allowed & batch.reporting_dropout_valid.reshape(-1)
    dropout = full.sum() * 0
    if valid.any():
        df = spectrum_nll(batch.reporting_dropout_q, full)
        dm = spectrum_nll(batch.reporting_dropout_q, missing)
        dp = model.reporting_head(families, conditions, True)
        dropout = state_nll(df, dm, dp, batch.reporting_dropout_k.reshape(-1))[valid].mean()
    # 稀少条件采用共享小头和logit收缩，避免将所有谱无代价判成不报告。
    regularizer = (torch.logit(p[allowed].clamp(.001, .999)) - math.log(9.)).square().mean()
    auxiliary = .005 * delta + .001 * dropout + .0001 * regularizer
    return dict(parts, loss=parts['loss'] + auxiliary, report_aux=auxiliary)


def reliable_methyl_targets(row, labels, graph):
    sites = labels.get(make_key(row), {}).get('groups', {})
    result = {}
    for nitrogen, parents in amide_groups(graph):
        values = [sites.get(p, []) for p in parents]
        if all(len(v) == 3 and np.ptp(v) <= .03 for v in values):
            centres = sorted(float(np.median(v)) / 10 for v in values)
            result[nitrogen] = dict(parents=parents, mean=float(np.mean(centres)), gap=centres[1] - centres[0])
    return result


def build_pairs(rows, labels, graphs, metadata):
    grouped = collections.defaultdict(list)
    for i, row in enumerate(rows):
        if row['split'] != 'train' or row['domain'] != 'experimental':
            raise ValueError('cross_solvent_pairs_require_experimental_train')
        meta = metadata.get(make_key(row), {})
        family = meta.get('family', 'unknown')
        if (family in ('unknown', 'other') or '+' in family
                or row['observation']['windows'] or row.get('review_spectral_weight', 1.) == 0
                or not acyl_environment(row['smiles'])[:, 0].any()):
            continue
        grouped[row['smiles']].append((i, family))
    pairs = []
    rejected = collections.Counter()
    for smiles, items in sorted(grouped.items()):
        for a, (i, left) in enumerate(items):
            for j, right in items[a + 1:]:
                if left == right:
                    continue
                lrow, rrow = rows[i], rows[j]
                lm = labels.get(make_key(lrow), {})
                rm = labels.get(make_key(rrow), {})
                full = all(bool(x.get('full_atom_labels') or x.get('integral_consistent')) for x in (lm, rm))
                # 无积分峰高不是H质量；完整标注仍须存在审核后的有效原子组。
                full &= all(r['source'] != 'HMDB' and (m.get('groups') or m.get('integral_consistent'))
                            for r, m in [(lrow, lm), (rrow, rm)])
                lt = reliable_methyl_targets(lrow, labels, graphs[smiles])
                rt = reliable_methyl_targets(rrow, labels, graphs[smiles])
                methyl = {n: dict(parents=lt[n]['parents'], mean=rt[n]['mean'] - lt[n]['mean'],
                                  gap=rt[n]['gap'] - lt[n]['gap']) for n in lt.keys() & rt.keys()}
                if not full and not methyl:
                    rejected['unassigned_or_incomplete_observation'] += 1
                    continue
                pairs.append(dict(left=i, right=j, smiles=smiles, group=lrow['group'],
                                  solvents=[left, right], complete_distribution=bool(full), methyl=methyl))
    counts = collections.Counter(p['group'] for p in pairs)
    return pairs, dict(accepted_pairs=len(pairs), connectivities=len(counts),
                       full_distribution_pairs=sum(p['complete_distribution'] for p in pairs),
                       methyl_pairs=sum(bool(p['methyl']) for p in pairs), rejected=dict(rejected),
                       cross_solvent_structures=sum(len({x[1] for x in v}) > 1 for v in grouped.values()))


def pair_loss(model, batch, pairs):
    details = model(batch, details=True)
    losses = []
    for a, pair in enumerate(pairs):
        i, j = 2 * a, 2 * a + 1
        if pair['complete_distribution']:
            predicted = details['quantiles'][j] - details['quantiles'][i]
            target = batch.q[j] - batch.q[i]
            losses.append((predicted - target).abs().mean())
        for value in pair['methyl'].values():
            pq = []
            for graph_id in (i, j):
                owners = batch.ptr[graph_id] + torch.tensor(value['parents'], device=batch.x.device)
                keep = torch.isin(details['owners'], owners)
                pq.append(weighted_quantiles(details['values'][keep], details['weights'][keep],
                          torch.zeros(int(keep.sum()), dtype=torch.long, device=batch.x.device), 1, 2)[0])
            mean = pq[1].mean() - pq[0].mean()
            gap = (pq[1][1] - pq[1][0]) - (pq[0][1] - pq[0][0])
            losses.extend([(mean - value['mean']).abs(), (gap - value['gap']).abs()])
    return torch.stack(losses).mean() if losses else details['quantiles'].sum() * 0

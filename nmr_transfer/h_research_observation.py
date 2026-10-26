"""功能概述：在固定候选预测上联合学习来源无关的观测过程，供新 extract/ranker 调用。

prepare_observation 将查询谱表转为可序列化正观测：xs 使用 ppm，weights 归一化，
raw_weights 保留原口径；assigned_labels、integral、line_positions、quantile_support
明确区分。整数标签从不自动成为积分。HMDB 的 source/record_id 仅用于取原始线表，
峰高不用作积分，按既有最大簇宽 0.08 ppm 聚类；这些身份字段不进入返回值或网络。

TrainableObservationModel 同时接受 F0 分位数与 F1 物理环境中心，输入形状 [B,P]，
prediction_mass 可传组 H 质量，per_quantile_sigma 可传 F1 自身宽度。模型按四种观测
语义学习九类局部宽度倍率、报告倾向、Gaussian/Student-t(3) 尾部及原子/环境报告混合。
trust 若提供只校验结构侧表范围，不改变位移宽度；结构分区未知不等于位移不可信。forward
返回固定三列 O_FEATURE_NAMES，O 关闭时调用方追加同维零值以保持 ranker 容量。
未知覆盖只解释正观测，显式窗口按 PDF/CDF 同一混合条件化；没有未报峰反向惩罚。
报告数量项仅用于无显式窗/删峰的整数原子标签，H/K 都来自各候选结构；未知 K 退回
H 报告过程。稳健几何污染允许标签多报，不硬排除候选。未标注原子不等同测量缺峰。

regularization 对报告、宽度、尾部和混合向预声明初值收缩，防止藏峰与无界展宽。
报告倾向由位置和总标签数弱监督，不代表逐原子缺峰已校准。pack/load_pack 保存纯
配置和 CPU 参数；fit_observation_model 是可选 train-only 小助手，默认建议与 ranker
联合训练。training_line_variants 仅在原表明确 MHz 时生成预声明 Hz 间隔的训练线表
扰动；training_environment_variant 只在有认证物理组映射时整组删除报告。缺证据均
返回明确 skip，不按位移相近造环境、不假定 MHz。两者只接受实验 train 查询、返回
prepared observation 及审计，不改原查询或假造积分。模块不读数据文件、不联网、
不选择设备、不产生文件副作用。
"""
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn

from .h_research_recall import _windows


O_FEATURE_NAMES = ('observation_positive_nll', 'observation_visibility_delta',
                   'observation_report_nll')
SEMANTICS = ('assigned_labels', 'integral', 'line_positions', 'quantile_support')
LINE_SPLIT_HZ = (4., 8., 12.)
ENVIRONMENT_DROP_FRACTION = .25


def _cluster(xs, width):
    xs = np.unique(np.asarray(xs, dtype=float))
    result, start = [], 0
    for i in range(1, len(xs)):
        if xs[i] - xs[start] > width + 1e-8:
            result.append(float(xs[start:i].mean())); start = i
    if len(xs):
        result.append(float(xs[start:].mean()))
    return np.asarray(result)


def prepare_observation(query, raw=None, line_resolution=.08):
    """只输出公开观测字段；不推断未知窗口、积分单位或未报告峰。"""
    state = query.get('observation', query.get('query_observation', {}))
    windows = _windows(state.get('windows'))
    partial = bool(state.get('partial_peaks', False))
    measure = query.get('measure', '')
    line_table = query.get('source') == 'HMDB' or measure == 'line_positions'
    if line_table:
        if not math.isfinite(float(line_resolution)) or line_resolution <= 0:
            raise ValueError('observation_invalid_line_resolution')
        if raw is not None and query.get('record_id') in raw:
            xs = np.asarray(raw[query['record_id']]['value']['x'], dtype=float)
        else:
            xs = np.asarray(query.get('observed_xs', query.get('q', [])), dtype=float)
        if partial:
            retained = np.unique(np.asarray(query.get('q', []), dtype=float))
            xs = xs[np.isclose(xs[:, None], retained[None], atol=1e-4, rtol=0).any(1)]
        weights = np.ones_like(xs)
        semantics = 'line_positions'
    elif len(query.get('observed_xs', [])):
        xs = np.asarray(query['observed_xs'], dtype=float)
        weights = np.asarray(query.get('observed_weights', np.ones(len(xs))), dtype=float)
        semantics = ('integral' if measure == 'integral_weighted' else
                     'assigned_labels' if measure == 'observed_assigned_atom_equal_partial_possible'
                     else 'quantile_support')
    else:
        xs = np.asarray(query.get('q', []), dtype=float)
        weights = np.ones_like(xs)
        semantics = 'quantile_support'
    if (xs.ndim != 1 or weights.shape != xs.shape or not len(xs)
            or not np.isfinite(xs).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('observation_invalid_reported_support')
    if windows:
        inside = np.zeros(len(xs), dtype=bool)
        for a, b in windows:
            inside |= (xs >= a) & (xs <= b)
        xs, weights = xs[inside], weights[inside]
    keep = weights > 0
    xs, weights = xs[keep], weights[keep]
    if not len(xs):
        raise ValueError('observation_no_reported_support_in_declared_window')
    integer_labels = (semantics == 'assigned_labels'
                      and np.allclose(weights, weights.round(), atol=.001, rtol=0))
    if line_table:
        xs = _cluster(xs, line_resolution)
        weights = np.ones_like(xs)
    else:
        unique, inverse = np.unique(xs, return_inverse=True)
        weights = np.bincount(inverse, weights=weights, minlength=len(unique))
        xs = unique
    labels = (int(round(float(weights.sum())))
              if integer_labels
              else None)
    return dict(xs=xs.tolist(), weights=(weights / weights.sum()).tolist(),
                raw_weights=weights.tolist(), semantics=semantics, kind=SEMANTICS.index(semantics),
                windows=windows, partial_peaks=partial, reported_count=labels,
                coverage='explicit_window' if windows else 'declared_partial' if partial else 'unknown')


def _logit(probability):
    return math.log(probability / (1 - probability))


def _augmentation_guard(query, seed):
    if query.get('split') != 'train' or query.get('domain') != 'experimental':
        raise ValueError('observation_augmentation_requires_experimental_train_only')
    if type(seed) is not int or seed < 0:
        raise ValueError('observation_augmentation_requires_nonnegative_integer_seed')


def training_line_variants(query, raw=None, seed=0):
    """明确 MHz 才将报告簇细拆成单位线；返回一个训练扰动或可审计 skip。"""
    _augmentation_guard(query, seed)
    if query.get('source') != 'HMDB' and query.get('measure') != 'line_positions':
        return dict(variants=[], audit=dict(skipped='not_explicit_line_table'))
    from .h_global_field_capacity import frequency_mhz
    record = (raw or {}).get(query.get('record_id'), query)
    frequency = frequency_mhz(record)
    if frequency is None:
        return dict(variants=[], audit=dict(skipped='explicit_MHz_unavailable'))
    base = prepare_observation(query, raw)
    width_hz = float(np.random.default_rng(seed).choice(LINE_SPLIT_HZ))
    xs = np.asarray(base['xs'])
    fine = np.sort(np.concatenate([xs - width_hz / (2 * frequency), xs + width_hz / (2 * frequency)]))
    if base['windows']:
        inside = np.zeros(len(fine), dtype=bool)
        for a, b in base['windows']:
            inside |= (fine >= a) & (fine <= b)
        fine = fine[inside]
    if not len(fine):
        return dict(variants=[], audit=dict(skipped='no_synthetic_line_in_explicit_window'))
    fine = np.unique(fine)
    observation = dict(base, xs=fine.tolist(), weights=np.full(len(fine), 1 / len(fine)).tolist(),
                       raw_weights=np.ones(len(fine)).tolist(), reported_count=None)
    return dict(variants=[dict(name='training_fine_line_split', observation=observation)],
                audit=dict(seed=seed, frequency_MHz=frequency, synthetic_spacing_Hz=width_hz,
                           original_positions=len(xs), augmented_line_positions=len(fine),
                           semantics='unit reported lines, no integral or inferred coupling',
                           skipped=None))


def training_environment_variant(query, raw=None, mapping=None, seed=0):
    """仅按带证据的物理组映射删除报告；不把 unique ppm 直接视作环境。"""
    _augmentation_guard(query, seed)
    if not mapping or mapping.get('certified') is not True:
        return dict(variants=[], audit=dict(skipped='certified_environment_mapping_unavailable'))
    evidence = mapping.get('evidence_sha256', '')
    if (not isinstance(evidence, str) or len(evidence) != 64
            or any(c not in '0123456789abcdef' for c in evidence)):
        return dict(variants=[], audit=dict(skipped='environment_mapping_evidence_unavailable'))
    base = prepare_observation(query, raw)
    groups = mapping.get('groups')
    positions = mapping.get('positions_ppm')
    try:
        aligned = (np.asarray(positions, dtype=float).shape == (len(base['xs']),)
                   and np.allclose(positions, base['xs'], atol=1e-6, rtol=0))
    except (TypeError, ValueError):
        aligned = False
    if (not isinstance(groups, (list, tuple)) or len(groups) != len(base['xs'])
            or not aligned or any(type(x) not in (str, int) or (isinstance(x, str) and not x) for x in groups)):
        return dict(variants=[], audit=dict(skipped='ambiguous_or_misaligned_environment_mapping'))
    keys = [(type(x).__name__, str(x)) for x in groups]
    unique = sorted(set(keys))
    if len(unique) < 2:
        return dict(variants=[], audit=dict(skipped='fewer_than_two_reported_environments'))
    count = min(len(unique) - 1, max(1, int(round(len(unique) * ENVIRONMENT_DROP_FRACTION))))
    selected = np.random.default_rng(seed).choice(len(unique), size=count, replace=False)
    removed = {unique[int(i)] for i in selected}
    keep = np.asarray([g not in removed for g in keys])
    xs = np.asarray(base['xs'])[keep]; weights = np.asarray(base['raw_weights'])[keep]
    labels = int(round(float(weights.sum()))) if base['reported_count'] is not None else None
    observation = dict(base, xs=xs.tolist(), weights=(weights / weights.sum()).tolist(),
                       raw_weights=weights.tolist(), reported_count=labels,
                       partial_peaks=True, coverage='explicit_window' if base['windows'] else 'declared_partial')
    return dict(variants=[dict(name='training_reported_environment_dropout', observation=observation)],
                audit=dict(seed=seed, mapping_evidence_sha256=evidence,
                           reported_environments=len(unique), dropped_environments=count,
                           original_positions=len(base['xs']), retained_positions=len(xs),
                           semantics='synthetic deletion of certified reported groups, not evidence of measurement missingness',
                           skipped=None))


class TrainableObservationModel(nn.Module):
    """固定三列观测特征；模型不接受 source、ID 或真结构索引。"""
    def __init__(self, initial_sigmas=None):
        super().__init__()
        widths = torch.as_tensor([.2] * 9 if initial_sigmas is None else initial_sigmas,
                                 dtype=torch.float32)
        if widths.shape != (9,) or not torch.isfinite(widths).all() or (widths <= 0).any():
            raise ValueError('observation_requires_nine_positive_sigmas')
        self.register_buffer('initial_sigmas', widths)
        self.register_buffer('report_prior', torch.tensor(_logit((.8 - .05) / .945)))
        self.register_buffer('tail_prior', torch.tensor(_logit((.05 - .001) / .199)))
        self.width_raw = nn.Parameter(torch.zeros(len(SEMANTICS), 9))
        self.report_logits = nn.Parameter(self.report_prior.expand(len(SEMANTICS), 9).clone())
        self.tail_logit = nn.Parameter(self.tail_prior.expand(len(SEMANTICS)).clone())
        self.environment_logit = nn.Parameter(torch.zeros(len(SEMANTICS)))

    def regularization(self):
        """无量纲均方先验；调用方必须记录外部损失系数。"""
        return (self.width_raw.square().mean()
                + (self.report_logits - self.report_prior).square().mean()
                + (self.tail_logit - self.tail_prior).square().mean()
                + self.environment_logit.square().mean())

    def _profile(self, centers, sigma, mass, xs, windows, tail):
        z = (xs[None, None] - centers[:, :, None]) / sigma[:, :, None]
        gaussian = -.5 * z.square() - sigma[:, :, None].log() - .5 * math.log(2 * math.pi)
        student = (math.log(2 / (math.pi * math.sqrt(3))) - sigma[:, :, None].log()
                   - 2 * torch.log1p(z.square() / 3))
        component = torch.logaddexp(gaussian + torch.log1p(-tail), student + tail.log())
        logmass = mass.clamp_min(1e-30).log().masked_fill(mass <= 0, -torch.inf)
        result = torch.logsumexp(component + logmass[:, :, None], 1)
        if windows:
            probability = centers.new_zeros(centers.shape)
            for a, b in windows:
                za, zb = (a - centers) / sigma, (b - centers) / sigma
                normal = .5 * (torch.erf(zb / math.sqrt(2)) - torch.erf(za / math.sqrt(2)))
                def student_cdf(z):
                    u = z / math.sqrt(3)
                    return .5 + (torch.atan(u) + u / (1 + u.square())) / math.pi
                robust = student_cdf(zb) - student_cdf(za)
                probability = probability + (1 - tail) * normal + tail * robust
            visible_mass = (probability.clamp(0, 1) * mass).sum(1)
            result = result - visible_mass.clamp_min(1e-12).log()[:, None]
            result = torch.where((visible_mass >= 1e-8)[:, None], result,
                                 result.new_full((), -1e6))
        return result

    def _count_nll(self, H, K, reported, probability, kind):
        k = probability.new_tensor(float(reported))
        def binomial(n):
            value = (torch.lgamma(n + 1) - torch.lgamma(k + 1)
                     - torch.lgamma((n - k).clamp_min(0) + 1)
                     + k * probability.log() + (n - k) * torch.log1p(-probability))
            # 两个不可能的核心分量同时为 -inf 会使 logaddexp 的反向导数 NaN。
            # 固定污染尾部使它们的精确极限为零；有限下界不改变浮点概率。
            return value.masked_fill(k > n, -1e6)
        atom = binomial(H)
        if K is None:
            core = atom
        else:
            known = torch.isfinite(K) & (K > 0) & (K <= H) & ((K - K.round()).abs() <= .001)
            environment = binomial(torch.where(known, K.round(), H))
            mixture = self.environment_logit[kind].sigmoid()
            mixed = torch.logaddexp(atom + torch.log1p(-mixture), environment + mixture.log())
            core = torch.where(known, mixed, atom)
        # 固定且归一化的几何污染分布，避免人工多报使 NLL 无穷。
        contamination = math.log(.1) + k * math.log(.9)
        robust = torch.logaddexp(core + math.log(.95), contamination + math.log(.05))
        return -robust / k.clamp_min(1)

    def forward(self, predicted_ppm, tags, observation, H=None, K=None,
                prediction_mass=None, trust=None, per_quantile_sigma=None):
        """[B,P] ppm/class → [B,3]；同一查询的显式观测供整个候选 batch 共用。"""
        if predicted_ppm.ndim != 2 or tags.shape != predicted_ppm.shape or not len(predicted_ppm):
            raise ValueError('observation_invalid_prediction_shape')
        if observation.get('semantics') not in SEMANTICS:
            raise ValueError('observation_explicit_semantics_required')
        kind = SEMANTICS.index(observation['semantics'])
        if 'kind' in observation and (type(observation['kind']) is not int or observation['kind'] != kind):
            raise ValueError('observation_kind_semantics_mismatch')
        device = predicted_ppm.device
        dtype = torch.float64 if predicted_ppm.dtype == torch.float64 else torch.float32
        centers = predicted_ppm.to(dtype)
        tags = tags.to(device=device)
        if (tags.dtype not in (torch.int32, torch.int64) or (tags < 0).any()
                or (tags >= 9).any() or not torch.isfinite(centers).all()):
            raise ValueError('observation_invalid_predictions_or_classes')
        xs = torch.as_tensor(observation['xs'], device=device, dtype=dtype).reshape(-1)
        weights = torch.as_tensor(observation['weights'], device=device, dtype=dtype).reshape(-1)
        if (not len(xs) or xs.shape != weights.shape or not torch.isfinite(xs).all()
                or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError('observation_invalid_normalized_support')
        weights = weights / weights.sum()
        windows = _windows(observation.get('windows'))
        if windows and not torch.stack([(xs >= a) & (xs <= b) for a, b in windows]).any(0).all():
            raise ValueError('observation_support_outside_explicit_window')
        mass = torch.ones_like(centers) if prediction_mass is None else torch.as_tensor(prediction_mass, device=device, dtype=dtype)
        if (mass.shape != centers.shape or not torch.isfinite(mass).all()
                or (mass < 0).any() or (mass.sum(1) <= 0).any()):
            raise ValueError('observation_invalid_prediction_mass')
        mass = mass / mass.sum(1, keepdim=True)
        if per_quantile_sigma is None:
            base_sigma = self.initial_sigmas[tags.long()].to(dtype)
        else:
            base_sigma = torch.as_tensor(per_quantile_sigma, device=device, dtype=dtype)
            if base_sigma.shape != centers.shape or not torch.isfinite(base_sigma).all() or (base_sigma <= 0).any():
                raise ValueError('observation_invalid_per_quantile_sigma')
        sigma = base_sigma * torch.exp(math.log(4) * self.width_raw[kind].tanh()[tags.long()])
        if trust is not None:
            confidence = torch.as_tensor(trust, device=device, dtype=dtype)
            if (confidence.shape != centers.shape or not torch.isfinite(confidence).all()
                    or (confidence < 0).any() or (confidence > 1).any()):
                raise ValueError('observation_invalid_prediction_trust')
            # 这里只校验接口，不把结构 partition trust 当成位移置信度。
        probability = .05 + .945 * self.report_logits[kind].sigmoid()[tags.long()]
        visible_mass = mass * probability
        visible_mass = visible_mass / visible_mass.sum(1, keepdim=True)
        tail = .001 + .199 * self.tail_logit[kind].sigmoid()
        logp = self._profile(centers, sigma, visible_mass, xs, windows, tail)
        neutral = self._profile(centers, sigma, mass, xs, windows, tail)
        positive = -(logp * weights[None]).sum(1)
        visibility_delta = positive + (neutral * weights[None]).sum(1)
        report = positive.new_zeros(len(centers))
        k = observation.get('reported_count')
        if (observation.get('semantics') == 'assigned_labels' and not windows
                and not observation.get('partial_peaks', False) and k is not None and H is not None):
            if type(k) is not int or k < 1:
                raise ValueError('observation_invalid_reported_label_count')
            H = torch.as_tensor(H, device=device, dtype=dtype)
            if H.shape != (len(centers),) or not torch.isfinite(H).all() or (H < 0).any() or ((H - H.round()).abs() > .001).any():
                raise ValueError('observation_invalid_candidate_H')
            K = None if K is None else torch.as_tensor(K, device=device, dtype=dtype)
            if K is not None and K.shape != H.shape:
                raise ValueError('observation_invalid_candidate_K_shape')
            if K is not None:
                specified = torch.isfinite(K) & (K >= 0)
                if ((K[specified] > H[specified])
                        | ((K[specified] - K[specified].round()).abs() > .001)).any():
                    raise ValueError('observation_invalid_certified_K')
            report = self._count_nll(H.round(), K, k, (mass * probability).sum(1), kind)
        return torch.stack([positive, visibility_delta, report], 1)

    def pack(self):
        """纯 CPU 参数对象，外部工作流负责冻结 hash 与持久化。"""
        return dict(format='research_observation_v1', feature_names=list(O_FEATURE_NAMES),
                    config=dict(initial_sigmas=self.initial_sigmas.detach().cpu().tolist()),
                    state_dict={k: v.detach().cpu().clone() for k, v in self.state_dict().items()})

    @classmethod
    def load_pack(cls, value):
        if value.get('format') != 'research_observation_v1' or value.get('feature_names') != list(O_FEATURE_NAMES):
            raise ValueError('observation_pack_schema_changed')
        model = cls(**value['config'])
        model.load_state_dict(value['state_dict'], strict=True)
        return model


def fit_observation_model(records, initial_sigmas=None, epochs=10, lr=.001,
                          regularization_weight=.01, report_weight=.1):
    """可选 CPU train-only 拟合；联合 ranker 可直接使用 forward/regularization。"""
    if not records or any(r.get('split') != 'train' or r.get('domain') != 'experimental' for r in records):
        raise ValueError('observation_fit_requires_experimental_train_only')
    if type(epochs) is not int or epochs < 1 or lr <= 0 or regularization_weight <= 0 or report_weight < 0:
        raise ValueError('observation_invalid_fit_settings')
    model = TrainableObservationModel(initial_sigmas)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        total = 0.
        for row in records:
            optimizer.zero_grad()
            inputs = {name: row[name] for name in ('prediction_mass', 'trust', 'per_quantile_sigma') if name in row}
            output = model(torch.as_tensor(row['predicted_ppm'], dtype=torch.float32).reshape(1, -1),
                           torch.as_tensor(row['tags'], dtype=torch.long).reshape(1, -1), row['observation'],
                           H=None if row.get('H') is None else torch.tensor([row['H']]),
                           K=None if row.get('K') is None else torch.tensor([row['K']]),
                           **{k: torch.as_tensor(v).reshape(1, -1) for k, v in inputs.items()})
            loss = output[:, 0].mean() + report_weight * output[:, 2].mean() + regularization_weight * model.regularization()
            if not torch.isfinite(loss):
                raise ValueError('observation_nonfinite_training_loss')
            loss.backward(); optimizer.step(); total += float(loss.detach())
        losses.append(total / len(records))
    def numeric_value(value):
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        if isinstance(value, dict):
            return {k: numeric_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [numeric_value(v) for v in value]
        return value
    serializable = [{k: numeric_value(v)
                     for k, v in row.items() if k not in ('source', 'record_id', 'smiles', 'group')}
                    for row in records]
    fingerprint = hashlib.sha256(json.dumps(serializable, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return model, dict(training_records=len(records), epochs=epochs, losses=losses,
                       training_numeric_sha256=fingerprint, regularization_weight=regularization_weight,
                       report_weight=report_weight, source_or_id_model_input=False)

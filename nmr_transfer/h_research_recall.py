"""功能概述：为新一轮固定候选库检索提供双通道召回，不修改旧冻结评分器。

stable_order 对小分数优先且按原候选索引打破平分；mixed_pool_details / combine_shortlists
先各取原 coarse 与正观测通道的 256 名，去重后交替下探补满 512。函数没有真实结构、
来源或查询 ID 参数，不补正例；默认检查完整 114982 候选。返回池按 coarse 的稳定次序
存放，池的成员身份与池内重排次序分离。新 ranker 应将整池前置、其余候选保留 coarse
次序，不能沿用旧 top512 前缀假设来计算池外名次。

directional_kernel_scores 直接读取既有 class-density 的 logp/CDF，按已报告峰计算正向
负对数密度；只有明确 windows 才做窗口条件化，unknown 不推断频窗或惩罚未报预测。
directional_scores 是 ppm 支持点的分块高斯版本，便于 CPU 单测与无密度缓存调用。
它不使用氢总数、H/K 报告先验、来源或真候选；不训练、不写文件、不主动选择设备。
prepare_observation 产生的 xs 与归一化 weights 可直接作为本模块输入。
"""
import math

import torch


FIXED_CANDIDATES = 114982


def stable_order(scores):
    """小分数优先，完全相等时原候选索引优先。"""
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError('recall_requires_finite_score_vector')
    return torch.argsort(scores, stable=True)


def mixed_pool_details(coarse, positive, pool_size=512, per_channel=256,
                       expected_candidates=FIXED_CANDIDATES):
    """返回精确大小的唯一候选池与两通道 provenance；不接受标签。"""
    if coarse.shape != positive.shape or coarse.device != positive.device:
        raise ValueError('recall_channel_shape_or_device_mismatch')
    if expected_candidates is not None and len(coarse) != expected_candidates:
        raise ValueError('recall_candidate_library_size_changed')
    if (type(pool_size) is not int or type(per_channel) is not int
            or not 0 < per_channel <= pool_size or 2 * per_channel > pool_size
            or len(coarse) < pool_size):
        raise ValueError('recall_invalid_pool_sizes')
    orders = [stable_order(coarse), stable_order(positive)]
    # 只将排序索引搬到 CPU；不搬运密度、预测谱或候选身份。
    lists = [o.detach().cpu().tolist() for o in orders]
    selected, origin = set(), {}
    for channel in range(2):
        for index in lists[channel][:per_channel]:
            selected.add(index)
            origin[index] = origin.get(index, 0) | (1 << channel)
    initial_origin = dict(origin)
    cursor = [per_channel, per_channel]
    channel = 0
    while len(selected) < pool_size:
        while cursor[channel] < len(coarse) and lists[channel][cursor[channel]] in selected:
            cursor[channel] += 1
        if cursor[channel] < len(coarse):
            index = lists[channel][cursor[channel]]
            cursor[channel] += 1
            selected.add(index)
            origin[index] = origin.get(index, 0) | (1 << channel)
        channel = 1 - channel
    coarse_rank = torch.empty_like(orders[0])
    positive_rank = torch.empty_like(orders[1])
    coarse_rank[orders[0]] = torch.arange(len(coarse), device=coarse.device)
    positive_rank[orders[1]] = torch.arange(len(coarse), device=coarse.device)
    members = torch.tensor(sorted(selected), device=coarse.device, dtype=torch.long)
    pool = members[torch.argsort(coarse_rank[members], stable=True)]
    return dict(indices=pool, coarse_rank=coarse_rank[pool] + 1,
                positive_rank=positive_rank[pool] + 1,
                initial_membership=torch.tensor([initial_origin.get(int(i), 0) for i in pool.cpu()],
                                                device=pool.device, dtype=torch.uint8),
                refill_channel=torch.tensor([origin[int(i)] for i in pool.cpu()],
                                            device=pool.device, dtype=torch.uint8),
                pool_size=pool_size, per_channel=per_channel)


def combine_shortlists(coarse, positive, pool_size=512, per_channel=256,
                       expected_candidates=FIXED_CANDIDATES):
    """仅返回 pool LongTensor，详细 provenance 由 mixed_pool_details 提供。"""
    return mixed_pool_details(coarse, positive, pool_size, per_channel,
                              expected_candidates)['indices']


def _observation_tensors(xs, weights, device, dtype):
    xs = torch.as_tensor(xs, device=device, dtype=dtype).reshape(-1)
    weights = (torch.ones_like(xs) if weights is None
               else torch.as_tensor(weights, device=device, dtype=dtype).reshape(-1))
    if (not len(xs) or xs.shape != weights.shape or not torch.isfinite(xs).all()
            or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('recall_invalid_positive_observation')
    return xs, weights / weights.sum()


def _windows(windows):
    result = []
    for pair in sorted(windows or []):
        if len(pair) != 2 or not all(math.isfinite(float(x)) for x in pair) or pair[0] >= pair[1]:
            raise ValueError('recall_invalid_explicit_window')
        a, b = map(float, pair)
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def directional_kernel_scores(kernel, observation):
    """复用现有离散密度，只按公开正观测与显式窗口评分。"""
    logp, cdf = kernel['logp'], kernel['cdf']
    if logp.ndim != 2 or cdf.shape != (len(logp), logp.shape[1] + 1):
        raise ValueError('recall_invalid_kernel_shape')
    xs, weights = _observation_tensors(observation['xs'], observation.get('weights'),
                                       logp.device, logp.dtype)
    low, step = float(kernel['low']), float(kernel['step'])
    if not step > 0:
        raise ValueError('recall_invalid_kernel_step')
    u = (xs - low) / step
    left = u.floor().long().clamp(0, logp.shape[1] - 1)
    right = (left + 1).clamp_max(logp.shape[1] - 1)
    fraction = (u - left).clamp(0, 1)
    values = logp[:, left] * (1 - fraction) + logp[:, right] * fraction
    values = torch.where(((u >= 0) & (u <= logp.shape[1] - 1))[None],
                         values, values.new_full((), math.log(1e-12)))
    nll = -(values * weights[None]).sum(1)
    windows = _windows(observation.get('windows'))
    if windows:
        contained = torch.zeros_like(xs, dtype=torch.bool)
        mass = logp.new_zeros(len(logp))
        def at(value):
            position = min(max((value - low) / step + .5, 0.), float(logp.shape[1]))
            i = int(math.floor(position)); j = min(i + 1, logp.shape[1])
            return cdf[:, i] * (1 - (position - i)) + cdf[:, j] * (position - i)
        for a, b in windows:
            contained |= (xs >= a) & (xs <= b)
            mass += at(b) - at(a)
        if not contained.all():
            raise ValueError('recall_observation_outside_declared_window')
        nll = nll + mass.clamp_min(1e-12).log()
        nll = torch.where(mass >= 1e-8, nll, nll.new_full((), 1e6))
    return nll


def directional_scores(predicted_ppm, tags, observed_ppm, weights=None, windows=None,
                       sigmas=None, chunk_size=512):
    """分块正观测 KDE；窗口按支持点显式可见性过滤，未知窗口保留全部支持。"""
    if predicted_ppm.ndim != 2 or tags.shape != predicted_ppm.shape:
        raise ValueError('recall_invalid_prediction_shape')
    if not torch.isfinite(predicted_ppm).all() or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('recall_invalid_predictions_or_chunk')
    device = predicted_ppm.device
    # 半精度只用于缓存，指数和密度计算始终至少 float32。
    dtype = torch.float64 if predicted_ppm.dtype == torch.float64 else torch.float32
    xs, weights = _observation_tensors(observed_ppm, weights, device, dtype)
    widths = torch.as_tensor([.2] * 9 if sigmas is None else sigmas, device=device, dtype=dtype)
    if widths.ndim != 1 or not len(widths) or not torch.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError('recall_invalid_sigmas')
    if tags.dtype not in (torch.int32, torch.int64) or (tags < 0).any() or (tags >= len(widths)).any():
        raise ValueError('recall_invalid_local_classes')
    windows = _windows(windows)
    if windows and not torch.stack([(xs >= a) & (xs <= b) for a, b in windows]).any(0).all():
        raise ValueError('recall_observation_outside_declared_window')
    pieces = []
    for start in range(0, len(predicted_ppm), chunk_size):
        centers = predicted_ppm[start:start + chunk_size].to(dtype)
        sigma = widths[tags[start:start + chunk_size].long()]
        visible = torch.ones_like(centers, dtype=torch.bool)
        if windows:
            visible = torch.zeros_like(visible)
            for a, b in windows:
                visible |= (centers >= a) & (centers <= b)
        mass = visible.to(dtype)
        mass /= mass.sum(1, keepdim=True).clamp_min(1)
        log_component = (-.5 * ((xs[None, None] - centers[:, :, None]) / sigma[:, :, None]).square()
                         - sigma[:, :, None].log() - .5 * math.log(2 * math.pi))
        logmass = mass.clamp_min(1e-30).log().masked_fill(mass <= 0, -torch.inf)
        logp = torch.logsumexp(log_component + logmass[:, :, None], 1)
        score = -(logp.clamp_min(math.log(1e-12)) * weights[None]).sum(1)
        pieces.append(torch.where(visible.any(1), score, score.new_full((), 1e6)))
    return torch.cat(pieces)

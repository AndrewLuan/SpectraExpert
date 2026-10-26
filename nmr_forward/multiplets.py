# 功能概述：把实验峰表 annotations 规范化，并在条件足够时生成一阶理论棒状谱。
# 主要职责：MULTIPLICITY_CLASSES 固定分类顺序；normalize_multiplicity 归一名称；
# parse_annotation 校验位移区间、积分和 J，保留原标签与异常；simulate_sticks 按
# 二项式卷积和仪器 MHz 生成 ppm/氢面积，合并简并跃迁；evaluate_matched_annotations
# 对调用方已确定对应关系的峰组评估位移、积分、多重性与 J，不自行进行峰组匹配。
# 代码库关系：供 nmr_forward 的峰表训练/评估使用；test_multiplets.py 验证数学契约。
# 输入输出与副作用：只用标准库，接收字典并返回可 JSON 序列化字典，无 I/O；
# 不猜缺失 J，不从 m/br 生成内部峰，不把理论 sticks 当作实测曲线。J 次序沿用
# 标签中的顺序（dt 首 J 对应 d、第二 J 对应 t），不擅自排序改变物理含义。
# 本模块无法从峰表判断强耦合，所有模拟均明确标记 strong_coupling_unknown。
import math
import re

MULTIPLICITY_CLASSES = (
    's', 'd', 't', 'q', 'dd', 'dt', 'td', 'tt', 'dq', 'qd',
    'quint', 'sept', 'm', 'br', 'other',
)
_SPLITTERS = {'s': (), 'd': (1,), 't': (2,), 'q': (3,), 'dd': (1, 1),
              'dt': (1, 2), 'td': (2, 1), 'tt': (2, 2), 'dq': (1, 3),
              'qd': (3, 1), 'quint': (4,), 'sept': (6,)}
_ALIASES = {'singlet': 's', 'doublet': 'd', 'triplet': 't', 'quartet': 'q',
            'quintet': 'quint', 'quint': 'quint', 'pentet': 'quint',
            'septet': 'sept', 'heptet': 'sept', 'multiplet': 'm',
            'broad': 'br', 'doublet of doublets': 'dd',
            'doublet of triplets': 'dt', 'triplet of doublets': 'td',
            'triplet of triplets': 'tt', 'doublet of quartets': 'dq',
            'quartet of doublets': 'qd'}


def normalize_multiplicity(value):
    """Return one of the fixed classes; broad labels retain broad classification."""
    if not isinstance(value, str):
        return 'other'
    text = re.sub(r'\s+', ' ', value.strip().lower())
    if re.match(r'^(br\.?|broad)(?:\s|$|[sdtqm])', text):
        return 'br'
    text = _ALIASES.get(text, text)
    return text if text in MULTIPLICITY_CLASSES else 'other'


def _number(value):
    """Convert finite scalar numbers without accepting booleans or units."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def parse_annotation(annotation):
    """Canonical label and flags; invalid data are retained as unknown, never guessed."""
    if not isinstance(annotation, dict):
        annotation = {}
    flags = []
    mid, low, high = (_number(annotation.get(k)) for k in
                      ('shift_mid', 'shift_min', 'shift_max'))
    if mid is None and low is not None and high is not None and low <= high:
        mid = (low + high) / 2
        flags.append('midpoint_from_interval')
    if mid is None:
        flags.append('invalid_shift')
    if low is None:
        low = mid
    if high is None:
        high = mid
    if low is not None and high is not None and (
            low > high or (mid is not None and not low <= mid <= high)):
        flags.append('invalid_shift_interval')
    area = _number(annotation.get('integration_h'))
    if area is None or area <= 0:
        flags.append('missing_or_invalid_integration')
        area = None
    raw_multiplicity = annotation.get('multiplicity')
    multiplicity = normalize_multiplicity(raw_multiplicity)
    if multiplicity == 'other':
        flags.append('unparsed_multiplicity')
    raw_j = annotation.get('coupling_hz')
    if raw_j is None:
        raw_j = []
    if not isinstance(raw_j, (tuple, list)):
        raw_j = []
        flags.append('invalid_coupling_format')
    coupling = [_number(v) for v in raw_j]
    if any(v is None for v in coupling):
        flags.append('invalid_coupling_value')
    expected = len(_SPLITTERS.get(multiplicity, ()))
    if multiplicity in _SPLITTERS and len(coupling) < expected:
        flags.append('missing_J')
    if multiplicity in _SPLITTERS and len(coupling) > expected:
        flags.append('extra_J')
    return {'shift_mid': mid, 'shift_min': low, 'shift_max': high,
            'integration_h': area, 'multiplicity': multiplicity,
            'coupling_hz': coupling, 'raw_multiplicity': raw_multiplicity,
            'flags': flags}


def simulate_sticks(annotation, frequency_mhz, merge_tolerance_ppm=1e-10):
    """Generate first-order theoretical sticks, conserving supplied integration.

    Return no sticks for insufficient labels. No linewidth, empirical intensity,
    strong-coupling correction or missing J is inferred. Signed J is accepted;
    first-order stick patterns cannot determine its sign.
    """
    label = parse_annotation(annotation)
    flags = list(label['flags']) + ['strong_coupling_unknown']
    frequency = _number(frequency_mhz)
    tolerance = _number(merge_tolerance_ppm)
    if tolerance is None or tolerance < 0:
        raise ValueError('merge_tolerance_ppm must be finite and nonnegative')
    if frequency is None or frequency <= 0:
        flags.append('missing_or_invalid_frequency')
    if label['multiplicity'] not in _SPLITTERS:
        flags.append('unsupported_internal_pattern')
    result = {'representation': 'theoretical_first_order_sticks',
              'x_ppm': [], 'area_h': [], 'flags': flags, 'label': label,
              'frequency_mhz': frequency}
    allowed = {'midpoint_from_interval', 'strong_coupling_unknown'}
    if any(flag not in allowed for flag in flags):
        return result
    sticks = [(0.0, label['integration_h'])]
    for n, coupling in zip(_SPLITTERS[label['multiplicity']], label['coupling_hz']):
        sticks = [(offset + (k - n / 2) * coupling / frequency,
                   area * math.comb(n, k) / (2 ** n))
                  for offset, area in sticks for k in range(n + 1)]
    sticks.sort()
    merged = []
    for offset, area in sticks:
        if merged and abs(offset - merged[-1][0]) <= tolerance:
            old_offset, old_area = merged[-1]
            merged[-1] = ((old_offset * old_area + offset * area) / (old_area + area),
                          old_area + area)
        else:
            merged.append((offset, area))
    result['x_ppm'] = [label['shift_mid'] + offset for offset, _ in merged]
    result['area_h'] = [area for _, area in merged]
    return result


def evaluate_matched_annotations(predictions, targets):
    """Evaluate already matched groups; masks and counts expose missing labels.

    J MAE is evaluated only for the same supported multiplicity, complete J lists,
    and finite values. Its comparison uses absolute J because a first-order
    pattern cannot identify signs; dd and tt allow exchange of identical stages.
    """
    if len(predictions) != len(targets):
        raise ValueError('Predictions and targets must be explicitly matched first')
    errors = {'shift_mae_ppm': [], 'integration_mae_h': [], 'J_magnitude_mae_hz': []}
    correct = total = 0
    for pred, target in zip(predictions, targets):
        p, t = parse_annotation(pred), parse_annotation(target)
        for field, metric in [('shift_mid', 'shift_mae_ppm'),
                              ('integration_h', 'integration_mae_h')]:
            if p[field] is not None and t[field] is not None:
                errors[metric].append(abs(p[field] - t[field]))
        if t['multiplicity'] != 'other':
            total += 1
            correct += p['multiplicity'] == t['multiplicity']
        m = t['multiplicity']
        expected = len(_SPLITTERS.get(m, ()))
        if expected and p['multiplicity'] == m and all(
                len(z['coupling_hz']) == expected and
                all(j is not None for j in z['coupling_hz']) for z in (p, t)):
            pj, tj = [abs(j) for j in p['coupling_hz']], [abs(j) for j in t['coupling_hz']]
            if m in ('dd', 'tt'):
                pj.sort()
                tj.sort()
            errors['J_magnitude_mae_hz'].extend(abs(a-b) for a, b in zip(pj, tj))
    result = {key: sum(values) / len(values) if values else None
              for key, values in errors.items()}
    result['counts'] = {key: len(values) for key, values in errors.items()}
    result['multiplicity_accuracy'] = correct / total if total else None
    result['counts']['multiplicity'] = total
    result['matched_groups'] = len(targets)
    return result

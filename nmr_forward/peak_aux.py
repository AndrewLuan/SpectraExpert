# 功能概述：对无序氢 slot 提供峰组多重性及局部 J 的辅助监督，不声称原子归属。
# 主要职责：peak_aux_loss 先验证每条记录的正整数积分及结构氢数一致性，将峰组
# 复制到每个氢，按位移与预测有效 slot 排序匹配，再计算多重性 CE 和 J Huber。
# 代码库关系：类别沿用 multiplets.MULTIPLICITY_CLASSES；输入来自 GNN 输出和
# batch['rows'] 的 annotations/protons/integral，由训练入口决定辅助损失权重。
# 输入输出：shifts/proton_mask[B,N,4]；可选 mult_logits[B,N,4,C]、
# couplings[B,N,4,3]。返回 loss、multiplicity_loss、j_loss 张量及 counts 字典。
# J 是峰组局部劈裂大小，不是核对耦合矩阵；缺失值不补零，m/br/other不监督J。
# 非空J列表长度必须符合多重性的阶段数；dd/tt在两阶段间最小Huber置换匹配，
# dt/td等保持阶段顺序；显式None保留缺失掩码。缺失/未解析多重性不参与CE或accuracy。
# counts报告已知/未知多重性、J缺失/格式/数量矛盾和实际监督覆盖；默认CE+0.05*Huber。
# 标签先在 CPU 构造；整批排序和向量 gather，避免按 slot 发起 GPU 索引 kernel。
# 返回 detached multiplicity_correct/mult_slots/j_abs_error_sum/j_values 用于验证汇总。
# joint_multiplicity_correct_at_0_1ppm仅计位移误差<=0.1且类别正确的已知槽；
# joint_J_abs_error_sum/joint_J_values在相同联合条件下累计J误差与覆盖数，分母覆盖
# 参照原有mult_slots/j_values；联合统计不修改训练损失，缺多重性头时联合计数为零。
# 排序索引不可微；梯度仅传向匹配后的辅助预测。重合位移使用稳定排序，内部
# 顺序无物理可辨识性。无文件、网络或全局状态副作用；非法形状立即报错。
import math
import torch
import torch.nn.functional as F
try:
    from .multiplets import MULTIPLICITY_CLASSES, normalize_multiplicity
except ImportError:
    from multiplets import MULTIPLICITY_CLASSES, normalize_multiplicity


_J_STAGES = {'s': 0, 'd': 1, 't': 1, 'q': 1, 'dd': 2, 'dt': 2,
             'td': 2, 'tt': 2, 'dq': 2, 'qd': 2, 'quint': 1, 'sept': 1}


def _finite(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def peak_aux_loss(output, batch, j_weight=0.05):
    """Return auxiliary loss components and explicit supervision/skip counts."""
    shifts, mask = output['shifts'], output['proton_mask'].bool()
    if shifts.ndim != 3 or shifts.shape[-1] != 4 or mask.shape != shifts.shape:
        raise ValueError('Expected shifts and proton_mask shaped [B,N,4]')
    rows = batch['rows']
    if len(rows) != shifts.shape[0]:
        raise ValueError('rows must match batch dimension')
    logits, coupling = output.get('mult_logits'), output.get('couplings')
    if logits is not None and logits.shape != (*shifts.shape, len(MULTIPLICITY_CLASSES)):
        raise ValueError('mult_logits has incompatible shape')
    if coupling is not None and coupling.shape != (*shifts.shape, 3):
        raise ValueError('couplings has incompatible shape')
    # 零损失仍连到可用输出，允许整批均无监督时正常 backward。
    zero = shifts.reshape(-1)[:0].sum()
    if logits is not None:
        zero = zero + logits.reshape(-1)[:0].sum()
    if coupling is not None:
        zero = zero + coupling.reshape(-1)[:0].sum()
    counts = dict(records=len(rows), eligible_records=0, skipped_integral=0,
                  skipped_annotations=0, skipped_slot_count=0,
                  skipped_nonfinite_shifts=0, mult_slots=0, j_values=0,
                  j_overflow_groups=0, invalid_j_values=0, unknown_mult_groups=0,
                  annotation_groups=0, known_mult_groups=0, unknown_mult_slots=0,
                  j_supported_groups=0, j_supervised_groups=0, j_missing_groups=0,
                  j_count_mismatch_groups=0, j_invalid_format_groups=0)
    # 一次读取整批资格统计；无按 slot 的 GPU->CPU 同步。
    flat_mask = mask.flatten(1)
    flat_shifts = shifts.detach().flatten(1)
    metadata = torch.stack((flat_mask.sum(1),
                           ((~torch.isfinite(flat_shifts)) & flat_mask).sum(1)), 1).cpu().tolist()
    ordering = torch.argsort(flat_shifts.masked_fill(~flat_mask, float('inf')),
                            dim=1, stable=True)
    positions, labels = [], []
    known_j_count = 0
    mult_slot_count = 0
    for i, row in enumerate(rows):
        annotations = row.get('annotations')
        if not isinstance(annotations, (list, tuple)) or not annotations:
            counts['skipped_annotations'] += 1
            continue
        protons, integral = row.get('protons'), row.get('integral')
        if not _finite(protons) or float(protons) <= 0 or not float(protons).is_integer():
            counts['skipped_integral'] += 1
            continue
        expanded = []
        invalid = False
        for a in annotations:
            if not isinstance(a, dict):
                invalid = True
                break
            area, shift = a.get('integration_h'), a.get('shift_mid')
            if not _finite(area) or float(area) <= 0 or not float(area).is_integer():
                invalid = True
                break
            if not _finite(shift):
                invalid = True
                break
            if len(expanded) + int(float(area)) > int(protons):
                invalid = True
                break
            expanded.extend([(float(shift), a)] * int(float(area)))
        if (invalid or len(expanded) != int(protons) or not _finite(integral)
                or not math.isclose(float(integral), float(protons), rel_tol=0, abs_tol=1e-6)):
            counts['skipped_integral'] += 1
            continue
        if metadata[i][0] != len(expanded):
            counts['skipped_slot_count'] += 1
            continue
        if metadata[i][1]:
            counts['skipped_nonfinite_shifts'] += 1
            continue
        expanded.sort(key=lambda item: item[0])
        counts['eligible_records'] += 1
        # 原始峰组计数，不按积分复制倍增异常计数。
        for a in annotations:
            counts['annotation_groups'] += 1
            multiplicity = normalize_multiplicity(a.get('multiplicity'))
            if multiplicity == 'other':
                counts['unknown_mult_groups'] += 1
                counts['unknown_mult_slots'] += int(float(a['integration_h']))
            else:
                counts['known_mult_groups'] += 1
            js = a.get('coupling_hz')
            if isinstance(js, (list, tuple)):
                counts['j_overflow_groups'] += len(js) > 3
                counts['invalid_j_values'] += sum(not _finite(j) for j in js)
            if multiplicity in _J_STAGES:
                expected = _J_STAGES[multiplicity]
                counts['j_supported_groups'] += expected > 0
                if js is None or (isinstance(js, (list, tuple)) and not js):
                    counts['j_missing_groups'] += expected > 0
                elif not isinstance(js, (list, tuple)):
                    counts['j_invalid_format_groups'] += 1
                elif len(js) != expected:
                    counts['j_count_mismatch_groups'] += 1
                elif coupling is not None and any(_finite(j) for j in js):
                    counts['j_supervised_groups'] += 1
        for rank, (target_shift, a) in enumerate(expanded):
            multiplicity = normalize_multiplicity(a.get('multiplicity'))
            known_mult = multiplicity != 'other'
            mult_slot_count += int(known_mult and logits is not None)
            # 类别、三个J、三个J掩码、多重性掩码、两阶段可交换标记、目标位移。
            label = [float(MULTIPLICITY_CLASSES.index(multiplicity)), 0., 0., 0., 0., 0., 0.,
                     float(known_mult), float(multiplicity in ('dd', 'tt')), target_shift]
            js = a.get('coupling_hz')
            if (coupling is not None and multiplicity in _J_STAGES
                    and isinstance(js, (list, tuple)) and len(js) == _J_STAGES[multiplicity]):
                for k, value in enumerate(js):
                    if _finite(value):
                        label[1 + k] = abs(float(value))
                        label[4 + k] = 1.
                        known_j_count += 1
            positions.append((i, rank))
            labels.append(label)
    mult_loss, j_loss = zero, zero
    correct = torch.zeros((), dtype=torch.long, device=shifts.device)
    abs_error = torch.zeros((), dtype=shifts.dtype, device=shifts.device)
    joint_correct = correct.clone()
    joint_j_values = correct.clone()
    joint_j_abs_error = abs_error.clone()
    if positions and (logits is not None or coupling is not None):
        # 各种标签合并一次传输；索引同样整批传输，不遍历 CUDA 标量。
        position = torch.tensor(positions, dtype=torch.long, device=shifts.device)
        target = torch.tensor(labels, dtype=shifts.dtype, device=shifts.device)
        indices = ordering[position[:, 0], position[:, 1]]
        flat_indices = position[:, 0] * ordering.shape[1] + indices
        joint_mask = torch.zeros(len(labels), dtype=torch.bool, device=shifts.device)
        if logits is not None and mult_slot_count:
            mult_mask = target[:, 7].bool()
            pred = logits.reshape(-1, len(MULTIPLICITY_CLASSES))[flat_indices][mult_mask]
            classes = target[mult_mask, 0].long()
            mult_loss = F.cross_entropy(pred, classes)
            correct = (pred.detach().argmax(-1) == classes).sum()
            matched_shifts = flat_shifts.reshape(-1)[flat_indices]
            joint_mask[mult_mask] = ((pred.detach().argmax(-1) == classes)
                                    & ((matched_shifts[mult_mask] - target[mult_mask, 9]).abs() <= 0.1))
            joint_correct = joint_mask.sum()
        if coupling is not None and known_j_count:
            pred = coupling.reshape(-1, 3)[flat_indices]
            jmask = target[:, 4:7].bool()
            expected = target[:, 1:4]
            # 同类劈裂阶段可交换；同时交换目标及其缺失掩码，避免补造缺失J。
            swapped = expected[:, [1, 0, 2]]
            swapped_mask = jmask[:, [1, 0, 2]]
            direct_loss = (F.huber_loss(pred, expected, reduction='none', delta=1.0) * jmask).sum(1)
            swapped_loss = (F.huber_loss(pred, swapped, reduction='none', delta=1.0) * swapped_mask).sum(1)
            use_swap = target[:, 8].bool() & (swapped_loss < direct_loss)
            expected = torch.where(use_swap[:, None], swapped, expected)
            jmask = torch.where(use_swap[:, None], swapped_mask, jmask)
            joint_jmask = jmask & joint_mask[:, None]
            joint_j_abs_error = (pred.detach() - expected).abs()[joint_jmask].sum()
            joint_j_values = joint_jmask.sum()
            selected = pred[jmask]
            expected = expected[jmask]
            j_loss = F.huber_loss(selected, expected, delta=1.0)
            abs_error = (selected.detach() - expected).abs().sum()
    counts['mult_slots'] = mult_slot_count
    counts['j_values'] = known_j_count
    return dict(loss=mult_loss + j_weight * j_loss,
                multiplicity_loss=mult_loss, j_loss=j_loss, counts=counts,
                multiplicity_correct=correct.detach(), mult_slots=counts['mult_slots'],
                j_abs_error_sum=abs_error.detach(), j_values=counts['j_values'],
                joint_multiplicity_correct_at_0_1ppm=joint_correct.detach(),
                joint_J_abs_error_sum=joint_j_abs_error.detach(),
                joint_J_values=joint_j_values.detach())

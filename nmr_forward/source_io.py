"""功能概述：统一 NMRexp 与 WanJuan 峰表字段，供 prepare.convert 读取。
主要职责：normalize_record 返回 (副本, None) 或 (None, 排除原因)；_number
核验有限数值，_frequency 解析 MHz，_restore_stereo 仅在非立体结构完全一致时
恢复 provenance.smiles_std 的立体信息。区间使用中点并保留两端与原始证据。
关系与边界：训练只读取规范化峰表，不将积分当峰高、不补造缺失 J 或完整曲线。
输入输出：不改原记录，不访问文件；保留来源及实验条件，新增 normalization_flags
与 normalization_evidence。无效核种、数组或注释明确排除；缺失频率仅标记。
"""
import copy
import math
import re


def _number(value):
    if isinstance(value, bool):
        raise ValueError('boolean_numeric')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite_numeric')
    return result


def _frequency(value):
    if value is None or value == '':
        return None
    match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(?:MHz)?\s*', str(value), re.I)
    if not match:
        return None
    result = float(match.group(1))
    return result if result > 0 else None


def _restore_stereo(out, flags, evidence):
    candidate = (out.get('provenance') or {}).get('smiles_std')
    original = out.get('smiles')
    if not candidate or candidate == original:
        return
    from rdkit import Chem
    a, b = Chem.MolFromSmiles(original or ''), Chem.MolFromSmiles(candidate)
    if a is None or b is None:
        flags.append('unverifiable_provenance_structure')
        return
    if Chem.MolToSmiles(a, isomericSmiles=False) != Chem.MolToSmiles(b, isomericSmiles=False):
        flags.append('provenance_structure_differs')
        return
    # 若两边都带立体描述而彼此冲突，不静默选择其中一个。
    original_iso = Chem.MolToSmiles(a, isomericSmiles=True)
    candidate_iso = Chem.MolToSmiles(b, isomericSmiles=True)
    if any(c in original for c in '@/\\') and original_iso != candidate_iso:
        flags.append('provenance_stereochemistry_conflict')
        return
    out['smiles'] = candidate_iso
    evidence['structure'] = {'original_smiles': original, 'selected': 'provenance.smiles_std',
                             'nonstereo_canonical_identity_verified': True}


def normalize_record(record):
    """规范化一个 JSON 记录；返回值沿用 prepare.convert 的 (记录, 原因) 契约。"""
    out = copy.deepcopy(record)
    flags, evidence = [], {}
    conditions = out.get('experimental_conditions') or {}
    top, nested = out.get('nucleus'), conditions.get('nucleus')
    if top and nested and top != nested:
        return None, 'conflicting_nucleus'
    out['nucleus'] = top or nested
    if out['nucleus'] != '1H':
        return None, 'not_1H'
    evidence['nucleus_field'] = 'nucleus' if top else 'experimental_conditions.nucleus'
    if out.get('representation') != 'integral_peak_spectrum':
        return None, 'not_integral_peak_spectrum'
    raw_frequency = out.get('frequency') or conditions.get('frequency')
    out['frequency'] = _frequency(raw_frequency)
    if out['frequency'] is None:
        flags.append('missing_frequency' if not raw_frequency else 'unparseable_frequency')
    evidence['raw_frequency'] = raw_frequency
    out['solvent'] = out.get('solvent') or conditions.get('solvent')
    annotations = out.get('annotations') or []
    value = out.get('value') or {}
    xs, ys = value.get('x', []), value.get('y', [])
    if not xs or len(xs) != len(ys):
        return None, 'invalid_arrays'
    if annotations and len(annotations) != len(xs):
        return None, 'annotation_array_length_mismatch'
    normalized = []
    try:
        for i, (x, y) in enumerate(zip(xs, ys)):
            a = copy.deepcopy(annotations[i]) if annotations else {}
            weight = _number(y)
            if weight <= 0:
                return None, 'invalid_integral'
            if a.get('integration_h') is not None and abs(_number(a['integration_h']) - weight) > 1e-6:
                return None, 'annotation_integral_mismatch'
            if a.get('shift') is not None:
                start = _number(a['shift'])
                end = _number(a['shift_end']) if a.get('shift_end') is not None else start
            elif a.get('shift_min') is not None and a.get('shift_max') is not None:
                start, end = _number(a['shift_min']), _number(a['shift_max'])
            else:
                start = end = _number(a.get('shift_mid', x))
            a.update(shift_min=min(start, end), shift_max=max(start, end),
                     shift_mid=(start + end) / 2, integration_h=weight)
            raw_j = a.get('coupling_hz', a.get('j_hz', [])) or []
            a['coupling_hz'] = [_number(j) for j in raw_j]
            a.setdefault('multiplicity', None)
            if not raw_j and re.search(r'\bJ\s*=', a.get('raw') or '', re.I):
                flags.append('raw_J_present_but_unparsed')
            normalized.append(a)
    except (ValueError, TypeError, OverflowError):
        return None, 'invalid_annotation_numeric'
    out['annotations'] = normalized
    out['value'] = {'x': [a['shift_mid'] for a in normalized],
                    'y': [a['integration_h'] for a in normalized]}
    evidence['original_value'] = copy.deepcopy(value)
    evidence['interval_policy'] = 'midpoint_preserving_endpoints'
    _restore_stereo(out, flags, evidence)
    locator = copy.deepcopy(out.get('source_locator') or {})
    if out.get('provenance'):
        locator['provenance'] = copy.deepcopy(out['provenance'])
    out['source_locator'] = locator
    out['normalization_flags'] = sorted(set(flags))
    out['normalization_evidence'] = evidence
    return out, None

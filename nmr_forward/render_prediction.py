# 功能概述：把predict.py的二维结构预测JSON转换为一阶理论棒谱和可选Lorentzian曲线。
# 主要职责：render_prediction按预测类别截取所需J候选并调用multiplets.simulate_sticks；
# lorentzian_curve使用用户指定FWHM与ppm网格生成理论曲线；write_outputs拒绝覆盖并
# 原子写各文件；main提供CLI，保留输入JSON哈希、checkpoint SHA与全部近似假设。
# 代码库关系：只依赖标准库与multiplets，不读取标签/训练缓存，不加载模型或远端资源。
# 输入输出与边界：频率必须由JSON或参数明确提供；m/br/other、缺J等列入unresolved，
# 不造中心占位峰；面积优先采用predicted_reported_integration_h，否则用结构积分。
# 每条Lorentzian在无限域积分等于棒面积；有限窗口不重归一化，另报告覆盖面积。
# J候选不是核对矩阵，不处理强耦合/交换，指定线宽不是学习结果，也不是实测完整谱。
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

try:
    from .multiplets import _SPLITTERS, normalize_multiplicity, simulate_sticks
except ImportError:
    from multiplets import _SPLITTERS, normalize_multiplicity, simulate_sticks


def _finite(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return number


def render_prediction(prediction, frequency_mhz=None):
    """渲染支持的位点；未解决位点保留面积和原因，返回可JSON序列化字典。"""
    if prediction.get('nucleus') != '1H':
        raise ValueError('Only 1H predictions are supported')
    supplied = (prediction.get('input') or {}).get('frequency_mhz')
    frequency = _finite(frequency_mhz if frequency_mhz is not None else supplied,
                        'frequency_mhz', positive=True)
    checkpoint = prediction.get('checkpoint') or {}
    if not checkpoint.get('sha256'):
        raise ValueError('Prediction must retain checkpoint.sha256')
    sites = prediction.get('proton_sites')
    if not isinstance(sites, list):
        raise ValueError('Prediction must contain proton_sites list')
    sticks, unresolved, resolved = [], [], []
    total = 0.
    for index, site in enumerate(sites):
        site_id = site.get('site_id', f'site_{index}')
        shift = _finite(site.get('shift_ppm'), f'{site_id}.shift_ppm')
        area_field = ('predicted_reported_integration_h'
                      if 'predicted_reported_integration_h' in site else 'structural_integration_h')
        area = _finite(site.get(area_field), f'{site_id}.{area_field}')
        if area < 0:
            raise ValueError(f'{site_id} has negative integration')
        total += area
        multiplicity = normalize_multiplicity(site.get('multiplicity_argmax'))
        record = dict(site_id=site_id, shift_ppm=shift, area_h=area,
                      area_source=area_field, multiplicity=multiplicity)
        if multiplicity not in _SPLITTERS:
            unresolved.append(dict(record, reasons=['unsupported_or_unknown_multiplicity']))
            continue
        required = len(_SPLITTERS[multiplicity])
        candidates = site.get('peak_group_j_candidates_hz', [])
        if not isinstance(candidates, (list, tuple)) or len(candidates) < required:
            unresolved.append(dict(record, reasons=['insufficient_J_candidates']))
            continue
        try:
            js = [abs(_finite(j, f'{site_id}.J')) for j in candidates[:required]]
        except ValueError:
            unresolved.append(dict(record, reasons=['invalid_required_J_candidate']))
            continue
        if area == 0:
            resolved.append(dict(record, coupling_hz=js, sticks_count=0))
            continue
        simulated = simulate_sticks(dict(shift_mid=shift, integration_h=area,
                                         multiplicity=multiplicity, coupling_hz=js), frequency)
        if not simulated['x_ppm']:
            unresolved.append(dict(record, reasons=simulated['flags']))
            continue
        resolved.append(dict(record, coupling_hz=js, sticks_count=len(simulated['x_ppm'])))
        sticks.extend(dict(site_id=site_id, ppm=x, area_h=w)
                      for x, w in zip(simulated['x_ppm'], simulated['area_h']))
    sticks.sort(key=lambda s: s['ppm'])
    return dict(schema_version=1, representation='theoretical_first_order_sticks',
                source_checkpoint_sha256=checkpoint['sha256'], frequency_mhz=frequency,
                frequency_source='explicit_override' if frequency_mhz is not None else 'prediction.input',
                prediction_frequency_mhz=supplied,
                source_model_boundaries=prediction.get('model_boundaries', []),
                assumptions=['first_order_weak_coupling_only_strong_coupling_unknown',
                             'multiplicity_argmax_selects_pattern_without_uncertainty_propagation',
                             'first_N_J_candidates_selected_by_pattern_unused_candidates_discarded',
                             'J_magnitudes_are_local_pattern_parameters_not_nuclear_pair_matrix',
                             'predicted_reported_area_preferred_else_structural_unit_area',
                             'unresolved_sites_excluded_no_placeholder_peaks',
                             'independent_latent_sites_summed_not_validated_atom_assignments',
                             'no_exchange_or_relaxation_simulation_not_measured_spectrum'],
                input_area_h=total, resolved_area_h=sum(s['area_h'] for s in sticks),
                unresolved_area_h=sum(s['area_h'] for s in unresolved),
                resolved_sites=resolved, unresolved=unresolved, sticks=sticks,
                all_sites_have_supported_first_order_patterns=not unresolved)


def lorentzian_curve(rendered, linewidth_hz, ppm_min, ppm_max, points=16384):
    """FWHM使用Hz；返回(ppm,intensity_h_per_ppm)列表及有限窗口积分元数据。"""
    linewidth = _finite(linewidth_hz, 'linewidth_hz', positive=True)
    low, high = _finite(ppm_min, 'ppm_min'), _finite(ppm_max, 'ppm_max')
    if high <= low or isinstance(points, bool) or not isinstance(points, int) or points < 2:
        raise ValueError('Need ppm_max > ppm_min and integer points >= 2')
    frequency = _finite(rendered['frequency_mhz'], 'frequency_mhz', positive=True)
    gamma = linewidth / frequency / 2  # 半高半宽，ppm。
    step = (high-low)/(points-1)
    curve = []
    for index in range(points):
        x = low+index*step
        y = sum(s['area_h']*gamma/(math.pi*((x-s['ppm'])**2+gamma**2))
                for s in rendered['sticks'])
        curve.append((x,y))
    analytic = sum(s['area_h']/math.pi*(math.atan((high-s['ppm'])/gamma)
                   - math.atan((low-s['ppm'])/gamma)) for s in rendered['sticks'])
    numerical = step*(sum(y for _,y in curve)-(curve[0][1]+curve[-1][1])/2)
    return curve, dict(representation='theoretical_Lorentzian_resolved_sites_only',
                       linewidth_fwhm_hz=linewidth, linewidth_source='user_specified_not_learned',
                       ppm_min=low, ppm_max=high, points=points, step_ppm=step,
                       kernel_normalization='unit_area_on_infinite_ppm_domain_no_window_renormalization',
                       resolved_infinite_area_h=rendered['resolved_area_h'],
                       analytic_window_area_h=analytic, trapezoid_grid_area_h=numerical,
                       points_per_fwhm=(2*gamma)/step,
                       warnings=['grid_underresolves_linewidth'] if step > gamma/2 else [])


def _write_new(path, text):
    """同目录临时文件加硬链接发布；目标已存在时原子失败，不覆写。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_outputs(rendered, out, curve_csv=None, linewidth_hz=None, ppm_min=None,
                  ppm_max=None, points=16384):
    """验证所有参数后写新JSON及可选CSV；JSON写入失败时回滚本次创建的CSV。"""
    out = Path(out)
    csv_path = Path(curve_csv) if curve_csv is not None else None
    if csv_path is not None and csv_path.resolve() == out.resolve():
        raise ValueError('JSON and CSV paths must differ')
    for path in (out,csv_path):
        if path is not None and path.exists():
            raise FileExistsError(path)
    result = dict(rendered)
    csv_text = None
    if csv_path is not None:
        curve, metadata = lorentzian_curve(rendered,linewidth_hz,ppm_min,ppm_max,points)
        result['curve'] = dict(metadata, csv_path=str(csv_path.resolve()))
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(['ppm','theoretical_intensity_h_per_ppm'])
        writer.writerows(curve)
        csv_text = buffer.getvalue()
    elif linewidth_hz is not None or ppm_min is not None or ppm_max is not None:
        raise ValueError('Curve parameters require curve_csv')
    encoded = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    wrote_csv = False
    try:
        if csv_text is not None:
            _write_new(csv_path,csv_text)
            wrote_csv = True
        _write_new(out,encoded)
    except Exception:
        if wrote_csv:
            csv_path.unlink()
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description='Render predicted NMR sites as theoretical first-order spectra')
    parser.add_argument('--prediction',required=True,help='predict.py output JSON')
    parser.add_argument('--out',required=True,help='new JSON output')
    parser.add_argument('--frequency',type=float,help='explicit override in MHz; otherwise use prediction JSON')
    parser.add_argument('--curve-csv',help='optional new theory curve CSV')
    parser.add_argument('--linewidth-hz',type=float,help='user-specified Lorentzian FWHM, required for CSV')
    parser.add_argument('--ppm-min',type=float)
    parser.add_argument('--ppm-max',type=float)
    parser.add_argument('--points',type=int,default=16384)
    args = parser.parse_args()
    raw = Path(args.prediction).read_bytes()
    rendered = render_prediction(json.loads(raw),args.frequency)
    rendered['source_prediction_sha256'] = hashlib.sha256(raw).hexdigest()
    rendered['source_prediction_path'] = str(Path(args.prediction).resolve())
    result = write_outputs(rendered,args.out,args.curve_csv,args.linewidth_hz,
                           args.ppm_min,args.ppm_max,args.points)
    print(json.dumps(dict(out=str(Path(args.out).resolve()),sticks=len(result['sticks']),
                          unresolved_sites=len(result['unresolved']))))


if __name__ == '__main__':
    main()

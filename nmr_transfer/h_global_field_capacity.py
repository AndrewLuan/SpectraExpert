"""功能概述：仅在固定开发验证集检验公开场强条件化的 Hz 容量假设。

frequency_mhz 只解析 HMDB 原表 experimental_conditions.frequency 中带 MHz 单位的
正数；不从谱位、结构、旧条件向量或默认仪器推断。configs 在读指标前固定
16/24/32 Hz × gamma .25/1，观测与预测用同一 width_Hz/nu_MHz 聚簇。
field_capacity 沿用已审核的单次容量 DP，sigma=.2 ppm、污染阈值=.5 ppm 不变；
未知场强、非 HMDB 和显式频窗修正为 0，不把离散细线跨度称为实测 FWHM。

read_hmdb_records 只解析所需开发记录 ID 的原始 JSONL 行。main 只接受
original_val/tune；非 HMDB 逐条复用旧 study/report_g1.0，HMDB 对完整 114982
候选评分并强制 control 与旧 report1 的 rank/Top10 相同。每个配置报告整体及
HMDB 的 Top1/5/10、MRR；冻结原始输入、参考结果、源码、候选和查询 SHA。
load_reference 同时核对旧 study 的 cohort、候选顺序、查询和模型摘要，防止误用
其他 seed、其他报告强度或旧查询的结果作为非 HMDB 基座。
本模块不打开 verify/test、不训练、不启动子任务、不改写旧 cache/模型或外部服务。
GPU 仅由调用者显式运行 main 时使用；辅助函数和合成单测支持 CPU。
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from .h_condinfer import condition_key
from .h_eval import LazyFP
from .h_global_bank import load_bank
from .h_global_capacity import capacity_excess, cluster_predictions
from .h_global_extract import Scorer
from .h_global_protocol import actual_inputs, scoring_inputs_paths
from .h_global_scoring import configs as coarse_configs
from .h_observation_finalize import visible_lines
from .h_observation_study import entry, stamp_queries, summarize_sources, uniform_peaks, write_rows
from .h_observed import cluster_lines
from .prepare import sha


WIDTHS_HZ = (16., 24., 32.)
GAMMAS = (.25, 1.)
SIGMA_PPM = .2
OUTLIER_PPM = .5
FREQUENCY_FIELD = 'experimental_conditions.frequency'


def configs():
    """返回预声明的 report1 控制和六个统一场强容量干预。"""
    return [dict(name='report_control', width_Hz=0., gamma=0.)] + [
        dict(name=f'field_hz{int(width)}_g{gamma}', width_Hz=width, gamma=gamma)
        for width in WIDTHS_HZ for gamma in GAMMAS
    ]


def frequency_mhz(record):
    """只接受原记录公开的显式 MHz 字符串；缺失、非正或未注明单位返回 None。"""
    value = (record.get('experimental_conditions') or {}).get('frequency')
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*MHz\s*', value, re.IGNORECASE)
    if match is None:
        return None
    number = float(match.group(1))
    return number if math.isfinite(number) and number > 0 else None


def resolution_ppm(record, width_Hz):
    """按 ppm × MHz = Hz 换算统一分辨率；不提供未知场强的默认值。"""
    if not math.isfinite(width_Hz) or width_Hz <= 0:
        raise ValueError('field_capacity_invalid_width_Hz')
    frequency = frequency_mhz(record)
    return None if frequency is None else width_Hz / frequency


def read_hmdb_records(path, queries):
    """扫描原文件但只解析当前开发 HMDB ID；禁止静默重复或缺失记录。"""
    wanted = {row['record_id'] for row in queries if row['source'] == 'HMDB'}
    if not wanted:
        return {}
    encoded = {value.encode() for value in wanted}
    pattern = re.compile(rb'"record_id"\s*:\s*"([^"\\]+)"')
    result = {}
    with Path(path).open('rb') as stream:
        for line in stream:
            match = pattern.search(line)
            if match is None or match.group(1) not in encoded:
                continue
            row = json.loads(line)
            record_id = row['record_id']
            if row.get('source') != 'HMDB' or row.get('nucleus') != '1H':
                raise ValueError('field_capacity_raw_record_scope_changed')
            if record_id in result:
                raise ValueError('field_capacity_duplicate_raw_record')
            result[record_id] = row
    if result.keys() != wanted:
        raise ValueError('field_capacity_missing_raw_development_record')
    return result


@torch.inference_mode()
def field_capacity(predictions, query, record, width_Hz, cluster_cache=None):
    """返回每候选容量差值；缓存必须仅在同一条件预测张量内复用。

    predictions 是原 bank 的归一化 128 分位点，乘 10 转 ppm。窗口或未知场强
    精确返回 0；HMDB 只读取真实可见细线，不赋峰高积分、不读正确候选身份。
    """
    result = predictions.new_zeros(len(predictions))
    if query['source'] != 'HMDB' or query['observation']['windows']:
        return result
    width = resolution_ppm(record, width_Hz)
    if width is None:
        return result
    cache = {} if cluster_cache is None else cluster_cache
    key = (float(width_Hz), frequency_mhz(record))
    if key not in cache:
        cache[key] = cluster_predictions(uniform_peaks(predictions) * 10, width)
    centers, valid = cache[key]
    lines = visible_lines(query, {query['record_id']: record})
    observed, _ = cluster_lines(lines, width)
    return capacity_excess(centers, valid, observed, sigma=SIGMA_PPM, outlier=OUTLIER_PPM)


def _identity(row):
    return row['source'], row['record_id'], row['smiles']


def load_reference(path, queries, cohort, order_sha256, checkpoint_sha256):
    """严格读取对应旧 report1；来源摘要和逐查询身份必须同时匹配。"""
    path = Path(path)
    if path.name != 'predictions.jsonl' or path.parent.name != 'report_g1.0':
        raise ValueError('field_capacity_requires_report1_reference')
    summary = json.loads((path.parent.parent / 'summary.json').read_text())
    expected = dict(cohort=cohort, candidate_count=114982,
                    candidate_order_sha256=order_sha256, checkpoint_sha256=checkpoint_sha256,
                    query_sha256=stamp_queries(queries))
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError('field_capacity_reference_provenance_changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    reference = {_identity(row): row for row in rows}
    if (len(reference) != len(rows) or reference.keys() != {_identity(row) for row in queries}
            or any(row.get('variant', 'observed') != 'observed' for row in rows)):
        raise ValueError('field_capacity_reference_development_identity_changed')
    return reference


def main():
    parser = argparse.ArgumentParser()
    for name in ('base', 'bank', 'widths', 'cohorts', 'reference', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cohort', choices=['original_val', 'tune'], required=True)
    parser.add_argument('--extension')
    args = parser.parse_args()
    required = actual_inputs(args.base, args.bank, args.widths, args.cohorts, args.extension)
    required += [Path(args.reference), Path(args.reference).parent.parent / 'summary.json',
                 Path(__file__), Path(__file__).with_name('h_global_capacity.py')]
    required = sorted({path.resolve() for path in required})
    pinned = {str(path): sha(path) for path in required}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    def status(stage, **fields):
        temporary = out / 'status.tmp'
        temporary.write_text(json.dumps(dict(stage=stage, **fields)))
        temporary.replace(out / 'status.json')

    try:
        choices = configs()
        (out / 'plan.json').write_text(json.dumps(dict(configs=choices, inputs_sha256=pinned,
            frequency_field=FREQUENCY_FIELD, sigma_ppm=SIGMA_PPM, outlier_ppm=OUTLIER_PPM,
            baseline='corresponding prior observation study/report_g1.0',
            selection_scope='original_val+tune only; no test or verify', external_api_calls=0), indent=2))
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('high')
        device = torch.device('cuda')
        bank = load_bank(args.bank, args.extension)
        if sha(bank['checkpoint_path']) != bank['checkpoint_sha256']:
            raise ValueError('field_capacity_prediction_checkpoint_changed')
        payload = torch.load(Path(args.cohorts) / 'cohorts.pt', map_location='cpu', weights_only=False)
        queries = payload['cohorts'][args.cohort]
        if len(queries) != (320 if args.cohort == 'original_val' else 1024):
            raise ValueError('field_capacity_development_cohort_changed')
        indices = bank['single_indices']
        smiles = [bank['smiles'][i] for i in indices]
        if len(smiles) != 114982:
            raise ValueError('field_capacity_candidate_count_changed')
        lookup = {value: i for i, value in enumerate(smiles)}
        order_sha256 = hashlib.sha256('\n'.join(smiles).encode()).hexdigest()
        reference = load_reference(args.reference, queries, args.cohort, order_sha256, bank['checkpoint_sha256'])
        paths = scoring_inputs_paths(args.base)
        raw = read_hmdb_records(paths[-1], queries)
        inputs = (json.loads(paths[0].read_text()), json.loads(paths[1].read_text()),
                  json.loads(paths[2].read_text())['config'], json.loads(paths[3].read_text()), raw)
        widths = json.loads(Path(args.widths).read_text())['selected_ppm']
        choice = next(item for item in coarse_configs() if item['name'] == 'atom_count_1.0')
        scorer = Scorer(bank, payload, widths, inputs, choice, device)
        fps = LazyFP()
        records = {item['name']: [] for item in choices}
        hmdb = [(i, row) for i, row in enumerate(queries) if row['source'] == 'HMDB']
        frequency_counts = collections.Counter(str(frequency_mhz(raw[row['record_id']])) for _, row in hmdb)
        for order, row in enumerate(queries):
            if row['source'] != 'HMDB':
                for config in choices:
                    records[config['name']].append(dict(reference[_identity(row)], query_order=order, variant='observed'))
        metadata = payload['metadata']
        for key in sorted({condition_key(row, metadata, payload['key_mode']) for _, row in hmdb}):
            group = [(i, row) for i, row in hmdb if condition_key(row, metadata, payload['key_mode']) == key]
            scorer.group(key, True)
            cluster_cache = {}
            for order, query in group:
                baseline, _ = scorer.score(query)
                original = dict(query_order=order, variant='observed', **entry(baseline, query, lookup, smiles, fps))
                old = reference[_identity(query)]
                if original['rank'] != old['rank'] or original['top10'] != old['top10']:
                    raise ValueError('field_capacity_report_control_drift:' + query['record_id'])
                terms = {width: field_capacity(scorer.kernels['pred'], query, raw[query['record_id']], width, cluster_cache)
                         for width in WIDTHS_HZ}
                for config in choices:
                    if config['gamma'] == 0:
                        item = original
                    else:
                        values = baseline + config['gamma'] * terms[config['width_Hz']]
                        item = dict(query_order=order, variant='observed', **entry(values, query, lookup, smiles, fps))
                    records[config['name']].append(item)
            status('ranking', queries=len(records['report_control']), total=len(queries))
            scorer.kernels = scorer.tags = scorer.variable = None
            cluster_cache.clear()
            torch.cuda.empty_cache()
        report = dict(cohort=args.cohort, configs=choices, metrics={}, frequency_field=FREQUENCY_FIELD,
            frequency_counts=dict(frequency_counts), HMDB_queries=len(hmdb),
            control_rank_top10_exact=True, reference_sha256=sha(args.reference),
            candidates=len(smiles), candidate_order_sha256=order_sha256,
            query_sha256=stamp_queries(queries), checkpoint_sha256=bank['checkpoint_sha256'],
            inputs_sha256=pinned, selection_scope='train_val_only', external_api_calls=0)
        for name, rows in records.items():
            rows.sort(key=lambda row: row['query_order'])
            if len(rows) != len(queries):
                raise ValueError('field_capacity_output_cohort_changed')
            folder = out / name
            folder.mkdir()
            write_rows(folder / 'predictions.jsonl', rows)
            report['metrics'][name] = summarize_sources(rows)
        for path, digest in pinned.items():
            if sha(path) != digest:
                raise ValueError('field_capacity_input_changed_during_run:' + path)
        if sha(bank['checkpoint_path']) != bank['checkpoint_sha256']:
            raise ValueError('field_capacity_weight_changed_during_run')
        (out / 'summary.json').write_text(json.dumps(report, indent=2))
        status('complete')
        (out / 'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed', error=repr(exc))
        (out / 'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

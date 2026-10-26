# 功能概述：在模型和测试集冻结后，为逐记录预测补充可核查的数据质量线索，不筛除测试样本。
# 主要职责：annotate_record 对积分、结构和来源片段差异分类；review_file核对评估分母与
# 验证集导出的阈值，写出全部记录审计及类别摘要；main提供只读输入/新文件输出CLI。
# 代码库关系：读取train.evaluate_test的test_predictions.jsonl/test_metrics.json，复用
# audit_quality.source_fragment_candidate的结构证据，依赖RDKit；不加载模型或训练缓存。
# 边界：高误差、氢数不符、来源裁剪均不是自动确诊；exchangeable候选仅指N/O/S上H。
# 输出保留原始预测与flags，并始终confirmed_data_error=False；优先级只用于人工复查。
# 副作用：验证输入及全部记录数后原子创建JSONL和JSON，拒绝覆写，记录输入SHA256；
# 不修改原谱表、模型指标或拆分，候选分层不构成删除数据后的新性能报告。
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from rdkit import Chem
try:
    from .audit_quality import source_fragment_candidate
except ImportError:
    from audit_quality import source_fragment_candidate


def annotate_record(record, threshold):
    """返回保留原记录的诊断副本；优先级1最高，4为未见重点复查线索。"""
    result = dict(record)
    error = float(record['error_w1_ppm'])
    integral, hydrogens = float(record['reported_integral']), int(record['structure_protons'])
    if not all(math.isfinite(v) for v in (error, integral, threshold)) or threshold < 0:
        raise ValueError('finite error/integral and nonnegative validation threshold required')
    mol = Chem.MolFromSmiles(record['smiles'])
    if mol is None:
        raise ValueError('prediction contains invalid structure')
    exchangeable = sum(a.GetTotalNumHs() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8, 16))
    delta = integral-hydrogens
    categories = []
    if delta > .25:
        categories.append('integral_exceeds_structure_H')
    elif delta < -.25:
        categories.append('deficit_compatible_with_exchangeable_H' if -delta <= exchangeable+.25
                          else 'deficit_exceeds_exchangeable_H_candidates')
    source = source_fragment_candidate(record)
    if source:
        categories.append('source_structure_fragment_omitted')
    high = error > threshold
    if high:
        categories.append('high_error_relative_to_validation_p99')
    mismatch = any(x in categories for x in ('integral_exceeds_structure_H',
                                           'deficit_exceeds_exchangeable_H_candidates',
                                           'source_structure_fragment_omitted'))
    priority = 1 if high and mismatch else 2 if mismatch else 3 if high else 4
    result.update(review_high_error=high, confirmed_data_error=False,
                  review_priority=priority, review_categories=categories,
                  diagnostic_evidence={'reported_minus_structure_H':delta,
                      'potentially_exchangeable_NOSH_H':exchangeable,
                      'source_fragment_candidate':source},
                  review_interpretation='Diagnostic clues only; source adjudication required. '
                      'High model error alone does not establish incorrect data.')
    return result


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _write_new(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as handle:
        name=handle.name;handle.write(text)
    try:os.link(name,path)
    finally:os.unlink(name)


def review_file(predictions, metrics, out, summary):
    """校验分母后输出所有记录及摘要；不改变原测试指标或标签。"""
    inputs=[Path(predictions),Path(metrics)];outputs=[Path(out),Path(summary)]
    if len({p.resolve() for p in inputs+outputs}) != 4:
        raise ValueError('input and output paths must be distinct')
    for path in outputs:
        if path.exists():raise FileExistsError(path)
    metric=json.loads(inputs[1].read_text())
    threshold=float(metric['review_threshold_from_validation_p99'])
    records=[annotate_record(json.loads(line),threshold) for line in inputs[0].read_text().splitlines() if line.strip()]
    if len(records)!=metric['n']:raise ValueError('test denominator does not match predictions')
    counts=collections.Counter(c for r in records for c in r['review_categories'])
    report={'schema_version':1,'rule_version':'structure_integral_source_v1','n':len(records),
            'review_threshold_from_validation_p99':threshold,
            'category_counts':dict(counts),
            'priority_counts':dict(collections.Counter(r['review_priority'] for r in records)),
            'predictions_sha256':_sha(inputs[0]),'metrics_sha256':_sha(inputs[1]),
            'all_records_retained':True,'automatically_confirmed_errors':0,
            'selection_uses_test_error_for_review_only_not_model_selection_or_exclusion':True}
    encoded=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records)
    summary_text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    _write_new(outputs[0],encoded)
    try:_write_new(outputs[1],summary_text)
    except Exception:
        outputs[0].unlink()
        raise
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',required=True);parser.add_argument('--metrics',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--summary',required=True)
    args=parser.parse_args()
    print(json.dumps(review_file(args.predictions,args.metrics,args.out,args.summary),ensure_ascii=False))


if __name__=='__main__':main()

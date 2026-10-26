"""功能概述：只用训练来源比例审计固定验证集的模型选择口径，不读取测试结果。
training_prior从prepared元数据审计取实验train行数，排除理论、val和test；
prior_score把同一320-val的分来源MRR按该先验加权，与原五来源平衡准则并列。
main只读取各模型val/summary.json，要求查询hash一致；所有旧选择文件保持不变。
本审计在后续三seed骨干/上下文模型test前登记，检查来源均衡val与自然来源混合
目标之间的差异，不按已见test分数选择模型或改测试分母。
输出独占模型表、两个验证排序、train先验和源码hash；无训练、外部调用或旧结果写入。
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from .prepare import sha


def training_prior(audit):
    counts={key.split('|',1)[1]:int(n) for key,n in audit['training_rows'].items() if key.startswith('experimental|')};total=sum(counts.values())
    if not total or any(n<=0 for n in counts.values()):raise ValueError('invalid_training_source_counts')
    return counts,{s:n/total for s,n in counts.items()}


def prior_score(metrics,prior):
    by=metrics['by_source']
    if set(by)!=set(prior) or any(by[s]['queries']<=0 for s in prior):raise ValueError('validation_source_coverage_changed')
    return sum(prior[s]*by[s]['MRR'] for s in prior)


def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata-audit',required=True);p.add_argument('--evaluation',action='append',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    try:
        counts,prior=training_prior(json.loads(Path(a.metadata_audit).read_text()));rows=[];hashes=set()
        for root in map(Path,a.evaluation):
            for path in sorted(root.glob('*/val/summary.json')):
                if path.parents[1].name.startswith('previous_'):continue
                summary=json.loads(path.read_text());metrics=summary['metrics'];hashes.add(summary['query_sha256']);rows.append(dict(evaluation=str(root),variant=path.parents[1].name,checkpoint_sha256=summary['checkpoint_sha256'],source_balanced_MRR=metrics['overall']['MRR'],training_prior_MRR=prior_score(metrics,prior),by_source_MRR={s:v['MRR'] for s,v in metrics['by_source'].items()},validation_summary_sha256=sha(path)))
        if len(hashes)!=1 or not rows:raise ValueError('validation_query_hash_changed_or_empty')
        report=dict(registered_utc=datetime.now(timezone.utc).isoformat(),training_counts=counts,source_prior=prior,validation_query_sha256=next(iter(hashes)),models=rows,source_balanced_order=sorted(range(len(rows)),key=lambda i:-rows[i]['source_balanced_MRR']),training_prior_order=sorted(range(len(rows)),key=lambda i:-rows[i]['training_prior_MRR']),test_results_read=False,scope='additional train-frequency validation criterion; original criterion retained and no test cohort/denominator change',source_sha256=sha(__file__));(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:(out/'error.json').write_text(json.dumps(dict(error=repr(exc))));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

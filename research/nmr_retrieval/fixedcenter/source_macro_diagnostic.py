"""功能概述：只用已完成验证summary与1188行metrics.tsv分解source宏MRR差值。
read_inputs同次读取输入字节并固定SHA；calculate核对source/seed/cohort样本与指标，
先按query合并original_val+tune，再两seed等权、五source等权，输出每层实际贡献。
markdown保留20条source×seed×cohort事实表及5条合并贡献，不作评分规则或研究方案选择。
main独占输出JSON/Markdown/内部EXIT，始末核验全部输入与自身源码；异常保存失败证据。
本文件只处理train/val正常observed汇总，不读模型、谱值、test预测或执行训练/推理。
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import datetime

CANDIDATE='original_listwise_single'
BASELINE='frozen_single'
SEEDS=(193,211)
COHORTS=('original_val','tune')
SOURCES=('HMDB','NMRexp','WanJuan','nmrshiftdb2_2018','nmrshiftdb2_2024')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def close(a,b,label):
    if not math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12):
        raise ValueError('metric_mismatch:'+label)


def read_inputs(summary_path,metrics_path):
    pins={};values=[]
    for path in (summary_path,metrics_path):
        path=Path(path).resolve();raw=path.read_bytes();pins[str(path)]=digest(raw);values.append(raw)
    summary=json.loads(values[0])
    rows=list(csv.DictReader(io.StringIO(values[1].decode()),delimiter='\t'))
    return summary,rows,pins


def calculate(summary,raw_rows):
    index={}
    for row in raw_rows:
        key=tuple(row[k] for k in ('config','seed','cohort','variant','source'))
        if key in index:raise ValueError('duplicate_metric_stratum')
        index[key]=row
    if len(raw_rows)!=summary['metric_rows'] or len(raw_rows)!=1188:raise ValueError('incomplete_metric_table')
    def get(config,seed,cohort,source):
        row=index[(config,str(seed),cohort,'observed',source)]
        result={k:float(row[k]) for k in ('MRR','top1')}
        result['queries']=int(row['queries'])
        if result['queries']<=0 or not all(math.isfinite(result[k]) for k in ('MRR','top1')):
            raise ValueError('invalid_metric_stratum')
        return result
    detailed=[];combined=[];source_means=[]
    for source in SOURCES:
        seed_rows=[]
        for seed in SEEDS:
            parts=[]
            for cohort in COHORTS:
                a=get(CANDIDATE,seed,cohort,source);b=get(BASELINE,seed,cohort,source)
                if a['queries']!=b['queries']:raise ValueError('candidate_baseline_query_count')
                parts.append(dict(source=source,seed=seed,cohort=cohort,queries=a['queries'],
                    baseline_MRR=b['MRR'],candidate_MRR=a['MRR'],delta_MRR=a['MRR']-b['MRR'],
                    baseline_top1=b['top1'],candidate_top1=a['top1'],delta_top1=a['top1']-b['top1']))
            n=sum(r['queries'] for r in parts)
            value=dict(source=source,seed=seed,queries=n)
            for name in ('MRR','top1'):
                for role,config in (('baseline',BASELINE),('candidate',CANDIDATE)):
                    key=role+'_'+name
                    value[key]=sum(r[key]*r['queries'] for r in parts)/n
                    close(value[key],get(config,seed,'combined',source)[name],key+':combined')
                value['delta_'+name]=value['candidate_'+name]-value['baseline_'+name]
            value['macro_MRR_contribution']=value['delta_MRR']/len(SEEDS)/len(SOURCES)
            for r in parts:
                r['within_source_seed_query_weight']=r['queries']/n
                r['macro_MRR_contribution']=r['delta_MRR']*r['queries']/n/len(SEEDS)/len(SOURCES)
                detailed.append(r)
            combined.append(value);seed_rows.append(value)
        mean=dict(source=source,queries_by_seed={str(r['seed']):r['queries'] for r in seed_rows})
        for name in ('MRR','top1'):
            for role in ('baseline','candidate'):
                mean[role+'_'+name]=sum(r[role+'_'+name] for r in seed_rows)/len(SEEDS)
            mean['delta_'+name]=mean['candidate_'+name]-mean['baseline_'+name]
        mean['macro_MRR_contribution']=mean['delta_MRR']/len(SOURCES)
        source_means.append(mean)
    overall=[]
    for seed in SEEDS:
        for cohort,expected_n in [('original_val',320),('tune',1024),('combined',1344)]:
            result=dict(seed=seed,cohort=cohort,queries=expected_n)
            for role,config in (('baseline',BASELINE),('candidate',CANDIDATE)):
                parts=[get(config,seed,cohort,s) for s in SOURCES]
                if sum(p['queries'] for p in parts)!=expected_n:raise ValueError('cohort_query_count')
                for metric in ('MRR','top1'):
                    value=sum(p['queries']*p[metric] for p in parts)/expected_n
                    close(value,get(config,seed,cohort,'all')[metric],role+':'+cohort+':'+metric)
                    result[role+'_'+metric]=value
            for metric in ('MRR','top1'):
                result['delta_'+metric]=result['candidate_'+metric]-result['baseline_'+metric]
            overall.append(result)
    final={}
    for metric in ('MRR','top1'):
        for role,config in (('baseline',BASELINE),('candidate',CANDIDATE)):
            value=sum(r[role+'_'+metric] for r in overall if r['cohort']=='combined')/len(SEEDS)
            close(value,summary['signals'][config][metric],role+':summary:'+metric)
            final[role+'_'+metric]=value
        final['delta_'+metric]=final['candidate_'+metric]-final['baseline_'+metric]
    for role,config in (('baseline',BASELINE),('candidate',CANDIDATE)):
        final[role+'_macro_MRR']=sum(r[role+'_MRR'] for r in source_means)/len(SOURCES)
        close(final[role+'_macro_MRR'],summary['signals'][config]['macro_MRR'],role+':macro')
    final['delta_macro_MRR']=final['candidate_macro_MRR']-final['baseline_macro_MRR']
    contribution=sum(r['macro_MRR_contribution'] for r in detailed)
    close(contribution,final['delta_macro_MRR'],'20_contribution_sum')
    close(sum(r['macro_MRR_contribution'] for r in source_means),contribution,'5_contribution_sum')
    return dict(schema='source_macro_diagnostic_20261023',scope='train_val_observed_only',
        comparison=dict(candidate=CANDIDATE,baseline=BASELINE,seeds=list(SEEDS),cohorts=list(COHORTS),sources=list(SOURCES)),
        aggregation='query-weighted cohort merge within source/seed; equal mean across 2 seeds; equal 1/5 source mean',
        overall=final,by_source_seed_cohort=detailed,by_source_seed_combined=combined,
        by_source_two_seed_mean=source_means,overall_by_seed_cohort=overall,
        contribution_sum=contribution,source_macro_gate_reference=summary['selection']['configurations'][CANDIDATE]['gates']['source_macro'],
        scientific_decision_performed=False,model_inference_calls=0,test_data_read=False,external_api_calls=0)


def markdown(result):
    total=result['overall']
    text=['# Source宏MRR分解','',
        f"{CANDIDATE} 相对 {BASELINE}：正常MRR差值 {total['delta_MRR']:+.12f}，Top1差值 {total['delta_top1']:+.12f}，source宏MRR差值 {total['delta_macro_MRR']:+.12f}。",
        '以下Top1为0–1比例。两个cohort先按query合并，再对两seed取均值；五个source各占1/5。','',
        '| Source | 基准MRR | 候选MRR | ΔMRR | 宏MRR贡献（Δ/5） |',
        '|---|---:|---:|---:|---:|']
    for r in result['by_source_two_seed_mean']:
        text.append('| '+r['source']+' | '+' | '.join(f"{r[k]:+.12f}" for k in ('baseline_MRR','candidate_MRR','delta_MRR','macro_MRR_contribution'))+' |')
    text += ['', '| Source | Seed | Cohort | N | MRR 基准→候选（Δ） | Top1 基准→候选（Δ） | 宏MRR贡献 |',
             '|---|---:|---|---:|---|---|---:|']
    for r in result['by_source_seed_cohort']:
        text.append(f"| {r['source']} | {r['seed']} | {r['cohort']} | {r['queries']} | {r['baseline_MRR']:.9f}→{r['candidate_MRR']:.9f}（{r['delta_MRR']:+.9f}） | {r['baseline_top1']:.9f}→{r['candidate_top1']:.9f}（{r['delta_top1']:+.9f}） | {r['macro_MRR_contribution']:+.12f} |")
    text += ['',f"20项贡献之和：{result['contribution_sum']:+.12f}。与完整验证summary的source宏MRR差值一致。",'']
    return '\n'.join(text)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summary',required=True);parser.add_argument('--metrics',required=True);parser.add_argument('--out-prefix',required=True)
    args=parser.parse_args();prefix=Path(args.out_prefix).resolve()
    exit_path=prefix.with_suffix('.EXIT_CODE')
    try:
        summary,rows,pins=read_inputs(args.summary,args.metrics)
        source=Path(__file__).resolve();source_hash=digest(source.read_bytes())
        result=calculate(summary,rows)
        if any(digest(Path(p).read_bytes())!=h for p,h in pins.items()) or digest(source.read_bytes())!=source_hash:
            raise ValueError('input_or_source_changed')
        result.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_sha256=pins,
                      source_sha256={str(source):source_hash},inputs_unchanged=True)
        with prefix.with_suffix('.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        with prefix.with_suffix('.md').open('x') as f:f.write(markdown(result))
        with exit_path.open('x') as f:f.write('0\n')
        print(json.dumps(dict(status='PASS',delta_macro_MRR=result['overall']['delta_macro_MRR'],
            detailed_rows=len(result['by_source_seed_cohort']),source_rows=len(result['by_source_two_seed_mean']))))
    except BaseException as exc:
        with prefix.with_suffix('.failure.json').open('x') as f:f.write(json.dumps(dict(error=repr(exc)))+'\n')
        if not exit_path.exists():exit_path.write_text('1\n')
        raise


if __name__=='__main__':main()

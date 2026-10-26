"""功能概述：冻结方案后统一评估完整库和原缺峰变体，避免重贴标签恢复删峰。

main 只接受verify/test，并逐项核验冻结bank、宽度、cohort、先验、源码和本入口SHA；
配置只读冻结列表，不能在本入口搜索。Scorer 对全部114982候选打分，真结构只用于
排名。测试仍是原1000条以及前200条生成的200裁窗/171整峰删除变体；先贴观测侧表
再生成掩码，之后不再贴原始完整侧表，保证缺峰与窗口始终生效。verify只用封存1024。
输出每个冻结评分臂的完整Top1/5/10/MRR和逐查询Top10，并保存候选/查询/权重SHA；
可核对旧test基线逐条一致。对新GNN基线没有旧test参考时保留全指标但不伪造控制。
不改写任何训练/验证/测试/候选，不访问网络或外部服务。
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .evaluate import missing_queries,summarize
from .h_condinfer import condition_key,scoring_inputs
from .h_conditions import SOLVENT_DIM
from .h_eval import LazyFP
from .h_global_bank import load_bank
from .h_global_extract import Scorer
from .h_global_protocol import actual_inputs
from .h_global_scoring import frozen_choices
from .h_labels import make_key
from .h_observation_study import entry,write_rows,summarize_sources,stamp_queries
from .h_score import attach_observation
from .prepare import sha


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    for name in ('base','bank','widths','cohorts','frozen-selection','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--extension');parser.add_argument('--cohort',choices=['verify','test'],required=True)
    args=parser.parse_args()
    required=actual_inputs(args.base,args.bank,args.widths,args.cohorts,args.extension)+[Path(__file__).resolve()]
    choices=frozen_choices(args.frozen_selection,required)
    selection_sha256=sha(args.frozen_selection)
    if any(c['kind'] not in ('baseline','atom','integral_count','outlier') for c in choices):
        raise ValueError('locked_evaluation_unsupported_frozen_kind')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):
        tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,**kw)));tmp.replace(out/'status.json')
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda')
        bank=load_bank(args.bank,args.extension)
        if sha(bank['checkpoint_path'])!=bank['checkpoint_sha256']:raise ValueError('locked_prediction_checkpoint_changed')
        payload=torch.load(Path(args.cohorts)/'cohorts.pt',map_location='cpu',weights_only=False)
        queries=bank['test'] if args.cohort=='test' else payload['cohorts']['verify']
        if len(queries)!=(1000 if args.cohort=='test' else 1024):raise ValueError('locked_query_count_drift')
        widths=json.loads(Path(args.widths).read_text())['selected_ppm']
        inputs=scoring_inputs(Path(args.base));metadata=payload['metadata'];labels=payload['annotations']
        owner=next((c for c in choices if c.get('condition')=='train_mixture'),choices[0])
        scorer=Scorer(bank,payload,widths,inputs,owner,device)
        indices=bank['single_indices'];smiles=[bank['smiles'][i] for i in indices];lookup={s:i for i,s in enumerate(smiles)};fps=LazyFP()
        if len(smiles)!=114982:raise ValueError('locked_candidate_count_drift')
        records={c['name']:[] for c in choices};stress={c['name']:[] for c in choices}
        for key in sorted({condition_key(r,metadata,payload['key_mode']) for r in queries}):
            group=[(i,r) for i,r in enumerate(queries) if condition_key(r,metadata,payload['key_mode'])==key]
            scorer.group(key,any(r['source']=='HMDB' for _,r in group))
            for order,row in group:
                query=attach_observation(row,labels.get(make_key(row),{}))
                query['_condition_known']=bool(key[4+SOLVENT_DIM-1]>0)
                variants=[('observed',query)]
                if args.cohort=='test' and order<200:variants+=missing_queries(query,'1H')
                for variant,observed in variants:
                    for choice in choices:
                        scorer.choice=choice;values,_=scorer.score(observed)
                        item=dict(query_order=order,variant=variant,**entry(values,observed,lookup,smiles,fps))
                        (records if variant=='observed' else stress)[choice['name']].append(item)
            status('ranking',ranked=len(records[choices[0]['name']]),total=len(queries))
            scorer.kernels=scorer.tags=scorer.variable=None;torch.cuda.empty_cache()
        report=dict(cohort=args.cohort,metrics={},stress={},configs=choices,
            candidate_count=len(smiles),candidate_order_sha256=hashlib.sha256('\n'.join(smiles).encode()).hexdigest(),
            query_sha256=stamp_queries(queries),checkpoint_sha256=bank['checkpoint_sha256'],
            inputs_sha256={str(p):sha(p) for p in required},frozen_selection_sha256=sha(args.frozen_selection),external_api_calls=0)
        for name,rows in records.items():
            rows.sort(key=lambda r:r['query_order']);folder=out/name;folder.mkdir()
            write_rows(folder/'predictions.jsonl',rows);report['metrics'][name]=summarize_sources(rows)
            if args.cohort=='test':
                if len(stress[name])!=371 or sum(r['variant']=='frequency_crop' for r in stress[name])!=200:
                    raise ValueError('locked_stress_cohort_drift')
                stress[name].sort(key=lambda r:(r['query_order'],r['variant']))
                write_rows(folder/'missing_predictions.jsonl',stress[name])
                report['stress'][name]={kind:summarize([r for r in stress[name] if r['variant']==kind]) for kind in ('frequency_crop','peak_dropout')}
        reference=Path(args.bank).parent/'test/width_0.5/predictions.jsonl'
        if args.cohort=='test' and reference.exists():
            old={(r['source'],r['record_id'],r['smiles']):r for r in map(json.loads,reference.read_text().splitlines())}
            exact=all(r['rank']==old[(r['source'],r['record_id'],r['smiles'])]['rank'] and
                      r['top10']==old[(r['source'],r['record_id'],r['smiles'])]['top10'] for r in records['baseline'])
            report['old_baseline_rank_top10_exact']=exact
            if not exact:raise ValueError('locked_old_test_baseline_drift')
            reference_missing=reference.with_name('missing_predictions.jsonl')
            if not reference_missing.exists():raise ValueError('locked_old_stress_control_missing')
            identity=lambda r:(r['source'],r['record_id'],r['smiles'],r['variant'])
            old_missing={identity(r):r for r in map(json.loads,reference_missing.read_text().splitlines())}
            new_missing={identity(r):r for r in stress['baseline']}
            exact_missing=len(old_missing)==371 and old_missing.keys()==new_missing.keys() and all(
                r['rank']==old_missing[k]['rank'] and r['top10']==old_missing[k]['top10']
                for k,r in new_missing.items())
            report['old_baseline_stress_rank_top10_exact']=exact_missing
            if not exact_missing:raise ValueError('locked_old_test_stress_baseline_drift')
        frozen_choices(args.frozen_selection,required)
        if sha(args.frozen_selection)!=selection_sha256:raise ValueError('locked_selection_changed_during_evaluation')
        if sha(bank['checkpoint_path'])!=bank['checkpoint_sha256']:raise ValueError('locked_checkpoint_changed_during_evaluation')
        (out/'summary.json').write_text(json.dumps(report,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':
    main()

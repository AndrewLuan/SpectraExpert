"""功能概述：独立消融等价环境报告混合与 HMDB 正观测匹配容量。

main 复用固定atom_count_1.0基座，比较train拟合P(k|H,K)的替换强度以及HMDB
宽核重复使用的容量修正；所有数值网格在本入口声明。未知K、非原子表、显式窗口、
人工partial_peaks计数路径精确回退旧报告先验。容量项只约束已报告的HMDB峰簇，
每个分辨率同时聚簇观测与预测；显式窗精确回退原评分，避免跨窗聚簇挪动支持。
候选未报峰可免费跳过，不产生完整谱负标签。候选侧表与模型只读，SHA与顺序核对。
train/val阶段只开放original_val/tune；verify/test要求完整冻结，保留原1000+371。
不修改原始谱、mask、候选、测试或其他模型，不访问外部API。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import missing_queries,summarize
from .h_condinfer import condition_key,scoring_inputs
from .h_conditions import SOLVENT_DIM
from .h_eval import LazyFP
from .h_global_bank import load_bank
from .h_global_capacity import cluster_predictions,capacity_excess
from .h_global_extract import Scorer
from .h_global_protocol import actual_inputs
from .h_global_reporting import reporting_delta
from .h_global_scoring import configs as coarse_configs,frozen_choices
from .h_labels import make_key
from .h_latent_score import eligible
from .h_observation_study import entry,write_rows,summarize_sources,stamp_queries,uniform_peaks
from .h_observation_finalize import visible_lines
from .h_observed import cluster_lines
from .h_score import attach_observation
from .prepare import sha


def configs():
    result=[dict(name='count_control',report=0.,capacity=0.,width=.08)]
    for width in (.04,.08,.16):
        for weight in (.25,1.,4.):
            result.append(dict(name=f'capacity_w{width}_g{weight}',report=0.,capacity=weight,width=width))
    for weight in (.25,1.):
        result.append(dict(name=f'report_g{weight}',report=weight,capacity=0.,width=.08))
        for capacity in (.25,1.):
            result.append(dict(name=f'report_g{weight}_capacity_g{capacity}',report=weight,capacity=capacity,width=.08))
    return result


def artifact_inputs(folder):
    return [Path(folder)/name for name in ('table.pt','models.json','manifest.json')]


def corrections(scorer,observed,environments,models,resolutions):
    """只用查询和候选公开特征计算报告/容量差值，供评分与重排特征入口共同复用。"""
    delta=torch.zeros_like(scorer.counts);capacity={}
    if eligible(observed) and not observed['observation'].get('partial_peaks') and observed['source'] in models:
        weights=np.asarray(observed['observed_weights']);k=int(round(float(weights.sum())))
        if k>0 and np.allclose(weights,weights.round(),atol=.001):
            delta=reporting_delta(scorer.counts,environments,k,models[observed['source']],observed['observation'])
    if observed['source']=='HMDB' and not observed['observation']['windows']:
        lines=visible_lines(observed,scorer.inputs[-1])
        for width,(centers,valid) in resolutions.items():
            positions,_=cluster_lines(lines,width)
            capacity[width]=capacity_excess(centers,valid,positions)
    return delta,capacity


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ('base','bank','widths','cohorts','artifact','cohort','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--extension');p.add_argument('--frozen-selection');p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    if args.cohort not in ('original_val','tune','verify','test'):raise ValueError('observation_study_wrong_cohort')
    required=actual_inputs(args.base,args.bank,args.widths,args.cohorts,args.extension)+artifact_inputs(args.artifact)
    required += [Path(__file__).resolve(),Path(__file__).with_name('h_global_capacity.py'),Path(__file__).with_name('h_global_reporting.py')]
    required=sorted({p.resolve() for p in required});selected=configs();selection_sha=None
    choice=next(c for c in coarse_configs() if c['name']=='atom_count_1.0')
    if args.cohort in ('verify','test'):
        if args.limit or not args.frozen_selection:raise ValueError('observation_study_not_fully_frozen')
        if choice not in frozen_choices(args.frozen_selection,required):raise ValueError('observation_study_coarse_not_frozen')
        frozen=json.loads(Path(args.frozen_selection).read_text());selected=frozen['observation_configs'];selection_sha=sha(args.frozen_selection)
        registered={c['name']:c for c in configs()}
        if not selected or any(c!=registered.get(c['name']) for c in selected):raise ValueError('observation_study_unregistered_config')
        if not any(c['name']=='count_control' for c in selected):raise ValueError('observation_study_control_required')
    pinned={str(p):sha(p) for p in required};out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):
        temp=out/'status.tmp';temp.write_text(json.dumps(dict(stage=stage,**kw)));temp.replace(out/'status.json')
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda')
        bank=load_bank(args.bank,args.extension);payload=torch.load(Path(args.cohorts)/'cohorts.pt',map_location='cpu',weights_only=False)
        if sha(bank['checkpoint_path'])!=bank['checkpoint_sha256']:raise ValueError('observation_study_checkpoint_drift')
        indices=bank['single_indices'];smiles=[bank['smiles'][i] for i in indices];lookup={s:i for i,s in enumerate(smiles)}
        order_sha=hashlib.sha256('\n'.join(smiles).encode()).hexdigest()
        artifact=json.loads((Path(args.artifact)/'manifest.json').read_text())
        if artifact['candidate_order_sha256']!=order_sha:raise ValueError('observation_study_candidate_order_changed')
        table=torch.load(Path(args.artifact)/'table.pt',map_location='cpu',weights_only=False)
        if not torch.allclose(table['H'],bank['physics'][indices,0].cpu()*40,atol=.001,rtol=0):
            raise ValueError('observation_study_candidate_H_changed')
        environments=table['K'].to(device);models=json.loads((Path(args.artifact)/'models.json').read_text())
        queries=bank['test'] if args.cohort=='test' else payload['cohorts'][args.cohort]
        if args.limit:queries=queries[:args.limit]
        widths=json.loads(Path(args.widths).read_text())['selected_ppm'];inputs=scoring_inputs(Path(args.base))
        scorer=Scorer(bank,payload,widths,inputs,choice,device);fps=LazyFP();records={c['name']:[] for c in selected};stress={c['name']:[] for c in selected}
        for key in sorted({condition_key(r,payload['metadata'],payload['key_mode']) for r in queries}):
            group=[(i,r) for i,r in enumerate(queries) if condition_key(r,payload['metadata'],payload['key_mode'])==key]
            hmdb=any(r['source']=='HMDB' for _,r in group);scorer.group(key,hmdb)
            resolutions={}
            if hmdb:
                distinct=uniform_peaks(scorer.kernels['pred'])*10
                for width in {c['width'] for c in selected if c['capacity']}:
                    resolutions[width]=cluster_predictions(distinct,width)
            for order,row in group:
                query=attach_observation(row,payload['annotations'].get(make_key(row),{}))
                query['_condition_known']=bool(key[4+SOLVENT_DIM-1]>0)
                variants=[('observed',query)]
                if args.cohort=='test' and order<200:variants+=missing_queries(query,'1H')
                for variant,observed in variants:
                    base,_=scorer.score(observed)
                    delta,capacity=corrections(scorer,observed,environments,models,resolutions)
                    for config in selected:
                        values=base+config['report']*delta
                        if config['capacity'] and config['width'] in capacity:values=values+config['capacity']*capacity[config['width']]
                        item=dict(query_order=order,variant=variant,**entry(values,observed,lookup,smiles,fps))
                        (records if variant=='observed' else stress)[config['name']].append(item)
                if len(records[selected[0]['name']])%25==0:status('ranking',queries=len(records[selected[0]['name']]),total=len(queries))
            scorer.kernels=scorer.tags=scorer.variable=None;resolutions={};torch.cuda.empty_cache()
        report=dict(cohort=args.cohort,metrics={},stress={},configs=selected,candidate_count=len(smiles),candidate_order_sha256=order_sha,
            query_sha256=stamp_queries(queries),checkpoint_sha256=bank['checkpoint_sha256'],inputs_sha256=pinned,external_api_calls=0)
        for name,rows in records.items():
            rows.sort(key=lambda r:r['query_order']);folder=out/name;folder.mkdir();write_rows(folder/'predictions.jsonl',rows)
            report['metrics'][name]=summarize_sources(rows)
            if args.cohort=='test':
                if len(rows)!=1000 or len(stress[name])!=371:raise ValueError('observation_study_test_cohort_changed')
                stress[name].sort(key=lambda r:(r['query_order'],r['variant']));write_rows(folder/'missing_predictions.jsonl',stress[name])
                report['stress'][name]={v:summarize([r for r in stress[name] if r['variant']==v]) for v in ('frequency_crop','peak_dropout')}
        for path,digest in pinned.items():
            if sha(path)!=digest:raise ValueError('observation_study_input_changed_during_run:'+path)
        if selection_sha and sha(args.frozen_selection)!=selection_sha:raise ValueError('observation_study_selection_changed')
        if sha(bank['checkpoint_path'])!=bank['checkpoint_sha256']:raise ValueError('observation_study_weight_changed_during_run')
        (out/'summary.json').write_text(json.dumps(report,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

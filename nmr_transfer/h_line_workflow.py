"""功能概述：在固定均值模型和既有稀疏原子评分上，只改变HMDB细线置信度。
hmdb_score沿用已冻结的水相可见性/等峰核/参考偏移，原峰高只用于污染概率或
细线阈值，不作H积分。select先在原320-val比较十个预声明配置，再冻结后复评
1000条test和同一171删峰/200裁窗；非HMDB单核排名直接保持原记录。
双核复用h_uncertainty_workflow.joint的固定候选/C逐查询控制，并只覆盖HMDB的
H评分；其他来源继续冻结latent配置。所有控制、查询hash和开发复评属性保存。
输出独占状态、配置、逐查询结果、消融和退出码；不训练均值或使用测试选参。
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .h_line_confidence import GRID,score
from .h_latent_workflow import Context,load_reference,query_observation
from .h_latent_score import bank_key
from .h_condinfer import scoring_inputs
from .h_observation_finalize import visible_lines,blend,query_score
from .h_observation_study import write_rows,summarize_sources,stamp_queries
from .h_diagnose import cond_key
from .h_uncertainty_workflow import joint
from .evaluate import missing_queries,summarize
from .h_labels import make_key
from .prepare import sha


def visible_raw(query,raw):
    r=raw[query['record_id']];xs=np.asarray(r['value']['x']);selected=visible_lines(query,raw);keep=np.isin(xs,selected);ys=np.asarray(r['value'].get('y',[]));value=dict(x=xs[keep].tolist())
    if ys.ndim==1 and len(ys)==len(xs):value['y']=ys[keep].tolist()
    return dict(value=value)


def hmdb_score(query,context,choice,inputs):
    frozen,prior,config,priors,raw=inputs
    if choice['kind']=='baseline':return query_score(query,context.given,context.counts,prior,config,frozen,priors,raw)
    cfg=frozen['water_HMDB'] if cond_key(query)[1]>0 and frozen['water_HMDB'] is not None else frozen['HMDB'];kernel=context.given[cfg['kernel']]
    if 'visibility' in cfg:kernel=blend(kernel,context.given['fixed_'+cfg['kernel']],cfg['visibility'])
    return score(kernel,query,visible_raw(query,raw),choice,cfg['shift_sigma'])


@torch.inference_mode()
def single(bank,context,baseline,oldmissing,choice,inputs,split,stress=False):
    lookup={(r['source'],r['record_id'],r['smiles']):r for r in baseline};oldstress={(r['variant'],r['record_id'],r['smiles']):r for r in oldmissing};rows=[];missing=[]
    for key in sorted({bank_key(bank,r,context.metadata) for r in bank[split]}):
        group=[(i,r) for i,r in enumerate(bank[split]) if bank_key(bank,r,context.metadata)==key]
        if any(r['source']=='HMDB' for _,r in group):context.set_condition(key,True)
        for i,r in group:
            q=query_observation(r,context.metadata,context.labels);result=context.rank(hmdb_score(q,context,choice,inputs),q) if q['source']=='HMDB' else dict(lookup[r['source'],r['record_id'],r['smiles']]);rows.append(dict(result,query_order=i))
            if stress and i<200:
                for variant,rr in missing_queries(q,'1H'):
                    result=context.rank(hmdb_score(rr,context,choice,inputs),rr) if q['source']=='HMDB' else dict(oldstress[variant,r['record_id'],r['smiles']]);missing.append(dict(result,variant=variant))
    rows.sort(key=lambda r:r['query_order']);return rows,missing


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--reference-latent',required=True);p.add_argument('--out',required=True);a=p.parse_args();base=Path(a.base);ref=Path(a.reference_latent);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        status('waiting_reference');deadline=time.monotonic()+10800
        while not (ref/'EXIT_CODE').exists():
            if time.monotonic()>deadline:raise TimeoutError('reference_timeout')
            time.sleep(10)
        if (ref/'EXIT_CODE').read_text().strip()!='0':raise RuntimeError('reference_failed')
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');evaluation=Path(json.loads((ref/'protocol.json').read_text())['reference_evaluation']);bank,_,_,_,_=load_reference(base,evaluation);metadata=torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False);labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False);normal=json.loads((base/'hydrogen_analysis/routed_evaluation/site/count_prior.json').read_text());reporting=json.loads((ref/'reporting_priors.json').read_text());cfg=json.loads((ref/'frozen_selection.json').read_text())['selected_config'];inputs=scoring_inputs(base);valbase=[json.loads(l) for l in (ref/'validation_selected.jsonl').read_text().splitlines()];baseline=[json.loads(l) for l in (ref/'selected_predictions.jsonl').read_text().splitlines()];oldmissing=[json.loads(l) for l in (ref/'missing_predictions.jsonl').read_text().splitlines()];context=Context(bank,metadata,labels,normal,reporting,bank['single_indices'],device);trials={};valrows={}
        for i,choice in enumerate(GRID):
            status('validation',configuration=i);rs,_=single(bank,context,valbase,[],choice,inputs,'val');trials[str(i)]=summarize_sources(rs);valrows[str(i)]=rs
        controls={(r['source'],r['record_id'],r['smiles']):r['rank'] for r in valbase}
        if any(r['rank']!=controls[r['source'],r['record_id'],r['smiles']] for r in valrows['0']):raise ValueError('baseline_HMDB_rank_changed')
        selected=max(trials,key=lambda k:trials[k]['overall']['MRR']);choice=GRID[int(selected)];frozen=dict(selected_index=selected,selected_config=choice,validation=trials,grid=GRID,reference=str(ref),checkpoint_sha256=bank['checkpoint_sha256'],query_sha256=stamp_queries(bank['test']),source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},selection='same320 validation MRR before test; only HMDB changes; height is confidence, never integral',development_test_reuse=True);(out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));write_rows(out/'validation_selected.jsonl',valrows[selected]);status('test');rows,missing=single(bank,context,baseline,oldmissing,choice,inputs,'test',True)
        if len(rows)!=1000 or sum(r['variant']=='peak_dropout' for r in missing)!=171:raise ValueError('query_cohort_changed')
        write_rows(out/'predictions.jsonl',rows);write_rows(out/'missing_predictions.jsonl',missing);(out/'summary.json').write_text(json.dumps(dict(candidates=len(context.smiles),baseline=summarize_sources(baseline),selected=summarize_sources(rows),missing_stress={v:dict(baseline=summarize([r for r in oldmissing if r['variant']==v]),selected=summarize([r for r in missing if r['variant']==v])) for v in ['peak_dropout','frequency_crop']}),indent=2));del context;torch.cuda.empty_cache();context=Context(bank,metadata,labels,normal,reporting,bank['joint_indices'],device);joint(base,bank,context,cfg,out,status,h_override=lambda q,c:hmdb_score(q,c,choice,inputs));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

"""功能概述：训练报告过程后，在同库同测试上检验HMDB的环境部分匹配。
fit_reporting从1568条train HMDB和冻结条件模型预测拟合正常/整峰删除报告过程；
不使用实验积分、原子归属或测试拟合位移。EnvContext按候选预测生成一单位一簇的
位置集，并为水相保留全H/非交换H两种可见性；空非交换支持只按候选自身回退。
validation在原320查询上比较预声明37个配置，非HMDB保持旧排名，基线直接复用。
先冻结再复评原1000查询/171缺峰/200裁窗，双核仍原792对和冻结C；显式频窗沿用
旧评分。公共偏移仅为统计误差潜变量，不解释为已测参考校正或纯溶剂效应。
输出训练报告计数、模型/数据/源码身份、逐查询结果和退出码；旧结果、候选与
数据均不改写。--matching soft检验有序子集求和归一化，density保留旧密度并增加
报告计数/宽公共偏移；--reporting-reference仅在
模型、参考和点集/计数拟合函数AST完全一致时复用训练先验，不重复拟合数据。
test可接收score_function，供h_formyl_workflow复用同一冻结查询/压力循环，
默认仍使用本模块routed_score；不通过进程全局替换改变评分函数。
依赖h_env_report核心、原latent观测路由和joint的逐查询C控制。
"""
import argparse
import ast
import collections
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from .h_env_report import GRID,SOFT_GRID,environment_points,visible_fallback,fit_count_mixture,environment_features,environment_score
from .h_env_density import GRID as DENSITY_GRID,density_features,density_score
from .h_latent_workflow import Context,load_reference,query_observation
from .h_latent_score import bank_key
from .h_condinfer import scoring_inputs
from .h_observation_finalize import query_score,visible_lines
from .h_observation_study import write_rows,summarize_sources,stamp_queries
from .h_gapworkflow import resolve_checkpoint
from .h_conditions import ConditionsSpectra
from .h_labels import make_key
from .emission_model import make_model,weighted_quantiles
from .train import collate
from .h_diagnose import cond_key
from .h_uncertainty_workflow import joint
from .evaluate import missing_queries,summarize
from .prepare import sha


@torch.inference_mode()
def fit_reporting(base,prepared,evaluation,metadata,raw,device,out,status):
    checkpoint=resolve_checkpoint(base,evaluation);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict']);cache=torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True);rows=[r for r in cache['rows'] if r['split']=='train' and r['source']=='HMDB' and r['domain']=='experimental'];loader=DataLoader(ConditionsSpectra(rows,cache['graphs'],'1H',metadata),batch_size=128,num_workers=2,collate_fn=collate);samples={p:{f:{v:[] for v in ['full','fixed']} for f in ['water','nonwater']} for p in ['normal','dropout']};records=[];offset=0;status('fitting_train_reporting',records=len(rows))
    for b in loader:
        b=b.to(device);d=model(b,details=True);full=d['quantiles'];keep=~d['exchangeable'];fixed=weighted_quantiles(d['values'][keep],d['weights'][keep],d['groups'][keep],b.num_graphs);fixed,empty=visible_fallback(full,fixed);_,fm=environment_points(full);_,vm=environment_points(fixed);nn={'full':fm.sum(1).cpu().tolist(),'fixed':vm.sum(1).cpu().tolist()}
        for j,r in enumerate(rows[offset:offset+b.num_graphs]):
            family='water' if cond_key(r)[1]>0 else 'nonwater';centers=metadata[make_key(r)]['line_centers'];k=len(centers);rec=dict(record_id=r['record_id'],smiles=r['smiles'],family=family,reported_clusters=k,predicted_clusters={v:nn[v][j] if family=='water' else nn['full'][j] for v in nn},empty_fixed_fallback=bool(empty[j]))
            for v in nn:samples['normal'][family][v].append((rec['predicted_clusters'][v],k))
            for variant,rr in missing_queries(r,'1H'):
                if variant!='peak_dropout':continue
                from .h_observed import cluster_lines
                kk=len(cluster_lines(visible_lines(rr,raw),.08)[0]);rec['dropout_reported_clusters']=kk
                for v in nn:samples['dropout'][family][v].append((rec['predicted_clusters'][v],kk))
            records.append(rec)
        offset+=b.num_graphs
    priors={p:{f:{v:fit_count_mixture(pairs) for v,pairs in vv.items()} for f,vv in ff.items()} for p,ff in samples.items()};write_rows(out/'training_reporting_records.jsonl',records);(out/'training_reporting_priors.json').write_text(json.dumps(priors,indent=2));del model,cache;torch.cuda.empty_cache();return priors


class EnvContext(Context):
    def __init__(self,*args,matching='hard',**kwargs):
        super().__init__(*args,**kwargs);self.matching=matching
    def set_condition(self,key,need_hmdb=False):
        super().set_condition(key,need_hmdb);self.full_points=environment_points(self.given['pred']);self.water=key[1]>0;self.density_kernels=None;self.empty_fixed=torch.zeros(len(self.counts),dtype=torch.bool,device=self.device)
        if self.water:
            fixed=self.bank['fixed_predictions'][key][self.indices].to(self.device);fixed,empty=visible_fallback(self.given['pred'],fixed);nonexchange=(self.bank['physics'][self.indices,0]*40-self.bank['physics'][self.indices,2]*8).to(self.device)
            if (empty&(nonexchange>.5)).any():raise ValueError('nonempty_structural_support_lost')
            self.fixed_points=environment_points(fixed);self.empty_fixed=empty
        else:self.fixed_points=self.full_points
    def env_features(self,q,priors,raw):return environment_features(self.full_points,self.fixed_points,q,visible_lines(q,raw),priors,self.water,matching=self.matching)


def routed_score(q,context,choice,priors,inputs):
    frozen,prior,config,oldpriors,raw=inputs
    if choice['kind']=='baseline' or q['observation']['windows']:return query_score(q,context.given,context.counts,prior,config,frozen,oldpriors,raw)
    return proposed_score(proposed_features(q,context,priors,inputs),choice)


def proposed_features(q,context,priors,inputs):
    return density_features(context,q,priors,inputs) if context.matching=='density' else context.env_features(q,priors,inputs[-1])


def proposed_score(features,choice):return density_score(features,choice) if choice['kind']=='density_reporting' else environment_score(features,choice)


@torch.inference_mode()
def validation(bank,context,baseline,priors,inputs,out,status,grid=GRID):
    lookup={(r['source'],r['record_id'],r['smiles']):r for r in baseline};trials={str(i):[] for i in range(len(grid))}
    for key in sorted({bank_key(bank,r,context.metadata) for r in bank['val']}):
        group=[(i,r) for i,r in enumerate(bank['val']) if bank_key(bank,r,context.metadata)==key]
        if any(r['source']=='HMDB' for _,r in group):context.set_condition(key,context.matching=='density')
        for i,r in group:
            q=query_observation(r,context.metadata,context.labels);old=lookup[r['source'],r['record_id'],r['smiles']];ff=proposed_features(q,context,priors,inputs) if q['source']=='HMDB' and not q['observation']['windows'] else None
            for j,cfg in enumerate(grid):
                result=dict(old) if ff is None or j==0 else context.rank(proposed_score(ff,cfg),q);trials[str(j)].append(dict(result,query_order=i))
        status('validation',ranked=len(trials['0']))
    for rs in trials.values():rs.sort(key=lambda r:r['query_order'])
    metrics={k:summarize_sources(rs) for k,rs in trials.items()};selected=max(metrics,key=lambda k:metrics[k]['overall']['MRR']);frozen=dict(selected_index=selected,selected_config=grid[int(selected)],validation=metrics,grid=grid,selection='same320 validation before test; only HMDB environment-reporting process changes',development_test_reuse=True);(out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));write_rows(out/'validation_selected.jsonl',trials[selected]);return frozen


def reporting_priors(a,base,prepared,evaluation,metadata,inputs,device,out,status,checkpoint_sha):
    if not a.reporting_reference:return fit_reporting(base,prepared,evaluation,metadata,inputs[-1],device,out,status)
    reference=Path(a.reporting_reference);protocol=json.loads((reference/'protocol.json').read_text())
    if protocol['H_checkpoint_sha256']!=checkpoint_sha or protocol['reference']!=a.reference_latent:raise ValueError('reporting_reference_mismatch')
    def functions(path):
        tree=ast.parse(path.read_text());return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    old=functions(reference/'source/nmr_transfer/h_env_report.py');new=functions(Path(__file__).parent/'h_env_report.py')
    for name in ['environment_points','visible_fallback','fit_count_mixture']:
        if old[name]!=new[name]:raise ValueError('reporting_feature_changed:'+name)
    for name in ['training_reporting_priors.json','training_reporting_records.jsonl']:shutil.copy2(reference/name,out/name)
    (out/'reporting_reuse.json').write_text(json.dumps(dict(reference=str(reference),prior_sha256=sha(reference/'training_reporting_priors.json'),verified_functions=['environment_points','visible_fallback','fit_count_mixture'])))
    return json.loads((out/'training_reporting_priors.json').read_text())


@torch.inference_mode()
def test(bank,context,baseline,oldmissing,choice,priors,inputs,out,score_function=None):
    scorer=score_function or routed_score
    lookup={(r['source'],r['record_id'],r['smiles']):r for r in baseline};oldstress={(r['variant'],r['record_id'],r['smiles']):r for r in oldmissing};rows=[];stress=[]
    for key in sorted({bank_key(bank,r,context.metadata) for r in bank['test']}):
        group=[(i,r) for i,r in enumerate(bank['test']) if bank_key(bank,r,context.metadata)==key]
        if any(r['source']=='HMDB' for _,r in group):context.set_condition(key,True)
        for i,r in group:
            q=query_observation(r,context.metadata,context.labels);old=lookup[r['source'],r['record_id'],r['smiles']];result=context.rank(scorer(q,context,choice,priors,inputs),q) if r['source']=='HMDB' and choice['kind']!='baseline' else dict(old);rows.append(dict(result,query_order=i))
            if i<200:
                for variant,rr in missing_queries(q,'1H'):
                    result=context.rank(scorer(rr,context,choice,priors,inputs),rr) if r['source']=='HMDB' and choice['kind']!='baseline' else dict(oldstress[variant,r['record_id'],r['smiles']]);stress.append(dict(result,variant=variant))
    rows.sort(key=lambda r:r['query_order'])
    if len(rows)!=1000 or sum(r['variant']=='peak_dropout' for r in stress)!=171:raise ValueError('cohort_changed')
    write_rows(out/'predictions.jsonl',rows);write_rows(out/'missing_predictions.jsonl',stress);(out/'summary.json').write_text(json.dumps(dict(candidates=len(context.smiles),baseline=summarize_sources(baseline),selected=summarize_sources(rows),missing_stress={v:dict(baseline=summarize([r for r in oldmissing if r['variant']==v]),selected=summarize([r for r in stress if r['variant']==v])) for v in ['peak_dropout','frequency_crop']}),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--reference-latent',required=True);p.add_argument('--out',required=True);p.add_argument('--matching',choices=['hard','soft','density'],default='hard');p.add_argument('--reporting-reference');a=p.parse_args();base=Path(a.base);prepared=Path(a.prepared);ref=Path(a.reference_latent);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');evaluation=Path(json.loads((ref/'protocol.json').read_text())['reference_evaluation']);bank,_,_,_,_=load_reference(base,evaluation);metadata=torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False);labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False);normal=json.loads((base/'hydrogen_analysis/routed_evaluation/site/count_prior.json').read_text());reporting=json.loads((ref/'reporting_priors.json').read_text());cfg=json.loads((ref/'frozen_selection.json').read_text())['selected_config'];inputs=scoring_inputs(base);status('train_reporting');priors=reporting_priors(a,base,prepared,evaluation,metadata,inputs,device,out,status,bank['checkpoint_sha256']);valbase=[json.loads(l) for l in (ref/'validation_selected.jsonl').read_text().splitlines()];baseline=[json.loads(l) for l in (ref/'selected_predictions.jsonl').read_text().splitlines()];oldmissing=[json.loads(l) for l in (ref/'missing_predictions.jsonl').read_text().splitlines()];context=EnvContext(bank,metadata,labels,normal,reporting,bank['single_indices'],device,matching=a.matching);protocol=dict(reference=str(ref),matching=a.matching,H_checkpoint_sha256=bank['checkpoint_sha256'],query_sha256=stamp_queries(bank['test']),candidates=len(context.smiles),source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')});(out/'protocol.json').write_text(json.dumps(protocol,indent=2));frozen=validation(bank,context,valbase,priors,inputs,out,status,dict(soft=SOFT_GRID,density=DENSITY_GRID,hard=GRID)[a.matching]);status('test');test(bank,context,baseline,oldmissing,frozen['selected_config'],priors,inputs,out);del context;torch.cuda.empty_cache();context=EnvContext(bank,metadata,labels,normal,reporting,bank['joint_indices'],device,matching=a.matching);joint(base,bank,context,cfg,out,status,h_override=lambda q,c:routed_score(q,c,frozen['selected_config'],priors,inputs));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

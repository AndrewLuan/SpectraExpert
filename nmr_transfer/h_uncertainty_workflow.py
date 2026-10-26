"""功能概述：在条件均值模型和观测评分冻结后，单独检验局部环境误差宽度。
fit_widths只用train实验原子侧表的母节点混合似然；
--fit-site-quality可对甲基标签冲突记录停用位点拟合，原谱和查询标签保留；predict_tags在原候选次序上
优先读取同次发射保存的分位点类别；旧bank无类别时才尝试重建。固定均值不重算，
按最近显式发射匹配类别并核对支持
距离；CDF边界浮点微扰另存审计，不替换冻结均值。不会将实验归属给查询评分。
VariableContext只替换稀疏原子谱的密度；积分谱、HMDB和显式频窗继续用原评分。
原latent条件/报告计数配置保持固定，统一宽度插值lambda=0/0.5/1只由原320-val
MRR选择。--fixed-lambda可预先声明迁移旧轮插值作为单独消融，仍保存完整val网格；不按test选值。
baseline零插值直接调用旧密度，保持原评分路径；然后才评1000条test、
固定缺峰/裁窗和792对双核。C模型/候选/逐查询控制不变。
恢复溶剂侧表按bank.condition_key_mode投影，旧bank使用旧字段，已知条件与训练先验保持一致。
输出独占训练宽度、标签、验证选择、逐条排名、source/checkpoint/query身份和退出码；
失败保留日志，不静默替换测试或模型。依赖h_latent_workflow的固定cohort接口。
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .h_uncertainty import CLASSES,WIDTH_GRID,parent_grid_nll,select_widths,quantile_classes,class_density
from .h_latent_workflow import Context,load_reference,query_observation,prepare_carbon_queries
from .h_latent_score import eligible,features,score,bank_key
from .h_condinfer import candidate_batches,scoring_inputs
from .h_conditions import ConditionsSpectra
from .h_labels import make_key
from .h_site_quality import apply_quality
from .h_gapworkflow import resolve_checkpoint
from .emission_model import make_model
from .train import collate
from .h_observation_study import write_rows,summarize_sources,stamp_queries
from .h_observation_finalize import query_score
from .h_diagnose import cond_key
from .likelihood import density,log_score
from .evaluate import missing_queries,summarize
from .prepare import sha
from .h_metadata import metadata_view


@torch.inference_mode()
def fit_widths(model,cache,metadata,labels,device,out,site_quality='original'):
    rows=[r for r in cache['rows'] if r['split']=='train' and r['domain']=='experimental' and labels.get(make_key(r),{}).get('groups')];fit_labels=dict(labels);quality=apply_quality(fit_labels,rows,site_quality);rows=[r for r in rows if fit_labels.get(make_key(r),{}).get('groups')];loader=DataLoader(ConditionsSpectra(rows,cache['graphs'],'1H',metadata,annotations=fit_labels),batch_size=256,num_workers=2,collate_fn=collate);totals=torch.zeros(len(CLASSES),len(WIDTH_GRID),device=device);counts=torch.zeros(len(CLASSES),device=device)
    for b in loader:
        b=b.to(device);d=model(b,details=True);costs,nn=parent_grid_nll(d,b);totals+=costs;counts+=nn
    widths=select_widths(totals,counts);fit=dict(site_quality=quality,classes=CLASSES,width_grid=WIDTH_GRID,selected_ppm=widths,training_parent_counts=counts.cpu().tolist(),training_nll_sums=totals.cpu().tolist(),prior_strength_parents=50.,training_records=len(rows),training_query_sha256=stamp_queries(rows),scope='supplied-parent likelihood in experimental train with explicit consistency policy; no test or val labels fit widths');(out/'width_fit.json').write_text(json.dumps(fit,indent=2));return widths


@torch.inference_mode()
def predict_tags(model,bank,graphs,device,status,out):
    if 'candidate_quantile_classes' in bank:
        tags=bank['candidate_quantile_classes']
        if tags.keys()!=bank['predictions'].keys():raise ValueError('same_forward_tag_conditions_changed')
        for key,value in tags.items():
            if value.shape!=bank['predictions'][key].shape or (value>=len(CLASSES)).any():raise ValueError('invalid_same_forward_tags')
        (out/'quantile_support_audit.json').write_text(json.dumps(dict(source='same forward pass as frozen prediction bank',means='frozen bank values retained exactly',conditions=len(tags))))
        return tags
    batches=candidate_batches(bank,graphs,device);tags={};audit=[]
    for j,key in enumerate(sorted(bank['predictions'])):
        if len(key)!=31:raise ValueError('conditional_bank_required')
        status('candidate_environment_tags',condition=j+1,total=len(bank['predictions']));pieces=[];support_error=0.;cdf_error=0.;offset=0
        for b in batches:
            b.cond=torch.tensor(key[:4],device=device).reshape(1,4).expand(b.num_graphs,-1);b.condition_vector=torch.tensor(key[4:],device=device).reshape(1,-1).expand(b.num_graphs,-1);d=model(b,details=True);frozen=bank['predictions'][key][offset:offset+b.num_graphs].to(device);cdf_error=max(cdf_error,float((d['quantiles']-frozen).abs().max()));c,support=quantile_classes(d,b,reference=frozen);support_error=max(support_error,float((support-frozen).abs().max()));pieces.append(c.cpu());offset+=b.num_graphs
        if support_error>2e-6:raise ValueError('frozen_quantiles_off_emission_support:'+str(support_error))
        tags[key]=torch.cat(pieces);audit.append(dict(condition=list(key),max_CDF_recompute_difference_ppm=cdf_error*10,max_nearest_support_distance_ppm=support_error*10,means='frozen bank values retained exactly'))
    (out/'quantile_support_audit.json').write_text(json.dumps(audit,indent=2));del batches;torch.cuda.empty_cache();return tags


class VariableContext(Context):
    def __init__(self,*args,tags,widths,**kwargs):
        super().__init__(*args,**kwargs);self.tags=tags;self.widths=widths
        for name,key in self.keys.items():self.prototypes[name]=class_density(self.bank['predictions'][key][self.indices].to(self.device),self.tags[key][self.indices].to(self.device),self.widths)
    def set_condition(self,key,need_hmdb=False):
        super().set_condition(key,need_hmdb);self.variable_given=class_density(self.bank['predictions'][key][self.indices].to(self.device),self.tags[key][self.indices].to(self.device),self.widths)
    def featurize(self,q):return features(q,dict(self.prototypes,given=self.variable_given),self.counts,self.normal_prior,self.reporting)


def context_for(lam,bank,metadata,labels,normal,reporting,indices,device,tags,widths):
    if lam==0:return Context(bank,metadata,labels,normal,reporting,indices,device)
    interpolated=(.2**(1-lam)*np.asarray(widths)**lam).tolist();return VariableContext(bank,metadata,labels,normal,reporting,indices,device,tags=tags,widths=interpolated)


@torch.inference_mode()
def rank_single(bank,context,baseline,oldmissing,cfg,split,stress=False):
    rows=bank[split];lookup={(r['source'],r['record_id'],r['smiles']):r for r in baseline};oldstress={(r['variant'],r['record_id'],r['smiles']):r for r in oldmissing};records=[];missing=[]
    for key in sorted({bank_key(bank,r,context.metadata) for r in rows}):
        group=[(i,r) for i,r in enumerate(rows) if bank_key(bank,r,context.metadata)==key]
        if any(eligible(query_observation(r,context.metadata,context.labels)) for _,r in group):context.set_condition(key)
        for i,r in group:
            q=query_observation(r,context.metadata,context.labels);old=lookup[r['source'],r['record_id'],r['smiles']];result=context.rank(score(context.featurize(q),cfg),q) if eligible(q) else dict(old);records.append(dict(result,query_order=i))
            if stress and i<200:
                for variant,rr in missing_queries(q,'1H'):
                    result=context.rank(score(context.featurize(rr),cfg),rr) if eligible(rr) else dict(oldstress[variant,r['record_id'],r['smiles']]);missing.append(dict(result,variant=variant))
    records.sort(key=lambda r:r['query_order']);return records,missing


@torch.inference_mode()
def joint(base,bank,context,cfg,out,status,h_override=None):
    folder=out/'joint';folder.mkdir();carbon=torch.load(base/'hydrogen_analysis/observation_comparison/joint/carbon_bank.pt',map_location='cpu',weights_only=False,mmap=True)
    if context.smiles!=carbon['smiles']:raise ValueError('C_candidate_order_changed')
    cqueries=prepare_carbon_queries(base,bank,out);ck={key:density(q.to(context.device),'13C',2.) for key,q in carbon['predictions'].items()};old=[json.loads(l) for l in (base/'hydrogen_analysis/observation_comparison/joint/predictions.jsonl').read_text().splitlines()];control={(r['H_source'],r['H_record_id'],r['smiles']):r['13C']['rank'] for r in old};frozen,prior,config,priors,raw=scoring_inputs(base);records=[];stress=[]
    for key in sorted({bank_key(bank,r,context.metadata) for r in bank['paired']}):
        group=[(i,r,cqueries[i]) for i,r in enumerate(bank['paired']) if bank_key(bank,r,context.metadata)==key];context.set_condition(key,any(r['source']=='HMDB' for _,r,_ in group));status('joint_ranking',ranked=len(records))
        for i,r,c in group:
            q=query_observation(r,context.metadata,context.labels)
            def hs(query):
                if h_override is not None and query['source']=='HMDB':return h_override(query,context)
                return score(context.featurize(query),cfg) if eligible(query) else query_score(query,context.given,context.counts,prior,config,frozen,priors,raw)
            h=hs(q);cs=log_score(ck[cond_key(c)],c,'13C');item=dict(query_order=i,smiles=r['smiles'],H_source=r['source'],H_record_id=r['record_id'])
            for name,s in [('1H',h),('13C',cs),('joint',.5*(h+cs))]:item[name]=context.rank(s,r)
            if item['13C']['rank']!=control[r['source'],r['record_id'],r['smiles']]:raise ValueError('C_rank_changed')
            records.append(item)
            if i<200:
                for variant,rr in missing_queries(q,'1H'):stress.append(dict(variant='1H_'+variant,**context.rank(.5*(hs(rr)+cs),r)))
                for variant,rr in missing_queries(c,'13C'):stress.append(dict(variant='13C_'+variant,**context.rank(.5*(h+log_score(ck[cond_key(c)],rr,'13C')),r)))
    records.sort(key=lambda r:r['query_order'])
    if len(records)!=792:raise ValueError('joint_cohort_changed')
    write_rows(folder/'predictions.jsonl',records);write_rows(folder/'missing_predictions.jsonl',stress);(folder/'summary.json').write_text(json.dumps(dict(queries=792,candidates=len(context.smiles),metrics={n:summarize([r[n] for r in records]) for n in ['1H','13C','joint']},missing_stress={v:summarize([r for r in stress if r['variant']==v]) for v in sorted({r['variant'] for r in stress})},C_per_query_control_equal=True),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit-site-quality',choices=['original','methyl_record_mask'],default='original');p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--reference-latent',required=True);p.add_argument('--out',required=True);p.add_argument('--fixed-lambda',type=float,choices=[0.,.5,1.]);a=p.parse_args();base=Path(a.base);ref=Path(a.reference_latent);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        status('waiting_reference');deadline=time.monotonic()+10800
        while not (ref/'EXIT_CODE').exists():
            if time.monotonic()>deadline:raise TimeoutError('reference_timeout')
            time.sleep(10)
        if (ref/'EXIT_CODE').read_text().strip()!='0':raise RuntimeError('reference_failed')
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');protocol=json.loads((ref/'protocol.json').read_text());evaluation=Path(protocol['reference_evaluation']);status('loading');bank,_,_,_,_=load_reference(base,evaluation);cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=metadata_view(torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False),bank.get('condition_key_mode'));labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False);checkpoint=resolve_checkpoint(base,evaluation);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict']);status('train_widths');widths=fit_widths(model,cache,metadata,labels,device,out,a.fit_site_quality);tags=predict_tags(model,bank,cache['graphs'],device,status,out);torch.save(tags,out/'candidate_quantile_classes.pt');del cache,model;torch.cuda.empty_cache()
        normal=json.loads((base/'hydrogen_analysis/routed_evaluation/site/count_prior.json').read_text());reporting=json.loads((ref/'reporting_priors.json').read_text());cfg=json.loads((ref/'frozen_selection.json').read_text())['selected_config'];baseline=[json.loads(l) for l in (ref/'selected_predictions.jsonl').read_text().splitlines()];valbase=[json.loads(l) for l in (ref/'validation_selected.jsonl').read_text().splitlines()];oldmissing=[json.loads(l) for l in (ref/'missing_predictions.jsonl').read_text().splitlines()];trials={};valrows={}
        for lam in [0.,.5,1.]:
            status('validation',width_lambda=lam);context=context_for(lam,bank,metadata,labels,normal,reporting,bank['single_indices'],device,tags,widths);rs,_=rank_single(bank,context,valbase,[],cfg,'val');trials[str(lam)]=summarize_sources(rs);valrows[str(lam)]=rs;del context;torch.cuda.empty_cache()
        baseline_map={(r['source'],r['record_id'],r['smiles']):r['rank'] for r in valbase}
        if any(r['rank']!=baseline_map[r['source'],r['record_id'],r['smiles']] for r in valrows['0.0']):raise ValueError('zero_width_baseline_rank_changed')
        selected=max(trials,key=lambda k:trials[k]['overall']['MRR']) if a.fixed_lambda is None else str(a.fixed_lambda);frozen=dict(selected_lambda=float(selected),validation=trials,scoring_config=cfg,reference=str(ref),checkpoint_sha256=sha(checkpoint),query_sha256=stamp_queries(bank['test']),source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},selection='only fixed320 validation MRR; all widths fit experimental train, scoring unchanged' if a.fixed_lambda is None else 'predeclared fixed lambda transfer ablation; val grid retained; no test selection');(out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));write_rows(out/'validation_selected.jsonl',valrows[selected])
        status('test');context=context_for(float(selected),bank,metadata,labels,normal,reporting,bank['single_indices'],device,tags,widths);records,missing=rank_single(bank,context,baseline,oldmissing,cfg,'test',True)
        if len(records)!=1000 or sum(r['variant']=='peak_dropout' for r in missing)!=171:raise ValueError('single_query_cohort_changed')
        write_rows(out/'predictions.jsonl',records);write_rows(out/'missing_predictions.jsonl',missing);(out/'summary.json').write_text(json.dumps(dict(candidates=len(context.smiles),baseline=summarize_sources(baseline),selected=summarize_sources(records),missing_stress={v:dict(baseline=summarize([r for r in oldmissing if r['variant']==v]),selected=summarize([r for r in missing if r['variant']==v])) for v in ['peak_dropout','frequency_crop']}),indent=2));del context;torch.cuda.empty_cache();context=context_for(float(selected),bank,metadata,labels,normal,reporting,bank['joint_indices'],device,tags,widths);joint(base,bank,context,cfg,out,status);status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

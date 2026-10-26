"""功能概述：在冻结均值和候选库上检验羰基CH整峰未报告的潜状态模型。
fit_reporting只读实验train HMDB，模型同次发射得到full/no-carbonyl两种似然；
按甲酰胺/醛、水/非水及明确溶剂分层，连接结构均权，稀有条件向训练父层收缩。另对原
missing_queries整峰删除增强拟合压力先验；不把峰高当积分或未知频窗当完整谱。
FormylContext复用原latent Context及同次分位环境tag，只改变HMDB报告似然；
显式频窗沿用旧评分，原逐原子/积分观测路由不改。每个候选使用同一规则。
main在固定320条val选择baseline/半强度/全强度，然后原1000查询及固定压力变体、
792双核对复评；joint复用原逐查询C控制。配置选择在test前写入磁盘。
--fixed-strength只用于预先声明的非选型消融，仍保存完整val网格；不能按测试表现选强度。
输出独占、源码/权重/查询身份和EXIT_CODE；不改旧bank、候选、原始谱或测试身份。
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .h_formyl_reporting import GRID,candidate_families,counterfactual_quantiles,mixture_score,fit_probability
from .h_latent_workflow import Context,load_reference,query_observation
from .h_latent_score import bank_key
from .h_conditions import ConditionsSpectra,SOLVENTS
from .h_metadata import metadata_view
from .h_uncertainty import quantile_classes
from .h_condinfer import scoring_inputs
from .h_gapworkflow import resolve_checkpoint
from .h_labels import make_key
from .h_observation_finalize import query_score,visible_lines,blend
from .h_observed import line_score,cluster_lines
from .h_observation_study import uniform_peaks,SHIFTS,write_rows,summarize_sources,stamp_queries
from .h_uncertainty_workflow import joint
from .likelihood import density
from .emission_model import make_model,weighted_quantiles
from .train import collate
from .evaluate import missing_queries
from .prepare import sha
from .h_env_workflow import test as fixed_test


def nearest_tags(reference,full,tags):
    right=torch.searchsorted(full.contiguous(),reference.contiguous()).clamp_max(full.shape[1]-1);left=(right-1).clamp_min(0)
    idx=torch.where((full.gather(1,right)-reference).abs()<(full.gather(1,left)-reference).abs(),right,left)
    return tags.gather(1,idx)


def reporting_kernel(full,fixed,tags,families,cfg):
    missing,active=counterfactual_quantiles(full,tags,families);kind=cfg['kernel']
    kernel=density(uniform_peaks(missing) if kind=='distinct' else missing,'1H',.2)
    if 'visibility' in cfg:
        ftags=nearest_tags(fixed,full,tags);altered,_=counterfactual_quantiles(fixed,ftags,families)
        fk=density(uniform_peaks(altered) if kind=='distinct' else altered,'1H',.2);kernel=blend(kernel,fk,cfg['visibility'])
    return kernel,active


def selected_hmdb_config(key,inputs):
    frozen=inputs[0];return frozen['water_HMDB'] if key[1]>0 and frozen['water_HMDB'] is not None else frozen['HMDB']


def solvent_group(key):
    return '+'.join(name for name,v in zip(SOLVENTS,key[4:4+len(SOLVENTS)]) if v>0) or 'unknown'


@torch.inference_mode()
def fit_reporting(base,prepared,evaluation,metadata,inputs,device,out,status):
    checkpoint=resolve_checkpoint(base,evaluation);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict']);cache=torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True)
    rows=[r for r in cache['rows'] if r['domain']=='experimental' and r['split']=='train' and r['source']=='HMDB'];samples=[]
    loader=DataLoader(ConditionsSpectra(rows,cache['graphs'],'1H',metadata),batch_size=64,num_workers=2,collate_fn=collate);offset=0
    for b in loader:
        d=model(b.to(device),details=True);tags,checked=quantile_classes(d,b)
        if not torch.equal(checked,d['quantiles']):raise ValueError('training_tag_support_changed')
        full=d['quantiles'];keep=~d['exchangeable'];fixed=weighted_quantiles(d['values'][keep],d['weights'][keep],d['groups'][keep],b.num_graphs)
        for j,r in enumerate(rows[offset:offset+b.num_graphs]):
            family=candidate_families([r['smiles']])[0]
            if family=='other' or r['observation']['windows']:continue
            key=tuple(b.cond[j].cpu().tolist());cfg=selected_hmdb_config(key,inputs)
            if cfg is None:continue
            pp=full[j:j+1];ff=fixed[j:j+1];tt=tags[j:j+1];missing,active=reporting_kernel(pp,ff,tt,[family],cfg)
            if not bool(active[0]):continue
            original=density(uniform_peaks(pp) if cfg['kernel']=='distinct' else pp,'1H',.2)
            if 'visibility' in cfg:original=blend(original,density(uniform_peaks(ff) if cfg['kernel']=='distinct' else ff,'1H',.2),cfg['visibility'])
            for name,q in [('normal',r)]+[(v,rr) for v,rr in missing_queries(r,'1H') if v=='peak_dropout']:
                xs=visible_lines(q,inputs[-1]);k=len(cluster_lines(xs,cfg['width'])[0])
                if not k:continue
                f=float(line_score(original,q,xs,cfg['width'],SHIFTS,cfg['shift_sigma'])[0]);m=float(line_score(missing,q,xs,cfg['width'],SHIFTS,cfg['shift_sigma'])[0])
                samples.append(dict(group=r['group'],record_id=r['record_id'],family=family,water=key[1]>0,solvent=solvent_group(tuple(b.cond[j].cpu().tolist())+tuple(b.condition_vector[j].cpu().tolist())),process='dropout' if name=='peak_dropout' else 'normal',full_loglike=-k*f,missing_loglike=-k*m))
        offset+=b.num_graphs;status('fit_reporting',processed=offset,total=len(rows))
    priors={}
    for process in ['normal','dropout']:
        priors[process]={}
        for family in ['formamide','aldehyde']:
            ss=[s for s in samples if s['process']==process and s['family']==family];parent=fit_probability(ss) if ss else fit_probability([],prior=1.);levels={'all':parent}
            for water in [False,True]:levels['water' if water else 'nonwater']=fit_probability([s for s in ss if s['water']==water],parent['probability'])
            for solvent in sorted({s['solvent'] for s in ss}):
                subset=[s for s in ss if s['solvent']==solvent];levels['solvent:'+solvent]=fit_probability(subset,levels['water' if subset[0]['water'] else 'nonwater']['probability'])
            priors[process][family]=levels
    write_rows(out/'training_reporting_records.jsonl',samples);(out/'training_reporting_priors.json').write_text(json.dumps(priors,indent=2));del model,cache;torch.cuda.empty_cache();return priors


class FormylContext(Context):
    def __init__(self,*args,inputs,**kwargs):
        super().__init__(*args,**kwargs);self.inputs=inputs;self.families=candidate_families(self.smiles)
        if 'candidate_quantile_classes' not in self.bank:raise ValueError('same_forward_tags_required')

    def set_condition(self,key,need_hmdb=False):
        super().set_condition(key,need_hmdb);self.formyl_kernel=None
        if not need_hmdb:return
        cfg=selected_hmdb_config(key,self.inputs)
        if cfg is None:return
        pred=self.given['pred'];fixed=self.bank['fixed_predictions'][key][self.indices].to(self.device) if key[1]>0 else pred;tags=self.bank['candidate_quantile_classes'][key][self.indices].to(self.device)
        self.formyl_kernel,self.active=reporting_kernel(pred,fixed,tags,self.families,cfg)

    def reporting_features(self,q,priors):
        frozen,prior,config,oldpriors,raw=self.inputs;full=query_score(q,self.given,self.counts,prior,config,frozen,oldpriors,raw)
        if self.formyl_kernel is None or q['observation']['windows']:return dict(full=full,missing=full,p=torch.ones_like(full),k=1)
        cfg=selected_hmdb_config(self.current,self.inputs);xs=visible_lines(q,raw);k=len(cluster_lines(xs,cfg['width'])[0]);missing=line_score(self.formyl_kernel,q,xs,cfg['width'],SHIFTS,cfg['shift_sigma']);process='dropout' if q['observation'].get('partial_peaks') else 'normal';water='water' if self.current[1]>0 else 'nonwater';solvent='solvent:'+solvent_group(self.current);p=torch.tensor([priors[process][f].get(solvent,priors[process][f][water])['probability'] if f!='other' else 1. for f in self.families],device=self.device)
        return dict(full=full,missing=torch.where(self.active,missing,full),p=p,k=k)


def reporting_score(features,choice):
    return features['full'] if choice['kind']=='baseline' else mixture_score(features['full'],features['missing'],features['p'],features['k'],choice['strength'])


def routed(q,context,choice,priors,inputs):
    return reporting_score(context.reporting_features(q,priors),choice)


def main():
    p=argparse.ArgumentParser()
    for name in ['base','prepared','reference-latent','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--fixed-strength',type=float,choices=[.5,1.])
    a=p.parse_args();base=Path(a.base);prepared=Path(a.prepared);ref=Path(a.reference_latent);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');evaluation=Path(json.loads((ref/'protocol.json').read_text())['reference_evaluation']);bank,_,_,_,_=load_reference(base,evaluation);metadata=metadata_view(torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False),bank.get('condition_key_mode'));labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False);inputs=scoring_inputs(base);priors=fit_reporting(base,prepared,evaluation,metadata,inputs,device,out,status);normal=json.loads((base/'hydrogen_analysis/routed_evaluation/site/count_prior.json').read_text());reporting=json.loads((ref/'reporting_priors.json').read_text());latent_cfg=json.loads((ref/'frozen_selection.json').read_text())['selected_config']
        def context(indices):return FormylContext(bank,metadata,labels,normal,reporting,indices,device,inputs=inputs)
        c=context(bank['single_indices']);valbase=[json.loads(l) for l in (ref/'validation_selected.jsonl').read_text().splitlines()];lookup={(r['source'],r['record_id'],r['smiles']):r for r in valbase};trials={str(i):[] for i in range(len(GRID))}
        for key in sorted({bank_key(bank,r,metadata) for r in bank['val']}):
            group=[(i,r) for i,r in enumerate(bank['val']) if bank_key(bank,r,metadata)==key]
            if any(r['source']=='HMDB' for _,r in group):c.set_condition(key,True)
            for order,r in group:
                q=query_observation(r,metadata,labels);old=lookup[r['source'],r['record_id'],r['smiles']];ff=c.reporting_features(q,priors) if r['source']=='HMDB' else None
                for i,cfg in enumerate(GRID):trials[str(i)].append(dict(c.rank(reporting_score(ff,cfg),q) if ff is not None and i else old,query_order=order))
            status('validation',records=len(trials['0']))
        for rs in trials.values():rs.sort(key=lambda r:r['query_order'])
        metrics={i:summarize_sources(rs) for i,rs in trials.items()};best=max(metrics,key=lambda i:metrics[i]['overall']['MRR']) if a.fixed_strength is None else str(next(i for i,cfg in enumerate(GRID) if cfg.get('strength')==a.fixed_strength));choice=GRID[int(best)];frozen=dict(selected_config=choice,grid=GRID,validation=metrics,query_sha256=stamp_queries(bank['test']),H_checkpoint_sha256=bank['checkpoint_sha256'],source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},selection='same320 val; train-only reporting fit; no test selection' if a.fixed_strength is None else 'predeclared fixed-strength diagnostic ablation; complete val grid retained; not model selection',fixed_strength=a.fixed_strength);(out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));write_rows(out/'validation_selected.jsonl',trials[best]);baseline=[json.loads(l) for l in (ref/'selected_predictions.jsonl').read_text().splitlines()];oldmissing=[json.loads(l) for l in (ref/'missing_predictions.jsonl').read_text().splitlines()]
        fixed_test(bank,c,baseline,oldmissing,choice,priors,inputs,out,score_function=routed)
        del c;torch.cuda.empty_cache();c=context(bank['joint_indices']);joint(base,bank,c,latent_cfg,out,status,h_override=lambda q,c:routed(q,c,choice,priors,inputs));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

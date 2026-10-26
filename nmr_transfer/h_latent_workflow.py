"""功能概述：固定模型均值后，对稀疏逐原子谱表做缺失条件/参考偏移/报告计数消融。
load_reference只取上一轮验证优胜者的bank、原320/1000排名；模型、候选身份不改。
select遍历预先声明的48种评分并保留原baseline，非原子标注记录直接沿用原排名；
固定val选型后才评原1000查询和171删峰/200裁窗，另保存移除每个机制的消融。
Context缓存三个溶剂原型密度，每个查询只使用实验条件、报告位置/计数和候选模型。
prepare_carbon_queries仅缓存原792对的C观测，后续双核不重复加载大缓存；H/C模型和
C候选预测、双核等权都固定。所有控制按记录身份核对，不依赖条件分组后的文件顺序。
恢复溶剂侧表按bank.condition_key_mode投影，旧bank使用旧字段，已知条件与训练先验保持一致。
输出独占、明确退出码/源码hash/冻结配置。该流程是开发复评，不把测试作为选参集。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .h_latent_score import CONFIGS,PROTOTYPES,prototype_keys,bank_key,fit_reporting_priors,eligible,features,score
from .h_score import attach_observation
from .h_labels import make_key
from .h_condinfer import query_kernels,scoring_inputs
from .h_observation_finalize import query_score
from .h_observation_study import entry,write_rows,summarize_sources,stamp_queries
from .h_eval import LazyFP
from .evaluate import missing_queries,summarize
from .h_diagnose import cond_key
from .likelihood import density,log_score
from .prepare import sha
from .h_metadata import metadata_view


def load_reference(base,reference):
    selected=json.loads((reference/'frozen_selection.json').read_text())['selected_variant']
    if selected=='baseline':
        bank=torch.load(base/'hydrogen_analysis/evaluation_dropout/site/bank.pt',map_location='cpu',weights_only=False,mmap=True)
        fixed=torch.load(base/'hydrogen_analysis/water_quantile_visibility/bank.pt',map_location='cpu',weights_only=False,mmap=True)
        if fixed['smiles']!=bank['smiles']:raise ValueError('fixed_candidate_order_mismatch')
        bank['fixed_predictions']=fixed['predictions'];folder=base/'hydrogen_analysis/observation_comparison'
        validation=[json.loads(l) for l in (reference/'baseline_validation.jsonl').read_text().splitlines()]
        test=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()];missing=[json.loads(l) for l in (folder/'missing_predictions.jsonl').read_text().splitlines()]
    else:
        folder=reference/selected;bank=torch.load(folder/'bank.pt',map_location='cpu',weights_only=False,mmap=True)
        validation=[json.loads(l) for l in (folder/'val/predictions.jsonl').read_text().splitlines()];test=[json.loads(l) for l in (folder/'test/predictions.jsonl').read_text().splitlines()];missing=[json.loads(l) for l in (folder/'test/missing_predictions.jsonl').read_text().splitlines()]
    return bank,validation,test,missing,selected


def query_observation(row,metadata,labels):
    q=attach_observation(row,labels.get(make_key(row),{}));meta=metadata.get(make_key(row));q['_condition_known']=meta['metadata_known'] if meta is not None else bool(np.any(np.asarray(row['cond'])[1:]>0));return q


class Context:
    def __init__(self,bank,metadata,labels,normal_prior,reporting,indices,device):
        self.bank=bank;self.metadata=metadata;self.labels=labels;self.normal_prior=normal_prior;self.reporting=reporting;self.indices=indices;self.device=device
        self.smiles=[bank['smiles'][i] for i in indices];self.lookup={s:i for i,s in enumerate(self.smiles)};self.counts=bank['physics'][indices,0].to(device)*40;self.fps=LazyFP();self.prototypes={}
        self.keys=prototype_keys(bank)
        for name,key in self.keys.items():
            if key not in bank['predictions']:raise ValueError('missing_prototype:'+name)
            self.prototypes[name]=density(bank['predictions'][key][indices].to(device),'1H',.2)
        self.current=None;self.given=None
    def set_condition(self,key,need_hmdb=False):
        self.current=key;self.given=query_kernels(self.bank,self.indices,key,self.device,need_hmdb)
    def featurize(self,q):
        return features(q,dict(self.prototypes,given=self.given['mass']),self.counts,self.normal_prior,self.reporting)
    def rank(self,scores,q):return entry(scores,q,self.lookup,self.smiles,self.fps)


@torch.inference_mode()
def validation(bank,context,baseline,out):
    rows=bank['val'];by_id={(r['source'],r['record_id'],r['smiles']):r for r in baseline};trials={str(i):[] for i in range(len(CONFIGS))}
    for key in sorted({bank_key(bank,r,context.metadata) for r in rows}):
        group=[r for r in rows if bank_key(bank,r,context.metadata)==key]
        if any(eligible(query_observation(r,context.metadata,context.labels)) for r in group):context.set_condition(key)
        for r in group:
            q=query_observation(r,context.metadata,context.labels);old=by_id[r['source'],r['record_id'],r['smiles']];ff=context.featurize(q) if eligible(q) else None
            for i,cfg in enumerate(CONFIGS):trials[str(i)].append(dict(old) if ff is None or i==0 else context.rank(score(ff,cfg),q))
        print(json.dumps(dict(validation_ranked=len(trials['0']))),flush=True)
    order={(r['source'],r['record_id'],r['smiles']):i for i,r in enumerate(rows)}
    if len(order)!=len(rows):raise ValueError('duplicate_validation_identity')
    for rs in trials.values():rs.sort(key=lambda r:order[r['source'],r['record_id'],r['smiles']])
    metrics={i:summarize_sources(rs) for i,rs in trials.items()};best=max(metrics,key=lambda i:metrics[i]['overall']['MRR']);selected=CONFIGS[int(best)]
    ablations=dict(selected=selected,no_condition=dict(selected,condition='given'),no_offset=dict(selected,shift_sigma=0.),no_count=dict(selected,beta=0.))
    frozen=dict(selected_index=best,selected_config=selected,ablations=ablations,config_grid=CONFIGS,validation=metrics,reference_only_baseline=metrics['0'],selection='same 320 validation queries; all model/score choices frozen before test',development_test_reuse=True)
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));write_rows(out/'validation_baseline.jsonl',baseline);write_rows(out/'validation_selected.jsonl',trials[best])
    np.savez_compressed(out/'validation_ranks.npz',ranks=np.array([[r['rank'] for r in trials[str(i)]] for i in range(len(CONFIGS))],np.int32))
    return frozen


@torch.inference_mode()
def test(bank,context,baseline,oldmissing,frozen,out):
    by_id={(r['source'],r['record_id'],r['smiles']):r for r in baseline};missing_id={(r['variant'],r['record_id'],r['smiles']):r for r in oldmissing};records={n:[] for n in frozen['ablations']};stress=[]
    for key in sorted({bank_key(bank,r,context.metadata) for r in bank['test']}):
        group=[(i,r) for i,r in enumerate(bank['test']) if bank_key(bank,r,context.metadata)==key]
        if any(eligible(query_observation(r,context.metadata,context.labels)) for _,r in group):context.set_condition(key)
        for order,r in group:
            q=query_observation(r,context.metadata,context.labels);old=by_id[r['source'],r['record_id'],r['smiles']];ff=context.featurize(q) if eligible(q) else None
            for name,cfg in frozen['ablations'].items():
                result=dict(old) if ff is None or cfg==CONFIGS[0] else context.rank(score(ff,cfg),q);records[name].append(dict(result,query_order=order))
            if order>=200:continue
            for variant,rr in missing_queries(q,'1H'):
                old=missing_id[variant,r['record_id'],r['smiles']];result=dict(old);cfg=frozen['selected_config']
                if eligible(rr) and cfg!=CONFIGS[0]:result=context.rank(score(context.featurize(rr),cfg),rr)
                stress.append(dict(result,variant=variant,source=r['source']))
        print(json.dumps(dict(test_ranked=len(records['selected']))),flush=True)
    for name,rs in records.items():rs.sort(key=lambda r:r['query_order']);write_rows(out/(name+'_predictions.jsonl'),rs)
    if len(records['selected'])!=1000 or sum(r['variant']=='peak_dropout' for r in stress)!=171:raise ValueError('query_cohort_changed')
    report=dict(candidates=len(context.smiles),baseline=summarize_sources(baseline),optimized={name:summarize_sources(rs) for name,rs in records.items()},missing_stress={v:dict(baseline=summarize([r for r in oldmissing if r['variant']==v]),selected=summarize([r for r in stress if r['variant']==v])) for v in ['peak_dropout','frequency_crop']},reference_H_checkpoint_sha256=bank['checkpoint_sha256'],query_sha256=stamp_queries(bank['test']))
    write_rows(out/'missing_predictions.jsonl',stress);(out/'summary.json').write_text(json.dumps(report,indent=2));return report


def prepare_carbon_queries(base,bank,out):
    path=base/'hydrogen_analysis/condition_response/carbon_queries.pt'
    original=[json.loads(l) for l in (base/'atom_emission/joint_likelihood/predictions.jsonl').read_text().splitlines()]
    if path.exists():rows=torch.load(path,map_location='cpu',weights_only=False)
    else:
        cache=torch.load(base/'prepared_coverage/cache.pt',map_location='cpu',weights_only=False,mmap=True);lookup={(r['smiles'],r['source'],r['record_id']):r for r in cache['rows'] if r['nucleus']=='13C' and r['domain']=='experimental'};rows=[lookup[r['smiles'],r['sources']['13C'],r['record_ids']['13C']] for r in original];torch.save(rows,path)
    if [(r['smiles'],r['source'],r['record_id']) for r in rows]!=[(r['smiles'],r['sources']['13C'],r['record_ids']['13C']) for r in original]:raise ValueError('C_query_cache_mismatch')
    return rows


@torch.inference_mode()
def joint(base,bank,metadata,labels,normal_prior,reporting,frozen,out,device,status):
    folder=out/'joint';folder.mkdir();context=Context(bank,metadata,labels,normal_prior,reporting,bank['joint_indices'],device);carbon=torch.load(base/'hydrogen_analysis/observation_comparison/joint/carbon_bank.pt',map_location='cpu',weights_only=False,mmap=True)
    if context.smiles!=carbon['smiles']:raise ValueError('C_candidates_changed')
    cqueries=prepare_carbon_queries(base,bank,out);ck={key:density(q.to(device),'13C',2.) for key,q in carbon['predictions'].items()};old=[json.loads(l) for l in (base/'hydrogen_analysis/observation_comparison/joint/predictions.jsonl').read_text().splitlines()];c_control={(r['H_source'],r['H_record_id'],r['smiles']):r['13C']['rank'] for r in old};oldfrozen,prior,config,priors,raw=scoring_inputs(base);records=[];stress=[];cfg=frozen['selected_config']
    for key in sorted({bank_key(bank,r,metadata) for r in bank['paired']}):
        group=[(i,r,cqueries[i]) for i,r in enumerate(bank['paired']) if bank_key(bank,r,metadata)==key];context.set_condition(key,any(r['source']=='HMDB' for _,r,_ in group));status('joint_ranking',ranked=len(records))
        for i,r,c in group:
            q=query_observation(r,metadata,labels)
            def hs(query):
                if eligible(query) and cfg!=CONFIGS[0]:return score(context.featurize(query),cfg)
                return query_score(query,context.given,context.counts,prior,config,oldfrozen,priors,raw)
            h=hs(q);carbon_score=log_score(ck[cond_key(c)],c,'13C');item=dict(query_order=i,smiles=r['smiles'],H_source=r['source'],H_record_id=r['record_id'])
            for name,scores in [('1H',h),('13C',carbon_score),('joint',.5*(h+carbon_score))]:item[name]=context.rank(scores,r)
            if item['13C']['rank']!=c_control[r['source'],r['record_id'],r['smiles']]:raise ValueError('C_control_rank_changed')
            records.append(item)
            if i<200:
                for variant,rr in missing_queries(q,'1H'):stress.append(dict(variant='1H_'+variant,**context.rank(.5*(hs(rr)+carbon_score),r)))
                for variant,rr in missing_queries(c,'13C'):stress.append(dict(variant='13C_'+variant,**context.rank(.5*(h+log_score(ck[cond_key(c)],rr,'13C')),r)))
        print(json.dumps(dict(joint_ranked=len(records))),flush=True)
    records.sort(key=lambda r:r['query_order'])
    if len(records)!=792:raise ValueError('joint_query_cohort_changed')
    report=dict(queries=792,candidates=len(context.smiles),metrics={n:summarize([r[n] for r in records]) for n in ['1H','13C','joint']},missing_stress={v:summarize([r for r in stress if r['variant']==v]) for v in sorted({r['variant'] for r in stress})},C_per_query_control_equal=True,C_checkpoint_sha256=carbon['checkpoint_sha256'],H_checkpoint_sha256=bank['checkpoint_sha256'])
    write_rows(folder/'predictions.jsonl',records);write_rows(folder/'missing_predictions.jsonl',stress);(folder/'summary.json').write_text(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--reference-evaluation',required=True);p.add_argument('--out',required=True);a=p.parse_args();base=Path(a.base);reference=Path(a.reference_evaluation);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        status('waiting_reference');path=reference/'EXIT_CODE';deadline=time.monotonic()+10800
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('reference_timeout')
            time.sleep(10)
        if path.read_text().strip()!='0':raise RuntimeError('reference_failed')
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading');bank,val,baseline,oldmissing,selected=load_reference(base,reference);metadata=metadata_view(torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False),bank.get('condition_key_mode'));labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False);normal_prior=json.loads((base/'hydrogen_analysis/routed_evaluation/site/count_prior.json').read_text())
        status('fitting_training_reporting_process');reporting=fit_reporting_priors(labels,metadata);(out/'reporting_priors.json').write_text(json.dumps(reporting,indent=2));context=Context(bank,metadata,labels,normal_prior,reporting,bank['single_indices'],device)
        protocol=dict(reference_evaluation=str(reference),reference_variant=selected,H_checkpoint_sha256=bank['checkpoint_sha256'],validation_query_sha256=stamp_queries(bank['val']),test_query_sha256=stamp_queries(bank['test']),single_candidates=len(bank['single_indices']),joint_candidates=len(bank['joint_indices']),source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},scope='only sparse atom annotation reporting scores change; previous scoring and query identities preserved elsewhere')
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2));status('validation');frozen=validation(bank,context,val,out);status('test');test(bank,context,baseline,oldmissing,frozen,out);del context;torch.cuda.empty_cache();status('joint_loading');joint(base,bank,metadata,labels,normal_prior,reporting,frozen,out,device,status);status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

"""功能概述：固定候选库、查询、模型权重的¹H观测口径开发复评。
audit核对HMDB原始ppm/峰高/积分/条件，记录参考偏移oracle及随机删峰质量契约。
validate仅用原320条val及其原哈希删峰变体，比较有界峰簇、公共偏移边缘化、
train整峰遗漏计数先验；所有选项在test前写入frozen_selection.json。
test复用上轮1000查询和171条删峰查询，候选顺序不变；正常非HMDB与裁窗评分保持
原路线，并与原逐条结果核对。HMDB峰高不作积分，不从峰端点推测测量窗口。
候选仅用缓存预测和候选结构H数；真结构用于排名/审计，绝不进入评分或偏移估计。
main创建独占输出目录、源码快照、hash协议、进度/退出码；旧结果不覆盖。
此复评使用已查看过的开发测试集，参数仍只由train/val确定，不称新独立测试。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .data import quantiles
from .evaluate import missing_queries,summarize
from .h_diagnose import cond_key
from .h_eval import LazyFP
from .h_labels import make_key
from .h_score import attach_observation,score_features,select_score,query_config
from .h_observed import cluster_lines,fit_dropout_prior,count_nll,shifted_transport,marginal_shift,line_score
from .likelihood import density,log_score
from .joint import rank_entry
from .prepare import sha

SHIFTS=[-.2,-.1,0.,.1,.2]

def stamp_queries(rows):
    return hashlib.sha256(json.dumps([(r['source'],r['record_id'],r['smiles'],np.asarray(r['q']).tolist()) for r in rows],separators=(',',':')).encode()).hexdigest()


def uniform_peaks(pred):
    mask=torch.cat([torch.ones(len(pred),1,device=pred.device,dtype=torch.bool),torch.diff(pred,dim=1)>1e-7],1)
    cdf=mask.cumsum(1)/mask.sum(1)[:,None];u=((torch.arange(128,device=pred.device)+.5)/128).expand(len(pred),-1).contiguous()
    return pred.gather(1,torch.searchsorted(cdf.contiguous(),u).clamp_max(127))


def summarize_sources(rows):
    return dict(overall=summarize(rows),by_source={s:summarize([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})})


def audit(bank,labels,raw,out):
    hmdb=[];index={s:i for i,s in enumerate(bank['smiles'])}
    for q in bank['test']:
        if q['source']!='HMDB':continue
        r=raw[q['record_id']];x=r['value']['x'];centers,groups=cluster_lines(x,.08)
        pred=bank['predictions'][cond_key(q)][index[q['smiles']]].numpy()*10
        shifts=np.arange(-.6,.601,.025);errors=np.abs(centers[None,:,None]+shifts[:,None,None]-pred[None,None,:]).min(2).mean(1)
        hmdb.append(dict(record_id=q['record_id'],smiles=q['smiles'],lines=len(x),unique_lines=len(set(x)),clusters=groups,
            centers=centers.tolist(),predicted=np.unique(pred).tolist(),nonnumeric_units='x=ppm; y=peak_height, integration_h=null',
            conditions=r.get('experimental_conditions'),metadata={k:r.get('source_metadata',{}).get(k) for k in ['distinct-peaks','peak-counter','sweep-width-ppm','center-frequency-ppm','notes']},
            labelled_integrals=sum(a.get('integration_h') is not None for a in r.get('annotations',[])),
            nearest_error_ppm=float(errors[24]),oracle_shift_ppm=float(shifts[errors.argmin()]),oracle_nearest_error_ppm=float(errors.min()),
            oracle_is_truth_guided_diagnostic_only=True))
    dropped=[]
    for r in bank['test'][:200]:
        q=attach_observation(r,labels.get(make_key(r),{}))
        for name,rr in missing_queries(q,'1H'):
            if name!='peak_dropout':continue
            item=dict(source=r['source'],record_id=r['record_id'],smiles=r['smiles'],original_peaks=len(np.unique(q['q'])),retained_peaks=len(np.unique(rr['q'])),remaining_q=rr['q'].tolist())
            if 'observed_xs' in rr:
                xs=np.asarray(rr['observed_xs']);ws=np.asarray(rr['observed_weights']);uq,uc=np.unique(rr['q'],return_counts=True)
                item.update(original_mass=sum(q['observed_weights']),remaining_mass=float(ws.sum()),remaining_xs=xs.tolist(),remaining_weights=ws.tolist(),
                    outside_remaining_q=int((~np.isclose(xs[:,None],uq[None,:],atol=1e-4,rtol=0).any(1)).sum()))
            dropped.append(item)
    report=dict(HMDB=hmdb,dropout=dropped,contracts=dict(test=stamp_queries(bank['test']),validation=stamp_queries(bank['val']),
        candidate_smiles_sha256=hashlib.sha256('\n'.join(bank['smiles'][i] for i in bank['single_indices']).encode()).hexdigest(),candidate_count=len(bank['single_indices'])),
        raw_HMDB_sha256=sha('/data/SpecGPT/dataset/NMR_exprimental/experimental_finetune/data.jsonl'))
    (out/'observation_audit.json').write_text(json.dumps(report,indent=2));return report


def baseline_features(pred,kernel,counts,q,prior,config):
    cfg=query_config(config,q)
    if cfg['kind']=='likelihood':return log_score(kernel,q,'1H')
    return select_score(score_features(pred,kernel,counts,q,prior),cfg)


def entry(score,q,lookup,smiles,fps):
    return dict(record_id=q['record_id'],smiles=q['smiles'],source=q['source'],**rank_entry(score,lookup[q['smiles']],smiles,fps))


def write_rows(path,rows):
    with path.open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')


@torch.inference_mode()
def validate(bank,labels,raw,prior,config,out,device,status):
    ids=bank['single_indices'];smiles=[bank['smiles'][i] for i in ids];lookup={s:i for i,s in enumerate(smiles)};fps=LazyFP();counts=bank['physics'][ids,0].to(device)*40
    priors={str(c):fit_dropout_prior(labels,c) for c in [None,10.,2.]};(out/'training_dropout_priors.json').write_text(json.dumps(priors,indent=2))
    drop_configs=[dict(prior=p,beta=b,shift_sigma=s) for p in priors for b in [.25,1.,4.] for s in [0.,.1,.2]]
    line_configs=[dict(width=w,kernel=k,shift_sigma=s) for w in [0.,.04,.08,.16] for k in ['mass','distinct'] for s in [0.,.1,.2]]
    drops={'baseline':[]};drops.update({str(i):[] for i in range(len(drop_configs))})
    lines={'baseline':[]};lines.update({str(i):[] for i in range(len(line_configs))})
    for cond in sorted({cond_key(r) for r in bank['val']}):
        pred=bank['predictions'][cond][ids].to(device);kernel=density(pred,'1H',.2);distinct=density(uniform_peaks(pred),'1H',.2)
        for r in [r for r in bank['val'] if cond_key(r)==cond]:
            q=attach_observation(r,labels.get(make_key(r),{}))
            if r['source']=='HMDB':
                lines['baseline'].append(entry(log_score(kernel,q,'1H'),q,lookup,smiles,fps))
                for i,cfg in enumerate(line_configs):
                    score=line_score(kernel if cfg['kernel']=='mass' else distinct,q,raw[r['record_id']]['value']['x'],cfg['width'],SHIFTS,cfg['shift_sigma'])
                    lines[str(i)].append(entry(score,q,lookup,smiles,fps))
            for name,rr in missing_queries(q,'1H'):
                if name!='peak_dropout':continue
                old=baseline_features(pred,kernel,counts,rr,prior,config);drops['baseline'].append(entry(old,rr,lookup,smiles,fps))
                costs=shifted_transport(pred,counts,rr,SHIFTS) if r['measure']=='integral_weighted' and 'observed_xs' in rr else None
                count={}
                if costs is not None:
                    k=int(round(sum(rr['observed_weights'])))
                    count={p:count_nll(counts,k,v[rr['source']]) for p,v in priors.items()}
                for i,cfg in enumerate(drop_configs):
                    score=old if costs is None else marginal_shift(costs,SHIFTS,cfg['shift_sigma'],max(1,len(np.unique(rr['observed_xs']))))+cfg['beta']*count[cfg['prior']]
                    drops[str(i)].append(entry(score,rr,lookup,smiles,fps))
        print(json.dumps(dict(validation_condition=list(cond),drop_queries=len(drops['baseline']),HMDB_queries=len(lines['baseline']))),flush=True)
    metrics={kind:{k:summarize_sources(v) for k,v in trials.items()} for kind,trials in [('dropout',drops),('HMDB',lines)]}
    chosen={kind:max(v,key=lambda k:v[k]['overall']['MRR']) for kind,v in metrics.items()}
    frozen=dict(dropout=None if chosen['dropout']=='baseline' else drop_configs[int(chosen['dropout'])],HMDB=None if chosen['HMDB']=='baseline' else line_configs[int(chosen['HMDB'])],
        validation=metrics,choices=chosen,shift_grid_ppm=SHIFTS,selection='validation MRR with baseline included; fixed site checkpoint; training whole-peak dropout counts only; same candidate/query identities',
        development_test_reuse=True)
    for kind,rows in [('dropout',drops),('HMDB',lines)]:
        write_rows(out/(kind+'_validation_baseline.jsonl'),rows['baseline']);write_rows(out/(kind+'_validation_selected.jsonl'),rows[chosen[kind]])
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));status('validation_complete',choices=chosen);return frozen,priors


@torch.inference_mode()
def test(bank,labels,raw,prior,config,frozen,priors,root,out,device,status):
    ids=bank['single_indices'];smiles=[bank['smiles'][i] for i in ids];lookup={s:i for i,s in enumerate(smiles)};fps=LazyFP();counts=bank['physics'][ids,0].to(device)*40
    old=[json.loads(l) for l in (root/'site/test/selected_predictions.jsonl').read_text().splitlines()];oldlookup={(r['source'],r['record_id'],r['smiles']):r for r in old}
    oldmissing=[json.loads(l) for l in (root/'site/test/missing_predictions.jsonl').read_text().splitlines()];misslookup={(r['variant'],r['record_id'],r['smiles']):r for r in oldmissing}
    records=[];drops=[];audit=[]
    for cond in sorted({cond_key(r) for r in bank['test']}):
        pred=bank['predictions'][cond][ids].to(device);kernel=density(pred,'1H',.2)
        distinct=density(uniform_peaks(pred),'1H',.2) if frozen['HMDB'] and frozen['HMDB']['kernel']=='distinct' else None
        for j,r in enumerate(bank['test']):
            if cond_key(r)!=cond:continue
            q=attach_observation(r,labels.get(make_key(r),{}));previous=oldlookup[r['source'],r['record_id'],r['smiles']]
            result=dict(previous)
            if r['source']=='HMDB' and frozen['HMDB']:
                cfg=frozen['HMDB'];score=line_score(kernel if cfg['kernel']=='mass' else distinct,q,raw[r['record_id']]['value']['x'],cfg['width'],SHIFTS,cfg['shift_sigma'])
                result=entry(score,q,lookup,smiles,fps);audit.append(dict(record_id=r['record_id'],old_rank=previous['rank'],new_rank=result['rank']))
            records.append(result)
            if j>=200:continue
            for name,rr in missing_queries(q,'1H'):
                previous=misslookup[name,r['record_id'],r['smiles']];item=dict(previous)
                cfg=frozen['dropout']
                if name=='peak_dropout' and cfg and r['measure']=='integral_weighted' and 'observed_xs' in rr:
                    costs=shifted_transport(pred,counts,rr,SHIFTS)
                    if costs is not None:
                        k=int(round(sum(rr['observed_weights'])));cnt=count_nll(counts,k,priors[cfg['prior']][r['source']])
                        score=marginal_shift(costs,SHIFTS,cfg['shift_sigma'],max(1,len(np.unique(rr['observed_xs']))))+cfg['beta']*cnt
                        item=dict(variant=name,**entry(score,rr,lookup,smiles,fps))
                        oldscore=baseline_features(pred,kernel,counts,rr,prior,config)
                        check=entry(oldscore,rr,lookup,smiles,fps)
                        if check['rank']!=previous['rank']:raise ValueError('baseline_ranking_drift')
                        audit.append(dict(record_id=r['record_id'],source=r['source'],variant=name,old_rank=previous['rank'],new_rank=item['rank'],
                            original_mass=sum(q['observed_weights']),remaining_mass=sum(rr['observed_weights']),retained_peaks=len(np.unique(rr['q'])),original_peaks=len(np.unique(q['q']))))
                drops.append(item)
        print(json.dumps(dict(test_condition=list(cond),ranked=len(records),stress=len(drops))),flush=True)
    records.sort(key=lambda r:next(i for i,q in enumerate(bank['test']) if (q['source'],q['record_id'],q['smiles'])==(r['source'],r['record_id'],r['smiles'])))
    if len(records)!=1000 or len([r for r in drops if r['variant']=='peak_dropout'])!=171:raise ValueError('query_cohort_drift')
    report=dict(candidates=len(smiles),checkpoint_sha256=bank['checkpoint_sha256'],normal=dict(baseline=summarize_sources(old),optimized=summarize_sources(records)),
        stress={v:dict(baseline=summarize([r for r in oldmissing if r['variant']==v]),optimized=summarize([r for r in drops if r['variant']==v])) for v in ['peak_dropout','frequency_crop']},
        controls={v:summarize([r for r in records if any(m['record_id']==r['record_id'] and m['smiles']==r['smiles'] and m['variant']==v for m in drops)]) for v in ['peak_dropout','frequency_crop']},
        note='development reevaluation; same 1000 queries, 171 random dropout masks, 200 frequency crops, 114982 candidates, checkpoint and conditions')
    write_rows(out/'predictions.jsonl',records);write_rows(out/'missing_predictions.jsonl',drops);write_rows(out/'paired_failure_audit.jsonl',audit)
    (out/'summary.json').write_text(json.dumps(report,indent=2));status('complete');return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    try:
        torch.set_num_threads(4);device=torch.device('cuda');status('loading')
        root=base/'hydrogen_analysis/routed_evaluation';bank=torch.load(root/'site/bank.pt',map_location='cpu',weights_only=False,mmap=True)
        labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        raw={r['record_id']:r for r in map(json.loads,open('/data/SpecGPT/dataset/NMR_exprimental/experimental_finetune/data.jsonl')) if r['nucleus']=='1H'}
        prior=json.loads((root/'site/count_prior.json').read_text());config=json.loads((root/'frozen_selection.json').read_text())['config']
        status('observation_audit');audit(bank,labels,raw,out)
        status('validation');frozen,priors=validate(bank,labels,raw,prior,config,out,device,status)
        status('test');test(bank,labels,raw,prior,config,frozen,priors,root,out,device,status)
        (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

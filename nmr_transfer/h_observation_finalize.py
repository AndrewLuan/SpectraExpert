"""功能概述：验证水相交换H可见性，再在原单核/双核候选库上完成观测过程对照。
blend按可见性混合全H与非交换H密度；water_validation只比较原64条HMDB验证谱，
非水相沿用第一轮冻结峰簇配置，保留第一轮作为选择候选。single只改水相HMDB，
复用第一轮1000查询/171缺峰结果；joint对原792配对、138783候选及缺测掩码复评。
碳模型、2ppm带宽和双核等权不变；碳单核指标与原结果核对。query_score只读取
观测条件、位置/积分和已冻结配置；不利用真结构进行缺峰或参考偏移推断。
visible_lines使HMDB原始细线与裁窗/删峰掩码同步，防止压力评测重新引入未观测峰。
main将配置先冻结再执行test，保存逐查询预测、C预测缓存、hash、源码及退出码。
所有目录独占；原始数据、模型权重、候选身份、旧结果不改写。依赖第一轮
h_observation_study与h_nonexchange输出，所有测试结果明确属于开发集复评。
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from .h_observation_study import uniform_peaks,entry,write_rows,summarize_sources,baseline_features,SHIFTS,stamp_queries
from .h_observed import line_score,shifted_transport,marginal_shift,count_nll
from .h_score import attach_observation
from .h_labels import make_key
from .h_diagnose import cond_key
from .h_eval import LazyFP
from .evaluate import summarize,missing_queries
from .likelihood import density,log_score
from .emission_model import make_model
from .train import Spectra,collate
from .prepare import sha


def blend(full,fixed,visibility):
    if visibility==1:return full
    if visibility==0:return fixed
    result=dict(full);result['logp']=torch.logaddexp(full['logp']+np.log(visibility),fixed['logp']+np.log1p(-visibility));result['cdf']=visibility*full['cdf']+(1-visibility)*fixed['cdf'];return result


@torch.inference_mode()
def water_validation(bank,fixed,raw,first,out,device):
    ids=bank['single_indices'];smiles=[bank['smiles'][i] for i in ids];lookup={s:i for i,s in enumerate(smiles)};fps=LazyFP()
    original=[json.loads(l) for l in (first/'HMDB_validation_selected.jsonl').read_text().splitlines()];byid={r['record_id']:r for r in original}
    frozen=json.loads((first/'frozen_selection.json').read_text());basecfg=frozen['HMDB']
    configs=[dict(width=w,kernel=k,shift_sigma=s,visibility=v) for w in [.04,.08,.16] for k in ['mass','distinct'] for s in [0.,.1,.2] for v in [0.,.5]]
    rows={'baseline':original};rows.update({str(i):[] for i in range(len(configs))})
    for cond in sorted({cond_key(r) for r in bank['val'] if r['source']=='HMDB'}):
        water=cond[1]>0
        if water:
            p=bank['predictions'][cond][ids].to(device);f=fixed['predictions'][cond][ids].to(device)
            kernels={k:{v:blend(density(p if k=='mass' else uniform_peaks(p),'1H',.2),density(f if k=='mass' else uniform_peaks(f),'1H',.2),v) for v in [0.,.5]} for k in ['mass','distinct']}
        for r in bank['val']:
            if r['source']!='HMDB' or cond_key(r)!=cond:continue
            for i,cfg in enumerate(configs):
                result=byid[r['record_id']]
                if water:
                    score=line_score(kernels[cfg['kernel']][cfg['visibility']],r,raw[r['record_id']]['value']['x'],cfg['width'],SHIFTS,cfg['shift_sigma'])
                    result=entry(score,r,lookup,smiles,fps)
                rows[str(i)].append(result)
    metrics={k:summarize_sources(rs) for k,rs in rows.items()};best=max(metrics,key=lambda k:metrics[k]['overall']['MRR'])
    frozen['water_HMDB']=None if best=='baseline' else configs[int(best)];frozen['water_validation']=dict(best=best,configs=configs,metrics=metrics)
    frozen['contracts']=dict(test_query_sha256=stamp_queries(bank['test']),validation_query_sha256=stamp_queries(bank['val']),single_candidates=len(bank['single_indices']),joint_candidates=len(bank['joint_indices']),H_checkpoint_sha256=bank['checkpoint_sha256'])
    frozen['water_selection_scope']='same 64 HMDB validation queries; first-round frozen scorer included; visibility selected globally, never by true test structure'
    write_rows(out/'water_validation_selected.jsonl',rows[best]);(out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));return frozen


def get_h_kernels(bank,fixed,indices,cond,device):
    pred=bank['predictions'][cond][indices].to(device)
    result=dict(pred=pred,mass=density(pred,'1H',.2),distinct=density(uniform_peaks(pred),'1H',.2))
    if cond[1]>0:
        p=fixed['predictions'][cond][indices].to(device);result['fixed_mass']=density(p,'1H',.2);result['fixed_distinct']=density(uniform_peaks(p),'1H',.2)
    return result


def visible_lines(q,raw):
    xs=np.asarray(raw[q['record_id']]['value']['x']);windows=q['observation']['windows']
    if windows:
        keep=np.zeros(len(xs),bool)
        for a,b in windows:keep|=(xs>=a)&(xs<=b)
        xs=xs[keep]
    if q['observation'].get('partial_peaks'):
        xs=xs[np.isclose(xs[:,None],np.unique(q['q'])[None,:],atol=1e-4,rtol=0).any(1)]
    return xs


def query_score(q,kernels,counts,prior,config,frozen,priors,raw):
    if q['source']=='HMDB':
        cfg=frozen['water_HMDB'] if cond_key(q)[1]>0 and frozen['water_HMDB'] is not None else frozen['HMDB']
        if cfg is not None:
            kernel=kernels[cfg['kernel']]
            if 'visibility' in cfg:kernel=blend(kernel,kernels['fixed_'+cfg['kernel']],cfg['visibility'])
            return line_score(kernel,q,visible_lines(q,raw),cfg['width'],SHIFTS,cfg['shift_sigma'])
    cfg=frozen['dropout']
    if q['observation'].get('partial_peaks') and q['measure']=='integral_weighted' and cfg and 'observed_xs' in q:
        costs=shifted_transport(kernels['pred'],counts,q,SHIFTS)
        if costs is not None:
            k=int(round(sum(q['observed_weights'])))
            return marginal_shift(costs,SHIFTS,cfg['shift_sigma'],max(1,len(np.unique(q['observed_xs']))))+cfg['beta']*count_nll(counts,k,priors[cfg['prior']][q['source']])
    return baseline_features(kernels['pred'],kernels['mass'],counts,q,prior,config)


@torch.inference_mode()
def single(bank,fixed,raw,first,frozen,out,device):
    rows=[json.loads(l) for l in (first/'predictions.jsonl').read_text().splitlines()];lookup_row={r['record_id']:i for i,r in enumerate(rows)}
    ids=bank['single_indices'];smiles=[bank['smiles'][i] for i in ids];lookup={s:i for i,s in enumerate(smiles)};fps=LazyFP();changes=[]
    if frozen['water_HMDB'] is not None:
        for cond in sorted({cond_key(r) for r in bank['test'] if r['source']=='HMDB' and cond_key(r)[1]>0}):
            kernels=get_h_kernels(bank,fixed,ids,cond,device);cfg=frozen['water_HMDB'];kernel=blend(kernels[cfg['kernel']],kernels['fixed_'+cfg['kernel']],cfg['visibility'])
            for q in bank['test']:
                if q['source']!='HMDB' or cond_key(q)!=cond:continue
                score=line_score(kernel,q,raw[q['record_id']]['value']['x'],cfg['width'],SHIFTS,cfg['shift_sigma']);result=entry(score,q,lookup,smiles,fps)
                j=lookup_row[q['record_id']];changes.append(dict(record_id=q['record_id'],previous_rank=rows[j]['rank'],new_rank=result['rank']));rows[j]=result
    report=json.loads((first/'summary.json').read_text());report['normal']['optimized']=summarize_sources(rows);report['water_changes']=changes
    write_rows(out/'predictions.jsonl',rows);shutil.copy2(first/'missing_predictions.jsonl',out/'missing_predictions.jsonl');(out/'summary.json').write_text(json.dumps(report,indent=2))


@torch.inference_mode()
def joint(base,bank,fixed,raw,labels,prior,config,frozen,priors,out,device,status):
    joint_out=out/'joint';joint_out.mkdir();cache=torch.load(base/'prepared_coverage/cache.pt',map_location='cpu',weights_only=False,mmap=True)
    previous=[json.loads(l) for l in (base/'hydrogen_analysis/routed_evaluation/joint/predictions.jsonl').read_text().splitlines()]
    original=[json.loads(l) for l in (base/'atom_emission/joint_likelihood/predictions.jsonl').read_text().splitlines()]
    crows={(r['smiles'],r['source'],r['record_id']):r for r in cache['rows'] if r['domain']=='experimental' and r['nucleus']=='13C'}
    cqueries=[crows[r['smiles'],r['sources']['13C'],r['record_ids']['13C']] for r in original]
    indices=bank['joint_indices'];smiles=[bank['smiles'][i] for i in indices];lookup={s:i for i,s in enumerate(smiles)};counts=bank['physics'][indices,0].to(device)*40
    checkpoint=base/'atom_emission/13C/domain/best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict'])
    template={r['smiles']:r for r in cache['rows'] if r['domain']=='experimental'};cp={};ck={}
    for cond in sorted({cond_key(r) for r in cqueries}):
        status('joint_C_prediction',condition=cond);rows=[dict(template[s],cond=np.array(cond,np.float32),q=np.zeros(128,np.float32),domain='experimental') for s in smiles]
        loader=DataLoader(Spectra(rows,cache['graphs'],'13C'),batch_size=256,num_workers=2,collate_fn=collate)
        cp[cond]=torch.cat([model(b.to(device)).cpu() for b in loader]);ck[cond]=density(cp[cond].to(device),'13C',2.)
    torch.save(dict(smiles=smiles,predictions=cp,checkpoint_sha256=sha(checkpoint)),joint_out/'carbon_bank.pt');del model,cache,cp
    fps=LazyFP();records=[];stress=[];c_drift=[]
    for cond in sorted({cond_key(r) for r in bank['paired']}):
        hk=get_h_kernels(bank,fixed,indices,cond,device);status('joint_ranking',H_condition=cond)
        for j,(r,c) in enumerate(zip(bank['paired'],cqueries)):
            if cond_key(r)!=cond:continue
            q=attach_observation(r,labels.get(make_key(r),{}));hs=query_score(q,hk,counts,prior,config,frozen,priors,raw);cs=log_score(ck[cond_key(c)],c,'13C');idx=lookup[r['smiles']]
            item=dict(smiles=r['smiles'],H_source=r['source'],H_record_id=r['record_id'])
            for name,score in [('1H',hs),('13C',cs),('joint',.5*(hs+cs))]:item[name]=entry(score,r,lookup,smiles,fps)
            if item['13C']['rank']!=previous[j]['13C']['rank']:c_drift.append(dict(record_id=r['record_id'],old=previous[j]['13C']['rank'],new=item['13C']['rank']))
            records.append(item)
            if j<200:
                for name,rr in missing_queries(q,'1H'):
                    stress.append(dict(variant='1H_'+name,**entry(.5*(query_score(rr,hk,counts,prior,config,frozen,priors,raw)+cs),r,lookup,smiles,fps)))
                for name,rr in missing_queries(c,'13C'):
                    stress.append(dict(variant='13C_'+name,**entry(.5*(hs+log_score(ck[cond_key(c)],rr,'13C')),r,lookup,smiles,fps)))
        print(json.dumps(dict(joint_ranked=len(records))),flush=True)
    oldsummary=json.loads((base/'hydrogen_analysis/routed_evaluation/joint/summary.json').read_text());metrics={n:summarize([r[n] for r in records]) for n in ['1H','13C','joint']}
    for k in ['top1','top5','top10']:
        if abs(metrics['13C'][k]-oldsummary['metrics']['13C'][k])>1.01/len(records):raise ValueError('C_control_changed')
    bysmi={r['smiles']:r['joint'] for r in records};variants=sorted({r['variant'] for r in stress})
    report=dict(queries=len(records),candidates=len(smiles),baseline=oldsummary['metrics'],optimized=metrics,C_rank_differences=c_drift,
        missing_stress={v:dict(baseline=oldsummary['missing_stress'][v],optimized=summarize([r for r in stress if r['variant']==v])) for v in variants},
        matched_controls={v:summarize([bysmi[r['smiles']] for r in stress if r['variant']==v]) for v in variants},C_checkpoint_sha256=sha(checkpoint),H_checkpoint_sha256=bank['checkpoint_sha256'])
    write_rows(joint_out/'predictions.jsonl',records);write_rows(joint_out/'missing_predictions.jsonl',stress);(joint_out/'summary.json').write_text(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    try:
        status('loading');torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda')
        h=base/'hydrogen_analysis';first=h/'observation_process';bank=torch.load(h/'evaluation_dropout/site/bank.pt',map_location='cpu',weights_only=False,mmap=True)
        fixed=torch.load(h/'water_quantile_visibility/bank.pt',map_location='cpu',weights_only=False,mmap=True)
        if fixed['smiles']!=bank['smiles'] or fixed['checkpoint_sha256']!=bank['checkpoint_sha256']:raise ValueError('candidate_or_checkpoint_changed')
        raw={r['record_id']:r for r in map(json.loads,open('/data/SpecGPT/dataset/NMR_exprimental/experimental_finetune/data.jsonl')) if r['nucleus']=='1H'}
        labels=torch.load(h/'labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        prior=json.loads((h/'routed_evaluation/site/count_prior.json').read_text());config=json.loads((h/'routed_evaluation/frozen_selection.json').read_text())['config'];priors=json.loads((first/'training_dropout_priors.json').read_text())
        status('water_validation');frozen=water_validation(bank,fixed,raw,first,out,device)
        status('single_test');single(bank,fixed,raw,first,frozen,out,device)
        joint(base,bank,fixed,raw,labels,prior,config,frozen,priors,out,device,status)
        status('complete',query_sha256=stamp_queries(bank['test']));(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

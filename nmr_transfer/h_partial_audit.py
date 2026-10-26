"""功能概述：对冻结随机缺峰失败做真结构指导的投影诊断，分离位置误差与部分谱混淆。
project_ordered通过CPU有序注入DP，把保留的实验积分核投影到真实候选预测的不同H；
投影保留总观测质量，但位置来自真实候选，因此是乐观诊断，不能部署或用于选参。
main只处理原171掩码中的整数积分谱，k超过候选H数时显式标记多报并跳过投影。
复用冻结模型、库、训练计数先验和已由val选择的评分；报告实验排名、投影排名、
正确候选的零偏移/有界参考偏移残差及匹配不更差的候选数。该数衡量当前模型下
的部分谱混淆，不等于物理不可识别性证明。结果另存，绝不替换正式评测或调参。
依赖h_observation_study/score输出及标签侧表；无训练、无外部API、无原始数据修改。
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .h_score import attach_observation,unit_emissions
from .h_labels import make_key
from .h_diagnose import cond_key
from .h_observed import shifted_transport,marginal_shift,count_nll
from .h_observation_study import SHIFTS,write_rows
from .evaluate import missing_queries
from .data import quantiles


def project_ordered(atoms,observed):
    a=np.sort(np.asarray(atoms));x=np.sort(np.asarray(observed));n=len(a);k=len(x)
    if k>n:raise ValueError('more_observed_H_than_candidate')
    dp=np.full((k+1,n+1),np.inf);dp[0]=0.;match=np.zeros((k+1,n+1),bool)
    for i in range(1,k+1):
        for j in range(1,n+1):
            value=dp[i-1,j-1]+abs(x[i-1]-a[j-1]);match[i,j]=value<=dp[i,j-1];dp[i,j]=min(value,dp[i,j-1])
    selected=[];i=k;j=n
    while i:
        if match[i,j]:selected.append(a[j-1]);i-=1
        j-=1
    return np.array(selected[::-1]),float(dp[k,n]/k)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4);device=torch.device('cuda');h=base/'hydrogen_analysis';first=h/'observation_process';status('loading')
        bank=torch.load(h/'evaluation_dropout/site/bank.pt',map_location='cpu',weights_only=False,mmap=True);labels=torch.load(h/'labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        cfg=json.loads((first/'frozen_selection.json').read_text())['dropout'];priors=json.loads((first/'training_dropout_priors.json').read_text())[cfg['prior']]
        actual={(r['record_id'],r['smiles']):r for r in map(json.loads,(first/'missing_predictions.jsonl').read_text().splitlines()) if r['variant']=='peak_dropout'}
        ids=bank['single_indices'];smiles=[bank['smiles'][i] for i in ids];lookup={s:i for i,s in enumerate(smiles)};counts=bank['physics'][ids,0].to(device)*40;records=[]
        for cond in sorted({cond_key(r) for r in bank['test'][:200]}):
            pred=bank['predictions'][cond][ids].to(device);atoms,mask=unit_emissions(pred,counts)
            for r in bank['test'][:200]:
                if cond_key(r)!=cond or r['measure']!='integral_weighted':continue
                q=attach_observation(r,labels.get(make_key(r),{}))
                for name,rr in missing_queries(q,'1H'):
                    if name!='peak_dropout' or 'observed_xs' not in rr:continue
                    ws=np.asarray(rr['observed_weights']);k=int(round(ws.sum()))
                    if not 1<=k<=96 or not np.allclose(ws,ws.round(),atol=.001):continue
                    true=lookup[r['smiles']];n=int(mask[true].sum());observed=np.repeat(rr['observed_xs'],ws.round().astype(int));row=dict(record_id=r['record_id'],smiles=r['smiles'],source=r['source'],actual_rank=actual[r['record_id'],r['smiles']]['rank'],
                        candidate_H=n,observed_H=k,original_observed_H=sum(q['observed_weights']),remaining_fraction=k/sum(q['observed_weights']))
                    if k>n:records.append(dict(row,projection='ineligible_overreport'));continue
                    projected,error=project_ordered(atoms[true,:n].cpu().numpy(),observed);xs,weights=np.unique(projected,return_counts=True)
                    oracle=dict(rr,q=quantiles(xs,weights),observed_xs=xs.tolist(),observed_weights=weights.tolist())
                    costs=shifted_transport(pred,counts,oracle,SHIFTS);effective=max(1,len(np.unique(rr['observed_xs'])))
                    score=marginal_shift(costs,SHIFTS,cfg['shift_sigma'],effective)+cfg['beta']*count_nll(counts,k,priors[r['source']])
                    originalcost=shifted_transport(pred,counts,rr,SHIFTS)
                    records.append(dict(row,projection='truth_guided_optimistic',projected_rank=int((score<=score[true]+1e-7).sum()),zero_shift_match_error_ppm=error,
                        bounded_shift_match_error_ppm=float(originalcost[:,true].min()*.2),diagnostic_best_shift_ppm=SHIFTS[int(originalcost[:,true].argmin())],
                        candidates_with_transport_no_worse_than_true=int((originalcost.min(0).values<=originalcost[:,true].min()+1e-7).sum())))
            status('auditing',queries=len(records));print(json.dumps(dict(audited=len(records))),flush=True)
        valid=[r for r in records if 'projected_rank' in r]
        def metrics(rs):
            return dict(queries=len(rs),actual_top10=sum(r['actual_rank']<=10 for r in rs)/max(1,len(rs)),projected_top10=sum(r['projected_rank']<=10 for r in rs)/max(1,len(rs)),
                median_zero_shift_error_ppm=float(np.median([r['zero_shift_match_error_ppm'] for r in rs])),median_bounded_shift_error_ppm=float(np.median([r['bounded_shift_match_error_ppm'] for r in rs])),
                median_transport_confusable_candidates=float(np.median([r['candidates_with_transport_no_worse_than_true'] for r in rs])))
        report=dict(scope='truth-guided optimistic projection, not deployment or heldout accuracy; no parameters selected here',records=len(records),overreported=len(records)-len(valid),all=metrics(valid),
            retained_mass_strata={str(lo)+'-'+str(hi):metrics([r for r in valid if lo<=r['remaining_fraction']<hi]) for lo,hi in [(0.,.5),(.5,.75),(.75,1.01)]})
        write_rows(out/'records.jsonl',records);(out/'summary.json').write_text(json.dumps(report,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

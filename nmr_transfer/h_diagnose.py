"""功能概述：固定¹H权重/候选库，量化注释缺失、整体位移偏差和谱形表达问题。
main保留旧1000条H查询、128条val及792条双核查询身份，预测候选并保存可复用bank.pt；
get_queries只按旧hash规则取样，绝不按错误大小筛选评估集。diagnose比较旧似然排名、
真结构指导的±0.5ppm参考偏移oracle、模型自身谱的W1碰撞，以及仅供诊断的完整W1排名。
oracle显式使用真结构，只用于可解释性上界，不能作为部署算法或验证选模分数。
分源/标注覆盖报告和失败案例同时保存观测/预测位移及候选计数；atom注释部分覆盖
不被解释为实测漏峰。图/核数来自候选结构，测试原子归属只用于事后审计。
依赖h_labels侧表、likelihood评分与既有cache；新目录独占，不覆盖旧训练/检索结果。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from .train import Spectra,collate
from .emission_model import make_model,weighted_quantiles
from .likelihood import density,log_score
from .evaluate import summarize
from .joint import rank_entry
from .h_labels import make_key
from .prepare import sha


def get_queries(rows,split,cap):
    return sorted([r for r in rows if r['split']==split],key=lambda r:hashlib.sha256((r['record_id']+r['smiles']+r['source']).encode()).hexdigest())[:cap]


def cond_key(r):
    return tuple([0.,*np.asarray(r['cond'])[1:].tolist()])


def compact_quantiles(q):
    x,w=np.unique(np.round(np.asarray(q)*10,4),return_counts=True)
    return list(zip(x.tolist(),w.tolist()))


@torch.inference_mode()
def diagnose(bank,labels,out,device):
    smiles=bank['smiles'];index={s:i for i,s in enumerate(smiles)}
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fps={s:fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles}
    source=[];kernels={c:density(q.to(device),'1H',.2) for c,q in bank['predictions'].items()}
    for j,r in enumerate(bank['test']):
        cond=cond_key(r);pred=bank['predictions'][cond].to(device);kernel=kernels[cond]
        allowed=bank['single_indices'].to(device);local_index=bank['single_lookup'][r['smiles']]
        score=log_score(kernel,r,'1H')[allowed]
        candidate_smiles=[smiles[i] for i in bank['single_indices'].tolist()]
        baseline=rank_entry(score,local_index,candidate_smiles,fps);true=index[r['smiles']]
        q=np.asarray(r['q']);modelq=pred[true]*10
        # 真结构指导偏移仅为oracle：不把test拟合的偏移或阈值交给后续模型。
        trials=[]
        true_kernel={k:(v[true:true+1] if torch.is_tensor(v) else v) for k,v in kernel.items()}
        for shift in np.linspace(-.5,.5,21):
            rr=dict(r,q=q+shift);trials.append((float(log_score(true_kernel,rr,'1H')[0]),float(shift)))
        best_shift=min(trials)[1];shifted=rank_entry(log_score(kernel,dict(r,q=q+best_shift),'1H')[allowed],local_index,candidate_smiles,fps)
        target=torch.tensor(q,device=device)/10
        balanced=(pred-target).abs().mean(1);w1_rank=int((balanced[allowed]<=balanced[true]+1e-7).sum())
        collision=int(((pred[allowed]-pred[true]).abs().mean(1)<=1e-7).sum())
        nearest=(modelq[:,None]-torch.tensor(q,device=device)[None,:]).abs().min(0).values
        meta=labels.get(make_key(r),{});frac=meta.get('observed_H',0)/max(1,meta.get('structural_H',1)) if 'observed_H' in meta else None
        tags=[]
        if frac is not None and frac<1:tags.append('partial_atom_annotation_evidenced; acquisition_missingness_unknown')
        if baseline['rank']>10 and shifted['rank']<=10:tags.append('global_reference_oracle_recovers_top10')
        if collision>1:tags.append('exact_predicted_spectrum_collision')
        if baseline['rank']>10 and shifted['rank']>10:tags.append('global_reference_shift_insufficient')
        top=baseline['top10'][0];topidx=index[top]
        item=dict(record_id=r['record_id'],smiles=r['smiles'],source=r['source'],condition=list(cond),
            original=baseline,reference_oracle=shifted,oracle_shift_ppm=best_shift,diagnostic_full_W1_rank=w1_rank,
            exact_model_self_query_tie_count=collision,nearest_observed_error_ppm=float(nearest.mean()),
            full_W1_ppm=float(balanced[true]*10),annotated_H_fraction=frac,
            structural_H=float(r['physics'][0]*40),top1_structural_H=bank['physics'][topidx,0].item()*40,
            observation=r['observation'],annotation_metadata={k:v for k,v in meta.items() if k not in ['groups','xs','weights']},
            observed_peaks=compact_quantiles(q/10),predicted_true=compact_quantiles(pred[true].cpu().numpy()),
            predicted_top1=compact_quantiles(pred[topidx].cpu().numpy()),tags=tags)
        source.append(item)
        if (j+1)%100==0:print(json.dumps({'diagnosed':j+1}),flush=True)
    with (out/'failure_records.jsonl').open('x') as f:
        for r in source:f.write(json.dumps(r)+'\n')
    strata=collections.defaultdict(list)
    for r in source:
        strata['source:'+r['source']].append(r)
        frac=r['annotated_H_fraction']
        strata['annotation:'+('complete' if frac==1 else 'partial' if frac is not None else 'no_atom_mask')].append(r)
        strata['exchange_H:'+str(bank['physics'][index[r['smiles']],2].item()>0)].append(r)
    def aggregate(rs):
        return dict(n=len(rs),original=summarize([r['original'] for r in rs]),
            oracle_global_shift=summarize([r['reference_oracle'] for r in rs]),
            full_W1_top10=sum(r['diagnostic_full_W1_rank']<=10 for r in rs)/len(rs),
            exact_self_query_collision_fraction=sum(r['exact_model_self_query_tie_count']>1 for r in rs)/len(rs),
            median_nearest_error_ppm=float(np.median([r['nearest_observed_error_ppm'] for r in rs])),
            median_full_W1_ppm=float(np.median([r['full_W1_ppm'] for r in rs])),
            median_oracle_offset_abs_ppm=float(np.median([abs(r['oracle_shift_ppm']) for r in rs])),
            median_top1_H_count_ratio=float(np.median([r['top1_structural_H']/max(1,r['structural_H']) for r in rs])))
    cases=[]
    for tag in ['global_reference_oracle_recovers_top10','partial_atom_annotation_evidenced; acquisition_missingness_unknown','global_reference_shift_insufficient','exact_predicted_spectrum_collision']:
        eligible=[r for r in source if tag in r['tags'] and r['original']['rank']>10]
        cases.extend(dict(case_selection=tag,**r) for r in eligible[:3])
    report=dict(overall=aggregate(source),strata={k:aggregate(v) for k,v in strata.items()},
        examples=cases,oracle_scope='true-structure diagnostic only; no test-fitted transform is a production scorer',
        annotation_scope='partial atom assignment is observed missing annotation, not proof of a missing measured peak')
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--labels',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage):
        (out/'status.json').write_text(json.dumps({'stage':stage,'time':time.time()}))
    try:
        status('loading');cache=torch.load(base/'prepared_coverage/cache.pt',map_location='cpu',weights_only=False,mmap=True)
        annotation_path=Path(a.labels)/'annotations.pt'
        while not annotation_path.exists():time.sleep(5)
        labels=torch.load(annotation_path,map_location='cpu',weights_only=False)
        exp=[r for r in cache['rows'] if r['domain']=='experimental'];single=[r for r in exp if r['nucleus']=='1H']
        test=get_queries(single,'test',1000);val=get_queries(single,'val',128)
        single_smiles=sorted({r['smiles'] for r in single});joint_smiles=sorted({r['smiles'] for r in exp if r['physics'][0]>0 and r['physics'][1]>0})
        candidate={r['smiles']:r for r in exp};smiles=sorted(set(single_smiles)|set(joint_smiles));index={s:i for i,s in enumerate(smiles)}
        paired=list(map(json.loads,(base/'atom_emission/joint_likelihood/predictions.jsonl').read_text().splitlines()))
        lookup={(r['smiles'],r['source'],r['record_id']):r for r in single}
        paired_queries=[lookup[r['smiles'],r['sources']['1H'],r['record_ids']['1H']] for r in paired]
        checkpoint=base/'atom_emission/1H/domain/best.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        device=torch.device('cuda');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
        model=make_model(payload['args']).to(device);model.load_state_dict(payload['state_dict']);model.eval()
        predictions={};fixed={}
        for cond in sorted({cond_key(r) for r in test+val+paired_queries}):
            status('predicting_'+str(cond));rows=[dict(candidate[s],q=np.zeros(128,np.float32),cond=np.array(cond,np.float32),domain='experimental') for s in smiles]
            loader=DataLoader(Spectra(rows,cache['graphs'],'1H'),batch_size=256,num_workers=2,collate_fn=collate)
            allq=[];fixedq=[]
            with torch.inference_mode():
                for b in loader:
                    b=b.to(device);d=model(b,details=True);allq.append(d['quantiles'].cpu())
                    keep=~d['exchangeable'];fixedq.append(weighted_quantiles(d['values'][keep],d['weights'][keep],d['groups'][keep],b.num_graphs).cpu())
            predictions[cond]=torch.cat(allq);fixed[cond]=torch.cat(fixedq)
        bank=dict(smiles=smiles,physics=torch.tensor(np.stack([candidate[s]['physics'] for s in smiles])),predictions=predictions,fixed_predictions=fixed,
                  test=test,val=val,paired=paired_queries,single_indices=torch.tensor([index[s] for s in single_smiles]),joint_indices=torch.tensor([index[s] for s in joint_smiles]),
                  single_lookup={s:i for i,s in enumerate(single_smiles)},joint_lookup={s:i for i,s in enumerate(joint_smiles)},checkpoint_sha256=sha(checkpoint))
        torch.save(bank,out/'bank.pt');del model,cache
        status('diagnosing');report=diagnose(bank,labels,out,device)
        status('complete');(out/'EXIT_CODE').write_text('0\n');print(json.dumps(report['overall']),flush=True)
    except Exception as exc:
        status('failed:'+repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

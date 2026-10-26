"""功能概述：将验证选定的¹H模型/缺测评分与冻结¹³C模型结合，复用原792条双核查询。
main核对全局val选模冻结文件，保持138783个候选、原H/C记录配对和固定等权组合；
¹H只使用h_score的观测位置/质量，¹³C继续用旧2ppm核密度似然。C-only指标与原结果
核对，防止候选库或配对口径漂移。缺测变体只改一个核种，保存相同查询的未扰动对照。
真实结构只用于rank/指纹相似度，不传给观测过程模型。新目录独占，异常记录退出码。
依赖h_eval输出bank/selection、h_labels侧表、旧cache及旧双核predictions.jsonl。
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .h_eval import LazyFP
from .h_labels import make_key
from .h_diagnose import cond_key
from .h_score import attach_observation,score_features,select_score,query_config
from .likelihood import density,log_score
from .emission_model import make_model
from .train import Spectra,collate
from .joint import rank_entry
from .evaluate import summarize,missing_queries
from .prepare import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();base=Path(a.base);evaluation=Path(a.evaluation);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        status('loading');frozen=json.loads((evaluation/'frozen_selection.json').read_text());variant=frozen['selected_variant'];root=evaluation/variant
        bank=torch.load(root/'bank.pt',map_location='cpu',weights_only=False,mmap=True);config=json.loads((root/'selection.json').read_text())['config']
        prior=json.loads((root/'count_prior.json').read_text());labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        cache=torch.load(base/'prepared_coverage/cache.pt',map_location='cpu',weights_only=False,mmap=True)
        original=list(map(json.loads,(base/'atom_emission/joint_likelihood/predictions.jsonl').read_text().splitlines()))
        crows={(r['smiles'],r['source'],r['record_id']):r for r in cache['rows'] if r['domain']=='experimental' and r['nucleus']=='13C'}
        cqueries=[crows[r['smiles'],r['sources']['13C'],r['record_ids']['13C']] for r in original]
        device=torch.device('cuda');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
        indices=bank['joint_indices'];smiles=[bank['smiles'][i] for i in indices.tolist()];index={s:i for i,s in enumerate(smiles)}
        hp={c:q[indices].to(device) for c,q in bank['predictions'].items()};hk={c:density(q,'1H',.2) for c,q in hp.items()}
        counts=bank['physics'][indices,0].to(device)*40
        checkpoint=base/'atom_emission/13C/domain/best.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cp=make_model(payload['args']).to(device);cp.load_state_dict(payload['state_dict']);cp.eval();ck={}
        candidate={r['smiles']:r for r in cache['rows'] if r['domain']=='experimental'}
        with torch.inference_mode():
            for cond in sorted({cond_key(r) for r in cqueries}):
                status('predicting_C',condition=cond)
                rows=[dict(candidate[s],cond=np.array(cond,np.float32),q=np.zeros(128,np.float32),domain='experimental') for s in smiles]
                loader=DataLoader(Spectra(rows,cache['graphs'],'13C'),batch_size=256,num_workers=2,collate_fn=collate)
                cq=torch.cat([cp(b.to(device)).cpu() for b in loader]).to(device);ck[cond]=density(cq,'13C',2.)
        del cache,cp;fps=LazyFP();records=[];stress=[];status('ranking')
        for j,(r,c) in enumerate(zip(bank['paired'],cqueries)):
            h=attach_observation(r,labels.get(make_key(r),{}));cond=cond_key(r)
            def hscore(query):
                selected=query_config(config,query)
                if selected['kind']=='likelihood':return log_score(hk[cond],query,'1H')
                return select_score(score_features(hp[cond],hk[cond],counts,query,prior),selected)
            hs=hscore(h);cs=log_score(ck[cond_key(c)],c,'13C');true=index[r['smiles']]
            item=dict(smiles=r['smiles'],H_source=r['source'],H_record_id=r['record_id'])
            for name,score in [('1H',hs),('13C',cs),('joint',.5*(hs+cs))]:item[name]=rank_entry(score,true,smiles,fps)
            records.append(item)
            if j<200:
                for name,query in missing_queries(h,'1H'):
                    stress.append(dict(smiles=r['smiles'],variant='1H_'+name,**rank_entry(.5*(hscore(query)+cs),true,smiles,fps)))
                for name,query in missing_queries(c,'13C'):
                    stress.append(dict(smiles=r['smiles'],variant='13C_'+name,**rank_entry(.5*(hs+log_score(ck[cond_key(c)],query,'13C')),true,smiles,fps)))
            if (j+1)%100==0:print(json.dumps(dict(ranked=j+1,total=len(bank['paired']))),flush=True)
        variants=sorted({r['variant'] for r in stress});lookup={r['smiles']:r['joint'] for r in records}
        report=dict(selected_variant=variant,H_config=config,queries=len(records),candidates=len(smiles),C_checkpoint_sha256=sha(checkpoint),
            metrics={n:summarize([r[n] for r in records]) for n in ['1H','13C','joint']},
            missing_stress={v:summarize([r for r in stress if r['variant']==v]) for v in variants},
            missing_same_query_controls={v:summarize([lookup[r['smiles']] for r in stress if r['variant']==v]) for v in variants},
            selection='all H architecture/score selection frozen by validation before test; C and joint weights unchanged')
        old=json.loads((base/'atom_emission/joint_likelihood/summary.json').read_text())
        for k in ['top1','top5','top10']:
            if abs(report['metrics']['13C'][k]-old['metrics']['13C'][k])>1.01/len(records):raise ValueError('C_control_or_cohort_changed')
        for name,rows in [('predictions.jsonl',records),('missing_predictions.jsonl',stress)]:
            with (out/name).open('x') as f:
                for r in rows:f.write(json.dumps(r)+'\n')
        (out/'summary.json').write_text(json.dumps(report,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

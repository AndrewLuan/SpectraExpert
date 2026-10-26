"""功能概述：对验证选定的条件模型执行原792双核查询与原缺峰掩码复评。
main只在全局选择冻结后读取所选¹H候选bank；条件取自H查询，全部候选共享。
¹³C直接复用上轮carbon_bank，维持同模型、同预测、同138783候选及2ppm密度，
双核组合仍为等权；逐查询C排名按来源/记录ID/结构索引核对，不能假定上轮
按条件分组的输出顺序等于bank查询顺序。保留200/147/164条缺测变体及
相同查询的正常对照。baseline入选时复制上轮结果，明确表示没有新模型替代。
输出逐查询、summary和退出码；无训练或调参，不重写旧数据/结果。
--variant允许对全局冻结validation表中已登记的非优胜臂作同库消融；默认仍评所选。
复用prepare_carbon_queries已核验的固定C查询缓存，逐查询核对C排名及Top-10。
summary保存完整候选次序、H/C查询与H物理输入hash，供所有臂的最终冻结库审计。
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from .h_condinfer import condition_key,query_kernels,scoring_inputs
from .h_labels import make_key
from .h_score import attach_observation
from .h_observation_finalize import query_score
from .h_observation_study import entry,write_rows,stamp_queries
from .h_eval import LazyFP
from .evaluate import missing_queries,summarize
from .h_diagnose import cond_key
from .likelihood import density,log_score
from .h_latent_workflow import prepare_carbon_queries


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True);p.add_argument('--variant');a=p.parse_args();base=Path(a.base);evaluation=Path(a.evaluation);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        choice=json.loads((evaluation/'frozen_selection.json').read_text());mode=choice['selected_variant'];oldroot=base/'hydrogen_analysis/observation_comparison/joint'
        if a.variant:
            if a.variant not in choice['validation']:raise ValueError('ablation_variant_not_frozen')
            mode=a.variant
        if mode=='baseline':
            for name in ['summary.json','predictions.jsonl','missing_predictions.jsonl']:shutil.copy2(oldroot/name,out/name)
            (out/'selection_note.json').write_text(json.dumps(dict(selected='baseline',new_model_promoted=False)));status('complete');(out/'EXIT_CODE').write_text('0\n');return
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading')
        bank=torch.load(evaluation/mode/'bank.pt',map_location='cpu',weights_only=False,mmap=True);key_mode=bank.get('condition_key_mode',bank['condition_mode']);metadata=torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False);labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        carbon=torch.load(oldroot/'carbon_bank.pt',map_location='cpu',weights_only=False,mmap=True);indices=bank['joint_indices'];smiles=[bank['smiles'][i] for i in indices]
        if carbon['smiles']!=smiles:raise ValueError('candidate_C_bank_changed')
        old=[json.loads(l) for l in (oldroot/'predictions.jsonl').read_text().splitlines()];old_by_id={(r['H_source'],r['H_record_id'],r['smiles']):r for r in old}
        cqueries=prepare_carbon_queries(base,bank,out)
        ck={key:density(q.to(device),'13C',2.) for key,q in carbon['predictions'].items()};frozen,prior,config,priors,raw=scoring_inputs(base);lookup={s:i for i,s in enumerate(smiles)};counts=bank['physics'][indices,0].to(device)*40;fps=LazyFP();records=[];missing=[]
        for key in sorted({condition_key(r,metadata,key_mode) for r in bank['paired']}):
            group=[(i,r,cqueries[i]) for i,r in enumerate(bank['paired']) if condition_key(r,metadata,key_mode)==key];kernels=query_kernels(bank,indices,key,device,any(r['source']=='HMDB' for _,r,_ in group));status('ranking',ranked=len(records))
            for i,r,c in group:
                q=attach_observation(r,labels.get(make_key(r),{}));hs=query_score(q,kernels,counts,prior,config,frozen,priors,raw);cs=log_score(ck[cond_key(c)],c,'13C')
                item=dict(query_order=i,smiles=r['smiles'],H_source=r['source'],H_record_id=r['record_id'])
                for name,scores in [('1H',hs),('13C',cs),('joint',.5*(hs+cs))]:item[name]=entry(scores,r,lookup,smiles,fps)
                if item['13C']['rank']!=old_by_id[r['source'],r['record_id'],r['smiles']]['13C']['rank']:raise ValueError('C_query_rank_changed')
                if item['13C']['top10']!=old_by_id[r['source'],r['record_id'],r['smiles']]['13C']['top10']:raise ValueError('C_query_top10_changed')
                records.append(item)
                if i<200:
                    for name,rr in missing_queries(q,'1H'):
                        ss=.5*(query_score(rr,kernels,counts,prior,config,frozen,priors,raw)+cs);missing.append(dict(variant='1H_'+name,**entry(ss,r,lookup,smiles,fps)))
                    for name,rr in missing_queries(c,'13C'):
                        ss=.5*(hs+log_score(ck[cond_key(c)],rr,'13C'));missing.append(dict(variant='13C_'+name,**entry(ss,r,lookup,smiles,fps)))
            print(json.dumps(dict(joint_ranked=len(records),total=len(bank['paired']))),flush=True)
        records.sort(key=lambda r:r['query_order'])
        if len(records)!=792:raise ValueError('paired_cohort_changed')
        metrics={n:summarize([r[n] for r in records]) for n in ['1H','13C','joint']};old_summary=json.loads((oldroot/'summary.json').read_text());variants=sorted({r['variant'] for r in missing});by_smi={r['smiles']:r['joint'] for r in records}
        summary=dict(selected_variant=mode,queries=len(records),candidates=len(smiles),baseline=old_summary['optimized'],optimized=metrics,C_per_query_rank_equal=True,C_per_query_top10_equal=True,C_checkpoint_sha256=carbon['checkpoint_sha256'],H_checkpoint_sha256=bank['checkpoint_sha256'],missing_stress={v:summarize([r for r in missing if r['variant']==v]) for v in variants},same_query_controls={v:summarize([by_smi[r['smiles']] for r in missing if r['variant']==v]) for v in variants})
        summary.update(candidate_order_sha256=hashlib.sha256('\n'.join(smiles).encode()).hexdigest(),physics_sha256=hashlib.sha256(bank['physics'][indices].contiguous().cpu().numpy().tobytes()).hexdigest(),H_query_sha256=stamp_queries(bank['paired']),C_query_sha256=stamp_queries(cqueries))
        write_rows(out/'predictions.jsonl',records);write_rows(out/'missing_predictions.jsonl',missing);(out/'summary.json').write_text(json.dumps(summary,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

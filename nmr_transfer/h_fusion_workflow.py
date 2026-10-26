"""功能概述：用独立固定320双核val区分局部误差宽度与H/C整体融合尺度。
两种H评分分别是原0.2ppm和已由单核val选出的局部宽度；均值、潜在条件和报告
过程完全相同。run_split比较预声明的H:C比例0/.05/.1/.25/.5/1/2/4，各自只用
双核val选比例；固定等权与校准比例在同一792 test和相同缺峰/裁窗上比较。
--joint-width-grid额外在同一双核val比较lambda=0/.5/1与融合权重，
单核的宽度选择保持原样；新增分支逐查询核对C控制，已有分支同时核对H/融合。
两个等权分支逐查询核对既有latent/width排名，C控制也逐查询核对。参数冻结前
不读test结果选比例；paired_bootstrap仅作完成后的同查询差值描述，不再选型。
本入口要求预先固定val所需条件都已有缓存，不过滤查询；新增条件缺失则明确失败。
恢复溶剂侧表按bank.condition_key_mode投影，旧bank使用旧字段，已知条件与训练先验保持一致。
--reporting-evaluation可组合已由单核val冻结的甲酰报告模型；只覆盖HMDB评分，
局部宽度仍只作用于稀疏原子记录。覆盖时继续逐查询核对C，H的改动明确标注，
不要求新H评分等于旧控制。routed_h_score集中执行这项来源路由，其他观测不变。
输出独占配置、逐条rank、压力变体、配对差值和退出码；无新训练、外部API或
原bank/候选/测试改写。复用Context/VariableContext，窗口谱仍按原观测路由处理。
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .h_latent_workflow import load_reference,query_observation,prepare_carbon_queries
from .h_latent_score import bank_key,eligible,score
from .h_uncertainty_workflow import context_for
from .h_condinfer import scoring_inputs
from .h_observation_finalize import query_score
from .h_observation_study import write_rows
from .h_diagnose import cond_key
from .likelihood import density,log_score
from .evaluate import summarize,missing_queries
from .prepare import sha
from .h_metadata import metadata_view

WEIGHTS=[0.,.05,.1,.25,.5,1.,2.,4.]


def combine(h,c,weight):return c if weight==0 else (weight*h+c)/(weight+1)


def routed_h_score(q,context,cfg,inputs,hmdb_override=None):
    if hmdb_override is not None and q['source']=='HMDB':return hmdb_override(q,context)
    frozen,prior,oldconfig,priors,raw=inputs
    return score(context.featurize(q),cfg) if eligible(q) else query_score(q,context.given,context.counts,prior,oldconfig,frozen,priors,raw)


def paired_bootstrap(a,b,seed=97):
    aa=np.array([r['rank'] for r in a]);bb=np.array([r['rank'] for r in b])
    if len(aa)!=len(bb):raise ValueError('bootstrap_pair_count_mismatch')
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(aa),(5000,len(aa)));result={}
    for name,diff in [('MRR',1/aa-1/bb),('top1',(aa==1).astype(float)-(bb==1)),('top10',(aa<=10).astype(float)-(bb<=10))]:
        draws=diff[indices].mean(1);result[name]=dict(difference=float(diff.mean()),paired_95_percentile_interval=np.quantile(draws,[.025,.975]).tolist())
    return result


@torch.inference_mode()
def run_split(base,bank,context,pairs,ck,cfg,weights,inputs,split,controls=None,stress=False,check_h_control=True,hmdb_override=None):
    frozen,prior,oldconfig,priors,raw=inputs;records={str(w):[] for w in weights};missing={str(w):[] for w in weights};hrows=[];crows=[]
    for key in sorted({bank_key(bank,h,context.metadata) for h,c in pairs}):
        group=[(i,h,c) for i,(h,c) in enumerate(pairs) if bank_key(bank,h,context.metadata)==key];context.set_condition(key,any(h['source']=='HMDB' for _,h,_ in group))
        for i,h,c in group:
            q=query_observation(h,context.metadata,context.labels)
            def hs(rr):return routed_h_score(rr,context,cfg,inputs,hmdb_override)
            hv=hs(q);cv=log_score(ck[cond_key(c)],c,'13C');identity=dict(query_order=i,smiles=h['smiles'],H_source=h['source'],H_record_id=h['record_id'],C_source=c['source'],C_record_id=c['record_id']);hh=context.rank(hv,h);cc=context.rank(cv,h);hrows.append(dict(identity,**hh));crows.append(dict(identity,**cc))
            for w in weights:
                result=dict(identity,**context.rank(combine(hv,cv,w),h));records[str(w)].append(result)
                if controls is not None and w==1:
                    old=controls[h['source'],h['record_id'],h['smiles']]
                    if cc['rank']!=old['13C']['rank'] or (check_h_control and (result['rank']!=old['joint']['rank'] or hh['rank']!=old['1H']['rank'])):raise ValueError(split+'_same_query_control_changed')
            if stress and i<200:
                for variant,rr in missing_queries(q,'1H'):
                    hh=hs(rr)
                    for w in weights:missing[str(w)].append(dict(variant='1H_'+variant,**context.rank(combine(hh,cv,w),h)))
                for variant,rr in missing_queries(c,'13C'):
                    cc=log_score(ck[cond_key(c)],rr,'13C')
                    for w in weights:missing[str(w)].append(dict(variant='13C_'+variant,**context.rank(combine(hv,cc,w),h)))
    for rs in list(records.values())+[hrows,crows]:rs.sort(key=lambda r:r['query_order'])
    return records,missing,hrows,crows


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--reference-latent',required=True);p.add_argument('--width-evaluation',required=True);p.add_argument('--validation',required=True);p.add_argument('--out',required=True);p.add_argument('--joint-width-grid',action='store_true');p.add_argument('--reporting-evaluation');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');base=Path(a.base);ref=Path(a.reference_latent);width_root=Path(a.width_evaluation);vp=Path(a.validation)
        for root in [ref,width_root,vp]:
            if (root/'EXIT_CODE').read_text().strip()!='0':raise RuntimeError('dependency_incomplete:'+str(root))
        evaluation=Path(json.loads((ref/'protocol.json').read_text())['reference_evaluation']);bank,_,_,_,_=load_reference(base,evaluation);wf=json.loads((width_root/'frozen_selection.json').read_text());widths=json.loads((width_root/'width_fit.json').read_text())['selected_ppm'];vp_protocol=json.loads((vp/'protocol.json').read_text())
        if wf['checkpoint_sha256']!=bank['checkpoint_sha256'] or vp_protocol['H_checkpoint_sha256']!=bank['checkpoint_sha256']:raise ValueError('H_checkpoint_mismatch')
        if vp_protocol['H_condition_addons'] or vp_protocol['C_condition_addons']:raise ValueError('additional_validation_condition_prediction_required')
        metadata=metadata_view(torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False),bank.get('condition_key_mode'));labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False);reporting=json.loads((ref/'reporting_priors.json').read_text());normal=json.loads((base/'hydrogen_analysis/routed_evaluation/site/count_prior.json').read_text());cfg=json.loads((ref/'frozen_selection.json').read_text())['selected_config'];inputs=scoring_inputs(base);carbon=torch.load(base/'hydrogen_analysis/observation_comparison/joint/carbon_bank.pt',map_location='cpu',weights_only=False,mmap=True)
        if [bank['smiles'][i] for i in bank['joint_indices']]!=carbon['smiles']:raise ValueError('C_candidate_order_changed')
        ck={key:density(q.to(device),'13C',2.) for key,q in carbon['predictions'].items()};vpairs=[(rs['1H'],rs['13C']) for s,rs in torch.load(vp/'pairs.pt',map_location='cpu',weights_only=False)];modes={'fixed':0.,'hetero':wf['selected_lambda']};trials={};selections={};validation_rows={};tags=bank['candidate_quantile_classes']
        hmdb_override=None;reporting_config=None
        if a.reporting_evaluation:
            from .h_formyl_workflow import FormylContext,reporting_score
            rr=Path(a.reporting_evaluation)
            if (rr/'EXIT_CODE').read_text().strip()!='0':raise ValueError('reporting_evaluation_incomplete')
            reporting_config=json.loads((rr/'frozen_selection.json').read_text())
            if reporting_config['H_checkpoint_sha256']!=bank['checkpoint_sha256']:raise ValueError('reporting_checkpoint_mismatch')
            reporting_fit=json.loads((rr/'training_reporting_priors.json').read_text());reporter=FormylContext(bank,metadata,labels,normal,reporting,bank['joint_indices'],device,inputs=inputs)
            def hmdb_override(q,context):
                if reporter.current!=context.current:reporter.set_condition(context.current,True)
                return reporting_score(reporter.reporting_features(q,reporting_fit),reporting_config['selected_config'])
        if a.joint_width_grid:
            for lam in [0.,.5,1.]:
                if lam not in modes.values():modes['lambda'+str(lam)]=lam
        for mode,lam in modes.items():
            status('validation',mode=mode);context=context_for(lam,bank,metadata,labels,normal,reporting,bank['joint_indices'],device,tags,widths);rs,_,hh,cc=run_split(base,bank,context,vpairs,ck,cfg,WEIGHTS,inputs,'val',hmdb_override=hmdb_override);trials[mode]={w:summarize(rows) for w,rows in rs.items()};chosen=max(trials[mode],key=lambda w:trials[mode][w]['MRR']);selections[mode]=float(chosen);validation_rows[mode]=rs;write_rows(out/(mode+'_validation_selected.jsonl'),rs[chosen]);del context;torch.cuda.empty_cache()
        selected_joint_mode=max(trials,key=lambda mode:trials[mode][str(selections[mode])]['MRR'])
        frozen=dict(selected_joint_mode=selected_joint_mode,H_C_weight_grid=WEIGHTS,selected_H_C_weights=selections,H_width_lambdas=modes,validation=trials,validation_query_sha256=vp_protocol['query_sha256'],candidate_order_sha256=vp_protocol['candidate_order_sha256'],H_checkpoint_sha256=bank['checkpoint_sha256'],C_checkpoint_sha256=carbon['checkpoint_sha256'],reference_latent=str(ref),width_evaluation=str(width_root),selection='width and weight selected only on pre-fixed320 paired validation; original single-H width selection preserved; equal-weight controls retained; no test selection' if a.joint_width_grid else 'weights selected only on pre-fixed320 validation pairs; equal-weight controls retained; no test selection',development_test_reuse=True,source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')});(out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));np.savez_compressed(out/'validation_ranks.npz',ranks=np.array([[[r['rank'] for r in validation_rows[mode][str(w)]] for w in WEIGHTS] for mode in modes],np.int32))
        if reporting_config is not None:
            frozen['reporting_evaluation']=a.reporting_evaluation;frozen['reporting_selected_config']=reporting_config['selected_config'];(out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2))
        cqueries=prepare_carbon_queries(base,bank,out);pairs=list(zip(bank['paired'],cqueries));summary={};allrecords={}
        for mode,lam in modes.items():
            status('test',mode=mode);context=context_for(lam,bank,metadata,labels,normal,reporting,bank['joint_indices'],device,tags,widths);control_root=width_root if mode=='hetero' else ref;control={(r['H_source'],r['H_record_id'],r['smiles']):r for r in map(json.loads,(control_root/'joint/predictions.jsonl').read_text().splitlines())};weights=sorted({0.,1.,selections[mode]});check_h=mode in ['fixed','hetero'] and (reporting_config is None or reporting_config['selected_config']['kind']=='baseline');rs,missing,hh,cc=run_split(base,bank,context,pairs,ck,cfg,weights,inputs,'test',control,True,check_h_control=check_h,hmdb_override=hmdb_override)
            if len(hh)!=792 or any(len(rows)!=792 for rows in rs.values()):raise ValueError('test_cohort_changed')
            for w,rows in rs.items():write_rows(out/(mode+'_weight'+w+'_predictions.jsonl'),rows);write_rows(out/(mode+'_weight'+w+'_missing.jsonl'),missing[w])
            summary[mode]=dict(H_only=summarize(hh),C_only=summarize(cc),equal_weight=summarize(rs['1.0']),calibrated=summarize(rs[str(selections[mode])]),by_weight={w:summarize(rows) for w,rows in rs.items()},missing_stress={w:{v:summarize([r for r in rows if r['variant']==v]) for v in sorted({r['variant'] for r in rows})} for w,rows in missing.items()},same_query_equal_weight_controls=check_h,C_same_query_control_verified=True);allrecords[mode]=rs;del context;torch.cuda.empty_cache()
        comparisons=dict(hetero_vs_fixed_equal=paired_bootstrap(allrecords['hetero']['1.0'],allrecords['fixed']['1.0']),hetero_vs_fixed_calibrated=paired_bootstrap(allrecords['hetero'][str(selections['hetero'])],allrecords['fixed'][str(selections['fixed'])]),fixed_calibrated_vs_equal=paired_bootstrap(allrecords['fixed'][str(selections['fixed'])],allrecords['fixed']['1.0']));comparisons['joint_validation_selected_vs_fixed_calibrated']=paired_bootstrap(allrecords[selected_joint_mode][str(selections[selected_joint_mode])],allrecords['fixed'][str(selections['fixed'])]);(out/'summary.json').write_text(json.dumps(dict(selected_joint_mode=selected_joint_mode,queries=792,candidates=len(carbon['smiles']),metrics=summary,paired_differences=comparisons),indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

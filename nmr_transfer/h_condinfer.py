"""功能概述：冻结评分配置下，预测条件残差消融的同库候选并评估固定查询。
condition_key只编码查询条件，按消融屏蔽未启用特征；candidate_batches一次构建GPU
结构批次并在不同条件间复用。批次缓存以单进程构建，避免保留多进程共享
存储的文件描述符；绝不读取候选自己的实验条件或谱标签。
predict在原候选并集顺序导出全H/非交换H谱，使用同一模型成分避免两次前向漂移。
rank_queries逐条件构建核密度，复用上轮冻结观测评分与缺峰掩码，保存逐查询结果。
diagnostics报告正确母节点残差与DMF三条件的预测，仅作诊断，不参与模型选择。
predict遇到更新过GINE的权重时先重算每个候选batch的结构表示，再在各条件复用，
不能沿用prepared中旧骨干的缓存；退出前恢复模型的缓存策略。
局部苯残差可在逐张量核对冻结基座和候选次序后复用其他条件的同次发射bank，
仅重算苯条件；不对有基座变化的模型复用预测。预测时同时保存同次
发射的分位点环境标签，使误差宽度消融不依赖GPU重新前向的微小数值差异。
无条件上下文模型保存final_context_embedding；溶剂消息模型只保存冻结骨干的
unconditioned_context_embedding，每个条件重新传播，不能共享带条件最终表示。
恢复元数据仅由checkpoint的condition_key_mode启用；旧bank继续使用原字段。
main的val阶段只读训练完成权重、固定320条val；test必须已有全局冻结选择。
输出独占，记录权重与查询hash；不修改旧模型、原始谱、库或测试查询。
每次排名另记录完整候选次序与原物理输入字节hash，供跨消融冻结库核验。
"""
import argparse
import collections
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .h_conditions import ConditionsSpectra,condition_from_old,SOLVENT_DIM,CONDITION_DIM,LOCAL_NAMES,SOLVENTS
from .h_condmodel import MODES
from .h_labels import make_key
from .h_score import attach_observation
from .h_observation_finalize import query_score
from .h_observation_study import uniform_peaks,entry,write_rows,summarize_sources,stamp_queries
from .h_eval import LazyFP
from .evaluate import missing_queries,summarize
from .emission_model import make_model,weighted_quantiles
from .train import collate
from .likelihood import density
from .h_uncertainty import quantile_classes
from .prepare import sha
from .h_metadata import metadata_view, RECOVERED_MODE


def condition_key(row,metadata,mode):
    meta=metadata.get(make_key(row))
    if meta is None:meta=condition_from_old(row['cond'])
    old=np.asarray(row['cond'],np.float32).copy();old[0]=0;extra=np.asarray(meta.get('recovered_vector',meta['vector']) if mode==RECOVERED_MODE else meta['vector'],np.float32).copy()
    if mode=='control':extra[:]=0
    elif mode=='solvent':extra[SOLVENT_DIM:]=0
    return tuple(np.concatenate([old,extra]).tolist())


def candidate_batches(bank,graphs,device):
    rows=[dict(smiles=s,physics=p.numpy(),cond=np.zeros(4,np.float32),condition_vector=np.zeros(CONDITION_DIM,np.float32),q=np.zeros(128,np.float32),source='structure_only',domain='experimental',record_id='',observation=dict(kind='unknown',windows=[])) for s,p in zip(bank['smiles'],bank['physics'])]
    loader=DataLoader(ConditionsSpectra(rows,graphs,'1H',metadata={}),batch_size=1024,num_workers=0,collate_fn=collate,pin_memory=False)
    return [b.to(device) for b in loader]


@torch.inference_mode()
def predict(model,bank,graphs,metadata,mode,device,status,reuse=None):
    if reuse is not None and not getattr(model,'invariant_outside_benzene',False):raise ValueError('prediction_reuse_requires_invariance_contract')
    batches=candidate_batches(bank,graphs,device);predictions={};fixed={};environment_tags={}
    if getattr(model,'conditioned_messages',False):
        if not model.use_frozen_embedding:raise ValueError('conditional_messages_require_frozen_base_GINE')
        status('caching_unconditioned_context_embeddings',batches=len(batches))
        for b in batches:b.unconditioned_context_embedding=model.unconditioned_embedding(b).detach()
    elif getattr(model,'context_depth',0)>0:
        status('caching_final_context_embeddings',batches=len(batches))
        for b in batches:b.final_context_embedding=model.structure_embedding(b).detach()
    original_cache=model.use_frozen_embedding
    if not original_cache:
        status('refreshing_updated_GINE_embeddings',batches=len(batches))
        for b in batches:b.frozen_embedding=model.structure_embedding(b).detach()
        model.use_frozen_embedding=True
    keys=sorted({condition_key(r,metadata,mode) for r in bank['val']+bank['test']+bank['paired']})
    for j,key in enumerate(keys):
        if reuse is not None and key[4+SOLVENTS.index('benzene')]==0:
            predictions[key]=reuse['predictions'][key];environment_tags[key]=reuse['candidate_quantile_classes'][key]
            if key in reuse['fixed_predictions']:fixed[key]=reuse['fixed_predictions'][key]
            continue
        status('predicting_candidates',condition=j+1,total_conditions=len(keys));all_q=[];fixed_q=[];all_tags=[]
        for b in batches:
            b.cond=torch.tensor(key[:4],device=device).reshape(1,4).expand(b.num_graphs,-1)
            b.condition_vector=torch.tensor(key[4:],device=device).reshape(1,-1).expand(b.num_graphs,-1)
            d=model(b,details=True);all_q.append(d['quantiles'].cpu());tags,checked=quantile_classes(d,b)
            if not torch.equal(checked,d['quantiles']):raise ValueError('same_forward_environment_quantiles_changed')
            all_tags.append(tags.cpu())
            if key[1]>0:
                keep=~d['exchangeable'];fixed_q.append(weighted_quantiles(d['values'][keep],d['weights'][keep],d['groups'][keep],b.num_graphs).cpu())
        predictions[key]=torch.cat(all_q)
        environment_tags[key]=torch.cat(all_tags)
        if fixed_q:fixed[key]=torch.cat(fixed_q)
        print(json.dumps(dict(predicted_conditions=j+1,total=len(keys))),flush=True)
    model.use_frozen_embedding=original_cache;del batches;torch.cuda.empty_cache();return predictions,fixed,environment_tags


def query_kernels(bank,indices,key,device,need_hmdb=True):
    pred=bank['predictions'][key][indices].to(device);k=dict(pred=pred,mass=density(pred,'1H',.2))
    if need_hmdb:
        k['distinct']=density(uniform_peaks(pred),'1H',.2)
        if key[1]>0:
            fixed=bank['fixed_predictions'][key][indices].to(device);k['fixed_mass']=density(fixed,'1H',.2);k['fixed_distinct']=density(uniform_peaks(fixed),'1H',.2)
    return k


def scoring_inputs(base):
    h=base/'hydrogen_analysis';frozen=json.loads((h/'observation_comparison/frozen_selection.json').read_text());prior=json.loads((h/'routed_evaluation/site/count_prior.json').read_text());config=json.loads((h/'routed_evaluation/frozen_selection.json').read_text())['config'];priors=json.loads((h/'observation_process/training_dropout_priors.json').read_text())
    raw={r['record_id']:r for r in map(json.loads,open('/data/SpecGPT/dataset/NMR_exprimental/experimental_finetune/data.jsonl')) if r['nucleus']=='1H'}
    return frozen,prior,config,priors,raw


@torch.inference_mode()
def rank_queries(base,bank,queries,metadata,labels,mode,device,out,stress=False):
    key_mode=bank.get('condition_key_mode',mode);frozen,prior,config,priors,raw=scoring_inputs(base);indices=bank['single_indices'];smiles=[bank['smiles'][i] for i in indices];lookup={s:i for i,s in enumerate(smiles)};counts=bank['physics'][indices,0].to(device)*40;fps=LazyFP();records=[];missing=[]
    for key in sorted({condition_key(r,metadata,key_mode) for r in queries}):
        group=[(i,r) for i,r in enumerate(queries) if condition_key(r,metadata,key_mode)==key];kernels=query_kernels(bank,indices,key,device,any(r['source']=='HMDB' for _,r in group))
        for i,r in group:
            q=attach_observation(r,labels.get(make_key(r),{}));score=query_score(q,kernels,counts,prior,config,frozen,priors,raw);records.append(dict(query_order=i,**entry(score,q,lookup,smiles,fps)))
            if stress and i<200:
                for name,rr in missing_queries(q,'1H'):
                    ss=query_score(rr,kernels,counts,prior,config,frozen,priors,raw);missing.append(dict(variant=name,**entry(ss,rr,lookup,smiles,fps)))
        print(json.dumps(dict(ranked=len(records),total=len(queries))),flush=True)
    records.sort(key=lambda r:r['query_order']);write_rows(out/'predictions.jsonl',records)
    report=dict(mode=mode,candidates=len(smiles),metrics=summarize_sources(records),query_sha256=stamp_queries(queries),checkpoint_sha256=bank['checkpoint_sha256'],candidate_order_sha256=hashlib.sha256('\n'.join(smiles).encode()).hexdigest(),physics_sha256=hashlib.sha256(bank['physics'][indices].contiguous().cpu().numpy().tobytes()).hexdigest())
    if stress:
        if len(records)!=1000 or sum(r['variant']=='peak_dropout' for r in missing)!=171:raise ValueError('cohort_drift')
        report['missing_stress']={v:summarize([r for r in missing if r['variant']==v]) for v in ['frequency_crop','peak_dropout']};write_rows(out/'missing_predictions.jsonl',missing)
    (out/'summary.json').write_text(json.dumps(report,indent=2));return report


@torch.inference_mode()
def diagnostics(model,bank,graphs,metadata,labels,out,device):
    rows=bank['test'];dataset=ConditionsSpectra(rows,graphs,'1H',metadata,annotations=labels);loader=DataLoader(dataset,batch_size=128,num_workers=2,collate_fn=collate);offset=0;records=[];strata=collections.defaultdict(list)
    for b in loader:
        b=b.to(device);d=model(b,details=True);pred=weighted_quantiles(d['values'],d['weights'],d['owners'],len(b.x),points=6)*10
        for j,r in enumerate(rows[offset:offset+b.num_graphs]):
            sites=[]
            for parent,values in labels.get(make_key(r),{}).get('groups',{}).items():
                idx=int(b.ptr[j])+int(parent);pp=pred[idx];target=torch.tensor(values,device=device);err=float((pp[:,None]-target[None,:]).abs().min(0).values.mean());z=int(b.x[idx,:54].argmax())+1;tag='exchange_H' if z in [7,8,16] else 'C_H';strata[tag].append(err)
                if b.local_environment[idx,LOCAL_NAMES.index('amide_N_methyl')]>0:strata['amide_N_methyl'].append(err)
                sites.append(dict(parent=int(parent),observed=values,predicted=pp.cpu().tolist(),error_ppm=err,kind=tag))
            if sites:records.append(dict(record_id=r['record_id'],smiles=r['smiles'],source=r['source'],sites=sites))
        offset+=b.num_graphs
    dmf=[]
    for key,meta in metadata.items():
        if key[1]!='HMDB' or key[3]!='CN(C)C=O':continue
        r=dict(domain=key[0],source=key[1],record_id=key[2],smiles=key[3],physics=bank['physics'][bank['smiles'].index(key[3])].numpy(),cond=meta['old_cond'],q=np.zeros(128,np.float32),observation=dict(kind='unknown',windows=[]))
        b=collate([ConditionsSpectra([r],graphs,'1H',metadata)[0]]).to(device);d=model(b,details=True);values=d['values'].cpu().numpy()*10;weights=d['weights'].cpu().numpy();peaks=np.unique(np.round(values,4))
        dmf.append(dict(record_id=key[2],solvent=meta['solvent'],pH=meta['pH'],observed_centers=meta['line_centers'],predicted=[dict(ppm=float(v),H_mass=float(weights[np.isclose(values,v,atol=5e-5)].sum())) for v in peaks]))
    report=dict(parent_errors={k:dict(sites=len(v),mean_ppm=float(np.mean(v)),median_ppm=float(np.median(v))) for k,v in strata.items()},records=records,DMF=dmf,scope='test diagnostics only, not numerical selection')
    (out/'site_diagnostics.json').write_text(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--training',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',required=True);p.add_argument('--stage',choices=['val','test'],required=True);a=p.parse_args();base=Path(a.base);root=Path(a.out);root.mkdir(parents=True,exist_ok=True);out=root/a.stage;out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading');metadata=torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False);labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        if a.stage=='val':
            cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);bank=torch.load(base/'hydrogen_analysis/evaluation_dropout/site/bank.pt',map_location='cpu',weights_only=False,mmap=True);checkpoint=Path(a.training)/a.mode/'train/best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict'])
            key_mode=saved['args'].get('condition_key_mode',saved['args']['condition_mode']);metadata=metadata_view(metadata,key_mode);reuse=None
            if saved['args'].get('prediction_reuse_bank'):
                reuse=torch.load(saved['args']['prediction_reuse_bank'],map_location='cpu',weights_only=False,mmap=True);owner=torch.load(reuse['checkpoint_path'],map_location='cpu',weights_only=False)
                if not getattr(model,'invariant_outside_benzene',False) or reuse['smiles']!=bank['smiles'] or not torch.equal(reuse['physics'],bank['physics']) or any(not torch.equal(v,saved['state_dict'][k]) for k,v in owner['state_dict'].items()):raise ValueError('unsafe_prediction_reuse')
                if any(saved['args'].get(k)!=owner['args'].get(k) for k in ['nucleus','width','layers','context_depth','condition_mode','response_mode','amide_gap_scale','use_frozen_embedding']):raise ValueError('prediction_reuse_configuration_mismatch')
                if sha(reuse['checkpoint_path'])!=reuse['checkpoint_sha256'] or reuse['condition_key_mode']!=key_mode:raise ValueError('prediction_reuse_provenance_mismatch')
            pred,fixed,tags=predict(model,bank,cache['graphs'],metadata,key_mode,device,status,reuse);bank['predictions']=pred;bank['fixed_predictions']=fixed;bank['candidate_quantile_classes']=tags;bank['checkpoint_sha256']=sha(checkpoint);bank['checkpoint_path']=str(checkpoint);bank['condition_mode']=a.mode;bank['condition_key_mode']=key_mode;torch.save(bank,root/'bank.pt');del cache,model;gc.collect()
        else:
            if not (root.parent/'frozen_selection.json').exists():raise ValueError('selection_not_frozen')
            bank=torch.load(root/'bank.pt',map_location='cpu',weights_only=False,mmap=True)
        metadata=metadata_view(metadata,bank.get('condition_key_mode'));status('ranking');report=rank_queries(base,bank,bank['val'] if a.stage=='val' else bank['test'],metadata,labels,a.mode,device,out,a.stage=='test')
        if a.stage=='test':
            status('parent_diagnostics');cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);saved=torch.load(Path(a.training)/a.mode/'train/best.pt',map_location='cpu',weights_only=False);model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict']);diagnostics(model,bank,cache['graphs'],metadata,labels,out,device)
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

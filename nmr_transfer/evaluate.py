"""功能概述：在所有候选训练/验证选模结束并冻结后，一次执行实验保留集评估。
主要职责：main核对freeze/checkpoint/cache身份，报告全实验test的128分位数误差；
retrieve将实验谱在预先确定的全部实验结构候选库中排序，最多1000条记录hash固定查询。
候选只有结构及结构算法特征，候选谱全部由锁定模型预测，不从候选实验标签取谱。
按输入实验溶剂状态分组预测，ppm模型中仪器频率置零。条件来自查询谱，不借用目标结构。
指标含候选覆盖、Top1/5/10、MRR、Top1 Morgan相似度，按source/measure分层并保存逐条。
相等距离采用保守并列排名；一谱多解不会靠任意索引打破平局。报告闭库条件与明确分母。
test不参与任何选择或域拟合；freeze文件必须已经存在。输出独占，不覆盖已有结果。
"""
import argparse
import collections
import json
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from .data import SCALES,quantiles
from .model import SpecGNN
from .train import Spectra, collate
from .prepare import sha
from .observation import KINDS,observed_prediction,observation_loss


def observed_variant(query,variant):
    """同步显式观测质量，供¹H部分匹配压力测试使用；无新增字段时保留旧行为。"""
    if 'observed_xs' not in query:return variant
    xs=np.asarray(query['observed_xs']);weights=np.asarray(query['observed_weights']);state=variant['observation']
    if state['windows']:
        keep=np.zeros(len(xs),bool)
        for lo,hi in state['windows']:keep|=(xs>=lo)&(xs<=hi)
    else:keep=np.isclose(xs[:,None],np.unique(variant['q'])[None,:],atol=1e-4,rtol=0).any(1)
    return dict(variant,observed_xs=xs[keep].tolist(),observed_weights=weights[keep].tolist())


def distances_for_query(predictions,query,nucleus):
    scale=SCALES[nucleus];state=query['observation'];n=len(predictions)
    windows=torch.zeros(n,8,2,device=predictions.device);mask=torch.zeros(n,8,device=predictions.device,dtype=torch.bool)
    if state['windows']:
        w=torch.tensor(state['windows'],device=predictions.device)/scale
        windows[:,:len(w)]=w;mask[:,:len(w)]=True
        conditioned,penalty,count=observed_prediction(predictions,windows,mask)
    else:
        conditioned=predictions;penalty=torch.zeros(n,device=predictions.device);count=torch.ones(n,device=predictions.device)
    target=torch.tensor(query['q'],device=predictions.device)/scale
    if state['kind'] in ('unknown','partial_window'):
        targets=target[None,:].expand(n,-1).contiguous();values=conditioned.contiguous()
        pos=torch.searchsorted(values,targets).clamp_max(values.shape[1]-1)
        left=values.gather(1,(pos-1).clamp_min(0));right=values.gather(1,pos)
        distance=torch.minimum((targets-left).abs(),(targets-right).abs()).mean(1)
    else:
        distance=(conditioned-target).abs().mean(1)
    return distance+torch.where(count==0,penalty,torch.zeros_like(penalty))


def missing_queries(query,nucleus):
    result=[];scale=SCALES[nucleus];q=np.array(query['q'],copy=True)
    low,high=(-2.,16.) if nucleus=='1H' else (-30.,250.)
    pivot=6. if nucleus=='1H' else 100.
    state=query['observation']
    for lo,hi in [(low,pivot),(pivot,high)]:
        windows=[[lo,hi]]
        if state['windows']:
            windows=[[max(lo,a),min(hi,b)] for a,b in state['windows'] if max(lo,a)<min(hi,b)]
        keep=np.zeros(len(q),bool)
        for a,b in windows:keep|=(q>=a)&(q<=b)
        if keep.sum()>=4:
            result.append(('frequency_crop',observed_variant(query,dict(query,q=quantiles(q[keep]),
                observation={'kind':'window' if state['kind'] in ('complete_atoms','window') else 'partial_window','windows':windows}))))
            break
    rng=np.random.default_rng(int(hashlib.sha256((query['record_id']+query['smiles']).encode()).hexdigest()[:8],16))
    unique=np.unique(q);kept=unique[rng.random(len(unique))>.25]
    remaining=q[np.isin(q,kept)]
    if len(remaining)>=4 and len(kept)<len(unique):
        dropped=dict(query,q=quantiles(remaining),observation={'kind':'partial_window' if state['windows'] else 'unknown','windows':state['windows'],'partial_peaks':True})
        # 缺峰与频段裁剪不同：即使已有窗口，也必须删除被抽掉峰组的观测质量。
        if 'observed_xs' in query:
            xs=np.asarray(query['observed_xs']);weights=np.asarray(query['observed_weights'])
            keep=np.isclose(xs[:,None],np.unique(remaining)[None,:],atol=1e-4,rtol=0).any(1)
            dropped.update(observed_xs=xs[keep].tolist(),observed_weights=weights[keep].tolist())
        result.append(('peak_dropout',dropped))
    return result


def summarize(items):
    n=len(items)
    if not n:return {'queries':0}
    covered=[r for r in items if r['rank'] is not None]
    return dict(queries=n,covered=len(covered),candidate_coverage=len(covered)/n,
        **{f'top{k}':sum(r['rank'] is not None and r['rank']<=k for r in items)/n for k in [1,5,10]},
        MRR=sum(1/r['rank'] for r in covered)/n,mean_top1_Morgan_Tanimoto=float(np.mean([r['top1_tanimoto'] for r in items])))


@torch.inference_mode()
def retrieve(model, queries, candidate_rows, graphs, nucleus, device, output):
    condition_groups=collections.defaultdict(list)
    for r in queries:
        c=np.asarray(r['cond']).copy();c[0]=0
        # 与Spectra一致：只用实验溶剂状态；没有目标结构条件。
        condition_groups[tuple(float(x) for x in c)].append(r)
    identities=[r['smiles'] for r in candidate_rows];index={s:i for i,s in enumerate(identities)}
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);fp_cache={}
    results=[];stress=[];stress_ids={id(r) for r in queries[:200]}
    for cond,group in condition_groups.items():
        rows=[dict(r,cond=np.array(cond,np.float32),domain='experimental') for r in candidate_rows]
        loader=DataLoader(Spectra(rows,graphs,nucleus),batch_size=256,num_workers=2,collate_fn=collate)
        predictions=[]
        model.eval()
        for b in loader:
            predictions.append(model(b.to(device)).cpu())
        predictions=torch.cat(predictions).to(device)
        for r in group:
            variants=[('observed',r)]
            if id(r) in stress_ids:variants+=missing_queries(r,nucleus)
            for variant,query in variants:
                distances=distances_for_query(predictions,query,nucleus)
                smallest=distances.topk(min(10,len(rows)),largest=False).indices.cpu().tolist()
                true=index.get(r['smiles'])
                rank=int((distances<=distances[true]+1e-7).sum()) if true is not None else None
                best=identities[smallest[0]]
                for s in [best,r['smiles']]:
                    if s not in fp_cache:fp_cache[s]=fpgen.GetFingerprint(Chem.MolFromSmiles(s))
                entry=dict(record_id=r['record_id'],smiles=r['smiles'],source=r['source'],measure=r['measure'],
                    rank=rank,top10=[identities[i] for i in smallest],top1_tanimoto=DataStructs.TanimotoSimilarity(fp_cache[best],fp_cache[r['smiles']]),
                    condition=list(cond),best_distance_ppm=float(distances[smallest[0]])*SCALES[nucleus],observation=query['observation'],variant=variant)
                (results if variant=='observed' else stress).append(entry)
        print(json.dumps({'retrieval_conditions_done':list(cond),'queries_done':len(results),'candidates':len(rows)}),flush=True)
    with output.open('x') as f:
        for r in results:f.write(json.dumps(r)+'\n')
    strata=collections.defaultdict(list)
    for r in results:
        strata['source:'+r['source']].append(r);strata['measure:'+r['measure']].append(r)
        strata['coverage:'+r['observation']['kind']].append(r)
    with output.with_name(output.stem+'_missing_stress.jsonl').open('x') as f:
        for r in stress:f.write(json.dumps(r)+'\n')
    return dict(overall=summarize(results),strata={k:summarize(v) for k,v in strata.items()},
                candidate_count=len(candidate_rows),candidate_library='all unique isomeric canonical structures in experimental corpus, all splits; no measured candidate spectra used',
                query_selection='up to 1000 experimental test records, lowest sha256(record_id+smiles+source)',
                missing_stress={v:summarize([r for r in stress if r['variant']==v]) for v in ['frequency_crop','peak_dropout']},
                tie_policy='pessimistic rank: all candidates within normalized distance 1e-7 count ahead; top1 structure shown separately',
                candidate_structure_sha256=hashlib.sha256('\n'.join(identities).encode()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--cache',required=True);p.add_argument('--nucleus',required=True)
    a=p.parse_args();root=Path(a.root);out=root/'evaluation';out.mkdir(exist_ok=False)
    frozen=json.loads((root/'frozen_selection.json').read_text());cache=torch.load(a.cache,map_location='cpu',weights_only=False)
    if sha(a.cache)!=frozen['cache_sha256']:raise ValueError('cache_changed')
    rows=[r for r in cache['rows'] if r['nucleus']==a.nucleus and r['domain']=='experimental']
    test=[r for r in rows if r['split']=='test'];candidate={r['smiles']:r for r in rows}
    candidates=[candidate[s] for s in sorted(candidate)]
    queries=sorted(test,key=lambda r:hashlib.sha256((r['record_id']+r['smiles']+r['source']).encode()).hexdigest())[:1000]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high');summary={}
    for mode,entry in frozen['runs'].items():
        ckpt=root/mode/'best.pt'
        if sha(ckpt)!=entry['checkpoint_sha256']:raise ValueError('checkpoint_changed')
        payload=torch.load(ckpt,map_location='cpu',weights_only=False);conf=payload['args']
        model=SpecGNN(conf['width'],conf['layers'],adapter=mode=='domain').to(device);model.load_state_dict(payload['state_dict']);model.eval()
        loader=DataLoader(Spectra(test,cache['graphs'],a.nucleus),batch_size=256,num_workers=2,collate_fn=collate)
        errors=[];strata=collections.defaultdict(list)
        with torch.inference_mode(),(out/(mode+'_forward.jsonl')).open('x') as f:
            offset=0
            for b in loader:
                b=b.to(device);e=(observation_loss(model(b),b)*SCALES[a.nucleus]).cpu().tolist()
                for r,value in zip(test[offset:offset+len(e)],e):
                    errors.append(value);strata['source:'+r['source']].append(value);strata['measure:'+r['measure']].append(value)
                    strata['coverage:'+r['observation']['kind']].append(value)
                    f.write(json.dumps({'record_id':r['record_id'],'source':r['source'],'observation_error_ppm':value,'coverage':r['observation']})+'\n')
                offset+=len(e)
        retrieval=retrieve(model,queries,candidates,cache['graphs'],a.nucleus,device,out/(mode+'_retrieval.jsonl'))
        summary[mode]=dict(forward=dict(records=len(errors),mean_observation_error_ppm=float(np.mean(errors)),
             metric='conditional quantile L1 if complete/window; directed observed-to-prediction match if unknown/partial',
             strata={k:{'records':len(v),'mean_observation_error_ppm':float(np.mean(v))} for k,v in strata.items()}),retrieval=retrieval,
             selected_by_validation=mode==frozen['selected_mode'])
        (out/'summary_progress.json').write_text(json.dumps(summary,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'complete.json').write_text(json.dumps({'modes':list(summary)}))


if __name__=='__main__':main()

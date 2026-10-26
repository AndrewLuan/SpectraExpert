"""功能概述：固定训练权重后，用验证集选择核密度带宽，补充缺测感知谱似然排序。
主要职责：density把预测分位数转为归一化高斯混合密度；log_score仅使用观测峰，
已知窗口按窗口内预测质量条件化；未知覆盖对预先指定的全窗/半窗先验做边缘化，
不把首末观测点当测量边界，也没有对未观测峰逐一置零的损失。
query_scores/measure报告闭库检索，val的128条hash固定查询选择带宽，之后才评test。
main比较实验直接训练和已由主验证选定的适配模型；原始单向距离评测完整保留。
这一补充评测是在原始检索诊断之后开发，选择只使用val，输出明确标记为补充复评。
CDF用于窗口质量积分，log密度用于观测似然；候选只输入结构和查询条件，目标置零。
输出独占目录，保存选择、参数、密度协议、权重/数据hash及逐查询结果，不重新训练。
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from .model import SpecGNN
from .data import SCALES
from .train import Spectra,collate
from .prepare import sha
from .evaluate import summarize,missing_queries
from .emission_model import make_model

GRID_SIZE=2048
BOUNDS={'1H':(-10.,30.),'13C':(-100.,400.)}
BANDWIDTHS={'1H':(.05,.1,.2),'13C':(.5,1.,2.)}
HALVES={'1H':((-2.,6.),(6.,16.)),'13C':((-30.,100.),(100.,250.))}


@torch.inference_mode()
def density(predictions,nucleus,sigma):
    low,high=BOUNDS[nucleus];step=(high-low)/(GRID_SIZE-1)
    p=predictions*SCALES[nucleus];position=(p-low)/step;left=position.floor().long();fraction=position-left
    histogram=torch.zeros(len(p),GRID_SIZE,device=p.device)
    for offset,weight in [(0,1-fraction),(1,fraction)]:
        idx=left+offset;valid=(idx>=0)&(idx<GRID_SIZE)
        histogram.scatter_add_(1,idx.clamp(0,GRID_SIZE-1),weight*valid/p.shape[1])
    radius=int(math.ceil(4*sigma/step));x=torch.arange(-radius,radius+1,device=p.device)*step
    kernel=(torch.exp(-.5*(x/sigma).square())/(math.sqrt(2*math.pi)*sigma)).reshape(1,1,-1)
    pieces=[]
    for start in range(0,len(p),1024):pieces.append(F.conv1d(histogram[start:start+1024,None,:],kernel,padding=radius).squeeze(1))
    profile=torch.cat(pieces)
    return dict(logp=profile.clamp_min(1e-12).log(),cdf=torch.cat([torch.zeros(len(p),1,device=p.device),profile.cumsum(1)*step],1),
                low=low,high=high,step=step,sigma=sigma)


def interval_mass(kernel,windows):
    cdf=kernel['cdf'];low=kernel['low'];step=kernel['step']
    def at(x):
        u=(x-low)/step+.5;i=int(np.clip(math.floor(u),0,GRID_SIZE));j=min(i+1,GRID_SIZE)
        f=float(np.clip(u-i,0,1));return cdf[:,i]*(1-f)+cdf[:,j]*f
    merged=[]
    for a,b in sorted(windows):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    result=torch.zeros(len(cdf),device=cdf.device)
    for a,b in merged:result+=at(b)-at(a)
    return result.clamp_min(1e-12)


def log_score(kernel,query,nucleus):
    q=np.asarray(query['q']);device=kernel['logp'].device
    u=torch.tensor((q-kernel['low'])/kernel['step'],device=device).clamp(0,GRID_SIZE-1)
    i=u.floor().long();j=(i+1).clamp_max(GRID_SIZE-1);f=u-i
    base=(kernel['logp'][:,i]*(1-f)+kernel['logp'][:,j]*f).mean(1)
    state=query['observation']
    if state['windows']:
        mass=interval_mass(kernel,state['windows'])
        return -(base-mass.log()).masked_fill(mass<1e-8,-1e6)
    if state['kind']=='complete_atoms':return -base
    # 窗口假设在看查询之前固定；只将与全部观测位置相容的假设纳入边缘化。
    effective=max(1,len(np.unique(q)));hypotheses=[effective*base+math.log(.7)]
    for window in HALVES[nucleus]:
        if q.min()>=window[0] and q.max()<=window[1]:
            mass=interval_mass(kernel,[window]);conditional=(base-mass.log()).masked_fill(mass<1e-8,-1e6)
            hypotheses.append(effective*conditional+math.log(.15))
    return -torch.logsumexp(torch.stack(hypotheses),0)/effective


@torch.inference_mode()
def measure(queries,kernels,nucleus,model,smiles,index,fpgen,fp_cache,output=None,stress=False):
    records=[];missing=[]
    for number,r in enumerate(queries):
        c=np.asarray(r['cond']).copy();c[0]=0
        cond=tuple(float(x) for x in c) if model.adapter is not None else (0.,0.,0.,0.)
        for variant,query in [('observed',r)]+(missing_queries(r,nucleus) if stress and number<200 else []):
            score=log_score(kernels[cond],query,nucleus);true=index[r['smiles']]
            top=score.topk(min(10,len(smiles)),largest=False).indices.cpu().tolist()
            rank=int((score<=score[true]+1e-7).sum())
            best=smiles[top[0]]
            for s in [best,r['smiles']]:
                if s not in fp_cache:fp_cache[s]=fpgen.GetFingerprint(Chem.MolFromSmiles(s))
            entry=dict(record_id=r['record_id'],smiles=r['smiles'],source=r['source'],rank=rank,top10=[smiles[t] for t in top],
                       top1_tanimoto=DataStructs.TanimotoSimilarity(fp_cache[best],fp_cache[r['smiles']]),variant=variant)
            (records if variant=='observed' else missing).append(entry)
    if output is not None:
        with output.open('x') as f:
            for r in records:f.write(json.dumps(r)+'\n')
        with output.with_name(output.stem+'_missing.jsonl').open('x') as f:
            for r in missing:f.write(json.dumps(r)+'\n')
    return dict(overall=summarize(records),missing_stress={v:summarize([r for r in missing if r['variant']==v]) for v in ['frequency_crop','peak_dropout']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--nucleus',required=True);p.add_argument('--out',required=True)
    p.add_argument('--selected-only',action='store_true')
    a=p.parse_args();base=Path(a.root).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(registered_utc=datetime.now(timezone.utc).isoformat(),grid=GRID_SIZE,
        bandwidths=BANDWIDTHS[a.nucleus],full_window_prior=.7,half_window_priors=[.15,.15],halves=HALVES[a.nucleus],
        validation_queries=128,test_queries=1000,development='supplementary scoring developed after inspecting original retrieval diagnostics; no weight retraining; bandwidth uses validation only',source_sha256=sha(__file__)),indent=2))
    try:
        status('waiting_for_frozen_selection');deadline=time.monotonic()+10800;path=base/a.nucleus/'frozen_selection.json'
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('frozen_selection_unavailable')
            time.sleep(15)
        frozen=json.loads(path.read_text());cache_path=base/'prepared_coverage/cache.pt';status('loading')
        cache=torch.load(cache_path,map_location='cpu',weights_only=False)
        if sha(cache_path)!=frozen['cache_sha256']:raise ValueError('cache_changed')
        exp=[r for r in cache['rows'] if r['domain']=='experimental' and r['nucleus']==a.nucleus]
        candidates={r['smiles']:r for r in exp};smiles=sorted(candidates);index={s:i for i,s in enumerate(smiles)}
        key=lambda r:hashlib.sha256((r['record_id']+r['smiles']+r['source']).encode()).hexdigest()
        val=sorted([r for r in exp if r['split']=='val'],key=key)[:128];test=sorted([r for r in exp if r['split']=='test'],key=key)[:1000]
        conditions={tuple([0.,*r['cond'][1:].tolist()]) for r in val+test}
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
        fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);fp_cache={};report={}
        for mode in ([frozen['selected_mode']] if a.selected_only else dict.fromkeys(['scratch',frozen['selected_mode']])):
            checkpoint=base/a.nucleus/mode/'best.pt'
            if sha(checkpoint)!=frozen['runs'][mode]['checkpoint_sha256']:raise ValueError('checkpoint_changed')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['args']
            model=make_model(cfg).to(device);model.load_state_dict(saved['state_dict']);model.eval()
            conds=conditions if model.adapter is not None else {(0.,0.,0.,0.)};predictions={}
            with torch.inference_mode():
                for cond in conds:
                    status('candidate_prediction',mode=mode,condition=list(cond))
                    rows=[dict(candidates[s],q=np.zeros(128,np.float32),source='structure_only',cond=np.array(cond,np.float32),domain='experimental',observation={'kind':'unknown','windows':[]}) for s in smiles]
                    loader=DataLoader(Spectra(rows,cache['graphs'],a.nucleus),batch_size=256,num_workers=2,collate_fn=collate)
                    predictions[cond]=torch.cat([model(b.to(device)).cpu() for b in loader]).to(device)
            trials=[];best=None;best_kernels=None
            for sigma in BANDWIDTHS[a.nucleus]:
                status('validation_bandwidth_selection',mode=mode,bandwidth=sigma)
                kernels={c:density(pred,a.nucleus,sigma) for c,pred in predictions.items()}
                metrics=measure(val,kernels,a.nucleus,model,smiles,index,fpgen,fp_cache)['overall'];trials.append(dict(sigma=sigma,validation=metrics))
                if best is None or metrics['MRR']>best['validation']['MRR']:
                    best=trials[-1];best_kernels=kernels
                print(json.dumps(dict(mode=mode,bandwidth=sigma,validation=metrics)),flush=True)
            selection=dict(selected=best,trials=trials,checkpoint_sha256=sha(checkpoint),cache_sha256=frozen['cache_sha256'],test_used_for_selection=False)
            (out/(mode+'_selection.json')).write_text(json.dumps(selection,indent=2))
            status('test_rescoring',mode=mode,bandwidth=best['sigma'])
            metrics=measure(test,best_kernels,a.nucleus,model,smiles,index,fpgen,fp_cache,out/(mode+'_predictions.jsonl'),True)
            report[mode]=dict(selection=selection,metrics=metrics,candidates=len(smiles),supplementary_reevaluation=True)
            (out/'summary_progress.json').write_text(json.dumps(report,indent=2));del best_kernels,kernels,predictions,model
        (out/'summary.json').write_text(json.dumps(report,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

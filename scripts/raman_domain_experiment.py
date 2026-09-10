"""功能概述：执行理论→实验 Raman 的配对诊断、三组受控训练和一次冻结测试。
主要职责：prepare 只读测试身份以隔离分子组，并从其他配对数据拟合域扰动；
MaskedSpectrumEncoder 在归一化、注意力与池化中处理缺失覆盖；multi_positive_loss 防止重复身份互为负样本；
train_arm 以共同初始化训练 A/B/C、用独立实验验证 exact 选 checkpoint；最终评分由独立的
evaluate_raman_domain 入口在冻结选择后执行。fit_domain/augment_domain 的参数仅从实验训练组估计。
代码库关系：复用 SpectrumTransformer 参数结构、旧增强、精确闭库数据加载及标准结构指标；
独立 checkpoint schema 不冒充 spectrum_graph_v2，固定测试候选仍为 10000，验证池额外包含验证理论参考。
输入输出与副作用：所有命令只写独立实验目录，保存来源哈希、划分、参数、日志与索引；
不访问 LLM API，不用实验测试谱训练或选型，不覆盖原数据、checkpoint 或历史结果。
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.repair_closed_raman_exact import canonical, ordered, usable, record_key
from scripts.evaluate_closed_raman import load_role, digest
from scripts.run_closed_llm_direct import evaluate, load_queries, save
from spectraexpert.models.spectrum_encoder import SpectrumTransformer
from spectraexpert.utils.augmentations import create_positive_pair
from spectraexpert.eval.metrics import tanimoto_similarity

AXIS = np.arange(500., 4001.)
MODEL_CONFIG = dict(signal_length=3501, patch_size=16, stride=8, embed_dim=128,
                    depth=3, num_heads=4, projection_dim=64)


def coverage_mask(coverage):
    original = np.linspace(0,4000,3000)
    lo, hi = coverage
    return (AXIS >= original[lo]) & (AXIS <= original[hi-1])


def unit(x, mask, center=True):
    x=np.asarray(x,dtype=np.float32).copy()
    if not np.any(mask):raise ValueError('empty spectral coverage')
    if center:x-=x[mask].mean()
    x[~mask]=0
    return x/max(float(np.linalg.norm(x)),1e-8)


class MaskedSpectrumEncoder(SpectrumTransformer):
    """保持原 Transformer 大小，以有效值计算标准化并屏蔽全缺失 patch。"""
    def forward(self,x,mask):
        mask=mask.bool()
        count=mask.sum(1,keepdim=True)
        if (count==0).any():raise ValueError('empty spectral coverage')
        x=torch.where(mask,x,0.)
        mean=x.sum(1,keepdim=True)/count
        variance=torch.where(mask,(x-mean).square(),0.).sum(1,keepdim=True)/count
        x=torch.where(mask,(x-mean)/variance.sqrt().clamp_min(1e-6),0.)
        weights=mask.unfold(1,self.patch_size,self.stride).float().mean(-1)
        if (weights.sum(1)==0).any():raise ValueError('empty patch coverage')
        h=self.pos_drop(self.patch_embed(x)+self.pos_embed)
        h=self.encoder(h,src_key_padding_mask=weights==0)
        h=self.norm(h)
        pooled=(h*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True)
        return self.project(pooled)


def multi_positive_loss(left,right,identities,temperature=.1):
    logits=left.float()@right.float().T/temperature
    positive=identities[:,None]==identities[None,:]
    def direction(z,p):
        return (torch.logsumexp(z,1)-torch.logsumexp(z.masked_fill(~p,-torch.inf),1)).mean()
    return .5*(direction(logits,positive)+direction(logits.T,positive.T))


def peak_info(x,mask):
    y=np.where(mask,x,0).astype(float)
    span=max(float(np.ptp(y[mask])),1e-8)
    peaks,_=find_peaks(y,prominence=.05*span,distance=5)
    peaks=peaks[mask[peaks]]
    peaks=peaks[np.argsort(y[peaks])[-20:]]
    widths=peak_widths(y,peaks)[0] if len(peaks) else np.array([])
    return peaks,widths


def fit_domain(theory,tm,experiment,em):
    """逐训练配对搜索全局仿射峰位偏差；低相关/低增益拟合不进入增强频率范围。"""
    details=[]
    small=AXIS[::4]
    for t,mt,e,me in zip(theory,tm,experiment,em):
        common=mt&me
        before=float(unit(t,common)@unit(e,common))
        best=(before,1.,0.)
        for scale in np.linspace(.94,1.06,13):
            for shift in range(-40,41,5):
                source=(small-shift)/scale
                mask=(np.interp(source,AXIS,mt.astype(float),left=0,right=0)>.999)&me[::4]
                if mask.sum()<100:continue
                warped=np.interp(source,AXIS,t,left=0,right=0)
                corr=float(unit(warped,mask)@unit(e[::4],mask))
                if corr>best[0]:best=(corr,float(scale),float(shift))
        score,scale,shift=best
        source=(AXIS-shift)/scale
        warped=np.interp(source,AXIS,t,left=0,right=0)
        wm=(np.interp(source,AXIS,mt.astype(float),left=0,right=0)>.999)&me
        tp,tw=peak_info(warped,wm);ep,ew=peak_info(e,wm)
        reliable=score>=.25 and score-before>=.05
        broad=math.sqrt(max(0.,float(np.median(ew))**2-float(np.median(tw))**2))/2.355 if len(tw) and len(ew) else 0.
        amplitude=max(float(np.ptp(e[me])),1e-6)
        residual=(e/amplitude)-(warped/max(float(np.ptp(warped[wm])),1e-6))
        coeff=np.polyfit(np.linspace(-1,1,len(e))[wm],gaussian_filter1d(residual,80)[wm],2)[:2]
        ratio=[];peak_residual=[]
        for peak in tp:
            if not len(ep):continue
            match=ep[np.argmin(abs(ep-peak))]
            if abs(match-peak)<=15 and warped[peak]>0 and e[match]>0:
                ratio.append(float(np.log((e[match]/amplitude)/(warped[peak]/max(warped.max(),1e-6)))))
                peak_residual.append(float(match-peak))
        low=e[me]<np.quantile(e[me],.4)
        noise=float(np.median(abs(np.diff(e[me])[low[:-1]&low[1:]]))/.954/amplitude) if (low[:-1]&low[1:]).any() else 0.
        details.append(dict(correlation_before=before,correlation_aligned=score,scale=scale,shift=shift,
                            reliable=reliable,broadening_sigma=broad,baseline_coeff=coeff.tolist(),noise=noise,
                            log_intensity_ratios=ratio,local_peak_residuals=peak_residual,
                            theory_peak_count=len(tp),experiment_peak_count=len(ep)))
    reliable=[d for d in details if d['reliable']]
    interval=lambda values:np.quantile(values,[.05,.95]).tolist()
    # 安全范围限制来自固定搜索窗口；数据不足时关闭新增频率扰动并明确标记。
    scales=interval([d['scale'] for d in reliable]) if len(reliable)>=20 else [1.,1.]
    shifts=interval([d['shift'] for d in reliable]) if len(reliable)>=20 else [0.,0.]
    residuals=[v for d in reliable for v in d['local_peak_residuals']]
    ratios=[v for d in reliable for v in d['log_intensity_ratios']]
    params=dict(scale=scales,shift=shifts,local_shift_max=min(10.,float(np.quantile(abs(np.array(residuals)),.9))) if residuals else 0.,
                broadening_max=min(15.,float(np.quantile([d['broadening_sigma'] for d in details],.9))),
                intensity_log_sigma=min(.5,float(np.std(ratios))) if ratios else 0.,
                baseline_abs=np.minimum(.3,np.quantile(abs(np.array([d['baseline_coeff'] for d in details])),.9,axis=0)).tolist(),
                noise_sigma=min(.03,float(np.quantile([d['noise'] for d in details],.9))),
                reliable_fits=len(reliable),fitted_pairs=len(details),fit_uses='experimental training groups only',
                frequency_fit_enabled=len(reliable)>=20)
    params['affine_profiles']=[[d['scale'],d['shift']] for d in reliable
                              if scales[0]<=d['scale']<=scales[1] and shifts[0]<=d['shift']<=shifts[1]] if len(reliable)>=20 else [[1.,0.]]
    if not params['affine_profiles']:params['affine_profiles']=[[1.,0.]]
    return params,details


def augment_domain(x,mask,params,rng,exp_masks):
    # 缩放与平移共同拟合，按整对重采样，避免独立抽样拼出训练中没有的畸变。
    scale,shift=params['affine_profiles'][rng.integers(len(params['affine_profiles']))]
    local=np.interp(np.linspace(0,1,len(x)),np.linspace(0,1,8),rng.uniform(-params['local_shift_max'],params['local_shift_max'],8))
    source=(AXIS-shift-local)/scale
    result=np.interp(source,AXIS,x,left=0,right=0)
    valid=np.interp(source,AXIS,mask.astype(float),left=0,right=0)>.999
    valid &= exp_masks[rng.integers(len(exp_masks))]
    if valid.sum()<32:return np.where(mask,x,0).astype(np.float32),mask.copy()
    sigma=rng.uniform(0,params['broadening_max'])
    if sigma>.1:result=gaussian_filter1d(result,sigma)
    gain=np.exp(np.interp(np.linspace(0,1,len(x)),np.linspace(0,1,12),rng.normal(0,params['intensity_log_sigma'],12)))
    result*=gain
    u=np.linspace(-1,1,len(x));a,b=params['baseline_abs']
    amplitude=max(float(np.ptp(result[valid])),1e-6)
    result+=amplitude*(rng.uniform(-a,a)*u*u+rng.uniform(-b,b)*u+rng.normal(0,params['noise_sigma'],len(x)))
    result[~valid]=0
    return result.astype(np.float32),valid


def prepare(root,pilot,out):
    out.mkdir(parents=True,exist_ok=False)
    sample=json.loads((root/'sample_manifest.json').read_text())
    split=json.loads((root/'training_split.json').read_text())
    rows,matrix=load_role(root,'theory_train_candidate',sample)
    test_groups=set(sample['test_groups'])  # 不读取实验测试谱。
    if digest(pilot/'records.json')!=sample['pilot_records_sha256']:raise ValueError('pilot binding mismatch')
    records=json.loads((pilot/'records.json').read_text())
    arrays=np.load(pilot/'overlap_arrays.npz',allow_pickle=False)
    matches={};exp=[]
    for role in ('candidate','query'):
        values=arrays[role]
        for i,r in enumerate(records[role]):
            if r['group_hash'] in test_groups or not usable(values[i],r['coverage']):continue
            key=(r['group_hash'],canonical(r['smiles']))
            item=dict(r,array=values[i],canonical=key[1],cache_index=i)
            if role=='candidate':matches.setdefault(key,[]).append(item)
            else:exp.append(item)
    exp=[r for r in exp if (r['group_hash'],r['canonical']) in matches]
    groups=ordered({r['group_hash'] for r in exp},0,'domain-experimental-split')
    val_groups=set(groups[:math.ceil(len(groups)*.25)]);train_groups=set(groups)-val_groups
    if len(val_groups)<50 or len(train_groups)<100:raise ValueError('insufficient independent paired groups')
    masks=[coverage_mask(r) for r in rows['coverage']]
    names=[canonical(s) for s in rows['smiles']];pool=list(matrix);lookup={s:i for i,s in enumerate(names)}
    meta=rows.to_dict('records')
    for row in meta:row['coverage']=[int(i) for i in row['coverage']]
    exp_values=[];exp_masks=[];exp_targets=[];exp_meta=[]
    for r in exp:
        name=r['canonical']
        if name not in lookup:
            refs=matches[r['group_hash'],name]
            ref=min(refs,key=lambda x:(x['split'],x['line']))
            lookup[name]=len(pool);names.append(name)
            pool.append(np.interp(AXIS,np.linspace(0,4000,3000),ref['array']).astype(np.float32))
            masks.append(coverage_mask(ref['coverage']))
            meta.append({k:v for k,v in ref.items() if k!='array'})
        exp_values.append(np.interp(AXIS,np.linspace(0,4000,3000),r['array']).astype(np.float32))
        exp_masks.append(coverage_mask(r['coverage']));exp_targets.append(lookup[name])
        exp_meta.append({k:v for k,v in r.items() if k!='array'})
    pool=np.stack(pool);masks=np.stack(masks);exp_values=np.stack(exp_values);exp_masks=np.stack(exp_masks)
    targets=np.array(exp_targets);train=np.array([i for i,r in enumerate(exp_meta) if r['group_hash'] in train_groups]);val=np.array([i for i,r in enumerate(exp_meta) if r['group_hash'] in val_groups])
    assert not (train_groups|val_groups)&test_groups and not train_groups&val_groups
    main_train=[i for i,mid in enumerate(rows['mol_id']) if mid in set(split['train'])]
    theory_train=np.array(main_train+list(range(10000,len(pool))))
    params,details=fit_domain(pool[targets[train]],masks[targets[train]],exp_values[train],exp_masks[train])
    np.savez_compressed(out/'prepared.npz',theory=pool,theory_mask=masks,experiment=exp_values,experiment_mask=exp_masks,
                        targets=targets,train=train,validation=val,theory_train=theory_train)
    save(out/'domain_parameters.json',params);save(out/'domain_diagnostics.json',details)
    save(out/'protocol.json',dict(schema='raman_domain_spectrum_v1',seed=0,data_root=str(root.resolve()),
        sample_manifest_sha256=digest(root/'sample_manifest.json'),training_split_sha256=digest(root/'training_split.json'),
        pilot_records_sha256=digest(pilot/'records.json'),pilot_arrays_sha256=digest(pilot/'overlap_arrays.npz'),
        prepared_sha256=digest(out/'prepared.npz'),parameters_sha256=digest(out/'domain_parameters.json'),
        script_sha256=digest(Path(__file__)),candidate_names=names,candidate_metadata=meta,experimental_metadata=exp_meta,
        train_groups=sorted(train_groups),validation_groups=sorted(val_groups),excluded_test_groups=sorted(test_groups),
        theory_training_count=len(theory_train),experimental_train_count=len(train),experimental_validation_count=len(val),
        validation_candidate_count=len(pool),final_test_candidate_count=10000,model_config=MODEL_CONFIG,
        theory_reference_policy='9500 frozen train rows plus supplemental exact theory references, including validation identities; experimental validation observations withheld',
        selection='validation exact match, then Tanimoto, then lower training loss; frozen test only after arm selection',
        graph_baseline='original graph checkpoint allowed only as frozen diagnostic, not trained on revised library'))
    print(json.dumps(dict(prepared=True,train_groups=len(train_groups),validation_groups=len(val_groups),
                         train_spectra=len(train),validation_spectra=len(val),pool=len(pool),
                         domain_parameters={k:v for k,v in params.items() if k!='affine_profiles'})),flush=True)


def load_prepared(out):
    protocol=json.loads((out/'protocol.json').read_text())
    if digest(out/'prepared.npz')!=protocol['prepared_sha256'] or digest(out/'domain_parameters.json')!=protocol['parameters_sha256']:
        raise ValueError('prepared data changed')
    with np.load(out/'prepared.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    return protocol,data,json.loads((out/'domain_parameters.json').read_text())


def encode(model,x,m,device,batch=128):
    model.eval();result=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch):
            result.append(model(torch.as_tensor(x[start:start+batch],device=device),torch.as_tensor(m[start:start+batch],device=device)).float().cpu().numpy())
    return np.concatenate(result)


def rank_metrics(query,reference,targets,names):
    scores=query@reference.T
    predicted=np.argmax(scores,axis=1)
    exact=(predicted==targets)
    tanimoto=[tanimoto_similarity(names[t],names[p]) for t,p in zip(targets,predicted)]
    return dict(exact_match=float(exact.mean()),exact_count=int(exact.sum()),n=len(targets),
                tanimoto=float(np.mean(tanimoto))),predicted


def train_arm(out,arm,minutes=20,epochs=20):
    directory=out/arm;directory.mkdir(exist_ok=False)
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    protocol,data,params=load_prepared(out);device='cuda' if torch.cuda.is_available() else 'cpu'
    if device!='cuda':raise RuntimeError('formal experiment requires server GPU')
    model=MaskedSpectrumEncoder(**MODEL_CONFIG).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    rng=np.random.default_rng(0);start=time.monotonic();best=None;bad=0;history=[]
    train=data['train'];val=data['validation'];theory_train=data['theory_train']
    t,tm,e,em=data['theory'],data['theory_mask'],data['experiment'],data['experiment_mask']
    initial={k:v.cpu().clone() for k,v in model.state_dict().items()}
    torch.save(initial,directory/'initial.pt')
    save(directory/'config.json',dict(arm=arm,learning_rate=3e-4,batch_size=64,temperature=.1,
        minutes=minutes,epochs=epochs,seed=0,experimental_fraction=.5 if arm=='C' else 0,
        initial_sha256=digest(directory/'initial.pt'),prepared_sha256=protocol['prepared_sha256'],
        script_sha256=digest(Path(__file__)),model_config=MODEL_CONFIG,device=device))
    for epoch in range(epochs):
        model.train();losses=[]
        width=32 if arm=='C' else 64
        indices=rng.permutation(theory_train)
        # C 与 A/B 使用相同步数，避免半批理论导致更新次数翻倍。
        for step in range(math.ceil(len(theory_train)/64)):
            if time.monotonic()-start>=minutes*60:break
            ids=indices[step*width:(step+1)*width]
            if len(ids)<2:continue
            query=[];qm=[];reference=[];rm=[];identity=[]
            for idx in ids:
                _,aug=create_positive_pair(np.where(tm[idx],t[idx],0),AXIS)
                mask=tm[idx].copy()
                if arm in ('B','C'):aug,mask=augment_domain(aug,mask,params,rng,em[train])
                query.append(aug);qm.append(mask);reference.append(t[idx]);rm.append(tm[idx]);identity.append(idx)
            if arm=='C':
                for idx in rng.choice(train,len(ids),replace=True):
                    target=data['targets'][idx]
                    query.append(e[idx]);qm.append(em[idx]);reference.append(t[target]);rm.append(tm[target]);identity.append(target)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                left=model(torch.as_tensor(np.stack(query),device=device),torch.as_tensor(np.stack(qm),device=device))
                right=model(torch.as_tensor(np.stack(reference),device=device),torch.as_tensor(np.stack(rm),device=device))
                loss=multi_positive_loss(left,right,torch.tensor(identity,device=device))
            if not torch.isfinite(loss):raise FloatingPointError('non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        if not losses:break
        reference=encode(model,t,tm,device)
        query=encode(model,e[val],em[val],device)
        metrics,_=rank_metrics(query,reference,data['targets'][val],protocol['candidate_names'])
        key=(metrics['exact_match'],metrics['tanimoto'],-float(np.mean(losses)))
        record=dict(epoch=epoch,loss=float(np.mean(losses)),validation=metrics,elapsed_seconds=time.monotonic()-start,steps=len(losses))
        history.append(record)
        with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if best is None or key>best:
            best=key;bad=0
            torch.save(dict(schema='raman_domain_spectrum_v1',state_dict=model.state_dict(),model_config=MODEL_CONFIG,
                prepared_sha256=protocol['prepared_sha256'],sample_manifest_sha256=protocol['sample_manifest_sha256'],
                parameters_sha256=protocol['parameters_sha256'],arm=arm,epoch=epoch,validation=metrics,loss=record['loss']),directory/'best.pt')
        else:bad+=1
        if bad>=6 or time.monotonic()-start>=minutes*60:break
    if best is None:raise RuntimeError('no completed training validation')
    save(directory/'summary.json',dict(arm=arm,best_key=best,epochs=len(history),seconds=time.monotonic()-start,
        checkpoint_sha256=digest(directory/'best.pt'),initial_sha256=digest(directory/'initial.pt')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','train'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--pilot',type=Path)
    parser.add_argument('--arm',choices=['A','B','C'])
    parser.add_argument('--minutes',type=float,default=20)
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.data_root,args.pilot,args.out)
    else:train_arm(args.out,args.arm,args.minutes,args.epochs)


if __name__=='__main__':main()

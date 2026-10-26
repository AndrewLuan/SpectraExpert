"""功能概述：相同冻结骨干上训练四种等容量条件残差消融，所有选模只使用val。
sampling_weights对实验来源平衡，并对其溶剂族频次做平方根平衡；理论重放总质量
保持一半，训练身份与prepared一致。每模式使用同一seed、抽样序列、轮数和目标。
validate_sources分别计算五来源目标并取宏平均选epoch，保存全局和分源误差；
train使用已有正确母节点监督、积分分布约束与HMDB峰簇/水相可见性，无新增真值。
main核对权重/缓存身份，冻结所有旧参数，只训练零初始化条件残差头；原谱与标签
侧表只读。输出配置、每轮日志、最佳权重、完成文件和退出码，目录独占。
不调用外部API；候选检索/模型间选择由后续固定val查询完成，不在本文件读test。
"""
import argparse
import collections
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from .h_conditions import ConditionsSpectra,condition_from_old
from .h_condmodel import ConditionSpecGNN,MODES,training_details
from .h_labels import make_key
from .h_objective import objective
from .train import collate,seed_worker
from .observation import observation_loss
from .prepare import sha


def sampling_weights(rows,metadata):
    exp=[r for r in rows if r['domain']=='experimental'];theory=[r for r in rows if r['domain']=='theory'];counts=collections.Counter((r['source'],metadata[make_key(r)]['family']) for r in exp)
    denom=collections.defaultdict(float)
    for (source,family),n in counts.items():denom[source]+=np.sqrt(n)
    sources=len(denom);weights=[]
    for r in rows:
        if r['domain']=='theory':weights.append(.5/max(1,len(theory)))
        else:
            family=metadata[make_key(r)]['family'];weights.append(.5/sources/denom[r['source']]/np.sqrt(counts[r['source'],family]))
    return torch.tensor(weights,dtype=torch.float64)


@torch.inference_mode()
def validate_sources(model,rows,graphs,metadata,labels,device,batch_size):
    model.eval();report={}
    for source in sorted({r['source'] for r in rows}):
        rs=[r for r in rows if r['source']==source];loader=DataLoader(ConditionsSpectra(rs,graphs,'1H',metadata,annotations=labels),batch_size=batch_size,num_workers=2,collate_fn=collate)
        total=0.;error=0.;n=0
        for b in loader:
            b=b.to(device);d=training_details(model(b,details=True),b);loss=objective(d,b,'site')['loss'];total+=float(loss)*b.num_graphs;error+=float(observation_loss(d['quantiles'],b).sum())*10;n+=b.num_graphs
        report[source]=dict(records=n,objective=total/n,observed_error_ppm=error/n)
    return dict(by_source=report,macro_objective=float(np.mean([r['objective'] for r in report.values()])),mean_observed_error_ppm=sum(r['records']*r['observed_error_ppm'] for r in report.values())/len(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--epochs',type=int,default=25);p.add_argument('--samples',type=int,default=65536);p.add_argument('--seed',type=int,default=31);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--smoke',action='store_true');a=p.parse_args();base=Path(a.base);prepared=Path(a.prepared);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,seconds=time.time()-start,**kw)))
    try:
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading')
        cache=torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False)
        labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        train=[r for r in cache['rows'] if r['split']=='train'];val=[r for r in cache['rows'] if r['split']=='val']
        if a.smoke:train=train[:128]+[r for r in train if r['source']=='HMDB'][:32];val=val[:16]+[r for r in val if r['source']=='HMDB'][:8];a.samples=256;a.epochs=2
        checkpoint=base/'hydrogen_analysis/site_supervision/train/best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=ConditionSpecGNN('1H',saved['args']['width'],saved['args']['layers'],a.mode).to(device)
        missing,unexpected=model.load_state_dict(saved['state_dict'],strict=False)
        if unexpected or any(not k.startswith('condition_head.') for k in missing):raise ValueError('warmstart_mismatch')
        weights=sampling_weights(train,metadata);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=1e-4)
        args=dict(model='condition_atom',nucleus='1H',width=saved['args']['width'],layers=saved['args']['layers'],condition_mode=a.mode,seed=a.seed)
        manifest=json.loads((prepared/'manifest.json').read_text());contract=dict(args=vars(a),model_args=args,train=len(train),validation=len(val),baseline_sha256=sha(checkpoint),prepared_sha256=manifest['cache_sha256'],metadata_sha256=manifest['metadata_sha256'],trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),selection='equal-source macro validation objective; fixed 25 epochs; retrieval model selection separate',source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
        (out/'config.json').write_text(json.dumps(contract,indent=2));best=float('inf');best_epoch=0;dataset=ConditionsSpectra(train,cache['graphs'],'1H',metadata,annotations=labels)
        for epoch in range(1,a.epochs+1):
            generator=torch.Generator().manual_seed(a.seed+epoch);sampler=WeightedRandomSampler(weights,a.samples,replacement=True,generator=generator)
            loader=DataLoader(dataset,batch_size=a.batch_size,sampler=sampler,num_workers=2,collate_fn=collate,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(a.seed+epoch),pin_memory=True)
            model.train();loss_sum=0.;n=0;status('training',epoch=epoch)
            for j,b in enumerate(loader):
                b=b.to(device);optimizer.zero_grad(set_to_none=True);details=model(b,details=True);visible=training_details(details,b)
                if (visible['quantiles'].abs()>1e4).any():raise ValueError('empty_nuclear_support_in_training')
                parts=objective(visible,b,'site');loss=parts['loss']
                if not torch.isfinite(loss):raise ValueError('nonfinite_loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.condition_head.parameters(),2.);optimizer.step();loss_sum+=float(loss.detach())*b.num_graphs;n+=b.num_graphs
                if (j+1)%200==0:print(json.dumps(dict(epoch=epoch,step=j+1,steps=len(loader),loss=loss_sum/n,seconds=time.time()-start)),flush=True)
            status('validation',epoch=epoch);metrics=validate_sources(model,val,cache['graphs'],metadata,labels,device,a.batch_size);row=dict(epoch=epoch,train_loss=loss_sum/n,validation=metrics,seconds=time.time()-start)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if metrics['macro_objective']<best:
                best=metrics['macro_objective'];best_epoch=epoch;torch.save(dict(state_dict=model.state_dict(),args=args,epoch=epoch,validation=metrics,baseline_sha256=contract['baseline_sha256'],prepared_sha256=contract['prepared_sha256']),out/'best.pt')
            print(json.dumps(row),flush=True)
        (out/'complete.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_macro_objective=best,checkpoint_sha256=sha(out/'best.pt'),seconds=time.time()-start)));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

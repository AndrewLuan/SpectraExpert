"""功能概述：构造训练分区内同数据库分子记录的条件对比，训练可观测谱形响应。
condition_pairs只使用prepared保留身份，要求同HMDB分子ID、同SMILES、相同峰簇数
且至少两个簇。pH对比要求完全相同溶剂及已知不同pH；不把Water与D2O的差异归给
pH。溶剂族不同的记录可构成联合条件对比，不能称纯溶剂因果效应。
pair_response_loss比较两条件下可见谱分位数之差，目标来自各自已观测簇中心；
不增加缺失峰、不声明原子归属，也不将DMF测试结构或val/test记录放入训练对比。
main在冻结骨干和相同条件残差头上执行有/无对比损失的匹配继续训练，复用原监督，
按同一个宏平均val目标选epoch。val对比只作额外诊断；检索选择留给h_condeval。
输出独占权重、配置、对比清单、日志与退出码；无外部API或原始数据改写。
"""
import argparse
import collections
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
from .h_conditions import ConditionsSpectra
from .h_condmodel import training_details
from .h_condtrain import sampling_weights,validate_sources
from .h_labels import make_key
from .h_objective import objective
from .emission_model import make_model
from .train import collate,seed_worker
from .prepare import sha


def condition_pairs(rows,metadata,split='train'):
    groups=collections.defaultdict(list)
    for r in rows:
        if r['split']!=split or r['source']!='HMDB':continue
        meta=metadata[make_key(r)];parts=r['record_id'].split(':')
        if len(parts)<3:continue
        groups[r['smiles'],parts[1]].append((r,meta))
    pairs=[]
    for (smi,database),rs in groups.items():
        for i,(a,ma) in enumerate(rs):
            for b,mb in rs[i+1:]:
                if len(ma['line_centers'])!=len(mb['line_centers']) or len(ma['line_centers'])<2:continue
                exact=ma['solvent'].strip().lower()==mb['solvent'].strip().lower()
                ph=exact and ma['pH'] is not None and mb['pH'] is not None and abs(ma['pH']-mb['pH'])>=.3
                solvent=ma['family']!=mb['family']
                if not ph and not solvent:continue
                if np.allclose(ma['vector'],mb['vector']):continue
                pairs.append(dict(smiles=smi,database_id=database,kind='pH' if ph else 'joint_condition',a=a,b=b,solvents=[ma['solvent'],mb['solvent']],pH=[ma['pH'],mb['pH']]))
    return pairs


def pair_response_loss(model,pairs,graphs,metadata,device):
    rows=[p['a'] for p in pairs]+[p['b'] for p in pairs];b=collate([ConditionsSpectra(rows,graphs,'1H',metadata)[i] for i in range(len(rows))]).to(device)
    d=training_details(model(b,details=True),b);n=len(pairs);pred=d['quantiles'][n:]-d['quantiles'][:n];target=b.q[n:]-b.q[:n]
    return F.smooth_l1_loss(pred,target,beta=.02)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--init',required=True);p.add_argument('--out',required=True);p.add_argument('--pair-weight',type=float,default=1.);p.add_argument('--epochs',type=int,default=8);p.add_argument('--samples',type=int,default=32768);p.add_argument('--seed',type=int,default=47);a=p.parse_args();base=Path(a.base);prepared=Path(a.prepared);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,seconds=time.time()-start,**kw)))
    try:
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading')
        cache=torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False);labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        train=[r for r in cache['rows'] if r['split']=='train'];val=[r for r in cache['rows'] if r['split']=='val'];pairs=condition_pairs(cache['rows'],metadata,'train');valpairs=condition_pairs(cache['rows'],metadata,'val')
        if not pairs:raise ValueError('no_training_pairs')
        (out/'condition_pairs.json').write_text(json.dumps(dict(train=[dict(p,a=p['a']['record_id'],b=p['b']['record_id']) for p in pairs],validation=[dict(p,a=p['a']['record_id'],b=p['b']['record_id']) for p in valpairs]),indent=2))
        saved=torch.load(a.init,map_location='cpu',weights_only=False);model=make_model(saved['args']).to(device);model.load_state_dict(saved['state_dict']);weights=sampling_weights(train,metadata);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
        kinds=collections.Counter(p['kind'] for p in pairs);pw=np.array([1/kinds[p['kind']] for p in pairs],float);pw/=pw.sum();best=float('inf');best_epoch=0
        (out/'config.json').write_text(json.dumps(dict(args=vars(a),model_args=saved['args'],init_sha256=sha(a.init),prepared_sha256=saved['prepared_sha256'],pairs=len(pairs),pair_kinds=dict(kinds),validation_pairs=len(valpairs),selection='same macro source validation objective as preceding conditional training; no test choice',source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')}),indent=2))
        dataset=ConditionsSpectra(train,cache['graphs'],'1H',metadata,annotations=labels)
        for epoch in range(1,a.epochs+1):
            generator=torch.Generator().manual_seed(a.seed+epoch);sampler=WeightedRandomSampler(weights,a.samples,replacement=True,generator=generator);loader=DataLoader(dataset,batch_size=128,sampler=sampler,num_workers=2,collate_fn=collate,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(a.seed+epoch),pin_memory=True);rng=np.random.default_rng(a.seed+epoch);model.train();total=0.;pair_total=0.;n=0;status('training',epoch=epoch)
            for step,b in enumerate(loader):
                b=b.to(device);optimizer.zero_grad(set_to_none=True);d=training_details(model(b,details=True),b)
                if (d['quantiles'].abs()>1e4).any():raise ValueError('empty_nuclear_support')
                loss=objective(d,b,'site')['loss'];pair_loss=loss*0
                if step%4==0:
                    chosen=[pairs[i] for i in rng.choice(len(pairs),16,p=pw)];pair_loss=pair_response_loss(model,chosen,cache['graphs'],metadata,device);loss=loss+a.pair_weight*pair_loss
                if not torch.isfinite(loss):raise ValueError('nonfinite_loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.condition_head.parameters(),2.);optimizer.step();total+=float(loss.detach())*b.num_graphs;pair_total+=float(pair_loss.detach());n+=b.num_graphs
            metrics=validate_sources(model,val,cache['graphs'],metadata,labels,device,128)
            if valpairs:
                with torch.inference_mode():metrics['condition_pair_loss']=float(pair_response_loss(model,valpairs,cache['graphs'],metadata,device))
            row=dict(epoch=epoch,train_loss=total/n,train_pair_loss=pair_total/max(1,(step//4+1)),validation=metrics,seconds=time.time()-start)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if metrics['macro_objective']<best:
                best=metrics['macro_objective'];best_epoch=epoch;torch.save(dict(state_dict=model.state_dict(),args=saved['args'],epoch=epoch,validation=metrics,prepared_sha256=saved['prepared_sha256'],baseline_sha256=saved['baseline_sha256']),out/'best.pt')
            print(json.dumps(row),flush=True)
        (out/'complete.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_macro_objective=best,checkpoint_sha256=sha(out/'best.pt'),seconds=time.time()-start)));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

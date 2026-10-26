"""功能概述：在同一实验曝光量下区分条件残差容量和GINE表达误差。
configure按control/heads/last/all逐级解冻：条件头、原发射头、末层GINE、全部GINE。
更新GINE时显式禁用旧结构缓存；推断入口h_condinfer会按新权重重建候选表示。
所有分支用相同实验样本序列/联合甲基监督/40轮预算，只用原实验val选epoch；
保留理论预训练初始化，适配阶段不重放存在静态甲基口径差异的理论谱。
main保存参数量、分组学习率、seed、初始化/数据/源码hash、逐epoch指标及退出码；
检索仍由固定320-val选择，测试/候选身份不改。不读取测试或调用外部服务。
"""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .h_replaytrain import loader_for,grouped_objective,validate
from .h_condtrain import sampling_weights
from .h_condmodel import training_details
from .emission_model import make_model
from .prepare import sha

BACKBONE_MODES=['control','heads','last','all']


def configure(model,mode):
    if mode not in BACKBONE_MODES:raise ValueError('invalid_backbone_mode')
    groups={'condition':[],'emission':[],'GINE':[]};last=model.num_layer-1 if hasattr(model,'num_layer') else len(model.atom_bond_block_list)-1
    for name,param in model.named_parameters():
        conditional=name.startswith(('condition_head.','response_head.'));emission=name.startswith(('shift_head.','split_head.'));gnn=not conditional and not emission
        enabled=conditional or (mode!='control' and emission) or (mode=='all' and gnn) or (mode=='last' and name.startswith((f'bond_emb_list.{last}.',f'atom_bond_block_list.{last}.')))
        param.requires_grad_(enabled)
        if enabled:groups['condition' if conditional else 'emission' if emission else 'GINE'].append(param)
    model.use_frozen_embedding=mode in ['control','heads'];rates={'condition':1e-4,'emission':1e-5,'GINE':3e-6}
    return [dict(params=ps,lr=rates[k],name=k) for k,ps in groups.items() if ps]


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--init',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',choices=BACKBONE_MODES,required=True);p.add_argument('--seed',type=int,default=79);p.add_argument('--epochs',type=int,default=40);p.add_argument('--samples',type=int,default=65536);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,seconds=time.time()-start,**kw)))
    try:
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading');cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False);labels=torch.load(Path(a.base)/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False);train=[r for r in cache['rows'] if r['split']=='train' and r['domain']=='experimental'];val=[r for r in cache['rows'] if r['split']=='val']
        if a.smoke:train=train[:64];val=val[:32];a.epochs=1;a.samples=128
        saved=torch.load(a.init,map_location='cpu',weights_only=False);model=make_model(saved['args']).to(device);model.load_state_dict(saved['state_dict']);groups=configure(model,a.mode);optimizer=torch.optim.AdamW(groups,weight_decay=1e-4);weights=sampling_weights(train,metadata);cfg=dict(saved['args'],use_frozen_embedding=model.use_frozen_embedding,backbone_mode=a.mode,seed=a.seed);best=float('inf');best_epoch=0
        manifest=dict(args=vars(a),model_args=cfg,init_sha256=sha(a.init),train_records=len(train),validation_records=len(val),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),parameter_groups={g['name']:dict(lr=g['lr'],parameters=sum(p.numel() for p in g['params'])) for g in groups},selection='same full experimental grouped-site source macro validation objective; retrieval selection separate',source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')});(out/'config.json').write_text(json.dumps(manifest,indent=2))
        for epoch in range(1,a.epochs+1):
            loader=loader_for(train,cache['graphs'],metadata,labels,weights,a.seed+epoch,a.samples,64,False,True);model.train();total=0.;steps=0;status('training',epoch=epoch)
            for step,b in enumerate(loader):
                b=b.to(device);optimizer.zero_grad(set_to_none=True);d=training_details(model(b,details=True),b);loss=grouped_objective(d,b)['loss']
                if not torch.isfinite(loss):raise ValueError('nonfinite_backbone_loss')
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],2.);optimizer.step();total+=float(loss.detach());steps+=1
                if steps%256==0:print(json.dumps(dict(epoch=epoch,step=steps,loss=total/steps,seconds=time.time()-start)),flush=True)
            status('validation',epoch=epoch);metrics=validate(model,val,cache['graphs'],metadata,labels,device);row=dict(epoch=epoch,train_loss=total/steps,validation=metrics,seconds=time.time()-start)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if metrics['macro_objective']<best:
                best=metrics['macro_objective'];best_epoch=epoch;torch.save(dict(state_dict=model.state_dict(),args=cfg,epoch=epoch,validation=metrics,baseline_sha256=saved['baseline_sha256'],prepared_sha256=saved['prepared_sha256']),out/'best.pt')
            print(json.dumps(row),flush=True)
        (out/'complete.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_macro_objective=best,checkpoint_sha256=sha(out/'best.pt'),seconds=time.time()-start)));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

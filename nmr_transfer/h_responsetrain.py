"""功能概述：只训练明确条件记录上的锚定响应头，基础模型完全冻结。
main以同一实验样本序列比较ph/local/pairs/solvent四个等容量分支；缺少条件的
样本无需参与更新，因为模型按契约严格保持其原预测。普通训练复用原母节点目标。
pairs使用同HMDB身份、完全相同溶剂的六个纯pH比较；solvent另使用已有联合
条件比较。对比不被称为独立溶剂因果效应，所有比较均来自train连接分区。
验证仍用全体原实验val、来源宏平均选epoch。输出完整配对支持、曝光量和权重SHA，
不读取测试，不更改原缓存/权重；候选排序由h_condeval完成。
"""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from .h_response import AnchoredSpecGNN,RESPONSE_MODES
from .h_conditions import ConditionsSpectra
from .h_condtrain import sampling_weights,validate_sources
from .h_condmodel import training_details
from .h_condpairs import condition_pairs,pair_response_loss
from .h_labels import make_key
from .h_objective import objective
from .train import collate,seed_worker
from .prepare import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--init',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',choices=RESPONSE_MODES,required=True);p.add_argument('--epochs',type=int,default=25);p.add_argument('--samples',type=int,default=32768);p.add_argument('--seed',type=int,default=67);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,seconds=time.time()-start,**kw)))
    try:
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading');cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False);labels=torch.load(Path(a.base)/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        train=[r for r in cache['rows'] if r['split']=='train' and r['domain']=='experimental' and metadata[make_key(r)]['metadata_known']];val=[r for r in cache['rows'] if r['split']=='val'];pairs=condition_pairs(cache['rows'],metadata,'train');ph_pairs=[x for x in pairs if x['kind']=='pH']
        saved=torch.load(a.init,map_location='cpu',weights_only=False);cfg=dict(saved['args'],model='anchored_atom',response_mode=a.mode);model=AnchoredSpecGNN('1H',cfg['width'],cfg['layers'],cfg['condition_mode'],a.mode).to(device);model.amide_gap_scale=float(cfg.get('amide_gap_scale',1.));missing,unexpected=model.load_state_dict(saved['state_dict'],strict=False)
        if unexpected or any(not n.startswith('response_head.') for n in missing):raise ValueError('response_warmstart_mismatch')
        if a.smoke:train=train[:64]+[r for r in train if r['source']=='HMDB'][:16];val=val[:32];a.epochs=1;a.samples=128
        weights=sampling_weights(train,metadata);optimizer=torch.optim.AdamW(model.response_head.parameters(),lr=3e-4,weight_decay=1e-3);best=float('inf');best_epoch=0
        contract=dict(args=vars(a),model_args=cfg,init_sha256=sha(a.init),train_records=len(train),pure_pH_pairs=len(ph_pairs),pure_pH_structures=len({x['smiles'] for x in ph_pairs}),joint_condition_pairs=len(pairs)-len(ph_pairs),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),selection='original full experimental val source macro objective, then fixed retrieval val',source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')});(out/'config.json').write_text(json.dumps(contract,indent=2));(out/'pure_pH_pairs.json').write_text(json.dumps([dict(x,a=x['a']['record_id'],b=x['b']['record_id']) for x in ph_pairs],indent=2));dataset=ConditionsSpectra(train,cache['graphs'],'1H',metadata,annotations=labels)
        for epoch in range(1,a.epochs+1):
            sampler=WeightedRandomSampler(weights,a.samples,replacement=True,generator=torch.Generator().manual_seed(a.seed+epoch));loader=DataLoader(dataset,batch_size=128,sampler=sampler,num_workers=2,collate_fn=collate,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(a.seed+epoch),pin_memory=True);rng=np.random.default_rng(a.seed+epoch);model.train();total=0.;pair_total=0.;n=0;status('training',epoch=epoch)
            for step,b in enumerate(loader):
                b=b.to(device);optimizer.zero_grad(set_to_none=True);d=training_details(model(b,details=True),b);loss=objective(d,b,'site')['loss'];pair_loss=loss*0
                if step%4==0 and ph_pairs:
                    chosen=[ph_pairs[i] for i in rng.choice(len(ph_pairs),16)];pair_loss=pair_response_loss(model,chosen,cache['graphs'],metadata,device)
                    if a.mode in ['pairs','solvent']:loss=loss+2*pair_loss
                    if a.mode=='solvent':
                        chosen=[pairs[i] for i in rng.choice(len(pairs),16)];loss=loss+pair_response_loss(model,chosen,cache['graphs'],metadata,device)
                if not torch.isfinite(loss):raise ValueError('nonfinite_response_loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.response_head.parameters(),1.);optimizer.step();total+=float(loss.detach())*b.num_graphs;pair_total+=float(pair_loss.detach());n+=b.num_graphs
            metrics=validate_sources(model,val,cache['graphs'],metadata,labels,device,128)
            with torch.inference_mode():metrics['training_pure_pH_pair_loss']=float(pair_response_loss(model,ph_pairs,cache['graphs'],metadata,device)) if ph_pairs else None
            row=dict(epoch=epoch,train_loss=total/n,pair_loss=pair_total/max(1,step//4+1),validation=metrics,seconds=time.time()-start)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if metrics['macro_objective']<best:
                best=metrics['macro_objective'];best_epoch=epoch;torch.save(dict(state_dict=model.state_dict(),args=cfg,epoch=epoch,validation=metrics,baseline_sha256=saved['baseline_sha256'],prepared_sha256=saved['prepared_sha256']),out/'best.pt')
            print(json.dumps(row),flush=True)
        (out/'complete.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_macro_objective=best,checkpoint_sha256=sha(out/'best.pt'),seconds=time.time()-start)));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

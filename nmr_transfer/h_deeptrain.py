"""功能概述：以已完成的冻结骨干控制组为共同起点，训练额外消息上下文深度。
main先逐张量核对旧GINE与prepared缓存所属checkpoint，再加载完全相同初始
条件/发射权重；0/1/3/6深度都保留六块布局和零门控，只训练条件头及新上下文。
各分支同seed、同65536实验曝光/轮、40轮与联合甲基目标，完整val宏目标选epoch。
metadata-mode与sampling-mode分别控制输入条件与抽样分布；freeze-context固定新骨干，
只微调已有条件头，并在进程内一次缓存train/val结构的最终表示，不改磁盘cache。
旧/恢复条件使用相同样本身份、seed与曝光量，抽样变化单列消融。
balance-local-support仅在原source/solvent质量内平方根平衡NMe2酰胺样本。
不重放静态理论甲基、不读取测试；候选评测通过支持最终上下文缓存的h_condinfer。
输出独占配置/所有权证据/日志/权重/退出码；原始数据、缓存与起点权重只读。
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch_geometric.data import Batch
from .h_deep_context import DeepConditionSpecGNN,DEPTHS,verify_cached_backbone
from .h_replaytrain import loader_for,grouped_objective,validate
from .h_condtrain import sampling_weights
from .h_condmodel import training_details
from .prepare import sha
from .h_observation_study import stamp_queries
from .h_metadata import metadata_view,RECOVERED_MODE,balance_local_support


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--init',required=True);p.add_argument('--out',required=True);p.add_argument('--depth',type=int,choices=DEPTHS,required=True);p.add_argument('--seed',type=int,default=101);p.add_argument('--epochs',type=int,default=40);p.add_argument('--samples',type=int,default=65536);p.add_argument('--metadata-mode',choices=['legacy',RECOVERED_MODE],default='legacy');p.add_argument('--sampling-mode',choices=['legacy',RECOVERED_MODE],default='legacy');p.add_argument('--balance-local-support',action='store_true');p.add_argument('--freeze-context',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,seconds=time.time()-start,**kw)))
    try:
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading');base=Path(a.base);prepared=Path(a.prepared);cache=torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False);labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False);saved=torch.load(a.init,map_location='cpu',weights_only=False);owner=base/'hydrogen_analysis/site_supervision/train/best.pt';reference=torch.load(owner,map_location='cpu',weights_only=False);manifest=json.loads((prepared/'manifest.json').read_text())
        if manifest['H_checkpoint_sha256']!=sha(owner) or manifest['cache_sha256']!=saved['prepared_sha256']:raise ValueError('prepared_embedding_manifest_mismatch')
        verify_cached_backbone(saved['state_dict'],reference['state_dict']);cfg=dict(saved['args'],model='deep_condition_atom',context_depth=a.depth,seed=a.seed,use_frozen_embedding=True,condition_key_mode=RECOVERED_MODE if a.metadata_mode==RECOVERED_MODE else saved['args']['condition_mode']);model=DeepConditionSpecGNN('1H',cfg['width'],cfg['layers'],cfg['condition_mode'],cfg['response_mode'],a.depth).to(device);model.amide_gap_scale=float(cfg.get('amide_gap_scale',1.));missing,unexpected=model.load_state_dict(saved['state_dict'],strict=False)
        if unexpected or any(not n.startswith(('context_bonds.','context_blocks.','context_gates')) for n in missing):raise ValueError('deep_warmstart_mismatch')
        sampling_metadata=metadata_view(metadata,a.sampling_mode);metadata=metadata_view(metadata,a.metadata_mode)
        if a.freeze_context:
            for n,p in model.named_parameters():
                if n.startswith('context_'):p.requires_grad_(False)
        conditional=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('context_')];context=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('context_')];optimizer=torch.optim.AdamW([dict(params=conditional,lr=1e-4),dict(params=context,lr=5e-5)],weight_decay=1e-4);train=[r for r in cache['rows'] if r['split']=='train' and r['domain']=='experimental'];val=[r for r in cache['rows'] if r['split']=='val']
        if a.smoke:train=train[:64];val=val[:32];a.epochs=1;a.samples=128
        if a.freeze_context:
            required=sorted({r['smiles'] for r in train+val});model.eval();status('caching_frozen_context',structures=len(required))
            with torch.no_grad():
                for offset in range(0,len(required),512):
                    names=required[offset:offset+512];b=Batch.from_data_list([cache['graphs'][s] for s in names]).to(device);h=model.structure_embedding(b).detach().cpu();ptr=b.ptr.cpu().tolist()
                    for i,s in enumerate(names):cache['graphs'][s].final_context_embedding=h[ptr[i]:ptr[i+1]].clone()
            torch.cuda.empty_cache()
        weights=sampling_weights(train,sampling_metadata)
        if a.balance_local_support:weights=balance_local_support(train,sampling_metadata,weights)
        sample_hash=hashlib.sha256()
        for epoch in range(1,a.epochs+1):
            draw=torch.multinomial(weights,a.samples,replacement=True,generator=torch.Generator().manual_seed(a.seed+epoch));sample_hash.update(draw.numpy().tobytes())
        sampling_sha256=sample_hash.hexdigest();best=float('inf');best_epoch=0;(out/'config.json').write_text(json.dumps(dict(args=vars(a),model_args=cfg,init_sha256=sha(a.init),cached_GINE_owner_sha256=sha(owner),GINE_tensor_equality_verified=True,training_query_sha256=stamp_queries(train),sampling_sequence_sha256=sampling_sha256,metadata_sha256=manifest['metadata_sha256'],prepared_sha256=manifest['cache_sha256'],trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),selection='same full experimental grouped-site macro validation objective; no test choice',source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')}),indent=2))
        for epoch in range(1,a.epochs+1):
            loader=loader_for(train,cache['graphs'],metadata,labels,weights,a.seed+epoch,a.samples,64,False,True);model.train();total=0.;steps=0;status('training',epoch=epoch)
            for b in loader:
                b=b.to(device);optimizer.zero_grad(set_to_none=True);d=training_details(model(b,details=True),b);loss=grouped_objective(d,b)['loss']
                if not torch.isfinite(loss):raise ValueError('nonfinite_context_loss')
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],2.);optimizer.step();total+=float(loss.detach());steps+=1
                if steps%256==0:print(json.dumps(dict(epoch=epoch,step=steps,loss=total/steps,seconds=time.time()-start)),flush=True)
            metrics=validate(model,val,cache['graphs'],metadata,labels,device);row=dict(epoch=epoch,train_loss=total/steps,context_gates=model.context_gates.detach().cpu().tolist(),validation=metrics,seconds=time.time()-start)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if metrics['macro_objective']<best:
                best=metrics['macro_objective'];best_epoch=epoch;torch.save(dict(state_dict=model.state_dict(),args=cfg,epoch=epoch,validation=metrics,baseline_sha256=saved['baseline_sha256'],prepared_sha256=saved['prepared_sha256']),out/'best.pt')
            print(json.dumps(row),flush=True)
        (out/'complete.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_macro_objective=best,checkpoint_sha256=sha(out/'best.pt'),seconds=time.time()-start)));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

"""功能概述：同曝光比较原发射、软计数惩罚和结构K容量硬投影的¹H训练。
main读取冻结prepared和同一checkpoint，只训练发射/条件/响应/消息头，冻结GINE及
旧深上下文以安全复用原结构缓存；每臂同seed、采样序列、优化器与轮数。
软约束只约束全结构预测的冗余位置；硬约束在NuclearSpecGNN.forward同步投影
位移及分位点。不把未观测峰数用作结构K，不删除未报告H。--target-sidecar/--target-mode显式切换原始/校准/物理组平均
监督，apply_variants只改进程内train标签；原实验val保持原值，新增理论val独立诊断，
缺图使用核对权重的原GINE编码，源文件和prepared不改写。
validate报告所有来源，epoch只按原实验来源宏目标选，step0参与；理论val单列诊断。
输出独占目录的历史、配置、best.pt、状态和退出码；验证权重/cache身份及冻结体。
由h_condinfer后续在固定库/val/test复评；不在训练中读取test谱，不调用外部API。--equivalence-mode切换同头局部槽与认证
物理环境槽；--equivalence-only仅训练环境头，冻结体逐张量核验；
--theory-fraction显式控制理论重放抽样质量，各对照保持相同曝光。
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch
from .emission_model import make_model
from .data import graph,identity
from .h_conditions import local_environment
from .h_labels import make_key
from .h_count_constraint import count_penalty
from .h_deep_context import verify_cached_backbone
from .h_replaytrain import loader_for,grouped_objective,validate
from .h_condtrain import sampling_weights
from .h_condmodel import training_details
from .h_metadata import metadata_view,RECOVERED_MODE
from .h_observation_study import stamp_queries
from .prepare import sha


def apply_variants(rows,labels,sidecar,mode):
    result=[];matched=0
    for r in rows:
        key=make_key(r);v=sidecar["variants"].get(key)
        if v is not None:
            if r["split"]=="test" or identity(r["smiles"])[2]!=r["split"]:raise ValueError("target_partition_mismatch")
            rr=dict(r,q=v[mode]["q"].copy());labels[key]=v[mode]["annotation"];matched+=1;result.append(rr)
        else:result.append(r)
    return result,matched


def main():
    p=argparse.ArgumentParser()
    for k in ['base','prepared','init','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--equivalence-only',action='store_true');p.add_argument('--equivalence-mode',choices=['local','physical']);p.add_argument('--equivalence-registry');p.add_argument('--theory-fraction',type=float,default=.5);p.add_argument('--target-sidecar');p.add_argument('--target-mode',choices=['unchanged','raw','calibrated','physical'],default='unchanged');p.add_argument('--arm',choices=['control','soft1','soft10','hard'],required=True);p.add_argument('--epochs',type=int,default=12);p.add_argument('--samples',type=int,default=32768);p.add_argument('--seed',type=int,default=163);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,seconds=time.time()-start,**kw)))
    try:
        torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');base=Path(a.base);prepared=Path(a.prepared);status('loading')
        cache=torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=metadata_view(torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False),RECOVERED_MODE);labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False);saved=torch.load(a.init,map_location='cpu',weights_only=False);owner=base/'hydrogen_analysis/site_supervision/train/best.pt';manifest=json.loads((prepared/'manifest.json').read_text())
        if manifest['H_checkpoint_sha256']!=sha(owner) or manifest['cache_sha256']!=saved['prepared_sha256']:raise ValueError('prepared_embedding_manifest_mismatch')
        verify_cached_backbone(saved['state_dict'],torch.load(owner,map_location='cpu',weights_only=False)['state_dict']);cfg=dict(saved['args'],count_decode=a.arm=='hard')
        if a.equivalence_mode:
            if not a.equivalence_registry or a.arm=='hard':raise ValueError('equivalence_registry_required_and_separate_decoder')
            cfg.update(model='certified_group_atom',equivalence_mode=a.equivalence_mode,equivalence_registry_path=a.equivalence_registry,equivalence_registry_sha256=sha(a.equivalence_registry))
        model=make_model(cfg).to(device);missing,extra=model.load_state_dict(saved['state_dict'],strict=False)
        if extra or any(not k.startswith('equivalence_head.') for k in missing):raise ValueError('count_warmstart_mismatch')
        train_prefixes=('shift_head.','split_head.','condition_head.','response_head.','solvent_message_','equivalence_head.')
        if a.equivalence_only:
            if not a.equivalence_mode:raise ValueError('equivalence_only_requires_decoder')
            train_prefixes=('equivalence_head.',)
        for name,param in model.named_parameters():param.requires_grad_(name.startswith(train_prefixes))
        train=[r for r in cache['rows'] if r['split']=='train' and r['nucleus']=='1H'];val=[r for r in cache['rows'] if r['split']=='val' and r['nucleus']=='1H']
        theory_val=[];target_audit={}
        if a.target_mode!='unchanged':
            if not a.target_sidecar:raise ValueError('target_sidecar_required')
            side=torch.load(a.target_sidecar,map_location='cpu',weights_only=False);train,matched=apply_variants(train,labels,side,a.target_mode);theory_val,matched_val=apply_variants(side['theory_validation'],labels,side,a.target_mode);target_audit=dict(mode=a.target_mode,train_replaced=matched,theory_validation=matched_val,sidecar_sha256=sha(a.target_sidecar))
            encoder=make_model(torch.load(owner,map_location='cpu',weights_only=False)['args']).to(device).eval();encoder.load_state_dict(torch.load(owner,map_location='cpu',weights_only=False)['state_dict'])
            with torch.no_grad():
                for r in theory_val:
                    s=r['smiles']
                    if s not in cache['graphs']:
                        g=graph(s);g.local_environment=torch.tensor(local_environment(s));g.frozen_embedding=encoder.structure_embedding(Batch.from_data_list([g]).to(device)).cpu();cache['graphs'][s]=g
            del encoder;torch.cuda.empty_cache()
        if a.smoke:theory_val=theory_val[:16];train=[r for r in train if r['physics'][5]>0][:128]+[r for r in train if r['domain']=='experimental'][:128];val=train[:16]+[r for r in val if r['domain']=='experimental'][:16];a.samples=128;a.epochs=1
        weights=sampling_weights(train,metadata)
        if not 0<a.theory_fraction<1:raise ValueError('invalid_theory_sampling_fraction')
        theory_mask=torch.tensor([r['domain']=='theory' for r in train]);weights[theory_mask]*=a.theory_fraction/.5;weights[~theory_mask]*=(1-a.theory_fraction)/.5
        required=sorted({r['smiles'] for r in train+val+theory_val});model.eval();status('caching_frozen_context',structures=len(required))
        with torch.no_grad():
            for offset in range(0,len(required),512):
                names=required[offset:offset+512];b=Batch.from_data_list([cache['graphs'][s] for s in names]).to(device)
                if getattr(model,'conditioned_messages',False):h=model.unconditioned_embedding(b).cpu();attr='unconditioned_context_embedding'
                else:h=model.structure_embedding(b).cpu();attr='final_context_embedding'
                ptr=b.ptr.cpu().tolist()
                for j,s in enumerate(names):setattr(cache['graphs'][s],attr,h[ptr[j]:ptr[j+1]].clone())
        digest=hashlib.sha256()
        for epoch in range(1,a.epochs+1):digest.update(torch.multinomial(weights,a.samples,replacement=True,generator=torch.Generator().manual_seed(a.seed+epoch)).numpy().tobytes())
        parameters=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(parameters,lr=3e-5,weight_decay=1e-4);strength={'control':0.,'soft1':1.,'soft10':10.,'hard':0.}[a.arm]
        config=dict(target_audit=target_audit,args=vars(a),model_args=cfg,init_sha256=sha(a.init),sampling_sequence_sha256=digest.hexdigest(),training_query_sha256=stamp_queries(train),trainable_parameters=sum(p.numel() for p in parameters),source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},selection='original experimental validation source macro only including step0; theoretical target mode explicit in target_audit; fixed retrieval val later')
        (out/'config.json').write_text(json.dumps(config,indent=2));best=float('inf');best_epoch=0
        for epoch in range(a.epochs+1):
            total=penalty=0.;steps=0
            if epoch:
                model.train();status('training',epoch=epoch)
                for b in loader_for(train,cache['graphs'],metadata,labels,weights,a.seed+epoch,a.samples,64,False,True):
                    b=b.to(device);opt.zero_grad(set_to_none=True);d=model(b,details=True);regular=count_penalty(d,b) if strength else d['values'].sum()*0;loss=grouped_objective(training_details(d,b),b)['loss']+strength*regular+parameters[0].sum()*0
                    if not torch.isfinite(loss):raise ValueError('nonfinite_count_loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(parameters,2.);opt.step();steps+=1;total+=float(loss.detach());penalty+=float(regular.detach())
                    if steps%100==0:status('training',epoch=epoch,steps=steps,loss=total/steps)
            status('validation',epoch=epoch);metrics=validate(model,val,cache['graphs'],metadata,labels,device);sources={r['source'] for r in val if r['domain']=='experimental'};selection=float(np.mean([v['objective'] for k,v in metrics['by_source'].items() if k in sources]));row=dict(epoch=epoch,training_loss=total/max(1,steps),count_penalty=penalty/max(1,steps),validation=metrics,experimental_macro=selection,seconds=time.time()-start)
            if theory_val:row['theory_validation']=validate(model,theory_val,cache['graphs'],metadata,labels,device)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if selection<best:
                best=selection;best_epoch=epoch;torch.save(dict(state_dict=model.state_dict(),args=cfg,epoch=epoch,validation=metrics,baseline_sha256=saved['baseline_sha256'],prepared_sha256=saved['prepared_sha256']),out/'best.pt')
            print(json.dumps(row),flush=True)
        final=model.state_dict();frozen=[k for k,p in model.named_parameters() if not p.requires_grad]
        if any(not torch.equal(saved['state_dict'][k],final[k].cpu()) for k in frozen):raise ValueError('frozen_context_changed')
        (out/'complete.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_experimental_macro=best,checkpoint_sha256=sha(out/'best.pt'),frozen_body_exact=True,seconds=time.time()-start)));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

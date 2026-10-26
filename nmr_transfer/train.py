"""功能概述：以同一SpecGNN骨干完成理论预训练及四类实验训练对照，支持DDP。
--model atom使用先按核质量合并成分的NuclearSpecGNN；从原预训练模型只迁移GINE
骨干，原子输出头新建；同架构初始化时才建立冻结教师，防止把旧读出重新作为硬目标。
主要职责：Spectra返回CPU图/原始实验目标；augmentation仅在训练时扰动目标；
theory_replay根据训练分区配对拟合的源/核种仿射和分位残差变换理论标签，不改变实验标签；
main在train训练、完整val选模和早停，保存checkpoint、epoch日志、数据和源码sha。
模式：pretrain/scratch/finetune/augment/domain；后两者用等量理论重放，domain另加
训练配对校正、按残差MAD确定的扰动及条件残差头。模型比较不宣称单因素因果效应。
DDP每rank独立sampler，rank0完整验证并广播停止状态。评价始终eval/inference_mode；
不每batch清CUDA缓存，不反复取验证首批。输出目录必须新建；complete仅成功结束写入。
--h-labels加载独立¹H标签侧表；--h-objective site启用母节点/软分布/似然监督，
presence为同数据同训练轮数的继续训练对照。完整性只参与训练/验证，不传给检索。
Spectra同时附加canonical structure_smiles字符串，供认证环境解码器读取结构侧表；
原模型忽略该字段，标签/结构缓存及数值目标保持原契约。
modality_is_theory按已知理论/实验采集域逐记录生成；只供可选域条件解码器使用，
候选批次由预测器统一声明experimental，绝不继承候选自己的谱表/训练分区。
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch_geometric.data import Batch
from .model import SpecGNN
from .data import SCALES,quantiles
from .prepare import sha
from .observation import KINDS,observation_loss
from .emission_model import make_model
from .h_labels import make_key
from .h_objective import objective as h_objective


class Spectra(Dataset):
    def __init__(self, rows, graphs, nucleus, mode='', fits=None, annotations=None):
        self.rows=rows; self.graphs=graphs; self.nucleus=nucleus; self.mode=mode; self.fits=fits or {}
        self.annotations=annotations

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r=self.rows[i]; g=self.graphs[r['smiles']].clone()
        # 已知采集域是查询输入；候选预测统一由调用方标为experimental，不读取候选记录。
        g.modality_is_theory=torch.tensor([r['domain']=='theory'],dtype=torch.bool)
        q=np.array(r['q'],copy=True)/SCALES[self.nucleus]
        g.structure_smiles=r['smiles']
        g.cond=torch.tensor(r['cond']).reshape(1,4)
        # 目标是ppm位置分布；仪器频率不应被学成经验位移偏差，保留溶剂状态。
        g.cond[:,0]=0
        if r['domain']=='theory':
            g.cond.zero_()
        fit=self.fits.get(r['source']+'|'+self.nucleus)
        if self.mode=='domain' and r['domain']=='theory' and 'domain_q' in r:
            q=np.asarray(r['domain_q']).copy()/SCALES[self.nucleus]
        original_q=q.copy()
        observation=r.get('observation',{'kind':'unknown','windows':[]})
        kind=observation['kind'];windows=np.asarray(observation['windows'],np.float32).reshape(-1,2)/SCALES[self.nucleus]
        if self.mode in ('augment','domain'):
            # 缺峰先按相同观测位移整组删除，再做位移扰动；随机删除分位数点不能模拟整峰缺失。
            if np.random.random()<.2:
                positions=np.unique(q);kept=positions[np.random.random(len(positions))>.15]
                remaining=q[np.isin(q,kept)]
                if len(remaining)>=4 and len(kept)<len(positions):
                    q=quantiles(remaining)
                    kind='partial_window' if len(windows) else 'unknown'
            # 固定增强对照用小扰动；domain强度来自训练配对MAD，界限防止生成离谱目标。
            sigma=.003
            if self.mode=='domain' and fit and fit['molecules'] >= 20:
                sigma=float(np.clip(fit['residual_mad_ppm']/SCALES[self.nucleus],.001,.03))
            # 同一观测峰的所有分位数共享局部位移扰动，避免把单峰误造成连续谱。
            positions,inverse=np.unique(q,return_inverse=True)
            local=np.random.normal(0,sigma/3,len(positions))[inverse]
            q=q*(1+np.random.normal(0,.005))+np.random.normal(0,sigma)+local
            q=np.sort(q)
            # 裁掉随机频段；选择由物理ppm范围产生，不把原观测首末点当测量范围。
            if np.random.random()<.4:
                low,high=((-2.,16.) if self.nucleus=='1H' else (-30.,250.))
                low/=SCALES[self.nucleus];high/=SCALES[self.nucleus]
                pivot=np.random.uniform(low,high)
                crop=np.array([[low,pivot]] if np.random.random()<.5 else [[pivot,high]],np.float32)
                if len(windows):
                    intersections=[]
                    for l,u in windows:
                        ll,uu=max(l,crop[0,0]),min(u,crop[0,1])
                        if ll<uu:intersections.append([ll,uu])
                    crop=np.asarray(intersections,np.float32).reshape(-1,2)
                keep=np.zeros(len(q),bool)
                for l,u in crop:keep|=(q>=l)&(q<=u)
                if keep.sum()>=4:
                    q=quantiles(q[keep])
                    windows=crop;kind='window' if kind in ('complete_atoms','window') else 'partial_window'
        g.q=torch.tensor(q,dtype=torch.float32).reshape(1,-1)
        g.physics=torch.tensor(r['physics']).reshape(1,8)
        g.windows=torch.zeros(1,8,2);g.window_mask=torch.zeros(1,8,dtype=torch.bool)
        if len(windows):
            g.windows[0,:len(windows)]=torch.tensor(windows);g.window_mask[0,:len(windows)]=True
        g.coverage_kind=torch.tensor([KINDS[kind]],dtype=torch.long)
        if self.annotations is not None:
            meta=self.annotations.get(make_key(r),{});g.site_targets=torch.zeros(len(g.x),6);g.site_mask=torch.zeros(len(g.x),6,dtype=torch.bool)
            for node,values in meta.get('groups',{}).items():
                if len(values)>6:raise ValueError('more_than_six_site_targets')
                g.site_targets[int(node),:len(values)]=torch.tensor(values)/SCALES[self.nucleus]
                g.site_mask[int(node),:len(values)]=True
            z=g.x[:,:54].argmax(1)+1;src,tgt=g.edge_index;valid=(z[src]==1)&(z[tgt]!=1)
            g.site_expected=torch.bincount(tgt[valid],minlength=len(g.x))
            g.shape_supervision=torch.tensor([bool(meta.get('full_atom_labels') or meta.get('integral_consistent'))])
            g.supervised_q=torch.tensor(original_q,dtype=torch.float32).reshape(1,-1)
        return g


def collate(items):
    return Batch.from_data_list(items)


@torch.inference_mode()
def validate(model, loader, device, scale, h_kind=None):
    model.eval(); errors=[];objectives=[]
    for b in loader:
        b=b.to(device)
        if h_kind:
            details=model(b,details=True);pred=details['quantiles'];obj=h_objective(details,b,h_kind)
            objectives.extend([float(obj['loss'])]*b.num_graphs)
        else:pred=model(b)
        errors.extend((observation_loss(pred,b)*scale).cpu().tolist())
    if not errors:
        raise ValueError('empty_validation')
    result=dict(records=len(errors),mean_observation_error_ppm=float(np.mean(errors)),median_observation_error_ppm=float(np.median(errors)),
                metric='conditional quantile L1 for complete/window; directed observed-to-predicted matching for unknown/partial')
    if objectives:result['selection_objective_normalized']=float(np.mean(objectives))
    return result


def seed_worker(worker):
    seed=torch.initial_seed()%(2**32); np.random.seed(seed); random.seed(seed)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--nucleus',choices=list(SCALES),required=True);p.add_argument('--mode',choices=['pretrain','scratch','finetune','augment','domain'],required=True)
    p.add_argument('--init'); p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=6)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--seed',type=int,default=21);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--width',type=int,default=256);p.add_argument('--layers',type=int,default=3);p.add_argument('--smoke',action='store_true')
    p.add_argument('--model',choices=['specgnn','atom'],default='specgnn')
    p.add_argument('--h-labels');p.add_argument('--h-objective',choices=['presence','site'],default='presence')
    a=p.parse_args(); rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    if world>1:
        dist.init_process_group('nccl'); torch.cuda.set_device(local)
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    out=Path(a.out)
    if rank==0:
        out.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    cache=torch.load(a.cache,map_location='cpu',weights_only=False)
    domain='theory' if a.mode=='pretrain' else 'experimental'
    train=[r for r in cache['rows'] if r['nucleus']==a.nucleus and r['split']=='train' and r['domain']==domain]
    val=[r for r in cache['rows'] if r['nucleus']==a.nucleus and r['split']=='val' and r['domain']==domain]
    fits=json.loads((Path(a.cache).parent/'domain_shift.json').read_text())['fits']
    if a.mode in ['augment','domain']:
        # 固定hash选择等量理论重放，各对照重放身份相同；从不借入保留结构。
        theory=[r for r in cache['rows'] if r['nucleus']==a.nucleus and r['split']=='train' and r['domain']=='theory']
        theory.sort(key=lambda r:hashlib.sha256((r['smiles']+r['source']+r['record_id']).encode()).hexdigest())
        train += theory[:len(train)]
    annotations=torch.load(a.h_labels,map_location='cpu',weights_only=False) if a.h_labels else None
    structural_exclusions=0
    if annotations is not None:
        if a.nucleus!='1H' or a.model!='atom':raise ValueError('h_labels_require_H_atom_model')
        structural_exclusions=int(sum(r['physics'][0]<=0 for r in train))
        train=[r for r in train if r['physics'][0]>0]
        val=[r for r in val if r['physics'][0]>0]
    if a.smoke:
        train=train[:256];val=val[:64]
    if not train or not val:
        raise ValueError('empty_partition')
    model=make_model(vars(a)).to(device)
    if a.init:
        weights=torch.load(a.init,map_location='cpu',weights_only=False)
        init_kind=weights['args'].get('model','specgnn')
        state=weights['state_dict']
        if a.model=='atom' and init_kind!='atom':state={k:v for k,v in state.items() if k.startswith(('init_atom_emb.','bond_emb_list.','atom_bond_block_list.'))}
        missing,unexpected=model.load_state_dict(state,strict=False)
        allowed=('shift_head.','split_head.') if a.model=='atom' and init_kind!='atom' else ('adapter.',)
        if unexpected or any(not s.startswith(allowed) for s in missing):
            raise ValueError(('incompatible_warmstart',missing,unexpected))
    teacher=None
    if a.init and weights['args'].get('model','specgnn')==a.model:
        teacher=make_model(weights['args']).to(device);teacher.load_state_dict(weights['state_dict']);teacher.eval()
        for param in teacher.parameters():param.requires_grad_(False)
    contract=dict(args=vars(a),world_size=world,cache_sha256=sha(a.cache) if rank==0 else None,
                  init_sha256=sha(a.init) if a.init else None,
                  source_sha256={str(x.name):sha(x) for x in Path(__file__).parent.glob('*.py')},
                  train=len(train),validation=len(val),test_used=False,
                  structural_zero_H_training_exclusions=structural_exclusions,
                  h_labels_sha256=sha(a.h_labels) if a.h_labels else None,
                  train_sources=dict(collections.Counter(r['source'] for r in train)))
    if rank==0:(out/'config.json').write_text(json.dumps(contract,indent=2))
    train_ds=Spectra(train,cache['graphs'],a.nucleus,a.mode,fits,annotations)
    val_ds=Spectra(val,cache['graphs'],a.nucleus,annotations=annotations)
    sampler=DistributedSampler(train_ds,num_replicas=world,rank=rank,seed=a.seed,shuffle=True) if world>1 else None
    generator=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(train_ds,batch_size=a.batch_size,shuffle=sampler is None,sampler=sampler,num_workers=a.workers,
                      collate_fn=collate,worker_init_fn=seed_worker,generator=generator,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    valid=DataLoader(val_ds,batch_size=a.batch_size,num_workers=a.workers,collate_fn=collate,pin_memory=device.type=='cuda') if rank==0 else None
    network=DistributedDataParallel(model,device_ids=[local]) if world>1 else model
    optimizer=torch.optim.AdamW(network.parameters(),lr=5e-4 if a.mode in ['pretrain','scratch'] else 1e-4,weight_decay=1e-5)
    best=float('inf');best_epoch=0;best_metrics=None;start=time.time()
    for epoch in range(1,a.epochs+1):
        network.train()
        if sampler is not None:sampler.set_epoch(epoch)
        sums=torch.zeros(2,device=device)
        for step,b in enumerate(loader,1):
            b=b.to(device);optimizer.zero_grad(set_to_none=True)
            if annotations is not None:
                details=network(b,details=True);pred=details['quantiles'];loss=h_objective(details,b,a.h_objective)['loss']
            else:pred=network(b);loss=observation_loss(pred,b).mean()
            if teacher is not None:
                unknown=(b.coverage_kind==KINDS['unknown']) | (b.coverage_kind==KINDS['partial_window'])
                if unknown.any():
                    with torch.no_grad():prior=teacher(b)
                    loss=loss+.1*(pred[unknown]-prior[unknown]).square().mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite_loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5.);optimizer.step()
            sums[0]+=loss.detach()*b.num_graphs;sums[1]+=b.num_graphs
            if rank==0 and step%200==0:
                print(json.dumps({'epoch':epoch,'step':step,'steps':len(loader),'loss':float(loss),'seconds':time.time()-start}),flush=True)
        if world>1:dist.reduce(sums,dst=0)
        stop=torch.zeros(1,device=device)
        if rank==0:
            metrics=validate(model,valid,device,SCALES[a.nucleus],a.h_objective if annotations is not None else None)
            value=metrics.get('selection_objective_normalized',metrics['mean_observation_error_ppm'])
            row=dict(epoch=epoch,train_loss_normalized=float(sums[0]/sums[1]),validation=metrics,seconds=time.time()-start)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if value<best:
                best=value;best_epoch=epoch;best_metrics=metrics
                payload=dict(state_dict=model.state_dict(),args=vars(a),epoch=epoch,validation=metrics,cache_sha256=contract['cache_sha256'])
                torch.save(payload,out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch),out/'last.tmp');os.replace(out/'last.tmp',out/'last.pt')
            stop.fill_(epoch-best_epoch>=a.patience)
            print(json.dumps(row),flush=True)
        if world>1:dist.broadcast(stop,src=0)
        if stop.item():break
    if rank==0:
        (out/'complete.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_validation_ppm=best_metrics['mean_observation_error_ppm'],
            best_selection_value=best,selection_metric='h_objective_normalized' if annotations is not None else 'observation_error_ppm',
            checkpoint_sha256=sha(out/'best.pt'),seconds=time.time()-start)))
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()

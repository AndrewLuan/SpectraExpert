"""功能概述：固定实验抽样序列，区分理论目标校准、重放强度和酰胺联合监督。
ReplaySpectra只在训练理论记录显式选择已有domain_q，不附加扰动或改写缓存；
所有分支每步读取相同64条实验和64条理论记录，损失为Lexp+lambda*Ltheory。
grouped_objective对稳定的酰胺NMe2甲基标注使用无序联合监督，避免二维等价节点
只能输出同一分裂混合而单母节点标签强制选择某一支。单个甲基标注仍是部分观测，
不补出另一甲基的位移。此监督只用于实验标签；理论静态构象不作等价峰扩写。
显式theory_orbits开关可将理论完整母节点标签在二维自同构轨道中无序合并，
保留原位移及原母节点总权重而避免任意源原子编号定向监督；不将缺失位点扩写成完整标签。
六组共享初始参数、实验曝光量、优化器和验证目标；选epoch只用原实验val，
检索选型由后续固定320条val进行。输出独占日志/权重/身份hash和退出码。
ReplaySpectra将显式review_spectral_weight逐图传递给审核训练入口；默认1，原目标
函数忽略该字段，审核目标只把已裁决的实验谱损失归零，不改变抽样或原q。
"""
import argparse
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from .data import SCALES
from .h_conditions import ConditionsSpectra
from .h_condmodel import training_details
from .h_condtrain import sampling_weights
from .h_amide_audit import amide_groups
from .h_objective import objective,site_loss
from .h_labels import make_key
from .emission_model import make_model
from .train import collate,seed_worker
from .observation import observation_loss
from .prepare import sha
from .h_symmetry_supervision import apply_theory_orbits

VARIANTS={'raw1':(False,1.,False),'cal1':(True,1.,False),'cal01':(True,.1,False),'cal0':(True,0.,False),'cal0_group':(True,0.,True),'cal01_group':(True,.1,True)}


class ReplaySpectra(ConditionsSpectra):
    def __init__(self,*args,calibrated=False,grouped=False,theory_orbits=False,**kwargs):
        super().__init__(*args,**kwargs);self.calibrated=calibrated;self.grouped=grouped;self.theory_orbits=theory_orbits
        if theory_orbits and not grouped:raise ValueError('theory_orbits_require_grouped_objective')
    def __getitem__(self,i):
        g=super().__getitem__(i);r=self.rows[i]
        g.review_spectral_weight=torch.tensor([r.get('review_spectral_weight',1.)],dtype=torch.float32)
        if self.calibrated and r['domain']=='theory' and 'domain_q' in r:
            g.q=torch.tensor(np.asarray(r['domain_q']).copy()/SCALES['1H'],dtype=torch.float32).reshape(1,-1);g.supervised_q=g.q.clone()
        if self.grouped:
            g.group_targets=g.site_targets.clone();g.group_mask=g.site_mask.clone();g.group_expected=g.site_expected.clone();g.group_owner=torch.arange(len(g.x));g.group_site_weight=torch.ones(len(g.x))
            if r['domain']=='experimental':
                for n,parents in amide_groups(g):
                    if g.site_expected[n]>0:continue
                    observed=[g.site_targets[p][g.site_mask[p]] for p in parents]
                    if not any(len(v) for v in observed) or any(len(v) and float(v.max()-v.min())>.003 for v in observed):continue
                    values=torch.cat([v.median().repeat(3) for v in observed if len(v)])
                    g.group_targets[parents]=0;g.group_mask[parents]=False;g.group_expected[parents]=0;g.group_targets[n]=0;g.group_mask[n]=False;g.group_targets[n,:len(values)]=values;g.group_mask[n,:len(values)]=True;g.group_expected[n]=6;g.group_owner[parents]=n
            elif self.theory_orbits:apply_theory_orbits(g,r['smiles'])
        return g


def grouped_objective(details,batch):
    parts=objective(details,batch,'site')
    # PyG对非索引字段不偏移；按各图ptr恢复局部母节点到批次全局编号。
    owners=batch.group_owner+batch.ptr[batch.batch]
    proxy=SimpleNamespace(x=batch.x,site_targets=batch.group_targets,site_mask=batch.group_mask,site_expected=batch.group_expected,site_weights=batch.group_site_weight,batch=batch.batch,num_graphs=batch.num_graphs)
    grouped,valid=site_loss(dict(details,owners=owners[details['owners']]),proxy);site=grouped[valid].mean() if valid.any() else parts['site']*0
    return dict(parts,loss=parts['loss']-.75*parts['site']+.75*site,site=site)


def loader_for(rows,graphs,metadata,labels,weights,seed,samples,batch_size,calibrated,grouped,theory_orbits=False):
    dataset=ReplaySpectra(rows,graphs,'1H',metadata,annotations=labels,calibrated=calibrated,grouped=grouped,theory_orbits=theory_orbits)
    sampler=WeightedRandomSampler(weights,samples,replacement=True,generator=torch.Generator().manual_seed(seed))
    return DataLoader(dataset,batch_size=batch_size,sampler=sampler,num_workers=2,collate_fn=collate,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed),pin_memory=True)


@torch.inference_mode()
def validate(model,rows,graphs,metadata,labels,device):
    model.eval();metrics={}
    for source in sorted({r['source'] for r in rows}):
        rs=[r for r in rows if r['source']==source];loader=DataLoader(ReplaySpectra(rs,graphs,'1H',metadata,annotations=labels,grouped=True),batch_size=128,num_workers=2,collate_fn=collate);total=0.;error=0.;n=0
        for b in loader:
            b=b.to(device);d=training_details(model(b,details=True),b);total+=float(grouped_objective(d,b)['loss'])*b.num_graphs;error+=float(observation_loss(d['quantiles'],b).sum())*10;n+=b.num_graphs
        metrics[source]=dict(records=n,objective=total/n,observed_error_ppm=error/n)
    return dict(by_source=metrics,macro_objective=float(np.mean([v['objective'] for v in metrics.values()])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--init',required=True);p.add_argument('--out',required=True);p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--epochs',type=int,default=8);p.add_argument('--samples',type=int,default=32768);p.add_argument('--seed',type=int,default=59);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,seconds=time.time()-start,**kw)))
    try:
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading');cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False);labels=torch.load(Path(a.base)/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        exp=[r for r in cache['rows'] if r['split']=='train' and r['domain']=='experimental'];theory=[r for r in cache['rows'] if r['split']=='train' and r['domain']=='theory'];val=[r for r in cache['rows'] if r['split']=='val'];weights=sampling_weights(exp,metadata);tw=torch.ones(len(theory),dtype=torch.float64)
        if a.smoke:exp=exp[:64];theory=theory[:64];weights=weights[:64];tw=tw[:64];val=val[:32];a.epochs=1;a.samples=128
        saved=torch.load(a.init,map_location='cpu',weights_only=False);model=make_model(saved['args']).to(device);model.load_state_dict(saved['state_dict']);optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=1e-4);calibrated,weight,grouped=VARIANTS[a.variant];best=float('inf');best_epoch=0
        config=dict(args=vars(a),model_args=saved['args'],init_sha256=sha(a.init),lambda_theory=weight,calibrated_theory=calibrated,grouped_experimental_methyls=grouped,experimental_samples_per_epoch=a.samples,theory_samples_per_epoch=a.samples,batch_size_each=64,validation='same grouped experimental macro objective for every arm; no test',source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')});(out/'config.json').write_text(json.dumps(config,indent=2))
        for epoch in range(1,a.epochs+1):
            el=loader_for(exp,cache['graphs'],metadata,labels,weights,a.seed+epoch,a.samples,64,False,grouped);tl=loader_for(theory,cache['graphs'],metadata,labels,tw,a.seed+10000+epoch,a.samples,64,calibrated,False);model.train();totals=np.zeros(2);steps=0;status('training',epoch=epoch)
            for eb,tb in zip(el,tl):
                optimizer.zero_grad(set_to_none=True);eb=eb.to(device);ed=training_details(model(eb,details=True),eb);ep=grouped_objective(ed,eb) if grouped else objective(ed,eb,'site');loss=ep['loss'];theory_loss=loss.detach()*0
                # 仍读取相同理论批次；lambda=0不计算无效反向图，实验曝光保持一致。
                if weight:
                    tb=tb.to(device);td=training_details(model(tb,details=True),tb);theory_loss=objective(td,tb,'site')['loss'];loss=loss+weight*theory_loss
                if not torch.isfinite(loss):raise ValueError('nonfinite_loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.condition_head.parameters(),2.);optimizer.step();totals+=np.array([float(ep['loss'].detach()),float(theory_loss.detach())]);steps+=1
            metrics=validate(model,val,cache['graphs'],metadata,labels,device);row=dict(epoch=epoch,experimental_loss=totals[0]/steps,theory_loss=totals[1]/steps,validation=metrics,seconds=time.time()-start)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if metrics['macro_objective']<best:
                best=metrics['macro_objective'];best_epoch=epoch;torch.save(dict(state_dict=model.state_dict(),args=saved['args'],epoch=epoch,validation=metrics,baseline_sha256=saved['baseline_sha256'],prepared_sha256=saved['prepared_sha256']),out/'best.pt')
            print(json.dumps(row),flush=True)
        (out/'complete.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_macro_objective=best,checkpoint_sha256=sha(out/'best.pt'),seconds=time.time()-start)));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

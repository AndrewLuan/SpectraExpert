"""功能概述：同起点、同曝光训练酰基晚期头与无条件/溶剂条件消息三个消融。
main读取冻结prepared、恢复条件及旧权重，验证GINE/cache所有权；只缓存未条件化
的旧深骨干，不缓存新增消息。既有全部权重冻结，新模块由相同seed初始化。
support_audit按来源/溶剂/酰基类别清点train/val独立连接结构，不读取test峰标签。
原实验train抽样分布、grouped_objective和完整val宏目标保持一致；step0也参与
epoch选择，防止把退化模型替换基线。sample序列hash用于跨臂核对曝光完全相同。
--extra-support启用h_message_support经SHA/分区/积分合理性核对的甲酰胺记录，
按固定extra-fraction分配曝光，新增5结构val只做诊断，不参与epoch选型。
默认12轮×32768次曝光；输出目录独占，保存args、源码/数据hash、历史和退出码；
不改变原缓存、候选、测试或碳模型。h_condinfer负责后续固定检索验证与测试。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch
from .h_solvent_message import SolventMessageSpecGNN,MESSAGE_MODES
from .h_deep_context import verify_cached_backbone
from .h_acyl import acyl_environment,ACYL_NAMES
from .h_replaytrain import loader_for,grouped_objective,validate
from .h_condtrain import sampling_weights
from .h_condmodel import training_details
from .h_metadata import metadata_view,RECOVERED_MODE
from .h_labels import make_key
from .h_observation_study import stamp_queries
from .prepare import sha
from .h_message_support import add_support,extra_weights
from .h_solvent_train import validate as validate_support


def support_audit(rows,metadata):
    groups=collections.defaultdict(set);counts=collections.Counter()
    for r in rows:
        env=acyl_environment(r['smiles']);family=metadata.get(make_key(r),{}).get('family','unknown')
        for tag in ACYL_NAMES[1:7]:
            if env[:,ACYL_NAMES.index(tag)].any():
                key=(r['split'],r['source'],family,tag);counts[key]+=1;groups[key].add(r['group'])
    return [dict(split=k[0],source=k[1],solvent=k[2],acyl=k[3],records=counts[k],connectivities=len(v)) for k,v in sorted(groups.items())]


def main():
    p=argparse.ArgumentParser()
    for name in ['base','prepared','init','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--arm',choices=MESSAGE_MODES,required=True);p.add_argument('--epochs',type=int,default=12);p.add_argument('--samples',type=int,default=32768);p.add_argument('--seed',type=int,default=139);p.add_argument('--extra-support');p.add_argument('--extra-fraction',type=float,default=.05);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');base=Path(a.base);prepared=Path(a.prepared);status('loading')
        cache=torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=metadata_view(torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False),RECOVERED_MODE);labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False);saved=torch.load(a.init,map_location='cpu',weights_only=False);owner=base/'hydrogen_analysis/site_supervision/train/best.pt';manifest=json.loads((prepared/'manifest.json').read_text())
        if manifest['H_checkpoint_sha256']!=sha(owner) or manifest['cache_sha256']!=saved['prepared_sha256']:raise ValueError('prepared_embedding_manifest_mismatch')
        verify_cached_backbone(saved['state_dict'],torch.load(owner,map_location='cpu',weights_only=False)['state_dict'])
        cfg=dict(saved['args'],model='solvent_message_atom',message_mode=a.arm,condition_key_mode=RECOVERED_MODE,use_frozen_embedding=True)
        model=SolventMessageSpecGNN('1H',cfg['width'],cfg['layers'],cfg['condition_mode'],cfg['response_mode'],cfg['context_depth'],a.arm).to(device);model.amide_gap_scale=float(cfg.get('amide_gap_scale',1.));missing,extra=model.load_state_dict(saved['state_dict'],strict=False)
        if extra or any(not k.startswith('solvent_message_') for k in missing):raise ValueError('message_warmstart_mismatch')
        train=[r for r in cache['rows'] if r['split']=='train' and r['domain']=='experimental'];val=[r for r in cache['rows'] if r['split']=='val'];weights=sampling_weights(train,metadata);extra_val=[]
        if a.extra_support:
            extra,extra_val,audit=add_support(cache,metadata,labels,a.extra_support,manifest,owner,device);weights=extra_weights(weights,extra,a.extra_fraction);train=train+extra;(out/'extra_support_audit.json').write_text(json.dumps(audit,indent=2))
        required=sorted({r['smiles'] for r in train+val+extra_val});model.eval();status('caching_unconditioned_context',structures=len(required))
        with torch.no_grad():
            for offset in range(0,len(required),512):
                names=required[offset:offset+512];b=Batch.from_data_list([cache['graphs'][s] for s in names]).to(device);h=model.unconditioned_embedding(b).cpu();ptr=b.ptr.cpu().tolist()
                for j,s in enumerate(names):cache['graphs'][s].unconditioned_context_embedding=h[ptr[j]:ptr[j+1]].clone()
        digest=hashlib.sha256()
        for epoch in range(1,a.epochs+1):digest.update(torch.multinomial(weights,a.samples,replacement=True,generator=torch.Generator().manual_seed(a.seed+epoch)).numpy().tobytes())
        config=dict(args=vars(a),model_args=cfg,init_sha256=sha(a.init),training_query_sha256=stamp_queries(train),sampling_sequence_sha256=digest.hexdigest(),metadata_sha256=sha(prepared/'metadata.pt'),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},selection='full validation grouped source macro, including unchanged step0; no test selection')
        (out/'config.json').write_text(json.dumps(config,indent=2));(out/'support_audit.json').write_text(json.dumps(support_audit(train+[r for r in val if r['domain']=='experimental'],metadata),indent=2));parameters=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4)
        best=float('inf');best_epoch=0
        for epoch in range(a.epochs+1):
            total=0.;steps=0
            if epoch:
                model.train();status('training',epoch=epoch)
                for b in loader_for(train,cache['graphs'],metadata,labels,weights,a.seed+epoch,a.samples,64,False,True):
                    b=b.to(device);opt.zero_grad(set_to_none=True);loss=grouped_objective(training_details(model(b,details=True),b),b)['loss']
                    if not torch.isfinite(loss):raise ValueError('nonfinite_message_loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(parameters,2.);opt.step();steps+=1;total+=float(loss.detach())
            status('validation',epoch=epoch);metrics=validate(model,val,cache['graphs'],metadata,labels,device);row=dict(epoch=epoch,training_loss=total/max(1,steps),validation=metrics,gates=model.solvent_message_gates.detach().cpu().tolist())
            if extra_val:row['extra_validation_connectivity_macro']=validate_support(model,extra_val,cache['graphs'],metadata,labels,device)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if metrics['macro_objective']<best:
                best=metrics['macro_objective'];best_epoch=epoch;torch.save(dict(state_dict=model.state_dict(),args=cfg,epoch=epoch,validation=metrics,baseline_sha256=saved['baseline_sha256'],prepared_sha256=saved['prepared_sha256']),out/'best.pt')
            print(json.dumps(row),flush=True)
        final=torch.load(out/'best.pt',map_location='cpu',weights_only=False)['state_dict']
        if any(not torch.equal(v,final[k]) for k,v in saved['state_dict'].items()):raise ValueError('frozen_body_changed')
        (out/'complete.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_macro_objective=best,checkpoint_sha256=sha(out/'best.pt'),frozen_body_exact=True)));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

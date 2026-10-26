"""功能概述：用整体训练结构的检索负例与教师保持约束微调 SpecGNN 分组残差。

main 复用理论预训练和已审核实验权重；--scope=head 默认只训练 equivalence_head，
--scope=messages 另解冻 solvent_message_，原深层骨干和其他头仍逐张量冻结核对。
configure_trainable 返回明确的训练前缀及名称；assert_frozen_core 排除这些前缀后
逐张量检查原权重。相同 rank/anchor 配对臂保留原始抽样、负例曝光和缓存。使用全部训练来源，取消此前
苯中甲酰胺的定向 5% 支持。best 仅由本轮1024-tune前向指标选择，另保留固定
中间检查点；head沿用既有train/val选型，messages只评epoch24。原320和封存verify不进入本入口。
build_cases 只从2048-train难负例缓存选纯训练连接结构，排除同连接、无H与自身；
make_rank_batch 重建原哈希缺峰/裁窗查询，给负候选同一查询条件、目标置为查询谱，
不读取负候选实验标签。HMDB用实际等权峰簇并同步掩码；完整物理teacher保持全H。
缓存始终仅保存 unconditioned_context_embedding，条件消息在每次前向重新计算。
head 沿用旧选择记录；messages 仍保存8/16/24/best，但下游只评预声明epoch24。
独占保存initial/best/last、曲线、scope/前缀/源码SHA、抽样SHA和退出码。
不修改原始数据、候选、测试或旧检查点，不调用任何外部服务。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch_geometric.data import Batch

from .data import identity,quantiles
from .emission_model import make_model
from .evaluate import missing_queries
from .h_condinfer import scoring_inputs
from .h_condmodel import training_details
from .h_condtrain import sampling_weights
from .h_count_curriculum import validate_spectrum
from .h_global_features import observed_support
from .h_global_forward_loss import observed_nll,pairwise_rank_loss,teacher_anchor
from .h_labels import make_key
from .h_metadata import metadata_view,RECOVERED_MODE
from .h_observation_study import stamp_queries,uniform_peaks
from .h_replaytrain import ReplaySpectra
from .h_reviewed_supervision import load_artifact,apply_review,reviewed_objective
from .h_score import attach_observation
from .h_site_quality import apply_quality
from .prepare import sha
from .train import collate,seed_worker


TRAIN_PREFIXES={'head':('equivalence_head.',),'messages':('equivalence_head.','solvent_message_')}


def configure_trainable(model,scope='head'):
    """只按预声明名称解冻；默认路径与原逐参数 startswith 判断及顺序相同。"""
    if scope not in TRAIN_PREFIXES:raise ValueError('global_forward_unknown_scope:'+str(scope))
    prefixes=TRAIN_PREFIXES[scope];names=[]
    for name,param in model.named_parameters():
        enabled=name.startswith(prefixes);param.requires_grad_(enabled)
        if enabled:names.append(name)
    if any(not any(name.startswith(prefix) for name in names) for prefix in prefixes):
        raise ValueError('global_forward_missing_trainable_prefix:'+scope)
    return prefixes,names


def assert_frozen_core(initial,current,prefixes):
    """训练前缀之外的参数和buffer必须与初始化逐点一致，缺键/新增键同样失败。"""
    if initial.keys()!=current.keys():raise ValueError('global_forward_state_keys_changed')
    checked=0
    for name,value in initial.items():
        if not name.startswith(prefixes):
            if not torch.equal(value.cpu(),current[name].detach().cpu()):
                raise ValueError('global_forward_frozen_core_changed:'+name)
            checked+=1
    return checked


def build_cases(feature,exp,bank,negatives=4):
    lookup={(r['source'],r['record_id'],r['smiles']):r for r in exp if r['review_spectral_weight']>0}
    smiles=feature['smiles']; ids=bank['single_indices']; physics=bank['physics'][ids]
    groups={}; cases=[]; dropped=collections.Counter()
    for row in feature['rows']:
        key=(row['source'],row['record_id'],row['smiles'])
        if key not in lookup:dropped['unreviewed_positive']+=1;continue
        chosen=[];chosen_groups=set()
        for i in row['candidate_indices'].tolist():
            name=smiles[i]
            if name not in groups:groups[name]=identity(name)
            _,group,split=groups[name]
            if split!='train' or group==row['group'] or group in chosen_groups or physics[i,0]<=0:continue
            chosen.append(dict(smiles=name,group=group,physics=physics[i].numpy()))
            chosen_groups.add(group)
            if len(chosen)==negatives:break
        if len(chosen)!=negatives:dropped['insufficient_distinct_train_negatives']+=1;continue
        cases.append(dict(positive=lookup[key],variant=row['variant'],negative=chosen,group=row['group']))
    if not cases:raise ValueError('no_global_forward_training_pairs')
    return cases,dict(cases=len(cases),positive_groups=len({p['group'] for p in cases}),dropped=dict(dropped),
        negative_scope='train connectivity only; same connectivity conservatively excluded; no candidate labels read')


def make_rank_batch(cases,graphs,metadata,labels,raw):
    rows=[]; view={}
    for ordinal,case in enumerate(cases):
        original=case['positive']; query=attach_observation(original,labels.get(make_key(original),{}))
        variants={'observed':query}; variants.update(dict(missing_queries(query,'1H')))
        if case['variant'] not in variants:raise ValueError('ranking_observation_variant_drift')
        query=variants[case['variant']]
        meta=dict(metadata[make_key(original)])
        if original['source']=='HMDB':
            support=observed_support(query,raw); meta['line_centers']=support['xs'].tolist()
            query=dict(query,q=quantiles(support['xs']))
        for structure in [dict(smiles=original['smiles'],group=original['group'],physics=original['physics'])]+case['negative']:
            row=dict(query,**structure)
            row['record_id']='ranking_'+str(ordinal)+'_'+original['record_id']
            rows.append(row); view[make_key(row)]=meta
    dataset=ReplaySpectra(rows,graphs,'1H',view,annotations={},grouped=False)
    return collate([dataset[i] for i in range(len(dataset))]),len(cases),1+len(cases[0]['negative'])


def main():
    p=argparse.ArgumentParser()
    for name in ('base','prepared','cohorts','init','bank','features','review','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--rank',action='store_true');p.add_argument('--anchor',action='store_true')
    p.add_argument('--scope',choices=tuple(TRAIN_PREFIXES),default='head')
    p.add_argument('--seed',type=int,default=193);p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--samples',type=int,default=32768);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def status(stage,**kw):
        tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**kw)));tmp.replace(out/'status.json')
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda')
        status('loading');base=Path(args.base);prepared=Path(args.prepared)
        cache=torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True)
        metadata=metadata_view(torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False),RECOVERED_MODE)
        labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        artifact,artifact_meta=load_artifact(args.review)
        exp=[r for r in cache['rows'] if r['domain']=='experimental' and r['split']=='train' and r['nucleus']=='1H']
        exp,audit=apply_review(labels,exp,artifact,'domain');quality=apply_quality(labels,exp,'methyl_record_mask')
        full_train_records=len(exp)
        if args.scope=='messages' and not args.smoke and full_train_records!=94571:
            raise ValueError('global_message_training_record_count_drift:'+str(full_train_records))
        payload=torch.load(Path(args.cohorts)/'cohorts.pt',map_location='cpu',weights_only=False)
        val=payload['cohorts']['tune']
        if {r['group'] for r in exp}&{r['group'] for r in val}:raise ValueError('global_forward_train_val_overlap')
        saved=torch.load(args.init,map_location='cpu',weights_only=False)
        if saved['prepared_sha256']!=json.loads((prepared/'manifest.json').read_text())['cache_sha256']:
            raise ValueError('global_forward_prepared_mismatch')
        torch.manual_seed(args.seed);cfg=dict(saved['args'],reporting_enabled=False)
        model=make_model(cfg).to(device);model.load_state_dict(saved['state_dict'])
        teacher=make_model(cfg).to(device).eval();teacher.load_state_dict(saved['state_dict'])
        for parameter in teacher.parameters():parameter.requires_grad_(False)
        feature=torch.load(args.features,map_location='cpu',weights_only=False,mmap=True)
        if feature['manifest']['cohort']!='train' or feature['manifest']['checkpoint_sha256']!=sha(args.init):
            raise ValueError('global_forward_negative_cache_mismatch')
        bank=torch.load(args.bank,map_location='cpu',weights_only=False,mmap=True)
        if bank['checkpoint_sha256']!=sha(args.init):raise ValueError('global_forward_initial_bank_mismatch')
        if feature['smiles']!=[bank['smiles'][i] for i in bank['single_indices']]:
            raise ValueError('global_forward_candidate_physics_order_mismatch')
        cases,pair_audit=build_cases(feature,exp,bank)
        if args.smoke:
            names={make_key(c['positive']) for c in cases[:8]};exp=[r for r in exp if make_key(r) in names]+exp[:64]
            val=val[:32];cases=cases[:8];args.epochs=1;args.samples=128
        weights=sampling_weights(exp,metadata);dataset=ReplaySpectra(exp,cache['graphs'],'1H',metadata,annotations=labels,grouped=True)
        required={r['smiles'] for r in exp+val}|{n['smiles'] for c in cases for n in c['negative']}
        status('caching_unconditioned_context',structures=len(required));model.eval()
        with torch.no_grad():
            names=sorted(required)
            for offset in range(0,len(names),512):
                selection=names[offset:offset+512];batch=Batch.from_data_list([cache['graphs'][s] for s in selection]).to(device)
                h=model.unconditioned_embedding(batch).cpu();ptr=batch.ptr.cpu().tolist()
                for j,s in enumerate(selection):cache['graphs'][s].unconditioned_context_embedding=h[ptr[j]:ptr[j+1]].clone()
        prefixes,trainable_names=configure_trainable(model,args.scope)
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=3e-5,weight_decay=1e-4)
        raw=scoring_inputs(base)[-1];exposure=hashlib.sha256();rank_exposure=hashlib.sha256()
        config=dict(args=vars(args),model_args=cfg,init_sha256=sha(args.init),review=audit,quality=quality,pairs=pair_audit,
            scope=args.scope,train_prefixes=list(prefixes),trainable_parameter_names=trainable_names,
            trainable_parameters=sum(p.numel() for p in parameters),full_train_records=full_train_records,
            source_sha256={path.name:sha(path) for path in Path(__file__).parent.glob('*.py')},
            train_queries_sha256=stamp_queries(exp),validation_queries_sha256=stamp_queries(val),rank_weight=.005,anchor_weight=.5,
            rank_batch=8,rank_every_steps=4,extra_targeted_support=False,new_theory_training=False,
            selection=('best spectral objective on fixed1024-tune; initial and epoch8/16/24 retained for train/val retrieval selection' if args.scope=='head'
                       else 'predeclared epoch24 only for train/val scope comparison; best/epoch8/16 saved as archival checkpoints'),
            input_sha256={str(Path(path).resolve()):sha(path) for path in [args.init,args.features,args.bank,Path(args.cohorts)/'manifest.json']})
        (out/'config.json').write_text(json.dumps(config,indent=2));best=float('inf');best_epoch=0
        def save(name,epoch,metrics):
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},args=cfg,epoch=epoch,validation=metrics,
                training_scope=args.scope,train_prefixes=list(prefixes),
                baseline_sha256=saved['baseline_sha256'],prepared_sha256=saved['prepared_sha256']),out/name)
        for epoch in range(args.epochs+1):
            totals=np.zeros(4);steps=0
            if epoch:
                torch.manual_seed(args.seed+epoch);np.random.seed(args.seed+epoch)
                generator=torch.Generator().manual_seed(args.seed+2000+epoch)
                sequence=torch.multinomial(weights,args.samples,replacement=True,generator=generator);exposure.update(sequence.numpy().tobytes())
                loader=DataLoader(dataset,batch_size=64,sampler=sequence.tolist(),num_workers=2,collate_fn=collate,worker_init_fn=seed_worker,
                                  generator=torch.Generator().manual_seed(args.seed+2000+epoch))
                rank_rng=np.random.default_rng(args.seed+4000+epoch);model.train()
                for batch in loader:
                    batch=batch.to(device);optimizer.zero_grad(set_to_none=True)
                    details=model(batch,details=True);parts=reviewed_objective(training_details(details,batch),batch)
                    with torch.no_grad():reference=teacher(batch,details=True)['quantiles']
                    anchor=teacher_anchor(details['quantiles'],reference)
                    rank=parts['loss']*0
                    if steps%4==0:
                        chosen=rank_rng.integers(0,len(cases),size=8);rank_exposure.update(chosen.tobytes())
                        rb,n,stride=make_rank_batch([cases[int(i)] for i in chosen],cache['graphs'],metadata,labels,raw);rb=rb.to(device)
                        visible=training_details(model(rb,details=True),rb)['quantiles']
                        line=torch.tensor([r['positive']['source']=='HMDB' for r in [cases[int(i)] for i in chosen]],device=device).repeat_interleave(stride)
                        distinct=uniform_peaks(visible);visible=torch.where(line[:,None],distinct,visible)
                        costs=observed_nll(visible,rb.q,rb.windows.reshape(-1,8,2),rb.window_mask.reshape(-1,8))
                        rank=pairwise_rank_loss(costs.reshape(n,stride))
                    loss=parts['loss']+.005*int(args.rank)*rank+.5*int(args.anchor)*anchor+parameters[0].sum()*0
                    if not torch.isfinite(loss):raise ValueError('global_forward_nonfinite_loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(parameters,2.);optimizer.step()
                    totals+=[float(loss.detach()),float(parts['loss'].detach()),float(rank.detach()),float(anchor.detach())];steps+=1
                    if steps%100==0:status('training',epoch=epoch,steps=steps,loss=(totals/steps).tolist())
            status('validation',epoch=epoch)
            metrics=validate_spectrum(model,val,cache['graphs'],metadata,labels,device)
            if epoch==0:save('initial.pt',epoch,metrics)
            if metrics['macro_objective']<best:
                best=metrics['macro_objective'];best_epoch=epoch;save('best.pt',epoch,metrics)
            if epoch in (8,16,24):save('epoch_'+str(epoch)+'.pt',epoch,metrics)
            save('last.pt',epoch,metrics)
            record=dict(epoch=epoch,losses=(totals/max(1,steps)).tolist(),validation=metrics,seconds=time.monotonic()-started)
            with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        frozen_tensors=assert_frozen_core(saved['state_dict'],model.state_dict(),prefixes)
        (out/'summary.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_macro=best,checkpoint_sha256=sha(out/'best.pt'),
            forward_exposure_sha256=exposure.hexdigest(),negative_exposure_sha256=rank_exposure.hexdigest(),frozen_core_exact=True,
            scope=args.scope,train_prefixes=list(prefixes),frozen_tensors_checked=frozen_tensors,source_sha256=config['source_sha256'],
            seconds=time.monotonic()-started,external_api_calls=0),indent=2))
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':
    main()

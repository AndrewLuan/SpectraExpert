"""功能概述：固定组解码器先用多源理论谱预训练，再以原实验谱微调的配对消融。
main默认冻结已有GINE/深上下文/条件消息及原发射，只训练同容量equivalence_head；
物理/局部槽和静态校准/物理平均目标形成2×2对照，可再配对有/无新增甲酰胺支持。
先补充未在旧冻结抽样中的认证QM9训练谱；图与原子序号已独立核验。
预训练按理论来源均衡，按独立理论val宏目标选起点；实验阶段按原实验val宏目标
选最终权重，初始未训练模型也参与，新增支持val仅诊断。validate_spectrum使用
共同且与母节点标签解释无关的谱目标，避免把可疑标签重新用于选模。
--site-quality为甲基内部冲突记录提供位点监督降权配对；整谱目标保留。所有源结构分区与侧表SHA
核验，缺图仅用原GINE补编码，缓存只存未条件化上下文。未知分组保留旧谱，损失
加零梯度锚避免全未知批次反传失败。每阶段同seed曝光，保存initial/best/last、历史、
抽样hash和冻结体核对；--source-calibration显式对齐未归属理论源；
--train-messages只解冻新增两层消息模块，旧GINE/深上下文继续逐张量冻结，
缓存仍仅含未条件化表示。--message-mode切换同布局结构/溶剂消息对照；不改源数据、候选、测试或旧结果，无外部服务。
--theory-site-orbits对理论完整标签使用二维轨道无序监督；--theory-selection可
显式使用共同整谱目标选理论起点，避免位点编号本身决定该配对消融的验证选择。
--decoder preserved启用保留原分布的分组残差头；默认absolute保留原36臂语义。
--domain-conditioning为等容量shared/observed域输入对照；默认None不改已有参数布局。
--experimental-theory-replay在每个64实验样本批次配32理论样本，理论来源均衡且
独立seed，默认权重0.1。配置记录精确曝光hash；批次数不一致时拒绝继续，
每阶段验证与旧实验相同，不从测试选择重放权重。
--skip-theory-training提供同初始权重/同实验曝光的直接适配对照：理论阶段只作
epoch0验证与原样检查点保存，不进行理论优化；实验阶段仍走同一循环。配置记录
各阶段epoch数与空理论曝光hash，历史逐epoch记录真实步数，避免把跳过误写成预训练。
--review-artifact/--review-mode显式接入已审计的更正/归属mask/实验谱损失隔离。
所有审核臂共享排除11条观测口径待核val后的前向选模集合；原检索val/test不变。
抽样保持相同，以逐图零权重隔离实验域来源疑点；条件弱目标保持关闭。
已含equivalence_head的预训练权重可继续微调；末尾只核对实际冻结的参数与缓冲。
"""
import argparse,collections,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch
from .emission_model import make_model
from .h_count_train import apply_variants
from .h_count_sidecars import augment_theory_rows
from .h_deep_context import verify_cached_backbone
from .h_replaytrain import loader_for,grouped_objective,validate,ReplaySpectra
from .h_site_quality import apply_quality
from .h_theory_alignment import align_unassigned
from .train import collate
from .observation import observation_loss
from torch.utils.data import DataLoader
from .h_condtrain import sampling_weights
from .h_condmodel import training_details
from .h_metadata import metadata_view,RECOVERED_MODE
from .h_message_support import add_support,extra_weights
from .h_solvent_train import validate as validate_support
from .h_conditions import local_environment
from .data import graph
from .h_observation_study import stamp_queries
from .prepare import sha
from .h_labels import make_key
from .h_reviewed_supervision import MODES as REVIEW_MODES,load_artifact,apply_review,common_validation,reviewed_objective



@torch.inference_mode()
def validate_spectrum(model,rows,graphs,metadata,labels,device):
    model.eval();metrics={}
    for source in sorted({r['source'] for r in rows}):
        rs=[r for r in rows if r['source']==source];loader=DataLoader(ReplaySpectra(rs,graphs,'1H',metadata,annotations=labels,grouped=True),batch_size=128,num_workers=2,collate_fn=collate);total=site=error=0.;n=0
        for b in loader:
            b=b.to(device);d=training_details(model(b,details=True),b);parts=grouped_objective(d,b);total+=float(parts['presence']+.5*parts['shape']+.005*parts['nll'])*b.num_graphs;site+=float(parts['site'])*b.num_graphs;error+=float(observation_loss(d['quantiles'],b).sum())*10;n+=b.num_graphs
        metrics[source]=dict(records=n,objective=total/n,site_objective_diagnostic=site/n,observed_error_ppm=error/n)
    return dict(by_source=metrics,macro_objective=float(np.mean([v['objective'] for v in metrics.values()])),selection='same spectral objective excluding uncertain parent assignment; original observed spectra only')


def main():
    p=argparse.ArgumentParser()
    for k in ['base','prepared','init','sidecar','registry','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--decoder',choices=['absolute','preserved'],default='absolute')
    p.add_argument('--skip-theory-training',action='store_true')
    p.add_argument('--review-artifact');p.add_argument('--review-mode',choices=REVIEW_MODES,default='legacy')
    p.add_argument('--domain-conditioning',choices=['shared','observed']);p.add_argument('--experimental-theory-replay',action='store_true');p.add_argument('--replay-weight',type=float,default=.1)
    p.add_argument('--theory-site-orbits',action='store_true');p.add_argument('--theory-selection',choices=['site','spectral'],default='site')
    p.add_argument('--source-calibration',choices=['sidecar_only','train_affine_all'],default='sidecar_only');p.add_argument('--message-mode',choices=['late','structure','solvent'],default='late');p.add_argument('--train-messages',action='store_true');p.add_argument('--site-quality',choices=['original','methyl_record_mask'],default='original');p.add_argument('--feature-counts',required=True);p.add_argument('--equivalence-mode',choices=['local','physical'],required=True);p.add_argument('--target-mode',choices=['calibrated','physical'],required=True);p.add_argument('--extra-support');p.add_argument('--epochs',type=int,default=16);p.add_argument('--theory-samples',type=int,default=65536);p.add_argument('--experimental-samples',type=int,default=32768);p.add_argument('--seed',type=int,default=193);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,seconds=time.time()-start,**kw)))
    try:
        torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');base=Path(a.base);prepared=Path(a.prepared);status('loading');manifest=json.loads((prepared/'manifest.json').read_text());cache=torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=metadata_view(torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False),RECOVERED_MODE);labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False);saved=torch.load(a.init,map_location='cpu',weights_only=False);owner=base/'hydrogen_analysis/site_supervision/train/best.pt';reference=torch.load(owner,map_location='cpu',weights_only=False)
        if manifest['H_checkpoint_sha256']!=sha(owner) or manifest['cache_sha256']!=saved['prepared_sha256']:raise ValueError('prepared_embedding_manifest_mismatch')
        if a.domain_conditioning and a.decoder!='preserved':raise ValueError('domain_conditioning_requires_preserved_decoder')
        if not np.isfinite(a.replay_weight) or a.replay_weight<0:raise ValueError('invalid_theory_replay_weight')
        verify_cached_backbone(saved['state_dict'],reference['state_dict']);cfg=dict(saved['args'],site_quality_mode=a.site_quality,experimental_selection_mode='spectral_no_site',message_mode=a.message_mode,model='preserved_group_atom' if a.decoder=='preserved' else 'certified_group_atom',count_decode=False,equivalence_mode=a.equivalence_mode,equivalence_registry_path=a.registry,equivalence_registry_sha256=sha(a.registry));
        if a.domain_conditioning:cfg['domain_conditioning']=a.domain_conditioning
        model=make_model(cfg).to(device);missing,extra=model.load_state_dict(saved['state_dict'],strict=False)
        if extra or any(not k.startswith('equivalence_head.') for k in missing):raise ValueError('curriculum_warmstart_mismatch')
        if a.train_messages and a.message_mode=='late':raise ValueError('message_training_requires_active_messages')
        for name,param in model.named_parameters():param.requires_grad_(name.startswith('equivalence_head.') or (a.train_messages and name.startswith('solvent_message_')))
        side=torch.load(a.sidecar,map_location='cpu',weights_only=False);theory,added=augment_theory_rows([r for r in cache['rows'] if r['split']=='train' and r['domain']=='theory'],side,torch.load(a.feature_counts,map_location='cpu',weights_only=False));alignment_audit=dict(mode=a.source_calibration)
        if a.source_calibration=='train_affine_all':theory,alignment_audit=align_unassigned(theory,side['variants'],json.loads((base/'prepared_coverage/domain_shift.json').read_text())['fits'])
        theory,matched=apply_variants(theory,labels,side,a.target_mode);theory_val,_=apply_variants(side['theory_validation'],labels,side,a.target_mode);exp=[r for r in cache['rows'] if r['split']=='train' and r['domain']=='experimental'];val=[r for r in cache['rows'] if r['split']=='val' and r['domain']=='experimental'];ew=sampling_weights(exp,metadata);extra_val=[];support_audit=None
        if a.extra_support:
            support,extra_val,support_audit=add_support(cache,metadata,labels,a.extra_support,manifest,owner,device);ew=extra_weights(ew,support,.05);exp+=support
        review_audit=None;validation_audit=None
        if a.review_mode!='legacy' and not a.review_artifact:raise ValueError('review_mode_requires_artifact')
        if a.review_artifact:
            artifact,review_meta=load_artifact(a.review_artifact)
            if review_meta['input_sha256'][str(base/'hydrogen_analysis/labels_topology/annotations.pt')]!=sha(base/'hydrogen_analysis/labels_topology/annotations.pt'):
                raise ValueError('review_annotations_version_mismatch')
            exp,review_audit=apply_review(labels,exp,artifact,a.review_mode)
            val,validation_audit=common_validation(val,artifact)
            review_audit['artifact_sha256']=review_meta['artifact_sha256']
        quality_audit=apply_quality(labels,exp,a.site_quality)
        if a.smoke:
            theory=[r for r in theory if r['physics'][5]>0][:64]+[r for r in theory if r['source']=='QM9-NMR'][:64]
            if a.review_artifact:
                reviewed=[r for r in exp if make_key(r) in artifact['decisions'] or make_key(r) in artifact['patches']]
                other=[r for r in exp if make_key(r) not in artifact['decisions'] and make_key(r) not in artifact['patches']]
                exp=reviewed[:64]+other[:64]
            else:exp=exp[:128]
            ew=sampling_weights(exp,metadata);val=val[:32];theory_val=theory_val[:32];extra_val=extra_val[:4];a.epochs=1;a.theory_samples=a.experimental_samples=128
        if {r['group'] for r in theory+exp}&{r['group'] for r in val+theory_val+extra_val}:raise ValueError('train_validation_connectivity_overlap')
        required=sorted({r['smiles'] for r in theory+exp+val+theory_val+extra_val});encoder=make_model(reference['args']).to(device).eval();encoder.load_state_dict(reference['state_dict']);missing_graphs=[s for s in required if s not in cache['graphs']];status('encoding_missing_graphs',structures=len(missing_graphs))
        with torch.no_grad():
            for offset in range(0,len(missing_graphs),256):
                names=missing_graphs[offset:offset+256];gs=[graph(s) for s in names];b=Batch.from_data_list(gs).to(device);h=encoder.structure_embedding(b).cpu();ptr=b.ptr.cpu().tolist()
                for j,s in enumerate(names):g=gs[j];g.local_environment=torch.tensor(local_environment(s));g.frozen_embedding=h[ptr[j]:ptr[j+1]].clone();cache['graphs'][s]=g
        del encoder;model.eval();status('caching_frozen_context',structures=len(required))
        with torch.no_grad():
            for offset in range(0,len(required),512):
                names=required[offset:offset+512];b=Batch.from_data_list([cache['graphs'][s] for s in names]).to(device);h=model.unconditioned_embedding(b).cpu();ptr=b.ptr.cpu().tolist()
                for j,s in enumerate(names):cache['graphs'][s].unconditioned_context_embedding=h[ptr[j]:ptr[j+1]].clone()
        sources=collections.Counter(r['source'] for r in theory);tw=torch.tensor([1/sources[r['source']] for r in theory],dtype=torch.float64);parameters=[p for p in model.parameters() if p.requires_grad];config=dict(args=vars(a),model_args=cfg,init_sha256=sha(a.init),prepared_sha256=manifest['cache_sha256'],sidecar_sha256=sha(a.sidecar),matched_theory_targets=matched,source_alignment_audit=alignment_audit,added_QM9_train_records=added,feature_counts_sha256=sha(a.feature_counts),theory_sources=dict(sources),experimental_queries_sha256=stamp_queries(exp),trainable_parameters=sum(p.numel() for p in parameters),support_audit=support_audit,site_quality_audit=quality_audit,sampling_sequence_sha256={},selection='theory val macro selects pretraining; original experimental spectral val macro excluding parent-assignment loss selects fine-tuning including initial control; extra val diagnostic only',source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
        config['review_audit']=review_audit;config['common_validation_audit']=validation_audit
        config['experimental_validation_queries_sha256']=stamp_queries(val)
        config['phase_epochs']=dict(theory=0 if a.skip_theory_training else a.epochs,experimental=a.epochs)
        sampling_specs=[('theory',tw,a.theory_samples,1000),('experimental',ew,a.experimental_samples,2000)]
        if a.experimental_theory_replay:
            if a.experimental_samples%64:raise ValueError('replay_requires_complete_64_sample_experimental_batches')
            sampling_specs.append(('experimental_theory_replay',tw,a.experimental_samples//2,3000))
        config['held_spectrum_exposures_by_epoch']=[]
        for stage,weights,samples,seed_offset in sampling_specs:
            digest=hashlib.sha256()
            for epoch in range(1,config['phase_epochs'].get(stage,a.epochs)+1):
                indices=torch.multinomial(weights,samples,replacement=True,generator=torch.Generator().manual_seed(a.seed+seed_offset+epoch));digest.update(indices.numpy().tobytes())
                if stage=='experimental':config['held_spectrum_exposures_by_epoch'].append(sum(exp[i].get('review_spectral_weight',1.)==0 for i in indices.tolist()))
            config['sampling_sequence_sha256'][stage]=digest.hexdigest()
        (out/'config.json').write_text(json.dumps(config,indent=2))
        def checkpoint(path,stage,epoch,metrics):torch.save(dict(state_dict=model.state_dict(),args=cfg,epoch=epoch,stage=stage,validation=metrics,baseline_sha256=saved['baseline_sha256'],prepared_sha256=saved['prepared_sha256']),path)
        status('initial_validation');initial=validate_spectrum(model,val,cache['graphs'],metadata,labels,device);best_exp=initial['macro_objective'];best_phase='initial';best_epoch=0;checkpoint(out/'initial.pt','initial',0,initial);checkpoint(out/'best.pt','initial',0,initial);best_theory=float('inf')
        for stage,rows,weights,samples,seed_offset,lr in [('theory',theory,tw,a.theory_samples,1000,1e-4),('experimental',exp,ew,a.experimental_samples,2000,3e-5)]:
            if stage=='experimental':model.load_state_dict(torch.load(out/'pretrain_best.pt',map_location=device,weights_only=False)['state_dict'])
            opt=torch.optim.AdamW(parameters,lr=lr,weight_decay=1e-4)
            for epoch in range(config['phase_epochs'][stage]+1):
                total=primary_total=replay_total=0.;steps=replay_steps=0
                if epoch:
                    model.train();status(stage+'_training',epoch=epoch)
                    replay=iter(loader_for(theory,cache['graphs'],metadata,labels,tw,a.seed+3000+epoch,a.experimental_samples//2,32,False,True,theory_orbits=a.theory_site_orbits)) if stage=='experimental' and a.experimental_theory_replay else None
                    for b in loader_for(rows,cache['graphs'],metadata,labels,weights,a.seed+seed_offset+epoch,samples,64,False,True,theory_orbits=a.theory_site_orbits and stage=='theory'):
                        b=b.to(device);opt.zero_grad(set_to_none=True);d=model(b,details=True);objective=reviewed_objective if a.review_artifact else grouped_objective;primary=objective(training_details(d,b),b)['loss'];loss=primary+parameters[0].sum()*0;primary_total+=float(primary.detach())
                        if replay is not None:
                            tb=next(replay,None)
                            if tb is None:raise ValueError('theoretical_replay_ended_before_experimental_batch')
                            tb=tb.to(device);td=model(tb,details=True);auxiliary=grouped_objective(training_details(td,tb),tb)['loss'];loss=loss+a.replay_weight*auxiliary;replay_total+=float(auxiliary.detach());replay_steps+=1
                        if not torch.isfinite(loss):raise ValueError('nonfinite_curriculum_loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,2.);opt.step();steps+=1;total+=float(loss.detach())
                        if steps%200==0:status(stage+'_training',epoch=epoch,steps=steps,loss=total/steps)
                    if replay is not None and (replay_steps!=steps or next(replay,None) is not None):raise ValueError('theoretical_replay_exposure_mismatch')
                status(stage+'_validation',epoch=epoch);theory_validator=validate_spectrum if a.theory_selection=='spectral' else validate;metrics=theory_validator(model,theory_val,cache['graphs'],metadata,labels,device) if stage=='theory' else validate_spectrum(model,val,cache['graphs'],metadata,labels,device);row=dict(stage=stage,epoch=epoch,training_loss=total/max(1,steps),primary_steps=steps,training_samples=steps*64,validation=metrics,seconds=time.time()-start)
                if a.experimental_theory_replay:row.update(primary_training_loss=primary_total/max(1,steps),theoretical_replay_loss=replay_total/max(1,replay_steps),theoretical_replay_steps=replay_steps,primary_steps=steps)
                if stage=='theory' and metrics['macro_objective']<best_theory:best_theory=metrics['macro_objective'];checkpoint(out/'pretrain_best.pt',stage,epoch,metrics)
                if stage=='experimental' and metrics['macro_objective']<best_exp:best_exp=metrics['macro_objective'];best_phase=stage;best_epoch=epoch;checkpoint(out/'best.pt',stage,epoch,metrics)
                if stage=='experimental' and extra_val and epoch%2==0:row['extra_validation_diagnostic']=validate_support(model,extra_val,cache['graphs'],metadata,labels,device)
                checkpoint(out/'last.pt',stage,epoch,metrics)
                with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        final=model.state_dict()
        trainable_names={k for k,p in model.named_parameters() if p.requires_grad}
        if any(not torch.equal(v,final[k].cpu()) for k,v in saved['state_dict'].items() if k not in trainable_names):raise ValueError('frozen_curriculum_body_changed')
        report=dict(best_phase=best_phase,best_epoch=best_epoch,best_experimental_macro=best_exp,best_theory_macro=best_theory,checkpoint_sha256=sha(out/'best.pt'),frozen_body_exact=not a.train_messages,frozen_core_exact=True,train_messages=a.train_messages,seconds=time.time()-start);(out/'complete.json').write_text(json.dumps(report));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

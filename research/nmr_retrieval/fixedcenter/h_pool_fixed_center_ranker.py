"""功能概述：唯一追加mixed50:50/listwise排序器，训练/推理共享固定train-only中心。
candidate_qualification从train缓存自带的bank候选SMILES顺序派生connectivity groups，
核negative mask与heldout契约；eligible_positions只取自然512、~mask及非internal候选，
从不通过target_position选择统计候选。estimate_centers对原role的非internal query-variant
逐行计算single/dual合格候选中位数，池各.5再query-row等权平均，空集合失败。
FixedCenterRanker仅把旧row_delta的两个query-pool median替换成固定33+3 fp32 buffer；
其余输入、scale、clamp、network、O、alpha/gamma不变，forward不读取source/ID/标签。
fit完全保留旧group→variant/pool曝光、listwise损失、64×8192预算、优化器及internal
alpha选择；centers与估计输入SHA随initial/best/last保存。evaluate只开放train/val验证阶段。
本模块由独立D工具目录加载唯一科学checkout，复用h_pool_ranker及h_research_ranker接口，
不改旧pin源码、不读cohorts.pt/heldout谱值、不训练O、不添加自然候选或谱模型forward。
main独占新目录并核始末输入/源码SHA与真实EXIT；正式执行由已审计划和诊断gate控制。
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

for _root in (Path('/Users/andrewluan/Desktop/Projects/IRfingerprints/SpectrExpert'),
              Path('/home/user/IRfingerprints/SpectrExpert')):
    if (_root/'nmr_transfer/h_pool_ranker.py').is_file():
        sys.path.insert(0,str(_root));break
else:
    raise FileNotFoundError('unique_scientific_checkout_missing')
from nmr_transfer import h_pool_ranker as base
from nmr_transfer import data as identity_source
from nmr_transfer.h_research_ranker import ResearchRanker,internal_group,read_feature_cache,select_rows,validation,result_row
from nmr_transfer.h_research_contract import atomic_json,input_fingerprints,rank_shortlist,summarize_predictions

SCHEMA='nmr_pool_fixed_center_20261023'
CANDIDATE_COUNT=114982
CANDIDATE_SHA='72b72140c09ae14ae8d0aa4edad624a151093c9df246b2052de8a3a4ea616bad'
BUDGET=dict(epochs=64,samples=8192,batch_size=32,anchor=.1,lr=.001)
CENTER_BUFFERS={'fixed_x_center','fixed_o_center'}


def require(condition,message):
    if not condition:raise ValueError(message)


def ordered_sha(values):
    return hashlib.sha256('\n'.join(values).encode()).hexdigest()


def source_paths():
    return [Path(__file__).resolve()]+base.source_paths()+[Path(identity_source.__file__).resolve()]


def candidate_qualification(single,dual,contract,identity_fn=None):
    """只读取已存在train缓存的候选身份及JSON历史heldout契约，不打开bank/test/cohorts.pt。"""
    smiles=single['smiles']
    require(smiles==dual['smiles'] and len(smiles)==CANDIDATE_COUNT and len(set(smiles))==len(smiles),
            'fixed_candidate_identity_changed')
    order_sha=ordered_sha(smiles)
    require(order_sha==CANDIDATE_SHA and all(d['manifest']['candidate_order_sha256']==order_sha
            for d in (single,dual)) and contract['candidate_order_sha256']==order_sha,'candidate_order_SHA_mismatch')
    heldout=contract['training_excluded_groups']
    require(isinstance(heldout,list) and bool(heldout) and len(heldout)==len(set(heldout)),'heldout_contract_required')
    heldout=set(heldout);excluded_sha=ordered_sha(sorted(heldout))
    require(all(d['manifest']['negative_excluded_structure_sha256']==excluded_sha for d in (single,dual)),
            'negative_exclusion_contract_SHA_mismatch')
    fn=identity_fn or (lambda s:identity_source.identity(s)[1])
    groups=[fn(s) for s in smiles]
    require(all(isinstance(g,str) and bool(g) for g in groups),'candidate_group_mapping_invalid')
    internal=torch.tensor([internal_group(g) for g in groups],dtype=torch.bool)
    return dict(groups=groups,internal=internal,heldout=heldout,metadata=dict(candidate_order_sha256=order_sha,
        derived_candidate_groups_sha256=ordered_sha(groups),candidate_count=len(smiles),
        candidate_connectivity_groups=len(set(groups)),historical_heldout_sha256=excluded_sha,
        historical_heldout_groups=len(heldout),candidate_internal_groups=int(internal.sum()),
        identity_source_path=str(Path(identity_source.__file__).resolve())))


def eligible_positions(row,qualification):
    n=row['evaluation_pool_size'];indices=row['candidate_indices']
    require(type(n) is int and n==512 and len(indices) in (512,513),'center_requires_natural512')
    require(len(indices.unique())==len(indices) and indices.dtype in (torch.int32,torch.int64)
            and bool(((indices>=0)&(indices<len(qualification['groups']))).all()),'center_invalid_candidate_indices')
    mask=row['negative_exclusion_mask']
    require(mask.dtype==torch.bool and len(mask)==len(indices),'center_invalid_negative_mask')
    group=row['group']
    require(group not in qualification['heldout'] and not internal_group(group),'center_query_not_fit')
    expected=torch.tensor([qualification['groups'][int(i)]==group or
                           qualification['groups'][int(i)] in qualification['heldout'] for i in indices],dtype=torch.bool)
    require(torch.equal(mask.cpu(),expected),'center_negative_mask_semantics_mismatch')
    allowed=(~mask[:n].cpu()) & (~qualification['internal'][indices[:n].cpu()])
    positions=allowed.nonzero().flatten()
    require(len(positions)>0,'center_empty_eligible_natural_pool')
    return positions


def raw_O(model,row,positions,device):
    sigma=row.get('per_quantile_sigma')
    def selected(name):return row[name][positions].to(device)
    values=model.observation_model(selected('predicted_ppm'),selected('tags').long(),row['observation_input'],
        H=selected('H'),K=selected('K'),per_quantile_sigma=None if sigma is None else sigma[positions].to(device))
    require(values.shape==(len(positions),3) and torch.isfinite(values).all(),'center_nonfinite_raw_O')
    return values


@torch.inference_mode()
def estimate_centers(model,rows,dual_lookup,qualification,device):
    require(rows and all(not internal_group(r['group']) for r in rows),'center_fit_rows_required')
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    xsum=torch.zeros(33,dtype=torch.float64);osum=torch.zeros(3,dtype=torch.float64)
    audit=[]
    for row in sorted(rows,key=base.row_key):
        other=dual_lookup[base.row_key(row)]
        entry=dict(query_key=list(base.row_key(row)),pool_weight=.5,query_row_weight=1/len(rows),pools={})
        for pool,selected in (('single',row),('dual',other)):
            positions=eligible_positions(selected,qualification)
            x=selected['features'][positions]
            require(x.shape==(len(positions),33) and torch.isfinite(x).all(),'center_nonfinite_raw_X')
            o=raw_O(model,selected,positions,device)
            xsum += .5*x.median(0).values.cpu().double()
            osum += .5*o.median(0).values.cpu().double()
            ids=selected['candidate_indices'][positions].tolist()
            entry['pools'][pool]=dict(eligible_candidates=len(positions),
                eligible_candidate_order_sha256=ordered_sha([str(i) for i in ids]),
                appended_positive_excluded=len(selected['candidate_indices'])>512)
        audit.append(entry)
    xcenter=(xsum/len(rows)).float();ocenter=(osum/len(rows)).float()
    require(torch.isfinite(xcenter).all() and torch.isfinite(ocenter).all(),'center_nonfinite_constants')
    require(base.same_value(before,{k:v.detach().cpu() for k,v in model.state_dict().items()}),'center_estimator_changed_model')
    return xcenter,ocenter,dict(query_rows=len(rows),fit_groups=len({r['group'] for r in rows}),
        weighting='query-variant rows equal; single/dual each0.5; independent of unchanged SGD group->variant draws',
        median='torch lower median of eligible natural candidates; no fallback/skip',row_audit=audit,
        qualification=qualification['metadata'],model_unchanged=True)


class FixedCenterRanker(ResearchRanker):
    def __init__(self,initial_sigmas,xcenter,ocenter):
        super().__init__(True,initial_sigmas)
        x=torch.as_tensor(xcenter,dtype=torch.float32);o=torch.as_tensor(ocenter,dtype=torch.float32)
        require(x.shape==(33,) and o.shape==(3,) and torch.isfinite(x).all() and torch.isfinite(o).all(),
                'fixed_center_dimensions_or_finite')
        self.register_buffer('fixed_x_center',x.detach().clone())
        self.register_buffer('fixed_o_center',o.detach().clone())
        for p in self.observation_model.parameters():p.requires_grad_(False)

    def row_delta(self,row,scale,device):
        x=row['features'].to(device)
        z=((x-self.fixed_x_center)/scale).clamp(-20,20)
        sigma=row.get('per_quantile_sigma')
        extra=self.observation_model(row['predicted_ppm'].to(device),row['tags'].to(device).long(),
            row['observation_input'],H=row['H'].to(device),K=row['K'].to(device),
            per_quantile_sigma=None if sigma is None else sigma.to(device))
        extra=(extra-self.fixed_o_center).clamp(-20,20)
        logits=self.network(torch.cat((z,extra),1))[:,int(row['kind'])]
        return 5*torch.tanh(logits/5)


def inherit_model(initial_model,initial_saved,xcenter,ocenter,device):
    model=FixedCenterRanker(initial_saved['initial_sigmas'],xcenter,ocenter).to(device)
    result=model.load_state_dict(initial_model.state_dict(),strict=False)
    require(set(result.missing_keys)==CENTER_BUFFERS and not result.unexpected_keys,'fixed_initial_state_keys')
    inherited={k:v for k,v in model.state_dict().items() if k not in CENTER_BUFFERS}
    require(base.same_value({k:v.cpu() for k,v in inherited.items()},
                           {k:v.cpu() for k,v in initial_model.state_dict().items()}),'fixed_initial_tensors_changed')
    model.eval()
    return model


def load_checkpoint(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    require(saved['schema']==SCHEMA,'fixed_checkpoint_schema')
    state=saved['state_dict']
    model=FixedCenterRanker(saved['initial_sigmas'],state['fixed_x_center'],state['fixed_o_center']).to(device)
    model.load_state_dict(state,strict=True);model.eval()
    require(saved['center_estimate_sha256'] and saved['scale'].shape==(33,)
            and torch.isfinite(saved['scale']).all() and (saved['scale']>0).all(),'fixed_checkpoint_metadata')
    parent=Path(path).resolve().parent
    estimate=json.loads((parent/'centers.json').read_text())
    require(input_fingerprints([parent/'centers.json'])[str(parent/'centers.json')]==saved['center_estimate_sha256'],
            'fixed_center_estimate_SHA_mismatch')
    require(torch.equal(model.fixed_x_center.cpu(),torch.tensor(estimate['center33'],dtype=torch.float32))
            and torch.equal(model.fixed_o_center.cpu(),torch.tensor(estimate['O_center3'],dtype=torch.float32)),
            'fixed_center_checkpoint_constants_mismatch')
    require(input_fingerprints([parent/'center_rows.jsonl'])[str(parent/'center_rows.jsonl')]==estimate['row_audit_sha256'],
            'fixed_center_rows_SHA_mismatch')
    producer=json.loads((parent/'summary.json').read_text())
    require((parent/'EXIT_CODE').read_text().strip()=='0','fixed_checkpoint_producer_exit')
    require(producer['center_estimate_sha256']==saved['center_estimate_sha256'],'fixed_checkpoint_producer_center_SHA')
    return model,saved['scale'].to(device),saved


def save_checkpoint(path,model,scale,args,data,epoch,alpha,exposure,pools,pins,center_sha):
    torch.save(dict(schema=SCHEMA,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        scale=scale.detach().cpu(),observation=True,initial_sigmas=model.observation_model.initial_sigmas.cpu().tolist(),
        epoch=epoch,alpha=alpha,args=vars(args),exposure_sha256=exposure,pool_exposure_sha256=pools,
        train_sha256=pins[str(Path(args.train).resolve())],dual_train_sha256=pins[str(Path(args.dual_train).resolve())],
        initialization_sha256=pins[str(Path(args.initial_checkpoint).resolve())],center_estimate_sha256=center_sha,
        cache_protocol={k:data['manifest'][k] for k in base.IDENTITY},
        source_inputs_sha256=data['manifest']['source_inputs_sha256'],ranker_source_sha256=input_fingerprints(source_paths())),path)


def validate_contract_producer(args,data,pins):
    path=Path(args.cohort_contract).resolve()
    contract=json.loads(path.read_text());producer=json.loads((path.parent/'manifest.json').read_text())
    require((path.parent/'EXIT_CODE').read_text().strip()=='0','cohort_contract_producer_exit')
    require(producer['contract_sha256']==pins[str(path)],'cohort_contract_producer_SHA')
    payload=str(path.parent/'cohorts.pt')
    require(contract['cohorts_sha256']==producer['cohorts_sha256']
            and data['manifest']['source_inputs_sha256'].get(payload)==contract['cohorts_sha256'],
            'cohort_contract_training_payload_binding')
    return contract


def validate_fit_budget(args):
    require(args.seed in (193,211) and args.training_pool=='mixed' and args.loss=='listwise','only_registered_fixed_arm')
    require(all(getattr(args,k)==v for k,v in BUDGET.items()),'fixed_arm_full_budget_required')


def fit(args,out,device,status,pins):
    validate_fit_budget(args)
    single,_=read_feature_cache(args.train);dual,_=read_feature_cache(args.dual_train)
    dual_lookup=base.validate_caches(single,dual)
    contract=validate_contract_producer(args,single,pins)
    qualification=candidate_qualification(single,dual,contract)
    rows=select_rows(single,args.augment)
    require(all(r['group'] not in qualification['heldout'] for r in rows),'training_query_heldout_overlap')
    train=[r for r in rows if not internal_group(r['group'])]
    valid=[r for r in rows if internal_group(r['group'])]
    require(train and valid,'fixed_empty_internal_split')
    groups=collections.defaultdict(list)
    for row in train:groups[row['group']].append(row)
    names=sorted(groups)
    torch.manual_seed(args.seed)
    initial_model,scale,initial=base.load_checkpoint(args.initial_checkpoint,device)
    base.check_science(initial,single)
    producer=json.loads((Path(args.initial_checkpoint).parent/'summary.json').read_text())
    require(producer['best']['epoch']==initial['epoch'] and producer['best']['alpha']==initial['alpha']
            and producer['training_cache_sha256']==initial['train_sha256'],'fixed_initial_producer_binding')
    require(all(torch.isfinite(v).all() for v in initial_model.state_dict().values()),'fixed_initial_nonfinite')
    require(initial['schema']==base.OLD_SCHEMA and initial['args']['seed']==args.seed
            and initial['args']['augment'] is args.augment and initial['cache_protocol']['recall']=='single',
            'fixed_initial_checkpoint_role')
    require(initial['train_sha256']==pins[str(Path(args.train).resolve())],'fixed_initial_training_SHA')
    require(scale.shape==(33,) and torch.isfinite(scale).all() and (scale>0).all(),'fixed_initial_scale')
    xcenter,ocenter,estimate=estimate_centers(initial_model,train,dual_lookup,qualification,device)
    estimate.update(schema=SCHEMA,seed=args.seed,augment=args.augment,center33=xcenter.tolist(),O_center3=ocenter.tolist(),
                    input_sha256=pins,constant_fp32=36,raw_constant_bytes=144,selection_scope='fit_train_only')
    row_audit=estimate.pop('row_audit')
    with (out/'center_rows.jsonl').open('x') as stream:
        for item in row_audit:stream.write(json.dumps(item)+'\n')
    estimate['row_audit_sha256']=input_fingerprints([out/'center_rows.jsonl'])[str((out/'center_rows.jsonl').resolve())]
    atomic_json(out/'centers.json',estimate)
    center_sha=input_fingerprints([out/'centers.json'])[str((out/'centers.json').resolve())]
    model=inherit_model(initial_model,initial,xcenter,ocenter,device)
    del initial_model
    optimizer=torch.optim.AdamW(model.network.parameters(),lr=args.lr,weight_decay=.001)
    initial_o={k:v.detach().cpu().clone() for k,v in model.observation_model.state_dict().items()}
    frozen_centers={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in CENTER_BUFFERS}
    initial_scale=scale.detach().cpu().clone()
    exposure,pools=hashlib.sha256(),hashlib.sha256();alphas=[0.,.25,.5,1.]
    best,chosen=-float('inf'),None;started=time.monotonic()
    for epoch in range(args.epochs+1):
        total,steps=0.,0
        if epoch:
            model.train();model.observation_model.eval()
            schedule=base.training_schedule(names,groups,args.seed,epoch,args.samples)
            for begin in range(0,len(schedule),args.batch_size):
                optimizer.zero_grad(set_to_none=True);losses=[]
                for row,pool in schedule[begin:begin+args.batch_size]:
                    exposure.update(json.dumps(base.row_key(row),separators=(',',':')).encode()+b'\n')
                    pools.update(bytes([int(pool==1)]))
                    selected=dual_lookup[base.row_key(row)] if pool==1 else row
                    delta=model.row_delta(selected,scale,device)
                    losses.append(base.pool_loss(selected,delta,scale,args.anchor,'listwise'))
                loss=torch.stack(losses).mean()
                require(torch.isfinite(loss),'fixed_nonfinite_training_loss')
                loss.backward();nn.utils.clip_grad_norm_(model.network.parameters(),2.);optimizer.step()
                total+=float(loss.detach());steps+=1
        if epoch==0 or epoch%8==0 or epoch==args.epochs:
            _,reports=validation(model,valid,scale,device,alphas)
            alpha=max(alphas,key=lambda a:(reports[str(a)]['overall']['MRR'],-a))
            signal=reports[str(alpha)]['overall']['MRR']
            if epoch==0:
                save_checkpoint(out/'initial.pt',model,scale,args,single,0,initial['alpha'],exposure.hexdigest(),pools.hexdigest(),pins,center_sha)
            if signal>best+1e-5:
                best=signal;chosen=dict(epoch=epoch,alpha=alpha,MRR=signal)
                save_checkpoint(out/'best.pt',model,scale,args,single,epoch,alpha,exposure.hexdigest(),pools.hexdigest(),pins,center_sha)
            record=dict(epoch=epoch,training_loss=total/max(steps,1),internal=reports,best=chosen,seconds=time.monotonic()-started)
            with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            status('fit',**record);print(json.dumps({k:record[k] for k in ('epoch','best','seconds')}),flush=True)
    save_checkpoint(out/'last.pt',model,scale,args,single,args.epochs,chosen['alpha'],exposure.hexdigest(),pools.hexdigest(),pins,center_sha)
    changes={k:float((v.detach().cpu()-initial_o[k]).abs().max()) for k,v in model.observation_model.state_dict().items()}
    require(not any(changes.values()),'fixed_observation_changed')
    require(base.same_value(frozen_centers,{k:v.detach().cpu() for k,v in model.state_dict().items() if k in CENTER_BUFFERS}),'fixed_centers_changed')
    require(torch.equal(initial_scale,scale.detach().cpu()),'fixed_scale_changed')
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()),'fixed_nonfinite_final_model')
    require(input_fingerprints([out/'centers.json'])[str((out/'centers.json').resolve())]==center_sha,'fixed_center_artifact_changed')
    atomic_json(out/'summary.json',dict(best=chosen,training_groups=len(groups),internal_groups=len({r['group'] for r in valid}),
        exposure_sha256=exposure.hexdigest(),pool_exposure_sha256=pools.hexdigest(),observation_parameter_max_changes=changes,
        samples_per_epoch=args.samples,epochs=args.epochs,training_pool='mixed',loss='listwise',input_sha256=pins,
        center_estimate_sha256=center_sha,center_rows_sha256=estimate['row_audit_sha256'],
        centers_frozen=True,scale_frozen=True,initial_tensors_inherited_exact=True,
        center_query_rows=len(train),constant_fp32=36,raw_constant_bytes=144,elapsed_seconds=time.monotonic()-started,external_api_calls=0))


@torch.inference_mode()
def evaluate(args,out,device,status,pins):
    data,_=read_feature_cache(args.cache)
    require(data['manifest']['cohort'] in ('original_val','tune'),'fixed_validation_only_no_test')
    models=[load_checkpoint(p,device) for p in (args.no_aug_checkpoint,args.aug_checkpoint)]
    for role,(model,scale,saved) in zip((False,True),models):
        require(saved['args']['seed']==args.seed and saved['args']['augment'] is role,'fixed_eval_model_role')
        base.check_science(saved,data)
        require(all(saved['args'][k]==v for k,v in BUDGET.items())
                and saved['args']['training_pool']=='mixed' and saved['args']['loss']=='listwise','fixed_eval_budget')
        require(torch.isfinite(scale).all() and all(torch.isfinite(v).all() for v in model.state_dict().values()),'fixed_eval_nonfinite')
    require(models[0][2]['train_sha256']==models[1][2]['train_sha256']
            and models[0][2]['dual_train_sha256']==models[1][2]['dual_train_sha256'],'fixed_eval_training_pair_mismatch')
    rows=[];durations=[];started=time.monotonic()
    for i,row in enumerate(data['rows']):
        base.validate_row(row,False);begin=time.monotonic()
        correction=[saved['alpha']*model.row_delta(row,scale,device)*scale[0] for model,scale,saved in models]
        obs=row['query_observation'];gamma=.75 if obs.get('windows') or obs.get('partial_peaks') else .25
        costs=row['coarse_scores'].to(device)+(1-gamma)*correction[0]+gamma*correction[1]
        rank,top=rank_shortlist(row,costs);durations.append(time.monotonic()-begin)
        rows.append(dict(result_row(row,rank,top),gamma=gamma,mask_content_sha256=row['mask_content_sha256'],
            natural_candidate_indices=row['candidate_indices'][:512].tolist(),natural_candidate_costs=costs[:512].cpu().tolist()))
        if (i+1)%100==0:status('evaluating',records=i+1,total=len(data['rows']))
    with (out/'predictions.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    atomic_json(out/'summary.json',dict(metrics=summarize_predictions(rows),cohort=data['manifest']['cohort'],
        recall=data['manifest']['recall'],query_sha256=data['manifest']['query_sha256'],
        candidate_order_sha256=data['manifest']['candidate_order_sha256'],input_sha256=pins,
        checkpoints=input_fingerprints([args.no_aug_checkpoint,args.aug_checkpoint]),
        center_estimate_sha256=[s['center_estimate_sha256'] for _,_,s in models],
        reranking_seconds_mean=float(np.mean(durations)),reranking_seconds_p95=float(np.quantile(durations,.95)),
        elapsed_seconds=time.monotonic()-started,external_api_calls=0))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--stage',choices=('fit','evaluate'),required=True)
    parser.add_argument('--out',required=True)
    for name in ('train','dual-train','initial-checkpoint','cohort-contract','cache','no-aug-checkpoint','aug-checkpoint'):
        parser.add_argument('--'+name)
    parser.add_argument('--seed',type=int,choices=(193,211),required=True);parser.add_argument('--augment',action='store_true')
    parser.add_argument('--training-pool',choices=('mixed',),default='mixed');parser.add_argument('--loss',choices=('listwise',),default='listwise')
    for name,value in BUDGET.items():parser.add_argument('--'+name.replace('_','-'),type=int if isinstance(value,int) else float,default=value)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    require(Path(args.source).resolve()==Path(base.__file__).resolve().parent.parent,'wrong_scientific_source')
    paths=source_paths()
    caches=[args.train,args.dual_train] if args.stage=='fit' else [args.cache]
    models=[args.initial_checkpoint] if args.stage=='fit' else [args.no_aug_checkpoint,args.aug_checkpoint]
    paths += caches+models
    for path in models+caches:
        require(path is not None,'required_input_missing')
        parent=Path(path).resolve().parent
        require((parent/'EXIT_CODE').read_text().strip()=='0','fixed_input_producer_not_exit0')
        paths += [parent/'summary.json',parent/'EXIT_CODE']
        if args.stage=='evaluate' and path in models:
            paths += [parent/'centers.json',parent/'center_rows.jsonl']
    if args.stage=='fit':
        require(args.cohort_contract is not None,'cohort_contract_required')
        paths += [args.cohort_contract,Path(args.cohort_contract).parent/'manifest.json',Path(args.cohort_contract).parent/'EXIT_CODE']
    pins=input_fingerprints(paths)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    status=lambda stage,**kw:atomic_json(out/'status.json',dict(stage=stage,**kw))
    try:
        torch.set_num_threads(4);atomic_json(out/'protocol.json',dict(args=vars(args),inputs_sha256=pins))
        if args.stage=='fit':fit(args,out,torch.device(args.device),status,pins)
        else:evaluate(args,out,torch.device(args.device),status,pins)
        require(input_fingerprints(pins)==pins,'fixed_input_changed_during_run')
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except BaseException as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

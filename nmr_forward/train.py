"""功能概述：训练纯二维氢位移分布预测，使用实验积分峰表而非伪造采样谱线。
主要职责：collate 对图做padding；pred_quantiles 将无序氢位移转为等权质子分位数；
exact_w1精确计算两离散质量分布间W1；score计算记录平均误差；main只据验证集选择模型；evaluate_test单独
执行锁定模型的测试，保留所有测试记录和独立质量标记，高误差仅列为复查候选。
新增契约：--init-from 仅恢复架构兼容权重；--validation-cache 固定验证/测试身份并拒绝分组泄漏；
--resume 严格核对缓存、源码与配置，恢复 epoch 边界 last.pt 及随机状态，已完成运行不可续训。
filehash/resolve_splits/warmstart_model/build_contract/reconcile_history 分别负责身份、划分、
权重初始化、恢复契约与已提交历史；verify_committed_best 拒绝超过 last 的最佳权重；
辅助指标在有效标签数为零时返回 None；evaluate_test 在加载模型前核对注册测试缓存 SHA。
关系：读取 prepare.py cache.pt，调用 model.NMRForwardModel 和 training_state，保存 checkpoint、
逐epoch history、validation.json。训练不会访问test标签；--evaluate-test显式解锁评估。
默认每个氢质量1；--visibility仅允许N/O/S上H减少报告质量，并提供计数约束及预热冻结。
score同时报告总积分误差、监督覆盖和联合位移/多重性/J指标；不是原子归属MAE或完整谱误差。
"""
import argparse, json, pathlib, time, random, hashlib, os, collections, shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import NMRForwardModel
from visibility import VisibilityNMRModel, visibility_loss
from peak_aux import peak_aux_loss
from multiplets import MULTIPLICITY_CLASSES
from training_state import save_training_state, restore_training_state


def collate(rows):
    b=len(rows); n=max(len(r['x']) for r in rows)
    x=torch.zeros(b,n,rows[0]['x'].shape[1]);adj=torch.zeros(b,n,n,5);mask=torch.zeros(b,n,dtype=torch.bool);h=torch.zeros(b,n,dtype=torch.long)
    for i,r in enumerate(rows):
        k=len(r['x']);x[i,:k]=torch.from_numpy(r['x']);adj[i,:k,:k]=torch.from_numpy(r['adj']);mask[i,:k]=True;h[i,:k]=torch.from_numpy(r['hcount'])
    peaks=max(len(r['xs']) for r in rows);tx=torch.zeros(b,peaks);tw=torch.zeros(b,peaks)
    for i,r in enumerate(rows):
        tx[i,:len(r['xs'])]=torch.tensor(r['xs']);tw[i,:len(r['xs'])]=torch.tensor(r['weights'])
    return dict(x=x,adj=adj,mask=mask,hcount=h,cond=torch.tensor(np.stack([r['cond'] for r in rows])),target=torch.tensor(np.stack([r['target'] for r in rows])),target_x=tx,target_w=tw,rows=rows)


def exact_w1(out,batch):
    """积分峰表与质子点集的精确W1；可选proton_weights，padding质量为零。"""
    pred=out['shifts'].flatten(1); pm=out['proton_mask'].flatten(1)
    tx=batch['target_x'];tw=batch['target_w'].double()
    pw=out.get('proton_weights',out['proton_mask']).flatten(1).double()*pm
    pw=pw/pw.sum(1,keepdim=True).clamp_min(1e-12)
    tw=tw/tw.sum(1,keepdim=True)
    support=torch.cat([pred,tx],1).double()
    masses=torch.cat([pw,-tw],1)
    # 零质量padding搬到最右有效点，避免虚假尾区间和浮点累计残差放大。
    active=torch.cat([pm,tw>0],1)
    right=support.masked_fill(~active,-1e6).max(1,keepdim=True).values.detach()
    support=torch.where(active,support,right)
    ordered,idx=support.sort(1);cdf=masses.gather(1,idx).cumsum(1)
    return ((ordered[:,1:]-ordered[:,:-1])*cdf[:,:-1].abs()).sum(1).float()


def pred_quantiles(out):
    shifts=out['shifts'].flatten(1);mask=out['proton_mask'].flatten(1)
    ordered=shifts.masked_fill(~mask,1e6).sort(dim=1).values
    counts=mask.sum(1)
    q=(torch.arange(128,device=shifts.device)+.5)/128
    idx=(counts[:,None]*q).long().clamp_max(ordered.shape[1]-1)
    return ordered.gather(1,idx)


def move(batch,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


@torch.no_grad()
def score(model,loader,device,details=False):
    model.eval();errors=[];rows=[];aux_counts=collections.Counter();coverage=[];precision=[];integral_errors=[];visibility_counts=collections.Counter();strata=collections.defaultdict(list)
    for batch in loader:
        batch=move(batch,device);out=model(batch);err=exact_w1(out,batch)
        batch_errors=err.cpu().tolist();errors.extend(batch_errors)
        predicted_mass=out.get('proton_weights',out['proton_mask']).flatten(1).sum(1)
        integral_errors.extend((predicted_mass-batch['target_w'].sum(1)).abs().cpu().tolist())
        if 'proton_weights' in out:
            visibility_counts.update(visibility_loss(out,batch)['counts'])
        for row,error in zip(batch['rows'],batch_errors):
            strata['source:'+str(row.get('source'))].append(error)
            strata['H_integral_mismatch' if 'integral_vs_structure_H_mismatch' in row['flags'] else 'H_integral_match'].append(error)
        sites=out['shifts'].flatten(1);valid=out['proton_mask'].flatten(1)
        distance=(sites[:,:,None]-batch['target_x'][:,None,:]).abs()
        distance=distance.masked_fill(~valid[:,:,None],float('inf')).masked_fill(batch['target_w'][:,None,:]<=0,float('inf'))
        target_hit=(distance.min(1).values<=.1).float()
        coverage.extend(((target_hit*batch['target_w']).sum(1)/batch['target_w'].sum(1)).cpu().tolist())
        precision.extend(((distance.min(2).values<=.1).sum(1)/valid.sum(1)).cpu().tolist())
        if 'mult_logits' in out:
            aux=peak_aux_loss(out,batch)
            for key,value in aux['counts'].items():aux_counts[key]+=value
            aux_counts['multiplicity_correct']+=float(aux['multiplicity_correct'])
            aux_counts['j_abs_error_sum']+=float(aux['j_abs_error_sum'])
            for key in ('joint_multiplicity_correct_at_0_1ppm','joint_J_abs_error_sum','joint_J_values'):
                if key in aux:aux_counts[key]+=float(aux[key])
        if details:
            for i,r in enumerate(batch['rows']):
                rows.append(dict(record_id=r['record_id'],smiles=r['smiles'],group=r['group'],flags=r['flags'],source=r['source'],source_locator=r['source_locator'],
                    error_w1_ppm=float(err[i]),predicted_shifts=out['shifts'][i][out['proton_mask'][i]].cpu().tolist(),
                    predicted_reported_weights=out.get('proton_weights',out['proton_mask'])[i][out['proton_mask'][i]].float().cpu().tolist(),
                    target_centers=r['xs'],target_integrals=r['weights'],structure_protons=r['protons'],reported_integral=r['integral']))
    e=np.array(errors)
    metrics={'n':len(e),'integral_weighted_W1_ppm':float(e.mean()),'median_W1_ppm':float(np.median(e)),
        'p90_W1_ppm':float(np.quantile(e,.9)),'p99_W1_ppm':float(np.quantile(e,.99)),
        'target_integral_coverage_at_0.1ppm':float(np.mean(coverage)),
        'predicted_proton_precision_at_0.1ppm':float(np.mean(precision)),
        'strata':{k:{'n':len(v),'W1_ppm':float(np.mean(v))} for k,v in strata.items()}}
    metrics['reported_total_integral_MAE_H_all_records']=float(np.mean(integral_errors))
    if visibility_counts:metrics['visibility_supervision_counts']=dict(visibility_counts)
    if aux_counts:
        metrics['aux_counts']=dict(aux_counts)
        if 'multiplicity_correct' in aux_counts:metrics['matched_proton_multiplicity_accuracy']=aux_counts['multiplicity_correct']/aux_counts['mult_slots'] if aux_counts['mult_slots'] else None
        if 'j_abs_error_sum' in aux_counts:metrics['matched_known_J_MAE_Hz']=aux_counts['j_abs_error_sum']/aux_counts['j_values'] if aux_counts['j_values'] else None
        if 'joint_multiplicity_correct_at_0_1ppm' in aux_counts:
            metrics['joint_shift_multiplicity_accuracy_at_0.1ppm']=aux_counts['joint_multiplicity_correct_at_0_1ppm']/aux_counts['mult_slots'] if aux_counts['mult_slots'] else None
            metrics['joint_shift_multiplicity_J_MAE_Hz']=aux_counts['joint_J_abs_error_sum']/aux_counts['joint_J_values'] if aux_counts['joint_J_values'] else None
            metrics['joint_J_fraction_of_known_values']=aux_counts['joint_J_values']/aux_counts['j_values'] if aux_counts['j_values'] else None
    return metrics,rows


def evaluate_test(args,data,device):
    out=pathlib.Path(args.out)
    verify_test_cache(out, args.cache)
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    cfg=ck['args'];model=(VisibilityNMRModel if cfg.get('visibility',False) else NMRForwardModel)(feature_dim=ck['feature_dim'],hidden_dim=cfg['width'],depth=cfg['depth'],aux_classes=ck.get('aux_classes',0),tie_methyl=cfg.get('tie_methyl',False)).to(device);model.load_state_dict(ck['model'])
    val=json.loads((out/'validation.json').read_text());threshold=val['p99_W1_ppm']
    test=[r for r in data if r['split']=='test']
    metric,rows=score(model,DataLoader(test,batch_size=args.batch_size,collate_fn=collate),device,True)
    for r in rows:r['review_high_error']=r['error_w1_ppm']>threshold;r['confirmed_data_error']=False
    flaggroups=collections.defaultdict(list)
    for r in rows:
        flaggroups['all'].append(r['error_w1_ppm'])
        flaggroups['no_metadata_flags' if not r['flags'] else 'metadata_flagged'].append(r['error_w1_ppm'])
    metric.update(validation_selected_epoch=ck['epoch'],review_threshold_from_validation_p99=threshold,
        quality_strata={k:{'n':len(v),'W1_ppm':float(np.mean(v))} for k,v in flaggroups.items()},
        checkpoint_sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),
        evaluation_code_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        warning='峰表积分分布W1，非原子归属MAE或实测完整谱误差。高误差不等于错误标签。')
    (out/'test_metrics.json').write_text(json.dumps(metric,indent=2,ensure_ascii=False))
    with (out/'test_predictions.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (out/'test_review_candidates.jsonl').open('w') as f:
        for r in sorted(rows,key=lambda x:x['error_w1_ppm'],reverse=True):
            if r['flags'] or r['review_high_error']:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(metric,ensure_ascii=False),flush=True)


def filehash(path):
    sha=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):sha.update(chunk)
    return sha.hexdigest()


def verify_test_cache(out, cache):
    provenance=json.loads((pathlib.Path(out)/'provenance.json').read_text())
    expected=provenance.get('validation_cache_sha256', provenance.get('cache_sha256'))
    if not expected or filehash(cache)!=expected:
        raise ValueError('test cache SHA mismatch with registered evaluation cache')


def resolve_splits(data, validation_data=None, smoke=False):
    evaluation_data=data if validation_data is None else validation_data
    train=[r for r in data if r['split']=='train']
    val=[r for r in evaluation_data if r['split']=='val']
    train_groups={r['group'] for r in train}
    held_out={r['group'] for r in evaluation_data if r['split'] in ('val','test')}
    overlap=train_groups & held_out
    if overlap:
        raise ValueError(f'train/validation-or-test group overlap: {len(overlap)} groups')
    if smoke:train=train[:256];val=val[:128]
    if not train or not val:raise ValueError('nonempty train and validation sets required')
    return train,val


def warmstart_model(model, path, *, feature_dim, width, depth, aux_classes, tie_methyl):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    config=checkpoint['args']
    actual=(checkpoint['feature_dim'],config['width'],config['depth'],
            checkpoint.get('aux_classes',0),config.get('tie_methyl',False))
    expected=(feature_dim,width,depth,aux_classes,tie_methyl)
    if actual!=expected:raise ValueError(f'warmstart architecture mismatch: {actual} != {expected}')
    if isinstance(model,VisibilityNMRModel) and not config.get('visibility',False):
        expected_base={k for k in model.state_dict() if not k.startswith('visibility_head.')}
        if set(checkpoint['model'])!=expected_base:raise ValueError('warmstart base keys mismatch')
        model.load_state_dict(checkpoint['model'],strict=False)
    else:
        model.load_state_dict(checkpoint['model'],strict=True)
    return filehash(path)


def build_contract(args, provenance, feature_dim, aux_classes):
    config={key:value for key,value in vars(args).items()
            if key not in ('resume','evaluate_test','out','cache','validation_cache','init_from')}
    return {'config':config,'feature_dim':feature_dim,'aux_classes':aux_classes,
            'cache_sha256':provenance['cache_sha256'],
            'validation_cache_sha256':provenance['validation_cache_sha256'],
            'source_sha256':provenance['source_sha256'],
            'init_from_sha256':provenance['init_from_sha256'],
            'torch':provenance['torch'],'numpy':provenance['numpy'],'cuda':provenance['cuda']}


def verify_committed_best(out, completed_epoch, best):
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    validation=json.loads((out/'validation.json').read_text())
    if not 1<=checkpoint.get('epoch',0)<=completed_epoch:
        raise ValueError('best checkpoint is outside committed last.pt epochs; manual recovery required')
    if validation.get('integral_weighted_W1_ppm')!=best:
        raise ValueError('best validation differs from committed last.pt; manual recovery required')


def reconcile_history(path, completed_epoch):
    # history 先于 last.pt 落盘；崩溃可能留下尚未提交的尾行，只保留 checkpoint 已提交前缀。
    lines=path.read_text().splitlines() if path.exists() else []
    entries=[]
    for line in lines:
        if len(entries)>=completed_epoch:break
        entry=json.loads(line)
        if entry.get('epoch')!=len(entries)+1:raise ValueError('noncontiguous checkpoint history')
        entries.append(entry)
    if len(entries)!=completed_epoch:raise ValueError('history missing checkpoint epochs')
    temporary=path.with_name(path.name+'.reconcile.tmp')
    temporary.write_text(''.join(json.dumps(entry)+'\n' for entry in entries))
    os.replace(temporary,path)
    return entries[-1].get('elapsed_seconds',0.) if entries else 0.


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--width',type=int,default=128);p.add_argument('--depth',type=int,default=5);p.add_argument('--seed',type=int,default=21)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=0.0007)
    p.add_argument('--device',default='cuda');p.add_argument('--evaluate-test',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--aux-weight',type=float,default=0.)
    p.add_argument('--tie-methyl',action='store_true')
    p.add_argument('--init-from');p.add_argument('--validation-cache');p.add_argument('--resume',action='store_true')
    p.add_argument('--visibility',action='store_true');p.add_argument('--visibility-weight',type=float,default=1.)
    p.add_argument('--visibility-freeze-epochs',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    if args.visibility_freeze_epochs and (not args.visibility or not args.init_from):raise ValueError('visibility freeze requires visibility and pretrained init-from')
    if args.visibility_weight<0 or args.visibility_freeze_epochs<0:raise ValueError('visibility settings must be nonnegative')
    if args.epochs<1:raise ValueError('epochs must be positive')
    if args.evaluate_test and args.resume:raise ValueError('evaluate-test and resume are mutually exclusive')
    out=pathlib.Path(args.out)
    if args.resume and (out/'complete.json').exists():raise ValueError('completed run cannot resume; use a new warmstart run')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if args.evaluate_test:
        verify_test_cache(out,args.cache)
        data=torch.load(args.cache,weights_only=False)
        evaluate_test(args,data,args.device);return
    data=torch.load(args.cache,weights_only=False)
    validation_data=torch.load(args.validation_cache,weights_only=False) if args.validation_cache else None
    train,val=resolve_splits(data,validation_data,args.smoke)
    feature_dim=train[0]['x'].shape[1]
    aux_classes=len(MULTIPLICITY_CLASSES) if args.aux_weight else 0
    here=pathlib.Path(__file__).parent
    provenance={'cache_sha256':filehash(args.cache),
        'validation_cache_sha256':filehash(args.validation_cache or args.cache),
        'source_sha256':{path.name:filehash(path) for path in sorted(here.glob('*.py'))},
        'init_from_sha256':filehash(args.init_from) if args.init_from else None,
        'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,
        'target':'integral_peak_table_distribution','metric':'exact_discrete_Wasserstein_1_ppm'}
    contract=build_contract(args,provenance,feature_dim,aux_classes)
    model=(VisibilityNMRModel if args.visibility else NMRForwardModel)(feature_dim=feature_dim,hidden_dim=args.width,depth=args.depth,aux_classes=aux_classes,tie_methyl=args.tie_methyl).to(args.device)
    if args.init_from and not args.resume:
        warmstart_model(model,args.init_from,feature_dim=feature_dim,width=args.width,
                        depth=args.depth,aux_classes=aux_classes,tie_methyl=args.tie_methyl)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.lr*.04)
    generator=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=collate,generator=generator)
    val_loader=DataLoader(val,batch_size=args.batch_size,collate_fn=collate)
    best=float('inf');completed_epoch=0;elapsed_before=0.
    if args.resume:
        completed_epoch,best=restore_training_state(out/'last.pt',model=model,optimizer=opt,
            scheduler=sched,expected_contract=contract,loader_generator=generator)
        if completed_epoch>args.epochs:raise ValueError('checkpoint exceeds requested epochs')
        verify_committed_best(out,completed_epoch,best)
        elapsed_before=reconcile_history(out/'history.jsonl',completed_epoch)
    else:
        out.mkdir(parents=True,exist_ok=False)
        (out/'config.json').write_text(json.dumps(vars(args),indent=2))
        (out/'source').mkdir()
        for source_file in here.glob('*.py'):shutil.copy2(source_file,out/'source'/source_file.name)
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
        baseline=np.median(np.stack([r['target'] for r in train]),axis=0)
        baseline_errors=[]
        for batch in val_loader:
            values=torch.tensor(baseline).expand(len(batch['rows']),-1)
            baseline_errors.extend(exact_w1({'shifts':values,'proton_mask':torch.ones_like(values,dtype=torch.bool)},batch).tolist())
        (out/'baseline.json').write_text(json.dumps({'model':'training_quantile_median','validation_W1_ppm':float(np.mean(baseline_errors)),'train_n':len(train),'val_n':len(val)}))
    start=time.time()
    with (out/'history.jsonl').open('a' if args.resume else 'w',buffering=1) as history:
        for epoch in range(completed_epoch+1,args.epochs+1):
            if args.visibility:
                for name,param in model.named_parameters():
                    param.requires_grad_(epoch>args.visibility_freeze_epochs or name.startswith('visibility_head.'))
            model.train();total=0.;n=0;aux_total=0.;visibility_total=0.
            for batch in train_loader:
                batch=move(batch,args.device);opt.zero_grad(set_to_none=True)
                prediction=model(batch);w1=exact_w1(prediction,batch).mean();loss=w1
                if args.visibility:
                    visible_loss=visibility_loss(prediction,batch)['loss']
                    loss=loss+args.visibility_weight*visible_loss
                    visibility_total+=float(visible_loss.detach())*len(batch['rows'])
                if args.aux_weight:
                    aux_loss=peak_aux_loss(prediction,batch)['loss']
                    ramp=min(1.,max(0.,(epoch-3)/5))
                    loss=loss+args.aux_weight*ramp*aux_loss
                    aux_total+=float(aux_loss.detach())*len(batch['rows'])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
                total+=float(w1.detach())*len(batch['rows']);n+=len(batch['rows'])
            sched.step();metrics,_=score(model,val_loader,args.device)
            entry=dict(epoch=epoch,train_W1_ppm=total/n,train_aux_loss=aux_total/n,train_visibility_loss=visibility_total/n,validation=metrics,lr=sched.get_last_lr()[0],elapsed_seconds=elapsed_before+time.time()-start)
            history.write(json.dumps(entry)+'\n');history.flush();os.fsync(history.fileno());print(json.dumps(entry),flush=True)
            if metrics['integral_weighted_W1_ppm']<best:
                best=metrics['integral_weighted_W1_ppm'];torch.save({'model':model.state_dict(),'args':vars(args),'feature_dim':feature_dim,'aux_classes':aux_classes,'epoch':epoch,'provenance':provenance},out/'best.tmp.pt');os.replace(out/'best.tmp.pt',out/'best.pt')
                (out/'validation.json').write_text(json.dumps(metrics,indent=2))
            save_training_state(out/'last.pt',model=model,optimizer=opt,scheduler=sched,
                epoch=epoch,best=best,contract=contract,loader_generator=generator)
    (out/'complete.json').write_text(json.dumps({'epochs':args.epochs,'best_validation_W1_ppm':best,'seconds':elapsed_before+time.time()-start}))

if __name__=='__main__':main()

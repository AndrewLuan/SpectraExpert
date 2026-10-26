"""功能概述：仅用训练结构的难负样本学习低容量候选谱重排，保留完整候选库。

ResidualRanker 以原粗评分为基座，加共享/观测口径条件化的小残差；输入只含数值
匹配特征，不读取候选名称、分子 ID 或正确结构。feature_columns 预声明匹配/观测/
联合消融。fit 按结构哈希在既有 train 中留 20% 内部核验，标准化只拟合其余 train，
相同 group 抽样序列用于增强开关对照；难负样本 pairwise 损失与教师评分锚联合优化。
evaluate 只重排粗检索生成的短名单，名单外候选排名保持原值；alpha=0 精确返回
原完整库排名和 Top10，训练缓存补入的正例不得在任何指标中越过召回上界。
main 的 fit/train_val 仅处理 train/val 特征；locked 阶段需冻结模型/缓存 SHA。
全部输出独占，包含初始/最佳权重、训练曲线、内部选择、各验证指标和真实退出码。
不改写均值模型、原谱、候选或测试，不访问任何外部服务。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .h_global_features import FEATURE_NAMES
from .prepare import sha


def feature_columns(kind):
    if kind=='matching':return list(range(20))
    if kind=='observation':return [0]+list(range(20,len(FEATURE_NAMES)))
    if kind=='all':return list(range(len(FEATURE_NAMES)))
    raise ValueError('invalid_global_feature_family')


class ResidualRanker(nn.Module):
    def __init__(self,columns,architecture='linear',typed=True):
        super().__init__(); self.columns=columns; self.typed=typed
        groups=4 if typed else 1
        if architecture=='linear':self.network=nn.Linear(len(columns),groups,bias=False)
        elif architecture=='mlp':self.network=nn.Sequential(nn.Linear(len(columns),32),nn.SiLU(),nn.Linear(32,groups,bias=False))
        else:raise ValueError('invalid_ranker_architecture')
        last=self.network if architecture=='linear' else self.network[-1]
        nn.init.zeros_(last.weight)

    def forward(self,z,kinds):
        logits=self.network(z[:,:,self.columns])
        indices=kinds if self.typed else torch.zeros_like(kinds)
        raw=logits.gather(2,indices[:,None,None].expand(-1,z.shape[1],1)).squeeze(2)
        delta=5*torch.tanh(raw/5)
        return z[:,:,0],delta


def row_center(row):
    x=row['features']
    visible=x if row['target_in_coarse_pool'] else x[:-1] if row['target_position']>=0 else x
    return visible.median(0).values


def fit_scale(rows):
    samples=[]
    for row in rows:
        x=row['features']; n=len(x)-(not row['target_in_coarse_pool'] and row['target_position']>=0)
        indices=torch.linspace(0,max(0,n-1),min(16,n)).long()
        samples.append((x[indices]-row_center(row)).abs())
    return torch.cat(samples).quantile(.75,dim=0).clamp_min(.05)


def normalized_rows(rows,scale):
    result=[]
    for r in rows:
        raw=(r['features']-row_center(r))/scale; z=raw.clamp(-20,20); z[:,0]=raw[:,0]
        result.append(dict(r,z=z,score_scale=float(scale[0])))
    return result


def pack(rows,device):
    maximum=max(len(r['z']) for r in rows); n=len(rows)
    z=torch.zeros(n,maximum,len(FEATURE_NAMES)); mask=torch.zeros(n,maximum,dtype=torch.bool)
    for i,row in enumerate(rows):
        z[i,:len(row['z'])]=row['z']; mask[i,:len(row['z'])]=True
    return z.to(device),mask.to(device),torch.tensor([r['kind'] for r in rows],device=device)


def metrics(rows):
    def summarize(rs):
        n=len(rs)
        if not n:return dict(queries=0)
        return dict(queries=n,**{f'top{k}':sum(r['rank']<=k for r in rs)/n for k in (1,5,10)},
                    MRR=sum(1/r['rank'] for r in rs)/n)
    original=[r for r in rows if r['variant']=='observed']
    return dict(overall=summarize(original),by_source={s:summarize([r for r in original if r['source']==s]) for s in sorted({r['source'] for r in original})},
                stress={s:summarize([r for r in rows if r['variant']==s]) for s in ('frequency_crop','peak_dropout')})


def rank_one(row,base,delta,alpha):
    if alpha==0 or not row.get('rerank_eligible',True):
        return row['coarse_rank'],row['coarse_top10']
    size=len(row['candidate_indices'])
    if not row['target_in_coarse_pool'] and row['target_position']>=0:size-=1
    if bool((delta[:size]==delta[0]).all()):return row['coarse_rank'],row['coarse_top10']
    raw=row.get('coarse_scores',base)
    raw=raw[:size].to(delta.device)
    values=raw+alpha*delta[:size]*row.get('score_scale',1.)
    target=row['target_position']
    rank=int((values<=values[target]+1e-7).sum()) if row['target_in_coarse_pool'] else row['coarse_rank']
    top=values.topk(min(10,size),largest=False).indices.cpu()
    return rank,row['candidate_indices'][top].tolist()


@torch.inference_mode()
def evaluate(model,rows,device,alphas):
    model.eval(); outputs={str(a):[] for a in alphas}
    for start in range(0,len(rows),64):
        group=rows[start:start+64]; z,mask,kinds=pack(group,device); base,delta=model(z,kinds)
        for j,row in enumerate(group):
            for alpha in alphas:
                rank,top=rank_one(row,base[j],delta[j],alpha)
                outputs[str(alpha)].append(dict(source=row['source'],record_id=row['record_id'],smiles=row['smiles'],group=row['group'],
                    variant=row['variant'],rank=rank,top10_indices=top,query_order=row['query_order'],
                    baseline_rank=row['baseline_rank'],coarse_rank=row['coarse_rank']))
    return outputs,{a:metrics(rs) for a,rs in outputs.items()}


def fit(args,out,device,status):
    data=torch.load(args.train,map_location='cpu',weights_only=False)
    if data['manifest']['cohort']!='train':raise ValueError('ranker_fit_requires_train_cache')
    valid_groups={r['group'] for r in data['rows'] if r['variant']=='observed' and r.get('rerank_eligible',True)}
    rows=[r for r in data['rows'] if r['group'] in valid_groups and r.get('rerank_eligible',True) and (args.augment or r['variant']=='observed')]
    internal=lambda r:int(hashlib.sha256(('ranker_internal_317'+str(r['group'])).encode()).hexdigest()[:12],16)%5==0
    training=[r for r in rows if not internal(r)]; validation=[r for r in rows if internal(r)]
    if {r['group'] for r in training}&{r['group'] for r in validation}:raise ValueError('ranker_internal_group_overlap')
    scale=fit_scale(training); training=normalized_rows(training,scale); validation=normalized_rows(validation,scale)
    groups=collections.defaultdict(list)
    for row in training:groups[row['group']].append(row)
    group_names=sorted(groups)
    torch.manual_seed(args.seed); model=ResidualRanker(feature_columns(args.features),args.architecture,not args.shared).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003 if args.architecture=='linear' else .001,weight_decay=1e-3)
    alphas=[0.,.25,.5,1.]; best=-1.; chosen=None; start=time.monotonic(); exposure=hashlib.sha256()
    def save(name,epoch,alpha,report):
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},scale=scale,columns=model.columns,architecture=args.architecture,
            typed=not args.shared,epoch=epoch,alpha=alpha,internal=report,train_sha256=sha(args.train),args=vars(args),
            feature_names=FEATURE_NAMES,candidate_order_sha256=data['manifest']['candidate_order_sha256'],
            prediction_checkpoint_sha256=data['manifest']['checkpoint_sha256'],coarse_config=data['manifest']['coarse_config'],
            pool_size=data['manifest']['pool_size'],feature_code_sha256=data['manifest']['feature_code_sha256']),out/name)
    for epoch in range(args.epochs+1):
        total=0.; steps=0
        if epoch:
            rng=np.random.default_rng(args.seed+2000+epoch); vrng=np.random.default_rng(args.seed+4000+epoch)
            sequence=rng.integers(0,len(group_names),size=args.samples)
            exposure.update(sequence.tobytes()); model.train()
            for offset in range(0,len(sequence),32):
                batch=[]
                for index in sequence[offset:offset+32]:
                    options=groups[group_names[int(index)]]; batch.append(options[int(vrng.integers(len(options)))])
                z,mask,kinds=pack(batch,device); base,delta=model(z,kinds); cost=base+delta
                target=torch.tensor([r['target_position'] for r in batch],device=device)
                if (target<0).any():raise ValueError('ranker_training_positive_missing')
                positive=cost.gather(1,target[:,None]); negative_mask=mask.clone()
                for j,row in enumerate(batch):
                    if 'negative_exclusion_mask' in row:
                        negative_mask[j,:len(row['negative_exclusion_mask'])]&=~row['negative_exclusion_mask'].to(device)
                negative_mask.scatter_(1,target[:,None],False)
                negative=cost.masked_fill(~negative_mask,float('inf'))
                hard=negative.topk(min(16,cost.shape[1]-1),largest=False).values
                pairs=F.softplus((positive-hard)/.5)
                loss=pairs.mean()+args.anchor*(delta.square()*mask).sum()/mask.sum()
                if not torch.isfinite(loss):raise ValueError('ranker_nonfinite_training_loss')
                optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.); optimizer.step()
                total+=float(loss.detach()); steps+=1
        if epoch==0 or epoch%8==0 or epoch==args.epochs:
            _,reports=evaluate(model,validation,device,alphas)
            candidate=max(alphas,key=lambda a:(reports[str(a)]['overall']['MRR'],-a))
            signal=reports[str(candidate)]['overall']['MRR']
            if epoch==0:save('initial.pt',epoch,0.,reports)
            if signal>best+1e-5:
                best=signal; chosen=dict(epoch=epoch,alpha=candidate,MRR=signal); save('best.pt',epoch,candidate,reports)
            record=dict(epoch=epoch,training_loss=total/max(1,steps),internal=reports,best=chosen,seconds=time.monotonic()-start)
            with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            status('fit',**{k:record[k] for k in ('epoch','training_loss','best','seconds')})
            print(json.dumps(dict(epoch=epoch,best=chosen,seconds=record['seconds'])),flush=True)
    save('last.pt',args.epochs,chosen['alpha'],{})
    result=dict(best=chosen,training_groups=len(groups),internal_groups=len({r['group'] for r in validation}),
        training_rows=len(training),internal_rows=len(validation),group_exposure_sha256=exposure.hexdigest(),
        train_cache_sha256=sha(args.train),checkpoint_sha256=sha(out/'best.pt'),feature_names=FEATURE_NAMES,
        external_api_calls=0,selection='internal train structure split only; later original320/tune choose model')
    (out/'summary.json').write_text(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['fit','train_val','locked'],required=True)
    p.add_argument('--out',required=True); p.add_argument('--train'); p.add_argument('--checkpoint'); p.add_argument('--cache')
    p.add_argument('--features',choices=['matching','observation','all'],default='all')
    p.add_argument('--architecture',choices=['linear','mlp'],default='linear'); p.add_argument('--shared',action='store_true')
    p.add_argument('--augment',action='store_true'); p.add_argument('--seed',type=int,default=193)
    p.add_argument('--anchor',type=float,default=.1); p.add_argument('--epochs',type=int,default=64)
    p.add_argument('--samples',type=int,default=2048); p.add_argument('--frozen-selection')
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4); device=torch.device('cuda')
        if args.stage=='fit':fit(args,out,device,status)
        else:
            saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            data=torch.load(args.cache,map_location='cpu',weights_only=False)
            if data['manifest']['candidate_order_sha256']!=saved['candidate_order_sha256']:raise ValueError('ranker_candidate_order_drift')
            if args.stage=='train_val' and data['manifest']['cohort'] not in ('train','original_val','tune'):
                raise ValueError('ranker_train_val_access_violation')
            if args.stage=='locked':
                frozen=json.loads(Path(args.frozen_selection).read_text())
                if frozen.get('selection_scope')!='train_val_only':raise ValueError('ranker_invalid_frozen_selection')
                if data['manifest']['cohort'] not in ('verify','test'):raise ValueError('ranker_locked_cohort_invalid')
                if frozen['ranker_checkpoints_sha256'].get(str(Path(args.checkpoint).resolve()))!=sha(args.checkpoint):
                    raise ValueError('ranker_checkpoint_not_frozen')
                for path,digest in data['manifest']['source_inputs_sha256'].items():
                    if frozen['evaluation_inputs_sha256'].get(path)!=digest or sha(path)!=digest:
                        raise ValueError('ranker_feature_source_not_frozen:'+path)
            for field in ('coarse_config','pool_size','feature_code_sha256'):
                if saved[field]!=data['manifest'][field]:raise ValueError('ranker_feature_protocol_mismatch:'+field)
            if saved['prediction_checkpoint_sha256']!=data['manifest']['checkpoint_sha256']:
                raise ValueError('ranker_prediction_checkpoint_mismatch')
            if saved['feature_names']!=data['manifest']['feature_names']:raise ValueError('ranker_feature_schema_drift')
            model=ResidualRanker(saved['columns'],saved['architecture'],saved['typed']).to(device); model.load_state_dict(saved['state_dict'])
            normalized=normalized_rows(data['rows'],saved['scale'])
            results,reports=evaluate(model,normalized,device,[0.,saved['alpha']])
            for alpha,rs in results.items():
                for row in rs:row['top10']=[data['smiles'][i] for i in row.pop('top10_indices')]
                (out/('predictions_'+alpha+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rs))
            (out/'summary.json').write_text(json.dumps(dict(metrics=reports,alpha=saved['alpha'],checkpoint_sha256=sha(args.checkpoint),
                cache_sha256=sha(args.cache),queries=data['manifest']['queries'],selection_scope='frozen internal train alpha'),indent=2))
        status('complete'); (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc)); (out/'EXIT_CODE').write_text('1\n'); raise


if __name__=='__main__':
    main()

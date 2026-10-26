"""功能概述：从冻结¹H候选谱提取非交换H观测分布，供水相HMDB消融。
cached_nonexchange只从重新前向传播获取128个分位点的交换H掩码，峰位置全部取
原bank；这样GPU聚合的浮点顺序差异不会改变任何已有候选峰位置。先核对新旧完整
输出最大差异小于0.01ppm，以确保分位点掩码对应同一位置；差异写入审计，不用于评分。
main复用候选顺序、条件、权重和原图缓存；交换H按候选N/O/S母节点确定，不使用
真查询结构或实验标签。输出独占bank、状态和退出码，缓存含候选身份和权重hash。
下游必须通过val比较交换H可见性；不会直接断言所有水相交换H都消失。128分位数
可能已有质量量化，提取后保留这一口径；不重新训练，也不修改旧完整候选谱。
"""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_batch
from .emission_model import make_model,weighted_quantiles
from .train import Spectra,collate
from .prepare import sha


def cached_nonexchange(details,cached_quantiles,batches):
    if not len(details['values']):return cached_quantiles
    values,_=to_dense_batch(details['values'],details['groups'],fill_value=1e6,batch_size=batches)
    weights,_=to_dense_batch(details['weights'],details['groups'],fill_value=0.,batch_size=batches)
    exchange,_=to_dense_batch(details['exchangeable'],details['groups'],fill_value=True,batch_size=batches)
    _,order=values.sort(1);mass=weights.gather(1,order);cdf=mass.cumsum(1)/mass.sum(1,keepdim=True).clamp_min(1)
    u=((torch.arange(128,device=values.device)+.5)/128).expand(batches,-1).contiguous()
    idx=torch.searchsorted(cdf.contiguous(),u).clamp_max(values.shape[1]-1)
    hidden=exchange.gather(1,order).gather(1,idx);groups=torch.arange(batches,device=values.device)[:,None].expand_as(hidden)
    return weighted_quantiles(cached_quantiles[~hidden],torch.ones_like(cached_quantiles)[~hidden],groups[~hidden],batches)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');status('loading')
        bank=torch.load(base/'hydrogen_analysis/evaluation_dropout/site/bank.pt',map_location='cpu',weights_only=False,mmap=True)
        checkpoint=base/'hydrogen_analysis/site_supervision/train/best.pt'
        if sha(checkpoint)!=bank['checkpoint_sha256']:raise ValueError('checkpoint_changed')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_model(saved['args']).cuda().eval();model.load_state_dict(saved['state_dict'])
        cache=torch.load(base/'prepared_coverage/cache.pt',map_location='cpu',weights_only=False,mmap=True)
        template=dict(nucleus='1H',domain='experimental',source='structure_only',record_id='',q=np.zeros(128,np.float32),observation=dict(kind='unknown',windows=[]))
        predictions={};max_error=0.;differences=[]
        for cond in sorted(c for c in bank['predictions'] if c[1]>0):
            status('predicting',condition=cond)
            rows=[dict(template,smiles=s,physics=p.numpy(),cond=np.array(cond,np.float32)) for s,p in zip(bank['smiles'],bank['physics'])]
            loader=DataLoader(Spectra(rows,cache['graphs'],'1H'),batch_size=256,num_workers=2,collate_fn=collate);pieces=[];offset=0
            with torch.inference_mode():
                for b in loader:
                    b=b.cuda();d=model(b,details=True);difference=(d['quantiles'].cpu()-bank['predictions'][cond][offset:offset+b.num_graphs]).abs();error=float(difference.max());max_error=max(max_error,error)
                    if error>1e-3:
                        local=int(difference.max(1).values.argmax());differences.append(dict(offset=offset,smiles=bank['smiles'][offset+local],max_normalized=error,mean_normalized=float(difference.mean()),old=bank['predictions'][cond][offset+local].tolist(),new=d['quantiles'][local].cpu().tolist()))
                    cached=bank['predictions'][cond][offset:offset+b.num_graphs].to(b.x.device);pieces.append(cached_nonexchange(d,cached,b.num_graphs).cpu());offset+=b.num_graphs
            predictions[cond]=torch.cat(pieces)
        (out/'prediction_drift.json').write_text(json.dumps(dict(max_normalized=max_error,batches=differences)))
        if differences:raise ValueError('cached_prediction_changed; inspect prediction_drift.json')
        torch.save(dict(smiles=bank['smiles'],checkpoint_sha256=bank['checkpoint_sha256'],predictions=predictions),out/'bank.pt')
        status('complete',max_full_prediction_error=max_error);(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

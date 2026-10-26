"""功能概述：在相同¹H候选库上做验证选评分与冻结后的检索复评，原/继续训练/位点模型分开。
main的val阶段从原cache按每来源64条固定hash取验证谱，预测候选bank，只用val MRR
选择预登记的似然/部分传输β网格。test阶段必须有全局frozen_selection.json，复用bank
在旧1000条test查询和前200条缺测变体上评价，不再拟合任何参数。
evaluate仅将观测位置/质量传给h_score，真实结构只计算rank与事后分层；LazyFP按需算指纹。
site_audit诊断正确母节点的误差，不参与评分/选模；与score结果分开保存。
新输出目录独占；等待指定训练退出码，失败停止。依赖旧诊断bank、原始cache和标签侧表。
"""
import argparse
import collections
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from .train import Spectra,collate
from .emission_model import make_model,weighted_quantiles
from .h_labels import make_key
from .h_diagnose import get_queries,cond_key
from .h_score import CONFIGS,fit_count_prior,attach_observation,score_features,select_score,query_config
from .likelihood import density,log_score
from .joint import rank_entry
from .evaluate import summarize,missing_queries
from .prepare import sha


class LazyFP(dict):
    def __init__(self):super().__init__();self.generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    def __missing__(self,s):
        value=self.generator.GetFingerprint(Chem.MolFromSmiles(s));self[s]=value;return value


@torch.inference_mode()
def evaluate(bank,queries,labels,prior,configs,indices,device,out,stress=False):
    ids=indices.tolist();smiles=[bank['smiles'][i] for i in ids];index={s:i for i,s in enumerate(smiles)}
    counts=bank['physics'][indices,0].to(device)*40;fps=LazyFP();records={name:[] for name in configs};missing=[]
    predictions={c:p[indices].to(device) for c,p in bank['predictions'].items()}
    kernels={c:density(p,'1H',.2) for c,p in predictions.items()}
    for j,r in enumerate(queries):
        query=attach_observation(r,labels.get(make_key(r),{}));cond=cond_key(r)
        resolved={name:query_config(c,query) for name,c in configs.items()}
        need_transport=any(c['kind']=='partial_transport' for c in resolved.values())
        if need_transport:features=score_features(predictions[cond],kernels[cond],counts,query,prior)
        else:features=dict(likelihood=log_score(kernels[cond],query,'1H'),eligible=False)
        for name,config in resolved.items():
            scores=select_score(features,config)
            records[name].append(dict(record_id=r['record_id'],smiles=r['smiles'],source=r['source'],transport_eligible=features['eligible'],
                **rank_entry(scores,index[r['smiles']],smiles,fps)))
        if stress and j<200:
            for variant,rr in missing_queries(query,'1H'):
                config=query_config(configs['selected'],rr)
                if config['kind']=='partial_transport':ff=score_features(predictions[cond],kernels[cond],counts,rr,prior)
                else:ff=dict(likelihood=log_score(kernels[cond],rr,'1H'))
                missing.append(dict(record_id=r['record_id'],smiles=r['smiles'],variant=variant,**rank_entry(select_score(ff,config),index[r['smiles']],smiles,fps)))
        if (j+1)%100==0:print(json.dumps(dict(ranked=j+1,total=len(queries))),flush=True)
    report={}
    for name,rs in records.items():
        strata=collections.defaultdict(list)
        for r in rs:strata[r['source']].append(r)
        report[name]=dict(config=configs[name],overall=summarize(rs),by_source={s:summarize(xs) for s,xs in strata.items()})
        with (out/(name+'_predictions.jsonl')).open('x') as f:
            for r in rs:f.write(json.dumps(r)+'\n')
    if missing:
        report['missing_stress']={s:summarize([r for r in missing if r['variant']==s]) for s in ['frequency_crop','peak_dropout']}
        original={r['smiles']+r['record_id']:r for r in records['selected']}
        report['missing_same_query_controls']={s:summarize([original[r['smiles']+r['record_id']] for r in missing if r['variant']==s]) for s in ['frequency_crop','peak_dropout']}
        with (out/'missing_predictions.jsonl').open('x') as f:
            for r in missing:f.write(json.dumps(r)+'\n')
    return report


@torch.inference_mode()
def site_audit(model,rows,graphs,labels,device):
    loader=DataLoader(Spectra(rows,graphs,'1H',annotations=labels),batch_size=256,num_workers=2,collate_fn=collate)
    groups=collections.defaultdict(list);records=[];offset=0
    for b in loader:
        b=b.to(device);d=model(b,details=True)
        pred=weighted_quantiles(d['values'],d['weights'],d['owners'],len(b.x),points=6)*10
        for j,r in enumerate(rows[offset:offset+b.num_graphs]):
            local=[]
            for node,values in labels.get(make_key(r),{}).get('groups',{}).items():
                idx=int(b.ptr[j])+int(node);pp=pred[idx];target=torch.tensor(values,device=device)
                err=float((pp[:,None]-target[None,:]).abs().min(0).values.mean())
                z=int(b.x[idx,:54].argmax())+1;tag='exchange_H' if z in [7,8,16] else 'C_H'
                groups[tag].append(err);local.append(dict(parent=int(node),kind=tag,observed=values,predicted=pp.cpu().tolist(),error_ppm=err))
            if local:records.append(dict(record_id=r['record_id'],smiles=r['smiles'],source=r['source'],sites=local))
        offset+=b.num_graphs
    return dict(metric='observed target to correct-parent unordered slot; no atom identity within parent asserted',
                summary={k:dict(sites=len(v),mean_ppm=float(np.mean(v)),median_ppm=float(np.median(v))) for k,v in groups.items()},records=records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True)
    p.add_argument('--variant',choices=['original','presence','site'],required=True);p.add_argument('--stage',choices=['val','test'],required=True)
    p.add_argument('--checkpoint');p.add_argument('--wait-for')
    p.add_argument('--reuse-bank')
    a=p.parse_args();base=Path(a.base);root=Path(a.out);out=root/a.stage;out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        if a.wait_for:
            status('waiting_training');path=Path(a.wait_for);deadline=time.monotonic()+7200
            while not path.exists():
                if time.monotonic()>deadline:raise TimeoutError('training_timeout')
                time.sleep(10)
            if path.read_text().strip()!='0':raise RuntimeError('training_failed')
        device=torch.device('cuda');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
        labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        if a.stage=='val':
            status('loading');bank=torch.load(base/'hydrogen_analysis/diagnosis/bank.pt',map_location='cpu',weights_only=False,mmap=True)
            cache=torch.load(base/'prepared_coverage/cache.pt',map_location='cpu',weights_only=False,mmap=True)
            rows=[r for r in cache['rows'] if r['nucleus']=='1H' and r['domain']=='experimental'];by_source=collections.defaultdict(list)
            for r in rows:by_source[r['source']].append(r)
            bank['val']=[r for source in sorted(by_source) for r in get_queries(by_source[source],'val',64)]
            checkpoint=Path(a.checkpoint) if a.checkpoint else base/'atom_emission/1H/domain/best.pt'
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_model(payload['args']).to(device)
            model.load_state_dict(payload['state_dict']);model.eval()
            candidate={r['smiles']:r for r in cache['rows'] if r['domain']=='experimental'}
            conditions={cond_key(r) for r in bank['test']+bank['paired']+bank['val']}
            predictions=dict(bank['predictions']) if a.variant=='original' else {}
            if a.reuse_bank:
                reuse=torch.load(a.reuse_bank,map_location='cpu',weights_only=False,mmap=True)
                if reuse['checkpoint_sha256']!=sha(checkpoint) or reuse['smiles']!=bank['smiles']:raise ValueError('reuse_bank_identity_mismatch')
                predictions=dict(reuse['predictions'])
            for cond in sorted(conditions):
                if cond in predictions:continue
                status('predicting_candidates',condition=cond)
                rs=[dict(candidate[s],q=np.zeros(128,np.float32),cond=np.array(cond,np.float32),domain='experimental') for s in bank['smiles']]
                loader=DataLoader(Spectra(rs,cache['graphs'],'1H'),batch_size=256,num_workers=2,collate_fn=collate)
                with torch.inference_mode():predictions[cond]=torch.cat([model(b.to(device)).cpu() for b in loader])
            bank['predictions']=predictions;bank.pop('fixed_predictions',None);bank['checkpoint_sha256']=sha(checkpoint)
            torch.save(bank,root/'bank.pt');status('site_error_diagnosis')
            (root/'site_error_diagnosis.json').write_text(json.dumps(site_audit(model,bank['test'],cache['graphs'],labels,device)))
            del cache,model
            prior=fit_count_prior(labels);(root/'count_prior.json').write_text(json.dumps(prior,indent=2))
            status('validation_scoring');configs={str(i):c for i,c in enumerate(CONFIGS)}
            report=evaluate(bank,bank['val'],labels,prior,configs,bank['single_indices'],device,out)
            best=max(configs,key=lambda name:report[name]['overall']['MRR'])
            selection=dict(config=configs[best],validation=report[best],all_trials=report,checkpoint_sha256=bank['checkpoint_sha256'],
                selection='MRR on 64 fixed-hash validation queries per source; no test choice',query_count=len(bank['val']))
            (root/'selection.json').write_text(json.dumps(selection,indent=2))
        else:
            if not (root.parent/'frozen_selection.json').exists():raise RuntimeError('global_model_selection_not_frozen')
            bank=torch.load(root/'bank.pt',map_location='cpu',weights_only=False,mmap=True)
            selection=json.loads((root/'selection.json').read_text());prior=json.loads((root/'count_prior.json').read_text())
            status('test_scoring');report=evaluate(bank,bank['test'],labels,prior,dict(likelihood=CONFIGS[0],selected=selection['config']),bank['single_indices'],device,out,True)
        (out/'summary.json').write_text(json.dumps(report,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

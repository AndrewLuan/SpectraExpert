"""功能概述：两个核种流程成功后，对同分子实验记录做预先固定的双核闭库检索。
--likelihood采用各核种val选定带宽的归一化谱似然，支持核发射与原分布模型；
--wait-for额外等待已登记退出码，不改变模型或按test选择组合权重。
主要职责：main等待两条主流程退出码0，核对验证选定权重/数据哈希，再在相同候选库
比较H-only/C-only/joint。组合为H距离/10和C距离/200的等权均值，不按test调权。
同canonical结构的H/C各取record hash最小记录；可来自不同实验条件，不声称同次测量。
候选只传图、物理特征、查询条件和零占位目标，绝不使用候选实测谱；无H或C结构
依据结构核种计数排除。未知覆盖仍采用observation.py契约，窗外信号不作缺失反证。
最多1000个test分子由SMILES hash固定，前200个做单核裁剪/缺峰压力测试。
rank_entry保守处理并列并记录结构相似度；输出summary/逐分子rank及所有证据哈希。
任何主流程失败立即停止；新目录独占创建。无网络，不改训练、标签或已有结果。
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from .model import SpecGNN
from .train import Spectra,collate
from .evaluate import distances_for_query,missing_queries,summarize
from .prepare import sha
from .emission_model import make_model
from .likelihood import density,log_score


def rank_entry(distances,true,smiles,fps):
    top=distances.topk(min(10,len(smiles)),largest=False).indices.cpu().tolist()
    rank=int((distances<=distances[true]+1e-7).sum())
    return dict(rank=rank,top10=[smiles[i] for i in top],
                top1_tanimoto=DataStructs.TanimotoSimilarity(fps[smiles[true]],fps[smiles[top[0]]]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--likelihood',action='store_true')
    p.add_argument('--wait-for',action='append',default=[])
    a=p.parse_args();base=Path(a.root).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):
        (out/'status.json').write_text(json.dumps(dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**kw),indent=2))
    protocol=dict(combination='0.5*(H_negative_log_likelihood+C_negative_log_likelihood)' if a.likelihood else '0.5*(H_observation_distance/10 + C_observation_distance/200)',
                  selection='per-nucleus checkpoint frozen by primary validation selection; no test weight selection',
                  queries='up to1000 same canonical molecular identities with both experimental nuclei; hash order',
                  candidates='all experimental structures with H>0 and C>0; measured candidate spectra never used',
                  pairing='smallest record-id/source hash per nucleus; conditions retained independently; not necessarily co-measured',
                  missing_stress='first200 paired queries; crop/drop one modality at a time',
                  development='supplementary likelihood rescoring after original diagnostics' if a.likelihood else 'primary directed-distance protocol',
                  registered_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));status('waiting_for_primary_runs')
    try:
        deadline=time.monotonic()+10800
        while True:
            codes=[base/n/'EXIT_CODE' for n in ['1H','13C']]
            if a.likelihood:codes += [base/n/'likelihood/EXIT_CODE' for n in ['1H','13C']]
            codes += [Path(path) for path in a.wait_for]
            for path in codes:
                if path.exists() and path.read_text().strip()!='0':raise RuntimeError('primary run failed: '+str(path))
            if all(path.exists() for path in codes):break
            if time.monotonic()>deadline:raise TimeoutError('primary_runs_incomplete')
            time.sleep(15)
        status('loading_frozen_models');cache_path=base/'prepared_coverage/cache.pt'
        cache=torch.load(cache_path,map_location='cpu',weights_only=False);cache_hash=sha(cache_path)
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        selected={};models={};bandwidths={}
        for nucleus in ['1H','13C']:
            frozen=json.loads((base/nucleus/'frozen_selection.json').read_text())
            if frozen['cache_sha256']!=cache_hash:raise ValueError('cache_changed')
            mode=frozen['selected_mode'];path=base/nucleus/mode/'best.pt';entry=frozen['runs'][mode]
            if sha(path)!=entry['checkpoint_sha256']:raise ValueError('checkpoint_changed')
            payload=torch.load(path,map_location='cpu',weights_only=False);conf=payload['args']
            model=make_model(conf).to(device)
            model.load_state_dict(payload['state_dict']);model.eval();models[nucleus]=model
            selected[nucleus]=dict(mode=mode,checkpoint_sha256=entry['checkpoint_sha256'])
            if a.likelihood:
                report=json.loads((base/nucleus/'likelihood/summary.json').read_text())
                bandwidths[nucleus]=report[mode]['selection']['selected']['sigma']
        experimental=[r for r in cache['rows'] if r['domain']=='experimental']
        candidate={r['smiles']:r for r in experimental if r['physics'][0]>0 and r['physics'][1]>0}
        smiles=sorted(candidate);candidates=[candidate[s] for s in smiles];index={s:i for i,s in enumerate(smiles)}
        test={}
        for r in experimental:
            if r['split']!='test' or r['smiles'] not in index:continue
            bucket=test.setdefault(r['smiles'],{});nucleus=r['nucleus']
            key=hashlib.sha256((r['record_id']+r['source']).encode()).hexdigest()
            if nucleus not in bucket or key<bucket[nucleus][0]:bucket[nucleus]=(key,r)
        paired=[(s,{n:x[1] for n,x in rows.items()}) for s,rows in test.items() if '1H' in rows and '13C' in rows]
        paired.sort(key=lambda item:hashlib.sha256(item[0].encode()).hexdigest());paired=paired[:1000]
        if not paired:raise ValueError('no_paired_test_molecules')
        fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        fps={s:fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles}
        predictions={}
        with torch.inference_mode():
            for nucleus,model in models.items():
                conditions=set()
                for s,rs in paired:
                    c=np.array(rs[nucleus]['cond']).copy();c[0]=0
                    conditions.add(tuple(float(x) for x in c) if model.adapter is not None else (0.,0.,0.,0.))
                for cond in conditions:
                    status('predicting_candidates',nucleus=nucleus,condition=list(cond),candidates=len(candidates))
                    rows=[dict(r,q=np.zeros(128,np.float32),source='candidate_structure_only',domain='experimental',cond=np.array(cond,np.float32),
                               observation={'kind':'unknown','windows':[]}) for r in candidates]
                    loader=DataLoader(Spectra(rows,cache['graphs'],nucleus),batch_size=256,num_workers=2,collate_fn=collate)
                    pred=torch.cat([model(b.to(device)).cpu() for b in loader]).to(device)
                    predictions[nucleus,cond]=pred
        if a.likelihood:
            for (nucleus,cond),pred in list(predictions.items()):
                predictions[nucleus,cond]=density(pred,nucleus,bandwidths[nucleus])
        scorer=log_score if a.likelihood else distances_for_query
        results=[];stress=[];status('ranking',queries=len(paired),candidates=len(candidates))
        with torch.inference_mode():
            for j,(s,rs) in enumerate(paired):
                ds={};ps={}
                for nucleus,r in rs.items():
                    c=np.array(r['cond']).copy();c[0]=0
                    cond=tuple(float(x) for x in c) if models[nucleus].adapter is not None else (0.,0.,0.,0.)
                    ps[nucleus]=predictions[nucleus,cond];ds[nucleus]=scorer(ps[nucleus],r,nucleus)
                record=dict(smiles=s,record_ids={n:r['record_id'] for n,r in rs.items()},
                            sources={n:r['source'] for n,r in rs.items()},observation={n:r['observation'] for n,r in rs.items()})
                for name,distance in [('1H',ds['1H']),('13C',ds['13C']),('joint',.5*(ds['1H']+ds['13C']))]:
                    record[name]=rank_entry(distance,index[s],smiles,fps)
                results.append(record)
                if j<200:
                    for nucleus in ['1H','13C']:
                        other='13C' if nucleus=='1H' else '1H'
                        for variant,r in missing_queries(rs[nucleus],nucleus):
                            d=scorer(ps[nucleus],r,nucleus)
                            stress.append(dict(smiles=s,variant=nucleus+'_'+variant,
                                               **rank_entry(.5*(d+ds[other]),index[s],smiles,fps)))
                if (j+1)%100==0:print(json.dumps({'ranked':j+1,'total':len(paired)}),flush=True)
        with (out/'predictions.jsonl').open('x') as f:
            for r in results:f.write(json.dumps(r)+'\n')
        with (out/'missing_stress.jsonl').open('x') as f:
            for r in stress:f.write(json.dumps(r)+'\n')
        report=dict(selected=selected,scoring='likelihood' if a.likelihood else 'directed',bandwidths=bandwidths,paired_test_molecules=len(paired),candidate_count=len(smiles),cache_sha256=cache_hash,
                    candidate_structure_sha256=hashlib.sha256('\n'.join(smiles).encode()).hexdigest(),
                    metrics={name:summarize([r[name] for r in results]) for name in ['1H','13C','joint']},
                    missing_stress={v:summarize([r for r in stress if r['variant']==v]) for v in sorted({r['variant'] for r in stress})})
        (out/'summary.json').write_text(json.dumps(report,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

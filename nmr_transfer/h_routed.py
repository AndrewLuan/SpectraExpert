"""功能概述：按观测格式分离¹H积分谱与稀疏原子标注，并在val缺测变体上选参数。
normal_selection复用三个模型已保存的同组val候选排名；积分谱在固定β网格选择部分
传输，其余记录使用观测似然。模型仍只按val MRR选，不按test的分源结果选择权重。
calibrate_missing对同320条val施加原有裁窗/整峰丢失，分别选择积分谱的评分参数；
原子标注始终不被解释为积分。明确缺峰标志来自输入/受控变体，不来自真实结构。
main在所有选择写入frozen_selection后，才用h_eval复评原1000条查询及压力变体，
随后调用h_joint复评原792条双核查询。目录独占，旧统一评分的结果完整保留。
这属于查看上一轮开发结果后的观测模型修订，明确报告为开发复评，不称全新独立测试。
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from .h_eval import evaluate,LazyFP
from .h_diagnose import cond_key
from .h_labels import make_key
from .h_score import CONFIGS,attach_observation,score_features,select_score
from .likelihood import density,log_score
from .joint import rank_entry
from .evaluate import summarize,missing_queries


def normal_selection(root):
    reports={}
    for variant in ['original','presence','site']:
        d=root/variant;bank=torch.load(d/'bank.pt',map_location='cpu',weights_only=False,mmap=True)
        by_id={(r['record_id'],r['smiles'],r['source']):r for r in bank['val']}
        trials={i:[json.loads(line) for line in (d/'val'/(str(i)+'_predictions.jsonl')).read_text().splitlines()] for i in range(len(CONFIGS))}
        metrics={}
        for i in range(len(CONFIGS)):
            mixed=[]
            for baseline,candidate in zip(trials[0],trials[i]):
                q=by_id[baseline['record_id'],baseline['smiles'],baseline['source']]
                mixed.append(candidate if q['measure']=='integral_weighted' else baseline)
            metrics[str(i)]=summarize(mixed)
        best=max(metrics,key=lambda key:metrics[key]['MRR']);reports[variant]=dict(config=CONFIGS[int(best)],metrics=metrics[best],trials=metrics)
    selected=max(reports,key=lambda key:reports[key]['metrics']['MRR'])
    return selected,reports


@torch.inference_mode()
def calibrate_missing(bank,labels,prior,device,out):
    ids=bank['single_indices'];smiles=[bank['smiles'][i] for i in ids.tolist()];lookup={s:i for i,s in enumerate(smiles)};fps=LazyFP()
    counts=bank['physics'][ids,0].to(device)*40;pred={c:q[ids].to(device) for c,q in bank['predictions'].items()}
    kernels={c:density(q,'1H',.2) for c,q in pred.items()};records={v:{str(i):[] for i in range(len(CONFIGS))} for v in ['frequency_crop','peak_dropout']}
    for j,r in enumerate(bank['val']):
        query=attach_observation(r,labels.get(make_key(r),{}));cond=cond_key(r)
        for variant,rr in missing_queries(query,'1H'):
            if r['measure']=='integral_weighted':features=score_features(pred[cond],kernels[cond],counts,rr,prior)
            else:features=dict(likelihood=log_score(kernels[cond],rr,'1H'))
            for i,config in enumerate(CONFIGS):
                cfg=config if r['measure']=='integral_weighted' else CONFIGS[0]
                records[variant][str(i)].append(dict(record_id=r['record_id'],smiles=r['smiles'],**rank_entry(select_score(features,cfg),lookup[r['smiles']],smiles,fps)))
        if (j+1)%100==0:print(json.dumps(dict(validation_missing=j+1)),flush=True)
    choices={};report={}
    for variant,trials in records.items():
        metrics={i:summarize(rs) for i,rs in trials.items()};best=max(metrics,key=lambda i:metrics[i]['MRR'])
        choices[variant]=CONFIGS[int(best)];report[variant]=dict(config=choices[variant],metrics=metrics[best],trials=metrics)
        with (out/(variant+'_val_predictions.jsonl')).open('x') as f:
            for r in trials[best]:f.write(json.dumps(r)+'\n')
    return choices,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();base=Path(a.base);previous=Path(a.evaluation);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        status('validation_routing');selected,normal=normal_selection(previous)
        root=out/selected;root.mkdir();(root/'bank.pt').symlink_to(previous/selected/'bank.pt')
        prior=json.loads((previous/selected/'count_prior.json').read_text());(root/'count_prior.json').write_text(json.dumps(prior))
        bank=torch.load(root/'bank.pt',map_location='cpu',weights_only=False,mmap=True)
        labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        device=torch.device('cuda');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
        status('validation_missing',selected_variant=selected);choices,missing=calibrate_missing(bank,labels,prior,device,out)
        config=dict(kind='observation_router',full_or_unknown=normal[selected]['config'],**choices)
        frozen=dict(selected_variant=selected,normal_validation=normal,missing_validation=missing,config=config,
                    development='observation-format routing revision after earlier development results; all numerical choices use validation only; not a new independent test')
        (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));(root/'selection.json').write_text(json.dumps(dict(config=config)))
        status('test');test=root/'test';test.mkdir()
        report=evaluate(bank,bank['test'],labels,prior,dict(likelihood=CONFIGS[0],selected=config),bank['single_indices'],device,test,True)
        (test/'summary.json').write_text(json.dumps(report,indent=2));(test/'EXIT_CODE').write_text('0\n')
        del bank;torch.cuda.empty_cache();status('joint')
        with (out/'joint.log').open('x') as f:
            r=subprocess.run([sys.executable,'-u','-m','nmr_transfer.h_joint','--base',str(base),'--evaluation',str(out),'--out',str(out/'joint')],cwd=out/'source',stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('routed joint failed')
        status('complete',selected_variant=selected);(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

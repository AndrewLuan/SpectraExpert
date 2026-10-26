"""功能概述：组织¹H继续训练对照与位点监督实验，每个实验独立保存源码和日志。
main先用人工结构/标签执行两轮DDP训练冒烟，再在同一旧cache/同一初始权重上正式训练。
presence保留原观测目标；site启用h_objective。两者使用相同重放、轮数、随机种子及
零H结构排除规则；真实数据不重写。源快照用于避免运行中开发影响实验。
run_stage保存明确退出码；仅训练与smoke成功后写EXIT_CODE=0，检索复评另行保存。
依赖train/h_labels侧表和既有核发射checkpoint；NCCL用已验证的SHM传输。
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
from .data import graph,physical_features,quantiles
from .h_labels import make_key


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True)
    p.add_argument('--variant',choices=['presence','site'],required=True);p.add_argument('--gpus',required=True);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--labels')
    a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,OMP_NUM_THREADS='4',NCCL_P2P_DISABLE='1')
    def run_stage(stage,args):
        (out/'status.json').write_text(json.dumps(dict(stage=stage)))
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(len(a.gpus.split(','))),'-m','nmr_transfer.train',*args]
        with (out/(stage+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=out/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/(stage+'.exit_code')).write_text(str(r.returncode)+'\n')
        if r.returncode:raise RuntimeError(stage+' failed')
    try:
        synthetic=out/'synthetic';synthetic.mkdir();rows=[];annotations={};graphs={}
        for i,s in enumerate(['CO','CCO','CCN','COC']*8):
            g=graphs.setdefault(s,graph(s));z=g.x[:,:54].argmax(1)+1;src,tgt=g.edge_index
            parents=tgt[(z[src]==1)&(z[tgt]!=1)].tolist();groups={}
            for parent in parents:groups.setdefault(parent,[]).append(float(parent+1))
            for domain in ['experimental','theory']:
                r=dict(smiles=s,source='synthetic',record_id=str(i),domain=domain,nucleus='1H',split='train' if i<24 else 'val',
                       physics=physical_features(s),cond=np.zeros(4,np.float32),q=quantiles([v for vs in groups.values() for v in vs]),observation=dict(kind='unknown',windows=[]))
                rows.append(r);annotations[make_key(r)]=dict(groups=groups,full_atom_labels=True)
        torch.save(dict(rows=rows,graphs=graphs),synthetic/'cache.pt');torch.save(annotations,synthetic/'labels.pt')
        (synthetic/'domain_shift.json').write_text('{"fits":{}}')
        common=['--nucleus','1H','--mode','domain','--model','atom','--init',str(base/'atom_emission/1H/domain/best.pt'),
                '--h-objective',a.variant,'--workers','2','--batch-size','128','--patience','6']
        run_stage('smoke',['--cache',str(synthetic/'cache.pt'),'--h-labels',str(synthetic/'labels.pt'),'--out',str(out/'smoke'),
                           '--epochs','2',*common])
        run_stage('train',['--cache',str(base/'prepared_coverage/cache.pt'),'--h-labels',a.labels or str(base/'hydrogen_analysis/labels/annotations.pt'),
                           '--out',str(out/'train'),'--epochs',str(a.epochs),*common])
        (out/'status.json').write_text(json.dumps(dict(stage='training_complete')));(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        (out/'status.json').write_text(json.dumps(dict(stage='failed',error=repr(exc))));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

"""功能概述：补齐验证集领先 MLP 的容量、增强、观测分支和教师锚配对消融。

main 复用已冻结2048训练特征和原320/1024tune，不再生成特征或读取封存集。
两个seed各比较matching/observation、无增强、共享分支、强锚，联合增强MLP为
已有控制。使用相同训练代码/初始化/结构抽样；只并发两个小型重排作业以与本轮
GNN训练共享既有GPU，不调整其他进程。命令经技能审查，保存PID、日志与退出码。
选型仍用固定320/1344和1024/1344的MRR，完全不接触verify/test或外部API。
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def arms():
    return [dict(name=f'physical_mlp_{kind}_s{seed}',owner=f'physical_base_s{seed}',seed=seed,
        features=kind if kind in ('matching','observation') else 'all',anchor=1. if kind=='strong_anchor' else .1,
        augment=kind!='no_augmentation',shared=kind=='shared',architecture='mlp')
        for seed in (193,211) for kind in ('matching','observation','no_augmentation','shared','strong_anchor')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--gate',required=True)
    args=p.parse_args();root=Path(args.root);out=root/'mlp_ablation';out.mkdir(exist_ok=False)
    started=time.monotonic();active=[]
    def status(stage,**kw):
        temp=out/'status.tmp';temp.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**kw)))
        temp.replace(out/'status.json')
    def run_jobs(jobs,stage):
        for offset in range(0,len(jobs),2):
            active.clear()
            for gpu,(name,cmd) in enumerate(jobs[offset:offset+2]):
                gate=subprocess.run(['bash',args.gate,'screen-cmd',shlex.join(cmd)],capture_output=True,text=True)
                if gate.returncode or gate.stdout.strip()!='ok':raise ValueError('mlp_ablation_command_refused')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                with (out/(stage+'_'+name+'.log')).open('x') as stream:
                    process=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active.append((name,process))
            (out/(stage+f'_wave{offset//2}_pids.json')).write_text(json.dumps([dict(name=n,pid=p.pid,gpu=i,command=jobs[offset+i][1])
                for i,(n,p) in enumerate(active)],indent=2))
            while True:
                codes={n:p.poll() for n,p in active};status(stage,wave=offset//2,exit_codes=codes)
                if any(c not in (None,0) for c in codes.values()):raise RuntimeError('mlp_ablation_job_failed:'+str(codes))
                if all(c==0 for c in codes.values()):break
                time.sleep(5)
            active.clear()
    try:
        if (root/'calibration/EXIT_CODE').read_text().strip()!='0':raise ValueError('calibration_required')
        (out/'plan.json').write_text(json.dumps(dict(arms=arms(),control='calibration/physical_mlp_s193,s211',
            selection='fixed original320+tune1024 microMRR; no heldout access',external_api_calls=0),indent=2))
        jobs=[]
        for arm in arms():
            cmd=[sys.executable,'-m','nmr_transfer.h_global_ranker','--stage','fit','--train',
                str(root/'calibration/features'/arm['owner']/'train/features.pt'),'--out',str(out/'training'/arm['name']),
                '--features',arm['features'],'--architecture','mlp','--seed',str(arm['seed']),'--anchor',str(arm['anchor'])]
            if arm['augment']:cmd+=['--augment']
            if arm['shared']:cmd+=['--shared']
            jobs.append((arm['name'],cmd))
        run_jobs(jobs,'train');jobs=[]
        for arm in arms():
            for cohort in ('original_val','tune'):
                cmd=[sys.executable,'-m','nmr_transfer.h_global_ranker','--stage','train_val','--checkpoint',
                    str(out/'training'/arm['name']/'best.pt'),'--cache',
                    str(root/'calibration/features'/arm['owner']/cohort/'features.pt'),
                    '--out',str(out/'evaluation'/arm['name']/cohort)]
                jobs.append((arm['name']+'_'+cohort,cmd))
        run_jobs(jobs,'evaluate');results={}
        for arm in arms():
            data={c:json.loads((out/'evaluation'/arm['name']/c/'summary.json').read_text()) for c in ('original_val','tune')}
            alpha=str(data['tune']['alpha'])
            combined=sum(data[c]['metrics'][alpha]['overall']['MRR']*n for c,n in [('original_val',320),('tune',1024)])/1344
            results[arm['name']]=dict(arm=arm,validation=data,combined_MRR=combined)
            control=json.loads((root/'calibration/training'/f"physical_mlp_s{arm['seed']}"/'summary.json').read_text())
            training=json.loads((out/'training'/arm['name']/'summary.json').read_text())
            if training['group_exposure_sha256']!=control['group_exposure_sha256']:
                raise ValueError('mlp_ablation_group_exposure_drift')
        (out/'summary.json').write_text(json.dumps(dict(results=results,external_api_calls=0),indent=2))
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,process in active:
            if process.poll() is None:process.terminate()
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':
    main()

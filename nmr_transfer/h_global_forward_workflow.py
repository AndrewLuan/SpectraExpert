"""功能概述：在校准研究后执行 SpecGNN 排序目标×教师锚的八臂共性研究。

main 等待本轮calibration结束，先真实数据smoke，再对物理分组布局两seed跑四臂。
同一seed的forward/rank/anchor/both使用相同初始化、原始训练抽样和负例曝光。
保留epoch8/16/24和前向best，按实际张量去重后统一完整候选预测及train宽度拟合，
只对原320与1024tune检索评分。全部数值选择仍在train/val，封存verify与test不访问。
run_jobs 对每条命令做技能安全检查并记录PID/日志/退出码；只管理本轮子进程。
新训练/快照/候选bank独占保存，不改旧输出、原始数据、候选或生产配置，无外部API。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch

from .prepare import sha


def arms():
    return [dict(name=f'physical_{kind}_s{seed}',seed=seed,owner=f'physical_base_s{seed}',
                 rank=kind in ('rank','both'),anchor=kind in ('anchor','both'))
            for seed in (193,211) for kind in ('forward','rank','anchor','both')]


def signature(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    h=hashlib.sha256()
    for name,value in sorted(saved['state_dict'].items()):h.update(name.encode());h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('root','base','previous','gate'):p.add_argument('--'+name,required=True)
    args=p.parse_args();root=Path(args.root);out=root/'forward';out.mkdir(exist_ok=False)
    active=[];started=time.monotonic()
    def status(stage,**kw):
        tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**kw)));tmp.replace(out/'status.json')
    def run_jobs(jobs,stage):
        for wave in range(0,len(jobs),8):
            active.clear()
            for gpu,(name,cmd) in enumerate(jobs[wave:wave+8]):
                gate=subprocess.run(['bash',args.gate,'screen-cmd',shlex.join(cmd)],capture_output=True,text=True)
                if gate.returncode or gate.stdout.strip()!='ok':raise ValueError('global_forward_command_refused')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                with (out/(stage+'_'+name+'.log')).open('x') as stream:process=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active.append((name,process))
            (out/(stage+f'_wave{wave//8}_pids.json')).write_text(json.dumps([dict(name=n,pid=p.pid,gpu=i,command=jobs[wave+i][1],screened=True)
                for i,(n,p) in enumerate(active)],indent=2))
            while True:
                codes={n:p.poll() for n,p in active};status(stage,wave=wave//8,exit_codes=codes)
                if any(c not in (0,None) for c in codes.values()):raise RuntimeError('global_forward_stage_failed:'+str(codes))
                if all(c==0 for c in codes.values()):break
                time.sleep(5)
            active.clear()
    try:
        status('waiting_calibration');deadline=time.monotonic()+18000
        while not (root/'calibration/EXIT_CODE').exists():
            if time.monotonic()>deadline:raise TimeoutError('calibration_wait_budget_exhausted')
            time.sleep(15)
        if (root/'calibration/EXIT_CODE').read_text().strip()!='0':raise ValueError('calibration_failed_before_forward')
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        if len(devices.splitlines())<8 or any(int(line.split(',')[1])>512 for line in devices.splitlines()[:8]):
            raise RuntimeError('forward_requires_eight_idle_GPUs; inspect ownership before scheduling')
        base=Path(args.base);previous=Path(args.previous);prepared=base/'hydrogen_analysis/condition_response/solvent_resolved_prepared'
        review=base.parent/'nmr_assignment_training_20261020/review_artifact'
        (out/'plan.json').write_text(json.dumps(dict(arms=arms(),epochs=24,samples=32768,rank_weight=.005,anchor_weight=.5,
            checkpoint_epochs=[8,16,24],selection='original320 + fixed1024tune microMRR; same fixed mean/scoring ablations',
            new_theory_pretraining=False,extra_targeted_support=False,external_api_calls=0),indent=2))
        def training(arm,smoke=False):
            owner=previous/'study/evaluation'/arm['owner'];init=previous/'study/training'/arm['owner']/'train/best.pt'
            cmd=[sys.executable,'-m','nmr_transfer.h_global_forward_train','--base',str(base),'--prepared',str(prepared),
                '--cohorts',str(root/'cohorts'),'--init',str(init),'--bank',str(owner/'bank.pt'),
                '--features',str(root/'calibration/features'/arm['owner']/'train/features.pt'),'--review',str(review),
                '--out',str(out/'smoke' if smoke else out/'training'/arm['name']/'train'),'--seed',str(arm['seed'])]
            if arm['rank']:cmd+=['--rank']
            if arm['anchor']:cmd+=['--anchor']
            if smoke:cmd+=['--smoke']
            return arm['name'],cmd
        run_jobs([training(arms()[-1],True)],'smoke')
        run_jobs([training(a) for a in arms()],'train')
        audits={a['name']:json.loads((out/'training'/a['name']/'train/summary.json').read_text()) for a in arms()}
        for seed in (193,211):
            paired=[audits[a['name']] for a in arms() if a['seed']==seed]
            for field in ('forward_exposure_sha256','negative_exposure_sha256'):
                if len({r[field] for r in paired})!=1:raise ValueError('global_forward_paired_exposure_drift:'+field)
        (out/'paired_exposure.json').write_text(json.dumps(audits,indent=2))
        variants=[];aliases={}
        for arm in arms():
            seen={}
            for label in ('epoch_8','epoch_16','epoch_24','best'):
                source=out/'training'/arm['name']/'train'/(label+'.pt');sig=signature(source);name=arm['name']+'_'+label
                if sig in seen:aliases[name]=seen[sig];continue
                seen[sig]=name;folder=out/'evaluation_training'/name/'train';folder.mkdir(parents=True)
                os.link(source,folder/'best.pt');variants.append(dict(name=name,arm=arm,label=label,checkpoint_sha256=sha(source)))
        (out/'variants.json').write_text(json.dumps(dict(variants=variants,aliases=aliases),indent=2))
        jobs=[]
        for v in variants:
            cmd=[sys.executable,'-m','nmr_transfer.h_joint_evaluate','--base',str(base),'--prepared',str(prepared),
                '--training',str(out/'evaluation_training'),'--review',str(review),'--mode',v['name'],'--stage','val',
                '--out',str(out/'evaluation'/v['name'])]
            jobs.append((v['name'],cmd))
        run_jobs(jobs,'predict')
        jobs=[]
        for v in variants:
            bank=out/'evaluation'/v['name']/'bank.pt'
            cmd=[sys.executable,'-m','nmr_transfer.h_global_bank','--bank',str(bank),'--cohorts',str(root/'cohorts'),
                '--prepared',str(prepared),'--out',str(out/'extensions'/v['name'])]
            jobs.append((v['name'],cmd))
        run_jobs(jobs,'extend')
        jobs=[]
        for v in variants:
            bank=out/'evaluation'/v['name']/'bank.pt'
            for cohort in ('original_val','tune'):
                cmd=[sys.executable,'-m','nmr_transfer.h_global_scoring','--base',str(base),'--bank',str(bank),
                    '--cohorts',str(root/'cohorts'),'--extension',str(out/'extensions'/v['name']/'extension.pt'),
                    '--widths',str(bank.parent/'val/width_fit.json'),'--cohort',cohort,
                    '--out',str(out/'scoring'/v['name']/cohort)]
                jobs.append((v['name']+'_'+cohort,cmd))
        run_jobs(jobs,'score')
        report={v['name']:{c:json.loads((out/'scoring'/v['name']/c/'summary.json').read_text()) for c in ('original_val','tune')}
                for v in variants}
        (out/'summary.json').write_text(json.dumps(dict(variants=variants,aliases=aliases,results=report),indent=2))
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,process in active:
            if process.poll() is None:process.terminate()
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':
    main()

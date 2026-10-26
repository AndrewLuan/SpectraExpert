"""功能概述：编排三seed、四种解冻容量的匹配训练与固定检索复评。
main等待上一轮条件模型选择，冻结初始checkpoint和源码，先跑全GINE烟雾检查。
每seed的四模式独占四GPU、相同曝光量；seed顺序预先声明79/83/89。若距离
用户八小时截止不足30分钟，不再启动下一seed，明确记录尚未启动的变体；已启动
seed完整训练。是否启动仅由时钟决定，不读取测试或性能。
training/EXIT_CODE仅代表训练阶段；全流程EXIT_CODE在h_condeval全部同库同集
验证/测试结束后才写。逐模型/流程日志、PID、完成/未启动列表和失败码均保存。
输出独占，不修改旧缓存、数据、参数或候选库；无外部API调用。
"""
import argparse
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .h_backbonetrain import BACKBONE_MODES
from .h_gapworkflow import resolve_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--reference-evaluation',required=True);p.add_argument('--out',required=True);p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--deadline-utc',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True);training=out/'training';training.mkdir();cutoff=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00'))-timedelta(minutes=30)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**kw)))
    jobs=[];completed=[];not_started=[]
    try:
        status('waiting_reference');path=Path(a.reference_evaluation)/'EXIT_CODE';deadline=time.monotonic()+14400
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('reference_timeout')
            time.sleep(10)
        if path.read_text().strip()!='0':raise RuntimeError('reference_failed')
        checkpoint=resolve_checkpoint(Path(a.base),Path(a.reference_evaluation));(out/'initial_checkpoint.json').write_text(json.dumps(dict(path=str(checkpoint),seeds=[79,83,89],modes=BACKBONE_MODES,deadline_utc=a.deadline_utc)))
        def cmd(mode,seed,folder):return [sys.executable,'-u','-m','nmr_transfer.h_backbonetrain','--base',a.base,'--prepared',a.prepared,'--init',str(checkpoint),'--out',str(folder),'--mode',mode,'--seed',str(seed)]
        status('smoke')
        with (out/'smoke.log').open('x') as f:
            result=subprocess.run(cmd('all',79,out/'smoke')+['--smoke'],cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus.split(',')[0],OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('smoke_failed')
        if len(a.gpus.split(','))!=4:raise ValueError('four_GPUs_required')
        for seed in [79,83,89]:
            variants=[mode+'_s'+str(seed) for mode in BACKBONE_MODES]
            if datetime.now(timezone.utc)>=cutoff:not_started+=variants;continue
            jobs=[]
            for mode,variant,gpu in zip(BACKBONE_MODES,variants,a.gpus.split(',')):
                root=training/variant;root.mkdir();f=(root/'train.log').open('x');proc=subprocess.Popen(cmd(mode,seed,root/'train'),cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT);jobs.append((variant,proc,f))
            status('training',seed=seed,completed=completed,pids={name:p.pid for name,p,_ in jobs})
            for name,p,f in jobs:
                result=p.wait();f.close()
                if result:raise RuntimeError(name+' failed')
            completed+=variants;(training/'variants.json').write_text(json.dumps(dict(completed=completed,not_started=not_started)))
        (training/'variants.json').write_text(json.dumps(dict(completed=completed,not_started=not_started,reason='clock-only launch cutoff; no performance-based skip')))
        if not completed:raise RuntimeError('no_seed_started_before_cutoff')
        (training/'EXIT_CODE').write_text('0\n');status('evaluation',completed=completed,not_started=not_started)
        with (out/'evaluation.log').open('x') as f:
            result=subprocess.run([sys.executable,'-u','-m','nmr_transfer.h_condeval','--base',a.base,'--prepared',a.prepared,'--training',str(training),'--out',str(out/'evaluation'),'--gpus',a.gpus,'--variants',','.join(completed),'--reference-evaluation',a.reference_evaluation],cwd=out/'source',stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('evaluation_failed')
        status('complete',completed=completed,not_started=not_started);(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,p,_ in jobs:
            if p.poll() is None:p.terminate()
        if not (training/'EXIT_CODE').exists():(training/'EXIT_CODE').write_text('1\n')
        status('failed',error=repr(exc),completed=completed,not_started=not_started);(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

"""功能概述：编排0/1/3/6额外上下文层的两seed匹配训练和同库复评。
初始权重固定为已完成control_s79，仅该组旧GINE与公共表示缓存逐张量相同。
先对六层做GPU smoke，再按预声明101/103顺序、每seed四GPU40轮相同曝光量训练。
新增seed仅受用户截止前30分钟时钟约束，不按性能跳过。复评把完整三seed
骨干实验的验证优胜者作为参照，仍原320-val选择、1000/792 test和冻结C控制。
源码、PID、完成/未启动清单、阶段退出码全部独占保存；失败只终止本流程子任务。
不改旧数据/缓存/参数，不调用外部服务；依赖h_deeptrain和h_condeval的模型工厂。
"""
import argparse
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .h_deep_context import DEPTHS


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--init',required=True);p.add_argument('--reference-evaluation',required=True);p.add_argument('--out',required=True);p.add_argument('--gpus',default='4,5,6,7');p.add_argument('--deadline-utc',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True);training=out/'training';training.mkdir();cutoff=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00'))-timedelta(minutes=30)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**kw)))
    jobs=[];completed=[];not_started=[]
    def cmd(depth,seed,folder):return [sys.executable,'-u','-m','nmr_transfer.h_deeptrain','--base',a.base,'--prepared',a.prepared,'--init',a.init,'--out',str(folder),'--depth',str(depth),'--seed',str(seed)]
    try:
        if len(a.gpus.split(','))!=4:raise ValueError('four_GPUs_required')
        status('smoke')
        with (out/'smoke.log').open('x') as f:result=subprocess.run(cmd(6,101,out/'smoke')+['--smoke'],cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus.split(',')[0],OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('smoke_failed')
        for seed in [101,103]:
            variants=['depth'+str(depth)+'_s'+str(seed) for depth in DEPTHS]
            if datetime.now(timezone.utc)>=cutoff:not_started+=variants;continue
            jobs=[]
            for depth,variant,gpu in zip(DEPTHS,variants,a.gpus.split(',')):
                root=training/variant;root.mkdir();f=(root/'train.log').open('x');proc=subprocess.Popen(cmd(depth,seed,root/'train'),cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT);jobs.append((variant,proc,f))
            status('training',seed=seed,completed=completed,pids={name:p.pid for name,p,_ in jobs})
            for name,p,f in jobs:
                result=p.wait();f.close()
                if result:raise RuntimeError(name+' failed')
            completed+=variants;(training/'variants.json').write_text(json.dumps(dict(completed=completed,not_started=not_started)))
        (training/'variants.json').write_text(json.dumps(dict(completed=completed,not_started=not_started,reason='clock-only launch cutoff')))
        if not completed:raise RuntimeError('no_context_seed_started')
        (training/'EXIT_CODE').write_text('0\n');status('evaluation',completed=completed,not_started=not_started)
        with (out/'evaluation.log').open('x') as f:result=subprocess.run([sys.executable,'-u','-m','nmr_transfer.h_condeval','--base',a.base,'--prepared',a.prepared,'--training',str(training),'--out',str(out/'evaluation'),'--gpus',a.gpus,'--variants',','.join(completed),'--reference-evaluation',a.reference_evaluation],cwd=out/'source',stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('context_evaluation_failed')
        status('complete',completed=completed,not_started=not_started);(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,p,_ in jobs:
            if p.poll() is None:p.terminate()
        if not (training/'EXIT_CODE').exists():(training/'EXIT_CODE').write_text('1\n')
        status('failed',error=repr(exc),completed=completed,not_started=not_started);(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

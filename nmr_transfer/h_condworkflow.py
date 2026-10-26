"""功能概述：持久运行四个等容量条件残差训练，先验证缓存与短训练再做正式训练。
worker为control/solvent/ph/local分别执行两轮小样本冒烟和固定25轮训练，保存各阶段
日志与退出码。main等待prepared成功后，以指定四GPU启动独立worker；源代码快照
和随机种子在启动前固定。各消融使用同一缓存、采样预算、监督、初始化与评分协议。
失败明确终止该流程，不覆盖旧输出或静默重试。只管理自己启动的子进程，不改用户
其他任务；train/test的检索评估由后续h_condeval执行。
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .h_condmodel import MODES


def worker(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        for stage,extra in [('smoke',['--smoke']),('train',[])]:
            status(stage);cmd=[sys.executable,'-u','-m','nmr_transfer.h_condtrain','--base',a.base,'--prepared',a.prepared,'--out',str(out/stage),'--mode',a.mode,'--epochs',str(a.epochs),'--samples',str(a.samples),'--seed',str(a.seed),*extra]
            with (out/(stage+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            (out/(stage+'.exit_code')).write_text(str(r.returncode)+'\n')
            if r.returncode:raise RuntimeError(stage+' failed')
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--out',required=True);p.add_argument('--gpus',default='1,2,3,4');p.add_argument('--epochs',type=int,default=25);p.add_argument('--samples',type=int,default=65536);p.add_argument('--seed',type=int,default=31);p.add_argument('--mode',choices=MODES);a=p.parse_args()
    if a.mode:return worker(a)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    processes=[]
    try:
        status('waiting_preparation');deadline=time.monotonic()+1800;exit_path=Path(a.prepared)/'EXIT_CODE'
        while not exit_path.exists():
            if time.monotonic()>deadline:raise TimeoutError('preparation_timeout')
            time.sleep(5)
        if exit_path.read_text().strip()!='0':raise RuntimeError('preparation_failed')
        gpus=a.gpus.split(',')
        if len(gpus)!=len(MODES):raise ValueError('four_GPUs_required')
        for mode,gpu in zip(MODES,gpus):
            cmd=[sys.executable,'-u','-m','nmr_transfer.h_condworkflow','--base',a.base,'--prepared',a.prepared,'--out',str(out/mode),'--mode',mode,'--epochs',str(a.epochs),'--samples',str(a.samples),'--seed',str(a.seed)]
            f=(out/(mode+'.log')).open('x');proc=subprocess.Popen(cmd,cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT);processes.append((mode,proc,f))
        status('training',pids={mode:proc.pid for mode,proc,_ in processes})
        for mode,proc,f in processes:
            code=proc.wait();f.close()
            if code:raise RuntimeError(mode+' failed')
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,proc,_ in processes:
            if proc.poll() is None:proc.terminate()
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

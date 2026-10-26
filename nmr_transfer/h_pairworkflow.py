"""功能概述：首轮条件消融后，用ph/local各自最佳权重做有/无条件对比损失的匹配续训。
main等待上轮四组训练结束，冻结源码后启动ph_replay/ph_pairs/local_replay/local_pairs，
四组共享继续训练预算、抽样与优化器，配对权重分别为0/1；首轮训练权重不改写。
所有过程写入独占目录、日志、状态与退出码；失败停止自己启动的子任务，不静默重试。
训练后的模型/历史参照均由h_condeval在原320条val选取，再评原同库测试。
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


VARIANTS=['ph_replay','ph_pairs','local_replay','local_pairs']


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--initial-training',required=True);p.add_argument('--out',required=True);p.add_argument('--gpus',default='0,5,6,7');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    jobs=[]
    try:
        status('waiting_initial_training');path=Path(a.initial_training)/'EXIT_CODE';deadline=time.monotonic()+10800
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('initial_training_timeout')
            time.sleep(10)
        if path.read_text().strip()!='0':raise RuntimeError('initial_training_failed')
        for variant,gpu in zip(VARIANTS,a.gpus.split(',')):
            mode=variant.split('_')[0];root=out/variant;root.mkdir();weight='1' if variant.endswith('pairs') else '0'
            cmd=[sys.executable,'-u','-m','nmr_transfer.h_condpairs','--base',a.base,'--prepared',a.prepared,'--init',str(Path(a.initial_training)/mode/'train/best.pt'),'--out',str(root/'train'),'--pair-weight',weight]
            f=(root/'train.log').open('x');proc=subprocess.Popen(cmd,cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT);jobs.append((variant,proc,f))
        status('training',pids={name:proc.pid for name,proc,_ in jobs})
        for name,proc,f in jobs:
            result=proc.wait();f.close()
            if result:raise RuntimeError(name+' failed')
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,proc,_ in jobs:
            if proc.poll() is None:proc.terminate()
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

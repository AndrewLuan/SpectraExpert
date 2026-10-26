"""功能概述：编排理论校准/重放强度/甲基联合监督六组同曝光量继续训练。
先运行最复杂分支的一轮smoke，成功后六GPU各运行一个预声明分支，所有初始
参数和实验样本序列相同。保存源码快照、PID、日志和退出码，失败停止本流程子任务。
仅训练与实验val选epoch；后续h_condeval使用固定检索val选择，再复评原候选和测试。
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .h_replaytrain import VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--init',required=True);p.add_argument('--out',required=True);p.add_argument('--gpus',default='0,1,2,3,4,6');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    jobs=[]
    def command(variant,path):return [sys.executable,'-u','-m','nmr_transfer.h_replaytrain','--base',a.base,'--prepared',a.prepared,'--init',a.init,'--out',str(path),'--variant',variant]
    try:
        status('smoke')
        with (out/'smoke.log').open('x') as f:
            result=subprocess.run(command('cal0_group',out/'smoke')+['--smoke'],cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus.split(',')[0],OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('smoke_failed')
        if len(a.gpus.split(','))!=len(VARIANTS):raise ValueError('GPU_count_mismatch')
        for variant,gpu in zip(VARIANTS,a.gpus.split(',')):
            root=out/variant;root.mkdir();f=(root/'train.log').open('x');proc=subprocess.Popen(command(variant,root/'train'),cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT);jobs.append((variant,proc,f))
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

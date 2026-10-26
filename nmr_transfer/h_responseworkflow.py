"""功能概述：等待重放消融结束后，对验证优胜者启动四组锚定条件响应训练。
resolve_checkpoint仅由已冻结的检索val选择定位初始权重；不读取测试表现来选择。
四分支共用初始模型、25轮预算和seed；先完成一轮smoke，失败保留日志并退出。
输出独占源码快照、PID/状态/退出码；后续h_condeval按原同库同查询协议执行复评。
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .h_response import RESPONSE_MODES
from .h_gapworkflow import resolve_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--reference-evaluation',required=True);p.add_argument('--out',required=True);p.add_argument('--gpus',default='0,1,2,3');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    jobs=[]
    try:
        status('waiting_reference');path=Path(a.reference_evaluation)/'EXIT_CODE';deadline=time.monotonic()+10800
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('reference_timeout')
            time.sleep(10)
        if path.read_text().strip()!='0':raise RuntimeError('reference_failed')
        checkpoint=resolve_checkpoint(Path(a.base),Path(a.reference_evaluation));(out/'initial_checkpoint.json').write_text(json.dumps(dict(path=str(checkpoint))))
        def cmd(mode,folder):return [sys.executable,'-u','-m','nmr_transfer.h_responsetrain','--base',a.base,'--prepared',a.prepared,'--init',str(checkpoint),'--out',str(folder),'--mode',mode]
        status('smoke')
        with (out/'smoke.log').open('x') as f:
            result=subprocess.run(cmd('solvent',out/'smoke')+['--smoke'],cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus.split(',')[0],OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('smoke_failed')
        for mode,gpu in zip(RESPONSE_MODES,a.gpus.split(',')):
            root=out/mode;root.mkdir();f=(root/'train.log').open('x');proc=subprocess.Popen(cmd(mode,root/'train'),cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT);jobs.append((mode,proc,f))
        status('training',pids={name:p.pid for name,p,_ in jobs})
        for name,p,f in jobs:
            result=p.wait();f.close()
            if result:raise RuntimeError(name+' failed')
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,p,_ in jobs:
            if p.poll() is None:p.terminate()
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

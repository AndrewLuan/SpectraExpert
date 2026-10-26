"""功能概述：自动完成三个¹H模型的val评分、全局冻结、test对照及固定双核复评。
main复制源码并启动GPU0/1/2上的val任务；各任务等待对应训练成功退出，依赖失败即停。
三个val结果齐备后才按MRR选择模型并写frozen_selection.json，然后并行执行三个test。
最后只为val选中的模型执行h_joint，保留¹³C权重与组合权重。禁止test选模型或重跑择优。
所有子任务日志/退出码独立，输出新目录；源码快照不受后续开发影响，无外部服务调用。
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True)
    p.add_argument('--reuse-evaluation')
    a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    train=json.loads((base/'hydrogen_analysis/training_runs.json').read_text())
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    def launch(name,command,gpu):
        log=open(out/(name+'.log'),'x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4')
        return subprocess.Popen(command,cwd=out/'source',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    def collect(jobs):
        while jobs:
            for name,process in list(jobs.items()):
                code=process.poll()
                if code is not None:
                    (out/(name+'.exit_code')).write_text(str(code)+'\n')
                    if code:raise RuntimeError(name+' failed; inspect its preserved log')
                    del jobs[name]
            if jobs:time.sleep(10)
    try:
        status('validation');jobs={};variants=['original','presence','site']
        for gpu,variant in enumerate(variants):
            command=[sys.executable,'-u','-m','nmr_transfer.h_eval','--base',str(base),'--out',str(out/variant),'--variant',variant,'--stage','val']
            entry=train['presence' if variant=='original' else variant]
            command+=['--wait-for',entry['path']+'/EXIT_CODE']
            if variant!='original':command+=['--checkpoint',entry['path']+'/train/best.pt']
            if a.reuse_evaluation:
                reuse=Path(a.reuse_evaluation)/variant/'bank.pt'
                if reuse.exists():command+=['--reuse-bank',str(reuse)]
            jobs[variant+'_val']=launch(variant+'_val',command,gpu)
        collect(jobs)
        selections={v:json.loads((out/v/'selection.json').read_text()) for v in variants}
        selected=max(variants,key=lambda v:selections[v]['validation']['overall']['MRR'])
        frozen=dict(selected_variant=selected,variants=selections,selection='highest MRR on same source-balanced validation queries; all test workers start afterward')
        (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));status('test',selected_variant=selected)
        for gpu,variant in enumerate(variants):
            command=[sys.executable,'-u','-m','nmr_transfer.h_eval','--base',str(base),'--out',str(out/variant),'--variant',variant,'--stage','test']
            jobs[variant+'_test']=launch(variant+'_test',command,gpu)
        collect(jobs);status('joint',selected_variant=selected)
        command=[sys.executable,'-u','-m','nmr_transfer.h_joint','--base',str(base),'--evaluation',str(out),'--out',str(out/'joint')]
        collect({'joint':launch('joint',command,0)})
        status('complete',selected_variant=selected);(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

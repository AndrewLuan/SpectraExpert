"""功能概述：在 JJ 持久执行整体检索首轮固定均值消融，不自动打开测试集。

main 先为四个布局/seed 基线补条件 bank，执行真实数据 smoke，再用八张空闲 GPU
分别完成原 320-val 与哈希 tune。run_jobs 每个命令先经技能 command_gate 审查，
记录 PID、日志、退出码与阶段状态；只管理本轮进程，失败保留现场并终止本轮子任务。
全部验证结束写轻量 summary 供 Codex 选择下一轮；不运行 verify/test、不自动选型、
不改生产配置。资源启动前由调用者核实，入口不会访问 API 或下载任何数据。
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('root','base','previous','gate'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args(); root=Path(args.root); out=root/'phase_a'; out.mkdir(exist_ok=False)
    started=time.monotonic(); active=[]
    def status(stage,**kw):
        tmp=out/'status.tmp'; tmp.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**kw)))
        tmp.replace(out/'status.json')
    def run_jobs(jobs,stage):
        for wave in range(0,len(jobs),8):
            active.clear()
            for gpu,(name,cmd) in enumerate(jobs[wave:wave+8]):
                gate=subprocess.run(['bash',args.gate,'screen-cmd',shlex.join(cmd)],capture_output=True,text=True)
                if gate.returncode or gate.stdout.strip()!='ok':raise ValueError('global_derived_command_refused:'+name)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                with (out/(stage+'_'+name+'.log')).open('x') as stream:
                    process=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active.append((name,process))
            (out/(stage+'_pids.json')).write_text(json.dumps([dict(name=n,pid=p.pid,gpu=i,command=jobs[wave+i][1],screened=True)
                for i,(n,p) in enumerate(active)],indent=2))
            while True:
                codes={n:p.poll() for n,p in active}; status(stage,wave=wave//8,exit_codes=codes)
                if any(c not in (0,None) for c in codes.values()):raise RuntimeError('global_stage_failed:'+str(codes))
                if all(c==0 for c in codes.values()):break
                time.sleep(5)
            active.clear()
    try:
        previous=Path(args.previous); base=Path(args.base); prepared=base/'hydrogen_analysis/condition_response/solvent_resolved_prepared'
        models=[f'{layout}_base_s{seed}' for layout in ('local','physical') for seed in (193,211)]
        extension=[]
        for name in models:
            bank=previous/'study/evaluation'/name/'bank.pt'
            extension.append((name,[sys.executable,'-m','nmr_transfer.h_global_bank','--bank',str(bank),
                '--cohorts',str(root/'cohorts'),'--prepared',str(prepared),'--out',str(out/'extensions'/name)]))
        run_jobs(extension,'extend')
        def score_job(name,cohort,limit=False):
            bank=previous/'study/evaluation'/name/'bank.pt'
            cmd=[sys.executable,'-m','nmr_transfer.h_global_scoring','--base',str(base),'--bank',str(bank),
                '--cohorts',str(root/'cohorts'),'--extension',str(out/'extensions'/name/'extension.pt'),
                '--widths',str(bank.parent/'val/width_fit.json'),'--cohort',cohort,
                '--out',str(out/('smoke' if limit else 'evaluation')/name/cohort)]
            if limit:cmd+=['--limit','5']
            return (name+'_'+cohort,cmd)
        run_jobs([score_job('local_base_s211','original_val',True)],'smoke')
        run_jobs([score_job(n,c) for n in models for c in ('original_val','tune')],'score')
        report={name:{c:json.loads((out/'evaluation'/name/c/'summary.json').read_text()) for c in ('original_val','tune')}
                for name in models}
        (out/'summary.json').write_text(json.dumps(report,indent=2)); status('complete'); (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,process in active:
            if process.poll() is None:process.terminate()
        status('failed',error=repr(exc)); (out/'EXIT_CODE').write_text('1\n'); raise


if __name__=='__main__':
    main()

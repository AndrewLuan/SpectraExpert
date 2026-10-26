"""功能概述：只用固定train/val运行代表标签和峰簇容量的四个全库评分臂。

main 等待只含训练信息的reporting_artifact，先做4条真实验证smoke，再对两seed
物理布局分别评原320与1024tune。每次最多两个GPU作业，保留命令审查、PID、日志、
真实退出码和预声明320/1344、1024/1344合并MRR。不访问verify/test或外部API。
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
    for name in ('root','base','previous','gate'):p.add_argument('--'+name,required=True)
    args=p.parse_args();root=Path(args.root);out=root/'observation';out.mkdir(exist_ok=False)
    started=time.monotonic();active=[]
    def status(stage,**kw):
        temp=out/'status.tmp';temp.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**kw)))
        temp.replace(out/'status.json')
    def run_jobs(jobs,stage):
        for offset in range(0,len(jobs),2):
            active.clear()
            for gpu,(name,cmd) in enumerate(jobs[offset:offset+2]):
                gate=subprocess.run(['bash',args.gate,'screen-cmd',shlex.join(cmd)],capture_output=True,text=True)
                if gate.returncode or gate.stdout.strip()!='ok':raise ValueError('observation_workflow_command_refused')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu+2),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                with (out/(stage+'_'+name+'.log')).open('x') as stream:
                    process=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active.append((name,process))
            (out/(stage+f'_wave{offset//2}_pids.json')).write_text(json.dumps([dict(name=n,pid=p.pid,gpu=i+2,command=jobs[offset+i][1])
                for i,(n,p) in enumerate(active)],indent=2))
            while True:
                codes={n:p.poll() for n,p in active};status(stage,wave=offset//2,exit_codes=codes)
                if any(c not in (None,0) for c in codes.values()):raise RuntimeError('observation_workflow_job_failed:'+str(codes))
                if all(c==0 for c in codes.values()):break
                time.sleep(5)
            active.clear()
    try:
        deadline=time.monotonic()+3600;status('waiting_training_artifact')
        while not all((root/'reporting_artifact'/f).exists() for f in ('table.pt','models.json','manifest.json')):
            if time.monotonic()>deadline:raise TimeoutError('reporting_artifact_wait_exhausted')
            time.sleep(10)
        def job(owner,cohort,smoke=False):
            bank=Path(args.previous)/'study/evaluation'/owner/'bank.pt'
            cmd=[sys.executable,'-m','nmr_transfer.h_global_observation_study','--base',args.base,'--bank',str(bank),
                '--widths',str(bank.parent/'val/width_fit.json'),'--extension',str(root/'phase_a/extensions'/owner/'extension.pt'),
                '--cohorts',str(root/'cohorts'),'--artifact',str(root/'reporting_artifact'),'--cohort',cohort,
                '--out',str(out/'smoke' if smoke else out/owner/cohort)]
            if smoke:cmd+=['--limit','4']
            return owner+'_'+cohort,cmd
        run_jobs([job('physical_base_s211','original_val',True)],'smoke')
        owners=[f'physical_base_s{s}' for s in (193,211)]
        run_jobs([job(o,c) for o in owners for c in ('original_val','tune')],'score')
        results={}
        for owner in owners:
            reports={c:json.loads((out/owner/c/'summary.json').read_text()) for c in ('original_val','tune')}
            combined={config['name']:sum(reports[c]['metrics'][config['name']]['overall']['MRR']*n
                for c,n in [('original_val',320),('tune',1024)])/1344 for config in reports['tune']['configs']}
            results[owner]=dict(reports=reports,combined_MRR=combined)
        (out/'summary.json').write_text(json.dumps(dict(results=results,external_api_calls=0),indent=2))
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,process in active:
            if process.poll() is None:process.terminate()
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

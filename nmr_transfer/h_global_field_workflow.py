"""功能概述：按独立审查的固定计划运行两seed场强容量验证，保持GPU分区。

main 只执行plan中四个h_global_field_capacity开发集命令，先检查源码/计划SHA，
等待旧forward与新报告重排结束，再在GPU0/1各有70GiB空闲时启动两波。messages
重评固定GPU4..7，因此大密度作业不会与本入口争同一显卡。每条命令经技能gate，
保存PID/日志/退出码；输出固定1344微平均及各集合指标。只管理自己的子进程，
不打开verify/test、不训练、不调用任何外部API或改写旧结果。
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .prepare import sha


def main():
    p=argparse.ArgumentParser()
    for name in ('root','plan','gate'):p.add_argument('--'+name,required=True)
    args=p.parse_args();root=Path(args.root).resolve();out=root/'field_capacity';out.mkdir(exist_ok=False)
    started=time.monotonic();active=[];plan_sha=sha(args.plan)
    def status(stage,**kw):
        temp=out/'status.tmp';temp.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**kw)))
        temp.replace(out/'status.json')
    def screen(command):
        gate=subprocess.run(['bash',args.gate,'screen-cmd',shlex.join(command)],capture_output=True,text=True)
        if gate.returncode or gate.stdout.strip()!='ok':raise ValueError('field_workflow_command_refused')
    try:
        plan=json.loads(Path(args.plan).read_text())
        if plan['state']!='prepared_not_launched' or len(plan['waves'])!=2:raise ValueError('field_workflow_wrong_plan')
        for name,digest in plan['source_sha256'].items():
            if sha(Path(__file__).with_name(Path(name).name))!=digest:raise ValueError('field_workflow_source_drift:'+name)
        deadline=time.monotonic()+18000
        while True:
            codes={name:(root/name/'EXIT_CODE').read_text().strip() if (root/name/'EXIT_CODE').exists() else None
                   for name in ('forward','observation_calibration')}
            status('waiting_heavy_parents',exit_codes=codes)
            if any(code not in (None,'0') for code in codes.values()):raise RuntimeError('field_workflow_parent_failed:'+str(codes))
            if all(code=='0' for code in codes.values()):break
            if time.monotonic()>deadline:raise TimeoutError('field_workflow_parent_wait_exhausted')
            time.sleep(15)
        for wave,jobs in enumerate(plan['waves']):
            if len(jobs)!=2 or {j['GPU'] for j in jobs}!={0,1}:raise ValueError('field_workflow_GPU_partition_changed')
            deadline=time.monotonic()+3600
            while True:
                query=['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'];screen(query)
                result=subprocess.check_output(query,text=True)
                free={int(line.split(',')[0]):int(line.split(',')[1]) for line in result.splitlines()}
                if all(free.get(gpu,0)>=70*1024 for gpu in (0,1)):break
                if time.monotonic()>deadline:raise TimeoutError('field_workflow_GPU_wait_exhausted')
                status('waiting_GPU0_1',free_MiB=free);time.sleep(15)
            active.clear()
            for job in jobs:
                cmd=job['command'];name=job['owner']+'_'+job['cohort']
                if Path(cmd[0]).resolve()!=Path(sys.executable).resolve() or cmd[1:3]!=['-m','nmr_transfer.h_global_field_capacity'] or job['cohort'] not in ('original_val','tune'):
                    raise ValueError('field_workflow_unregistered_command')
                if Path(cmd[cmd.index('--out')+1]).resolve()!=out/job['owner']/job['cohort']:
                    raise ValueError('field_workflow_output_scope_changed')
                if job['environment']['CUDA_VISIBLE_DEVICES']!=str(job['GPU']):raise ValueError('field_workflow_GPU_environment_changed')
                screen(cmd)
                env=dict(os.environ,**job['environment'])
                with (out/(name+'.log')).open('x') as stream:
                    process=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active.append((name,process))
            (out/f'wave{wave}_pids.json').write_text(json.dumps([dict(name=n,pid=p.pid,command=jobs[i]['command'],gpu=jobs[i]['GPU'])
                for i,(n,p) in enumerate(active)],indent=2))
            while True:
                codes={n:p.poll() for n,p in active};status('scoring',wave=wave,exit_codes=codes)
                if any(code not in (None,0) for code in codes.values()):raise RuntimeError('field_workflow_job_failed:'+str(codes))
                if all(code==0 for code in codes.values()):break
                time.sleep(5)
            active.clear()
        results={}
        for owner in ('physical_base_s193','physical_base_s211'):
            reports={c:json.loads((out/owner/c/'summary.json').read_text()) for c in ('original_val','tune')}
            combined={config['name']:sum(reports[c]['metrics'][config['name']]['overall']['MRR']*n
                for c,n in [('original_val',320),('tune',1024)])/1344 for config in reports['tune']['configs']}
            results[owner]=dict(reports=reports,combined_MRR=combined)
        if sha(args.plan)!=plan_sha:raise ValueError('field_workflow_plan_changed')
        (out/'summary.json').write_text(json.dumps(dict(results=results,plan_sha256=plan_sha,selection_scope='train_val_only',external_api_calls=0),indent=2))
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,process in active:
            if process.poll() is None:process.terminate()
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

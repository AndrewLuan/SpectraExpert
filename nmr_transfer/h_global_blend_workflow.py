"""功能概述：在旧粗排和新报告粗排各自的同源缓存上比较两seed增强残差混合。

main 先运行已完成的no_aug/aug MLP两seed×原320/tune，再等待新报告重排结束后
在对应新cache重复同一预声明五权重消融。每次两CPU作业，不争抢GPU，来源路径
全部显式保存。只读train选出的权重和val缓存；不开放verify/test或外部API。
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--gate',required=True)
    args=p.parse_args();root=Path(args.root);out=root/'blend';out.mkdir(exist_ok=False);started=time.monotonic();active=[]
    def status(stage,**kw):
        temp=out/'status.tmp';temp.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**kw)))
        temp.replace(out/'status.json')
    def run(jobs,stage):
        for offset in range(0,len(jobs),2):
            active.clear()
            for name,cmd in jobs[offset:offset+2]:
                gate=subprocess.run(['bash',args.gate,'screen-cmd',shlex.join(cmd)],capture_output=True,text=True)
                if gate.returncode or gate.stdout.strip()!='ok':raise ValueError('blend_workflow_command_refused')
                with (out/(stage+'_'+name+'.log')).open('x') as stream:
                    process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
                active.append((name,process))
            (out/(stage+f'_wave{offset//2}_pids.json')).write_text(json.dumps([dict(name=n,pid=p.pid,command=jobs[offset+i][1])
                for i,(n,p) in enumerate(active)],indent=2))
            while True:
                codes={n:p.poll() for n,p in active};status(stage,wave=offset//2,exit_codes=codes)
                if any(c not in (None,0) for c in codes.values()):raise RuntimeError('blend_workflow_failed:'+str(codes))
                if all(c==0 for c in codes.values()):break
                time.sleep(5)
            active.clear()
    try:
        results={}
        for stage in ('base','observation'):
            if stage=='observation':
                status('waiting_observation_calibration');deadline=time.monotonic()+18000
                while not (root/'observation_calibration/EXIT_CODE').exists():
                    if time.monotonic()>deadline:raise TimeoutError('blend_observation_wait_exhausted')
                    time.sleep(15)
                if (root/'observation_calibration/EXIT_CODE').read_text().strip()!='0':raise RuntimeError('blend_observation_parent_failed')
                if not json.loads((root/'observation_calibration/summary.json').read_text())['results']:continue
            jobs=[]
            for seed in (193,211):
                owner=f'physical_base_s{seed}'
                if stage=='base':
                    no_aug=root/'mlp_ablation/training'/f'physical_mlp_no_augmentation_s{seed}'/'best.pt'
                    aug=root/'calibration/training'/f'physical_mlp_s{seed}'/'best.pt';cache=root/'calibration/features'/owner
                else:
                    no_aug=root/'observation_calibration/training'/f'physical_obs_no_augmentation_s{seed}'/'best.pt'
                    aug=root/'observation_calibration/training'/f'physical_obs_mlp_s{seed}'/'best.pt';cache=root/'observation_calibration/features'/owner
                for cohort in ('original_val','tune'):
                    cmd=[sys.executable,'-m','nmr_transfer.h_global_ranker_blend','--stage','train_val','--no-aug-checkpoint',str(no_aug),
                        '--aug-checkpoint',str(aug),'--cache',str(cache/cohort/'features.pt'),'--out',str(out/stage/owner/cohort)]
                    jobs.append((owner+'_'+cohort,cmd))
            run(jobs,stage);results[stage]={}
            for seed in (193,211):
                owner=f'physical_base_s{seed}';reports={c:json.loads((out/stage/owner/c/'summary.json').read_text()) for c in ('original_val','tune')}
                combined={str(g):sum(reports[c]['metrics'][str(g)]['overall']['MRR']*n for c,n in [('original_val',320),('tune',1024)])/1344
                    for g in reports['tune']['gammas']}
                results[stage][owner]=dict(reports=reports,combined_MRR=combined)
            (out/(stage+'_summary.json')).write_text(json.dumps(results[stage],indent=2))
        (out/'summary.json').write_text(json.dumps(dict(results=results,external_api_calls=0),indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,process in active:
            if process.poll() is None:process.terminate()
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

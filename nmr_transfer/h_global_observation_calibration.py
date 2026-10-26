"""功能概述：将两seed共同验证选定的观测粗排与低容量重排合并做配对消融。

main 等待观测评分结束，按两seed固定1344-val微平均MRR选择一个共用配置；
不超过1e-4的改善保留count_control并停止本支。选择文件先落盘，后生成新train/val
难负例。等待GNN研究完成、四GPU空闲后smoke及正式抽取，避免大核密度作业争抢显存。
linear/MLP、增强/无增强和共享观测头对照使用同一新粗排、结构抽样与初始化。
只访问train/original_val/tune；未来verify/test由独立冻结入口处理，无外部API。
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .h_global_observation_study import configs


def arms():
    return [dict(name=f'physical_obs_{kind}_s{seed}',owner=f'physical_base_s{seed}',seed=seed,
        architecture='linear' if kind=='linear' else 'mlp',augment=kind!='no_augmentation',shared=kind=='shared')
        for seed in (193,211) for kind in ('linear','mlp','no_augmentation','shared')]


def main():
    p=argparse.ArgumentParser()
    for name in ('root','base','previous','gate'):p.add_argument('--'+name,required=True)
    args=p.parse_args();root=Path(args.root);out=root/'observation_calibration';out.mkdir(exist_ok=False)
    active=[];started=time.monotonic();devices=[]
    def status(stage,**kw):
        temp=out/'status.tmp';temp.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**kw)))
        temp.replace(out/'status.json')
    def wait_for(folder):
        status('waiting_'+folder);deadline=time.monotonic()+18000
        while not (root/folder/'EXIT_CODE').exists():
            if time.monotonic()>deadline:raise TimeoutError('observation_calibration_wait_exhausted:'+folder)
            time.sleep(15)
        if (root/folder/'EXIT_CODE').read_text().strip()!='0':raise RuntimeError('required_parent_failed:'+folder)
    def run_jobs(jobs,stage):
        for offset in range(0,len(jobs),4):
            active.clear()
            for slot,(name,cmd) in enumerate(jobs[offset:offset+4]):
                gate=subprocess.run(['bash',args.gate,'screen-cmd',shlex.join(cmd)],capture_output=True,text=True)
                if gate.returncode or gate.stdout.strip()!='ok':raise ValueError('observation_calibration_command_refused')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(devices[slot]),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                with (out/(stage+'_'+name+'.log')).open('x') as stream:
                    process=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active.append((name,process))
            (out/(stage+f'_wave{offset//4}_pids.json')).write_text(json.dumps([dict(name=n,pid=p.pid,gpu=devices[i],command=jobs[offset+i][1])
                for i,(n,p) in enumerate(active)],indent=2))
            while True:
                codes={n:p.poll() for n,p in active};status(stage,wave=offset//4,exit_codes=codes)
                if any(c not in (None,0) for c in codes.values()):raise RuntimeError('observation_calibration_job_failed:'+str(codes))
                if all(c==0 for c in codes.values()):break
                time.sleep(5)
            active.clear()
    try:
        wait_for('observation');data=json.loads((root/'observation/summary.json').read_text())
        averages={c['name']:sum(v['combined_MRR'][c['name']] for v in data['results'].values())/len(data['results']) for c in configs()}
        best=max(configs(),key=lambda c:(averages[c['name']],-c['report']-c['capacity']))
        if averages[best['name']]<=averages['count_control']+1e-4:best=configs()[0]
        choice=dict(selection_scope='train_val_only',observation_config=best,validation_MRR=averages,
            selection='average of two seeds; each fixed original320+tune1024 microMRR; tolerance1e-4 keeps control',external_api_calls=0)
        choice_path=out/'feature_choice.json';choice_path.write_text(json.dumps(choice,indent=2))
        if best['name']=='count_control':
            (out/'summary.json').write_text(json.dumps(dict(choice=choice,results={},skipped='no common validation gain')))
            status('complete_control');(out/'EXIT_CODE').write_text('0\n');return
        wait_for('forward');status('waiting_four_idle_GPUs');deadline=time.monotonic()+3600
        while len(devices)<4:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            devices=[int(line.split(',')[0]) for line in text.splitlines() if int(line.split(',')[1])<512][:4]
            if time.monotonic()>deadline:raise TimeoutError('four_idle_GPUs_unavailable')
            if len(devices)<4:time.sleep(15)
        def extract(owner,cohort,smoke=False):
            bank=Path(args.previous)/'study/evaluation'/owner/'bank.pt'
            cmd=[sys.executable,'-m','nmr_transfer.h_global_observation_extract','--base',args.base,'--bank',str(bank),
                '--widths',str(bank.parent/'val/width_fit.json'),'--extension',str(root/'phase_a/extensions'/owner/'extension.pt'),
                '--cohorts',str(root/'cohorts'),'--artifact',str(root/'reporting_artifact'),'--feature-choice',str(choice_path),
                '--cohort',cohort,'--augment','--out',str(out/'smoke' if smoke else out/'features'/owner/cohort)]
            if smoke:cmd+=['--limit','8']
            return owner+'_'+cohort,cmd
        run_jobs([extract('physical_base_s211','train',True)],'smoke')
        run_jobs([extract(f'physical_base_s{s}',c) for s in (193,211) for c in ('train','original_val','tune')],'extract')
        jobs=[]
        for arm in arms():
            cmd=[sys.executable,'-m','nmr_transfer.h_global_ranker','--stage','fit','--train',str(out/'features'/arm['owner']/'train/features.pt'),
                '--out',str(out/'training'/arm['name']),'--features','all','--architecture',arm['architecture'],'--seed',str(arm['seed'])]
            if arm['augment']:cmd+=['--augment']
            if arm['shared']:cmd+=['--shared']
            jobs.append((arm['name'],cmd))
        run_jobs(jobs,'train');jobs=[]
        for arm in arms():
            for cohort in ('original_val','tune'):
                cmd=[sys.executable,'-m','nmr_transfer.h_global_ranker','--stage','train_val','--checkpoint',str(out/'training'/arm['name']/'best.pt'),
                    '--cache',str(out/'features'/arm['owner']/cohort/'features.pt'),'--out',str(out/'evaluation'/arm['name']/cohort)]
                jobs.append((arm['name']+'_'+cohort,cmd))
        run_jobs(jobs,'evaluate');results={}
        for arm in arms():
            validation={c:json.loads((out/'evaluation'/arm['name']/c/'summary.json').read_text()) for c in ('original_val','tune')}
            alpha=str(validation['tune']['alpha'])
            combined=sum(validation[c]['metrics'][alpha]['overall']['MRR']*n for c,n in [('original_val',320),('tune',1024)])/1344
            results[arm['name']]=dict(arm=arm,validation=validation,combined_MRR=combined)
        (out/'summary.json').write_text(json.dumps(dict(choice=choice,results=results),indent=2))
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,process in active:
            if process.poll() is None:process.terminate()
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

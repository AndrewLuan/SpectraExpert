"""功能概述：执行全局难负样本特征缓存及配对低容量校准，不打开 verify/test。

main 对四个固定布局/seed 使用同一预声明 atom_count_1.0 粗排，先跑真实 train/val
smoke，再缓存 train、320-val、1024-tune 与其固定缺峰/裁窗变体。训练比较匹配特征、
观测特征、联合特征、线性/MLP、教师锚、增强和观测口径条件化；所有正例/负例均
由训练结构监督，epoch/alpha 由 train 内部结构划分选择。最后只评 train_val，
按预先固定 320/1344 与 1024/1344 查询权重汇总 MRR，独立报告各集全部指标。
run_jobs 审查命令、记录 PID/退出码；只管理自身进程，失败保留现场。不自动通过
verify，不修改测试、候选、生产配置或均值模型，无外部 API 调用。
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def arms():
    result=[]
    for layout in ('local','physical'):
        for seed in (193,211):
            owner=f'{layout}_base_s{seed}'
            for family in ('matching','observation','all'):
                result.append(dict(name=f'{layout}_{family}_s{seed}',owner=owner,seed=seed,features=family,
                    architecture='linear',anchor=.1,augment=True,shared=False))
            result.append(dict(name=f'{layout}_mlp_s{seed}',owner=owner,seed=seed,features='all',
                architecture='mlp',anchor=.1,augment=True,shared=False))
    for seed in (193,211):
        for kind in ('strong_anchor','no_augmentation','shared'):
            result.append(dict(name=f'physical_{kind}_s{seed}',owner=f'physical_base_s{seed}',seed=seed,features='all',
                architecture='linear',anchor=1. if kind=='strong_anchor' else .1,
                augment=kind!='no_augmentation',shared=kind=='shared'))
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','base','previous','gate'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args(); root=Path(args.root); out=root/'calibration'; out.mkdir(exist_ok=False)
    active=[]; started=time.monotonic()
    def status(stage,**kw):
        tmp=out/'status.tmp'; tmp.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**kw)))
        tmp.replace(out/'status.json')
    def run_jobs(jobs,stage):
        for wave in range(0,len(jobs),8):
            active.clear()
            for gpu,(name,cmd) in enumerate(jobs[wave:wave+8]):
                gate=subprocess.run(['bash',args.gate,'screen-cmd',shlex.join(cmd)],capture_output=True,text=True)
                if gate.returncode or gate.stdout.strip()!='ok':raise ValueError('global_command_refused:'+name)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                with (out/(stage+'_'+name+'.log')).open('x') as stream:
                    process=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active.append((name,process))
            (out/(stage+f'_wave{wave//8}_pids.json')).write_text(json.dumps([dict(name=n,pid=p.pid,gpu=i,command=jobs[wave+i][1],screened=True)
                for i,(n,p) in enumerate(active)],indent=2))
            while True:
                codes={n:p.poll() for n,p in active}; status(stage,wave=wave//8,exit_codes=codes)
                if any(c not in (0,None) for c in codes.values()):raise RuntimeError('global_calibration_failed:'+str(codes))
                if all(c==0 for c in codes.values()):break
                time.sleep(5)
            active.clear()
    try:
        if (root/'phase_a/EXIT_CODE').read_text().strip()!='0':raise ValueError('phase_a_not_complete')
        previous=Path(args.previous); base=Path(args.base)
        plan=dict(arms=arms(),coarse='atom_count_1.0',pool_size=512,
            coarse_reason='report-count beta1 improved all4 paired validation arms; no test selection; simpler than prototype mixture',
            selection='micro MRR over original320+fixed1024tune; weights320/1344 and1024/1344; ties1e-4 prefer alpha0/linear/fewer features',
            independent_verify='one locked1024-val after model freeze; must have DeltaMRR>0 and DeltaTop1>=0 versus original system baseline; otherwise retain baseline and report diagnostic ablations',
            inherited_validation='native validation pool was used by earlier forward-model selection; independent means sealed during this round selection',
            external_api_calls=0)
        (out/'plan.json').write_text(json.dumps(plan,indent=2))
        def extract_job(owner,cohort,smoke=False):
            bank=previous/'study/evaluation'/owner/'bank.pt'
            cmd=[sys.executable,'-m','nmr_transfer.h_global_extract','--base',str(base),'--bank',str(bank),
                '--cohorts',str(root/'cohorts'),'--extension',str(root/'phase_a/extensions'/owner/'extension.pt'),
                '--widths',str(bank.parent/'val/width_fit.json'),'--cohort',cohort,'--coarse','atom_count_1.0',
                '--augment','--out',str(out/('smoke_features' if smoke else 'features')/owner/cohort)]
            if smoke:cmd+=['--limit','32' if cohort=='train' else '4']
            return owner+'_'+cohort,cmd
        run_jobs([extract_job('local_base_s211','train',True),extract_job('local_base_s211','original_val',True)],'smoke_extract')
        smoke_train=out/'smoke_features/local_base_s211/train/features.pt'
        smoke_cmd=[sys.executable,'-m','nmr_transfer.h_global_ranker','--stage','fit','--train',str(smoke_train),
            '--out',str(out/'smoke_fit'),'--epochs','1','--samples','64','--augment']
        run_jobs([('ranker',smoke_cmd)],'smoke_fit')
        owners=sorted({a['owner'] for a in arms()})
        run_jobs([extract_job(owner,cohort) for owner in owners for cohort in ('train','original_val','tune')],'extract')
        jobs=[]
        for arm in arms():
            cmd=[sys.executable,'-m','nmr_transfer.h_global_ranker','--stage','fit','--train',str(out/'features'/arm['owner']/'train/features.pt'),
                '--out',str(out/'training'/arm['name']),'--features',arm['features'],'--architecture',arm['architecture'],
                '--seed',str(arm['seed']),'--anchor',str(arm['anchor'])]
            if arm['augment']:cmd+=['--augment']
            if arm['shared']:cmd+=['--shared']
            jobs.append((arm['name'],cmd))
        run_jobs(jobs,'train')
        jobs=[]
        for arm in arms():
            for cohort in ('original_val','tune'):
                cmd=[sys.executable,'-m','nmr_transfer.h_global_ranker','--stage','train_val',
                    '--checkpoint',str(out/'training'/arm['name']/'best.pt'),
                    '--cache',str(out/'features'/arm['owner']/cohort/'features.pt'),
                    '--out',str(out/'evaluation'/arm['name']/cohort)]
                jobs.append((arm['name']+'_'+cohort,cmd))
        run_jobs(jobs,'evaluate')
        results={}
        for arm in arms():
            data={cohort:json.loads((out/'evaluation'/arm['name']/cohort/'summary.json').read_text()) for cohort in ('original_val','tune')}
            alpha=str(data['tune']['alpha'])
            combined=sum(data[c]['metrics'][alpha]['overall']['MRR']*n for c,n in [('original_val',320),('tune',1024)])/1344
            results[arm['name']]=dict(arm=arm,validation=data,combined_MRR=combined)
        (out/'summary.json').write_text(json.dumps(dict(plan=plan,results=results),indent=2))
        status('complete'); (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,process in active:
            if process.poll() is None:process.terminate()
        status('failed',error=repr(exc)); (out/'EXIT_CODE').write_text('1\n'); raise


if __name__=='__main__':
    main()

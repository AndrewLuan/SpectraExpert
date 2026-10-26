"""功能概述：在JJ执行审核监督的配对训练及固定候选库检索消融。

plan_arms从训练语义指纹去除等效干预；优先比较legacy/corrections/domain，再对
local/physical解码布局配对，余下两GPU做local的第二个微调seed。理论预训练检查点
由既有独立理论val选出，所有新臂仅微调16轮，不把复用权重写成新的预训练。
main先写冻结plan，再运行含真实审核样本的smoke；run_stage只管理自身启动的子进程，
按GPU独立启动并等待退出码，失败保存状态并停止该批自己的子进程。状态文件原子
替换，SSH断连不影响用start_new_session启动的主流程；不重试失败或跳过失败臂。
全部val完成后核验同库/同查询和同seed抽样，再冻结最佳MRR，之后才能打开test。
最终保留每臂1000原查询及371缺峰变体、DMF诊断、输入/受保护文件SHA和完成标记。
仅比较相同旧评分器下的均值模型，生产宽度/报告模型保持原样；不宣称自动部署。
依赖h_count_curriculum/h_condinfer及冻结审核衍生物；无外部API，不修改原谱和旧结果。
"""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .prepare import sha


def plan_arms(preflight):
    modes=[];seen=set()
    for mode in ('legacy','corrections','atom','domain'):
        fingerprint=preflight['modes'][mode]['effective_supervision_sha256']
        if fingerprint not in seen:modes.append(mode);seen.add(fingerprint)
    arms=[dict(name=f'{layout}_{mode}_s193',layout=layout,mode=mode,seed=193)
          for layout in ('local','physical') for mode in modes]
    if len(arms)<=6:
        arms.extend(dict(name=f'local_{mode}_s211',layout='local',mode=mode,seed=211)
                    for mode in ('legacy','domain'))
    if len(arms)>8:raise ValueError('review_workflow_exceeds_declared_GPUs')
    return arms


def main():
    p=argparse.ArgumentParser()
    for k in ('base','study','root'):p.add_argument('--'+k,required=True)
    a=p.parse_args();base=Path(a.base);study=Path(a.study);root=Path(a.root)
    out=root/'paired_study';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();active=[]
    def read(p):return json.loads(p.read_text())
    def status(stage,**kw):
        path=out/'status.tmp';path.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**kw)))
        path.replace(out/'status.json')
    def run_stage(jobs,phase):
        active.clear()
        for gpu,(name,cmd) in enumerate(jobs):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            with (out/(phase+'_'+name+'.log')).open('x') as f:
                child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
            active.append((name,child));status(phase,started=len(active),total=len(jobs))
        (out/(phase+'_pids.json')).write_text(json.dumps([
            dict(name=name,pid=child.pid,gpu=i,command=jobs[i][1]) for i,(name,child) in enumerate(active)],indent=2))
        while True:
            codes={name:child.poll() for name,child in active};status(phase,exit_codes=codes)
            bad={name:code for name,code in codes.items() if code not in (None,0)}
            if bad:raise RuntimeError('review_stage_failed:'+json.dumps(bad))
            if all(code==0 for code in codes.values()):break
            time.sleep(5)
        active.clear()
    try:
        preflight=read(root/'supervision_preflight.json');arms=plan_arms(preflight)
        prepared=base/'hydrogen_analysis/condition_response/solvent_resolved_prepared'
        artifact=root/'review_artifact';configs={}
        for layout in ('local','physical'):
            train=study/f'preserved_study/training/preserved_{layout}_head/train'
            configs[layout]=read(train/'config.json')['args']
        def train_command(arm,smoke=False):
            cfg=configs[arm['layout']];init=study/f"preserved_study/training/preserved_{arm['layout']}_head/train/pretrain_best.pt"
            dest=out/'smoke' if smoke else out/'training'/arm['name']/'train'
            cmd=[sys.executable,'-u','-m','nmr_transfer.h_count_curriculum']
            for key in ('base','prepared','sidecar','registry','feature_counts','extra_support'):
                cmd.extend(['--'+key.replace('_','-'),str(cfg[key])])
            cmd.extend(['--init',str(init),'--out',str(dest),'--decoder','preserved','--skip-theory-training',
                        '--theory-site-orbits','--theory-selection','spectral','--source-calibration','train_affine_all',
                        '--message-mode','solvent','--site-quality','methyl_record_mask','--equivalence-mode',arm['layout'],
                        '--target-mode','physical','--epochs','16','--theory-samples','65536','--experimental-samples','32768',
                        '--seed',str(arm['seed']),'--review-artifact',str(artifact),'--review-mode',arm['mode']])
            if smoke:cmd.append('--smoke')
            return cmd
        frozen_inputs={str(f):sha(f) for f in [root/'supervision_preflight.json',artifact/'artifact.pt',artifact/'manifest.json']}
        for layout in configs:
            f=study/f'preserved_study/training/preserved_{layout}_head/train/pretrain_best.pt';frozen_inputs[str(f)]=sha(f)
        plan=dict(arms=arms,effective_aliases=preflight['effective_aliases'],theory_training_epochs=0,
                  initialization='existing theory-val selected pretrain_best; no new theory optimization',
                  experimental_epochs=16,experimental_samples_per_epoch=32768,
                  common_forward_validation=preflight['common_validation'],
                  retrieval='unchanged114982 H candidates;320 val;1000 test;371 missing variants; fixed legacy scoring',
                  selection='all paired arms evaluated on same320 val first; select MRR with1e-4 ties favor legacy; no production promotion',
                  inputs_sha256=frozen_inputs,source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},extra_API_calls=0)
        (out/'plan.json').write_text(json.dumps(plan,indent=2))
        smoke_arm=next(x for x in arms if x['layout']=='physical' and x['mode']=='domain')
        run_stage([('physical_domain',train_command(smoke_arm,True))],'smoke')
        run_stage([(arm['name'],train_command(arm)) for arm in arms],'training')
        exposure={}
        for arm in arms:
            config=read(out/'training'/arm['name']/'train/config.json');s=config['sampling_sequence_sha256']
            key=arm['seed'];current=(s['experimental'],config['experimental_queries_sha256'],config['experimental_validation_queries_sha256'])
            if key in exposure and exposure[key]!=current:raise ValueError('review_paired_exposure_or_validation_drift')
            exposure[key]=current
            if config['phase_epochs']['theory']!=0:raise ValueError('unexpected_new_theory_training')
        (out/'paired_exposure_audit.json').write_text(json.dumps(exposure,indent=2))
        evaluation=out/'evaluation';evaluation.mkdir()
        def infer(arm,stage):return [sys.executable,'-u','-m','nmr_transfer.h_condinfer',
            '--base',str(base),'--prepared',str(prepared),'--training',str(out/'training'),
            '--out',str(evaluation/arm['name']),'--mode',arm['name'],'--stage',stage]
        run_stage([(arm['name'],infer(arm,'val')) for arm in arms],'validation')
        metrics={};reference=read(study/'preserved_study/evaluation/preserved_local_head/val/summary.json')
        for arm in arms:
            summary=read(evaluation/arm['name']/'val/summary.json')
            for key in ('candidate_order_sha256','query_sha256','physics_sha256','candidates'):
                if summary[key]!=reference[key]:raise ValueError('frozen_review_validation_drift:'+key)
            metrics[arm['name']]=summary['metrics']
        order=sorted(arms,key=lambda arm:(arm['mode']!='legacy',arm['layout']!='local',arm['seed']))
        selected=order[0]['name']
        for arm in order:
            if metrics[arm['name']]['overall']['MRR']>metrics[selected]['overall']['MRR']+.0001:selected=arm['name']
        selection=dict(selected_variant=selected,validation=metrics,arms={r['name']:'training' for r in arms},
                       selection='same320 val MRR; frozen before test; no production promotion',development_test_reuse=True)
        (evaluation/'frozen_selection.json').write_text(json.dumps(selection,indent=2))
        run_stage([(arm['name'],infer(arm,'test')) for arm in arms],'test')
        tests={};old=read(study/'preserved_study/evaluation/preserved_local_head/test/summary.json')
        for arm in arms:
            summary=read(evaluation/arm['name']/'test/summary.json')
            for key in ('candidate_order_sha256','query_sha256','physics_sha256','candidates'):
                if summary[key]!=old[key]:raise ValueError('frozen_review_test_drift:'+key)
            tests[arm['name']]=dict(metrics=summary['metrics'],missing_stress=summary['missing_stress'],
                                    checkpoint_sha256=summary['checkpoint_sha256'])
        status('integrity');protected=read(base.parent/'nmr_assignment_quality_20261020/remaining_resolution/final_integrity.json')['files']
        for f,expected in protected.items():
            if sha(f)!=expected['sha256']:raise ValueError('protected_file_changed:'+f)
        for f,h in frozen_inputs.items():
            if sha(f)!=h:raise ValueError('review_training_input_changed:'+f)
        summary=dict(selection=selection,tests=tests,paired_exposure_exact=True,protected_files_exact=len(protected),
                     candidate_order_sha256=old['candidate_order_sha256'],query_sha256=old['query_sha256'],
                     production_pipeline_modified=False,extra_API_calls=0,seconds=time.monotonic()-started)
        (out/'summary.json').write_text(json.dumps(summary,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n')
        for _,child in active:
            if child.poll() is None:child.terminate()
        raise


if __name__=='__main__':main()

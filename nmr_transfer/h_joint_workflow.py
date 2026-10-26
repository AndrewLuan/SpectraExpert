"""功能概述：冻结并执行16臂条件联合研究，管理自身GPU子进程及研究证据。

arms为2布局×2seed×位移/报告2×2；同布局从上一轮审核后的同一权重继续8轮。
run_jobs每次最多8臂，失败终止该批自己的进程并保留日志，SSH断连不影响外部会话。
main先确认真实数据smoke成功并写plan，训练后核对等曝光及初始张量，再完成所有320-val
宽度/模型选择，将选择写盘后才运行固定1000-test和371压力变体。原库/查询SHA须与上轮一致。
analysis保留width=0的严格2×2对照和val选择宽度对照，按连接结构bootstrap，不以DMF调参。
末尾重核20个受保护文件及输入SHA；明确记录复用理论权重、原测试重复评估和零额外API。
目录独占；不修改生产选择、原始谱、候选、已有输出或外部项目。无自动重试/覆盖。
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .h_count_finalization import cluster_bootstrap
from .prepare import sha


def arms():
    return [dict(name=f'{layout}_{kind}_s{seed}',layout=layout,kind=kind,seed=seed,
                 shift=kind in ('shift','both'),report=kind in ('report','both'))
            for seed in (193,211) for layout in ('local','physical')
            for kind in ('base','shift','report','both')]


def tensor_signature(path):
    saved = torch.load(path,map_location='cpu',weights_only=False)
    digest = hashlib.sha256()
    for name,value in sorted(saved['state_dict'].items()):
        digest.update(name.encode())
        digest.update(value.contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--base',required=True)
    args = parser.parse_args()
    root,base = Path(args.root),Path(args.base)
    out = root/'study'
    out.mkdir(exist_ok=False)
    prepared = base/'hydrogen_analysis/condition_response/solvent_resolved_prepared'
    previous = base.parent/'nmr_assignment_training_20261020'
    review = previous/'review_artifact'
    started = time.monotonic()
    active = []
    read = lambda p:json.loads(p.read_text())
    rows = lambda p:[json.loads(line) for line in p.read_text().splitlines()]
    def status(stage,**fields):
        tmp = out/'status.tmp'
        tmp.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**fields)))
        tmp.replace(out/'status.json')
    def run_jobs(jobs,stage):
        for wave in range(0,len(jobs),8):
            active.clear()
            for gpu,(name,cmd) in enumerate(jobs[wave:wave+8]):
                env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                with (out/(stage+'_'+name+'.log')).open('x') as stream:
                    process = subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active.append((name,process))
            (out/(stage+f'_wave{wave//8}_pids.json')).write_text(json.dumps([
                dict(name=name,pid=p.pid,gpu=gpu,command=jobs[wave+gpu][1])
                for gpu,(name,p) in enumerate(active)],indent=2))
            while True:
                codes = {name:p.poll() for name,p in active}
                status(stage,wave=wave//8,exit_codes=codes)
                bad = {n:c for n,c in codes.items() if c not in (0,None)}
                if bad:raise RuntimeError('joint_stage_failed:'+json.dumps(bad))
                if all(c==0 for c in codes.values()):break
                time.sleep(5)
            active.clear()
    try:
        status('waiting_smoke')
        deadline = time.monotonic()+1800
        while not (root/'smoke/EXIT_CODE').exists():
            if time.monotonic()>deadline:raise TimeoutError('joint_smoke_timeout')
            time.sleep(5)
        if (root/'smoke/EXIT_CODE').read_text().strip()!='0':raise RuntimeError('joint_smoke_failed')
        initial = {layout:previous/f'paired_study/training/{layout}_domain_s193/train/best.pt'
                   for layout in ('local','physical')}
        hashes = {str(p):sha(p) for p in list(initial.values())+[review/'artifact.pt',review/'manifest.json',prepared/'manifest.json']}
        plan = dict(arms=arms(),inputs_sha256=hashes,epochs=8,samples_per_epoch=32768,
                    new_theory_optimization=False,initialization='same prior reviewed domain checkpoint within layout',
                    cross_solvent_constraints='train only; same structure; trustworthy sites or consistent integer integrals; no HMDB forced alignment',
                    mean_lr=3e-5,message_lr=1e-5,report_lr=1e-3,
                    primary_contrasts='fixed width0 within layout/seed; all2x2 interactions and joint intervention',
                    width_ablation='re-fit on reviewed train then0/.5/1 selected only320-val',
                    selection='all16 validation first; MRR ties within1e-4 prefer base/local/seed193; freeze before all test',
                    no_DMF_specific_shifts=True,development_test_reuse=True,extra_API_calls=0,
                    source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
        (out/'plan.json').write_text(json.dumps(plan,indent=2))
        def train(arm):
            cmd = [sys.executable,'-u','-m','nmr_transfer.h_joint_train','--base',str(base),
                   '--prepared',str(prepared),'--review',str(review),'--init',str(initial[arm['layout']]),
                   '--out',str(out/'training'/arm['name']/'train'),'--seed',str(arm['seed'])]
            if arm['shift']:cmd.append('--shift')
            if arm['report']:cmd.append('--report')
            return cmd
        run_jobs([(arm['name'],train(arm)) for arm in arms()],'training')
        exposure = {}
        signatures = {}
        pair_audits = {}
        for arm in arms():
            folder = out/'training'/arm['name']/'train'
            config = read(folder/'config.json')
            fingerprint = (config['sampling_sha256'],config['train_queries_sha256'],config['val_queries_sha256'])
            if arm['seed'] in exposure and exposure[arm['seed']]!=fingerprint:
                raise ValueError('joint_paired_exposure_drift')
            exposure[arm['seed']] = fingerprint
            signature = tensor_signature(folder/'initial.pt')
            if arm['layout'] in signatures and signatures[arm['layout']]!=signature:
                raise ValueError('joint_initial_tensor_drift')
            signatures[arm['layout']] = signature
            pair_audits[arm['name']] = config['pairs']
        (out/'paired_data_audit.json').write_text(json.dumps(dict(exposure=exposure,initial=signatures,pairs=pair_audits),indent=2))
        evaluation = out/'evaluation'
        evaluation.mkdir()
        def evaluate(arm,stage):
            return [sys.executable,'-u','-m','nmr_transfer.h_joint_evaluate','--base',str(base),
                    '--prepared',str(prepared),'--training',str(out/'training'),'--review',str(review),
                    '--out',str(evaluation/arm['name']),'--mode',arm['name'],'--stage',stage]
        run_jobs([(arm['name'],evaluate(arm,'val')) for arm in arms()],'validation')
        reference = read(previous/'paired_study/evaluation/local_domain_s193/val/summary.json')
        validation = {}
        for arm in arms():
            result = read(evaluation/arm['name']/'val/frozen_selection.json')
            for report in result['validation'].values():
                for key in ('candidate_order_sha256','query_sha256','physics_sha256','candidates'):
                    if report[key]!=reference[key]:raise ValueError('joint_validation_identity_drift:'+key)
            validation[arm['name']] = result
        preferred = sorted(arms(),key=lambda a:(a['kind']!='base',a['layout']!='local',a['seed'],a['name']))
        chosen = preferred[0]['name']
        def metric(name):
            r = validation[name]
            return r['validation'][str(r['width_lambda'])]['metrics']['overall']['MRR']
        for arm in preferred:
            if metric(arm['name'])>metric(chosen)+.0001:chosen=arm['name']
        frozen = dict(selected=chosen,validation=validation,selected_width_lambda=validation[chosen]['width_lambda'],
                      selection=plan['selection'],development_test_reuse=True)
        (evaluation/'frozen_selection.json').write_text(json.dumps(frozen,indent=2))
        run_jobs([(arm['name'],evaluate(arm,'test')) for arm in arms()],'test')
        tests = {arm['name']:read(evaluation/arm['name']/'test/summary.json') for arm in arms()}
        reference = read(previous/'paired_study/evaluation/local_domain_s193/test/summary.json')
        predictions = {}
        missing = {}
        dmf = {}
        for arm in arms():
            name = arm['name']
            for report in tests[name].values():
                for key in ('candidate_order_sha256','query_sha256','physics_sha256','candidates'):
                    if report[key]!=reference[key]:raise ValueError('joint_test_identity_drift:'+key)
            folder = evaluation/name/'test/width_0.0'
            predictions[name] = rows(folder/'predictions.jsonl')
            missing[name] = rows(folder/'missing_predictions.jsonl')
            diag = read(evaluation/name/'test/site_diagnostics.json')
            ranks = {r['record_id']:r['rank'] for r in predictions[name]}
            dmf[name] = [dict(r,rank=ranks.get(r['record_id'])) for r in diag['DMF']]
        status('analysis')
        contrasts = {}
        for layout in ('local','physical'):
            for seed in (193,211):
                for new,old in [('shift','base'),('report','base'),('both','shift'),('both','report'),('both','base')]:
                    n,o = f'{layout}_{new}_s{seed}',f'{layout}_{old}_s{seed}'
                    contrasts[n+'__vs__'+o] = dict(
                        overall=cluster_bootstrap(predictions[n],predictions[o]),
                        missing={v:cluster_bootstrap([r for r in missing[n] if r['variant']==v],
                            [r for r in missing[o] if r['variant']==v],True) for v in ('frequency_crop','peak_dropout')})
        protected = read(base.parent/'nmr_assignment_quality_20261020/remaining_resolution/final_integrity.json')['files']
        status('protected_integrity')
        for path,expected in protected.items():
            if sha(path)!=expected['sha256']:raise ValueError('joint_protected_file_changed:'+path)
        for path,expected in hashes.items():
            if sha(path)!=expected:raise ValueError('joint_input_changed:'+path)
        result = dict(selection=frozen,tests=tests,contrasts=contrasts,DMF=dmf,paired_data_exact=True,
                      protected_files_exact=len(protected),source_counts=dict(collections.Counter(r['source'] for r in predictions[chosen])),
                      seconds=time.monotonic()-started,extra_API_calls=0,production_pipeline_modified=False)
        (out/'summary.json').write_text(json.dumps(result,indent=2))
        status('complete')
        (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc))
        (out/'EXIT_CODE').write_text('1\n')
        for _,process in active:
            if process.poll() is None:process.terminate()
        raise


if __name__=='__main__':
    main()

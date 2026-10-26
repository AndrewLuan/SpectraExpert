"""功能概述：比较固定 epoch24 的分组头微调与溶剂消息微调，研究整体位移表达误差。

arms 预声明两seed×forward/both四臂；forward不启用排序/教师，both沿用.005/.5，
其余初始化、94571条训练记录、24epochs×32768抽样与旧head同kind完全配对。
gpu_memory 只允许物理GPU4..7；训练每卡至少20GiB可用。重预测/补预测/评分先等
root/forward与root/observation_calibration结束，再等每卡70GiB可用，防止密度
作业并发耗尽显存。依赖失败留档原因，资源满足后仍评本轮模型，不重跑或终止他人。
run_jobs 每波重查显存，先经root/command_gate.sh筛查每条子命令，记录PID、
日志、退出码并仅管理自己的进程。真实数据smoke通过后才启动四臂正式训练。
assert_training_pair 核对原head抽样/负例曝光、初始化、数据与冻结检查；暖启动
另按实际state_dict签名核对，训练scope/参数前缀与运行源码SHA持久化。
之后复用h_joint_evaluate(stage=val)拟合训练宽度，再h_global_bank补预测与
h_global_scoring原21项评分；只读original_val/tune，候选顺序和数量保持114982。
仅epoch24进入复评，8/16/best只归档；scope_comparison按同评分逐项报告四个
检索指标差值和固定320/1024权重的合并指标；旧head失败导致的对照缺口单独留档。
所有新文件独占写入root/messages；失败保留日志并终止本入口所启动的子进程。
不读取verify/test排名，不改原谱、原bank、候选或构造器，不调用外部API。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch

from .h_global_forward_train import TRAIN_PREFIXES
from .h_global_forward_workflow import signature
from .h_global_protocol import CRITICAL_SOURCES
from .h_global_scoring import configs
from .prepare import sha


GPU_IDS=(4,5,6,7)
MIN_FREE_MIB=20*1024
MIN_HEAVY_FREE_MIB=70*1024
COHORTS={'original_val':320,'tune':1024}
METRICS=('top1','top5','top10','MRR')
SOURCE_NAMES=tuple(dict.fromkeys(CRITICAL_SOURCES+(
    'h_global_forward_train.py','h_global_forward_loss.py','h_global_forward_workflow.py',
    'h_global_message_workflow.py','h_joint_evaluate.py','h_preserved_group_model.py',
    'h_equivalence_model.py','h_solvent_message.py','h_deep_context.py','emission_model.py',
    'h_replaytrain.py','h_reviewed_supervision.py','h_count_curriculum.py','h_condmodel.py',
    'h_condtrain.py','train.py',
)))


def arms():
    return [dict(name=f'physical_messages_{kind}_s{seed}',seed=seed,owner=f'physical_base_s{seed}',
                 reference=f'physical_{kind}_s{seed}',rank=kind=='both',anchor=kind=='both')
            for seed in (193,211) for kind in ('forward','both')]


def gpu_memory(text,minimum=MIN_FREE_MIB):
    """解析nvidia-smi实际free MiB；不因其他GPU的占用拒绝指定的4..7。"""
    free={}
    for line in text.splitlines():
        index,amount=(int(value.strip()) for value in line.split(','))
        if index in free:raise ValueError('global_message_duplicate_gpu_index')
        free[index]=amount
    if any(free.get(gpu,-1)<minimum for gpu in GPU_IDS):
        raise RuntimeError('global_message_requires_'+str(minimum)+'MiB_free_on_GPUs4_7:'+str(free))
    return {gpu:free[gpu] for gpu in GPU_IDS}


def assert_training_pair(message,head,message_summary,head_summary):
    """核对scope之外的训练协议及曝光，兼容旧head没有显式scope元数据。"""
    if message.get('scope')!='messages' or head.get('scope','head')!='head':
        raise ValueError('global_message_scope_pair_invalid')
    if message.get('train_prefixes')!=list(TRAIN_PREFIXES['messages']):
        raise ValueError('global_message_train_prefixes_drift')
    for key in ('seed','epochs','samples','rank','anchor'):
        if message['args'][key]!=head['args'][key]:raise ValueError('global_message_training_args_drift:'+key)
    for key in ('init_sha256','input_sha256','model_args','review','quality','train_queries_sha256','validation_queries_sha256','pairs',
                'rank_weight','anchor_weight','rank_batch','rank_every_steps'):
        if message[key]!=head[key]:raise ValueError('global_message_training_protocol_drift:'+key)
    if message['full_train_records']!=94571 or message['args']['epochs']!=24 or message['args']['samples']!=32768:
        raise ValueError('global_message_training_budget_drift')
    for key in ('forward_exposure_sha256','negative_exposure_sha256'):
        if message_summary[key]!=head_summary[key]:raise ValueError('global_message_paired_exposure_drift:'+key)
    if not all(s['frozen_core_exact'] for s in (message_summary,head_summary)):
        raise ValueError('global_message_frozen_core_check_failed')


def scope_comparison(head,message,grid,cohort):
    """相同查询/完整候选/21评分下逐项比较，不在该函数选择模型或评分。"""
    if head['configs']!=grid or message['configs']!=grid:raise ValueError('global_message_scoring_grid_drift')
    for key in ('candidate_count','candidate_order_sha256','query_sha256'):
        if head[key]!=message[key]:raise ValueError('global_message_comparison_input_drift:'+key)
    if message['candidate_count']!=114982:raise ValueError('global_message_candidate_count_drift')
    result={}
    for choice in grid:
        name=choice['name'];a=head['metrics'][name];b=message['metrics'][name]
        if a['overall']['queries']!=COHORTS[cohort] or b['overall']['queries']!=COHORTS[cohort]:
            raise ValueError('global_message_query_count_drift:'+cohort)
        if a['by_source'].keys()!=b['by_source'].keys():raise ValueError('global_message_source_cohort_drift')
        result[name]=dict(head=a,messages=b,delta={key:b['overall'][key]-a['overall'][key] for key in METRICS},
            by_source_delta={source:{key:b['by_source'][source][key]-a['by_source'][source][key] for key in METRICS}
                             for source in a['by_source']})
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','base','previous','gate'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--head-root',help='原head研究目录；默认root/forward')
    args=parser.parse_args();root=Path(args.root).resolve();out=root/'messages';out.mkdir(exist_ok=False)
    head_root=Path(args.head_root).resolve() if args.head_root else root/'forward'
    active=[];started=time.monotonic();source_dir=Path(__file__).parent;sources={};gate_sha=None;dependency_failures={}
    def status(stage,**kw):
        tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**kw)));tmp.replace(out/'status.json')
    def write_record(record):
        path=Path(record['record']);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(path)
    def screen(cmd):
        if sha(args.gate)!=gate_sha:raise ValueError('global_message_command_gate_changed')
        check=subprocess.run(['bash',args.gate,'screen-cmd',shlex.join(cmd)],capture_output=True,text=True)
        if check.returncode or check.stdout.strip()!='ok':raise ValueError('global_message_command_refused:'+shlex.join(cmd))
    def launch(name,cmd,stage,gpu=None):
        screen(cmd);folder=out/'commands'/stage;folder.mkdir(parents=True,exist_ok=True)
        log=folder/(name+'.log');path=folder/(name+'.json')
        env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with log.open('x') as stream:process=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record=dict(name=name,pid=process.pid,gpu=gpu,command=cmd,log=str(log),record=str(path),screened=True,
            started_unix=time.time(),exit_code=None,source_sha256=sources)
        active.append((process,record));write_record(record)
        return process,record
    def finish(process,record):
        code=process.poll()
        if code is not None and record['exit_code'] is None:
            record.update(exit_code=code,finished_unix=time.time());write_record(record)
            Path(record['record']).with_suffix('.EXIT_CODE').write_text(str(code)+'\n')
        return code
    def check_memory(stage,minimum=MIN_FREE_MIB,wait=False):
        attempt=0;deadline=time.monotonic()+18000
        while True:
            if any(sha(source_dir/name)!=value for name,value in sources.items()):
                raise ValueError('global_message_runtime_source_changed')
            process,record=launch('gpu_memory_'+str(attempt),['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],stage)
            process.wait();code=finish(process,record);active.remove((process,record))
            if code:raise RuntimeError('global_message_gpu_memory_query_failed')
            try:free=gpu_memory(Path(record['log']).read_text(),minimum)
            except RuntimeError as exc:
                if not wait:raise
                if time.monotonic()>deadline:raise TimeoutError('global_message_heavy_gpu_memory_wait_budget_exhausted') from exc
                status('waiting_gpu_memory',next_stage=stage,minimum_free_MiB=minimum,error=str(exc));attempt+=1;time.sleep(15);continue
            status('memory_checked',next_stage=stage,free_MiB=free,minimum_free_MiB=minimum)
            return free
    def wait_heavy_dependencies(stage):
        dependencies={name:root/name for name in ('forward','observation_calibration')};deadline=time.monotonic()+18000
        while True:
            codes={name:(folder/'EXIT_CODE').read_text().strip() if (folder/'EXIT_CODE').exists() else None for name,folder in dependencies.items()}
            status('waiting_heavy_dependencies',next_stage=stage,exit_codes=codes)
            if all(code is not None for code in codes.values()):break
            if time.monotonic()>deadline:raise TimeoutError('global_message_heavy_dependency_wait_budget_exhausted')
            time.sleep(15)
        for name,code in codes.items():
            if code!='0':
                path=dependencies[name]/'status.json'
                dependency_failures[name]=dict(exit_code=code,status_text=path.read_text() if path.exists() else 'no status.json')
        (out/'dependencies.json').write_text(json.dumps(dict(exit_codes=codes,failures=dependency_failures,
            policy='do not restart or kill dependencies; own heavy stages require70GiB free onGPUs4..7'),indent=2))
    def run_jobs(jobs,stage,heavy=False):
        if heavy:wait_heavy_dependencies(stage)
        for offset in range(0,len(jobs),len(GPU_IDS)):
            wave=offset//len(GPU_IDS);check_memory(stage+'_wave'+str(wave),MIN_HEAVY_FREE_MIB if heavy else MIN_FREE_MIB,wait=heavy)
            for gpu,(name,cmd) in zip(GPU_IDS,jobs[offset:offset+len(GPU_IDS)]):launch(name,cmd,stage,gpu)
            while True:
                codes={r['name']:finish(p,r) for p,r in active};status(stage,wave=wave,exit_codes=codes)
                if any(code not in (0,None) for code in codes.values()):raise RuntimeError('global_message_stage_failed:'+str(codes))
                if all(code==0 for code in codes.values()):break
                time.sleep(5)
            active.clear()
    try:
        if Path(args.gate).resolve()!=root/'command_gate.sh':raise ValueError('global_message_requires_root_command_gate')
        sources={name:sha(source_dir/name) for name in SOURCE_NAMES};gate_sha=sha(args.gate)
        grid=configs()
        if len(grid)!=21:raise ValueError('global_message_requires_original_21_scoring_configs')
        base=Path(args.base);previous=Path(args.previous);prepared=base/'hydrogen_analysis/condition_response/solvent_resolved_prepared'
        review=base.parent/'nmr_assignment_training_20261020/review_artifact'
        reference={}
        for arm in arms():
            folder=head_root/'training'/arm['reference']/'train'
            if (folder/'EXIT_CODE').read_text().strip()!='0':raise ValueError('global_message_reference_training_not_complete')
            cfg=json.loads((folder/'config.json').read_text());summary=json.loads((folder/'summary.json').read_text())
            if cfg.get('scope','head')!='head':raise ValueError('global_message_reference_is_not_head_scope')
            expected=dict(seed=arm['seed'],rank=arm['rank'],anchor=arm['anchor'],epochs=24,samples=32768)
            if any(cfg['args'][key]!=value for key,value in expected.items()):
                raise ValueError('global_message_reference_training_args_drift:'+arm['reference'])
            init=previous/'study/training'/arm['owner']/'train/best.pt'
            if cfg['init_sha256']!=sha(init):raise ValueError('global_message_reference_initial_checkpoint_drift')
            if not summary['frozen_core_exact']:raise ValueError('global_message_reference_frozen_core_check_failed')
            reference[arm['name']]=dict(config=cfg,summary=summary,folder=str(folder),
                input_sha256={name:sha(folder/name) for name in ('config.json','summary.json','initial.pt','epoch_24.pt')})
        (out/'plan.json').write_text(json.dumps(dict(arms=arms(),scope='messages',train_prefixes=list(TRAIN_PREFIXES['messages']),
            source_sha256=sources,gate_sha256=gate_sha,head_root=str(head_root),train_records=94571,epochs=24,samples=32768,
            rank_weight=.005,anchor_weight=.5,checkpoint_epochs=[8,16,24],evaluated_checkpoint='epoch_24',scoring_configs=grid,
            selection='same kind/seed head epoch24; original320 + fixed1024tune only; report all21 scoring pairs',
            cuda_devices=list(GPU_IDS),minimum_train_free_MiB=MIN_FREE_MIB,minimum_heavy_free_MiB=MIN_HEAVY_FREE_MIB,
            heavy_dependencies=['root/forward/EXIT_CODE','root/observation_calibration/EXIT_CODE'],new_theory_pretraining=False,extra_targeted_support=False,
            external_api_calls=0,head_training=reference),indent=2))
        def training(arm,smoke=False):
            owner=previous/'study/evaluation'/arm['owner'];init=previous/'study/training'/arm['owner']/'train/best.pt'
            cmd=[sys.executable,'-m','nmr_transfer.h_global_forward_train','--base',str(base),'--prepared',str(prepared),
                '--cohorts',str(root/'cohorts'),'--init',str(init),'--bank',str(owner/'bank.pt'),
                '--features',str(root/'calibration/features'/arm['owner']/'train/features.pt'),'--review',str(review),
                '--out',str(out/'smoke' if smoke else out/'training'/arm['name']/'train'),'--seed',str(arm['seed']),
                '--scope','messages','--epochs','24','--samples','32768']
            if arm['rank']:cmd+=['--rank']
            if arm['anchor']:cmd+=['--anchor']
            if smoke:cmd+=['--smoke']
            return arm['name'],cmd
        run_jobs([training(arms()[-1],True)],'smoke')
        run_jobs([training(arm) for arm in arms()],'train')
        audits={};variants=[]
        for arm in arms():
            folder=out/'training'/arm['name']/'train';cfg=json.loads((folder/'config.json').read_text())
            summary=json.loads((folder/'summary.json').read_text());ref=reference[arm['name']]
            assert_training_pair(cfg,ref['config'],summary,ref['summary'])
            if any(sha(Path(ref['folder'])/name)!=digest for name,digest in ref['input_sha256'].items()):
                raise ValueError('global_message_head_reference_changed')
            warmstart=signature(folder/'initial.pt')
            if warmstart!=signature(Path(ref['folder'])/'initial.pt'):raise ValueError('global_message_warmstart_state_drift')
            audits[arm['name']]=dict(message=summary,head=ref['summary'],warmstart_state_sha256=warmstart,paired_exposure_exact=True)
            source=folder/'epoch_24.pt';saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
            if saved['epoch']!=24 or saved['training_scope']!='messages':raise ValueError('global_message_checkpoint_scope_drift')
            del saved
            name=arm['name']+'_epoch_24';target=out/'evaluation_training'/name/'train';target.mkdir(parents=True)
            os.link(source,target/'best.pt');variants.append(dict(name=name,arm=arm,label='epoch_24',checkpoint_sha256=sha(source)))
        for seed in (193,211):
            paired=[audits[arm['name']]['message'] for arm in arms() if arm['seed']==seed]
            for key in ('forward_exposure_sha256','negative_exposure_sha256'):
                if len({item[key] for item in paired})!=1:raise ValueError('global_message_kind_exposure_drift:'+key)
        (out/'paired_exposure.json').write_text(json.dumps(audits,indent=2));(out/'variants.json').write_text(json.dumps(variants,indent=2))
        jobs=[]
        for variant in variants:
            name=variant['name'];cmd=[sys.executable,'-m','nmr_transfer.h_joint_evaluate','--base',str(base),'--prepared',str(prepared),
                '--training',str(out/'evaluation_training'),'--review',str(review),'--mode',name,'--stage','val','--out',str(out/'evaluation'/name)]
            jobs.append((name,cmd))
        run_jobs(jobs,'predict',heavy=True)
        jobs=[]
        for variant in variants:
            name=variant['name'];bank=out/'evaluation'/name/'bank.pt'
            jobs.append((name,[sys.executable,'-m','nmr_transfer.h_global_bank','--bank',str(bank),'--cohorts',str(root/'cohorts'),
                '--prepared',str(prepared),'--out',str(out/'extensions'/name)]))
        run_jobs(jobs,'extend',heavy=True)
        jobs=[]
        for variant in variants:
            name=variant['name'];bank=out/'evaluation'/name/'bank.pt'
            for cohort in COHORTS:
                jobs.append((name+'_'+cohort,[sys.executable,'-m','nmr_transfer.h_global_scoring','--base',str(base),'--bank',str(bank),
                    '--cohorts',str(root/'cohorts'),'--extension',str(out/'extensions'/name/'extension.pt'),
                    '--widths',str(bank.parent/'val/width_fit.json'),'--cohort',cohort,'--out',str(out/'scoring'/name/cohort)]))
        run_jobs(jobs,'score',heavy=True)
        results={v['name']:{c:json.loads((out/'scoring'/v['name']/c/'summary.json').read_text()) for c in COHORTS} for v in variants}
        (out/'results.json').write_text(json.dumps(results,indent=2));comparisons={};reference_gaps=[]
        for variant in variants:
            name=variant['name'];reference_name=variant['arm']['reference']+'_epoch_24';paired={}
            for cohort in COHORTS:
                folder=head_root/'scoring'/reference_name/cohort;path=folder/'summary.json'
                code=(folder/'EXIT_CODE').read_text().strip() if (folder/'EXIT_CODE').exists() else None
                if code!='0' or not path.exists():
                    reference_gaps.append(dict(reference=reference_name,cohort=cohort,exit_code=code,folder=str(folder)));continue
                head=json.loads(path.read_text())
                paired[cohort]=scope_comparison(head,results[name][cohort],grid,cohort)
            combined={choice['name']:{metric:sum(COHORTS[c]*paired[c][choice['name']]['delta'][metric] for c in COHORTS)/sum(COHORTS.values())
                       for metric in METRICS} for choice in grid} if len(paired)==len(COHORTS) else None
            comparisons[name]=dict(reference=reference_name,cohorts=paired,micro_delta=combined)
        (out/'summary.json').write_text(json.dumps(dict(variants=variants,results=results,scope_comparisons=comparisons,
            evaluated_checkpoint='epoch_24',source_sha256=sources,external_api_calls=0,dependency_failures=dependency_failures,
            scope_comparison_complete=not reference_gaps,reference_gaps=reference_gaps),indent=2))
        status('complete',scope_comparison_complete=not reference_gaps,reference_gaps=reference_gaps);(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for process,record in active:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
            finish(process,record)
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':
    main()

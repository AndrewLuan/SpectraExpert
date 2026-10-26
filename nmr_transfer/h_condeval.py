"""功能概述：四组条件训练后的验证选择、固定测试消融和所选双核评测编排。
baseline_validation重建上轮固定评分的320条验证排名，HMDB用已冻结的可见性/峰簇
分支，积分谱与稀疏标注分别使用原正确口径。--variants可指定后续消融名称；--reference-evaluation把上一轮验证优胜者作为只读
参照，保留其实际条件键口径。main先完成全部val再冻结模型选择，
之后才运行四组相同1000测试、171缺峰、200裁窗和父节点诊断。双核仅评验证所选。
variants超过GPU数时分批运行，每个GPU同时只放一个任务，避免静默漏评或重叠。
全部阶段独立保存日志/退出码；允许baseline入选，不按test挑选。源代码启动前
快照固定；训练权重与评分配置不在本流程修改。等待训练有上限，失败如实退出。
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .h_condmodel import MODES
from .h_observation_study import summarize_sources,write_rows


def baseline_validation(base,out):
    h=base/'hydrogen_analysis';root=h/'evaluation_dropout/site/val';old=[[json.loads(l) for l in (root/(str(i)+'_predictions.jsonl')).read_text().splitlines()] for i in [0,4]]
    hmdb={(r['record_id'],r['smiles']):r for r in map(json.loads,(h/'observation_comparison/water_validation_selected.jsonl').read_text().splitlines())};rows=[]
    # val记录来源和measure的原顺序由bank固定；两种来源的积分谱恰为NMRexp/WanJuan。
    for baseline,integral in zip(*old):
        if baseline['source']=='HMDB':r=hmdb[baseline['record_id'],baseline['smiles']]
        elif baseline['source'] in ['NMRexp','WanJuan']:r=integral
        else:r=baseline
        rows.append(r)
    if len(rows)!=320:raise ValueError('baseline_validation_cohort_changed')
    write_rows(out/'baseline_validation.jsonl',rows);return dict(metrics=summarize_sources(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--training',required=True);p.add_argument('--out',required=True);p.add_argument('--gpus',default='1,2,3,4');p.add_argument('--variants',default=','.join(MODES));p.add_argument('--reference-evaluation');a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/f.name)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    processes=[]
    def launch(stage):
        variants=a.variants.split(',');gpus=a.gpus.split(',')
        for start in range(0,len(variants),len(gpus)):
            processes.clear()
            for mode,gpu in zip(variants[start:start+len(gpus)],gpus):
                cmd=[sys.executable,'-u','-m','nmr_transfer.h_condinfer','--base',a.base,'--prepared',a.prepared,'--training',a.training,'--out',str(out/mode),'--mode',mode,'--stage',stage]
                f=(out/(mode+'_'+stage+'.log')).open('x');proc=subprocess.Popen(cmd,cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT);processes.append((mode,proc,f))
            status(stage,batch_start=start,pids={mode:p.pid for mode,p,_ in processes})
            for mode,proc,f in processes:
                result=proc.wait();f.close()
                if result:raise RuntimeError(mode+' '+stage+' failed')
    try:
        status('waiting_training');path=Path(a.training)/'EXIT_CODE';deadline=time.monotonic()+10800
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('training_timeout')
            time.sleep(10)
        if path.read_text().strip()!='0':raise RuntimeError('training_failed')
        if a.reference_evaluation:
            ref_exit=Path(a.reference_evaluation)/'EXIT_CODE';status('waiting_reference_evaluation')
            while not ref_exit.exists():
                if time.monotonic()>deadline:raise TimeoutError('reference_evaluation_timeout')
                time.sleep(10)
            if ref_exit.read_text().strip()!='0':raise RuntimeError('reference_evaluation_failed')
        baseline=baseline_validation(base,out);launch('val');trials={'baseline':baseline}
        for mode in a.variants.split(','):trials[mode]=json.loads((out/mode/'val/summary.json').read_text())
        if a.reference_evaluation:
            reference=Path(a.reference_evaluation);previous=json.loads((reference/'frozen_selection.json').read_text())['selected_variant']
            if previous!='baseline':
                alias='previous_'+previous;trials[alias]=json.loads((reference/previous/'val/summary.json').read_text());(out/alias).symlink_to(reference/previous,target_is_directory=True)
        selected=max(trials,key=lambda k:trials[k]['metrics']['overall']['MRR']);frozen=dict(selected_variant=selected,validation=trials,selection='MRR on same 320 validation queries, baseline included; scoring fixed to prior observation study; test never selects',development_test_reuse=True)
        (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));launch('test');status('joint',selected=selected)
        with (out/'joint.log').open('x') as f:
            r=subprocess.run([sys.executable,'-u','-m','nmr_transfer.h_condjoint','--base',a.base,'--prepared',a.prepared,'--evaluation',str(out),'--out',str(out/'joint')],cwd=out/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus.split(',')[0]),stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('joint_failed')
        status('complete',selected=selected);(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        for _,proc,_ in processes:
            if proc.poll() is None:proc.terminate()
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

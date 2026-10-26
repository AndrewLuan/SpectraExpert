"""功能概述：JJ上串行组织预训练、四种实验训练、验证冻结和最终检索，跨SSH运行。
主要职责：main在数据complete和哈希验证后用torchrun运行各阶段；每阶段写状态和
独立日志/退出码，任何非零状态立即停止。run不自动重试失败训练、不覆盖旧模型。
两个核种可分配各四张不同GPU运行；由调用方nohup启动。本入口不设系统定时任务。
冻结选择只依据实验验证误差；预训练模型仍保留为零样本基线。全部训练结束后才评测test。
输入为已准备cache；运行目录不存在才创建。源码在启动前复制到独立source快照用于复现，
这是不可变实验记录，代码主线仍是现有本地checkout。输出status.json和EXIT_CODE。
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
from .prepare import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--out',required=True)
    p.add_argument('--nucleus',required=True,choices=['1H','13C']);p.add_argument('--gpus',required=True)
    a=p.parse_args();root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=False)
    source=root/'source'/'nmr_transfer';source.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,source/path.name)
    prepared=Path(a.prepared).resolve();env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,OMP_NUM_THREADS='4')
    env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';gpu_count=len(a.gpus.split(','))
    status={}
    def update(stage,**kw):
        status.update(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**kw)
        (root/'status.json').write_text(json.dumps(status,indent=2))
    def run(stage,cmd):
        update(stage,command=cmd)
        with (root/(stage+'.log')).open('x') as f:
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=root/'source')
        (root/(stage+'.exit_code')).write_text(str(result.returncode)+'\n')
        if result.returncode:raise RuntimeError(f'{stage} failed, exit={result.returncode}')
    try:
        update('waiting_for_prepared_data');deadline=time.monotonic()+7200
        while not (prepared/'complete.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('preparation_incomplete_after_2h')
            time.sleep(15)
        cache=prepared/'cache.pt';manifest=json.loads((prepared/'manifest.json').read_text())
        if sha(cache)!=manifest['cache_sha256']:raise ValueError('prepared_cache_hash_mismatch')
        for mode in ['pretrain','scratch','finetune','augment','domain']:
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={gpu_count}',
                 '-m','nmr_transfer.train','--cache',str(cache),'--out',str(root/mode),'--nucleus',a.nucleus,'--mode',mode,
                 '--epochs','30','--patience','6','--batch-size','128','--workers','2']
            if mode not in ['pretrain','scratch']:cmd+=['--init',str(root/'pretrain/best.pt')]
            run(mode,cmd)
        runs={m:json.loads((root/m/'complete.json').read_text()) for m in ['pretrain','scratch','finetune','augment','domain']}
        selected=min(['scratch','finetune','augment','domain'],key=lambda m:runs[m]['best_validation_ppm'])
        (root/'frozen_selection.json').write_text(json.dumps(dict(runs=runs,selected_mode=selected,cache_sha256=sha(cache),test_seen=False,
             selection='lowest full experimental validation observation error; conditional/distribution metric stratified by coverage',query_cap=1000,candidate_library='all experimental structures in all splits'),indent=2))
        run('evaluation',[sys.executable,'-m','nmr_transfer.evaluate','--root',str(root),'--cache',str(cache),'--nucleus',a.nucleus])
        update('complete',selected_mode=selected);(root/'EXIT_CODE').write_text('0\n')
    except Exception as e:
        update('failed',error=repr(e));(root/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

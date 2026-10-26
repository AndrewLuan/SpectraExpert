"""组织核发射输出的补充实验，不覆盖原SpecGNN分布模型或原始检索结果。
main复制轻量源码快照，引用同一数据缓存和原GINE预训练骨干，先训练新核输出头
及骨干10epoch，再做20epoch实验域适配。两阶段仅按各自val观测误差保存权重。
之后用likelihood在val选择带宽，再对相同test查询复评；记录为诊断后的补充实验。
每阶段独立日志/退出码，失败立即停止；不自动重跑，不创建新Git checkout。
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .prepare import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--nucleus',required=True);p.add_argument('--gpus',required=True)
    a=p.parse_args();base=Path(a.root).resolve();container=base/'atom_emission';container.mkdir(exist_ok=True)
    link=container/'prepared_coverage'
    try:link.symlink_to(base/'prepared_coverage',target_is_directory=True)
    except FileExistsError:pass
    out=container/a.nucleus;out.mkdir(exist_ok=False);source=out/'source/nmr_transfer';source.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,source/path.name)
    # 本机两卡P2P在NCCL初始化阻塞；SHM传输已通过同一两卡训练验证，作用域仅本进程树。
    cache=base/'prepared_coverage/cache.pt';env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,OMP_NUM_THREADS='4',NCCL_P2P_DISABLE='1')
    def state(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**kw),indent=2))
    def run(stage,command):
        state(stage)
        with (out/(stage+'.log')).open('x') as f:r=subprocess.run(command,cwd=out/'source',env=env,stdout=f,stderr=subprocess.STDOUT)
        (out/(stage+'.exit_code')).write_text(str(r.returncode)+'\n')
        if r.returncode:raise RuntimeError(stage+' failed')
    (out/'protocol.json').write_text(json.dumps(dict(development='supplementary architecture experiment after original retrieval diagnostics',
        readout='mass-conserving union of nuclear emissions; methyl equivalence and dimethylamide symmetric mixture',
        theory_epochs=10,domain_epochs=20,same_data_partitions=True,test_used_for_model_selection=False),indent=2))
    try:
        for mode,epochs,init in [('pretrain',10,base/a.nucleus/'pretrain/best.pt'),('domain',20,out/'pretrain/best.pt')]:
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(len(a.gpus.split(','))),'-m','nmr_transfer.train',
                     '--cache',str(cache),'--out',str(out/mode),'--nucleus',a.nucleus,'--mode',mode,'--model','atom','--init',str(init),
                     '--epochs',str(epochs),'--patience','5','--batch-size','128','--workers','2']
            run(mode,command)
        complete=json.loads((out/'domain/complete.json').read_text())
        (out/'frozen_selection.json').write_text(json.dumps(dict(runs={'domain':complete},selected_mode='domain',cache_sha256=sha(cache),
                                                               selection='domain-stage experimental validation observation error; supplement developed after original test diagnostics'),indent=2))
        run('likelihood',[sys.executable,'-m','nmr_transfer.likelihood','--root',str(container),'--nucleus',a.nucleus,'--out',str(out/'likelihood'),'--selected-only'])
        state('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        state('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

"""功能概述：为新资源就绪 operator 保存外层 PID、真实进程退出与内部 EXIT 收据。
main 只启动指定新目录的 supervise_controlled_queue，不重试；queue/source SHA
写入 launch_receipt 供独立消费者核验。它不代替 operator 的独审与科学输入准入。
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', required=True)
    a = p.parse_args()
    path = Path(a.queue).resolve()
    root = path.parent
    queue = json.loads(path.read_text())
    refs=queue['operator_paths']
    assert type(refs) is dict and set(refs)=={'supervisor','launcher'}
    for role,name in [('supervisor','supervise_resource_ready_queue.py'),('launcher','launch_resource_ready_stage.py')]:
        ref=refs[role];pinned=str(root/name)
        assert type(ref) is dict and set(ref)=={'path','sha256'}
        assert ref['path']==pinned and ref['sha256']==queue['source_sha256'][pinned]==hashlib.sha256(Path(pinned).read_bytes()).hexdigest()
    assert Path(__file__).resolve()==Path(refs['launcher']['path'])
    run_name = queue.get('run_name', queue['phase'])
    assert type(run_name) is str and re.fullmatch('[a-z][a-z0-9_]{0,63}', run_name)
    out = root / 'runs' / run_name
    assert not out.exists()
    assert not (root / (run_name + '.failed_launch_receipt.json')).exists()
    with (root / (run_name + '.launch_claim.json')).open('x') as stream:
        json.dump({'wrapper_pid': os.getpid(),
                   'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'queue_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, stream)
    command = [sys.executable, '-B', refs['supervisor']['path'],
               '--queue', str(path), '--out', str(out), '--execute']
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    process=None
    pending_signals=[];received_signals=[];forwarded=[]
    def forward(signum,frame):
        received_signals.append(signum)
        if process is None:pending_signals.append(signum)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid,signum)
                forwarded.append(signum)
            except ProcessLookupError:pass
    # 必须在Popen之前安装；spawn内部/返回边界的信号只能排队，holder建立后立即转发。
    previous={sig:signal.signal(sig,forward) for sig in (signal.SIGTERM,signal.SIGINT)}
    external=1
    try:
        if not pending_signals:
            process=subprocess.Popen(command,start_new_session=True)
            for signum in pending_signals[:]:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid,signum)
                        forwarded.append(signum)
                    except ProcessLookupError:pass
            external=process.wait()
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    internal = None
    if (out / 'EXIT_CODE').is_file():
        raw = (out / 'EXIT_CODE').read_text().strip()
        if raw in ('0', '1', '124'):
            internal = int(raw)
    receipt = dict(wrapper_pid=os.getpid(), child_pid=process.pid if process is not None else None, argv=command, phase=queue['phase'], run_name=run_name,
        started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        internal_exit=internal, external_exit=external,
        queue_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_sha256=queue['source_sha256'],operator_paths=refs,forwarded_signals=forwarded,cancellation_signals=received_signals)
    receipt_path = out / 'launch_receipt.json' if out.is_dir() else root / (run_name + '.failed_launch_receipt.json')
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    if out.is_dir():
        (out / 'EXTERNAL_EXIT_CODE').write_text(str(external) + '\n')
    print(json.dumps(receipt))
    raise SystemExit(0 if type(external) is int and external == 0 and type(internal) is int and internal == 0 else 1)


if __name__ == '__main__':
    main()

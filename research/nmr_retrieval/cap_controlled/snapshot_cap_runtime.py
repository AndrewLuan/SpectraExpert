"""功能概述：只读已登记队列的进程、退出文件与训练进度，保存紧凑运行快照。
main 从 queue 导出真实 run_name，读取 driver/job metadata、64epoch history计数及
任务自己日志的异常尾部；不读取预测内容、张量或旧test报告，不启动/取消任务。
输出runtime_snapshots新文件及JSON stdout，供当前会话人工编排监控；不建自动化。
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def read_json(path):
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except json.JSONDecodeError:
        return None  # 只读快照可能遇到写入中的status，等待下一次观察。


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', required=True)
    a = p.parse_args()
    root = Path(a.queue).resolve().parent
    q = read_json(Path(a.queue))
    run_name = q.get('run_name', q['phase'])
    run = root / 'runs' / run_name
    start = read_json(run / 'driver_start.json')
    status = read_json(run / 'status.json')
    completion = read_json(run / 'completion.json')
    jobs = []
    for j in q['jobs']:
        out = Path(j['job_out'])
        h = out / 'history.jsonl'
        content = h.read_text() if h.is_file() else ''
        lines = content.splitlines()
        if content and not content.endswith('\n'):
            lines = lines[:-1]
        last = json.loads(lines[-1]) if lines else {}
        rec = read_json(run / 'jobs' / (j['id'] + '.receipt.json'))
        launched = read_json(run / 'jobs' / (j['id'] + '.started.json'))
        log = run / 'jobs' / (j['id'] + '.log')
        tail = log.read_text(errors='replace')[-6000:] if log.is_file() else ''
        fatal = any(x in tail for x in ['Traceback', 'PermissionError', 'RuntimeError', 'AssertionError', 'CUDA out of memory'])
        jobs.append(dict(id=j['id'], output_exists=out.exists(), epochs=len(lines),
            epoch=last.get('epoch'), draws=last.get('draws', last.get('cumulative_draws')),
            optimizer_steps=last.get('optimizer_steps', last.get('cumulative_optimizer_steps')),
            pid=launched.get('pid') if launched else None,
            internal_exit=rec.get('internal_exit') if rec else None,
            external_exit=rec.get('external_exit') if rec else None,
            fatal_tail=tail[-1200:] if fatal else None))
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    result = dict(utc=now.isoformat(), phase=q['phase'], run_name=run_name,
        driver_pid=start.get('pid') if start else None,
        state=completion.get('status') if completion else (status.get('phase') if status else 'STARTING'),
        running=len(status.get('running', [])) if status else None,
        completed=status.get('completed') if status else None,
        jobs=jobs, gpu=gpu.stdout if gpu.returncode == 0 else None,
        prediction_contents_opened=False, model_calls=0)
    folder = root / 'runtime_snapshots'
    folder.mkdir(exist_ok=True)
    (folder / (run_name + '_' + now.strftime('%Y%m%dT%H%M%S%fZ') + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

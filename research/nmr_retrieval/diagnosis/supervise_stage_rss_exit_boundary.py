"""功能概述：在新诊断目录中有界运行一个已独审的冻结子进程。

main 校验 job、源码与独审 SHA 后启动唯一子进程。启动前安装取消处理；
以进程结束、deadline 或 RSS 超限为边界，必要时仅终止自己创建的进程组。
它保存原始 stdout/stderr、实际子退出和监督器退出，不重试、不调用科学代码。
本程序与 extract_frozen_diagnostics.py 的科学输入白名单互相独立。
修订通过至多0.25秒的进程完成事件等待，区分退出边界缺失RSS与存活时监测失效。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(tmp, path)


def rss_bytes(pid):
    status = Path('/proc') / str(pid) / 'status'
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    elif sys.platform == 'darwin':
        value = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(pid)],
                               capture_output=True, text=True, timeout=2)
        if value.returncode == 0 and value.stdout.strip():
            return int(value.stdout.strip()) * 1024
    return None


def output_bytes(root):
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in files:
            p = Path(directory) / name
            if p.is_file() and not p.is_symlink():
                try:
                    total += p.stat().st_size
                except FileNotFoundError:
                    pass
    return total


def terminate_owned(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--review', required=True)
    ap.add_argument('--review-sha256', required=True)
    args = ap.parse_args()
    assert sha(args.review) == args.review_sha256, 'review SHA'
    review = json.loads(Path(args.review).read_text())
    assert review['status'] == 'PASS' and review['independent'] is True
    assert review['execution_authorized'] is True
    job = json.loads(Path(args.job).read_text())
    assert review['pins']['job'] == sha(args.job), 'job SHA'
    assert review['pins']['supervisor'] == sha(__file__), 'supervisor SHA'
    assert job['schema'] == 'top10_bounded_stage_v1'
    assert 0 < job['max_seconds'] <= 3600
    assert 0 < job['max_rss_bytes'] <= 32 * 1024 ** 3
    assert job['max_cpu_threads'] == 4
    assert 0 < job['max_artifact_bytes'] <= 4 * 1024 ** 3
    for path, expected in job['source_sha256'].items():
        assert sha(path) == expected, 'source SHA: ' + path
    stage = Path(job['stage_dir']).resolve()
    assert stage.is_relative_to(Path(job['write_root']).resolve())
    stage.mkdir(parents=True, exist_ok=False)
    interrupted = []
    def on_cancel(sig, frame):
        interrupted.append(sig)
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous[sig] = signal.signal(sig, on_cancel)
    proc = None
    started = time.monotonic()
    outcome = dict(schema='top10_bounded_stage_execution_v1', job_sha256=sha(args.job),
                   review_sha256=args.review_sha256, pid=None, actual_inner_exit=None,
                   actual_external_exit=1, termination_reason=None, peak_observed_rss_bytes=0,
                   rss_monitor_samples=0,
                   max_seconds=job['max_seconds'], max_rss_bytes=job['max_rss_bytes'],
                   max_artifact_bytes=job['max_artifact_bytes'])
    save(stage / 'started.json', outcome)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    try:
        with (stage / 'stdout.log').open('x') as stdout, (stage / 'stderr.log').open('x') as stderr:
            if interrupted:
                outcome['termination_reason'] = 'cancel_before_start'
            else:
                proc = subprocess.Popen(job['argv'], cwd=job['cwd'], env=env,
                                        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                        start_new_session=True)
                outcome['pid'] = proc.pid
                save(stage / 'running.json', outcome)
                while proc.poll() is None:
                    elapsed = time.monotonic() - started
                    rss = rss_bytes(proc.pid)
                    # /proc may lose VmRSS just before waitpid reports completion.
                    # Wait on that exit event briefly; an alive unmonitored child still fails.
                    if rss is None and proc.poll() is None:
                        try:
                            proc.wait(timeout=min(.25, max(.001, job['max_seconds'] - elapsed)))
                            outcome['rss_exit_boundary_confirmed'] = True
                        except subprocess.TimeoutExpired:
                            outcome['rss_missing_while_alive'] = True
                        elapsed = time.monotonic() - started
                    if rss is not None:
                        outcome['rss_monitor_samples'] += 1
                        outcome['peak_observed_rss_bytes'] = max(outcome['peak_observed_rss_bytes'], rss)
                    total_bytes = output_bytes(Path(job['write_root']))
                    outcome['observed_output_bytes'] = total_bytes
                    if interrupted:
                        outcome['termination_reason'] = 'cancel_signal_' + str(interrupted[0])
                    elif elapsed >= job['max_seconds']:
                        outcome['termination_reason'] = 'deadline_exceeded'
                    elif rss is not None and rss > job['max_rss_bytes']:
                        outcome['termination_reason'] = 'rss_exceeded'
                    elif total_bytes > job['max_artifact_bytes']:
                        outcome['termination_reason'] = 'output_budget_exceeded'
                    elif outcome.get('rss_missing_while_alive', False):
                        outcome['termination_reason'] = 'rss_monitor_unavailable'
                    if outcome['termination_reason']:
                        terminate_owned(proc)
                        break
                    try:
                        proc.wait(timeout=min(.5, max(.001, job['max_seconds'] - elapsed)))
                    except subprocess.TimeoutExpired:
                        pass
                outcome['actual_inner_exit'] = proc.wait(timeout=5)
                outcome['actual_external_exit'] = (0 if outcome['actual_inner_exit'] == 0 and
                                                   outcome['termination_reason'] is None else 1)
    except BaseException as exc:
        outcome['error'] = type(exc).__name__ + ': ' + str(exc)
        if proc is not None:
            terminate_owned(proc)
            outcome['actual_inner_exit'] = proc.returncode
        outcome['actual_external_exit'] = 1
    finally:
        outcome['elapsed_seconds'] = time.monotonic() - started
        outcome['rss_monitor_status'] = ('observed' if outcome['rss_monitor_samples'] else
                                         'unavailable_while_alive' if outcome.get('rss_missing_while_alive') else
                                         'not_observed_child_already_exited')
        outcome['final_output_bytes_before_receipt'] = output_bytes(Path(job['write_root']))
        if outcome['final_output_bytes_before_receipt'] > job['max_artifact_bytes']:
            outcome['termination_reason'] = outcome['termination_reason'] or 'output_budget_exceeded'
            outcome['actual_external_exit'] = 1
        save(stage / 'actual_execution.json', outcome)
        (stage / 'EXIT_CODE').write_text(str(outcome['actual_inner_exit']) + '\n')
        if output_bytes(Path(job['write_root'])) > job['max_artifact_bytes']:
            outcome['termination_reason'] = outcome['termination_reason'] or 'output_budget_exceeded'
            outcome['actual_external_exit'] = 1
            save(stage / 'actual_execution.json', outcome)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps(outcome, ensure_ascii=False))
    return outcome['actual_external_exit']


if __name__ == '__main__':
    raise SystemExit(main())

"""功能概述：唯一执行已独审冻结的 C10/C5 阶段队列，支持独占空闲 GPU 并发。

sha/read/write 处理指纹与收据；gpu_idle 只读 GPU 状态；stop_owned 只终止本
operator 启动的进程组；main 检查精确 argv、独审绑定与所有输入始末 SHA。
每个任务经 isolation_entry 审计，保存 PID、内外退出和原始日志。任一失败停止
本队列尚未启动任务并清理自身子进程，不自动重跑、不触碰他人进程。
输入为单阶段 immutable queue；输出为新 runs/phase 的 driver/job/completion。
validation、mechanism、test 各需自己的独审与新 binding，不覆盖训练证据。
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
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def gpu_idle(index):
    out = subprocess.run(['nvidia-smi', '-i', str(index),
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True).stdout.strip()
    memory, utilization = [int(x.strip()) for x in out.split(',')]
    apps = subprocess.run(['nvidia-smi', '-i', str(index),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True).stdout.strip()
    return memory < 1024 and utilization < 5 and not apps


def stop_owned(running, records, out):
    for process, record, stream in running.values():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process, record, stream in running.values():
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        stream.close()
        if not any(x['id'] == record['id'] for x in records):
            access = read(record['access_log']) if Path(record['access_log']).is_file() else {}
            record.update(finished_utc=now(), external_exit=process.returncode,
                          internal_exit=access.get('internal_exit'), terminated_after_peer_failure=True)
            records.append(record)
            write(out / 'jobs' / (record['id'] + '.receipt.json'), record)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    queue_path = Path(a.queue).resolve()
    root = queue_path.parent
    out = Path(a.out).resolve()
    q = read(queue_path)
    assert q['schema'] == 'cap10_controlled_queue_20261024'
    run_name = q.get('run_name', q['phase'])
    assert type(run_name) is str and re.fullmatch('[a-z][a-z0-9_]{0,63}', run_name)
    assert out == root / 'runs' / run_name
    assert q['phase'] in ('train', 'validation', 'mechanism', 'test')
    protocol = read(root / 'protocol.json')
    binding = read(q['bindings'])
    assert q['input_sha256'] == binding['input_sha256']
    assert q['jobs'] and len({x['id'] for x in q['jobs']}) == len(q['jobs'])
    assert type(q['max_parallel']) is int and 0 < q['max_parallel'] <= 8
    assert 0 < q['maximum_seconds'] <= protocol['budget']['maximum_stage_seconds']
    assert len({x['job_out'] for x in q['jobs']}) == len(q['jobs'])
    assert len({x['access_log'] for x in q['jobs']}) == len(q['jobs'])
    expected_counts = {'train': 8, 'validation': 16, 'mechanism': 4}
    if q['phase'] in expected_counts:
        assert len(q['jobs']) == expected_counts[q['phase']]
    else:
        assert len(q['jobs']) in (8, 12)
        assert q['selection_locked_before_test'] is True
    local_root = Path(q['local_root'])
    review_paths = [Path(x) for x in q['review_paths']]
    assert review_paths
    required_sources = {str(root / name) for name in
        ('protocol.json', 'isolation_entry.py', 'supervise_controlled_queue.py',
         'launch_controlled_stage.py', 'cap_ranker.py')}
    required_sources.add(str(Path(q['bindings']).resolve()))
    required_sources.update(str(Path(x['command'][0]).resolve()) for x in q['jobs'])
    required_sources.update(binding['source_sha256'])
    assert required_sources.issubset(q['source_sha256']), sorted(required_sources - set(q['source_sha256']))
    assert all(q['source_sha256'][path] == digest for path, digest in binding['source_sha256'].items())
    required = dict(q['source_sha256'])
    required[str(queue_path)] = sha(queue_path)
    for review_path in review_paths:
        review = read(review_path)
        assert review['status'] == 'PASS' and review['independent'] is True
        assert review['admission_phase'] == q['phase']
        for path, digest in required.items():
            local = str(local_root / Path(path).relative_to(root)) if Path(path).is_relative_to(root) else path
            assert review['reviewed_sha256'].get(path, review['reviewed_sha256'].get(local)) == digest, (review_path, path)
    assert all(sha(path) == digest for path, digest in q['source_sha256'].items())
    for job in q['jobs']:
        assert job['argv'][0] == q['python']
        assert job['argv'][1:3] == ['-B', str(root / 'isolation_entry.py')]
        assert job['cwd'] == protocol['source_checkout']
        assert Path(job['job_out']).resolve().is_relative_to(root)
        assert Path(job['access_log']).resolve().is_relative_to(root / 'access_logs')
        assert all(path in q['input_sha256'] for path in job['allowed_reads'])
        expected = [q['python'], '-B', str(root / 'isolation_entry.py'),
                    '--bindings', q['bindings'], '--phase', q['phase'],
                    '--access-log', job['access_log'], '--output-root', job['job_out']]
        for path in job['allowed_reads']:
            expected += ['--allow-read', path]
        assert job['argv'] == expected + ['--'] + job['command']
    if not a.execute:
        print(json.dumps({'status': 'ADMISSION_METADATA_PASS', 'scientific_jobs_started': 0}))
        return
    assert not out.exists()
    assert all(not Path(x['job_out']).exists() and not Path(x['access_log']).exists() for x in q['jobs'])
    ps = subprocess.run(['ps', '-eo', 'pid,ppid,args'], check=True, capture_output=True, text=True).stdout
    duplicate = [line for line in ps.splitlines() if str(root) in line and
        any(script in line for script in ('train_cap_ranker.py', 'evaluate_cap_ranker.py', 'supervise_controlled_queue.py'))
        and int(line.split()[0]) != os.getpid() and ' -c ' not in line]
    assert not duplicate, duplicate
    claim = root / (run_name + '.operator_claim.json')
    with claim.open('x') as stream:
        json.dump({'pid': os.getpid(), 'utc': now(), 'queue_sha256': sha(queue_path)}, stream)
    out.mkdir(parents=True)
    (out / 'jobs').mkdir()
    (root / 'access_logs').mkdir(exist_ok=True)
    started = now()
    begin = time.monotonic()
    small = set(q['source_sha256']) | {str(queue_path)} | {str(x) for x in review_paths}
    small_before = {path: sha(path) for path in small}
    start = dict(pid=os.getpid(), parent_pid=os.getppid(), argv=sys.argv, phase=q['phase'], run_name=run_name,
        started_utc=started, queue_sha256=sha(queue_path),
        source_sha256=q['source_sha256'], input_sha256=q['input_sha256'])
    write(out / 'driver_start.json', start)
    running, records = {}, []
    pending = list(q['jobs'])
    code, status, error = 1, 'FAILED', None
    before, after = None, None
    try:
        before = {path: sha(path) for path in q['input_sha256']}
        write(out / 'input_fingerprints.before.json', before)
        assert before == q['input_sha256']
        while pending or running:
            assert time.monotonic() - begin < q['maximum_seconds'], 'stage_deadline'
            while pending and len(running) < q['max_parallel']:
                job = pending[0]
                if job['gpu'] is not None and any(r[1]['gpu'] == job['gpu'] for r in running.values()):
                    break
                if job['gpu'] is not None:
                    assert gpu_idle(job['gpu']), ('gpu_not_idle', job['gpu'])
                pending.pop(0)
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                    MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                if job['gpu'] is not None:
                    env['CUDA_VISIBLE_DEVICES'] = str(job['gpu'])
                else:
                    env['CUDA_VISIBLE_DEVICES'] = ''
                stream = (out / 'jobs' / (job['id'] + '.log')).open('wb')
                job_started = now()
                process = subprocess.Popen(job['argv'], cwd=job['cwd'], env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                record = dict(id=job['id'], argv=job['argv'], cwd=job['cwd'], gpu=job['gpu'],
                    pid=process.pid, parent_pid=os.getpid(), started_utc=job_started,
                    job_out=job['job_out'], access_log=job['access_log'],
                    queue_sha256=sha(queue_path), requested_inputs_sha256={path: before[path] for path in job['allowed_reads']})
                running[job['id']] = (process, record, stream)
                write(out / 'jobs' / (job['id'] + '.started.json'), record)
            finished = []
            for job_id, (process, record, stream) in running.items():
                result = process.poll()
                if result is None:
                    continue
                stream.close()
                access = read(record['access_log']) if Path(record['access_log']).is_file() else {}
                record.update(finished_utc=now(), external_exit=result, internal_exit=access.get('internal_exit'))
                write(out / 'jobs' / (job_id + '.receipt.json'), record)
                records.append(record)
                finished.append(job_id)
                assert type(result) is int and result == 0, record
                assert type(record['internal_exit']) is int and record['internal_exit'] == 0, record
                assert not access.get('rejected_paths'), access
                assert access['pid'] == process.pid and access['parent_pid'] == os.getpid()
                assert access['argv'] == record['argv'][record['argv'].index('--') + 1:]
                parse = datetime.datetime.fromisoformat
                assert parse(record['started_utc']) <= parse(access['started_utc']) <= parse(access['finished_utc']) <= parse(record['finished_utc'])
            for job_id in finished:
                del running[job_id]
            write(out / 'status.json', {'phase': q['phase'], 'completed': len(records),
                'running': [x[1] for x in running.values()], 'pending': len(pending), 'utc': now()})
            if running:
                time.sleep(0.2)
        after = {path: sha(path) for path in before}
        write(out / 'input_fingerprints.after.json', after)
        assert after == before
        small_after = {path: sha(path) for path in small}
        write(out / 'operator_fingerprints.before.json', small_before)
        write(out / 'operator_fingerprints.after.json', small_after)
        assert small_after == small_before
        status, code = 'COMPLETE', 0
    except BaseException as exc:
        error = repr(exc)
        stop_owned(running, records, out)
        raise
    finally:
        write(out / 'completion.json', dict(status=status, pid=os.getpid(), phase=q['phase'], run_name=run_name,
            started_utc=started, finished_utc=now(), jobs=records, error=error,
            queue_sha256=sha(queue_path), external_exit=code,
            inputs_unchanged=(after == before) if after is not None else None))
        (out / 'EXIT_CODE').write_text(str(code) + '\n')
        write(out / 'status.json', dict(phase=status, completed=len(records),
            running=[], pending=len(pending), finished_utc=now(), error=error))
    print(json.dumps({'status': status, 'jobs': len(records), 'external_exit': code}))


if __name__ == '__main__':
    main()

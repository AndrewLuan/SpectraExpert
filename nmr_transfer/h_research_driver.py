"""功能概述：执行本轮预声明研究DAG，持久保存逐任务及外层真实退出证据。

execute按依赖、GPU配额和截止时刻调度argv数组，不使用shell。每个job必须有独占
输出目录和科学脚本自身EXIT_CODE；父进程wait所得退出码与内部值同时为零才完成。
supervise作为独立外层等待driver，保存完整argv/PID/UTC/队列和源码SHA及外部EXIT。
exclusive文件阻止重复启动；只管理本进程创建的子任务，不触碰其他用户进程。
本模块只读既有科学输入、写本轮输出和日志，无网络或外部API，不选科学方案。
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .h_research_contract import atomic_json, input_fingerprints


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def deadline_passed(value):
    return bool(value) and dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def read_exit(path):
    path = Path(path)
    return int(path.read_text().strip()) if path.exists() else None


def validate_queue(queue):
    if queue.get('schema') != 'nmr_research_jobs_20261022' or not queue.get('jobs'):
        raise ValueError('research_invalid_job_queue')
    ids = [job['id'] for job in queue['jobs']]
    if len(ids) != len(set(ids)):
        raise ValueError('research_duplicate_job_id')
    outputs = [str(Path(j['out']).resolve()) for j in queue['jobs']]
    if len(outputs) != len(set(outputs)):
        raise ValueError('research_duplicate_job_output')
    for job in queue['jobs']:
        if (not isinstance(job['argv'], list) or not job['argv']
                or any(not isinstance(x, str) for x in job['argv'])
                or job['kind'] not in ('training', 'preparation', 'validation', 'sealed', 'audit')):
            raise ValueError('research_invalid_job_definition:' + job['id'])
        if any(dep not in ids or dep == job['id'] for dep in job.get('deps', [])):
            raise ValueError('research_invalid_dependency:' + job['id'])
    reached = set()
    while True:
        ready = {j['id'] for j in queue['jobs'] if set(j.get('deps', [])) <= reached}
        if ready <= reached:
            break
        reached |= ready
    if reached != set(ids):
        raise ValueError('research_cyclic_job_queue')


def execute(args):
    folder = Path(args.out); folder.mkdir(parents=True, exist_ok=True)
    with (folder/'driver.exclusive').open('x') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), started_utc=utc()))+'\n')
    queue = json.loads(Path(args.queue).read_text())
    validate_queue(queue)
    pinned = input_fingerprints([args.queue, __file__, Path(__file__).with_name('h_research_contract.py')])
    pinned.update(queue.get('source_inputs_sha256', {}))
    if input_fingerprints(pinned) != pinned:
        raise ValueError('research_driver_queue_source_changed')
    archive = folder/'source_snapshot'; archive.mkdir(exist_ok=False)
    archive_index = {}
    for source, digest in pinned.items():
        if Path(source).suffix == '.py':
            target = archive/(digest+'_'+Path(source).name)
            target.write_bytes(Path(source).read_bytes())
            archive_index[source] = dict(sha256=digest, archived_path=str(target))
    atomic_json(folder/'source_snapshot.json', archive_index)
    atomic_json(folder/'driver_start.json', dict(pid=os.getpid(), argv=sys.argv, utc=utc(), inputs_sha256=pinned,
                                               gpus=args.gpus, max_cpu=args.max_cpu))
    state = {job['id']: dict(state='pending', external_exit=None, internal_exit=None) for job in queue['jobs']}
    active = {}
    available = args.gpus.split(',') if args.gpus else []
    try:
        while True:
            changed = False
            for identifier, item in list(active.items()):
                result = item['process'].poll()
                if result is None:
                    continue
                job = item['job']; item['log'].close()
                internal = read_exit(Path(job['out'])/'EXIT_CODE')
                success = result == 0 and internal == 0
                state[identifier].update(state='complete' if success else 'failed', finished_utc=utc(),
                                         external_exit=result, internal_exit=internal)
                atomic_json(folder/(identifier+'.receipt.json'), state[identifier])
                if item['gpu'] is not None:
                    available.append(item['gpu']); available.sort(key=int)
                del active[identifier]; changed = True
            for job in queue['jobs']:
                identifier = job['id']
                if state[identifier]['state'] != 'pending':
                    continue
                deps = [state[dep]['state'] for dep in job.get('deps', [])]
                if any(value in ('failed', 'blocked', 'deadline') for value in deps):
                    state[identifier].update(state='blocked', reason='dependency_not_complete'); changed = True
                    continue
                if any(value != 'complete' for value in deps):
                    continue
                if deadline_passed(queue.get('end_utc')) or (job['kind'] == 'training' and deadline_passed(queue.get('training_cutoff_utc'))):
                    state[identifier].update(state='deadline', reason='admission_deadline'); changed = True
                    continue
                need_gpu = job.get('gpu', True)
                if need_gpu and not available:
                    continue
                if not need_gpu and sum(v['gpu'] is None for v in active.values()) >= args.max_cpu:
                    continue
                for marker in job.get('requires_exit0', []):
                    if read_exit(marker) != 0:
                        raise ValueError('research_external_dependency_not_exit0:' + marker)
                if Path(job['out']).exists():
                    raise ValueError('research_job_output_already_exists:' + job['out'])
                if input_fingerprints(job.get('inputs_sha256', {})) != job.get('inputs_sha256', {}):
                    raise ValueError('research_job_input_changed:' + identifier)
                gpu = available.pop(0) if need_gpu else None
                env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                env['CUDA_VISIBLE_DEVICES'] = gpu if gpu is not None else ''
                logfile = folder/(identifier+'.log')
                stream = logfile.open('xb')
                process = subprocess.Popen(job['argv'], cwd=job.get('cwd'), env=env,
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                state[identifier].update(state='running', argv=job['argv'], cwd=job.get('cwd'),
                    pid=process.pid, gpu=gpu, started_utc=utc(), log=str(logfile), output=job['out'],
                    queue_source_sha256=pinned, requested_inputs_sha256=job.get('inputs_sha256', {}))
                active[identifier] = dict(process=process, log=stream, job=job, gpu=gpu)
                atomic_json(folder/(identifier+'.receipt.json'), state[identifier]); changed = True
            if changed:
                atomic_json(folder/'status.json', dict(utc=utc(), jobs=state, driver_pid=os.getpid()))
            if not active and all(v['state'] != 'pending' for v in state.values()):
                break
            if deadline_passed(queue.get('end_utc')) and active:
                for item in active.values():
                    if item['process'].poll() is None:
                        item['process'].terminate()
                # 只等待自己的子进程；真实退出码仍在下一轮poll记录。
            time.sleep(3)
        source_ok = input_fingerprints(pinned) == pinned
        code = 0 if source_ok and all(v['state'] == 'complete' for v in state.values()) else 1
        atomic_json(folder/'completion.json', dict(utc=utc(), jobs=state, inputs_unchanged=source_ok,
                                                success=code == 0, external_api_calls=0))
        (folder/'EXIT_CODE').write_text(str(code)+'\n')
        return code
    except BaseException as exc:
        for item in active.values():
            if item['process'].poll() is None:
                item['process'].terminate()
        for identifier, item in active.items():
            try:
                result = item['process'].wait(timeout=20)
            except subprocess.TimeoutExpired:
                item['process'].kill(); result = item['process'].wait()
            item['log'].close()
            state[identifier].update(state='interrupted', external_exit=result,
                internal_exit=read_exit(Path(item['job']['out'])/'EXIT_CODE'), finished_utc=utc())
            atomic_json(folder/(identifier+'.receipt.json'), state[identifier])
        atomic_json(folder/'failure.json', dict(error=repr(exc), utc=utc(), jobs=state))
        (folder/'EXIT_CODE').write_text('1\n')
        raise


def supervise(args):
    folder = Path(args.out); folder.mkdir(parents=True, exist_ok=True)
    with (folder/'launch.exclusive').open('x') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), utc=utc()))+'\n')
    argv = [sys.executable, '-m', 'nmr_transfer.h_research_driver', '--queue', args.queue,
            '--out', args.out, '--gpus', args.gpus, '--max-cpu', str(args.max_cpu)]
    queue = json.loads(Path(args.queue).read_text())
    receipt = dict(schema='research_outer_launch_v1', wrapper_pid=os.getpid(), argv=argv, started_utc=utc(),
                   queue_sha256=input_fingerprints([args.queue]), source_sha256=input_fingerprints([__file__]),
                   bundle_sha256=queue.get('source_inputs_sha256', {}), external_exit=None)
    with (folder/'driver.log').open('xb') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        receipt['child_pid'] = child.pid; atomic_json(folder/'launch_receipt.json', receipt)
        code = child.wait()
    receipt.update(external_exit=code, internal_exit=read_exit(folder/'EXIT_CODE'), finished_utc=utc())
    atomic_json(folder/'launch_receipt.json', receipt)
    (folder/'EXTERNAL_EXIT_CODE').write_text(str(code)+'\n')
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7'); parser.add_argument('--max-cpu', type=int, default=4)
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    if args.max_cpu < 1 or (args.gpus and (any(not x.isdigit() for x in args.gpus.split(','))
                                          or len(set(args.gpus.split(','))) != len(args.gpus.split(',')))):
        raise ValueError('research_invalid_resource_arguments')
    sys.exit(supervise(args) if args.supervise else execute(args))


if __name__ == '__main__':
    main()

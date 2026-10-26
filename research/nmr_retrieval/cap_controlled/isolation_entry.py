"""功能概述：按单个已冻结任务的白名单审计 Python 文件访问并记录内层真实退出。

main 解析阶段、绑定清单、允许张量/文件和任务自己的输出目录，再用 runpy 执行
唯一登记脚本。guard 包装 builtins.open/io.open，拒绝未登记张量及提前读取 test；
sha 核对白名单张量的输入指纹。日志保存 PID/PPID、argv、UTC、实际退出和访问。
本文件是 supervise_controlled_queue 的子入口；不声称拦截 C++ 全部系统调用。
只允许任务读写自己的新输出，不能把旧阶段的零 test 接触声明扩展到本轮。
"""
import argparse
import builtins
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import stat
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bindings', required=True)
    p.add_argument('--phase', choices=['train', 'validation', 'mechanism', 'test'], required=True)
    p.add_argument('--access-log', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--allow-read', action='append', default=[])
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    assert command
    allowed = {str(Path(x).resolve()) for x in a.allow_read}
    output = Path(a.output_root).resolve()
    root = Path(__file__).resolve().parent
    assert output != root and output.is_relative_to(root)
    assert Path(a.bindings).resolve().parent == root
    assert Path(a.access_log).resolve().is_relative_to(root / 'access_logs')
    old_open, old_io = builtins.open, io.open
    accesses, rejected, devices = {}, [], {}
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    code, error = 1, None

    def protected(path):
        name = Path(path).name.lower()
        return any(x in path.lower() for x in ('/sealed', '/test/', '/cohorts.pt')) or name in (
            'test.pt', 'test.json', 'test.jsonl', 'predictions.jsonl', 'sealed_test_report.md')

    def guard(opener, filename, mode='r', *args, **kwargs):
        if isinstance(filename, (str, bytes, os.PathLike)):
            path = str(Path(os.fsdecode(filename)).resolve())
            # dill/PyTorch 用此字符设备探测文件类型；它不保存科学输入或输出。
            if path == str(Path(os.devnull).resolve()) and stat.S_ISCHR(Path(path).stat().st_mode):
                key = str(mode)
                devices[key] = devices.get(key, 0) + 1
                return opener(filename, mode, *args, **kwargs)
            read_mode = not any(x in str(mode) for x in ('w', 'a', 'x')) or '+' in str(mode)
            write_mode = any(x in str(mode) for x in ('w', 'a', 'x', '+'))
            own = Path(path).is_relative_to(output)
            if write_mode and not own:
                rejected.append(path)
                raise PermissionError('write outside task output: ' + path)
            if read_mode:
                tensor = path.endswith(('.pt', '.pth', '.ckpt', '.safetensors'))
                sensitive = protected(path)
                permitted = path in allowed or own
                if (tensor and not permitted) or (sensitive and not own and not (a.phase == 'test' and path in allowed)):
                    rejected.append(path)
                    raise PermissionError('unregistered scientific read: ' + path)
                if path in allowed or tensor or sensitive:
                    accesses[path] = accesses.get(path, 0) + 1
        return opener(filename, mode, *args, **kwargs)

    try:
        binding = json.loads(Path(a.bindings).read_text())
        pins = binding['input_sha256']
        # 所有路径先完成阶段准入，再读取任意白名单文件的字节。
        for path in allowed:
            if path not in pins or (protected(path) and a.phase != 'test'):
                rejected.append(path)
                raise PermissionError('pre-hash read admission refused: ' + path)
        for path in allowed:
            assert sha(path) == pins[path], path
        builtins.open = lambda filename, mode='r', *x, **kw: guard(old_open, filename, mode, *x, **kw)
        io.open = lambda filename, mode='r', *x, **kw: guard(old_io, filename, mode, *x, **kw)
        sys.argv = command
        runpy.run_path(command[0], run_name='__main__')
        code = 0
    except SystemExit as exc:
        code = exc.code if type(exc.code) is int else (0 if exc.code is None else 1)
        if code:
            error = repr(exc)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        builtins.open, io.open = old_open, old_io
        d = dict(schema='cap10_python_access_receipt_20261024', phase=a.phase,
                 pid=os.getpid(), parent_pid=os.getppid(), argv=command,
                 started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 internal_exit=code, error=error, allowed_reads=sorted(allowed),
                 recorded_reads=accesses, rejected_paths=rejected,
                 permitted_devnull_modes=devices,
                 boundary='Python open wrappers only; no claim of C++ syscall completeness')
        Path(a.access_log).parent.mkdir(parents=True, exist_ok=True)
        Path(a.access_log).write_text(json.dumps(d, indent=2) + '\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()

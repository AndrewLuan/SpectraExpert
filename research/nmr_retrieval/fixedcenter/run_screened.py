#!/usr/bin/env python3
"""功能概述：为本轮离线 NMR 自主研究执行经 autoresearch 审核的命令。

main 接收 --cwd 与余下命令参数，先用技能 scripts/orchestrate.sh screen-cmd
检查完整参数，再以参数数组执行，不经 shell 插值。每条命令、起止时间和退出码
写入本目录 commands.jsonl；不会主动访问 API、推送代码或修改研究输入。
研究脚本的 SSH 子命令也需在调用前通过此入口检查。异常保留退出码和原始产物。
"""
import argparse
import datetime
import json
import pathlib
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cwd', default=None)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('missing command')
    gate = '/Users/andrewluan/.codex/skills/autoresearch/scripts/orchestrate.sh'
    check = subprocess.run([gate, 'screen-cmd', shlex.join(command)], capture_output=True, text=True)
    if check.returncode or check.stdout.strip() != 'ok':
        raise SystemExit('autoresearch command refused: ' + check.stdout + check.stderr)
    root = pathlib.Path(__file__).resolve().parent
    now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = dict(started_utc=now(), cwd=args.cwd, command=command, screened=True)
    result = subprocess.run(command, cwd=args.cwd)
    record.update(finished_utc=now(), exit_code=result.returncode)
    with (root / 'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

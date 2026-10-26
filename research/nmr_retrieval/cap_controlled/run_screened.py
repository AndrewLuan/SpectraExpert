"""功能概述：为 C10/C5 受控研究执行命令安全检查并记录真实退出码。
main 用 argv 数组运行通过 autoresearch screen-cmd 的本地或 SSH 命令，日志
只追加到本阶段目录。具体训练、validation 和 test 的资格分别由 protocol、
独审及冻结清单控制；本包装器的 ok 只表示命令安全筛查通过，不授予派发资格。
本阶段禁止外部 LLM API、推送、部署、删除和正式系统变更。
"""
import argparse,datetime,json,shlex,subprocess
from pathlib import Path
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cwd',default=None)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:parser.error('missing command')
    gate='/Users/andrewluan/.codex/skills/autoresearch/scripts/orchestrate.sh'
    check=subprocess.run([gate,'screen-cmd',shlex.join(command)],capture_output=True,text=True)
    if check.returncode or check.stdout.strip()!='ok':raise SystemExit('autoresearch command refused: '+check.stdout+check.stderr)
    now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
    record=dict(started_utc=now(),cwd=args.cwd,command=command,screened=True)
    result=subprocess.run(command,cwd=args.cwd)
    record.update(finished_utc=now(),exit_code=result.returncode)
    with (Path(__file__).resolve().parent/'commands.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    raise SystemExit(result.returncode)
if __name__=='__main__':main()


"""功能概述：经本轮筛查器启动只读JJ身份扫描并保存真实退出凭据。
调用eligibility_identity_scan.py作为SSH标准输入，输出仅写本轮eligibility_*文件。
main保留argv/PID/UTC/sourceSHA/内外退出码；不修改JJ数据或旧轮封存文件。
"""
import subprocess,sys,json,hashlib,os
from pathlib import Path
from datetime import datetime,timezone
d=Path(__file__).resolve().parent
source=d/'eligibility_identity_scan.py'
argv=[sys.executable,str(d/'run_screened.py'),'--','ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','JJ','/home/user/.venvs/spectrexpert/bin/python -']
receipt={'argv':argv,'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
(d/'eligibility_launch_receipt.json').write_text(json.dumps(receipt,indent=2))
with source.open('rb') as inp,(d/'eligibility_identity_result.json').open('wb') as out,(d/'eligibility_identity_stderr.log').open('wb') as err:
 p=subprocess.Popen(argv,stdin=inp,stdout=out,stderr=err);receipt['child_pid']=p.pid
 (d/'eligibility_launch_receipt.json').write_text(json.dumps(receipt,indent=2))
 rc=p.wait()
receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(),actual_external_exit_code=rc,result_sha256=hashlib.sha256((d/'eligibility_identity_result.json').read_bytes()).hexdigest())
(d/'eligibility_launch_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
raise SystemExit(rc)

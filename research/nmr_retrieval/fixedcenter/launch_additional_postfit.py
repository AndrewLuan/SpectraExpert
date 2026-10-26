"""功能概述：只由根会话显式调用一次的additional postfit本地operator。
remote_main先核真实12任务driver双退出、completion/status/逐receipt、85产物存在、
250源pin、固定queue字节和UTC时限，再以独占claim仅派发一次CPU supervised driver。
main通过run_screened.py执行单次有界SSH，记录真实外部退出、stdout/stderr及operator/
queue SHA；每次尝试独占本地和远端receipt，失败不重试。无--execute时仅输出准备状态。
本文件不修改科学源、追加训练queue或其pins；不创建任何训练、GPU或sealed测试任务。
"""
import argparse
import datetime
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

D=Path(__file__).resolve().parent
LOCAL_PY='/Users/andrewluan/.venvs/spectrexpert-local/bin/python'
REMOTE_PY='/home/user/.venvs/spectrexpert/bin/python'
R='/home/user/IRfingerprints/SpectrExpert'
NR=R+'/outputs/nmr_pool_ranking_20261023'
NAME='additional_postfit_20261023T1205Z'
QUEUE_SHA='8c0ad960af2d8f5dcc927786e3d120f142f578f00ec48f6f637cb80188776468'
UPSTREAM_SHA='d1429dd48c70f0e85b840e605c4d0d15ef2c2bbfe21ac0b1687b7fc1c0522951'
DRIVER_SHA='611152435705e7d9406ff3e70fc3e8aad55aae4a4fdea104ba9897d9b65070c0'


def remote_main(payload):
    import datetime
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    def need(value,message):
        if not value:raise ValueError(message)
    def stamp():return datetime.datetime.now(datetime.timezone.utc)
    def utc(value):
        result=datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
        need(result.tzinfo is not None,'UTC_timezone_required')
        return result.astimezone(datetime.timezone.utc)
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
        return h.hexdigest()
    def read(path):return json.loads(Path(path).read_text())
    def double(record):
        need(all(type(record.get(k)) is int and record[k]==0 for k in ('external_exit','internal_exit')),
             'strict_integer_double_EXIT0_required')
    def exit0(path):need(Path(path).is_file() and Path(path).read_text().strip()=='0','EXIT_not_zero:'+str(path))
    nr=Path(payload['root']);name=payload['name']
    attempt=nr/'operator_attempts'/(name+'_'+payload['attempt']+'.json')
    attempt.parent.mkdir(exist_ok=True)
    record=dict(status='PREFLIGHT',operator_sha256=payload['operator_sha256'],queue_sha256=payload['queue_sha256'],
                attempt=payload['attempt'],started_utc=stamp().isoformat(),scientific_launch_count=0)
    with attempt.open('x') as stream:json.dump(record,stream,indent=2)
    def save():
        temp=attempt.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(attempt)
    try:
        qpath=nr/(name+'_queue.json');need(sha(qpath)==payload['queue_sha256'],'postfit_queue_SHA')
        queue=read(qpath);need(queue['schema']=='nmr_research_jobs_20261022','postfit_queue_schema')
        need(len(queue['source_inputs_sha256'])==250,'expected250_sourcepins')
        jobs=queue['jobs'];need(len(jobs)==2 and all(j['kind']=='audit' and j['gpu'] is False for j in jobs),
                               'strict2CPUjobs')
        need(jobs[0]['deps']==[] and jobs[1]['deps']==[jobs[0]['id']],'serial_audit_summary')
        need([Path(j['argv'][1]).name for j in jobs]==['audit_pool_fixed_center_training.py',
             'summarize_fixed_center_validation.py'],'exact_postfit_tools')
        upath=nr/'additional_fixedcenter_queue_20261023T1205Z.json'
        need(sha(upath)==payload['upstream_sha256'],'actual_upstream_queue_SHA')
        upstream=read(upath);ujobs=upstream['jobs']
        need(len(ujobs)==len({j['id'] for j in ujobs})==12,'upstream12unique')
        need(sum(j['kind']=='training' for j in ujobs)==4 and sum(j['kind']=='validation' for j in ujobs)==8,
             'upstream4fit8eval')
        run=nr/'runs/additional_fixedcenter_20261023T1205Z'
        for marker in ('EXIT_CODE','EXTERNAL_EXIT_CODE'):exit0(run/marker)
        launch,driver,completion,status=[read(run/n) for n in
            ('launch_receipt.json','driver_start.json','completion.json','status.json')]
        double(launch)
        need(launch['queue_sha256']=={str(upath):payload['upstream_sha256']},'actual_launch_queue_path')
        need(launch['bundle_sha256']==upstream['source_inputs_sha256'],'actual_launch_bundle')
        expected=dict(upstream['source_inputs_sha256'],**{str(upath):payload['upstream_sha256']})
        need(driver['inputs_sha256']==expected,'actual_driver_input_binding')
        need(type(driver['pid']) is int and driver['pid']>0 and type(launch['child_pid']) is int
             and driver['pid']==launch['child_pid'] and driver['argv'][1:]==launch['argv'][3:],
             'actual_driver_identity')
        need(status['driver_pid']==driver['pid'],'status_driver_identity')
        need(completion.get('success') is True and completion.get('inputs_unchanged') is True,'completion_PASS')
        ids={j['id'] for j in ujobs}
        need(set(completion['jobs'])==ids and status['jobs']==completion['jobs'],'completion_status12_equal')
        begin,end=utc(driver['utc']),utc(completion['utc'])
        need(utc(launch['started_utc'])<=begin<=utc(status['utc'])<=end<=utc(launch['finished_utc'])<=stamp(),
             'driver_UTC_order')
        metadata_sha={str(run/n):sha(run/n) for n in
            ('launch_receipt.json','driver_start.json','completion.json','status.json','EXIT_CODE','EXTERNAL_EXIT_CODE')}
        required=[run/n for n in ('launch_receipt.json','driver_start.json','completion.json','EXIT_CODE','EXTERNAL_EXIT_CODE')]
        for job in ujobs:
            rp=run/(job['id']+'.receipt.json');receipt=read(rp);double(receipt)
            need(receipt==completion['jobs'][job['id']] and receipt['state']=='complete','receipt_exact_complete')
            need(receipt['argv']==job['argv'] and receipt['output']==job['out'] and receipt['cwd']==job['cwd'],
                 'receipt_invocation_binding')
            need(receipt['queue_source_sha256']==expected
                 and receipt['requested_inputs_sha256']==job['inputs_sha256'],'receipt_inputs_binding')
            need(begin<=utc(receipt['started_utc'])<=utc(receipt['finished_utc'])<=end,'receipt_UTC_order')
            for dep in job.get('deps',[]):
                need(utc(completion['jobs'][dep]['finished_utc'])<=utc(receipt['started_utc']),'dependency_UTC_order')
            exit0(Path(job['out'])/'EXIT_CODE');metadata_sha[str(rp)]=sha(rp);required.append(rp)
            names=('summary.json','protocol.json','history.jsonl','initial.pt','best.pt','last.pt',
                   'centers.json','center_rows.jsonl','EXIT_CODE') if job['kind']=='training' else (
                   'summary.json','protocol.json','predictions.jsonl','EXIT_CODE')
            required.extend(Path(job['out'])/n for n in names)
        need(len(required)==len(set(required))==85 and all(p.is_file() for p in required),'85future_files_exist')
        actual={path:sha(path) for path in queue['source_inputs_sha256']}
        need(actual==queue['source_inputs_sha256'],'250sourcepins_changed')
        driverpath=Path(payload['repository'])/'nmr_transfer/h_research_driver.py'
        need(sha(driverpath)==payload['driver_sha256'],'pinned_driver_SHA')
        out=nr/'runs'/name;claim=nr/'runs'/(name+'.operator_claim.json')
        need(not out.exists() and not claim.exists() and all(not Path(j['out']).exists() for j in jobs),
             'postfit_already_exists_or_claimed')
        for job in jobs:
            need(job['argv'][job['argv'].index('--queue')+1]==str(upath)
                 and job['argv'][job['argv'].index('--run')+1]==str(run),'postfit_actual_upstream_paths')
            for marker in job['requires_exit0']:exit0(marker) if marker!=jobs[0]['out']+'/EXIT_CODE' else None
        now=stamp();root_now=utc(payload['root_utc']);conservative=max(now,root_now)+datetime.timedelta(seconds=60)
        deadline=utc('2026-10-23T14:29:41Z')
        need(conservative+datetime.timedelta(seconds=1200)<deadline,'CPU1200s_plus60s_deadline')
        need(utc(queue['end_utc'])==deadline,'postfit_deadline_fixed')
        argv=[payload['python'],'-m','nmr_transfer.h_research_driver','--queue',str(qpath),
              '--out',str(out),'--gpus','','--max-cpu','1','--supervise']
        record.update(status='PREFLIGHT_PASS',remote_utc=now.isoformat(),conservative_utc=conservative.isoformat(),
            forecast_complete_before=(conservative+datetime.timedelta(seconds=1200)).isoformat(),
            required_future_files=85,source_inputs_sha256=actual,upstream_metadata_sha256=metadata_sha,
            argv=argv,cwd=payload['repository'],claim=str(claim),run_out=str(out),queue_path=str(qpath))
        # 独占claim在Popen之前落盘。任何失败均保留，禁止自动重试实际派发。
        with claim.open('x') as stream:json.dump(record,stream,indent=2)
        save()
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        logpath=nr/'runs'/(name+'.supervisor.log')
        with logpath.open('xb') as log:
            child=subprocess.Popen(argv,cwd=payload['repository'],env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record.update(status='DISPATCHED_NOT_COMPLETE',scientific_launch_count=1,pid=child.pid,
                      dispatched_utc=stamp().isoformat(),supervisor_log=str(logpath))
        save();print(json.dumps(record),flush=True)
    except BaseException as exc:
        record.update(status='FAILED_NO_RETRY',error=repr(exc),finished_utc=stamp().isoformat())
        save();print(json.dumps(record),flush=True);raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true',help='root only; one real remote dispatch attempt')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',operator=str(Path(__file__).resolve()))));return 0
    now=datetime.datetime.now(datetime.timezone.utc)
    attempt=now.strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex
    folder=D/'postfit_operator_attempts';folder.mkdir(exist_ok=True)
    receipt=folder/(attempt+'.json')
    operator_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload=dict(root=NR,repository=R,name=NAME,attempt=attempt,operator_sha256=operator_sha,
        queue_sha256=QUEUE_SHA,upstream_sha256=UPSTREAM_SHA,driver_sha256=DRIVER_SHA,
        python=REMOTE_PY,root_utc=now.isoformat())
    record=dict(status='REQUESTED_NOT_RETRIED',operator_sha256=operator_sha,payload=payload,
                started_utc=now.isoformat(),remote_launch_attempts=1)
    with receipt.open('x') as stream:json.dump(record,stream,indent=2)
    code=inspect.getsource(remote_main)+'\nimport json,sys\nremote_main(json.loads(sys.argv[1]))\n'
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','JJ',
         shlex.join([REMOTE_PY,'-c',code,json.dumps(payload,separators=(',',':'))])]
    command=[LOCAL_PY,str(D/'run_screened.py'),'--',*ssh]
    record['screened_argv']=command
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=180)
        record.update(actual_exit=result.returncode,stdout=result.stdout,stderr=result.stderr,
                      status='SSH_RETURNED_NOT_JOB_COMPLETION')
        returncode=result.returncode
    except BaseException as exc:
        record.update(status='LOCAL_FAILURE_NO_RETRY',error=repr(exc),actual_exit=None,
            stdout=getattr(exc,'stdout',None),stderr=getattr(exc,'stderr',None),
            remote_state='UNKNOWN_DO_NOT_RETRY_WITHOUT_OWNER_CHECK')
        returncode=1
    record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    record['operator_unchanged']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==operator_sha
    receipt.write_text(json.dumps(record,indent=2,default=str)+'\n')
    print(json.dumps(dict(attempt_receipt=str(receipt),status=record['status'],
                         actual_exit=record.get('actual_exit'),operator_sha256=operator_sha)))
    return returncode


if __name__=='__main__':sys.exit(main())

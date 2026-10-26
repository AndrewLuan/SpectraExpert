"""功能概述：唯一source-only部署并执行已审native sealed工具的JJ50人工回归。
先核本地审查SHA，再远端核正在使用的val4/cap/train/operator小源码不变，执行rsync
dry-run→sync→逐文件SHA。只跑50个已审人工fixture/冷导入，CUDA_VISIBLE_DEVICES为空；
不打开实际test缓存、producer、预测或模型输出，不启动真实评价、训练或科学queue。
完整保留stdout/stderr、真实SSH退出、远端source首末SHA与客户端后验SSHack，不自动重试。
该文件仅写自己source-only回归目录与登记源码，不修改任何现有科学参数或运行产物。
"""
import argparse,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
REMOTE='/home/user/IRfingerprints/SpectrExpert/outputs/nmr_cap10_controlled_20261024T163633Z'
PY='/home/user/.venvs/spectrexpert/bin/python'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','JJ']
MODULES=['test_cap_test_evaluate_ranker','test_native_cost_test_precision','test_cap_test_evaluation_independent',
 'test_cap_test_native_independent','test_summarize_native_cost_results','test_summarize_native_cost_results_independent']
TRANSFER=['native_cost_evaluation/'+n for n in (
 'cap_test_evaluation_common.py','cap_test_evaluate_ranker.py','summarize_sealed_cap_results.py',
 'test_cap_test_evaluate_ranker.py','test_native_cost_test_precision.py','test_cap_test_evaluation_independent.py',
 'test_cap_test_native_independent.py','test_summarize_native_cost_results.py','test_summarize_native_cost_results_independent.py',
 'cap_test_evaluation_native_implementation_independent_review.json','sealed_cap_statistics_native_implementation_independent_review.json')]
TRANSFER+=['sealed_cap_pairmath.py','sealed_cap_matrix.py','test_summarize_sealed_cap_results_independent.py','test_summarize_sealed_cap_results.py',
 'cap_test_evaluation_common.py']  # 旧common只供冷负面fixture预载；生产入口强制native邻近common。
PROTECT=['cap_ranker.py','train_cap_ranker.py','isolation_entry.py','supervise_controlled_queue.py','launch_controlled_stage.py',
 'protocol.json','execution_bindings.json','train_after_import_fix_queue.json','cap_evaluation_common.py','evaluate_cap_ranker.py',
 'summarize_cap_validation.py','analyze_cap_mechanism.py','native_cost_validation_bindings.json',
 'validation_native_cost_queue.json','mechanism_native_cost_queue.json']
PROTECT+=['native_cost_evaluation/'+n for n in ('cap_evaluation_common.py','evaluate_cap_ranker.py','summarize_cap_validation.py',
 'analyze_cap_mechanism.py','test_cap_evaluation_entry.py','test_evaluate_cap_ranker.py')]

def utc():return datetime.now(timezone.utc).isoformat()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:f.write(json.dumps(v,indent=2)+'\n')
def run(argv,stem,out,timeout=180):
    began=utc();r=subprocess.run(argv,capture_output=True,timeout=timeout)
    (out/(stem+'.stdout.log')).write_bytes(r.stdout);(out/(stem+'.stderr.log')).write_bytes(r.stderr)
    record=dict(argv=argv,started_utc=began,finished_utc=utc(),actual_exit=r.returncode,
      stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),stderr_sha256=hashlib.sha256(r.stderr).hexdigest())
    write(out/(stem+'.json'),record)
    if r.returncode:raise RuntimeError(stem+'_actual_exit_'+str(r.returncode))
    return r
def ssh(code,stem,out,timeout=180):
    return run(SSH+[shlex.join([PY,'-B','-c',code])],stem,out,timeout)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute-source-only',action='store_true')
    ap.add_argument('--attempt-name',choices=('native_sealed_JJ50','native_sealed_JJ50_fixture_closure'),default='native_sealed_JJ50');a=ap.parse_args()
    if not a.execute_source_only:raise ValueError('explicit_source_only_execute_required')
    out=ROOT/a.attempt_name;out.mkdir(exist_ok=False)
    receipt=dict(schema='cap10_native_sealed_JJ50_source_only_deployment_20261025',started_utc=utc(),
      operator_sha256=digest(__file__),actual_test_inference=False,actual_scientific_job_started=False,automatic_retry=False,
      synthetic_operator_resource_scope='CPU fake-child fixtures only; no GPU-readiness or multi-wave production admission',
      known_GPU_readiness_failure_requires_separate_new_operator_review=True)
    try:
        local={rel:digest(ROOT/rel) for rel in TRANSFER}
        protect={rel:digest(ROOT/rel) for rel in PROTECT if (ROOT/rel).is_file()}
        before=json.loads((ROOT/'native_sealed_independent_final_test_execution.json').read_text())
        assert before['actual_exit']==0 and before['source_unchanged'] is True
        for rel,h in local.items():
            if rel=='cap_test_evaluation_common.py':
                prior=json.loads((ROOT/'cap_test_evaluation_implementation_independent_review.json').read_text())
                assert prior['status']=='PASS' and prior['reviewed_sha256'][str(ROOT/rel)]==h
            elif rel.endswith('.py'):assert before['source_sha256'][str(ROOT/rel)]==h
        for name in ('cap_test_evaluation_native_implementation_independent_review.json','sealed_cap_statistics_native_implementation_independent_review.json'):
            review=json.loads((ROOT/'native_cost_evaluation'/name).read_text());assert review['status']=='PASS' and review['independent'] is True
        # Native metadata was built remotely; its local copies live under native_cost_pre_admission.
        bundle=json.loads((ROOT/'native_cost_pre_admission/artifact_map.json').read_text())
        for rel in ('native_cost_validation_bindings.json','validation_native_cost_queue.json','mechanism_native_cost_queue.json'):
            remote=REMOTE+'/'+rel;protect[rel]=bundle['sha256'][remote]
        remote_protect={REMOTE+'/'+rel:h for rel,h in protect.items()}
        binding=json.loads((ROOT/'execution_bindings.json').read_text())
        fixed=[(path,h) for path,h in binding['source_sha256'].items() if Path(path).name=='h_pool_fixed_center_ranker.py']
        assert len(fixed)==1
        remote_protect[fixed[0][0]]=fixed[0][1]
        assert set(protect)==set(PROTECT)
        write(out/'manifest.json',dict(local_source_sha256=local,protected_source_sha256=remote_protect,modules=MODULES))
        check='import pathlib,hashlib,json; expected='+repr(remote_protect)+'; actual={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in expected}; assert actual==expected; print(json.dumps(actual,sort_keys=True))'
        ssh(check,'protected_before',out)
        # Preserve any earlier unpinned source draft that a precise final deployment replaces.
        archive='import pathlib,hashlib,json,shutil; root=pathlib.Path('+repr(REMOTE)+'); expected='+repr(local)+'; saved={};\nfor rel,h in expected.items():\n p=root/rel\n if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=h:\n  dest=root/'+repr('source_only_history/'+a.attempt_name+'_before')+'/rel\n  dest.parent.mkdir(parents=True,exist_ok=True)\n  assert not dest.exists()\n  shutil.copy2(p,dest);saved[rel]=hashlib.sha256(dest.read_bytes()).hexdigest()\nprint(json.dumps(saved,sort_keys=True))'
        ssh(archive,'preserve_prior_unpinned_drafts',out)
        files=out/'files.txt';files.write_text('\n'.join(TRANSFER)+'\n')
        base=['rsync','-a','--checksum','--itemize-changes','--files-from='+str(files),'-e','ssh -o BatchMode=yes -o ConnectTimeout=15',str(ROOT)+'/', 'JJ:'+REMOTE+'/']
        run(base[:2]+['--dry-run']+base[2:],'source_dry_run',out)
        run(base,'source_sync',out)
        all_expected={**remote_protect,**{REMOTE+'/'+rel:h for rel,h in local.items()}}
        verify='import pathlib,hashlib,json; expected='+repr(all_expected)+'; actual={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in expected}; assert actual==expected; print(json.dumps(actual,sort_keys=True))'
        ssh(verify,'all_remote_SHA_after_sync',out)
        testcode='''import pathlib,hashlib,json,subprocess,os,sys,datetime
root=pathlib.Path(REMOTE); directory=root/'native_cost_evaluation'; target=root/'synthetic'/ATTEMPT; target.mkdir(parents=True,exist_ok=False)
expected=EXPECTED
get=lambda:{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in expected}
before=get();assert before==expected
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
argv=[sys.executable,'-B','-m','unittest','-v']+MODULES
began=datetime.datetime.now(datetime.timezone.utc).isoformat();process=subprocess.Popen(argv,cwd=directory,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
stdout,stderr=process.communicate(timeout=180);finished=datetime.datetime.now(datetime.timezone.utc).isoformat()
(target/'stdout.log').write_bytes(stdout);(target/'stderr.log').write_bytes(stderr)
after=get();value=dict(schema='cap10_native_sealed_JJ50_synthetic_execution_20261025',argv=argv,cwd=str(directory),test_pid=process.pid,parent_pid=os.getpid(),started_utc=began,finished_utc=finished,actual_exit=process.returncode,actual_ssh_exit=None,tests=50,source_sha256=before,source_end_sha256=after,source_inputs_unchanged=before==after,env_overrides={k:env[k] for k in ('PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},stdout_sha256=hashlib.sha256(stdout).hexdigest(),stderr_sha256=hashlib.sha256(stderr).hexdigest(),actual_test_inference=False,actual_scientific_job_started=False,automatic_retry=False)
(target/'execution.pre_ssh_ack.json').write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value));assert process.returncode==0 and before==after
'''.replace('REMOTE',repr(REMOTE)).replace('EXPECTED',repr(all_expected)).replace('MODULES',repr(MODULES)).replace('ATTEMPT',repr(a.attempt_name))
        r=ssh(testcode,'JJ50_actual_synthetic_execution',out,timeout=240)
        observed=json.loads(r.stdout);assert observed['actual_exit']==0 and observed['source_inputs_unchanged'] is True
        ackcode='''import pathlib,hashlib,json,datetime
root=pathlib.Path(REMOTE)/'synthetic'/ATTEMPT; old=root/'execution.pre_ssh_ack.json'; value=json.loads(old.read_text());assert value['actual_ssh_exit'] is None
ack=dict(actual_ssh_exit=0,previous_receipt_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),client_observed_command=COMMAND,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/'ssh_ack.json').write_text(json.dumps(ack,indent=2)+'\\n');value.update(actual_ssh_exit=0,ssh_ack_sha256=hashlib.sha256((root/'ssh_ack.json').read_bytes()).hexdigest());(root/'execution.json').write_text(json.dumps(value,indent=2)+'\\n');print(json.dumps(dict(execution_sha256=hashlib.sha256((root/'execution.json').read_bytes()).hexdigest(),ack_sha256=hashlib.sha256((root/'ssh_ack.json').read_bytes()).hexdigest())))
'''.replace('REMOTE',repr(REMOTE)).replace('COMMAND',repr(str(out/'JJ50_actual_synthetic_execution.json'))).replace('ATTEMPT',repr(a.attempt_name))
        ssh(ackcode,'ack_actual_SSH0',out)
        copy=['rsync','-a','--itemize-changes','-e','ssh -o BatchMode=yes -o ConnectTimeout=15','JJ:'+REMOTE+'/synthetic/'+a.attempt_name+'/',str(out/'remote')+'/']
        run(copy,'copy_actual_receipts',out)
        final=json.loads((out/'remote/execution.json').read_text());ack=json.loads((out/'remote/ssh_ack.json').read_text())
        assert final['actual_exit']==final['actual_ssh_exit']==ack['actual_ssh_exit']==0
        assert final['source_sha256']==final['source_end_sha256']==all_expected
        assert digest(out/'remote/stdout.log')==final['stdout_sha256'] and digest(out/'remote/stderr.log')==final['stderr_sha256']
        assert digest(out/'remote/execution.pre_ssh_ack.json')==ack['previous_receipt_sha256']
        assert digest(out/'remote/ssh_ack.json')==final['ssh_ack_sha256']
        assert 'Ran 50 tests in ' in (out/'remote/stderr.log').read_text() and '\nOK\n' in (out/'remote/stderr.log').read_text()
        assert all(digest(ROOT/rel)==h for rel,h in local.items())
        ssh(verify,'all_remote_SHA_end',out)
        receipt.update(status='PASS',actual_exit=0,remote_execution_sha256=digest(out/'remote/execution.json'),
          local_source_sha256=local,protected_source_sha256=remote_protect,tests=50,source_input_unchanged=True,finished_utc=utc())
    except BaseException as exc:
        receipt.update(status='FAILED',actual_exit=1,error=repr(exc),finished_utc=utc());raise
    finally:write(out/'deployment_and_execution.json',receipt)
    print(json.dumps(dict(status='PASS',receipt=str(out/'deployment_and_execution.json'),sha256=digest(out/'deployment_and_execution.json'))))
if __name__=='__main__':main()

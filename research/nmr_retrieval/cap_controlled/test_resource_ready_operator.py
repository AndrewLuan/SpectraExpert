"""功能概述：新资源就绪operator的完整人工子进程fixture与有界条件等待验证。
继承旧fixture的admission/claim/失败清理/重入/多wave行为；fake nvidia-smi只返回
人工状态，真实子进程不导入模型、不读科学cache。等待READY/RELEASE等状态，
明确30秒测试外层预算与stage/resource子预算；不靠固定sleep声称就绪。
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import test_controlled_operator_independent as old

HERE=Path(__file__).resolve().parent
SUPERVISOR='supervise_resource_ready_queue.py'
LAUNCHER='launch_resource_ready_stage.py'


class Fixture(old.SupervisorFixture):
    def write(self):
        for name in (SUPERVISOR,LAUNCHER):
            (self.root/name).write_bytes((HERE/name).read_bytes())
        self.q['operator_paths']={key:dict(path=str(self.root/name),sha256=self.sha(self.root/name)) for key,name in [('supervisor',SUPERVISOR),('launcher',LAUNCHER)]}
        self.q.setdefault('resource_wait',dict(maximum_seconds=2.,poll_seconds=.02))
        super().write()
        self.q['source_sha256'].update({str(self.root/name):self.sha(self.root/name) for name in (SUPERVISOR,LAUNCHER)})
        self.queue.write_text(json.dumps(self.q))
        self.review.write_text(json.dumps(dict(status='PASS',independent=True,admission_phase='train',reviewed_sha256={**self.q['source_sha256'],str(self.queue):self.sha(self.queue)})))

    def argv(self,execute=False,wrapper=False):
        argv=[sys.executable,'-B',str(self.root/(LAUNCHER if wrapper else SUPERVISOR)),'--queue',str(self.queue)]
        if not wrapper:argv+=['--out',str(self.out)]+(['--execute'] if execute else [])
        return argv

    def run(self,execute=False,wrapper=False):
        return subprocess.run(self.argv(execute,wrapper),capture_output=True,text=True,timeout=30,env=self.env())

    def env(self):
        return dict(os.environ,PATH=str(self.root/'fakebin')+os.pathsep+os.environ.get('PATH',''))

    def gpu(self,plan):
        folder=self.root/'fakebin';folder.mkdir(exist_ok=True)
        (self.root/'probe_plan.json').write_text(json.dumps(plan))
        script=folder/'nvidia-smi'
        script.write_text('#!'+sys.executable+'\n'+FAKE_NVIDIA)
        script.chmod(0o755)

    def gpu_jobs(self):
        for i,job in enumerate(self.q['jobs']):job['gpu']=i%2
        self.write()


FAKE_NVIDIA=r'''import json,sys
from pathlib import Path
root=Path(__file__).resolve().parent.parent
index=sys.argv[sys.argv.index("-i")+1]
plan=json.loads((root/"probe_plan.json").read_text())
counts=json.loads((root/"probe_count.json").read_text()) if (root/"probe_count.json").exists() else {}
memory=any("query-gpu=" in a for a in sys.argv)
if memory:
 counts[index]=counts.get(index,0)+1;(root/"probe_count.json").write_text(json.dumps(counts))
states=plan.get(index,[{"memory_mib":0,"utilization_percent":0,"compute_pids":[]}])
state=states[min(max(0,counts.get(index,1)-1),len(states)-1)]
if state.get('hang'):
 import os,time
 (root/('probe_READY_'+index)).write_text(str(os.getpid()))
 while True:time.sleep(1)
if memory:print(str(state["memory_mib"])+", "+str(state["utilization_percent"]))
else:
 pids=state["compute_pids"]
 if pids=="retired":
  recs=[json.loads(p.read_text()) for p in (root/"runs").glob("*/jobs/*.receipt.json")]
  pids=[r["pid"] for r in recs if str(r["gpu"])==index][-1:]
 print("\n".join(str(p) for p in pids))
'''


def wait_until(check,deadline,message,process=None):
    while True:
        value=check()
        if value:return value
        if process is not None and process.poll() is not None:
            raise AssertionError(message+': process ended '+str(process.returncode))
        remaining=deadline-time.monotonic()
        if remaining<=0:raise AssertionError(message)
        time.sleep(min(.01,remaining))


class InheritedSupervisorTests(old.SupervisorAttacks):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.f=Fixture(self.root)

    def test_concurrent_wrappers_have_one_owner_and_one_driver(self):
        argv=self.f.argv(wrapper=True)
        children=[subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=self.f.env()) for _ in range(2)]
        deadline=time.monotonic()+30
        try:
            for p in children:p.communicate(timeout=max(.01,deadline-time.monotonic()))
        finally:
            for p in children:
                if p.poll() is None:p.kill();p.wait(timeout=5)
        self.assertEqual(sorted(p.returncode for p in children),[0,1])
        claim=json.loads((self.root/'train.launch_claim.json').read_text());launch=json.loads((self.f.out/'launch_receipt.json').read_text())
        self.assertEqual(claim['wrapper_pid'],launch['wrapper_pid'])
        self.assertEqual(len(json.loads((self.f.out/'completion.json').read_text())['jobs']),8)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.f=Fixture(self.root)
    def tearDown(self):self.tmp.cleanup()
    def result(self):return json.loads((self.f.out/'completion.json').read_text())
    def idle(self):return dict(memory_mib=0,utilization_percent=0,compute_pids=[])
    def busy(self,**kw):return dict(self.idle(),utilization_percent=8,**kw)

    def test_gpu_delay_and_retired_pid_converge_all_eight_children(self):
        self.f.gpu_jobs();idle=self.idle();busy=self.busy()
        self.f.gpu({str(g):[idle,busy,dict(busy,compute_pids='retired'),idle] for g in (0,1)})
        result=self.f.run(wrapper=True)
        self.assertEqual(result.returncode,0,result.stderr)
        done=self.result();self.assertEqual(done['status'],'COMPLETE');self.assertEqual(len(done['jobs']),8)
        self.assertTrue(any(e['probes']>=3 for e in done['resource_events']))
        self.assertTrue(all(e['snapshot']==idle for e in done['resource_events']))
        samples=[json.loads(x) for x in (self.f.out/'resource_samples.jsonl').read_text().splitlines()]
        self.assertTrue(any(s['snapshot']['compute_pids'] for s in samples))
        for rec in done['jobs']:
            self.assertEqual((rec['internal_exit'],rec['external_exit']),(0,0))
            self.assertEqual(json.loads((Path(rec['job_out'])/'env.json').read_text())['cuda'],str(rec['gpu']))

    def test_foreign_process_rejected_and_never_terminated(self):
        foreign=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read()'],stdin=subprocess.PIPE)
        try:
            self.f.gpu_jobs();self.f.gpu({'0':[dict(self.idle(),compute_pids=[foreign.pid])]})
            result=self.f.run(execute=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('foreign_gpu_process',self.result()['error'])
            self.assertEqual(self.result()['jobs'],[]);self.assertIsNone(foreign.poll())
            self.assertFalse((self.root/'outputs/0').exists())
            sample=json.loads((self.f.out/'resource_samples.jsonl').read_text().splitlines()[0]);self.assertEqual(sample['snapshot']['compute_pids'],[foreign.pid])
        finally:foreign.stdin.close();foreign.wait(timeout=5)

    def test_resource_timeout_preserves_zero_dispatch_and_threshold(self):
        self.f.gpu_jobs();self.f.q['resource_wait']=dict(maximum_seconds=.15,poll_seconds=.02);self.f.write();self.f.gpu({'0':[self.busy()]})
        result=self.f.run(execute=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('resource_ready_timeout',self.result()['error'])
        self.assertEqual(self.result()['jobs'],[])

    def test_stage_deadline_takes_precedence_over_resource_budget(self):
        self.f.gpu_jobs();self.f.q['maximum_seconds']=.12;self.f.write();self.f.gpu({'0':[self.busy()]})
        result=self.f.run(execute=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('stage_deadline',self.result()['error'])

    def test_peer_failure_observed_while_other_gpu_waits(self):
        self.f.q['jobs'][0]['gpu']=0;self.f.q['jobs'][0]['command'][-1]='fail';self.f.q['jobs'][0]['argv'][-1]='fail'
        self.f.q['jobs'][1]['gpu']=1;self.f.write();self.f.gpu({'0':[self.idle()],'1':[self.busy()]})
        result=self.f.run(execute=True);self.assertNotEqual(result.returncode,0)
        self.assertEqual(len(self.result()['jobs']),1);self.assertEqual(self.result()['jobs'][0]['internal_exit'],7)
        self.assertNotIn('resource_ready_timeout',self.result()['error']);self.assertFalse((self.root/'outputs/1').exists())

    def test_cancel_during_resource_wait_no_child_and_failure_receipt(self):
        self.f.gpu_jobs();self.f.gpu({'0':[self.busy()]})
        process=subprocess.Popen(self.f.argv(execute=True),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=self.f.env())
        try:
            wait_until(lambda:(self.f.out/'resource_samples.jsonl').exists(),time.monotonic()+10,'no resource observation',process)
            process.send_signal(signal.SIGTERM);process.communicate(timeout=10)
            self.assertNotEqual(process.returncode,0);self.assertIn('operator_cancelled',self.result()['error'])
            self.assertEqual(self.result()['jobs'],[])
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=5)
            process.communicate(timeout=5)

    def test_wrapper_cancel_cleans_ready_owned_peer_and_pending(self):
        self.f.q['jobs'][0]['command'][-1]='wait';self.f.q['jobs'][0]['argv'][-1]='wait';self.f.q['max_parallel']=1;self.f.write()
        process=subprocess.Popen(self.f.argv(wrapper=True),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=self.f.env())
        try:
            ready=self.root/'outputs/0/READY'
            wait_until(ready.exists,time.monotonic()+10,'owned peer not READY',process)
            owned=int(ready.read_text());process.send_signal(signal.SIGTERM);process.communicate(timeout=20)
            self.assertNotEqual(process.returncode,0);done=self.result();self.assertEqual(done['status'],'FAILED')
            self.assertEqual(len(done['jobs']),1);self.assertTrue(done['jobs'][0]['terminated_after_peer_failure'])
            with self.assertRaises(ProcessLookupError):os.kill(owned,0)
            self.assertFalse((self.root/'outputs/1').exists())
            launch=json.loads((self.f.out/'launch_receipt.json').read_text());self.assertEqual(launch['forwarded_signals'],[signal.SIGTERM])
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=5)
            process.communicate(timeout=5)

    def test_operator_ref_alias_and_source_drift_block_before_claim(self):
        self.f.q['operator_paths']['supervisor']['path']=str(self.root/'supervise_controlled_queue.py')
        self.f.queue.write_text(json.dumps(self.f.q))
        result=self.f.run(execute=True);self.assertNotEqual(result.returncode,0);self.assertFalse((self.root/'train.operator_claim.json').exists())


    def test_exact_idle_threshold_and_retired_pid_reuse(self):
        module=old.module('supervise_resource_ready_queue',HERE/SUPERVISOR)
        self.assertTrue(module.resource_ready(dict(memory_mib=1023,utilization_percent=4,compute_pids=[]),set()))
        for sample in (dict(memory_mib=1024,utilization_percent=4,compute_pids=[]),dict(memory_mib=1023,utilization_percent=5,compute_pids=[])):
            self.assertFalse(module.resource_ready(sample,set()))
        stale=dict(memory_mib=1,utilization_percent=0,compute_pids=[99123])
        with patch.object(module,'pid_exists',return_value=False):
            self.assertFalse(module.resource_ready(stale,{99123}))
        with patch.object(module,'pid_exists',return_value=True):
            with self.assertRaisesRegex(AssertionError,'foreign_gpu_process'):module.resource_ready(stale,{99123})


    def test_hung_probe_shares_resource_deadline_and_is_reaped(self):
        self.f.gpu_jobs();self.f.q['resource_wait']=dict(maximum_seconds=2.,poll_seconds=.02);self.f.write()
        self.f.gpu({'0':[dict(self.idle(),hang=True)]})
        result=self.f.run(execute=True);self.assertNotEqual(result.returncode,0)
        self.assertIn('resource_ready_timeout',self.result()['error']);self.assertEqual(self.result()['jobs'],[])
        pid=int((self.root/'probe_READY_0').read_text())
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_peer_failure_interrupts_hung_probe_without_waiting_its_budget(self):
        self.f.q['jobs'][0]['gpu']=0;self.f.q['jobs'][0]['command'][-1]='fail';self.f.q['jobs'][0]['argv'][-1]='fail'
        self.f.q['jobs'][1]['gpu']=1;self.f.write();self.f.gpu({'0':[self.idle()],'1':[dict(self.idle(),hang=True)]})
        result=self.f.run(execute=True);self.assertNotEqual(result.returncode,0)
        done=self.result();self.assertEqual(len(done['jobs']),1);self.assertEqual(done['jobs'][0]['internal_exit'],7)
        self.assertNotIn('gpu_probe_deadline',done['error'])
        marker=self.root/'probe_READY_1'
        if marker.exists():
            with self.assertRaises(ProcessLookupError):os.kill(int(marker.read_text()),0)
        self.assertFalse((self.root/'outputs/1').exists())

    def test_signal_interrupts_hung_probe_and_starts_no_scientific_child(self):
        self.f.gpu_jobs();self.f.gpu({'0':[dict(self.idle(),hang=True)]})
        process=subprocess.Popen(self.f.argv(execute=True),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=self.f.env())
        try:
            marker=self.root/'probe_READY_0'
            wait_until(marker.exists,time.monotonic()+10,'no probe READY',process)
            probe=int(marker.read_text());process.send_signal(signal.SIGTERM);process.communicate(timeout=10)
            self.assertNotEqual(process.returncode,0);self.assertEqual(self.result()['jobs'],[])
            self.assertIn('operator_cancelled',self.result()['error'])
            with self.assertRaises(ProcessLookupError):os.kill(probe,0)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=5)
            process.communicate(timeout=5)

    def test_launcher_signal_queued_before_spawn_starts_no_driver(self):
        module=old.module('launch_resource_ready_stage',self.root/LAUNCHER)
        original=signal.signal
        def install(signum,handler):
            prior=original(signum,handler)
            if signum==signal.SIGTERM and getattr(handler,'__name__','')=='forward':handler(signum,None)
            return prior
        with patch.object(module.signal,'signal',side_effect=install),patch.object(module.subprocess,'Popen',side_effect=AssertionError('spawn after cancellation')) as spawn,patch.object(sys,'argv',[str(self.root/LAUNCHER),'--queue',str(self.f.queue)]):
            with self.assertRaises(SystemExit) as raised:module.main()
        self.assertEqual(raised.exception.code,1);spawn.assert_not_called()
        receipt=json.loads((self.root/'train.failed_launch_receipt.json').read_text())
        self.assertIsNone(receipt['child_pid']);self.assertEqual(receipt['cancellation_signals'],[signal.SIGTERM])
        self.assertFalse(self.f.out.exists())

    def test_poll_contract_rejects_bool_and_over_budget(self):
        for policy in (dict(maximum_seconds=True,poll_seconds=.2),dict(maximum_seconds=121,poll_seconds=.2),dict(maximum_seconds=1,poll_seconds=2)):
            self.f.q['resource_wait']=policy;self.f.write()
            result=self.f.run();self.assertNotEqual(result.returncode,0);self.assertFalse(self.f.out.exists())


if __name__=='__main__':unittest.main()

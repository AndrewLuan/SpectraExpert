"""功能概述：非作者人工子进程核验资源operator的取消边界与资源/路径约束。
所有进程只消费临时fixture；等待PID/完成等显式状态并有外层预算，不靠固定sleep。
Popen边界取消用精确注入信号重现launcher尚未安装handler的窗口，不读科学数据。
"""
import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import test_resource_ready_operator as author

class IndependentOperatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='resource_operator_independent_');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.f=author.Fixture(self.root)
    def test_launcher_cancel_at_child_Popen_boundary_must_not_orphan_supervisor(self):
        self.f.q['jobs'][0]['command'][-1]='wait';self.f.q['jobs'][0]['argv'][-1]='wait';self.f.q['max_parallel']=1;self.f.write()
        pidfile=self.root/'injected_supervisor_pid.json'
        code='''import subprocess,sys,os,signal,runpy,json
from pathlib import Path
real=subprocess.Popen
def spawn(*a,**k):
 p=real(*a,**k);Path(PIDFILE).write_text(str(p.pid));os.kill(os.getpid(),signal.SIGTERM);return p
subprocess.Popen=spawn
sys.argv=[LAUNCHER,'--queue',QUEUE]
runpy.run_path(LAUNCHER,run_name='__main__')
'''.replace('PIDFILE',repr(str(pidfile))).replace('LAUNCHER',repr(str(self.root/author.LAUNCHER))).replace('QUEUE',repr(str(self.f.queue)))
        p=subprocess.Popen([sys.executable,'-B','-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=self.f.env())
        owned=None
        try:
            # wait() observes wrapper termination. communicate() would await an orphan's
            # inherited stdout EOF and could hide the orphan by letting it finish first.
            p.wait(timeout=5)
            self.assertTrue(pidfile.exists());owned=int(pidfile.read_text())
            self.assertNotEqual(p.returncode,0)
            try:os.kill(owned,0);alive=True
            except ProcessLookupError:alive=False
            self.assertFalse(alive,'launcher exited after Popen-boundary cancel but left owned supervisor alive')
        finally:
            if p.poll() is None:p.kill();p.wait(timeout=5)
            if owned is not None:
                try:os.killpg(owned,signal.SIGTERM)
                except ProcessLookupError:pass
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    try:os.kill(owned,0)
                    except ProcessLookupError:break
                    time.sleep(min(.02,max(0,deadline-time.monotonic())))
                else:
                    try:os.killpg(owned,signal.SIGKILL)
                    except ProcessLookupError:pass
            p.communicate(timeout=5)
    def test_policy_nan_and_bool_do_not_enter_wait(self):
        m=author.old.module('resource_policy_independent',author.HERE/author.SUPERVISOR)
        for value in (float('nan'),float('inf'),False):
            with self.assertRaises(AssertionError):m.resource_policy(dict(resource_wait=dict(maximum_seconds=value,poll_seconds=.2)))
    def test_stale_owned_compute_pid_requires_no_OS_process_and_still_not_idle(self):
        m=author.old.module('resource_pid_independent',author.HERE/author.SUPERVISOR)
        sample=dict(memory_mib=0,utilization_percent=0,compute_pids=[12345])
        with patch.object(m,'pid_exists',return_value=False):self.assertFalse(m.resource_ready(sample,{12345}))
        with patch.object(m,'pid_exists',return_value=True):
            with self.assertRaisesRegex(AssertionError,'foreign_gpu_process'):m.resource_ready(sample,{12345})

if __name__=='__main__':unittest.main()

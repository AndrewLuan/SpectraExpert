"""功能概述：独立验证有界监督器的合成启动、退出、deadline、取消和清理。
主要职责：run_case 为每个合成子进程创建独立 job/review；wait_ready 等待真实就绪
文件；gone 验证监督器拥有的子进程已结束；run 汇总精确退出及日志断言。
代码库关系：只调用同目录 supervise_stage_rss_exit_boundary.py。临时 job 中的 PASS 凭证仅对合成
命令有效，不读取模型、缓存或真实协议，不构成任何科学执行授权。
输入输出与副作用：仅写 N 的临时目录并自动清除；只对本文件创建的监督器发送
SIGTERM。所有前置/完成/清理等待有界；失败不重试，最终 receipt 保留源 SHA。
"""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SUPERVISOR = ROOT / "supervise_stage_rss_exit_boundary.py"
PYTHON = sys.executable
SETUP_TIMEOUT = 10
COMPLETION_TIMEOUT = 15


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def wait_ready(proc, ready, running):
    deadline = time.monotonic() + SETUP_TIMEOUT
    while time.monotonic() < deadline:
        if ready.exists() and running.exists():
            return
        if proc.poll() is not None:
            raise AssertionError("supervisor_exited_before_child_ready")
        time.sleep(.01)
    raise AssertionError("child_readiness_timeout")


def run_case(folder, name, child_code, max_seconds=30, cancel=False, invalid_argv=False,
             max_rss_bytes=32 * 1024 ** 3, max_artifact_bytes=4 * 1024 ** 3):
    case = folder / name
    case.mkdir()
    stage = case / "stage"
    ready = case / "child_ready"
    argv = ["/synthetic/nonexistent-executable"] if invalid_argv else [PYTHON, "-c", child_code, str(ready)]
    job = dict(schema="top10_bounded_stage_v1", stage_dir=str(stage), write_root=str(case),
               max_seconds=max_seconds, max_rss_bytes=max_rss_bytes, max_cpu_threads=4,
               max_artifact_bytes=max_artifact_bytes,
               source_sha256={str(SUPERVISOR): sha(SUPERVISOR)}, argv=argv, cwd=str(case))
    job_path = case / "job.json"
    job_path.write_text(json.dumps(job))
    review = dict(status="PASS", independent=True, execution_authorized=True,
                  synthetic_only=True, pins=dict(job=sha(job_path), supervisor=sha(SUPERVISOR)))
    review_path = case / "synthetic_only_review.json"
    review_path.write_text(json.dumps(review))
    proc = subprocess.Popen([PYTHON, str(SUPERVISOR), "--job", str(job_path), "--review",
                             str(review_path), "--review-sha256", sha(review_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if cancel:
            wait_ready(proc, ready, stage / "running.json")
            os.kill(proc.pid, signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=COMPLETION_TIMEOUT)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=COMPLETION_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    outcome = json.loads((stage / "actual_execution.json").read_text())
    assert proc.returncode == outcome["actual_external_exit"], (name, proc.returncode, outcome)
    assert (stage / "EXIT_CODE").read_text().strip() == str(outcome["actual_inner_exit"]), name
    if outcome["pid"] is not None:
        assert gone(outcome["pid"]), ("owned_child_remains", name, outcome)
    return outcome, stage, stdout, stderr



def deterministic_missing_rss(folder):
    """只用可控完成事件验证RSS缺失分支，绝不创建真实科学子进程。"""
    import importlib.util
    import types
    spec = importlib.util.spec_from_file_location("independent_rss_boundary_target", SUPERVISOR)
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    cases = {}
    for mode in ("confirmed_exit", "alive_timeout", "timeout_then_immediate_exit",
                 "deadline_during_probe", "cancel_during_probe", "output_during_probe"):
        case = folder / mode
        case.mkdir()
        stage = case / "stage"
        job = dict(schema="top10_bounded_stage_v1", stage_dir=str(stage), write_root=str(case),
                   max_seconds=1 if mode == "deadline_during_probe" else 30,
                   max_rss_bytes=32*1024**3, max_cpu_threads=4,
                   max_artifact_bytes=65536 if mode == "output_during_probe" else 4*1024**3,
                   source_sha256={str(SUPERVISOR):sha(SUPERVISOR)},
                   argv=[PYTHON,"-c","raise AssertionError('synthetic fake must intercept')"],cwd=str(case))
        jp=case/"job.json";jp.write_text(json.dumps(job))
        rp=case/"synthetic_only_review.json"
        rp.write_text(json.dumps(dict(status="PASS",independent=True,execution_authorized=True,
                                     synthetic_only=True,pins=dict(job=sha(jp),supervisor=sha(SUPERVISOR)))))
        clock=types.SimpleNamespace(first=True,value=.9 if mode=="deadline_during_probe" else .01)
        def monotonic():
            if clock.first:
                clock.first=False
                return 0.
            return clock.value
        class FakeOwnedProcess:
            pid=987654321
            returncode=None
            def __init__(self):self.waits=[];self.terminated=0
            def poll(self):return self.returncode
            def wait(self,timeout):
                self.waits.append(timeout)
                if len(self.waits)==1:
                    assert 0 < timeout <= .25
                    if mode=="deadline_during_probe":assert abs(timeout-.1)<1e-12
                    if mode in ("alive_timeout","timeout_then_immediate_exit"):
                        if mode=="timeout_then_immediate_exit":self.returncode=0
                        raise subprocess.TimeoutExpired("synthetic_owned_process",timeout)
                    if mode=="deadline_during_probe":clock.value=1.01
                    if mode=="cancel_during_probe":
                        signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
                    if mode=="output_during_probe":(case/"synthetic_large_output").write_bytes(b"x"*131072)
                    self.returncode=0
                assert self.returncode is not None,"unexpected_wait_of_live_fake"
                return self.returncode
        child=FakeOwnedProcess()
        def terminate_owned(proc):
            assert proc is child
            proc.terminated+=1
            if proc.returncode is None:proc.returncode=-signal.SIGTERM
        original=(supervisor.time,supervisor.subprocess,supervisor.rss_bytes,
                  supervisor.terminate_owned,sys.argv)
        supervisor.time=types.SimpleNamespace(monotonic=monotonic)
        supervisor.subprocess=types.SimpleNamespace(Popen=lambda *a,**k:child,
            TimeoutExpired=subprocess.TimeoutExpired,DEVNULL=subprocess.DEVNULL)
        supervisor.rss_bytes=lambda pid:None
        supervisor.terminate_owned=terminate_owned
        sys.argv=[str(SUPERVISOR),"--job",str(jp),"--review",str(rp),"--review-sha256",sha(rp)]
        try:external=supervisor.main()
        finally:
            supervisor.time,supervisor.subprocess,supervisor.rss_bytes,supervisor.terminate_owned,sys.argv=original
        out=json.loads((stage/"actual_execution.json").read_text())
        assert external==out["actual_external_exit"]
        assert out["rss_monitor_samples"]==0 and child.waits[0] <= .25
        if mode=="confirmed_exit":
            assert external==0 and out["actual_inner_exit"]==0 and out["termination_reason"] is None
            assert out["rss_exit_boundary_confirmed"] is True and child.terminated==0
            assert out["rss_monitor_status"]=="not_observed_child_already_exited"
        elif mode in ("alive_timeout","timeout_then_immediate_exit"):
            assert external==1 and out["termination_reason"]=="rss_monitor_unavailable"
            assert out["rss_missing_while_alive"] is True
            assert out["rss_monitor_status"]=="unavailable_while_alive"
            assert out["actual_inner_exit"]==(-signal.SIGTERM if mode=="alive_timeout" else 0)
        else:
            reason={"deadline_during_probe":"deadline_exceeded",
                    "cancel_during_probe":"cancel_signal_"+str(signal.SIGTERM),
                    "output_during_probe":"output_budget_exceeded"}[mode]
            assert external==1 and out["actual_inner_exit"]==0 and out["termination_reason"]==reason
            assert out["rss_exit_boundary_confirmed"] is True
        cases[mode]=dict(status="PASS",actual_inner_exit=out["actual_inner_exit"],
            actual_external_exit=external,termination_reason=out["termination_reason"],
            rss_monitor_status=out["rss_monitor_status"],probe_timeout=child.waits[0])
    return cases


def run():
    results = {}
    pause = "from pathlib import Path; import signal,sys; Path(sys.argv[1]).write_text('ready'); signal.pause()"
    with tempfile.TemporaryDirectory(prefix="reviewer_supervisor_", dir=ROOT) as temp:
        folder = Path(temp)
        outcome, stage, _, _ = run_case(folder, "success",
            "import sys; print('synthetic_stdout'); print('synthetic_stderr',file=sys.stderr)")
        assert outcome["actual_inner_exit"] == outcome["actual_external_exit"] == 0
        assert outcome["termination_reason"] is None
        assert (stage / "stdout.log").read_text() == "synthetic_stdout\n"
        assert (stage / "stderr.log").read_text() == "synthetic_stderr\n"
        results["success_exact_logs_and_exit"] = "PASS"
        outcome, _, _, _ = run_case(folder, "nonzero", "raise SystemExit(7)")
        assert outcome["actual_inner_exit"] == 7 and outcome["actual_external_exit"] == 1
        assert outcome["termination_reason"] is None
        results["nonzero_exact_inner_exit"] = "PASS"
        outcome, _, _, _ = run_case(folder, "deadline", pause, max_seconds=1.5)
        assert outcome["actual_inner_exit"] == -signal.SIGTERM and outcome["actual_external_exit"] == 1
        assert outcome["termination_reason"] == "deadline_exceeded"
        assert outcome["elapsed_seconds"] >= 1.5
        results["deadline_terminates_owned_child"] = "PASS"
        outcome, _, _, _ = run_case(folder, "cancel", pause, cancel=True)
        assert outcome["actual_inner_exit"] == -signal.SIGTERM and outcome["actual_external_exit"] == 1
        assert outcome["termination_reason"] == "cancel_signal_" + str(signal.SIGTERM)
        results["SIGTERM_after_readiness_cleanup"] = "PASS"
        outcome, _, _, _ = run_case(folder, "startup_error", "", invalid_argv=True)
        assert outcome["pid"] is None and outcome["actual_inner_exit"] is None
        assert outcome["actual_external_exit"] == 1 and outcome["error"].startswith("FileNotFoundError:")
        results["startup_failure_receipt"] = "PASS"
        outcome, _, _, _ = run_case(folder, "rss_limit", pause, max_rss_bytes=1)
        assert outcome["actual_external_exit"] == 1 and outcome["termination_reason"] == "rss_exceeded"
        assert outcome["peak_observed_rss_bytes"] > 1
        assert outcome["rss_monitor_samples"] > 0 and outcome["rss_monitor_status"] == "observed"
        results["RSS_monitored_and_low_limit_enforced"] = "PASS"
        burst_pause = "from pathlib import Path; import signal; Path('synthetic_large_output').write_bytes(b'x'*131072); signal.pause()"
        outcome, _, _, _ = run_case(folder, "output_limit", burst_pause, max_artifact_bytes=65536)
        assert outcome["actual_external_exit"] == 1 and outcome["termination_reason"] == "output_budget_exceeded"
        assert outcome["observed_output_bytes"] > 65536
        results["whole_write_root_output_limit"] = "PASS"
        fast_burst = "from pathlib import Path; Path('synthetic_large_output').write_bytes(b'x'*131072)"
        outcome, _, _, _ = run_case(folder, "fast_output_limit", fast_burst, max_artifact_bytes=65536)
        assert outcome["actual_external_exit"] == 1 and outcome["termination_reason"] == "output_budget_exceeded"
        assert outcome["final_output_bytes_before_receipt"] > 65536
        results["fast_exit_final_output_budget"] = "PASS"
    with tempfile.TemporaryDirectory(prefix="reviewer_supervisor_rss_deterministic_",dir=ROOT) as temp:
        results["deterministic_missing_RSS_exit_alive_timeoutrace_deadline_cancel_output"]=deterministic_missing_rss(Path(temp))
    receipt = dict(schema="independent_supervisor_rss_boundary_oracle_v1", status="PASS",
                   supervisor_sha256=sha(SUPERVISOR), reviewer_oracle_sha256=sha(__file__),
                   checks=results, synthetic_only=True, real_scientific_inputs_read=0,
                   scientific_execution_authorized=False)
    (ROOT / "independent_supervisor_rss_exit_boundary_oracle_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    run()

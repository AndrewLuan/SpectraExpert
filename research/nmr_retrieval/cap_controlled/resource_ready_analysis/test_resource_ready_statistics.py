"""功能概述：新资源operator的test统计gate人工12子进程联调和原生精度回归。
只替换metadata执行链，native dtype/rank/CI纯函数沿旧锁定版本。所有payload是临时
人工JSON/bytes，无真实test读取；仅最终耗时bootstrap由已审math测试单独覆盖。
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import summarize_sealed_cap_results as s

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('_resource_statistics_native_row_tests',ROOT/'native_cost_evaluation/test_summarize_native_cost_results.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
NativeRowTests=native.NativeRowTests

def full_fixture(root):
    spec=importlib.util.spec_from_file_location('_resource_statistics_metadata_fixture',ROOT/'test_summarize_sealed_cap_results_independent.py')
    fx=importlib.util.module_from_spec(spec);spec.loader.exec_module(fx)
    child=fx.CHILD.replace('natural_candidate_costs=[0.]*512,',"natural_candidate_costs=[0.]*512,natural_candidate_coarse_costs=[0.]*512,coarse_dtype='torch.float64',correction_dtypes=['torch.float32']*2,final_cost_dtype='torch.float64',")
    child=child.replace("(out/'summary.json').write_text(json.dumps(s));","s['evaluator']=b['evaluator']\n(out/'summary.json').write_text(json.dumps(s));")
    with patch.object(fx,'CHILD',child):f=fx.ExecutionFixture(root)
    f.bindings['evaluator']=dict(path=str(f.script),sha256=fx.digest(f.script))
    f.queue['operator_paths']={}
    for role,name in [('supervisor','supervise_resource_ready_queue.py'),('launcher','launch_resource_ready_stage.py')]:
        p=f.root/name;p.write_bytes((ROOT/name).read_bytes())
        f.queue['operator_paths'][role]=dict(path=str(p),sha256=fx.digest(p))
        f.bindings['source_sha256'][str(p)]=fx.digest(p)
    f.queue['resource_wait']=dict(maximum_seconds=120,poll_seconds=.2)
    fx.full_metadata(f)
    result=subprocess.run([sys.executable,'-B',str(f.root/'launch_resource_ready_stage.py'),'--queue',str(f.qp)],capture_output=True,text=True,timeout=30)
    if result.returncode:raise AssertionError(result.stdout[-1000:]+result.stderr[-1000:])
    f.snapshot={p:p.read_bytes() for p in f.root.rglob('*') if p.is_file()}
    return f


class ResourceStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.f=full_fixture(cls.tmp.name)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def tearDown(self):self.f.restore()

    def run_stats(self):
        out=self.f.root/'synthetic_statistics';out.mkdir(exist_ok=True)
        with patch.object(s.matrix,'analyze_matrix',return_value=dict(metrics=[dict(synthetic_only=True)])):
            return s.run(self.f.args,out)

    def test_real_new_operator12_native64_predictions_full_gate(self):
        r=self.run_stats();self.assertFalse(r['selection_performed']);self.assertEqual(r['test_cells'],12)

    def test_old_supervisor_argv_not_accepted_for_new_run(self):
        for name,index in [('launch_receipt.json',2),('driver_start.json',0)]:
            p=self.f.run/name;v=json.loads(p.read_text());v['argv'][index]=str(self.f.root/'supervise_controlled_queue.py');p.write_text(json.dumps(v))
        seen=[];original=s.c.Pins.raw
        def raw(pins,path,expected=None):
            if Path(path).name=='predictions.jsonl':seen.append(str(path))
            return original(pins,path,expected)
        with patch.object(s.c.Pins,'raw',raw),self.assertRaisesRegex(ValueError,'exact_driver_argv'):self.run_stats()
        self.assertEqual(seen,[])

    def test_completion_operator_refs_missing_rejected_before_payload(self):
        p=self.f.run/'completion.json';v=json.loads(p.read_text());v.pop('operator_paths');p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'resource_operator_actual_envelopes'):
            self.run_stats()

    def test_completion_resource_budget_drift_rejected(self):
        p=self.f.run/'completion.json';v=json.loads(p.read_text());v['resource_wait']['maximum_seconds']=121;p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'resource_wait_actual_envelopes'):self.run_stats()

    def test_scientific_AST_exact_old_native(self):
        get=lambda path:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before=get(ROOT/'native_cost_evaluation/summarize_sealed_cap_results.py');after=get(HERE/'summarize_sealed_cap_results.py')
        self.assertEqual(set(before),set(after))
        for name in before.keys()-{'execution'}:self.assertEqual(before[name],after[name],name)

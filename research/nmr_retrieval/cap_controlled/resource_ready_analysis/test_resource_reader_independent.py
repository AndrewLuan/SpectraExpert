"""功能概述：非作者核资源恢复reader的科学函数字节与人工执行链攻击。
只消费临时JSON/源码，不调用模型、不读真实cache或test，证明新operator refs
与policy不能换回旧版、角色互换或从实际回执删减。旧训练分支继续严格校验。
"""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
import cap_evaluation_common as c
from test_resource_ready_reader import install_resource_operator
from test_summarize_cap_validation import MatrixFixture,put

HERE=Path(__file__).resolve().parent

def synchronize(queue,run):
    q=json.loads(Path(queue).read_text());h=c.sha(queue)
    for name in ('driver_start.json','launch_receipt.json','completion.json'):
        p=run/name;v=json.loads(p.read_text());v['queue_sha256']=h
        if name!='completion.json':v['source_sha256']=q['source_sha256']
        if 'operator_paths' in v:v['operator_paths']=copy.deepcopy(q['operator_paths'])
        if 'resource_wait' in v:v['resource_wait']=copy.deepcopy(q['resource_wait'])
        put(p,v)
    return q

class IndependentReaderTests(unittest.TestCase):
    def test_scientific_AST_and_two_entry_bytes_exact_old_native(self):
        old=HERE.parent/'native_cost_evaluation'
        for name in ('summarize_cap_validation.py','analyze_cap_mechanism.py'):
            self.assertEqual((HERE/name).read_bytes(),(old/name).read_bytes())
        get=lambda p:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
        a,b=get(old/'cap_evaluation_common.py'),get(HERE/'cap_evaluation_common.py')
        self.assertEqual(set(a),set(b))
        for name in set(a)-{'driver_receipts'}:self.assertEqual(a[name],b[name],name)

    def fixture(self,tmp):
        f=MatrixFixture(tmp);f.run=install_resource_operator(f.queue,f.run,'validation_resource_ready')
        return f

    def test_swapped_operator_roles_even_with_synchronized_envelopes_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);q=json.loads(Path(f.queue).read_text())
            q['operator_paths']['launcher'],q['operator_paths']['supervisor']=q['operator_paths']['supervisor'],q['operator_paths']['launcher']
            put(f.queue,q);synchronize(f.queue,f.run)
            with self.assertRaisesRegex(ValueError,'canonical_resource_operator_path'):c.driver_receipts(c.Pins(),f.queue,f.run)

    def test_self_consistent_old_supervisor_alias_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);q=json.loads(Path(f.queue).read_text());p=f.root/'supervise_controlled_queue.py'
            p.write_text('# artificial legacy supervisor\n');q['source_sha256'][str(p)]=c.sha(p)
            q['operator_paths']['supervisor']={'path':str(p),'sha256':c.sha(p)}
            put(f.queue,q);synchronize(f.queue,f.run)
            with self.assertRaisesRegex(ValueError,'canonical_resource_operator_path'):c.driver_receipts(c.Pins(),f.queue,f.run)

    def test_boolean_or_extended_wait_is_not_registered_policy(self):
        for value in (True,121):
            with tempfile.TemporaryDirectory() as tmp:
                f=self.fixture(tmp);q=json.loads(Path(f.queue).read_text());q['resource_wait']['maximum_seconds']=value
                put(f.queue,q);synchronize(f.queue,f.run)
                with self.assertRaisesRegex(ValueError,'exact_bounded_resource_wait'):c.driver_receipts(c.Pins(),f.queue,f.run)

    def test_completion_cannot_erase_resource_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);p=f.run/'completion.json';v=json.loads(p.read_text());v.pop('resource_wait');put(p,v)
            with self.assertRaisesRegex(ValueError,'resource_wait_receipt_policy'):c.driver_receipts(c.Pins(),f.queue,f.run)

    def test_start_cannot_relabel_actual_operator_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);p=f.run/'driver_start.json';v=json.loads(p.read_text());v['operator_paths']={};put(p,v)
            with self.assertRaisesRegex(ValueError,'resource_operator_receipt_refs'):c.driver_receipts(c.Pins(),f.queue,f.run)

    def test_operator_actual_bytes_drift_before_any_output_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);q=json.loads(Path(f.queue).read_text());Path(q['operator_paths']['launcher']['path']).write_text('# drift\n')
            with self.assertRaisesRegex(ValueError,'input_sha'):c.driver_receipts(c.Pins(),f.queue,f.run)

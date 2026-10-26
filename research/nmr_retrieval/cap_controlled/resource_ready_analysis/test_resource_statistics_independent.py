"""功能概述：非作者核新资源operator统计gate在首预测读取前拒绝执行闭包篡改。
复用临时12人工子进程fixture，不读实际test；只以spy证明metadata失败时零预测读取。
精度/排名纯函数另与已独审native版本AST全等，未增加科学统计口径。
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import test_resource_ready_statistics as a

class IndependentResourceStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.f=a.full_fixture(cls.tmp.name)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def tearDown(self):self.f.restore()
    def run_reject(self,change,pattern):
        change();seen=[];original=a.s.c.Pins.raw
        def raw(pins,path,expected=None):
            if Path(path).name=='predictions.jsonl':seen.append(str(path))
            return original(pins,path,expected)
        with patch.object(a.s.c.Pins,'raw',raw),self.assertRaisesRegex(ValueError,pattern):
            with patch.object(a.s.matrix,'analyze_matrix',side_effect=AssertionError('math reached after bad metadata')):
                a.s.run(self.f.args,self.f.root/'independent_statistics')
        self.assertEqual(seen,[])
    def edit(self,path,mutator):
        value=json.loads(path.read_text());mutator(value);path.write_text(json.dumps(value))
    def test_outer_operator_role_swap_rejected_before_predictions(self):
        self.run_reject(lambda:self.edit(self.f.run/'launch_receipt.json',lambda v:v['operator_paths'].update(supervisor=v['operator_paths']['launcher'])),
          'resource_operator_actual_envelopes')
    def test_start_resource_policy_bool_rejected_before_predictions(self):
        # bool is not a substitute for the registered integer120 budget.
        self.run_reject(lambda:self.edit(self.f.run/'driver_start.json',lambda v:v['resource_wait'].update(maximum_seconds=True)),
          'resource_wait_actual_envelopes')
    def test_last_leaf_nonzero_still_blocks_first_prediction_read(self):
        done=json.loads((self.f.run/'completion.json').read_text());jid=done['jobs'][-1]['id']
        def change():
            self.edit(self.f.run/'completion.json',lambda v:v['jobs'][-1].update(external_exit=1))
        self.run_reject(change,'strict|actual|zero|EXIT|complete|receipt')
    def test_same_SHA_old_supervisor_alias_rejected_before_predictions(self):
        def change():
            # Even an otherwise valid envelope must use the exact registered new executable.
            self.edit(self.f.run/'launch_receipt.json',lambda v:v['argv'].__setitem__(2,str(self.f.root/'supervise_controlled_queue.py')))
        self.run_reject(change,'exact_driver_argv')

if __name__=='__main__':unittest.main()

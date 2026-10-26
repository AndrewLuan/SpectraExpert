"""功能概述：验证研究调度器真实子进程退出、依赖传播与训练截止门禁。

测试在临时目录启动最小本地CPU子进程，等候实际wait而非模拟完成状态；成功和
失败分别核对外层/内部退出证据，过期任务不得产生科学输出。无网络或GPU调用。
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import unittest

from .h_research_driver import execute, supervise, validate_queue


class DriverTests(unittest.TestCase):
    def queue(self, root, fail=False):
        out = root/'science'
        code = ('import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); '
                + ("sys.exit(7)" if fail else "(p/'EXIT_CODE').write_text('0\\n')"))
        return dict(schema='nmr_research_jobs_20261022', jobs=[dict(id='scientific_job',
            out=str(out), argv=[sys.executable, '-c', code, str(out)], kind='audit', gpu=False,
            cwd=str(Path(__file__).resolve().parents[1]))])

    def args(self, root, queue):
        path = root/'queue.json'; path.write_text(json.dumps(queue))
        return argparse.Namespace(queue=str(path), out=str(root/'run'), gpus='', max_cpu=1)

    def test_true_success_waited_and_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); args = self.args(root, self.queue(root))
            self.assertEqual(supervise(args), 0)
            receipt = json.loads((root/'run/launch_receipt.json').read_text())
            self.assertEqual((receipt['external_exit'], receipt['internal_exit']), (0, 0))
            child = json.loads((root/'run/scientific_job.receipt.json').read_text())
            self.assertEqual((child['external_exit'], child['internal_exit']), (0, 0))
            self.assertLess(child['started_utc'], child['finished_utc'])

    def test_nonzero_propagates_and_not_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); queue = self.queue(root, fail=True)
            queue['jobs'].append(dict(queue['jobs'][0], id='dependent', out=str(root/'never'), deps=['scientific_job']))
            args = self.args(root, queue)
            self.assertEqual(supervise(args), 1)
            completion = json.loads((root/'run/completion.json').read_text())
            self.assertEqual(completion['jobs']['scientific_job']['external_exit'], 7)
            self.assertIsNone(completion['jobs']['scientific_job']['internal_exit'])
            self.assertEqual(completion['jobs']['dependent']['state'], 'blocked')
            self.assertFalse((root/'never').exists())

    def test_deadline_prevents_new_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); queue = self.queue(root)
            queue['training_cutoff_utc'] = '2000-01-01T00:00:00Z'
            queue['jobs'][0]['kind'] = 'training'
            self.assertEqual(execute(self.args(root, queue)), 1)
            self.assertFalse((root/'science').exists())

    def test_dependency_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = self.queue(Path(folder)); queue['jobs'][0]['deps'] = ['scientific_job']
            with self.assertRaisesRegex(ValueError, 'dependency'):
                validate_queue(queue)


if __name__ == '__main__':
    unittest.main()

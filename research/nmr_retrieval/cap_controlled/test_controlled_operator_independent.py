"""功能概述：用临时目录独立攻击受控研究的 Python I/O 隔离边界。

只创建假输入和小型无模型子程序，核提前 test 哈希、非本任务写入、
未登记张量、合法输出及真实内外退出。不会连接远端或打开科学数据。
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name, path=None):
    spec = importlib.util.spec_from_file_location('_ind_' + name, path or HERE / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class IsolationAttacks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.output = self.root / 'job_output'
        self.output.mkdir()
        self.access = self.root / 'access_logs' / 'job.json'
        self.binding = self.root / 'bindings.json'
        self.script = self.root / 'small_child.py'
        self.isolation = self.root / 'isolation_entry.py'
        self.isolation.write_bytes((HERE / 'isolation_entry.py').read_bytes())

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, phase='train', allowed=()):
        return ['--bindings', str(self.binding), '--phase', phase,
                '--access-log', str(self.access), '--output-root', str(self.output)] + \
            sum((['--allow-read', str(p)] for p in allowed), []) + ['--', str(self.script)]

    def execute(self, source, phase='train', allowed=()):
        self.script.write_text(source)
        self.binding.write_text(json.dumps({'input_sha256': {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in allowed}}))
        result = subprocess.run([sys.executable, '-B', str(self.isolation)] +
            self.args(phase, allowed), capture_output=True, text=True, timeout=30)
        receipt = json.loads(self.access.read_text()) if self.access.exists() else None
        return result, receipt

    def test_valid_own_output_and_double_zero(self):
        p = self.output / 'summary.json'
        result, receipt = self.execute('from pathlib import Path\nPath(' + repr(str(p)) + ').write_text("ok")\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt['internal_exit'], 0)
        self.assertIs(type(receipt['internal_exit']), int)
        self.assertEqual(p.read_text(), 'ok')
        self.assertEqual(receipt['rejected_paths'], [])

    def test_write_outside_job_denied_without_corruption(self):
        p = self.root / 'immutable_source.json'
        p.write_text('original')
        result, receipt = self.execute('from pathlib import Path\nPath(' + repr(str(p)) + ').write_text("changed")\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(p.read_text(), 'original')
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt['internal_exit'], 1)
        self.assertIn(str(p), receipt['rejected_paths'])

    def test_unregistered_tensor_read_denied(self):
        p = self.root / 'unregistered.pt'
        p.write_bytes(b'fake; no torch')
        result, receipt = self.execute('from pathlib import Path\nPath(' + repr(str(p)) + ').read_bytes()\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(p), receipt['rejected_paths'])

    def test_symlink_write_resolves_outside_job(self):
        p = self.root / 'immutable_input.txt'
        p.write_text('original')
        link = self.output / 'escape.txt'
        link.symlink_to(p)
        result, receipt = self.execute('from pathlib import Path\nPath(' + repr(str(link)) + ').write_text("changed")\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(p.read_text(), 'original')
        self.assertIn(str(p), receipt['rejected_paths'])

    def test_train_test_allowlist_denied_before_first_hash(self):
        p = self.root / 'test' / 'features.pt'
        p.parent.mkdir()
        p.write_bytes(b'fake; no scientific data')
        self.binding.write_text(json.dumps({'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()}}))
        self.script.write_text('pass\n')
        isolation = module('isolation_entry', self.isolation)
        calls = []
        def forbidden_hash(path):
            calls.append(str(path))
            raise RuntimeError('pre-admission hash executed')
        with patch.object(sys, 'argv', ['isolation_entry.py'] + self.args('train', [p])), \
             patch.object(isolation, 'sha', forbidden_hash):
            with self.assertRaises((PermissionError, ValueError, AssertionError)):
                isolation.main()
        self.assertEqual(calls, [])

    def test_old_sealed_allowlist_denied_before_first_hash(self):
        p = self.root / 'sealed-20261024_fake' / 'summary.json'
        p.parent.mkdir()
        p.write_text('{}')
        self.binding.write_text(json.dumps({'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()}}))
        self.script.write_text('pass\n')
        isolation = module('isolation_entry', self.isolation)
        calls = []
        with patch.object(sys, 'argv', ['isolation_entry.py'] + self.args('train', [p])), \
             patch.object(isolation, 'sha', side_effect=lambda path: calls.append(str(path))):
            with self.assertRaises((PermissionError, ValueError, AssertionError)):
                isolation.main()
        self.assertEqual(calls, [])

    def test_other_sealed_root_and_prediction_basename_denied(self):
        for suffix in ('sealed_resume_20261024/results/predictions.jsonl', 'prior/metadata/predictions.jsonl',
                       'prior/metadata/sealed_test_report.md', 'prior/metadata/test.pt'):
            p = self.root / suffix
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('fake; no test data')
            result, receipt = self.execute('from pathlib import Path\nPath(' + repr(str(p)) + ').read_bytes()\n')
            self.assertNotEqual(result.returncode, 0, suffix)
            self.assertIn(str(p), receipt['rejected_paths'])

    def test_own_prediction_output_can_be_written_and_read(self):
        p = self.output / 'predictions.jsonl'
        result, receipt = self.execute('from pathlib import Path\np=Path(' + repr(str(p)) + ')\np.write_text("fake")\nassert p.read_text()=="fake"\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt['internal_exit'], 0)

    def test_cold_torch_dill_import_then_own_tensor_roundtrip(self):
        p = self.output / 'tiny_runtime_tensor.pt'
        source = ('import torch,dill\nfrom pathlib import Path\n'
                  'p=Path(' + repr(str(p)) + ')\n'
                  'torch.manual_seed(317)\n'
                  'model=torch.nn.Linear(2,1); before=model.weight.detach().clone()\n'
                  'opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)\n'
                  'loss=model(torch.tensor([[1.,2.]])).square().mean()\n'
                  'loss.backward(); opt.step()\n'
                  'assert not torch.equal(before,model.weight)\n'
                  'torch.save({"state_dict":model.state_dict(),"optimizer":opt.state_dict()},p)\n'
                  'saved=torch.load(p,weights_only=True)\n'
                  'assert torch.equal(saved["state_dict"]["weight"],model.weight)\n')
        result, receipt = self.execute(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt['internal_exit'], 0)
        self.assertEqual(receipt['rejected_paths'], [])

    def test_devnull_only_character_device_exception_not_regular_file(self):
        regular = self.root / 'devnull_lookalike'
        regular.write_text('original')
        source = ('import os\nfrom pathlib import Path\n'
                  'with open(os.devnull,"r+b") as f:f.write(b"discarded")\n'
                  'Path(' + repr(str(regular)) + ').write_text("changed")\n')
        result, receipt = self.execute(source)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(regular.read_text(), 'original')
        self.assertIsNotNone(receipt)
        self.assertIn(str(regular), receipt['rejected_paths'])

    def test_nonzero_child_preserved(self):
        result, receipt = self.execute('raise SystemExit(7)\n')
        self.assertEqual(result.returncode, 7)
        self.assertEqual(receipt['internal_exit'], 7)

    def test_boolean_exit_not_accepted_as_integer_zero(self):
        result, receipt = self.execute('raise SystemExit(False)\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt['internal_exit'], 1)


class SupervisorFixture:
    def __init__(self, root):
        self.root = root
        for name in ('isolation_entry.py', 'supervise_controlled_queue.py', 'launch_controlled_stage.py'):
            (root / name).write_bytes((HERE / name).read_bytes())
        (root / 'cap_ranker.py').write_text('# synthetic unused cap source\n')
        self.child = root / 'train_cap_ranker.py'
        self.child.write_text('import os,sys,json,time\nfrom pathlib import Path\n'
            'out=Path(sys.argv[1]); mode=sys.argv[2]\n'
            'if mode=="fail": raise SystemExit(7)\n'
            'if mode=="fail_after_peer":\n'
            ' deadline=time.monotonic()+10\n'
            ' while not (out.parent/"1"/"READY").exists():\n'
            '  assert time.monotonic()<deadline,"peer never reached READY"\n'
            '  time.sleep(.01)\n'
            ' raise SystemExit(7)\n'
            'out.mkdir(parents=True)\n'
            'if mode=="wait":\n'
            ' (out/"READY").write_text(str(os.getpid()))\n'
            ' deadline=time.monotonic()+10\n'
            ' while not (out/"RELEASE").exists():\n'
            '  assert time.monotonic()<deadline,"peer cleanup deadline"\n'
            '  time.sleep(.01)\n'
            '(out/"env.json").write_text(json.dumps({"cuda":os.environ.get("CUDA_VISIBLE_DEVICES")}))\n')
        self.protocol = root / 'protocol.json'
        self.protocol.write_text(json.dumps({'source_checkout': str(root), 'budget': {'maximum_stage_seconds': 60}}))
        self.input = root / 'immutable.json'
        self.input.write_text('{"input":"tiny synthetic"}')
        self.binding = root / 'bindings.json'
        self.binding.write_text(json.dumps({'source_sha256': {}, 'input_sha256': {str(self.input): self.sha(self.input)}}))
        self.queue = root / 'queue.json'
        self.review = root / 'review.json'
        self.out = root / 'runs' / 'train'
        jobs = []
        for i in range(8):
            output = root / 'outputs' / str(i)
            access = root / 'access_logs' / (str(i) + '.json')
            command = [str(self.child), str(output), 'ok']
            argv = [sys.executable, '-B', str(root / 'isolation_entry.py'), '--bindings', str(self.binding),
                '--phase', 'train', '--access-log', str(access), '--output-root', str(output), '--'] + command
            jobs.append({'id': 'tiny' + str(i), 'argv': argv, 'command': command, 'job_out': str(output),
                'access_log': str(access), 'allowed_reads': [], 'gpu': None, 'cwd': str(root)})
        self.q = {'schema': 'cap10_controlled_queue_20261024', 'phase': 'train',
            'bindings': str(self.binding), 'input_sha256': {str(self.input): self.sha(self.input)},
            'local_root': str(root), 'review_paths': [str(self.review)], 'python': sys.executable,
            'max_parallel': 2, 'maximum_seconds': 20, 'jobs': jobs}
        self.write()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write(self):
        self.q['source_sha256'] = {str(self.root / name): self.sha(self.root / name) for name in
            ('protocol.json', 'bindings.json', 'isolation_entry.py', 'supervise_controlled_queue.py',
             'launch_controlled_stage.py', 'cap_ranker.py', 'train_cap_ranker.py')}
        self.queue.write_text(json.dumps(self.q))
        self.review.write_text(json.dumps({'status': 'PASS', 'independent': True, 'admission_phase': 'train',
            'reviewed_sha256': dict(self.q['source_sha256'], **{str(self.queue): self.sha(self.queue)})}))

    def run(self, execute=False, wrapper=False):
        path = self.root / ('launch_controlled_stage.py' if wrapper else 'supervise_controlled_queue.py')
        argv = [sys.executable, '-B', str(path), '--queue', str(self.queue)]
        if not wrapper:
            argv += ['--out', str(self.out)] + (['--execute'] if execute else [])
        return subprocess.run(argv, capture_output=True, text=True, timeout=30)


class SupervisorAttacks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.f = SupervisorFixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_metadata_admission_starts_no_child(self):
        result = self.f.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.f.out.exists())
        self.assertFalse((self.root / 'train.operator_claim.json').exists())

    def test_eight_real_tiny_children_wrapper_double_exit_and_cpu_env(self):
        result = self.f.run(wrapper=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        completion = json.loads((self.f.out / 'completion.json').read_text())
        launch = json.loads((self.f.out / 'launch_receipt.json').read_text())
        self.assertEqual(launch['internal_exit'], 0)
        self.assertEqual(launch['external_exit'], 0)
        self.assertEqual(completion['status'], 'COMPLETE')
        self.assertEqual(len(completion['jobs']), 8)
        self.assertTrue(completion['inputs_unchanged'])
        for r in completion['jobs']:
            self.assertIs(type(r['internal_exit']), int)
            self.assertEqual((r['internal_exit'], r['external_exit']), (0, 0))
            self.assertEqual(json.loads((Path(r['job_out']) / 'env.json').read_text())['cuda'], '')
        repeated = self.f.run(wrapper=True)
        self.assertNotEqual(repeated.returncode, 0)

    def test_first_failure_stops_pending_and_owned_peer(self):
        for i, mode in [(0, 'fail_after_peer'), (1, 'wait')]:
            self.f.q['jobs'][i]['command'][-1] = mode
            self.f.q['jobs'][i]['argv'][-1] = mode
        self.f.write()
        result = self.f.run(execute=True)
        self.assertNotEqual(result.returncode, 0)
        completion = json.loads((self.f.out / 'completion.json').read_text())
        self.assertEqual(completion['status'], 'FAILED')
        self.assertEqual(len(completion['jobs']), 2)
        self.assertEqual(completion['jobs'][0]['internal_exit'], 7)
        self.assertTrue(completion['jobs'][1]['terminated_after_peer_failure'])
        peer_pid = int((self.root / 'outputs' / '1' / 'READY').read_text())
        self.assertEqual(peer_pid, completion['jobs'][1]['pid'])
        with self.assertRaises(ProcessLookupError):
            os.kill(peer_pid, 0)
        self.assertFalse((self.root / 'outputs' / '2').exists())

    def test_source_drift_rejected_before_claim(self):
        self.f.child.write_text('raise RuntimeError("must not run")\n')
        result = self.f.run(execute=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'train.operator_claim.json').exists())

    def test_missing_required_source_rejected(self):
        del self.f.q['source_sha256'][str(self.root / 'cap_ranker.py')]
        self.f.queue.write_text(json.dumps(self.f.q))
        self.f.review.write_text(json.dumps({'status': 'PASS', 'independent': True, 'admission_phase': 'train',
            'reviewed_sha256': dict(self.f.q['source_sha256'], **{str(self.f.queue): self.f.sha(self.f.queue)})}))
        self.assertNotEqual(self.f.run().returncode, 0)

    def test_review_wrong_phase_rejected(self):
        review = json.loads(self.f.review.read_text()); review['admission_phase'] = 'test'
        self.f.review.write_text(json.dumps(review))
        self.assertNotEqual(self.f.run().returncode, 0)

    def test_nonzero_input_hash_preserved_and_no_child(self):
        self.f.input.write_text('drift')
        result = self.f.run(execute=True)
        self.assertNotEqual(result.returncode, 0)
        completion = json.loads((self.f.out / 'completion.json').read_text())
        self.assertEqual(completion['jobs'], [])
        self.assertEqual((self.f.out / 'EXIT_CODE').read_text().strip(), '1')

    def test_forged_inner_argv_rejected_before_claim(self):
        self.f.q['jobs'][0]['argv'][-1] = 'unregistered-mode'
        self.f.write()
        result = self.f.run(execute=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'train.operator_claim.json').exists())

    def test_binding_source_digest_must_equal_queue_digest(self):
        binding = json.loads(self.f.binding.read_text())
        binding['source_sha256'][str(self.root / 'cap_ranker.py')] = '0' * 64
        self.f.binding.write_text(json.dumps(binding))
        self.f.write()
        result = self.f.run()
        self.assertNotEqual(result.returncode, 0)

    def test_preexisting_wrapper_claim_starts_no_driver(self):
        claim = self.root / 'train.launch_claim.json'
        claim.write_text('{"retained":"prior attempt"}')
        result = self.f.run(wrapper=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.f.out.exists())
        self.assertEqual(json.loads(claim.read_text()), {'retained': 'prior attempt'})

    def test_concurrent_wrappers_have_one_owner_and_one_driver(self):
        argv = [sys.executable, '-B', str(self.root / 'launch_controlled_stage.py'),
                '--queue', str(self.f.queue)]
        left = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        right = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 30
        try:
            left.communicate(timeout=max(.01, deadline - time.monotonic()))
            right.communicate(timeout=max(.01, deadline - time.monotonic()))
        finally:
            for child in (left, right):
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait(timeout=5)
        self.assertEqual(sorted([left.returncode, right.returncode]), [0, 1])
        claim = json.loads((self.root / 'train.launch_claim.json').read_text())
        launch = json.loads((self.f.out / 'launch_receipt.json').read_text())
        self.assertEqual(claim['wrapper_pid'], launch['wrapper_pid'])
        self.assertEqual(launch['wrapper_pid'], left.pid if left.returncode == 0 else right.pid)
        self.assertEqual(len(json.loads((self.f.out / 'completion.json').read_text())['jobs']), 8)

    def test_recovery_run_name_preserves_prior_run_and_claims(self):
        old = self.root / 'runs' / 'train'
        old.mkdir(parents=True)
        marker = old / 'completion.json'
        marker.write_text('{"status":"FAILED","scientific_steps":0}')
        originals = {str(marker): marker.read_bytes()}
        for suffix in ('operator_claim.json', 'launch_claim.json', 'failed_launch_receipt.json'):
            p = self.root / ('train.' + suffix)
            p.write_text('{"retained":"failed import"}')
            originals[str(p)] = p.read_bytes()
        self.f.q['run_name'] = 'train_after_import_fix'
        self.f.out = self.root / 'runs' / 'train_after_import_fix'
        (self.root / 'access_logs').mkdir()
        for job in self.f.q['jobs']:
            old_access = Path(job['access_log'])
            old_access.write_text('{"retained":"old failed attempt"}')
            originals[str(old_access)] = old_access.read_bytes()
            job['access_log'] = str(old_access.parent / 'train_after_import_fix' / old_access.name)
            position = job['argv'].index('--access-log') + 1
            job['argv'][position] = job['access_log']
        self.f.write()
        result = self.f.run(wrapper=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        launch = json.loads((self.f.out / 'launch_receipt.json').read_text())
        self.assertEqual(launch['run_name'], 'train_after_import_fix')
        self.assertEqual(launch['phase'], 'train')
        self.assertTrue((self.root / 'train_after_import_fix.launch_claim.json').exists())
        self.assertTrue((self.root / 'train_after_import_fix.operator_claim.json').exists())
        for path, raw in originals.items():
            self.assertEqual(Path(path).read_bytes(), raw)

    def test_recovery_with_old_access_path_is_rejected_without_overwrite(self):
        self.f.q['run_name'] = 'train_after_import_fix'
        self.f.out = self.root / 'runs' / 'train_after_import_fix'
        access = Path(self.f.q['jobs'][0]['access_log'])
        access.parent.mkdir()
        raw = '{"retained":"prior access"}'
        access.write_text(raw)
        self.f.write()
        result = self.f.run(wrapper=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.f.out.exists())
        self.assertEqual(access.read_text(), raw)

    def test_run_name_path_traversal_rejected_before_dispatch(self):
        self.f.q['run_name'] = '../train'
        self.f.write()
        result = self.f.run(wrapper=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.f.out.exists())


if __name__ == '__main__':
    unittest.main()

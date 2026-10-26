"""功能概述：独立攻击机制恢复准入 checker 的只读元数据边界和旧准入复用。
本文件不构造或运行真实科学任务。MetadataBoundaryTests 仅在临时目录创建标量文本，
检查 canonical 元数据别名不能打开本地预测或张量，并检查末端复核继续执行同一边界。
AdmissionReuseTests 用失败哨兵确认原资源准入 verify 必须先执行，并核原854a请求错误
在构造 Bundle 前拒绝。测试依赖 review_fixed_mechanism_admission.py 的实际入口和
ValidationMetadata 方法，不修改作者源码，不读取真实预测、缓存、模型或报告。
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import review_fixed_mechanism_admission as m


class MetadataBoundaryTests(unittest.TestCase):
    def fixture(self, root, basename):
        path = root / basename
        path.write_bytes(b'{"synthetic":true}\n')
        key = '/synthetic/driver_start.json'
        reader = object.__new__(m.ValidationMetadata)
        reader.path = root / 'metadata_map.json'
        reader.map = {'files': {key: str(path)}}
        reader.hashes = {key: m.digest(path.read_bytes())}
        reader.reads = {}
        reader.pins = {}
        return reader, key, path

    def assert_no_payload_read(self, operation, path):
        original = Path.read_bytes
        opened = []
        def guarded(current):
            if current.resolve() == path.resolve():
                opened.append(str(current))
            return original(current)
        with patch.object(Path, 'read_bytes', guarded):
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual(opened, [])

    def test_canonical_metadata_alias_cannot_open_local_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader, key, path = self.fixture(Path(tmp).resolve(), 'predictions.jsonl')
            self.assert_no_payload_read(lambda: reader.raw(key), path)

    def test_canonical_metadata_alias_cannot_open_local_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader, key, path = self.fixture(Path(tmp).resolve(), 'features.pt')
            self.assert_no_payload_read(lambda: reader.raw(key), path)

    def test_finish_cannot_open_prediction_in_consumed_metadata_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader, key, path = self.fixture(Path(tmp).resolve(), 'predictions.jsonl')
            reader.reads = {str(path): reader.hashes[key]}
            self.assert_no_payload_read(reader.finish, path)

    def test_registered_scalar_metadata_roundtrip_and_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader, key, path = self.fixture(Path(tmp).resolve(), 'driver_start.json')
            self.assertEqual(json.loads(reader.raw(key)), {'synthetic': True})
            reader.finish()
            path.write_bytes(b'{"synthetic":false}\n')
            with self.assertRaises(ValueError):
                reader.finish()


class AdmissionReuseTests(unittest.TestCase):
    def test_original_resource_verify_is_mandatory_before_new_gate(self):
        bundle = unittest.mock.Mock()
        protocol_ref = {'path': '/synthetic/protocol.json', 'sha256': 'a' * 64}
        request = {'schema': 'cap10_mechanism_fixed_rows_recovery_request_20261025',
                   'root': '/synthetic', 'refs': {'protocol': protocol_ref}}
        prior = {'original': '854a request sentinel'}
        with patch.object(m.resource, 'verify', side_effect=ValueError('legacy854a_called')) as old:
            with self.assertRaisesRegex(ValueError, 'legacy854a_called'):
                m.verify(bundle, request, prior, None)
        old.assert_called_once_with(bundle, prior)

    def test_wrong_original_request_sha_rejected_before_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            request = root / 'request.json'
            prior = root / 'prior.json'
            request.write_text('{}')
            prior.write_text('{"wrong":"original request"}')
            argv = ['review_fixed_mechanism_admission.py', '--request', str(request),
                    '--manifest', str(root / 'unopened_manifest.json'),
                    '--base-request', str(prior),
                    '--validation-manifest', str(root / 'unopened_validation_map.json'),
                    '--out', str(root / 'unused_out')]
            with patch.object(sys, 'argv', argv), patch.object(m.base, 'Bundle') as bundle:
                with self.assertRaisesRegex(ValueError, 'exact_original_resource_request'):
                    m.main()
            bundle.assert_not_called()



class RegisteredQueueAttacks(unittest.TestCase):
    def fixture(self):
        from test_review_fixed_mechanism_admission import AdmissionFixture
        return AdmissionFixture()

    def test_boolean_ssh_exit_is_not_integer_zero(self):
        f = self.fixture()
        f.mutate(f.refs['JJ_execution'], lambda jj: jj.update(actual_ssh_exit=False))
        with self.assertRaisesRegex(ValueError, 'strict_actual_exit0'):
            f.check()

    def test_old_operator_cannot_replace_registered_resource_operator(self):
        f = self.fixture()
        f.mutate(f.qp, lambda q: q.update(operator_paths={'supervisor': {
            'path': str(f.f.root / 'supervise_controlled_queue.py'), 'sha256': 'a' * 64}}))
        with self.assertRaisesRegex(ValueError, 'all_other_queue_fields_identical'):
            f.check()

    def test_new_binding_cannot_prune_old_scientific_source(self):
        f = self.fixture()
        original_path = next(iter(f.f.binding['source_sha256']))
        f.mutate(f.bp, lambda b: b['source_sha256'].pop(original_path))
        with self.assertRaisesRegex(ValueError, 'runtime_old15_plus_new_leaf'):
            f.check()

if __name__ == '__main__':
    unittest.main()

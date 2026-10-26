"""功能概述：未来12test结果reader的人工输入边界测试；不读取真实预测或张量。
仅验证metadata先于预测、精确12解锁、copy/inventory SHA、tensor不打开及导出比较。
完整冻结/driver gate与5000/317数学沿用其已独审suite；本文件不冒称新的full12 main fixture。
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import review_cap_test_results as r

class StoreFixture:
    def __init__(self,directory):
        self.root=Path(directory).resolve();self.remote='/synthetic/cap12';self.files={};self.hashes={}
        self.add('protocol.json',b'{"synthetic":true}')
        for n in range(12):self.add(f'cell{n}/predictions.jsonl',b'{"artificial":true}\n')
        self.tensor=self.remote+'/features.pt';self.h='a'*64;self.inventory={**self.hashes,self.tensor:self.h}
        self.path=self.root/'manifest.json';self.save()
    def add(self,name,data):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        key=self.remote+'/'+name;self.files[key]=str(p);self.hashes[key]=r.digest(data);return key
    def save(self,**change):
        m=dict(files=self.files,sha256=self.hashes,hash_only_inputs={self.tensor:self.h},remote_before_sha256=self.inventory,
          remote_after_sha256=self.inventory,remote_inventory_before_exit=0,remote_inventory_after_exit=0,rsync_exit=0)
        m.update(change);self.path.write_text(json.dumps(m))
    def store(self):return r.Copies([self.path])
    def refs(self):return [((f'synthetic{n}',193),self.remote+f'/cell{n}/predictions.jsonl',self.hashes[self.remote+f'/cell{n}/predictions.jsonl']) for n in range(12)]

class ResultStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cap12_reader_');self.addCleanup(self.tmp.cleanup);self.f=StoreFixture(self.tmp.name)
    def test_constructor_and_gate_failure_do_not_read_first_prediction_byte(self):
        old=Path.read_bytes;opened=[]
        def guarded(p):
            if 'prediction' in p.name:opened.append(str(p));raise AssertionError('unexpected_prediction_open')
            return old(p)
        with patch.object(Path,'read_bytes',guarded):
            s=self.f.store()
            with self.assertRaisesRegex(ValueError,'prediction_before_complete12'):s.raw(self.f.refs()[0][1],self.f.refs()[0][2])
            s.json(self.f.remote+'/protocol.json');s.finish()
        self.assertEqual(opened,[])
    def test_only_exact12_registered_prediction_files_can_be_unlocked(self):
        s=self.f.store()
        with self.assertRaisesRegex(ValueError,'exact12_prediction_unlock'):s.unlock(self.f.refs()[:11])
        s.unlock(self.f.refs());self.assertEqual(s.json(self.f.refs()[0][1],self.f.refs()[0][2]),{'artificial':True})
        with self.assertRaisesRegex(ValueError,'prediction_before_complete12'):s.raw('/synthetic/legacy/predictions.jsonl','b'*64)
    def test_tensor_add_uses_exact_remote_SHA_without_tensor_read(self):
        s=self.f.store();self.assertEqual(s.add(self.f.tensor,self.f.h),self.f.h)
        with self.assertRaisesRegex(ValueError,'exact_SHA'):s.add(self.f.tensor)
        with self.assertRaisesRegex(ValueError,'canonical_tensor_read_forbidden'):s.raw(self.f.tensor)
    def test_local_prediction_cannot_be_relabelled_as_metadata(self):
        key=self.f.remote+'/protocol.json';self.f.files[key]=self.f.files[self.f.refs()[0][1]]
        self.f.hashes[key]=self.f.hashes[self.f.refs()[0][1]];self.f.inventory[key]=self.f.hashes[key];self.f.save()
        with self.assertRaisesRegex(ValueError,'payload_class_mismatch'):self.f.store().json(key)
    def test_strict_actual_exit_rejects_boolean_false(self):
        self.f.save(rsync_exit=False)
        with self.assertRaisesRegex(ValueError,'strict_int_EXIT0'):self.f.store()
    def test_remote_union_and_consumed_end_drift_reject(self):
        self.f.save(remote_after_sha256={**self.f.inventory,self.f.tensor:'b'*64})
        with self.assertRaisesRegex(ValueError,'actual_begin_end'):self.f.store()
        self.f.save();s=self.f.store();s.json(self.f.remote+'/protocol.json');Path(self.f.files[self.f.remote+'/protocol.json']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'consumed_end_drift'):s.finish()
    def test_symlink_to_tensor_rejects_before_tensor_read(self):
        target=self.f.root/'weights.pt';target.write_bytes(b'synthetic tensor bytes');link=self.f.root/'link.json';link.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'symlink_redirect'):r.local(link)
    def test_export_comparison_keeps_undefined_and_boolean_types(self):
        r.close({'x':None,'float':.25,'count':2},{'x':None,'float':.25,'count':2})
        with self.assertRaisesRegex(ValueError,'exact_value'):r.close(False,0)
        with self.assertRaisesRegex(ValueError,'exact_dict_keys'):r.close({'x':None,'extra':1},{'x':None})
    def test_locked_reviewed_helpers_import_without_scientific_data(self):
        stats,execution=r.load_helpers();self.assertEqual(Path(stats.c.__file__).resolve(),r.HERE/'native_cost_evaluation/cap_test_evaluation_common.py')
        self.assertTrue(callable(execution.actual_stage));self.assertEqual(len(stats.matrix.COMPARISONS),10)

if __name__=='__main__':unittest.main()

"""功能概述：20历史frozen hash-only输入的人工provenance组件测试。
承接17项当前文件与远端namespace边界，历史proof只读小JSON/源码；不创建/打开tensor。
这是reader组件fixture，非伪造真实SSH或新数学oracle。
"""
import json,types,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import review_cap_test_results_historical_inventory as r
import test_review_cap_test_remote_paths as prior
prior.r=r;prior.previous.r=r;prior.previous.old.r=r
MetadataStoreTests=prior.MetadataStoreTests
RemoteNamespaceTests=prior.RemoteNamespaceTests

class HistoryFixture:
    def __init__(self,root):
        self.root=Path(root).resolve();self.required={p:'a'*64 for p in r.HISTORICAL_PATHS};self.dir=self.root/'history';self.dir.mkdir()
        source=self.dir/'collector.py';source.write_text('# artificial metadata fixture\n')
        request=self.dir/'collection_request.json';request.write_text('{}')
        self.m=dict(schema='cap10_evaluation_admission_artifact_map_20261025',current_source_input_unchanged=True,test_read=False,
          remote_before_sha256=self.required,remote_after_sha256=self.required,hash_only_inputs=self.required,
          actual_inventory_first_ssh_exit=0,source_sha256={str(source):r.digest(source.read_bytes())},provenance=dict(request=str(request)))
        self.actual=dict(actual_exit=0,model_calls=0,inputs_unchanged=True,test_read=False,tensor_deserialization=False,
          remote_before_sha256=self.required,remote_after_sha256=self.required,utc='synthetic historical UTC, not a new actual run')
        self.before=self.dir/'actual_inventory.before.json';self.after=self.dir/'actual_inventory.after.json';self.inventory=self.dir/'inventory.request.json';self.map=self.dir/'artifact_map.json'
        self.snapshot=self.dir/'execution_snapshot.json';self.rec=dict(raw_records_unchanged=True,records=[dict(exit_code=0,screened=True,command=['python','-B',str(source),'--request',str(request),'--out',str(self.dir)])])
        self.ref={'collector_source':dict(path=str(source),sha256=r.digest(source.read_bytes()))};self.save()
        base=prior.previous.old.StoreFixture(self.root/'current');self.store=base.store()
    def save(self):
        def put(p,obj):p.write_text(json.dumps(obj));return dict(path=str(p),sha256=r.digest(p.read_bytes()))
        self.ref['before_receipt']=put(self.before,self.actual);after=put(self.after,self.actual)
        self.m['actual_inventory']={**after,'ssh_actual_exit':0};self.ref['inventory_request']=put(self.inventory,dict(expected_sha256=self.required))
        self.ref['execution_snapshot']=put(self.snapshot,self.rec);self.ref['artifact_map']=put(self.map,self.m)
    def check(self,required=None):
        with patch.object(r,'HISTORICAL_MAP_PATH',self.map),patch.object(r,'HISTORICAL_MAP_SHA',self.ref['artifact_map']['sha256']),patch.object(r,'HISTORICAL_EXECUTION_SNAPSHOT',self.ref['execution_snapshot']):
            self.store.import_historical_frozen_inputs(self.ref,self.required if required is None else required)

class HistoricalInputTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='cap_history_');self.addCleanup(self.tmp.cleanup);self.f=HistoryFixture(self.tmp.name)
    def test_exact20_history_is_hash_only_and_current_inventory_unchanged(self):
        before=dict(self.f.store.current_inventory);self.f.check();self.assertEqual(self.f.store.current_inventory,before)
        self.assertEqual({p:self.f.store.inventory[p] for p in self.f.required},self.f.required)
        self.assertFalse(self.f.store.historical_records[0]['rehashed_during_test_result_review']);self.f.store.finish()
    def test_historical_ssh_boolean_false_rejected(self):
        self.f.m['actual_inventory_first_ssh_exit']=False;self.f.save()
        with self.assertRaisesRegex(ValueError,'strict_int_EXIT0'):self.f.check()
    def test_only_registered20_no_other_historical_payload(self):
        changed=dict(self.f.required);changed['/synthetic/test/features.pt']='b'*64
        with self.assertRaisesRegex(ValueError,'exact20_registered'):self.f.check(changed)
    def test_current_inventory_never_overridden(self):
        self.f.store.inventory[next(iter(self.f.required))]='b'*64
        with self.assertRaisesRegex(ValueError,'cannot_override_current'):self.f.check()
    def test_actual_collection_failure_not_relabelled_success(self):
        self.f.rec['records'][0]['exit_code']=1;self.f.save()
        with self.assertRaisesRegex(ValueError,'strict_int_EXIT0'):self.f.check()
    def test_preserved_before_after_mismatch_rejected(self):
        self.f.actual['remote_after_sha256']={**self.f.required,next(iter(self.f.required)):'b'*64};self.f.save()
        with self.assertRaisesRegex(ValueError,'receipt_full_inventory'):self.f.check()

if __name__=='__main__':unittest.main()

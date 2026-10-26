"""功能概述：非作者核历史20 hash-only来源边界，迁移10个现有first-read/path攻击。
只人工metadata；历史证明不能补本次运行缺项，完整执行snapshot必须精确登记。
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import review_cap_test_results_historical_inventory as r
import test_review_cap_test_remote_paths_independent as previous
import test_review_cap_test_historical_inventory as author
previous.r=r;previous.previous.r=r;previous.previous.previous.r=r

class HistoricalReaderIndependentTests(previous.RemoteReaderIndependentTests):
    def test_current_runtime_missing_never_filled_from_historical(self):
        store,request,stats,i=self.parts()
        stats.execution=lambda *a:(dict(input_sha256={'/synthetic/current_required.pt':'f'*64},source_sha256={}),{},self.f.refs())
        with patch.object(store,'import_historical_frozen_inputs',wraps=store.import_historical_frozen_inputs) as imported:
            with self.assertRaisesRegex(ValueError,'mandatory_current85_runtime'):
                r.verify(store,request,stats,i)
            imported.assert_not_called()
        self.assertFalse(store.metadata_complete)

class HistoricalProvenanceIndependentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cap_history_independent_');self.addCleanup(self.tmp.cleanup)
        self.f=author.HistoryFixture(self.tmp.name)
        original_snapshot=dict(self.f.ref['execution_snapshot'])
        def check(required=None):
            with patch.object(r,'HISTORICAL_MAP_PATH',self.f.map),patch.object(r,'HISTORICAL_MAP_SHA',self.f.ref['artifact_map']['sha256']),patch.object(r,'HISTORICAL_EXECUTION_SNAPSHOT',original_snapshot):
                self.f.store.import_historical_frozen_inputs(self.f.ref,self.f.required if required is None else required)
        self.f.check=check
    def test_execution_snapshot_alias_cannot_replace_registered_snapshot(self):
        alternative=self.f.dir/'invented_success_snapshot.json'
        alternative.write_bytes(self.f.snapshot.read_bytes())
        self.f.ref['execution_snapshot']=dict(path=str(alternative),sha256=r.digest(alternative.read_bytes()))
        with self.assertRaisesRegex(ValueError,'registered.*execution|exact.*snapshot|historical.*execution'):
            self.f.check()
    def test_original_before_receipt_exact_path_required(self):
        alternative=self.f.dir/'unregistered_before.json';alternative.write_bytes(self.f.before.read_bytes())
        self.f.ref['before_receipt']=dict(path=str(alternative),sha256=r.digest(alternative.read_bytes()))
        with self.assertRaisesRegex(ValueError,'original_historical_before_path'):self.f.check()
    def test_historical_inventory_request_different_bundle_rejected(self):
        bad={'expected_sha256':dict(self.f.required)};bad['expected_sha256'].pop(next(iter(self.f.required)))
        self.f.inventory.write_text(json.dumps(bad));self.f.ref['inventory_request']['sha256']=r.digest(self.f.inventory.read_bytes())
        with self.assertRaisesRegex(ValueError,'original_historical_inventory_request'):self.f.check()

if __name__=='__main__':unittest.main()

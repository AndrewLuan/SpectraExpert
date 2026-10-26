"""功能概述：人工验证冻结训练metadata JSONL的精确准入，不读真实test/训练值。
承接原9边界，并验证未准入、错SHA、错误namespace及local payload假冒仍首读前拒绝。
"""
from pathlib import Path
import unittest
import review_cap_test_results_metadata as r
import test_review_cap_test_results as old
old.r=r

class MetadataStoreTests(old.ResultStoreTests):
    def prepare_metadata(self,name='rankers/C5/s193/no_aug/history.jsonl'):
        key=self.f.add(name,b'{"artificial_epoch":64}\n');self.f.inventory[key]=self.f.hashes[key];self.f.save();return key,self.f.hashes[key]
    def test_frozen_history_requires_explicit_exact_admission(self):
        key,h=self.prepare_metadata();s=self.f.store()
        with self.assertRaisesRegex(ValueError,'prediction_before_complete12'):s.raw(key,h)
        s.admit_training_metadata({key:h});self.assertEqual(s.raw(key,h),b'{"artificial_epoch":64}\n');self.assertFalse(s.metadata_complete)
        with self.assertRaisesRegex(ValueError,'frozen_training_metadata_expected_SHA'):s.raw(key,'b'*64)
    def test_unregistered_or_outside_ranker_name_rejected(self):
        key,h=self.prepare_metadata('cache/train/history.jsonl');s=self.f.store()
        with self.assertRaisesRegex(ValueError,'only_frozen_ranker'):s.admit_training_metadata({key:h})
    def test_wrong_frozen_digest_rejected_without_data_read(self):
        key,h=self.prepare_metadata('rankers/C5/s193/aug/center_rows.jsonl');s=self.f.store()
        with self.assertRaisesRegex(ValueError,'actual_inventory_pin'):s.admit_training_metadata({key:'b'*64})
    def test_canonical_training_metadata_cannot_map_to_predictions(self):
        key,h=self.prepare_metadata();self.f.files[key]=self.f.files[self.f.refs()[0][1]]
        self.f.hashes[key]=self.f.hashes[self.f.refs()[0][1]];self.f.inventory[key]=self.f.hashes[key];self.f.save()
        s=self.f.store();s.admit_training_metadata({key:self.f.hashes[key]})
        with self.assertRaisesRegex(ValueError,'payload_class_mismatch'):s.raw(key)

if __name__=='__main__':unittest.main()

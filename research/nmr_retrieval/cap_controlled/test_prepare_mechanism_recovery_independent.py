"""功能概述：非作者核新机制metadata准备器拒绝路径重标、实施范围缩减和JJ源漂移。
复用人工旧213input/4角色fixture，不读tensor/预测，不准备真实队列。
"""
import unittest
import test_prepare_mechanism_fixed_rows_recovery as a

class IndependentPreparationTests(unittest.TestCase):
    def setUp(self):self.f=a.Fixture()
    def test_same_protocol_bytes_wrong_canonical_path_rejected(self):
        ref=self.f.refs['protocol'];self.f.refs['protocol']=self.f.reader.add(self.f.root/'alias_protocol.json',self.f.reader.data[ref['path']])
        with self.assertRaisesRegex(ValueError,'protocol|binding'):self.f.prepare()
    def test_preparer_must_be_in_independent_implementation_scope(self):
        value=self.f.reader.ref(self.f.refs['implementation_review']);value['reviewed_sha256'].pop(self.f.refs['preparer']['path']);self.f.replace('implementation_review',value)
        with self.assertRaisesRegex(ValueError,'complete_implementation_review'):self.f.prepare()
    def test_JJ_source_changed_after_suite_refused(self):
        value=self.f.reader.ref(self.f.refs['JJ_execution']);value['source_end_sha256']={};self.f.replace('JJ_execution',value)
        with self.assertRaisesRegex(ValueError,'actual_JJ_same_sources'):self.f.prepare()

if __name__=='__main__':unittest.main()

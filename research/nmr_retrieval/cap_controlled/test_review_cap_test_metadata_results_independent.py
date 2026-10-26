"""功能概述：非作者窄审训练JSONL metadata准入；继承旧6个读边界攻击。
仅人工小字节，确认具体冻结与分析独审之前不注册，test/sealed namespace无例外。
"""
import unittest
from unittest.mock import patch
import review_cap_test_results_metadata as r
import test_review_cap_test_results_independent as previous
previous.r=r

class MetadataAdmissionIndependentTests(previous.IndependentReadBoundaryTests):
    def test_test_or_sealed_ranker_history_never_admitted(self):
        for name in ('rankers/test/history.jsonl','rankers/sealed_resume_fake/center_rows.jsonl'):
            key=self.f.add(name,b'{"artificial":64}\n');self.f.inventory[key]=self.f.hashes[key];self.f.save()
            s=self.f.store()
            with self.assertRaisesRegex(ValueError,'only_frozen_ranker'):
                s.admit_training_metadata({key:self.f.hashes[key]})
            self.assertEqual(s.training_metadata,{})
    def test_specific_and_analysis_gate_failures_do_not_admit_training_metadata(self):
        for stage in ('specific','analysis'):
            s,q,stats,i=self.parts()
            original=stats.c.sealed_gate
            def gate(*a,**kw):
                if stage=='specific':raise ValueError('specific_review_rejected')
                result=original(*a,**kw);result[3]['input_sha256']={'/synthetic/cap12/rankers/C5/history.jsonl':'a'*64}
                return result
            stats.c.sealed_gate=gate
            if stage=='analysis':
                stats.c.reviewed=lambda *a:(_ for _ in ()).throw(ValueError('analysis_review_rejected'))
            with patch.object(s,'admit_training_metadata',wraps=s.admit_training_metadata) as admission:
                with self.assertRaisesRegex(ValueError,stage+'_review_rejected'):
                    r.verify(s,q,stats,i)
                admission.assert_not_called()
            self.assertFalse(s.metadata_complete)
            self.assertFalse(any(e[0]=='prediction' for e in s.events))

if __name__=='__main__':unittest.main()

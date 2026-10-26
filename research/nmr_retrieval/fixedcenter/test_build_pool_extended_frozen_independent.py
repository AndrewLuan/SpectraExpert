"""功能概述：扩展冻结器独立破坏性核验，仅读临时metadata，不运行真实test。"""
import json
from pathlib import Path
import tempfile
import unittest
import build_pool_extended_frozen as m
from test_build_pool_extended_frozen import Fixture


class IndependentTests(unittest.TestCase):
    def test_missing_initial_or_last_audit_pin_rejected(self):
        for name in ('initial.pt','last.pt'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                f=Fixture(tmp);path=str(Path(f.models[0]['output'])/name)
                del f.audit['input_sha256'][path];f.audit_path.write_text(json.dumps(f.audit))
                with self.assertRaisesRegex(ValueError,'additional_audit_missing_artifact'):
                    m.check_additional(f.root,f.queue_path,f.run,f.audit_path,{})
    def test_all_fixed_jobs_bind_centers_and_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);receipt,frozen,queue=f.build()
            self.assertEqual(set(frozen['sealed_evidence_paths']),{
                'main_validation_summary','training_audit','additional_training_audit','preregistration',
                'additional_preregistration','eligibility','builder','legacy_builder','joint_summarizer',
                'validation_summarizer','training_auditor','legacy_training_auditor','test_summarizer','validation_review'})
            jobs=[j for j in queue['jobs'] if '--config' in j['argv']]
            self.assertEqual(len(jobs),4)
            for job in jobs:
                config=job['argv'][job['argv'].index('--config')+1]
                seed=job['argv'][job['argv'].index('--seed')+1]
                fixed=frozen['fixed_center_configurations'][config][seed]
                for artifact in fixed['center_artifacts']:
                    for key in ('centers','row_audit'):
                        path=artifact[key];digest=artifact[key+'_sha256']
                        self.assertEqual(job['inputs_sha256'][path],digest)
                        self.assertEqual(frozen['reviewed_inputs_sha256'][path],digest)
                self.assertIn(str(f.review_path),job['inputs_sha256'])
                self.assertIn(job['argv'][1],job['inputs_sha256'])
            self.assertEqual(sum('-m' in j['argv'] for j in queue['jobs']),20)
    def test_plan_cannot_drop_critical_reviewed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);v=json.loads(f.review_path.read_text())
            del v['reviewed_sha256'][str(f.registration)]
            f.review_path.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'review_does_not_bind'):
                f.build()


if __name__=='__main__':unittest.main()

"""功能概述：非作者独立审计回归，使用作者tiny64 fixture复现provenance删减与安全边界。
本文件不修改审计源码，不读真实cache/test，不执行远程任务。每个攻击自行更新producer
JSON/byteSHA，要求被审计实现依据完整契约拒绝，不能只依赖旧摘要hash偶然失败。
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import audit_cap_training as a
import cap_ranker as cap
from test_audit_cap_training import ModelAuditFixture

class AuditProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        tiny=dict(cap.BUDGET,samples=2,batch_size=2,draws_per_fit=128,optimizer_steps=64)
        self.ps=[patch.object(cap,'BUDGET',tiny),patch.object(a,'BUDGET',tiny)]
        for p in self.ps:p.start()
        self.f=ModelAuditFixture(self.temp.name)
        initial=torch.load(self.f.f.out/'initial.pt',map_location='cpu',weights_only=False)
        self.admitted={**initial['input_sha256'],**initial['ranker_source_sha256']}
    def tearDown(self):
        for p in reversed(self.ps):p.stop()
        self.temp.cleanup()
    def check(self):
        return a.audit_model(self.f.f.out,self.f.record['checkpoint'],self.f.record,self.f.cap,193,
          self.f.role,self.f.expected,a.Inputs(),self.admitted)
    def test_complete_input_and_model_sources_cannot_be_erased(self):
        self.f.final(lambda v:v.update(input_sha256={},ranker_source_sha256={}))
        p=self.f.f.out/'summary.json';s=json.loads(p.read_text());s['source_inputs_start_sha256']={};s['source_inputs_end_sha256']={};p.write_text(json.dumps(s))
        with self.assertRaises(ValueError):self.check()
    def test_parent_forward_checkpoint_and_training_cache_lineage_cannot_change(self):
        self.f.final(lambda v:v.update(train_sha256='0'*64,dual_train_sha256='1'*64,cache_protocol={'forged':'different_forward'},source_inputs_sha256={}))
        with self.assertRaises(ValueError):self.check()
    def test_final_outpath_cannot_claim_another_model_directory(self):
        self.f.final(lambda v:v['args'].update(out=str(self.f.f.out/'another')))
        with self.assertRaises(ValueError):self.check()

class AuditPathTests(unittest.TestCase):
    def test_initial_forbidden_symlink_rejected_before_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'sealed_resume_forbidden/values.pt';target.parent.mkdir();target.write_text('synthetic sentinel')
            link=root/'metadata.json';link.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'forbidden'):a.Inputs().pin(link)
    def test_finish_revalidates_symlink_before_any_forbidden_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);metadata=root/'metadata.json';metadata.write_text('{}')
            inputs=a.Inputs();inputs.pin(metadata)
            target=root/'sealed_resume_forbidden/values.pt';target.parent.mkdir();target.write_text('synthetic sentinel')
            metadata.unlink();metadata.symlink_to(target)
            original=a.sha;bad=[]
            def spy(path):
                if '/sealed' in str(Path(path).resolve()):
                    bad.append(str(path));raise AssertionError('forbidden file would be read before rejection')
                return original(path)
            with patch.object(a,'sha',side_effect=spy):
                with self.assertRaises(ValueError):inputs.finish()
            self.assertEqual(bad,[])
if __name__=='__main__':unittest.main()

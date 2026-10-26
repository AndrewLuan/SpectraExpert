"""功能概述：独立攻击固定中心postfit审计的证据边界。
使用作者合成文件树，覆盖验证metadata缺失、最佳checkpoint冻结状态、
继承alpha与审计过程中输入漂移；不读取真实谱或执行科学训练。
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import audit_pool_fixed_center_training as a
import test_audit_pool_fixed_center_training as fixtures


class IndependentAuditTests(unittest.TestCase):
    def run_tree(self, mutate=None, drift=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            queue,run,jobs=fixtures.build_tree(root)
            if mutate: mutate(root,queue,run,jobs)
            real_sha=a.sha
            seen={}
            def drifting_sha(path):
                path=Path(path).resolve()
                value=real_sha(path)
                if path.name=="center_rows.jsonl":
                    seen[path]=seen.get(path,0)+1
                    if seen[path]>1:return "0"*64
                return value
            with patch.object(a.old,"recompute_exposure",return_value=(fixtures.oldfixture()[-1],1,1)):
                if drift:
                    with patch.object(a,"sha",side_effect=drifting_sha):
                        return a.audit(root,queue,run)
                return a.audit(root,queue,run)

    def test_validation_metadata_required(self):
        def mutate(root,queue,run,jobs):
            for job in jobs:
                if job["kind"]=="validation":
                    for name in ("summary.json","protocol.json"):
                        p=Path(job["out"])/name
                        if p.exists():p.unlink()
        with self.assertRaises((ValueError,FileNotFoundError,KeyError)):
            self.run_tree(mutate)

    def test_best_scale_freeze_rejected(self):
        v=fixtures.fixture()
        v[3]["best"]["scale"]=v[3]["best"]["scale"].clone()+1
        with self.assertRaisesRegex(ValueError,"scale_changed"):
            a.audit_model(*v)

    def test_inherited_alpha_rejected(self):
        v=fixtures.fixture()
        v[3]["initial"]["alpha"]=v[2]["alpha"]+0.25
        with self.assertRaisesRegex(ValueError,"initial_scale_alpha_changed"):
            a.audit_model(*v)

    def test_read_input_drift_rejected_at_final_pass(self):
        with self.assertRaisesRegex(ValueError,"audit_input_changed"):
            self.run_tree(drift=True)


if __name__=="__main__":unittest.main()

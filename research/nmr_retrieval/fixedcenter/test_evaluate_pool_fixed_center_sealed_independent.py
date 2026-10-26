"""功能概述：用扩展冻结器的真实合成输出独立检查sealed适配器的跨工具契约。
只操作临时合成缓存；不加载测试tensor。覆盖24job named映射、review缺具体计划的
先行拒绝和同权重跨pool重标拒绝，防止两个工具各自fixture通过却接口不兼容。
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate_pool_fixed_center_sealed as a
from test_build_pool_extended_frozen import Fixture


class CrossToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name)
        self.receipt,self.frozen,self.queue=self.f.build()
        config="fixedcenter_mixed_listwise_single"
        job=next(j for j in self.queue["jobs"] if j["id"]==f"sealed_{config}_s193")
        argv=job["argv"]
        option=lambda key:argv[argv.index("--"+key)+1]
        self.args=SimpleNamespace(source=option("source"),config=option("config"),seed=int(option("seed")),
            cache=option("cache"),no_aug_checkpoint=option("no-aug-checkpoint"),
            aug_checkpoint=option("aug-checkpoint"),frozen_selection=option("frozen-selection"),out=option("out"))
        self.sources=[self.f.adapter,self.f.fixed]
        self.sources += [self.f.f.source/"nmr_transfer"/n for n in
            ("h_pool_ranker.py","h_research_ranker.py","h_research_observation.py","h_research_contract.py",
             "h_global_ranker.py","h_global_features.py","data.py")]

    def call(self):
        # Relocate only source paths into the synthetic artifact tree. All metadata reads/hashes are real.
        with patch.object(a,"source_paths",return_value=self.sources):
            return a.preflight(self.args)

    def test_builder_output_passes_actual_preflight(self):
        frozen,selected,pins=self.call()
        self.assertEqual(selected["seed"],193)
        self.assertEqual(selected["recall"],"single")
        self.assertEqual(len(selected["center_artifacts"]),2)
        self.assertIn(self.args.cache,pins)

    def test_unrelated_PASS_review_rejected_before_test_hash(self):
        path=Path(self.f.f.base/"unrelated.json");path.write_text("{}")
        review_path=Path(self.frozen["freeze_review"]["path"])
        review=dict(status="PASS",independent=True,reviewer="independent_attack",
            scope="freeze_plan_and_implementation",reviewed_sha256={str(path):a.sha(path)})
        review_path.write_text(json.dumps(review))
        self.frozen["freeze_review"]["sha256"]=a.sha(review_path)
        self.frozen["reviewed_inputs_sha256"]={str(path):a.sha(path)}
        Path(self.args.frozen_selection).write_text(json.dumps(self.frozen))
        actual=a.sha
        def guard(path):
            if str(Path(path).resolve())==self.args.cache:
                raise AssertionError("test_cache_hashed_before_specific_plan_review")
            return actual(path)
        with patch.object(a,"sha",side_effect=guard):
            with self.assertRaises((ValueError,KeyError)):
                self.call()

    def test_config_pool_cannot_be_relabelled(self):
        self.args.config="fixedcenter_mixed_listwise_dual"
        with self.assertRaisesRegex(ValueError,"sealed_fixed_allowlist|sealed_named_allowlist"):
            self.call()


if __name__=="__main__":unittest.main()

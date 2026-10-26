"""功能概述：独立合成验证只读freeze输入inventory的停止与副作用边界。
实际调用完整合成builder直到review sentinel；攻击路径以临时假builder模拟写入、
低层文件变更与test读入，要求在副作用发生前拒绝。测试不生成真实freeze或运行模型。
"""
import builtins
import argparse
import sys
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import inventory_extended_freeze_inputs as op
from test_build_pool_extended_frozen import Fixture


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name);self.args=self.f.args()
        self.args.out=str(self.f.f.base/"never_construct_frozen")
        Path(self.args.review).write_text(json.dumps(dict(status="PENDING",scope="freeze_plan_and_implementation")))

    def capture(self):
        pending=json.loads(Path(self.args.review).read_text())
        op.builder.old.review(pending,{str(self.f.fixed):op.digest(self.f.fixed)})

    def fake(self,operation):
        def build(args,out):
            operation()
            self.capture()
        return build

    def test_actual_builder_stops_at_review_and_outputs_no_frozen(self):
        prior={str(p) for p in self.f.f.base.rglob("*")}
        value=op.collect(self.args)
        self.assertEqual(value["status"],"INVENTORY_ONLY_NOT_REVIEWED")
        self.assertTrue(value["review_not_approved"])
        self.assertFalse(value["actual_freeze"]);self.assertFalse(value["actual_test"])
        self.assertFalse(Path(self.args.out).exists())
        self.assertEqual(prior,{str(p) for p in self.f.f.base.rglob("*")})
        self.assertIn(str(self.f.adapter),value["required_sha256"])
        self.assertIn(self.f.plan["evaluations"][0]["cache"],value["required_sha256"])

    def test_normal_builder_return_fails_and_restores_guards(self):
        original=(builtins.open,io.open,op.builder.old.review)
        with patch.object(op.builder,"build",return_value={}):
            with self.assertRaisesRegex(AssertionError,"did_not_stop"):
                op.collect(self.args)
        self.assertEqual(original,(builtins.open,io.open,op.builder.old.review))

    def test_high_level_write_forbidden(self):
        target=self.f.f.base/"forbidden.json"
        with patch.object(op.builder,"build",side_effect=self.fake(lambda:target.write_text("{}"))):
            with self.assertRaises(AssertionError):op.collect(self.args)
        self.assertFalse(target.exists())

    def test_low_level_mkdir_forbidden_before_side_effect(self):
        target=self.f.f.base/"forbidden_directory"
        with patch.object(op.builder,"build",side_effect=self.fake(lambda:os.mkdir(target))):
            with self.assertRaises(AssertionError):op.collect(self.args)
        self.assertFalse(target.exists())

    def test_unlink_forbidden_before_side_effect(self):
        target=self.f.f.base/"keep_this_file";target.write_text("keep")
        with patch.object(op.builder,"build",side_effect=self.fake(lambda:target.unlink())):
            with self.assertRaises(AssertionError):op.collect(self.args)
        self.assertTrue(target.exists())

    def test_path_mkdir_forbidden_before_side_effect(self):
        target=self.f.f.base/"forbidden_path_directory"
        with patch.object(op.builder,"build",side_effect=self.fake(lambda:target.mkdir())):
            with self.assertRaises(AssertionError):op.collect(self.args)
        self.assertFalse(target.exists())

    def test_rename_forbidden_before_side_effect(self):
        source=self.f.f.base/"rename_source";source.write_text("keep")
        target=self.f.f.base/"rename_target"
        with patch.object(op.builder,"build",side_effect=self.fake(lambda:source.rename(target))):
            with self.assertRaises(AssertionError):op.collect(self.args)
        self.assertTrue(source.exists());self.assertFalse(target.exists())

    def test_os_open_write_flags_forbidden_before_side_effect(self):
        target=self.f.f.base/"low_level_write"
        def operation():
            fd=os.open(target,os.O_WRONLY|os.O_CREAT,0o600)
            os.close(fd)
        with patch.object(op.builder,"build",side_effect=self.fake(operation)):
            with self.assertRaises(AssertionError):op.collect(self.args)
        self.assertFalse(target.exists())

    def test_builder_sha_mismatch_fails(self):
        with patch.object(op,"EXPECTED_BUILDER","0"*64):
            with self.assertRaisesRegex(AssertionError,"builder_source_drift"):
                op.collect(self.args)

    def test_pass_review_cannot_replace_pending_placeholder(self):
        Path(self.args.review).write_text(json.dumps(dict(status="PASS",scope="freeze_plan_and_implementation")))
        with self.assertRaisesRegex(AssertionError,"explicit_pending"):op.collect(self.args)

    def test_sentinel_without_capture_cannot_succeed(self):
        with patch.object(op.builder,"build",side_effect=op.InventoryStop):
            with self.assertRaises(AssertionError):op.collect(self.args)

    def test_main_writes_only_separate_inventory_json(self):
        args_path=self.f.f.base/"builder_args.json";args_path.write_text(json.dumps(vars(self.args)))
        output=self.f.f.base/"inventory.json"
        with patch.object(sys,"argv",["inventory","--builder-args-file",str(args_path),"--inventory-output",str(output)]):
            op.main()
        value=json.loads(output.read_text())
        self.assertEqual(value["status"],"INVENTORY_ONLY_NOT_REVIEWED")
        self.assertEqual(value["args_file_sha256"],op.digest(args_path))
        self.assertEqual(value["operator_sha256"],op.digest(op.__file__))
        self.assertFalse(Path(self.args.out).exists())

    def test_pending_review_cannot_read_test_tensor(self):
        target=self.f.f.base/"old/sealed/features/inherited_s193/single/features.pt"
        target.parent.mkdir(parents=True);target.write_text(json.dumps(dict(status="PENDING",scope="freeze_plan_and_implementation")))
        self.args.review=str(target)
        with patch.object(op.builder,"build",side_effect=self.fake(lambda:None)):
            with self.assertRaises(AssertionError):op.collect(self.args)

    def test_test_predictions_forbidden(self):
        target=self.f.f.base/"test/predictions.jsonl"
        target.parent.mkdir();target.write_text("{}")
        with patch.object(op.builder,"build",side_effect=self.fake(lambda:target.read_text())):
            with self.assertRaises(AssertionError):op.collect(self.args)


if __name__=="__main__":unittest.main()

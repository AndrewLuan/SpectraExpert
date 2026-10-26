"""功能概述：独立核test metadata builder的准入、native路径与叶源/统计源分界。
全部人工metadata；不读取真实缓存、预测或报告，不派发任何科学任务。
"""
import copy
import json
import tempfile
from unittest import mock
from pathlib import Path
import unittest
import prepare_cap_test_metadata as m
from test_prepare_cap_test_metadata import Fixture

def rebind_registry(f):
    r=f.registry
    source=m.merge(r["leaf_source_sha256"],r["downstream_source_sha256"],r["operator_source_sha256"],{r["tool_paths"]["builder"]["path"]:r["tool_paths"]["builder"]["sha256"]})
    f.replace("source_registry",r)
    for key in ("test_implementation_review","statistics_implementation_review","builder_implementation_review"):
        old=f.reader.ref(f.refs[key]);old["reviewed_sha256"]=source;f.replace(key,old)

class IndependentMetadataTests(unittest.TestCase):
    def test_old_dtype_evaluator_paths_are_rejected(self):
        f=Fixture();r=f.registry
        for key in ("evaluator","common"):
            old=r["tool_paths"][key]
            ref=f.reader.add(f.root/Path(old["path"]).name,b"# synthetic old root precision source\n")
            r["leaf_source_sha256"].pop(old["path"]);r["leaf_source_sha256"][ref["path"]]=ref["sha256"];r["tool_paths"][key]=ref
        rebind_registry(f)
        with self.assertRaisesRegex(ValueError,"native|tool|path"):f.prepare()

    def test_downstream_statistics_cannot_alias_leaf_evaluator(self):
        f=Fixture();r=f.registry
        old=r["tool_paths"]["statistics"];r["downstream_source_sha256"].pop(old["path"])
        r["tool_paths"]["statistics"]=r["tool_paths"]["evaluator"]
        ref=r["tool_paths"]["statistics"];r["downstream_source_sha256"][ref["path"]]=ref["sha256"]
        rebind_registry(f)
        with self.assertRaisesRegex(ValueError,"downstream|source|tool"):f.prepare()

    def test_old_root_operators_cannot_replace_new_resource_roles(self):
        f=Fixture();registry=f.registry
        for oldname,newname in (("supervise_controlled_queue.py","supervise_resource_ready_queue.py"),("launch_controlled_stage.py","launch_resource_ready_stage.py")):
            registry["operator_source_sha256"].pop(str(f.root/newname))
            ref=f.reader.add(f.root/oldname,b"# synthetic self-consistent old operator\n")
            registry["operator_source_sha256"][ref["path"]]=ref["sha256"]
        rebind_registry(f)
        with self.assertRaisesRegex(ValueError,"exact_operator_sources"):f.prepare()

    def test_generated_queue_new_operator_refs_and_fixed_family_seed_weights(self):
        f=Fixture();req=f.freeze_request();frozen,q,required=m.freeze(f.reader,req)
        self.assertEqual(q["operator_paths"]["supervisor"]["path"],str(f.root/"supervise_resource_ready_queue.py"))
        self.assertEqual(q["operator_paths"]["launcher"]["path"],str(f.root/"launch_resource_ready_stage.py"))
        self.assertEqual(q["resource_wait"],dict(maximum_seconds=120,poll_seconds=.2))
        self.assertEqual(q["input_sha256"],f.prepare()[0]["input_sha256"])
        by={(e["config"],e["seed"]):e for e in frozen["entries"]}
        for family in ("fixed","cap5","cap10"):
            for seed in m.SEEDS:
                self.assertEqual(by[(family+"_single",seed)]["checkpoints"],by[(family+"_dual",seed)]["checkpoints"])
        self.assertTrue(set(f.registry["downstream_source_sha256"]).isdisjoint(q["input_sha256"]))
        self.assertTrue(set(f.registry["downstream_source_sha256"]) <= set(frozen["source_sha256"]))
        self.assertEqual(frozen["validation_selection"],f.refs["validation_lock"])
        self.assertEqual(frozen["validation_rule_applied_selection"],f.refs["validation_selection"])

    def test_no_lock_rejects_before_any_registered_test_metadata(self):
        f=Fixture();f.lock["status"]="PENDING";f.replace("validation_lock",f.lock)
        with self.assertRaises(ValueError):f.prepare()
        self.assertNotIn(f.refs["legacy_test_plan"]["path"],f.reader.reads)
        self.assertFalse(any("/old/sealed/" in p for p in f.reader.reads))

    def test_twelve_cells_are_full_cartesian_and_all_evaluate(self):
        f=Fixture();b,p,_=f.prepare()
        self.assertEqual({(e["config"],e["seed"]) for e in p["entries"]},{(c,s) for c in m.CONFIGS for s in m.SEEDS})
        self.assertEqual(len(p["entries"]),12)
        self.assertTrue(all(e["execution"]=="evaluate" for e in p["entries"]))
        self.assertTrue(set(f.registry["downstream_source_sha256"]).isdisjoint(b["source_sha256"]))


class ReaderSafetyTests(unittest.TestCase):
    def mapped(self,root,meta):
        mp=root/"metadata_map.json";h=m.sha(meta.read_bytes())
        mp.write_bytes(m.encoded(dict(files={str(meta):str(meta)},sha256={str(meta):h})))
        return m.Reader(dict(path=str(mp),sha256=m.sha(mp.read_bytes()))),h
    def test_tensor_symlink_rejected_before_tensor_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();tensor=root/"features.pt";tensor.write_bytes(b"forbidden synthetic tensor")
            meta=root/"metadata.json";meta.symlink_to(tensor)
            mp=root/"map.json";h=m.sha(b"forbidden synthetic tensor")
            mp.write_bytes(m.encoded(dict(files={str(meta):str(meta)},sha256={str(meta):h})))
            reader=m.Reader(dict(path=str(mp),sha256=m.sha(mp.read_bytes())))
            with mock.patch.object(Path,"read_bytes",side_effect=AssertionError("unexpected payload read")):
                with self.assertRaisesRegex(ValueError,"unsafe_metadata_path"):reader.raw(str(meta),h)
    def test_finish_refuses_tensor_symlink_drift_before_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();meta=root/"metadata.json";meta.write_text("{}")
            reader,h=self.mapped(root,meta);reader.raw(str(meta),h)
            tensor=root/"features.pt";tensor.write_bytes(b"synthetic blocked tensor")
            meta.unlink();meta.symlink_to(tensor)
            original=Path.read_bytes
            def guarded(p):
                if p.resolve()==tensor:raise AssertionError("tensor was read")
                return original(p)
            with mock.patch.object(Path,"read_bytes",guarded):
                with self.assertRaisesRegex(ValueError,"unsafe_metadata_path"):reader.finish()
    def test_locked_C5_is_never_an_alternative(self):
        f=Fixture();f.summary["preselected"]="cap5_dual";f.replace("validation_summary",f.summary)
        with self.assertRaisesRegex(ValueError,"locked_preselected"):f.prepare()

if __name__=="__main__":unittest.main()


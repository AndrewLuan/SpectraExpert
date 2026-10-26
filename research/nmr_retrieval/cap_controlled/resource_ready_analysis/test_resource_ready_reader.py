"""功能概述：完整16+4人工回执验证resource-ready operator显式路径与旧训练链兼容。
只更换独立queue/outer evidence；scientific leaf、binding输入、所有scalar结果保持原样。
覆盖精确operator SHA/argv/run_name/120秒预算与严格双退出，零模型、零真实cache读取。
"""
import copy,json,tempfile
from pathlib import Path
from unittest import mock
import unittest
import cap_evaluation_common as c
import summarize_cap_validation as s
from test_summarize_cap_validation import MatrixFixture,put,plain
from test_cap_mechanism_entry import MechanismFixture

def install_resource_operator(queue_path,run_path,run_name):
    queue_path=Path(queue_path);run=Path(run_path);root=queue_path.parent
    q=json.loads(queue_path.read_text());q["run_name"]=run_name
    newrun=root/"runs"/run_name
    if run!=newrun:run.rename(newrun)
    refs={}
    for role,name in (("supervisor","supervise_resource_ready_queue.py"),("launcher","launch_resource_ready_stage.py")):
        p=root/name;plain(p,"# synthetic resource-ready operator source\n");refs[role]=dict(path=str(p),sha256=c.sha(p))
    q["operator_paths"]=refs;q["resource_wait"]=dict(maximum_seconds=120,poll_seconds=.2)
    q["source_sha256"].update({r["path"]:r["sha256"] for r in refs.values()})
    for name in ("cap_evaluation_common.py","summarize_cap_validation.py","analyze_cap_mechanism.py"):
        p=Path(__file__).with_name(name).resolve();q["source_sha256"][str(p)]=c.sha(p)
    put(queue_path,q);qsha=c.sha(queue_path)
    argv=[q["python"],"-B",refs["supervisor"]["path"],"--queue",str(queue_path),"--out",str(newrun),"--execute"]
    for name in ("driver_start.json","launch_receipt.json","completion.json"):
        p=newrun/name;v=json.loads(p.read_text());v["queue_sha256"]=qsha
        v["operator_paths"]=refs
        if name!="launch_receipt.json":v["resource_wait"]=q["resource_wait"]
        if name!="completion.json":v["source_sha256"]=q["source_sha256"]
        if name=="driver_start.json":v["argv"]=argv[2:]
        if name=="launch_receipt.json":v["argv"]=argv
        if name=="completion.json":
            for rec in v["jobs"]:
                rec["queue_sha256"]=qsha;put(newrun/"jobs"/(rec["id"]+".receipt.json"),rec)
        put(p,v)
    for ref in q["review_paths"]:
        v=json.loads(Path(ref).read_text());v["reviewed_sha256"]={**q["source_sha256"],str(queue_path):qsha};put(ref,v)
    small={**q["source_sha256"],str(queue_path):qsha,**{ref:c.sha(ref) for ref in q["review_paths"]}}
    for when in ("before","after"):put(newrun/("operator_fingerprints."+when+".json"),small)
    return newrun

class ResourceReaderTests(unittest.TestCase):
    def test_full16_selection_new_operator_legacy_training_and_unchanged_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);before=Path(f.bindings).read_bytes();old_source=copy.deepcopy(json.loads(before)["source_sha256"])
            f.run=install_resource_operator(f.queue,f.run,"validation_resource_ready")
            r=f.execute()
            self.assertEqual(r["table_rows"],648);self.assertEqual(r["preselected"],"cap10_dual")
            self.assertEqual(r["validation_run"],str(f.run));self.assertEqual(Path(f.bindings).read_bytes(),before)
            self.assertEqual(json.loads(before)["source_sha256"],old_source)
    def test_full16plus4_mechanism_and_fixed400_87_new_operators(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MechanismFixture(tmp)
            f.f.run=install_resource_operator(f.f.queue,f.f.run,"validation_resource_ready")
            f.run=install_resource_operator(f.queue,f.run,"mechanism_resource_ready")
            r=f.execute()
            self.assertEqual(r["fixed_train_pair_analysis"]["all400"]["query_seed_pairs"],400)
            self.assertEqual(r["fixed_train_pair_analysis"]["original87"]["total_bound_release_count"],{"5":0,"10":87})
            self.assertEqual(len(r["common_candidate_invariance"]),8)
    def test_wrong_supervisor_SHA_ref_rejected_before_output_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);f.run=install_resource_operator(f.queue,f.run,"validation_resource_ready")
            q=json.loads(Path(f.queue).read_text());q["operator_paths"]["supervisor"]["sha256"]="a"*64;put(f.queue,q)
            h=c.sha(f.queue)
            for name in ("driver_start.json","launch_receipt.json","completion.json"):
                p=f.run/name;v=json.loads(p.read_text());v["queue_sha256"]=h;put(p,v)
            with self.assertRaisesRegex(ValueError,"operator_source_SHA_binding"):c.driver_receipts(c.Pins(),f.queue,f.run)
    def test_old_driver_argv_is_rejected_for_new_operator(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);f.run=install_resource_operator(f.queue,f.run,"validation_resource_ready")
            for name in ("driver_start.json","launch_receipt.json"):
                p=f.run/name;v=json.loads(p.read_text());v["argv"][0 if name=="driver_start.json" else 2]=str(f.root/"supervise_controlled_queue.py");put(p,v)
            with self.assertRaisesRegex(ValueError,"outer_exact_argv"):c.driver_receipts(c.Pins(),f.queue,f.run)
    def test_noncanonical_launcher_role_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);f.run=install_resource_operator(f.queue,f.run,"validation_resource_ready")
            q=json.loads(Path(f.queue).read_text());q["operator_paths"]["launcher"]["path"]=str(f.root/"other.py")
            put(f.queue,q)
            # keep queueSHA envelopes self-consistent, so rejection specifically tests the role contract.
            h=c.sha(f.queue)
            for name in ("driver_start.json","launch_receipt.json","completion.json"):
                p=f.run/name;v=json.loads(p.read_text());v["queue_sha256"]=h;put(p,v)
            with self.assertRaisesRegex(ValueError,"canonical_resource_operator_path"):c.driver_receipts(c.Pins(),f.queue,f.run)
    def test_wait_budget_requires_exact_registered_type_and_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);f.run=install_resource_operator(f.queue,f.run,"validation_resource_ready")
            q=json.loads(Path(f.queue).read_text());q["resource_wait"]["maximum_seconds"]=120.0;put(f.queue,q)
            h=c.sha(f.queue)
            for name in ("driver_start.json","launch_receipt.json","completion.json"):
                p=f.run/name;v=json.loads(p.read_text());v["queue_sha256"]=h;put(p,v)
            with self.assertRaisesRegex(ValueError,"exact_bounded_resource_wait"):c.driver_receipts(c.Pins(),f.queue,f.run)
    def test_missing_explicit_launcher_ref_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);f.run=install_resource_operator(f.queue,f.run,"validation_resource_ready")
            q=json.loads(Path(f.queue).read_text());q["operator_paths"].pop("launcher");put(f.queue,q)
            h=c.sha(f.queue)
            for name in ("driver_start.json","launch_receipt.json","completion.json"):
                p=f.run/name;v=json.loads(p.read_text());v["queue_sha256"]=h;put(p,v)
            with self.assertRaisesRegex(ValueError,"exact_operator_path_roles"):c.driver_receipts(c.Pins(),f.queue,f.run)
    def test_new_pure_reader_sources_must_be_in_admitted_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);f.run=install_resource_operator(f.queue,f.run,"validation_resource_ready")
            q=json.loads(Path(f.queue).read_text());q["source_sha256"].pop(str(Path(__file__).with_name("analyze_cap_mechanism.py").resolve()));put(f.queue,q)
            h=c.sha(f.queue)
            for name in ("driver_start.json","launch_receipt.json","completion.json"):
                p=f.run/name;v=json.loads(p.read_text());v["queue_sha256"]=h
                if name!="completion.json":v["source_sha256"]=q["source_sha256"]
                put(p,v)
            with self.assertRaisesRegex(ValueError,"resource_analysis_sources_pinned"):c.driver_receipts(c.Pins(),f.queue,f.run)
    def test_receipt_operator_role_cannot_differ_from_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);f.run=install_resource_operator(f.queue,f.run,"validation_resource_ready")
            p=f.run/"completion.json";v=json.loads(p.read_text());v["operator_paths"]["supervisor"]["sha256"]="a"*64;put(p,v)
            with self.assertRaisesRegex(ValueError,"resource_operator_receipt_refs"):c.driver_receipts(c.Pins(),f.queue,f.run)
    def test_old_legacy_envelope_without_new_paths_still_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);q,e=c.driver_receipts(c.Pins(),f.queue,f.run)
            self.assertEqual(len(e["completion"]["jobs"]),16);self.assertNotIn("operator_paths",q)
if __name__=="__main__":unittest.main()

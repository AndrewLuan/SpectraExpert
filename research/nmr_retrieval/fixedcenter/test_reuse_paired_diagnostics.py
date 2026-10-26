"""用途：以合成配对产物验证复用工具的权重、贡献恒等式与证据拒绝；不访问真实数据。"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import reuse_paired_diagnostics as m

def row(i=0,variant="observed",left=1,right=2,common=400,change=3.0):
    category={(True,True):"common",(False,True):"new",(True,False):"lost",(False,False):"neither"}[(left<=512,right<=512)]
    return dict(query_key=[str(i),variant,"synthetic_source",str(i),"synthetic_group"],
        source="synthetic_source",variant=variant,group="synthetic_group",category=category,
        left_rank=left,right_rank=right,delta_MRR=1/right-1/left,
        candidate_overlap_available=True,common_candidates=common,
        added_candidates=512-common,removed_candidates=512-common,
        common_candidate_mean_abs_rank_change=change)

def independent_summary(rows):
    n=len(rows)
    total=lambda group:sum(1/r["right_rank"]-1/r["left_rank"] for r in group)
    return dict(queries=n,left_MRR=sum(1/r["left_rank"] for r in rows)/n,
        right_MRR=sum(1/r["right_rank"] for r in rows)/n,delta_MRR=total(rows)/n,
        delta_recall512=sum(int(r["right_rank"]<=512)-int(r["left_rank"]<=512) for r in rows)/n,
        **{"delta_top"+str(k):sum(int(r["right_rank"]<=k)-int(r["left_rank"]<=k) for r in rows)/n for k in (1,5,10)},
        categories={c:dict(queries=len(cell),contribution_to_total_delta_MRR=total(cell)/n,
            conditional_delta_MRR=total(cell)/len(cell) if cell else None)
            for c in m.CATEGORIES for cell in [[r for r in rows if r["category"]==c]]})

def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v)+"\n")

SMALL_COUNTS={cohort:{v:n for v in m.VARIANTS} for cohort,n in (("original_val",4),("tune",8))}

def fixture(root,shift_seed211_identity=False):
    producer_bytes=(Path(__file__).parent/"pool_diagnostics.py").read_bytes()
    (root/"pool_diagnostics.py").write_bytes(producer_bytes)
    require_sha=hashlib.sha256(producer_bytes).hexdigest()
    assert require_sha==m.PRODUCER_SHA
    decompose=m.load_decompose(producer_bytes,root/"pool_diagnostics.py")
    jobs={}
    for seed in m.SEEDS:
        for cohort in m.COHORTS:
            out=root/f"diagnostics/paired/s{seed}"/cohort
            args=["python",str(root/"pool_diagnostics.py")]
            inputs={str(root/"pool_diagnostics.py"):require_sha}
            predictions={}
            for pool in ("single","dual"):
                e=root/f"diagnostics/frozen/s{seed}"/pool/cohort;e.mkdir(parents=True)
                dump(e/"summary.json",{"cohort":cohort,"query_sha256":"fixed_synthetic_query",
                    "candidate_order_sha256":"fixed_synthetic_order","checkpoints":{"forward":"fixed_by_seed_"+str(seed)}})
                cells=[]
                for variant in m.VARIANTS:
                    for i in range(SMALL_COUNTS[cohort][variant]):
                        left,right=((1,2),(1000,2),(2,1000),(1000,2000))[i%4]
                        common=400+i%4
                        candidates=list(range(512)) if pool=="single" else list(range(common))+list(range(512,1024-common))
                        costs=[float(k) if pool=="single" else -float(k) for k in candidates]
                        rank=left if pool=="single" else right
                        record_id=cohort+"_"+str(i)
                        if seed==211 and shift_seed211_identity:record_id+="_changed"
                        cells.append(dict(query_order=i,variant=variant,source="synthetic_source",
                            record_id=record_id,group="synthetic_group",smiles="CC",mask_content_sha256="synthetic_mask",
                            rank=rank,evaluation_pool_size=512,target_in_coarse_pool=rank<=512,
                            natural_candidate_indices=candidates,natural_candidate_costs=costs))
                predictions[pool]=cells
                (e/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in cells))
                (e/"EXIT_CODE").write_text("0\n")
                for name in ("summary.json","predictions.jsonl","EXIT_CODE"):
                    inputs[str(e/name)]=hashlib.sha256((e/name).read_bytes()).hexdigest()
                er=dict(state="complete",internal_exit=0,external_exit=0,output=str(e))
                job=f"eval_frozen_s{seed}_{pool}_{cohort}";jobs[job]=er
                dump(root/"runs/diagnostics"/(job+".receipt.json"),er)
                args+=["--"+pool,str(e)]
            args+=["--out",str(out)]
            receipt=dict(state="complete",internal_exit=0,external_exit=0,output=str(out),argv=args)
            job=f"decompose_s{seed}_{cohort}";jobs[job]=receipt
            dump(root/"runs/diagnostics"/(job+".receipt.json"),receipt)
            rows,_=decompose(predictions["single"],predictions["dual"])
            summary=dict(scope="train_val_only",cohort=cohort,inputs_unchanged=True,inputs_sha256=inputs,
                by_variant={v:independent_summary([r for r in rows if r["variant"]==v]) for v in m.VARIANTS},
                by_source={"synthetic_source":independent_summary([r for r in rows if r["variant"]=="observed"])})
            dump(out/"summary.json",summary)
            (out/"paired.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            (out/"EXIT_CODE").write_text("0\n")
    dump(root/"runs/diagnostics/completion.json",{"jobs":jobs})
    for name in ("EXIT_CODE","EXTERNAL_EXIT_CODE"):(root/"runs/diagnostics"/name).write_text("0\n")
    return require_sha

def mutate_paired(root,field,change):
    p=root/"diagnostics/paired/s193/original_val/paired.jsonl"
    rows=[json.loads(line) for line in p.read_text().splitlines()]
    rows[0][field]=change(rows[0][field])
    p.write_text("".join(json.dumps(r)+"\n" for r in rows))

class Arithmetic(unittest.TestCase):
    def test_query_weighted_cohort_merge(self):
        merged=m.stats([row(0,change=0)]+[row(i+1,change=4) for i in range(3)])
        self.assertEqual(merged["optional_pool"]["common_candidate_mean_abs_rank_change"],3.0)
        self.assertNotEqual(merged["optional_pool"]["common_candidate_mean_abs_rank_change"],2.0)
    def test_equal_seed_mean_after_merge(self):
        a=m.stats([row(0,left=1,right=2,change=1)])
        b=m.stats([row(0,left=2,right=1,change=5)])
        x=m.equal_seed_mean([a,b])
        self.assertEqual(x["delta_MRR"],0)
        self.assertEqual(x["optional_pool"]["common_candidate_mean_abs_rank_change"],3)
    def test_category_contribution_uses_total_denominator(self):
        x=m.stats([row(0,left=1,right=2),row(1,left=1000,right=2),row(2,left=2,right=1000),row(3,left=1000,right=2000)])
        self.assertAlmostEqual(x["categories"]["new"]["contribution_to_total_delta_MRR"],(.5-.001)/4)
        self.assertAlmostEqual(sum(c["contribution_to_total_delta_MRR"] for c in x["categories"].values()),x["delta_MRR"])
        self.assertEqual(sum(c["queries"] for c in x["categories"].values()),4)
    def test_empty_category_has_null_conditional(self):
        self.assertIsNone(m.stats([row()])["categories"]["new"]["conditional_delta_MRR"])
    def test_strict_integer_exits(self):
        for field in ("internal_exit","external_exit"):
            x=dict(state="complete",internal_exit=0,external_exit=0);x[field]=False
            with self.assertRaisesRegex(ValueError,"receipt_exit"):m.verify_receipt(x)
    def test_missing_commonrank_fails(self):
        x=row();del x["common_candidate_mean_abs_rank_change"]
        with self.assertRaises(KeyError):m.stats([x])
    def test_nan_commonrank_rejected(self):
        rows=[row(i,change=float("nan")) for i in range(320)]
        with self.assertRaisesRegex(ValueError,"common_rank_metric"):m.validate_rows(rows,"original_val")
    def test_intersection_candidate_rank_bounds(self):
        rows=[row(i,common=0,change=1) for i in range(320)]
        with self.assertRaisesRegex(ValueError,"common_rank_metric"):m.validate_rows(rows,"original_val")
    def test_wrong_membership_category_rejected(self):
        rows=[row(i) for i in range(320)];rows[0]["category"]="new"
        with self.assertRaisesRegex(ValueError,"category"):m.validate_rows(rows,"original_val")
    def test_no_feature_or_test_read(self):
        for path in ("/tmp/test/predictions.jsonl","/tmp/val/features.pt"):
            with self.assertRaisesRegex(ValueError,"forbidden_input"):m.Reader().read(path)

class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.counts=patch.object(m,"COUNTS",SMALL_COUNTS)
        self.counts.start()
        self.addCleanup(self.counts.stop)
    def test_complete_four_producer_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();digest=fixture(root)
            with patch.object(m,"PRODUCER_SHA",digest):
                x=m.run(root)
            self.assertTrue(x["inputs_unchanged"])
            self.assertTrue(x["paired_derivation_verified"])
            self.assertTrue(x["cross_seed_query_identity_verified"])
            self.assertEqual(len(x["by_seed_cohort_variant"]),12)
            self.assertEqual(len(x["by_seed_combined_variant"]),6)
            self.assertEqual(x["two_seed_query_weighted"]["observed"]["queries_per_seed"],12)
            self.assertEqual(x["two_seed_query_weighted"]["peak_dropout"]["queries_per_seed"],12)
            for v,z in x["two_seed_query_weighted"].items():
                self.assertAlmostEqual(z["optional_pool"]["common_candidates_mean"],401.5)
                self.assertAlmostEqual(z["optional_pool"]["common_candidate_mean_abs_rank_change"],sum(sum(abs(2*i-(c-1)) for i in range(c))/c for c in range(400,404))/4)
                self.assertEqual(sum(c["queries_mean_per_seed"] for c in z["categories"].values()),z["queries_per_seed"])
            self.assertEqual(x["input_sha256_before"],x["input_sha256_after"])
    def test_changed_producer_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();digest=fixture(root)
            (root/"diagnostics/frozen/s193/single/original_val/predictions.jsonl").write_text("changed\n")
            with patch.object(m,"PRODUCER_SHA",digest), self.assertRaisesRegex(ValueError,"producer_input_sha"):
                m.run(root)
    def test_receipt_completion_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();digest=fixture(root)
            p=root/"runs/diagnostics/decompose_s193_original_val.receipt.json"
            v=json.loads(p.read_text());v["unbound"]="new";dump(p,v)
            with patch.object(m,"PRODUCER_SHA",digest),self.assertRaisesRegex(ValueError,"completion_receipt_binding"):
                m.run(root)

    def test_optional_commonrank_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root)
            mutate_paired(root,"common_candidate_mean_abs_rank_change",lambda x:x+0.1)
            with self.assertRaisesRegex(ValueError,"derived_pair_content"):
                m.run(root)
    def test_optional_intersection_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root)
            p=root/"diagnostics/paired/s193/original_val/paired.jsonl"
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            rows[0]["common_candidates"]-=1
            rows[0]["added_candidates"]+=1
            rows[0]["removed_candidates"]+=1
            p.write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError,"derived_pair_content"):
                m.run(root)
    def test_query_identity_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root)
            mutate_paired(root,"query_key",lambda x:x[:3]+["tampered_record_id"]+x[4:])
            with self.assertRaisesRegex(ValueError,"derived_pair_identity"):
                m.run(root)
    def test_cross_seed_identity_exact_set_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root,shift_seed211_identity=True)
            with self.assertRaisesRegex(ValueError,"cross_seed_query_identity"):
                m.run(root)
    def test_extra_unbound_field_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root)
            p=root/"diagnostics/paired/s193/original_val/paired.jsonl"
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            rows[0]["unbound_extra"]="new"
            p.write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError,"derived_pair_content"):
                m.run(root)
    def test_changed_producer_source_rejected_before_exec(self):
        with self.assertRaisesRegex(ValueError,"producer_source"):
            m.load_decompose(b"raise RuntimeError('must not execute')",Path("fake.py"))

if __name__=="__main__":unittest.main()

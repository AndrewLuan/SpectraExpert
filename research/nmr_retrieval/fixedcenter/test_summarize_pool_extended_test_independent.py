"""功能概述：独立检查扩展sealed汇总与实际producer最小协议及自然池指标的兼容。
复用合成builder输出，只读取临时合成预测；不运行模型或真实sealed任务。
重点是old producer16输入、new adapter真实preflight闭包、rank/recall语义和空池内指标。
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import summarize_pool_extended_test as m
import evaluate_pool_fixed_center_sealed as adapter
from test_summarize_pool_extended_test import Fixture,synthetic_rows,write


class IndependentSummaryTests(unittest.TestCase):
    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        return Fixture(temp.name)

    def replace_protocol_inputs(self,job,inputs):
        out=Path(job["out"])
        protocol=json.loads((out/"protocol.json").read_text());protocol["inputs_sha256"]=inputs
        summary=json.loads((out/"summary.json").read_text());summary["input_sha256"]=inputs
        write(out/"protocol.json",protocol);write(out/"summary.json",summary)

    def test_actual_old_producer_minimum_input_map_passes(self):
        f=self.fixture();job=next(j for j in f.queue["jobs"] if j["id"].startswith("sealed_original_"))
        f.materialize(job)
        self.replace_protocol_inputs(job,dict(job["inputs_sha256"]))
        out=Path(job["out"]);role=(out.parent.parent.name,int(out.parent.name[1:]))
        with patch.object(m,"__file__",str(f.tool)),patch.object(m.validation,"CANDIDATE_SHA",f.builder.f.candidate):
            config,seed,rows,cost=m.read_test(job,f.frozen,f.review,f.joint,f.entries[role],{}, {},f.frozen_path)
        self.assertEqual((config,seed),role);self.assertEqual(len(rows),1371)

    def test_actual_adapter_preflight_minimum_input_map_passes(self):
        f=self.fixture();job=f.fixed_job();args,_=f.materialize(job)
        sources=[f.builder.adapter,f.builder.fixed]+[f.builder.f.source/"nmr_transfer"/n for n in m.SOURCE_NAMES if n!="h_research_driver.py"]
        with patch.object(adapter,"source_paths",return_value=sources):
            _,_,inputs=adapter.preflight(SimpleNamespace(**args))
        producer=json.loads((Path(args["cache"]).parent/"summary.json").read_text())
        inputs.update(producer["source_inputs_sha256"])
        self.replace_protocol_inputs(job,inputs)
        config,seed,rows,cost=f.read_fixed()
        self.assertEqual((config,seed),(m.ADDED[0],193));self.assertEqual(len(rows),1371)

    def test_rank_membership_conflict_rejected(self):
        f=self.fixture();job=f.fixed_job();f.materialize(job)
        path=Path(job["out"])/"predictions.jsonl"
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        rows[0]["target_in_coarse_pool"]=False
        write(path,"".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError,"natural_rank_membership"):
            f.read_fixed()

    def test_job_finishes_after_driver_rejected(self):
        f=self.fixture();job=f.queue["jobs"][0]
        path=f.run/(job["id"]+".receipt.json");record=json.loads(path.read_text())
        record["finished_utc"]="2026-10-23T12:03:01Z";write(path,record)
        path=f.run/"completion.json";value=json.loads(path.read_text());value["jobs"][job["id"]]=record;write(path,value)
        with self.assertRaisesRegex(ValueError,"job_time_order"):
            f.verify()

    def test_equal_seed_mean_preserves_outside_RR_and_null_in_pool(self):
        left=synthetic_rows(rank=1);right=synthetic_rows(rank=1000)
        for row in right:row["target_in_coarse_pool"]=False
        data={c:{193:left,211:right} for c in m.CONFIGS}
        result=m.aggregate(data)
        row=next(r for r in result if r["config"]==m.CONFIGS[0] and r["seed"]=="two_seed_mean"
                 and r["variant"]=="observed" and r["source"]=="all")
        self.assertAlmostEqual(row["MRR"],.5005)
        self.assertEqual(row["recall512"],.5);self.assertEqual(row["top1"],.5)
        self.assertIsNone(row["in_pool_MRR"]);self.assertEqual(row["queries"],1000)


if __name__=="__main__":unittest.main()

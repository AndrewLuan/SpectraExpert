"""用途：以真实扩展builder合成输出和24×1371生产器形状产物验证封存汇总。
不执行模型，不访问真实test；覆盖全矩阵、具体冻结review、新adapter/中心闭环和无重选。
"""
import copy,csv,hashlib,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import summarize_pool_extended_test as m
import evaluate_pool_fixed_center_sealed as adapter
from test_build_pool_extended_frozen import Fixture as BuildFixture

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(value if isinstance(value,str) else json.dumps(value)+"\n")

def synthetic_rows(rank=2):
    out=[];ids=list(range(512));costs=[0]*512
    for i in range(1000):
        for variant in sorted(m.VARIANTS):
            if i>=m.VARIANTS[variant]:continue
            out.append(dict(query_order=i,variant=variant,source=sorted(m.validation.SOURCES)[i%5],record_id="synthetic_"+str(i),
                smiles="C",group="synthetic_group_"+str(i),mask_content_sha256=hashlib.sha256((str(i)+variant).encode()).hexdigest(),
                rank=rank,target_in_coarse_pool=True,evaluation_pool_size=512,natural_candidate_indices=ids,natural_candidate_costs=costs))
    return out

class Fixture:
    def __init__(self,tmp,materialize_all=False):
        self.base=Path(tmp).resolve();self.builder=BuildFixture(str(self.base))
        b=self.builder;self.rows=synthetic_rows();fields=("query_order","variant","source","record_id","smiles","group")
        self.identity=dict(query_sha256="a"*64,
            identity_content_sha256=m.validation.content_sha([{k:r[k] for k in fields} for r in self.rows]),
            variant_mask_content_sha256=m.validation.content_sha([(tuple(r[k] for k in fields[:-1]),r["mask_content_sha256"]) for r in self.rows]))
        for cache in {e["cache"] for e in b.plan["evaluations"]}:
            p=Path(cache).parent/"summary.json";v=json.loads(p.read_text())
            v.update(self.identity,variants=m.VARIANTS,records=1371,base_queries=1000,appended_training_positives=0,matching_seconds_mean=.01)
            write(p,v)
        b.plan["fixed_test_identity"]=self.identity
        b.f.plan["fixed_test_identity"]=self.identity
        # Freeze实际执行中的summary工具字节，而非只给同名占位文件。
        self.tool=b.f.test_summarizer;self.tool.write_bytes(Path(m.__file__).read_bytes())
        b.plan["tool_sha256"]=b.tools();write(b.plan_path,b.plan);b.approve()
        self.receipt,self.frozen,self.queue=b.build()
        self.frozen_path=b.f.base/"freeze_extended/frozen_selection.json"
        self.queue_path=b.f.base/"freeze_extended/sealed_queue.json"
        write(self.frozen_path.parent/"EXIT_CODE","0\n")
        self.run=b.root/"runs/extended_sealed";self.out=self.base/"summary_output"
        sources=self.queue["source_inputs_sha256"];bound={**sources,str(self.queue_path):m.validation.sha(self.queue_path)}
        receipts={}
        for j in self.queue["jobs"]:
            record=dict(state="complete",internal_exit=0,external_exit=0,argv=j["argv"],output=j["out"],cwd=j["cwd"],
                queue_source_sha256=bound,requested_inputs_sha256=j["inputs_sha256"],pid=100,
                started_utc="2026-10-23T12:01:00Z",finished_utc="2026-10-23T12:02:00Z")
            receipts[j["id"]]=record;write(self.run/(j["id"]+".receipt.json"),record)
            write(Path(j["out"])/"EXIT_CODE","0\n")
        self.receipts=receipts
        argv=["python","-m","nmr_transfer.h_research_driver","--queue",str(self.queue_path),"--out",str(self.run)]
        write(self.run/"launch_receipt.json",dict(internal_exit=0,external_exit=0,queue_sha256={str(self.queue_path):m.validation.sha(self.queue_path)},
            bundle_sha256=sources,source_sha256={next(iter(sources)):next(iter(sources.values()))},child_pid=99,argv=argv,
            started_utc="2026-10-23T12:00:00Z",finished_utc="2026-10-23T12:04:00Z"))
        write(self.run/"driver_start.json",dict(inputs_sha256=bound,pid=99,argv=["driver.py"]+argv[3:],utc="2026-10-23T12:00:30Z"))
        write(self.run/"completion.json",dict(success=True,inputs_unchanged=True,jobs=receipts,utc="2026-10-23T12:03:00Z"))
        for name in ("EXIT_CODE","EXTERNAL_EXIT_CODE"):write(self.run/name,"0\n")
        self.entries={(e["config"],e["seed"]):e for e in b.plan["evaluations"]}
        self.joint=json.loads(Path(self.frozen["validation_summary_path"]).read_text())
        self.review=json.loads(Path(self.frozen["freeze_review"]["path"]).read_text())
        if materialize_all:
            for j in self.queue["jobs"]:self.materialize(j)
    def materialize(self,job):
        folder=Path(job["out"]);config=folder.parent.parent.name;seed=int(folder.parent.name[1:])
        argv=job["argv"];tail=argv[2:] if config in m.ADDED else argv[3:]
        args={k[2:].replace("-","_"):value for k,value in zip(tail[::2],tail[1::2])};args["seed"]=int(args["seed"])
        if config in m.ADDED:args["device"]="cuda"
        inputs={**self.frozen["evaluation_inputs_sha256"],str(self.frozen_path):m.validation.sha(self.frozen_path)}
        normal=[r for r in self.rows if r["variant"]=="observed"]
        metrics=dict(overall=m.validation.metrics(normal),by_source={s:m.validation.metrics([r for r in normal if r["source"]==s]) for s in sorted(m.validation.SOURCES)},
            stress={v:m.validation.metrics([r for r in self.rows if r["variant"]==v]) for v in list(m.VARIANTS)[1:]})
        summary=dict(metrics=metrics,cohort="test",recall=config.rsplit("_",1)[1],query_sha256=self.identity["query_sha256"],
            candidate_order_sha256=self.builder.f.candidate,input_sha256=inputs,
            checkpoints={args[k]:inputs[args[k]] for k in ("no_aug_checkpoint","aug_checkpoint")},
            reranking_seconds_mean=.02,reranking_seconds_p95=.04,elapsed_seconds=2.5,external_api_calls=0)
        if config in m.ADDED:
            summary.update(config=config,seed=seed,center_estimate_sha256=self.frozen["fixed_center_configurations"][config][str(seed)]["center_estimate_sha256"])
        write(folder/"protocol.json",dict(args=args,inputs_sha256=inputs));write(folder/"summary.json",summary)
        write(folder/"predictions.jsonl","".join(json.dumps(r)+"\n" for r in self.rows))
        return args,summary
    def fixed_job(self):
        return next(j for j in self.queue["jobs"] if j["id"]=="sealed_fixedcenter_mixed_listwise_single_s193")
    def verify(self):
        with patch.object(m,"__file__",str(self.tool)),patch.object(m.validation,"CANDIDATE_SHA",self.builder.f.candidate):
            return m.verify_run(self.frozen_path,self.queue_path,self.run,{}, {})
    def read_fixed(self):
        j=self.fixed_job()
        with patch.object(m,"__file__",str(self.tool)),patch.object(m.validation,"CANDIDATE_SHA",self.builder.f.candidate):
            return m.read_test(j,self.frozen,self.review,self.joint,self.entries[("fixedcenter_mixed_listwise_single",193)],{}, {},self.frozen_path)
    def run_summary(self):
        self.out.mkdir()
        args=types.SimpleNamespace(frozen=str(self.frozen_path),queue=str(self.queue_path),run=str(self.run))
        with patch.object(m,"__file__",str(self.tool)),patch.object(m.validation,"CANDIDATE_SHA",self.builder.f.candidate),patch.object(m.validation,"choose",side_effect=AssertionError("test_selection_forbidden")):
            return m.run(args,self.out)

class ExtendedSealed(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
    def test_actual_builder24_output_accepts_full_summary(self):
        f=Fixture(self.temp.name,materialize_all=True);x=f.run_summary()
        self.assertEqual(x["jobs"],24);self.assertEqual(x["metric_rows"],648)
        self.assertFalse(x["selection_performed"]);self.assertFalse(x["promotion"])
        self.assertEqual(x["independent_acceptance_status"],"unavailable")
        self.assertEqual(x["preselected"],f.frozen["preselected"])
        self.assertEqual(x["preselected"],x["frozen_joint_preselected"])
        self.assertEqual(set(x["raw_cost_records"]),set(m.CONFIGS))
        for seeds in x["raw_cost_records"].values():
            self.assertEqual(set(seeds),{193,211})
            for raw in seeds.values():
                self.assertEqual(raw["matching_seconds_mean"],.01);self.assertEqual(raw["reranking_seconds_mean"],.02)
                self.assertEqual(raw["reranking_seconds_p95"],.04);self.assertEqual(raw["evaluation_elapsed_seconds"],2.5)
        with (f.out/"metrics.tsv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream,delimiter="\t"))),648)
    def test_all4_actual_adapter_preflights_accept_same_builder_contract(self):
        f=Fixture(self.temp.name)
        sources=[f.builder.adapter,f.builder.fixed]+[f.builder.f.source/"nmr_transfer"/n for n in m.SOURCE_NAMES if n!="h_research_driver.py"]
        for job in f.queue["jobs"]:
            if not any(job["id"]==f"sealed_{c}_s{s}" for c in m.ADDED for s in m.validation.SEEDS):continue
            args,summary=f.materialize(job)
            with patch.object(adapter,"source_paths",return_value=sources):
                frozen,selected,pins=adapter.preflight(types.SimpleNamespace(**args))
            self.assertEqual(selected["seed"],args["seed"])
            self.assertEqual(summary["center_estimate_sha256"],selected["center_estimate_sha256"])
            self.assertIn(args["cache"],pins)
    def test_two_seed_arithmetic_is_equal_seed_mean(self):
        rows=synthetic_rows(rank=1);other=synthetic_rows(rank=4)
        data={c:{193:rows,211:other} for c in m.CONFIGS}
        table=m.aggregate(data)
        observed=[x for x in table if x["config"]==m.CONFIGS[0] and x["variant"]=="observed" and x["source"]=="all"]
        self.assertEqual([x["MRR"] for x in observed],[1,.25,.625])
        self.assertEqual([x["queries"] for x in observed],[1000,1000,1000])
    def test_bool_internal_job_exit_rejected(self):
        f=Fixture(self.temp.name);job=f.queue["jobs"][0]
        path=f.run/(job["id"]+".receipt.json");record=json.loads(path.read_text());record["internal_exit"]=False;write(path,record)
        path=f.run/"completion.json";completion=json.loads(path.read_text());completion["jobs"][job["id"]]=record;write(path,completion)
        with self.assertRaisesRegex(ValueError,"job_actual_double"):f.verify()
    def test_actual_builder24_driver_only(self):
        f=Fixture(self.temp.name);self.assertEqual(len(f.verify()[-1]),24)
    def test_bool_outer_exit_rejected(self):
        f=Fixture(self.temp.name);p=f.run/"launch_receipt.json";v=json.loads(p.read_text());v["internal_exit"]=False;write(p,v)
        with self.assertRaisesRegex(ValueError,"actual_outer_double"):f.verify()
    def test_one_missing_job_receipt_rejected(self):
        f=Fixture(self.temp.name);(f.run/(f.queue["jobs"][0]["id"]+".receipt.json")).unlink()
        with self.assertRaises(FileNotFoundError):f.verify()
    def test_missing_new_center_source_input_rejected(self):
        f=Fixture(self.temp.name);j=f.fixed_job();f.materialize(j)
        p=Path(j["out"])/"protocol.json";v=json.loads(p.read_text())
        center=f.frozen["fixed_center_configurations"][m.ADDED[0]]["193"]["center_artifacts"][0]["centers"]
        del v["inputs_sha256"][center];write(p,v)
        summary=Path(j["out"])/"summary.json";s=json.loads(summary.read_text());s["input_sha256"]=v["inputs_sha256"];write(summary,s)
        with self.assertRaisesRegex(ValueError,"requested_inputs|center_protocol"):f.read_fixed()
    def test_adapter_source_missing_even_when_job_pruned_rejected(self):
        f=Fixture(self.temp.name);j=f.fixed_job();f.materialize(j)
        p=Path(j["out"])/"protocol.json";v=json.loads(p.read_text());del v["inputs_sha256"][str(f.builder.adapter)]
        j["inputs_sha256"].pop(str(f.builder.adapter),None);write(p,v)
        s=Path(j["out"])/"summary.json";z=json.loads(s.read_text());z["input_sha256"]=v["inputs_sha256"];write(s,z)
        with self.assertRaisesRegex(ValueError,"source_or_evidence_closure"):f.read_fixed()
    def test_wrong_center_role_SHA_rejected(self):
        f=Fixture(self.temp.name);j=f.fixed_job();f.materialize(j)
        p=Path(j["out"])/"summary.json";s=json.loads(p.read_text());s["center_estimate_sha256"].reverse();write(p,s)
        with self.assertRaisesRegex(ValueError,"summary_centers"):f.read_fixed()
    def test_wrong_new_config_rejected(self):
        f=Fixture(self.temp.name);j=f.fixed_job();f.materialize(j)
        p=Path(j["out"])/"summary.json";s=json.loads(p.read_text());s["config"]=m.ADDED[1];write(p,s)
        with self.assertRaisesRegex(ValueError,"named_role"):f.read_fixed()
    def test_fixed_identity_tamper_rejected(self):
        f=Fixture(self.temp.name);j=f.fixed_job();f.materialize(j)
        p=Path(j["out"])/"predictions.jsonl";r=[json.loads(x) for x in p.read_text().splitlines()];r[0]["record_id"]="tampered";write(p,"".join(json.dumps(x)+"\n" for x in r))
        with self.assertRaisesRegex(ValueError,"identity_or_variant"):f.read_fixed()
    def test_missing_stress_variant_rejected(self):
        f=Fixture(self.temp.name);j=f.fixed_job();f.materialize(j)
        p=Path(j["out"])/"predictions.jsonl";r=[json.loads(x) for x in p.read_text().splitlines() if json.loads(x)["variant"]!="peak_dropout"];write(p,"".join(json.dumps(x)+"\n" for x in r))
        with self.assertRaisesRegex(ValueError,"identity_or_variant"):f.read_fixed()
    def test_negative_raw_time_rejected(self):
        f=Fixture(self.temp.name);j=f.fixed_job();f.materialize(j)
        p=Path(j["out"])/"summary.json";s=json.loads(p.read_text());s["reranking_seconds_mean"]=-1;write(p,s)
        with self.assertRaisesRegex(ValueError,"raw_cost"):f.read_fixed()
    def test_protocol_wrong_frozen_path_rejected(self):
        f=Fixture(self.temp.name);j=f.fixed_job();f.materialize(j)
        p=Path(j["out"])/"protocol.json";v=json.loads(p.read_text());v["args"]["frozen_selection"]="/other/frozen.json";write(p,v)
        with self.assertRaisesRegex(ValueError,"actual_argv|named_role"):f.read_fixed()

if __name__=="__main__":unittest.main()

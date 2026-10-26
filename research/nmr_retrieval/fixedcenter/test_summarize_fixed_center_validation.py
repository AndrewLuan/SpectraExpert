"""用途：合成验证追加家族的完整驱动闭环、已审audit依赖、旧helper适配及原guard/成本并列。
不读取真实谱值，不运行模型或科学选择；完整fixture使用实际320/1024查询与自然512。
"""
import collections,copy,hashlib,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import summarize_fixed_center_validation as m
v=m.v
def dump(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value)+"\n")
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value if isinstance(value,bytes) else value.encode())
def rows(cohort,rank):
    n=320 if cohort=="original_val" else 1024
    return [dict(query_order=i,variant=variant,source=sorted(v.SOURCES)[i%5],record_id=cohort+str(i),smiles="C"+str(i),
        group=cohort+"g"+str(i),rank=rank,target_in_coarse_pool=True,evaluation_pool_size=512,
        natural_candidate_indices=list(range(512)),natural_candidate_costs=[0]*512,mask_content_sha256=v.content_sha([cohort,i,variant]))
        for i in range(n) for variant in sorted(v.VARIANTS)]
def metrics(data):
    normal=[r for r in data if r["variant"]=="observed"]
    return dict(overall=v.metrics(normal),by_source={s:v.metrics([r for r in normal if r["source"]==s]) for s in sorted(v.SOURCES)},
        stress={x:v.metrics([r for r in data if r["variant"]==x]) for x in v.VARIANTS[1:]})
def default_args(expected):
    return dict(expected,training_pool="mixed",loss="listwise",epochs=64,samples=8192,batch_size=32,anchor=.1,lr=.001,augment=False,device="cuda")
def command(root,expected):
    return ["python",str(root/"h_pool_fixed_center_ranker.py")]+sum((["--"+k.replace("_","-"),str(value)] for k,value in expected.items()),[])
def simple_main():
    baseline=dict(MRR=.5,top1=.4,frequency_crop_MRR=.2,peak_dropout_MRR=.2,macro_MRR=.3,cost_seconds=.03)
    configs={c:dict(signal=dict(baseline),gates=dict(normal_MRR=False,top1=True,crop=True,dropout=True,source_macro=True),eligible=False,delta={k:0 for k in baseline}) for c in v.CONFIGS}
    return dict(selection=dict(preselected=None,eligible_configurations=[],baseline_signal=baseline,configurations=configs))
def improved():
    return {c:dict(MRR=.6,top1=.5,frequency_crop_MRR=.3,peak_dropout_MRR=.3,macro_MRR=.4,cost_seconds=.04) for c in m.CONFIGS}

class Selection(unittest.TestCase):
    def test_both_new_configs_and_main8_preserved(self):
        original=simple_main();frozen=copy.deepcopy(original);x=m.extend_selection(original,improved())
        self.assertEqual(x["preselected"],sorted(m.CONFIGS)[0])
        self.assertEqual({c:x["configurations"][c] for c in v.CONFIGS},original["selection"]["configurations"])
        self.assertEqual(original,frozen);self.assertFalse(x["promotion"])
    def test_each_original_guard_rejects(self):
        for field in ("MRR","top1","frequency_crop_MRR","peak_dropout_MRR","macro_MRR"):
            with self.subTest(field=field):
                signals=improved();baseline=simple_main()["selection"]["baseline_signal"]
                for c in signals:signals[c][field]=baseline[field] if field=="MRR" else baseline[field]-.001
                x=m.extend_selection(simple_main(),signals)
                self.assertEqual(x["eligible_configurations"],[]);self.assertIsNone(x["preselected"])
    def test_within_tolerance_uses_cost(self):
        x=improved();x[m.CONFIGS[0]]["MRR"]+=5e-13;x[m.CONFIGS[1]]["cost_seconds"]=.01
        self.assertEqual(m.extend_selection(simple_main(),x)["preselected"],m.CONFIGS[1])
    def test_outside_tolerance_uses_MRR(self):
        x=improved();x[m.CONFIGS[0]]["MRR"]+=2e-12;x[m.CONFIGS[1]]["cost_seconds"]=.01
        self.assertEqual(m.extend_selection(simple_main(),x)["preselected"],m.CONFIGS[0])
    def test_negative_cost_rejected(self):
        x=improved();x[m.CONFIGS[0]]["cost_seconds"]=-1
        with self.assertRaisesRegex(ValueError,"finite"):m.extend_selection(simple_main(),x)
    def test_exact_config_matrix(self):
        x=improved();x.pop(m.CONFIGS[0])
        with self.assertRaisesRegex(ValueError,"registered"):m.extend_selection(simple_main(),x)
    def test_unknown_config_rejected(self):
        x=improved();x["unregistered"]=x[m.CONFIGS[0]]
        with self.assertRaisesRegex(ValueError,"registered"):m.extend_selection(simple_main(),x)
    def test_nonfinite_rejected(self):
        x=improved();x[m.CONFIGS[0]]["MRR"]=float("nan")
        with self.assertRaisesRegex(ValueError,"finite"):m.extend_selection(simple_main(),x)

class Binder(unittest.TestCase):
    def setUp(self):
        self.root=Path("/synthetic/outputs/nmr_pool_ranking_20261023")
        self.expected=m.expected_args(self.root,m.CONFIGS[0],193,"original_val")
        self.args=default_args(self.expected);self.job=dict(argv=command(self.root,self.expected))
    def test_actual_script_argv(self):m.bind_fixed_evaluation(self.args,self.job,self.expected)
    def test_protocol_and_argv_wrong_checkpoint_rejected(self):
        self.args["aug_checkpoint"]="/other.pt";self.job["argv"]=command(self.root,{**self.expected,"aug_checkpoint":"/other.pt"})
        with self.assertRaisesRegex(ValueError,"fixed_family"):m.bind_fixed_evaluation(self.args,self.job,self.expected)
    def test_wrong_seed(self):
        self.args["seed"]=211
        with self.assertRaisesRegex(ValueError,"fixed_family"):m.bind_fixed_evaluation(self.args,self.job,self.expected)
    def test_extra_flag(self):
        self.job["argv"]+=["--train","illegal"]
        with self.assertRaisesRegex(ValueError,"unregistered"):m.bind_fixed_evaluation(self.args,self.job,self.expected)
    def test_duplicate_flag(self):
        self.job["argv"]+=["--seed","193"]
        with self.assertRaisesRegex(ValueError,"duplicate"):m.bind_fixed_evaluation(self.args,self.job,self.expected)
    def test_device_protocol_bound(self):
        self.args["device"]="cpu"
        with self.assertRaisesRegex(ValueError,"device_protocol"):m.bind_fixed_evaluation(self.args,self.job,self.expected)
    def test_fixed_module_only(self):
        self.job["argv"][1]="/other.py"
        with self.assertRaisesRegex(ValueError,"fixed_module"):m.bind_fixed_evaluation(self.args,self.job,self.expected)

class Fixture:
    def __init__(self,tmp):
        self.root=Path(tmp).resolve()/"outputs/nmr_pool_ranking_20261023"
        self.root.mkdir(parents=True);self.run=self.root/"runs/additional";self.queue=self.root/"queue.json"
        self.audit=self.root/"audits/fixed_training";self.audit.mkdir(parents=True)
        self.review=self.root/"audit_review.json";self.out=self.root/"validation/joint"
        self.original={c:rows(c,2) for c in v.COHORTS};self.new={c:rows(c,1) for c in v.COHORTS}
        base=Path(__file__).parent
        sources={}
        for name in ("h_pool_fixed_center_ranker.py","ADDITIONAL_PLAN.json","ADDITIONAL_ADMISSION_GATE.json","audit_pool_fixed_center_training.py"):
            p=self.root/name;write(p,(base/name).read_bytes());sources[str(p)]=v.sha(p)
        jobs=[];self.models=[];evals=[];auditpins=dict(sources)
        for seed in v.SEEDS:
            for aug in (False,True):
                folder=self.root/f"additional/rankers/fixedcenter_mixed_listwise/s{seed}"/("aug" if aug else "no_aug")
                for name in ("initial.pt","best.pt","last.pt"):write(folder/name,"synthetic opaque weights "+str(seed)+name)
                dump(folder/"centers.json",{"constant_fp32":36,"center33":[int(aug)]*33,"O_center3":[seed]*3})
                write(folder/"center_rows.jsonl","synthetic center provenance\n");write(folder/"EXIT_CODE","0\n")
                for p in folder.iterdir():auditpins[str(p)]=v.sha(p)
                role=[seed,aug];jobid=f"fit_fixedcenter_mixed_listwise_s{seed}_"+("aug" if aug else "no_aug")
                self.models.append(dict(job_id=jobid,role=role,output=str(folder),draws=524288,optimizer_steps=16384,constant_fp32=36,
                    center_estimate_sha256=v.sha(folder/"centers.json"),center_rows_sha256=v.sha(folder/"center_rows.jsonl")))
                jobs.append(dict(id=jobid,kind="training",out=str(folder),argv=["python",str(self.root/"h_pool_fixed_center_ranker.py"),"--stage","fit","--seed",str(seed)],
                    cwd=str(self.root.parent.parent),inputs_sha256={},deps=[]))
            for pool in ("single","dual"):
                for cohort in v.COHORTS:
                    expected=m.expected_args(self.root,"fixedcenter_mixed_listwise_"+pool,seed,cohort)
                    folder=Path(expected["out"]);cache=Path(expected["cache"])
                    write(cache,"synthetic opaque trainval features")
                    data=self.new[cohort];fields=("query_order","variant","source","record_id","smiles","group")
                    cs=dict(cohort=cohort,recall=pool,candidate_count=114982,pool_size=512,query_sha256=v.content_sha(cohort),
                        candidate_order_sha256=v.CANDIDATE_SHA,feature_cache_sha256=v.sha(cache),matching_seconds_mean=.01,
                        records=len(data),variants=dict(collections.Counter(r["variant"] for r in data)),
                        identity_content_sha256=v.content_sha([{k:r[k] for k in fields} for r in data]),
                        variant_mask_content_sha256=v.content_sha([(tuple(r[k] for k in fields[:-1]),r["mask_content_sha256"]) for r in data]))
                    dump(cache.parent/"summary.json",cs)
                    inputs={str(cache):v.sha(cache),str(self.root/"h_pool_fixed_center_ranker.py"):sources[str(self.root/"h_pool_fixed_center_ranker.py")]}
                    checkpoints={}
                    for role in ("no_aug_checkpoint","aug_checkpoint"):
                        p=Path(expected[role]);inputs[str(p)]=v.sha(p);checkpoints[str(p)]=v.sha(p)
                        for name in ("centers.json","center_rows.jsonl"):inputs[str(p.parent/name)]=v.sha(p.parent/name)
                    protocol=dict(args=default_args(expected),inputs_sha256=inputs);dump(folder/"protocol.json",protocol)
                    result=dict(metrics=metrics(data),cohort=cohort,recall=pool,query_sha256=cs["query_sha256"],candidate_order_sha256=v.CANDIDATE_SHA,
                        input_sha256=inputs,checkpoints=checkpoints,reranking_seconds_mean=.02,
                        center_estimate_sha256=[x["center_estimate_sha256"] for x in self.models if x["role"][0]==seed])
                    dump(folder/"summary.json",result);write(folder/"EXIT_CODE","0\n")
                    write(folder/"predictions.jsonl","".join(json.dumps(r)+"\n" for r in data))
                    jid=f"eval_fixedcenter_mixed_listwise_s{seed}_{pool}_{cohort}"
                    jobs.append(dict(id=jid,kind="validation",out=str(folder),argv=command(self.root,expected),cwd=str(self.root.parent.parent),
                        inputs_sha256={str(cache):v.sha(cache),**checkpoints},deps=[x["job_id"] for x in self.models if x["role"][0]==seed]))
                    evals.append(dict(job_id=jid,role=[seed,pool,cohort],output=str(folder),checkpoints=checkpoints))
        dump(self.queue,dict(schema="nmr_research_jobs_20261022",source_inputs_sha256=sources,jobs=jobs))
        self.queue_sha=v.sha(self.queue);allpins={**sources,str(self.queue):self.queue_sha};auditpins[str(self.queue)]=self.queue_sha
        self.jobs=jobs;receipts={}
        for j in jobs:
            r=dict(state="complete",internal_exit=0,external_exit=0,argv=j["argv"],output=j["out"],cwd=j["cwd"],
                queue_source_sha256=allpins,requested_inputs_sha256=j["inputs_sha256"],
                started_utc="2026-10-23T11:01:00Z" if j["kind"]=="training" else "2026-10-23T11:03:00Z",
                finished_utc="2026-10-23T11:02:00Z" if j["kind"]=="training" else "2026-10-23T11:04:00Z")
            dump(self.run/(j["id"]+".receipt.json"),r);receipts[j["id"]]=r
        argv=["python","-m","nmr_transfer.h_research_driver","--queue",str(self.queue),"--out",str(self.run)]
        dump(self.run/"launch_receipt.json",dict(internal_exit=0,external_exit=0,queue_sha256={str(self.queue):self.queue_sha},
            bundle_sha256=sources,source_sha256=sources,child_pid=9,argv=argv,started_utc="2026-10-23T10:59:00Z",finished_utc="2026-10-23T11:06:00Z"))
        dump(self.run/"driver_start.json",dict(inputs_sha256=allpins,pid=9,argv=["driver.py"]+argv[3:],utc="2026-10-23T11:00:00Z"))
        dump(self.run/"completion.json",dict(success=True,inputs_unchanged=True,jobs=receipts,utc="2026-10-23T11:05:00Z"))
        for n in ("EXIT_CODE","EXTERNAL_EXIT_CODE"):write(self.run/n,"0\n")
        checks={k:True for k in ("same_initial_tensors","frozen_observation","frozen_centers","matched_exposure","finite","full_budget","real_driver_double_exit","complete_validation_metadata")}
        dump(self.audit/"summary.json",dict(status="PASS",scope="fixedcenter_training_no_test",training_jobs=4,validation_jobs=8,
            inputs_unchanged=True,queue_sha256=self.queue_sha,input_sha256=auditpins,models=self.models,evaluations=evals,checks=checks))
        write(self.audit/"EXIT_CODE","0\n")
        dump(self.review,dict(status="PASS",independent=True,scope="local_source_and_synthetic_training_audit_review_only",reviewer="synthetic independent",reviewed_sha256={str(self.root/"audit_pool_fixed_center_training.py"):sources[str(self.root/"audit_pool_fixed_center_training.py")]}))
        baseline_data={s:self.original for s in v.SEEDS};baseline_meta={s:{c:dict(cost_seconds=.03,candidate_order_sha256=v.CANDIDATE_SHA,query_sha256=v.content_sha(c),checkpoints={}) for c in v.COHORTS} for s in v.SEEDS}
        alldata={c:baseline_data for c in list(v.CONFIGS)+["frozen_single","frozen_dual","incumbent_frozen"]}
        allmeta={c:baseline_meta for c in alldata};table,signals=v.aggregate(alldata,allmeta)
        historical=signals.pop("incumbent_frozen")
        main=dict(schema=v.SCHEMA,selection=v.choose(signals),signals=signals,metadata={c:baseline_meta for c in signals},
            metric_rows=1188,historical_incumbent=dict(signal=historical,selection_eligible=False),input_sha256={},
            scientific_inputs_sha256={},source_sha256={str(Path(v.__file__).resolve()):m.HELPER_SHA},driver_evidence=dict(formal_jobs=48,diagnostic_jobs=12,all_dual_exit0=True),independent_acceptance_status="not_performed")
        for seed in v.SEEDS:
            for cohort in v.COHORTS:
                p=self.root/f"diagnostics/frozen/s{seed}/single"/cohort/"predictions.jsonl"
                write(p,"".join(json.dumps(r)+"\n" for r in self.original[cohort]));main["input_sha256"][str(p)]=v.sha(p)
        directory=self.root/"validation/main_matrix";dump(directory/"summary.json",main);write(directory/"EXIT_CODE","0\n")
        import io,csv
        t=io.StringIO();w=csv.DictWriter(t,fieldnames=m.FIELDS,delimiter="\t");w.writeheader();w.writerows(table);write(directory/"metrics.tsv",t.getvalue())
        self.main=json.loads(json.dumps(main));self.main_sha=v.sha(directory/"summary.json");self.tsv_sha=v.sha(directory/"metrics.tsv")
    def run_summary(self):
        self.out.mkdir(parents=True)
        args=types.SimpleNamespace(root=str(self.root),queue=str(self.queue),run=str(self.run),training_audit=str(self.audit),training_audit_review=str(self.review))
        with patch.object(m,"QUEUE_SHA",self.queue_sha),patch.object(m,"MAIN_SHA",self.main_sha),patch.object(m,"MAIN_TSV_SHA",self.tsv_sha):
            return m.run(args,self.out)

class CompletePipeline(unittest.TestCase):
    def test_complete4fit8eval3201024_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);binder=v.bind_evaluation;oldconfigs=v.CONFIGS
            x=f.run_summary()
            self.assertEqual(x["schema"],"pool_joint_validation_20261023")
            self.assertEqual(len(x["selection"]["configurations"]),10);self.assertEqual(len(x["signals"]),12)
            self.assertEqual(len(x["metadata"]),12);self.assertEqual(x["metric_rows"],1404)
            self.assertEqual(x["independent_acceptance_status"],f.main["independent_acceptance_status"])
            self.assertEqual({k:x["driver_evidence"][k] for k in f.main["driver_evidence"]},f.main["driver_evidence"])
            self.assertEqual({k:x["metadata"][k] for k in f.main["metadata"]},f.main["metadata"])
            self.assertEqual(x["main_matrix_selection"],f.main["selection"])
            self.assertEqual(x["historical_incumbent"],f.main["historical_incumbent"])
            self.assertEqual(x["main_validation_summary"]["sha256"],f.main_sha)
            self.assertEqual(x["additional_queue"]["sha256"],f.queue_sha)
            self.assertIs(v.bind_evaluation,binder);self.assertIs(v.CONFIGS,oldconfigs)
            self.assertIn(x["selection"]["preselected"],m.CONFIGS);self.assertFalse(x["promotion"])
            self.assertEqual(len((f.out/"metrics.tsv").read_text().splitlines())-1,1404)
    def test_incomplete_driver_does_not_select(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);(f.run/"EXTERNAL_EXIT_CODE").unlink()
            with self.assertRaises(FileNotFoundError):f.run_summary()
    def test_bool_outer_exit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);p=f.run/"launch_receipt.json";x=json.loads(p.read_text());x["internal_exit"]=False;dump(p,x)
            with self.assertRaisesRegex(ValueError,"integer_double"):f.run_summary()
    def test_training_audit_not_PASS_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);p=f.audit/"summary.json";x=json.loads(p.read_text());x["status"]="FAIL";dump(p,x)
            with self.assertRaisesRegex(ValueError,"training_audit_required"):f.run_summary()
    def test_audit_missing_eval_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);p=f.audit/"summary.json";x=json.loads(p.read_text());x["evaluations"].pop();dump(p,x)
            with self.assertRaisesRegex(ValueError,"incomplete_audit_evaluations"):f.run_summary()

    def test_independent_training_audit_review_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);p=f.review;x=json.loads(p.read_text());x["independent"]=False;dump(p,x)
            with self.assertRaisesRegex(ValueError,"implementation_review"):f.run_summary()
    def test_prediction_mask_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);p=Path(m.expected_args(f.root,m.CONFIGS[0],193,"original_val")["out"])/"predictions.jsonl"
            rr=[json.loads(line) for line in p.read_text().splitlines()];rr[0]["mask_content_sha256"]="tampered"
            write(p,"".join(json.dumps(r)+"\n" for r in rr))
            with self.assertRaisesRegex(ValueError,"identity_or_variant"):f.run_summary()
    def test_center_role_binding_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);p=Path(m.expected_args(f.root,m.CONFIGS[0],193,"original_val")["out"])/"summary.json"
            x=json.loads(p.read_text());x["center_estimate_sha256"].reverse();dump(p,x)
            with self.assertRaisesRegex(ValueError,"center_role_SHA"):f.run_summary()

if __name__=="__main__":unittest.main()

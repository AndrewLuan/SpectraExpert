"""用途：只读复用四个已完成 train/val paired.jsonl，聚合交集与共同候选名次变化。
核对生产器双 EXIT、完整输入 SHA；调用 SHA 固定生产器的 decompose 从既有预测
复算完整 paired 行，逐查询核对身份与 optional 字段。先按查询合并 cohort，
再等权平均两个 seed；输出独占目录、始末输入 SHA 与执行 EXIT。此工具不调用模型，
不打开任何特征缓存或 test 数据，不运行选择、冻结或科学任务。
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

SEEDS = (193, 211)
COHORTS = ("original_val", "tune")
VARIANTS = ("observed", "frequency_crop", "peak_dropout")
CATEGORIES = ("common", "new", "lost", "neither")
COUNTS = {"original_val": {"observed": 320, "frequency_crop": 320, "peak_dropout": 264},
          "tune": {"observed": 1024, "frequency_crop": 1024, "peak_dropout": 888}}
PRODUCER_SHA = "b861ca3f0e05739885cd009c46d54fc608dd68202107d28d9c23affa99e0b3f7"

def require(condition, label):
    if not condition:
        raise ValueError(label)

def close(a, b, label):
    require(type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b) and abs(a-b) < 1e-12, label)

def stats(rows):
    n = len(rows)
    require(n > 0, "empty_stratum")
    def avg(key):
        return math.fsum(r[key] for r in rows)/n
    out = dict(queries=n, left_MRR=math.fsum(1/r["left_rank"] for r in rows)/n,
               right_MRR=math.fsum(1/r["right_rank"] for r in rows)/n,
               delta_MRR=avg("delta_MRR"),
               delta_recall512=sum((r["right_rank"] <= 512)-(r["left_rank"] <= 512) for r in rows)/n)
    out.update({"delta_top"+str(k):sum((r["right_rank"] <= k)-(r["left_rank"] <= k) for r in rows)/n for k in (1,5,10)})
    out["categories"] = {}
    for c in CATEGORIES:
        cell = [r for r in rows if r["category"] == c]
        total = math.fsum(r["delta_MRR"] for r in cell)
        out["categories"][c] = dict(queries=len(cell), fraction=len(cell)/n,
            contribution_to_total_delta_MRR=total/n, conditional_delta_MRR=total/len(cell) if cell else None)
    close(sum(v["contribution_to_total_delta_MRR"] for v in out["categories"].values()), out["delta_MRR"], "category_contribution_sum")
    out["optional_pool"] = dict(queries_with_overlap=n,
        common_candidates_mean=avg("common_candidates"),
        common_candidates_min=min(r["common_candidates"] for r in rows),
        common_candidates_max=max(r["common_candidates"] for r in rows),
        common_candidates_sum=sum(r["common_candidates"] for r in rows),
        added_candidates_mean=avg("added_candidates"), removed_candidates_mean=avg("removed_candidates"),
        common_candidate_mean_abs_rank_change=avg("common_candidate_mean_abs_rank_change"),
        query_weighting="one equal weight per query; each query metric already averages common candidates")
    return out

def validate_rows(rows, cohort):
    seen = set()
    for r in rows:
        key = r["query_key"]
        require(isinstance(key, list) and len(key) == 5 and all(isinstance(k, str) for k in key), "query_key")
        require(tuple(key) not in seen, "duplicate_query")
        seen.add(tuple(key))
        require((key[1], key[2], key[4]) == (r["variant"], r["source"], r["group"]), "query_identity")
        require(r["variant"] in VARIANTS, "variant")
        left, right = r["left_rank"], r["right_rank"]
        require(type(left) is int and type(right) is int and left > 0 and right > 0, "rank")
        category = {(True,True):"common", (False,True):"new", (True,False):"lost", (False,False):"neither"}[(left <= 512, right <= 512)]
        require(r["category"] == category, "category")
        close(r["delta_MRR"], 1/right-1/left, "delta_from_ranks")
        require(r["candidate_overlap_available"] is True, "overlap_unavailable")
        for k in ("common_candidates", "added_candidates", "removed_candidates"):
            require(type(r[k]) is int and 0 <= r[k] <= 512, "intersection_count")
        require(r["common_candidates"] + r["added_candidates"] == 512 and r["common_candidates"] + r["removed_candidates"] == 512, "pool512")
        z = r["common_candidate_mean_abs_rank_change"]
        require(type(z) in (int,float) and math.isfinite(z) and 0 <= z <= max(0,r["common_candidates"]-1), "common_rank_metric")
    for v in VARIANTS:
        require(sum(r["variant"] == v for r in rows) == COUNTS[cohort][v], "query_count:"+cohort+":"+v)
    return rows

def verify_summary(actual, expected):
    require(actual["queries"] == expected["queries"], "summary_count")
    for k in ("left_MRR", "right_MRR", "delta_MRR", "delta_recall512", "delta_top1", "delta_top5", "delta_top10"):
        close(actual[k], expected[k], "summary_"+k)
    require(set(actual["categories"]) == set(CATEGORIES), "summary_categories")
    for c in CATEGORIES:
        a,b = actual["categories"][c],expected["categories"][c]
        require(a["queries"] == b["queries"], "summary_category_count")
        close(a["contribution_to_total_delta_MRR"], b["contribution_to_total_delta_MRR"], "summary_contribution")
        if b["conditional_delta_MRR"] is None:
            require(a["conditional_delta_MRR"] is None, "summary_empty_conditional")
        else:
            close(a["conditional_delta_MRR"], b["conditional_delta_MRR"], "summary_conditional")

def equal_seed_mean(per_seed):
    a,b = per_seed
    require(a["queries"] == b["queries"], "seed_query_counts")
    n=a["queries"]
    out = {"queries_per_seed": n, "seeds": list(SEEDS)}
    for k in ("left_MRR","right_MRR","delta_MRR","delta_recall512","delta_top1","delta_top5","delta_top10"):
        out[k]=(a[k]+b[k])/2
    out["categories"]={}
    for c in CATEGORIES:
        x,y=a["categories"][c],b["categories"][c]
        out["categories"][c] = dict(queries_by_seed={str(SEEDS[0]):x["queries"],str(SEEDS[1]):y["queries"]},
            queries_mean_per_seed=(x["queries"]+y["queries"])/2,
            fraction=(x["fraction"]+y["fraction"])/2,
            contribution_to_total_delta_MRR=(x["contribution_to_total_delta_MRR"]+y["contribution_to_total_delta_MRR"])/2)
    close(sum(c["contribution_to_total_delta_MRR"] for c in out["categories"].values()),out["delta_MRR"],"two_seed_contributions")
    out["optional_pool"]={}
    for k in ("common_candidates_mean","added_candidates_mean","removed_candidates_mean","common_candidate_mean_abs_rank_change"):
        out["optional_pool"][k]=(a["optional_pool"][k]+b["optional_pool"][k])/2
    out["optional_pool"]["common_candidates_min"]=min(a["optional_pool"]["common_candidates_min"],b["optional_pool"]["common_candidates_min"])
    out["optional_pool"]["common_candidates_max"]=max(a["optional_pool"]["common_candidates_max"],b["optional_pool"]["common_candidates_max"])
    return out

class Reader:
    def __init__(self):
        self.pins={}
    def read(self,path):
        p=Path(path).resolve()
        require(not any(part in ("test","sealed","verify") for part in p.parts) and p.suffix != ".pt", "forbidden_input:"+str(p))
        data=p.read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        require(str(p) not in self.pins or self.pins[str(p)]==digest,"input_drift")
        self.pins[str(p)]=digest
        return data
    def obj(self,path):
        return json.loads(self.read(path))
    def check_end(self):
        after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in self.pins}
        require(after==self.pins,"input_changed")
        return after

def verify_receipt(v):
    require(v.get("state")=="complete","receipt_state")
    for k in ("external_exit","internal_exit"):
        require(type(v.get(k)) is int and v[k]==0,"receipt_exit:"+k)

def load_decompose(source_bytes, source_path):
    require(hashlib.sha256(source_bytes).hexdigest()==PRODUCER_SHA,"producer_source")
    namespace={"__name__":"_sha_bound_pool_diagnostics","__file__":str(source_path)}
    exec(compile(source_bytes,str(source_path),"exec"),namespace)
    return namespace["decompose"]

def verify_derived_pairs(actual, derived):
    expected=json.loads(json.dumps(derived,allow_nan=False))
    def index(rows):
        result={}
        for row in rows:
            key=tuple(row["query_key"])
            require(key not in result,"derived_pair_duplicate")
            result[key]=row
        return result
    a,b=index(actual),index(expected)
    require(a.keys()==b.keys(),"derived_pair_identity")
    for key in a:
        require(json.dumps(a[key],sort_keys=True,allow_nan=False)==json.dumps(b[key],sort_keys=True,allow_nan=False),
                "derived_pair_content:"+str(key))

def verify_cross_seed_identity(raw):
    for cohort in COHORTS:
        keys=[{tuple(r["query_key"]) for r in raw[(seed,cohort)]} for seed in SEEDS]
        require(keys[0]==keys[1],"cross_seed_query_identity:"+cohort)

def run(root):
    root=Path(root).resolve(); read=Reader()
    read.read(Path(__file__))
    completion=read.obj(root/"runs/diagnostics/completion.json")
    for name in ("EXIT_CODE","EXTERNAL_EXIT_CODE"):
        require(read.read(root/"runs/diagnostics"/name).strip()==b"0","driver_exit")
    details=[]; raw={}
    for seed in SEEDS:
        for cohort in COHORTS:
            paired=root/f"diagnostics/paired/s{seed}"/cohort
            job=f"decompose_s{seed}_{cohort}"
            receipt=read.obj(root/"runs/diagnostics"/(job+".receipt.json"))
            verify_receipt(receipt)
            require(completion["jobs"][job]==receipt,"completion_receipt_binding")
            require(Path(receipt["output"])==paired,"receipt_output")
            require(receipt["argv"][1]==str(root/"pool_diagnostics.py"),"producer_argv_source")
            require(receipt["argv"][receipt["argv"].index("--out")+1]==str(paired),"producer_argv_out")
            require(read.read(paired/"EXIT_CODE").strip()==b"0","producer_exit")
            summary=read.obj(paired/"summary.json")
            require(summary["scope"]=="train_val_only" and summary["cohort"]==cohort and summary["inputs_unchanged"] is True,"summary_scope")
            expected={str(root/"pool_diagnostics.py")}
            eval_summaries={}
            for pool in ("single","dual"):
                evaldir=root/f"diagnostics/frozen/s{seed}"/pool/cohort
                ejob=f"eval_frozen_s{seed}_{pool}_{cohort}"
                er=read.obj(root/"runs/diagnostics"/(ejob+".receipt.json"))
                verify_receipt(er)
                require(completion["jobs"][ejob]==er and Path(er["output"])==evaldir,"eval_receipt_binding")
                flag="--"+pool
                require(receipt["argv"][receipt["argv"].index(flag)+1]==str(evaldir),"producer_argv")
                es=read.obj(evaldir/"summary.json")
                require(es["cohort"]==cohort,"eval_cohort")
                eval_summaries[pool]=es
                require(read.read(evaldir/"EXIT_CODE").strip()==b"0","eval_exit")
                expected.update(str(evaldir/name) for name in ("summary.json","predictions.jsonl","EXIT_CODE"))
            require(set(summary["inputs_sha256"])==expected,"producer_input_set")
            verified_bytes={}
            for p,digest in summary["inputs_sha256"].items():
                verified_bytes[p]=read.read(p)
                require(read.pins[str(Path(p).resolve())]==digest,"producer_input_sha")
            for field in ("cohort","query_sha256","candidate_order_sha256","checkpoints"):
                require(field in eval_summaries["single"] and eval_summaries["single"][field]==eval_summaries["dual"].get(field),
                        "producer_fixed_identity:"+field)
            decompose=load_decompose(verified_bytes[str(root/"pool_diagnostics.py")],root/"pool_diagnostics.py")
            prediction_rows={}
            for pool in ("single","dual"):
                prediction_path=root/f"diagnostics/frozen/s{seed}"/pool/cohort/"predictions.jsonl"
                prediction_rows[pool]=[json.loads(line) for line in verified_bytes[str(prediction_path)].splitlines() if line]
            derived,_=decompose(prediction_rows["single"],prediction_rows["dual"])
            rows=validate_rows([json.loads(line) for line in read.read(paired/"paired.jsonl").splitlines() if line],cohort)
            verify_derived_pairs(rows,derived)
            require(set(summary["by_variant"])==set(VARIANTS),"summary_variants")
            for variant in VARIANTS:
                cell=[r for r in rows if r["variant"]==variant]
                result=stats(cell); verify_summary(summary["by_variant"][variant],result)
                details.append(dict(seed=seed,cohort=cohort,variant=variant,**result))
            require(set(summary["by_source"])=={r["source"] for r in rows},"summary_sources")
            for source in summary["by_source"]:
                verify_summary(summary["by_source"][source],stats([r for r in rows if r["source"]==source and r["variant"]=="observed"]))
            raw[(seed,cohort)]=rows
    verify_cross_seed_identity(raw)
    combined=[]; final={}
    for variant in VARIANTS:
        per_seed=[]
        for seed in SEEDS:
            rows=[r for cohort in COHORTS for r in raw[(seed,cohort)] if r["variant"]==variant]
            val=stats(rows);per_seed.append(val)
            combined.append(dict(seed=seed,cohort="query_weighted_combined",variant=variant,**val))
        final[variant]=equal_seed_mean(per_seed)
    before=dict(read.pins);after=read.check_end()
    return dict(schema="existing_paired_diagnostic_reuse_20261023",scope="completed_train_val_outputs_only",
        utc=datetime.now(timezone.utc).isoformat(),comparison="frozen_dual_minus_frozen_single",
        aggregation="cohorts merged per query within each seed and variant; then equal mean of seeds 193 and 211",
        common_rank_definition="per-query mean absolute rank change among common candidates only, then equal mean across queries",
        category_definition={"common":"target in both natural512 pools","new":"target only in dual","lost":"target only in single","neither":"target in neither"},
        by_seed_cohort_variant=details,by_seed_combined_variant=combined,two_seed_query_weighted=final,
        input_sha256_before=before,input_sha256_after=after,inputs_unchanged=True,
        paired_derivation_verified=True,cross_seed_query_identity_verified=True,
        derivation_method="SHA-bound producer decompose on already completed SHA-verified predictions; exact full JSON row equality",
        models_executed=False,test_values_read=False,test_predictions_read=False,selection_recomputed=False,freeze_executed=False)

def markdown(result):
    lines=["既有配对诊断复用","",
      "先在各 seed 内按 query 合并 original_val 与 tune，再等权平均两个 seed。ΔMRR 为 frozen_dual − frozen_single。",
      "共同候选名次变化先在每个 query 的候选交集内计算，再等权平均 query。","",
      "| variant | queries/seed | common candidates mean | mean absolute common rank change | ΔMRR |",
      "|---|---:|---:|---:|---:|"]
    for v,x in result["two_seed_query_weighted"].items():
        p=x["optional_pool"]
        lines.append(f"| {v} | {x['queries_per_seed']} | {p['common_candidates_mean']:.6f} | {p['common_candidate_mean_abs_rank_change']:.6f} | {x['delta_MRR']:.9f} |")
    lines += ["","| variant | target category | n seed193 | n seed211 | ΔMRR contribution |","|---|---|---:|---:|---:|"]
    for v,x in result["two_seed_query_weighted"].items():
        for c,z in x["categories"].items():
            lines.append(f"| {v} | {c} | {z['queries_by_seed']['193']} | {z['queries_by_seed']['211']} | {z['contribution_to_total_delta_MRR']:.9f} |")
    lines += ["","JSON 保存每个 seed/cohort/variant 明细及全部输入始末 SHA。生产器、评估任务与 diagnostics driver 内外 EXIT 均已核对。"]
    return "\n".join(lines)+"\n"

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    try:
        result=run(a.root)
        (out/"EXISTING_PAIRED_DIAGNOSTIC_REUSE.json").write_text(json.dumps(result,indent=2)+"\n")
        (out/"EXISTING_PAIRED_DIAGNOSTIC_REUSE.md").write_text(markdown(result))
        (out/"EXIT_CODE").write_text("0\n")
        print(json.dumps({"status":"PASS","out":str(out),"two_seed_query_weighted":result["two_seed_query_weighted"]}))
    except BaseException:
        (out/"EXIT_CODE").write_text("1\n")
        raise
if __name__=="__main__":
    main()

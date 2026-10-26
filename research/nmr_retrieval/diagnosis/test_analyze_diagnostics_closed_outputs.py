"""功能概述：封存保存tensor分析器的纯合成边界测试，不读取任何真实shard或模型。
make_saved_row按真实worker nullable loss/双shape/资格/trace字段创建内存tensor。
测试固定容差、native成本位值、所有字段有限性、双cap/双role、full513边界和unknown分母；
元数据测试只使用临时虚构对象；不调用run_analysis、worker_rows或preparer.main。
"""
import copy
import csv
import io
import math
import unittest
from pathlib import Path
import torch
import analyze_diagnostics_closed_outputs as a


def group_with_internal(wanted, prefix):
    return next(prefix+str(i) for i in range(10000) if a.internal_qualification(prefix+str(i))==wanted)


FIT=group_with_internal(False,"synthetic-fit-")
INTERNAL=group_with_internal(True,"synthetic-internal-")
NEG=group_with_internal(False,"synthetic-negative-")


def actual_loss(trace,coarse,strict,target,cap):
    allowed=~strict.clone();allowed[target]=True
    cost=coarse/trace["scale"][0]+trace["delta"]
    probe=a.bm.loss_probe(coarse.tolist(),trace["delta"].tolist(),float(trace["scale"][0]),target,
                          allowed.tolist(),trace["logit"].tolist(),cap,actual_costs=cost.tolist())
    v=lambda name:torch.tensor(probe[name],dtype=torch.float64)
    gl,ga=v("listwise_gradient_delta"),v("anchor_gradient_delta")
    derivative=v("tanh_derivative")
    return dict(allowed=allowed,actual_normalized_costs=cost,normalized_cost_dtype=str(cost.dtype),
                probability=v("probabilities"),probability_dtype="torch.float64",
                listwise_gradient_delta=gl,anchor_gradient_delta=ga,total_gradient_delta=gl+ga,
                tanh_local_derivative=derivative,listwise_gradient_logit=gl*derivative,
                anchor_gradient_logit=ga*derivative,total_gradient_logit=(gl+ga)*derivative,
                listwise_loss=probe["listwise_loss"],anchor_loss=probe["anchor_loss"],
                total_loss=probe["listwise_loss"]+probe["anchor_loss"],support_count=int(allowed.sum()),
                gradient_scope="current_single_role_row_before_batch_mean",
                actual_historical_exposure=None,actual_historical_exposure_status="unknown")


def make_saved_row(population="fit",miss=False,append=False):
    assert not append or population=="fit"
    qgroup=INTERNAL if population=="internal" else FIT
    external=population=="external";internal=population=="internal";fit=population=="fit"
    heldout={qgroup} if external else set()
    target=-1 if miss or append else 511
    groups=[NEG]*512
    if target>=0:groups[target]=qgroup
    strict=torch.tensor([g==qgroup or g in heldout or a.internal_qualification(g) for g in groups])
    row=dict(schema="top10_frozen_worker_v1_tensor_row",seed=193,cohort="original_val" if external else "train",
        population="fixed_fit" if fit else "internal" if internal else "original_val",
        query_identity=dict(query_order=0,variant="observed",source="HMDB",record_id="synthetic",smiles="C",group=qgroup),
        pool="single" if append else "dual",kind=0,fixed400_member=fit,original_bound_obstructed=True if fit else None,
        cached_outside_pool_rank=1000 if target<0 else None,mask_content_sha256=None,
        observation_semantics="line_positions",public_coverage="unknown",public_observation=dict(windows=None,partial_peaks=False),
        candidate_indices=torch.arange(512),candidate_groups=groups,candidate_smiles=["C"]*512,
        coarse_cost=torch.zeros(512,dtype=torch.float64),target_position=target,target_candidate_id=512 if append else 511 if target>=0 else None,
        target_natural=target>=0,target_input_status="present" if target>=0 else "null_natural_miss",
        cache_loss_only_positive_present=append,diagnostic_loss_only_positive_used=append,
        loss_target_position=512 if append else target if fit else None,
        loss_candidate_indices=torch.arange(513) if append else None,
        loss_candidate_groups=groups+[qgroup] if append else None,loss_candidate_smiles=["C"]*513 if append else None,
        loss_coarse_cost=torch.zeros(513,dtype=torch.float64) if append else None,
        loss_strict_excluded=torch.cat((strict,torch.ones(1,dtype=torch.bool))) if append else None,
        actual_train_query=fit,query_external_heldout=external,query_internal_group=internal,
        candidate_qualification_bits=dict(same_query_group=torch.tensor([g==qgroup for g in groups]),
            external_heldout=torch.tensor([g in heldout for g in groups]),
            internal_group=torch.tensor([a.internal_qualification(g) for g in groups])),
        strict_excluded=strict,gamma=.25,original_other_candidate_index=0,
        actual_historical_exposure=None,actual_historical_exposure_status="unknown",
        caps={},roles={},loss_roles={} if append else None)
    important=set(range(10))|({target} if target>=0 else set())
    def trace(n,cap,role):
        w=.75 if role=="no_aug" else .25
        positions=sorted(important|({512} if n==513 else set()))
        t=dict(raw_X=torch.zeros(n,33),raw_O=torch.zeros(n,3),input36=torch.zeros(n,36),
               logit=torch.zeros(n),delta=torch.zeros(n),scale=torch.ones(33),fixed_x_center=torch.zeros(33),
               fixed_o_center=torch.zeros(3),shape_kind="natural512_inference" if n==512 else "full513_loss_only",
               inference_role_weight=w if n==512 else None,actual_train_loss=None,
               actual_train_loss_status="fit_current_frozen_loss_full513" if n==513 else
                   "separate_loss_only_full513_shape" if append else "fit_current_frozen_loss_reuses_natural512" if fit else "not_training_query",
               input_jacobians=dict(positions=positions,dlogit_dinput36=torch.zeros(len(positions),36),
                  ddelta_dinput36=torch.zeros(len(positions),36),
                  weighted_dcost_dinput36=torch.zeros(len(positions),36) if n==512 else None))
        if fit and (n==513 or not append):
            t["actual_train_loss"]=actual_loss(t,row["loss_coarse_cost"] if n==513 else row["coarse_cost"],
                row["loss_strict_excluded"] if n==513 else strict,512 if n==513 else target,cap)
        t["trace_sha256_natural512" if n==512 else "trace_sha256_loss513"]=a.saved_trace_hashes(t)
        return t
    for cap in (5,10):
        row["caps"][cap]=dict(natural_cost=torch.zeros(512,dtype=torch.float64),top10_positions=list(range(10)),
            dtypes=dict(coarse_dtype="torch.float64",correction_dtypes=["torch.float32"]*2,final_cost_dtype="torch.float64"))
        row["roles"][cap]={role:trace(512,cap,role)for role in a.ROLES}
        if append:row["loss_roles"][cap]={role:trace(513,cap,role)for role in a.ROLES}
    return row,heldout


def facts(row):
    old={(193,a.identity(row)):dict(original_bound_obstructed=True,original_coarse_gap=0.,original_other_candidate_index=0)} if row["fixed400_member"] else {}
    previous={}
    for c in (5,10):
        obj=row["caps"][c];roles=row["roles"][c]
        fp=a.bm.cost_fingerprint(row["candidate_indices"].tolist(),row["coarse_cost"].tolist(),obj["natural_cost"].tolist(),obj["dtypes"],
                                 [roles[r]["trace_sha256_natural512"]for r in a.ROLES])
        rank=1000 if row["target_position"]<0 else a.bm.native_rank(obj["natural_cost"].tolist(),row["target_position"],"torch.float64")
        previous[a.key(193,c,row["pool"],row)]=dict(cost_fingerprint=fp,diagnostic=dict(rank=rank))
    return [fact for _,fact,_ in a.worker_facts(row,old,previous, {(193,c,a.identity(row)):list(range(10))for c in(5,10)})]


def fake_provenance():
    root=Path(a.__file__).resolve().parent
    stage=root/"synthetic_metadata_only_source_stage"
    names=("original_decision","original_decision_text","source_execution","closure_execution",
           "closure_report","closure_review","sealed_manifest")
    refs={name:dict(path=str(root/("synthetic_metadata_"+name+".json")),sha256=str(i)*64) for i,name in enumerate(names,1)}
    refs["original_decision"]["sha256"]=a.DECISION_SHA256
    refs["original_decision_text"]["sha256"]=a.DECISION_TEXT_SHA256
    refs["closure_review"]["sha256"]=a.CLOSURE_REVIEW_SHA256
    refs["source_execution"]["path"]=str(stage/"actual_execution.json")
    refs["source_stage_root"]=str(stage)
    contract=dict(path=str(root/"synthetic_metadata_contract.json"),sha256="a"*64)
    bindings=dict(path=str(root/"synthetic_metadata_bindings.json"),sha256="b"*64)
    query=dict(path=str(root/"synthetic_metadata_queries.json"),sha256="c"*64)
    decision=dict(schema="original_planning_scientific_decoding_decision_v1",status="AUTHORIZED_CLOSED_SAVED_TENSOR_SCIENTIFIC_ANALYSIS",
        decision_owner="original_planning_session",decision_text_path=refs["original_decision_text"]["path"],
        decision_text_sha256=a.DECISION_TEXT_SHA256,closure_independent_review_sha256=a.CLOSURE_REVIEW_SHA256,
        saved_tensor_scientific_decode_authorized=True,saved_tensor_algebra_statistics_authorized=True,
        new_model_forward_authorized=False,training_authorized=False,test_read_authorized=False,external_API_authorized=False,
        formal_system_changed=False,additional_sample_authorized=False,new_definition_authorized=False,new_NN_budget=0,
        query_seed_pairs=826,pool_count=2,source_NN_stage_status="FAIL_SUPERVISION",source_RSS_gap_status="UNKNOWN_NOT_CERTIFIED",
        model_state_assertion_scope="SOURCE_BOUND_NOT_INDEPENDENTLY_REMEASURED",source_actual_inner_exit=0,source_actual_external_exit=1)
    source=dict(actual_inner_exit=0,actual_external_exit=1,termination_reason="rss_monitor_unavailable",
                rss_missing_while_alive=True,job_sha256="d"*64,review_sha256="e"*64)
    execution=dict(actual_inner_exit=0,actual_external_exit=0,termination_reason=None,job_sha256="f"*64,review_sha256="0"*64)
    closure=dict(schema="failed_supervision_output_closure_v1",status="PASS_RAW_OUTPUT_CLOSURE",
        execution_scope="FAILED_SUPERVISION_OUTPUT_CLOSURE_ONLY",original_stage_status="FAIL_SUPERVISION",
        original_rss_gap_status="unknown_not_certified",original_resource_envelope_recertified=False,
        raw_shards_hashed=1652,closed_output_files=1661,original_actual_inner_exit=0,original_actual_external_exit=1,
        source_refs=dict(job=dict(sha256=source["job_sha256"]),replay_admission=dict(sha256=source["review_sha256"]),
                         query_manifest=query,remote_manifest=refs["sealed_manifest"]),
        checked_input_sha256={contract["path"]:contract["sha256"],bindings["path"]:bindings["sha256"]})
    review=dict(status="PASS_RAW_CLOSURE_REVIEW",independent=True,execution_authorized=False,scientific_execution_authorized=False,
        source_stage_status="FAIL_SUPERVISION_PRESERVED",original_RSS_gap_status="UNKNOWN_NOT_CERTIFIED",
        original_resource_envelope_recertified=False,closure_actual_inner_exit=0,closure_actual_external_exit=0,closure_actual_ssh_exit=0,
        source_stage_actual_inner_exit=0,source_stage_actual_external_exit=1,
        evidence_sha256={refs[name]["path"]:refs[name]["sha256"] for name in ("closure_execution","closure_report")},
        closure_resource_observations=execution.copy())
    files={"worker/tensor_shards/"+str(i)+".pt":dict(sha256="1"*64,bytes=1) for i in range(1652)}
    for k in ("actual_execution.json","worker/summary.json","worker/rows.jsonl")+tuple("other"+str(i)for i in range(6)):
        files[k]=dict(sha256="2"*64,bytes=1)
    files["actual_execution.json"]["sha256"]=refs["source_execution"]["sha256"]
    request=dict(completed_output_provenance=refs,query_manifest=query,qualification_contract=contract,qualification_bindings=bindings,
        worker_outputs=[dict(summary=dict(path=str(stage/"worker"/"summary.json"),sha256="2"*64),
                             index=dict(path=str(stage/"worker"/"rows.jsonl"),sha256="2"*64))])
    objects={refs[n]["path"]:v for n,v in (("original_decision",decision),("source_execution",source),
        ("closure_execution",execution),("closure_report",closure),("closure_review",review),("sealed_manifest",dict(files=files)))}
    objects[contract["path"]]=dict(training_excluded_groups=["synthetic-heldout"])
    objects[bindings["path"]]=dict(heldout_groups_sha256=a.hashlib.sha256(b"synthetic-heldout").hexdigest())
    class FakePins:
        def json(self,ref):return objects[ref["path"]]
        def verify(self,path,digest):return Path(path)
    return request,FakePins(),objects


class ClosedSavedTests(unittest.TestCase):
    def test_true_worker_nullable_internal_external_hit_miss(self):
        for population in ("internal","external"):
            for miss in (False,True):
                with self.subTest(population=population,miss=miss):
                    row,heldout=make_saved_row(population,miss=miss)
                    a.validate_saved_shard(row,heldout)
                    for fact in facts(row):
                        self.assertEqual(fact["query_qualification"],population)
                        self.assertEqual(fact["diagnostic"]["target_in_pool"],not miss)
                        self.assertEqual(fact["loss_evidence"]["no_aug"]["status"],"not_training_query")
                        self.assertIsNone(fact["loss_evidence"]["no_aug"]["target_probability"])
                        self.assertFalse(row["diagnostic_loss_only_positive_used"])
                        self.assertIsNone(row["loss_target_position"])

    def test_fit_natural_and_full513_valid_saved_loss(self):
        for append in (False,True):
            row,heldout=make_saved_row(append=append)
            a.validate_saved_shard(row,heldout)
            for fact in facts(row):
                info=fact["loss_evidence"]["aug"]
                self.assertEqual(info["support_count"],513 if append else 512)
                self.assertAlmostEqual(info["total_loss"],math.log(info["support_count"]))
                self.assertEqual(info["target_tanh_local_derivative"],1.)
                self.assertEqual(info["actual_loss_shape"],513 if append else 512)
                point=info["candidate_points"]["target"]
                self.assertEqual(point["absolute_anchor_to_listwise_delta_ratio"],0.)
                self.assertIsNone(point["ratio_undefined_reason"])
                self.assertEqual(info["critical_dual_membership_table"][0]["exclusive_reason"],"eligible_negative")

    def test_every_present_numeric_field_finite(self):
        row,heldout=make_saved_row()
        row["optional_metadata"]={"value":float("nan")}
        with self.assertRaisesRegex(ValueError,"nonfinite"):a.validate_saved_shard(row,heldout)

    def test_natural_extra_shape_head_role_rejected(self):
        for mutate in (lambda r:r["caps"].update({7:r["caps"][5]}),
                       lambda r:r["roles"][5].update(extra=r["roles"][5]["aug"]),
                       lambda r:r.update(candidate_indices=torch.arange(513))):
            row,h=make_saved_row();mutate(row)
            with self.assertRaises(ValueError):a.validate_saved_shard(row,h)

    def test_missing_required_trace_field_rejected(self):
        row,h=make_saved_row();del row["roles"][5]["aug"]["scale"]
        with self.assertRaises(KeyError):a.validate_saved_shard(row,h)

    def test_wrong_gamma_mask_and_target_identity_rejected(self):
        for mutate in (lambda r:r.update(gamma=.75),
                       lambda r:r["strict_excluded"].fill_(False),
                       lambda r:r.update(target_candidate_id=5)):
            row,h=make_saved_row();mutate(row)
            with self.assertRaises(ValueError):a.validate_saved_shard(row,h)

    def test_natural_miss_cost_still_native_checked(self):
        row,h=make_saved_row("external",miss=True)
        row["caps"][5]["natural_cost"][20]=1e-12
        with self.assertRaisesRegex(ValueError,"all512_native_cost"):a.validate_saved_shard(row,h)

    def test_trace_byte_hash_rejected(self):
        row,h=make_saved_row();row["roles"][5]["aug"]["trace_sha256_natural512"][0]="f"*64
        with self.assertRaisesRegex(ValueError,"saved_trace_typed"):a.validate_saved_shard(row,h)

    def test_loss_scalar_and_component_gradient_checked(self):
        for field in ("total_loss","support_count","anchor_gradient_logit","tanh_local_derivative","total_gradient_delta"):
            row,h=make_saved_row()
            loss=row["roles"][5]["aug"]["actual_train_loss"]
            if hasattr(loss[field],"shape"):loss[field][0]+=1
            else:loss[field]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):a.validate_saved_shard(row,h)

    def test_full513_prefix_ids_coarse_checked(self):
        for field in ("loss_candidate_indices","loss_coarse_cost"):
            row,h=make_saved_row(append=True);row[field][0]+=1
            with self.assertRaisesRegex(ValueError,"full513_natural_prefix"):a.validate_saved_shard(row,h)

    def test_separate_shape_saved_outputs_do_not_require_network_prefix_bits(self):
        row,h=make_saved_row(append=True)
        # 一个独立513 trace在固定容差内有自己的已保存值；绝不拿512输出强行替代。
        tr=row["loss_roles"][5]["aug"]
        tr["logit"][0]=.001;tr["delta"][0]=5*torch.tanh(tr["logit"][0]/5)
        tr["actual_train_loss"]=actual_loss(tr,row["loss_coarse_cost"],row["loss_strict_excluded"],512,5)
        tr["trace_sha256_loss513"]=a.saved_trace_hashes(tr)
        a.validate_saved_shard(row,h)
        self.assertNotEqual(float(tr["delta"][0]),float(row["roles"][5]["aug"]["delta"][0]))

    def test_all_unknown_nearest_container_counts_unknown(self):
        row,h=make_saved_row()
        fs=facts(row)
        for f in fs:
            f["nn_evidence"]=[dict(status="unknown",role=r,is_target=True,reason="no_out_of_group_sampled_reference")for r in a.ROLES]
        out=a.summarize(fs)
        self.assertEqual(out["nn_availability"]["unknown"],2)
        self.assertEqual(out["nn_availability"]["known"],0)
        self.assertEqual(out["core_crosscounts"]["S2_crossquery_collision_reference_unknown"],2)
        row,h=make_saved_row("external",miss=True)
        f=facts(row)[0]
        self.assertEqual(a.nearest_state(f,True),"not_applicable")

    def test_crosspool_critical_preserves_overlapping_bits(self):
        row,h=make_saved_row()
        row["candidate_qualification_bits"]["same_query_group"][0]=True
        row["candidate_qualification_bits"]["external_heldout"][0]=True
        row["roles"][5]["aug"]["actual_train_loss"]["allowed"][0]=False
        t=a.critical_membership_table(row,row["candidate_indices"].tolist(),row["roles"][5]["aug"]["actual_train_loss"],[0,800],511)
        self.assertEqual(t[0]["exclusive_reason"],"same_query_group")
        self.assertTrue(t[0]["exclusion_bits"]["external_heldout"])
        self.assertFalse(t[1]["in_natural"]);self.assertFalse(t[1]["in_loss"])

    def test_compact_table_is_direct_six_axis_projection(self):
        row,h=make_saved_row()
        fs=facts(row)
        parts=a.population_summaries(fs,{(193,a.identity(row))})
        compact=a.compact_population_rows(parts)
        unstratified=[p for p in parts if len(p["group"])==6]
        self.assertEqual(len(compact),len(unstratified))
        for r,p in zip(compact,unstratified):
            self.assertEqual(r["denominator_query_seed_pairs"],p["query_seed_pairs"])
            self.assertEqual(r["top10"],p["metrics"]["top10"])
            self.assertEqual(sum(r["bucket_"+b]for b in ("P0_target_absent","S0_actual_top10","S1_fail_static_blocked","S2_fail_static_open")),p["query_seed_pairs"])
        output=io.StringIO();w=csv.DictWriter(output,fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(output.getvalue())))),len(compact))

    def test_explicit_decision_plus_closure_keeps_source_failed(self):
        request,pins,objects=fake_provenance()
        result=a.validate_completed_provenance(request,pins)
        self.assertEqual(result["source_NN_stage_status"],"FAIL_SUPERVISION")
        self.assertEqual(result["source_actual_external_exit"],1)
        self.assertEqual(result["closure_actual_external_exit"],0)
        self.assertFalse(result["original_resource_envelope_recertified"])
        self.assertEqual(result["new_NN_calls"],0)

    def test_closure_alone_does_not_authorize_science(self):
        request,pins,objects=fake_provenance()
        objects[request["completed_output_provenance"]["original_decision"]["path"]]["saved_tensor_scientific_decode_authorized"]=False
        with self.assertRaisesRegex(ValueError,"decision_requires"):a.validate_completed_provenance(request,pins)

    def test_source_failure_or_RSS_gap_cannot_be_relabelled(self):
        for name,field,value in (("source_execution","actual_external_exit",0),
                                 ("source_execution","actual_inner_exit",False),
                                 ("closure_review","original_resource_envelope_recertified",True),
                                 ("closure_execution","actual_external_exit",1)):
            request,pins,objects=fake_provenance()
            objects[request["completed_output_provenance"][name]["path"]][field]=value
            with self.subTest(name=name,field=field),self.assertRaises(ValueError):a.validate_completed_provenance(request,pins)

    def test_heldout_contract_join_must_match_saved_binding(self):
        request,pins,objects=fake_provenance()
        objects[request["qualification_bindings"]["path"]]["heldout_groups_sha256"]="f"*64
        with self.assertRaisesRegex(ValueError,"source_bound_heldout"):a.validate_completed_provenance(request,pins)

    def test_structural_contained_unknown_and_not_applicable(self):
        row,h=make_saved_row();f=facts(row)[0]
        f["structural_evidence"]={"pairs":{k:{"status":"unknown"}for k in ("immutable_old_other","current_negative1","current_negative10")}}
        self.assertIsNone(a.structural_state(f))
        for v in f["structural_evidence"]["pairs"].values():v["status"]="not_applicable"
        self.assertEqual(a.structural_state(f),"not_applicable")
        f["structural_evidence"]["pairs"]["current_negative1"]["status"]="known"
        self.assertEqual(a.structural_state(f),"known")

    def test_exact_exit_rejects_bool_float_and_source_success(self):
        for value in (True,0.,None):
            with self.assertRaises(ValueError):a.exact_exit({"x":value},"x",0)
        a.exact_exit({"x":0},"x",0)
        with self.assertRaises(ValueError):a.exact_exit({"x":0},"x",1)


if __name__=="__main__":
    unittest.main()

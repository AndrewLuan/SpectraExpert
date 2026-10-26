"""功能概述：独立闭包结果checker的纯合成检查；只构造内存tensor和N内临时元数据。
以torch元素运算和当前单role objective独立核native dtype/loss；用虚构512/513行核契约负例。
不打开真实shard/历史预测/test/模型；不执行checker主入口，不调用NN或外部API。
"""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,math,random,tempfile,types
import numpy as np
import torch
torch.set_num_threads(4)
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("reviewer_closed_checker",ROOT/"independent_closed_outputs_checker.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.np=np;m.torch=torch
def rejected(fn):
    try:fn()
    except (AssertionError,KeyError,IndexError,TypeError,ValueError):return
    raise AssertionError("invalid synthetic accepted")
def numeric():
    rng=random.Random(317);cases=0
    for cd in ("torch.float32","torch.float64"):
        for d0 in ("torch.float32","torch.float64"):
            for d1 in ("torch.float32","torch.float64"):
                for gamma in (.25,.75):
                    kinds=[getattr(torch,s.split(".")[-1]) for s in (cd,d0,d1)]
                    base=torch.tensor([rng.uniform(-1e5,1e5) for _ in range(12)],dtype=kinds[0])
                    a=torch.tensor([rng.uniform(-5,5) for _ in range(12)],dtype=kinds[1])
                    b=torch.tensor([rng.uniform(-5,5) for _ in range(12)],dtype=kinds[2])
                    scales=[float(torch.tensor(1.37,dtype=kinds[1])),float(torch.tensor(2.41,dtype=kinds[2]))]
                    expected=base+(1-gamma)*(a*scales[0])+gamma*(b*scales[1])
                    d=dict(coarse_dtype=cd,correction_dtypes=[d0,d1],final_cost_dtype=str(expected.dtype))
                    assert np.array_equal(m.native_cost(base.numpy(),a.numpy(),b.numpy(),scales,gamma,d),expected.numpy())
                    cases+=1
    d=dict(coarse_dtype="torch.float64",correction_dtypes=["torch.float32"]*2,final_cost_dtype="torch.float64")
    diag=m.diagnostic(list(range(512)),[0.]*512,[0.]*512,511,.25,[1.,1.],5,d)
    assert diag["rank"]==512 and diag["bucket"]=="S2_fail_static_open" and diag["static_unbeatable_count"]==0
    coarse=[0.]*511+[100.]
    diag=m.diagnostic(list(range(512)),coarse,coarse,511,.25,[1.,1.],5,d)
    assert diag["static_unbeatable_count"]==511 and diag["bucket"]=="S1_fail_static_blocked"
    unknown=m.diagnostic(list(range(512)),[0.]*512,[0.]*512,-1,.25,[1.,1.],5,d,outside=None)
    assert unknown["rank"] is None and unknown["bucket"]=="P0_target_absent"
    m.CAP=5;z=torch.tensor([.2,-.7,.3,.8]);delta=5*torch.tanh(z/5)
    coarse=torch.tensor([.1,.3,.7,.8],dtype=torch.float64);scale=torch.ones(33)
    allowed=torch.tensor([True,False,True,True]);cost=coarse/scale[0]+delta
    p=torch.zeros_like(cost);p[allowed]=torch.softmax(-cost[allowed]/.5,0)
    ind=torch.zeros_like(p);ind[3]=1;gl=torch.where(allowed,(ind-p)/.5,0.)
    ga=torch.where(allowed,.2*delta/int(allowed.sum()),0.)
    derivative=1-torch.tanh(z/5).square()
    listloss=torch.nn.functional.cross_entropy((-cost[allowed]/.5)[None,:],torch.tensor([2]))
    anchor=.1*delta[allowed].square().mean()
    obj=dict(actual_normalized_costs=cost,allowed=allowed,probability=p,listwise_gradient_delta=gl,
             anchor_gradient_delta=ga,total_gradient_delta=gl+ga,total_gradient_logit=(gl+ga)*derivative,
             tanh_local_derivative=derivative,listwise_gradient_logit=gl*derivative,anchor_gradient_logit=ga*derivative,
             support_count=int(allowed.sum()),normalized_cost_dtype=str(cost.dtype),probability_dtype=str(p.dtype),
             listwise_loss=float(listloss),anchor_loss=float(anchor),total_loss=float(listloss+anchor))
    trace=dict(actual_train_loss=obj,delta=delta,logit=z,scale=scale)
    assert m.loss_check(trace,coarse.numpy(),3,np.array([False,True,False,False])).tolist()==allowed.tolist()


def trace(n,cap,weight,status,actual,coarse,target,excluded):
    raw=torch.arange(n*33,dtype=torch.float32).reshape(n,33)/10000
    o=torch.arange(n*3,dtype=torch.float32).reshape(n,3)/10000
    scale=torch.ones(33);xc=torch.zeros(33);oc=torch.zeros(3)
    inp=torch.cat((raw,o),1)
    logit=inp[:,0]*.3+inp[:,33]*.2;delta=cap*torch.tanh(logit/cap)
    positions=list(range(10))+([511] if n==512 else [512])
    j=torch.zeros(len(positions),36);j[:,0]=.3;j[:,33]=.2
    dj=j*(1-torch.tanh(logit[positions]/cap).square())[:,None]
    tr=dict(raw_X=raw,raw_O=o,input36=inp,logit=logit,delta=delta,scale=scale,
            fixed_x_center=xc,fixed_o_center=oc,inference_role_weight=weight,
            shape_kind="natural512_inference" if n==512 else "full513_loss_only",
            input_jacobians=dict(positions=positions,dlogit_dinput36=j,ddelta_dinput36=dj,
             weighted_dcost_dinput36=dj*weight if weight is not None else None),
            actual_train_loss_status=status,actual_train_loss=None)
    if actual:
        allowed=~excluded.clone();allowed[target]=True
        z=logit.detach().clone().requires_grad_();d=cap*torch.tanh(z/cap)
        native=coarse/scale[0]+d
        loss=torch.nn.functional.cross_entropy((-native[allowed]/.5)[None,:],torch.tensor([int(allowed[:target].sum())]))
        anch=.1*d[allowed].square().mean()
        gl=torch.autograd.grad(loss,d,retain_graph=True)[0];ga=torch.autograd.grad(anch,d,retain_graph=True)[0]
        derivative=1-torch.tanh(z/cap).square()
        prob=torch.zeros_like(native);prob[allowed]=torch.softmax(-native[allowed]/.5,0)
        tr["actual_train_loss"]=dict(allowed=allowed,actual_normalized_costs=native.detach(),probability=prob.detach(),
            listwise_gradient_delta=gl.detach(),anchor_gradient_delta=ga.detach(),total_gradient_delta=(gl+ga).detach(),
            tanh_local_derivative=derivative.detach(),listwise_gradient_logit=(gl*derivative).detach(),
            anchor_gradient_logit=(ga*derivative).detach(),total_gradient_logit=((gl+ga)*derivative).detach(),
            support_count=int(allowed.sum()),normalized_cost_dtype=str(native.dtype),probability_dtype=str(prob.dtype),
            listwise_loss=float(loss.detach()),anchor_loss=float(anch.detach()),total_loss=float((loss+anch).detach()))
    if n==513:tr["trace_sha256_loss513"]=m.trace_hash(tr)
    return tr

def row(append=False,fit=True):
    target=-1 if append else 511;ids=torch.arange(512);groups=["negative"]*512
    if target>=0:groups[target]="target"
    coarse=torch.arange(512,dtype=torch.float64)/100
    q=dict(query_order=0,variant="observed",source="synthetic",record_id="q",smiles="C",group="target")
    excluded=torch.tensor([g=="target" for g in groups])
    r=dict(seed=193,cohort="train" if fit else "tune",pool="single" if append else "dual",query_identity=q,
           candidate_indices=ids,candidate_groups=groups,candidate_smiles=["C"]*512,coarse_cost=coarse,
           target_position=target,target_natural=target>=0,target_candidate_id=512 if append else 511,
           candidate_qualification_bits=dict(same_query_group=excluded,external_heldout=torch.zeros(512,dtype=torch.bool),internal_group=torch.zeros(512,dtype=torch.bool)),
           strict_excluded=excluded,public_observation=dict(windows=None,partial_peaks=False),gamma=.25,
           actual_train_query=fit,caps={},roles={},loss_roles={} if append else None,
           loss_candidate_indices=torch.arange(513) if append else None,
           loss_candidate_groups=groups+["target"] if append else None,loss_candidate_smiles=["C"]*513 if append else None,
           loss_coarse_cost=torch.cat((coarse,torch.tensor([6.],dtype=torch.float64))) if append else None,
           loss_strict_excluded=torch.cat((excluded,torch.tensor([True]))) if append else None,
           loss_target_position=512 if append else target if fit else None)
    for cap in (5,10):
        status="separate_loss_only_full513_shape" if append else "fit_current_frozen_loss_reuses_natural512" if fit else "not_training_query"
        rr={role:trace(512,cap,w,status,fit and not append,coarse,target,excluded) for role,w in zip(m.ROLES,(.75,.25))}
        cost=coarse+.75*(rr["no_aug"]["delta"]*rr["no_aug"]["scale"][0])+.25*(rr["aug"]["delta"]*rr["aug"]["scale"][0])
        r["roles"][cap]=rr;r["caps"][cap]=dict(natural_cost=cost,top10_positions=sorted(range(512),key=lambda i:(float(cost[i]),i))[:10],
                 dtypes=dict(coarse_dtype="torch.float64",correction_dtypes=["torch.float32"]*2,final_cost_dtype="torch.float64"))
        if append:
            r["loss_roles"][cap]={role:trace(513,cap,None,"fit_current_frozen_loss_full513",True,r["loss_coarse_cost"],512,r["loss_strict_excluded"])for role in m.ROLES}
    e={k:r[k] for k in ("seed","cohort","pool","query_identity")};e["loss_only_role_row_forwards"]=4 if append else 0
    return r,e

def contracts():
    count=0
    for append,fit in ((False,True),(True,True),(False,False)):
        r,e=row(append,fit);m.row_contract(r,e);count+=1
        for cap in (5,10):
            for role in m.ROLES:
                tr=(r["loss_roles"] if append else r["roles"])[cap][role]
                if fit:
                    m.CAP=cap
                    m.loss_check(tr,(r["loss_coarse_cost"] if append else r["coarse_cost"]).numpy(),r["loss_target_position"],(r["loss_strict_excluded"] if append else r["strict_excluded"]).numpy())
        mutations=[
            lambda r:r.update(gamma=.75),
            lambda r:r.update(target_natural=not r["target_natural"]),
            lambda r:r["candidate_indices"].__setitem__(0,-1),
            lambda r:r["roles"][5]["no_aug"]["raw_X"].__setitem__((0,0),float("nan")),
            lambda r:r["roles"][5]["no_aug"]["input_jacobians"].update(dlogit_dinput36=torch.zeros(11,36,dtype=torch.int64)),
            lambda r:r["roles"][5]["no_aug"].update(actual_train_loss_status="wrong"),
            lambda r:r["caps"][5]["dtypes"].update(final_cost_dtype="torch.int32"),
            lambda r:r["strict_excluded"].__setitem__(0,False) if False else r.update(strict_excluded=torch.zeros(512)),
        ]
        if fit and not append:
            # Even a synchronized alternate saved loss cannot redefine the natural positive identity.
            def alternate_loss_target(r):
                r["loss_target_position"]=0
                for cap in (5,10):
                    for role,weight in zip(m.ROLES,(.75,.25)):
                        r["roles"][cap][role]=trace(512,cap,weight,"fit_current_frozen_loss_reuses_natural512",True,r["coarse_cost"],0,r["strict_excluded"])
            mutations.extend([
                alternate_loss_target,
                lambda r:r.update(loss_target_position=True),
                lambda r:r.update(loss_target_position=float(r["target_position"])),
                lambda r:r.update(loss_target_position=None),
            ])
        if not fit:mutations.append(lambda r:r.update(loss_target_position=r["target_position"]))
        if append:mutations.extend([
            lambda r:r.update(loss_candidate_indices=r["loss_candidate_indices"].float()),
            lambda r:r["loss_candidate_groups"].__setitem__(0,"wrong"),
            lambda r:r["loss_coarse_cost"].__setitem__(0,999.),
            lambda r:r["loss_roles"][5]["no_aug"]["raw_X"].__setitem__((0,0),.1),
            lambda r:r.update(loss_strict_excluded=torch.zeros(513)),
        ])
        for mutate in mutations:
            bad=copy.deepcopy(r);mutate(bad);rejected(lambda:m.row_contract(bad,e));count+=1
    # Tensor type/schema checks reject malformed loss support and components, not only incorrect numbers.
    r,e=row();tr=r["roles"][5]["no_aug"];m.CAP=5
    for name,value in (("allowed",torch.ones(512)),("support_count",True),("probability",torch.ones(1)),("listwise_loss",999.)):
        bad=copy.deepcopy(tr);bad["actual_train_loss"][name]=value
        rejected(lambda:m.loss_check(bad,r["coarse_cost"].numpy(),511,r["strict_excluded"].numpy()));count+=1
    # Native FP32 scalar cancellation is preserved while probability/derivative algebra stays checked.
    coarse=torch.full((512,),1e8,dtype=torch.float32);excluded=torch.zeros(512,dtype=torch.bool)
    tr=trace(512,5,.75,"fit_current_frozen_loss_reuses_natural512",True,coarse,511,excluded)
    obj=tr["actual_train_loss"];cost=obj["actual_normalized_costs"];mask=obj["allowed"]
    ll=torch.logsumexp(-cost[mask]/.5,0)+cost[511]/.5;al=.1*tr["delta"][mask].square().mean()
    obj.update(listwise_loss=float(ll),anchor_loss=float(al),total_loss=float(ll+al))
    assert float(ll)==0. and math.log(512)>6.
    m.CAP=5;m.loss_check(tr,coarse.numpy(),511,excluded.numpy());count+=1
    # Different network outputs across512/513 are legal when each saved algebra is internally valid.
    return count

def states():
    f=dict(diagnostic=dict(target_in_pool=True),nn_evidence=[dict(status="unknown")])
    assert m.nn_state(f) is None
    f["nn_evidence"].append(dict(status="known",is_target=False))
    assert m.nn_state(f)=="known" and m.nn_state(f,True) is None
    f["diagnostic"]["target_in_pool"]=False;assert m.nn_state(f,True)=="not_applicable"
    f["structural_evidence"]=dict(pairs={"a":dict(status="not_applicable"),"b":dict(status="unknown")})
    assert m.structure_state(f) is None
    f["structural_evidence"]["pairs"]["b"]["status"]="not_applicable";assert m.structure_state(f)=="not_applicable"
    d=m.distribution([None,1.,3.]);assert d["available"]==2 and d["unknown"]==1 and d["quantiles"]["median"]==2.
    a=m.availability([None,"known","not_applicable"]);assert a==dict(total=3,known=1,unknown=1,not_applicable=1)
    return 7

def gates():
    original_parse=m.argparse.ArgumentParser.parse_args;original_sha=m.sha
    m.argparse.ArgumentParser.parse_args=lambda self:types.SimpleNamespace(execute_after_closed=False)
    m.sha=lambda path:(_ for _ in ()).throw(AssertionError("premature input read"))
    try:
        try:m.main()
        except AssertionError as error:assert str(error)=="explicit closed-stage execution required"
        else:raise AssertionError("closed gate bypass")
    finally:m.argparse.ArgumentParser.parse_args=original_parse;m.sha=original_sha
    # Fake metadata result chain uses no real source outputs.
    with tempfile.TemporaryDirectory(dir=ROOT,prefix="synthetic-checker-") as temp:
        base=Path(temp);stage=base/"analysis";results=stage/"results";results.mkdir(parents=True)
        write=lambda p,v:p.write_text(json.dumps(v))
        source=base/"fake_author.py";source.write_text("# synthetic only")
        request=base/"request.json";write(request,{})
        scope="CLOSED_OUTPUT_SAVED_TENSOR_SCIENTIFIC_ANALYSIS_ONLY"
        A=dict(status="PASS",independent=True,execution_authorized=True,execution_scope=scope,
               request_sha256=m.sha(request),reviewed_source_sha256={str(source):m.sha(source)})
        ap=base/"A.json";write(ap,A)
        job=dict(stage_dir=str(stage),source_sha256={str(source):m.sha(source)},argv=["python",str(source),"--request",str(request),"--request-sha256",m.sha(request),"--admission",str(ap),"--admission-sha256",m.sha(ap),"--out",str(results)])
        jp=base/"job.json";write(jp,job)
        B=dict(status="PASS",independent=True,execution_authorized=True,execution_scope=scope,pins=dict(job=m.sha(jp)))
        bp=base/"B.json";write(bp,B)
        exe=dict(actual_inner_exit=0,actual_external_exit=0,termination_reason=None,elapsed_seconds=1,max_seconds=3600,
                 peak_observed_rss_bytes=1,max_rss_bytes=32*1024**3,final_output_bytes_before_receipt=1,max_artifact_bytes=4*1024**3,
                 job_sha256=m.sha(jp),review_sha256=m.sha(bp))
        write(stage/"actual_execution.json",exe);(stage/"EXIT_CODE").write_text("0\n")
        summary=dict(status="PASS",independent_admission=A,input_sha256={str(request):m.sha(request)},inputs_unchanged=True,
                     model_calls=0,new_spectrum_forward_calls=0,forward_bank_calls=0,optimizer_steps=0,weight_updates=0,external_api_calls=0,test_read=False)
        write(results/"summary.json",summary)
        package=dict(schema="closed_analysis_result_manifest_v1",stage_root=str(stage),
                     refs={k:dict(path=str(p),sha256=m.sha(p))for k,p in (("analysis_job",jp),("input_admission",ap),("stage_admission",bp))},
                     files={str(p.relative_to(stage)):dict(bytes=p.stat().st_size,sha256=m.sha(p))for p in stage.rglob("*")if p.is_file()})
        mp=base/"manifest.json";write(mp,package)
        args=types.SimpleNamespace(analysis_manifest=str(mp),analysis_manifest_sha256=m.sha(mp),analysis_stage=str(stage),request=str(request),request_sha256=m.sha(request))
        got=m.analysis_chain(args);m.recheck_analysis_files(got)
        (results/"summary.json").write_text("{}");rejected(lambda:m.recheck_analysis_files(got))
        write(results/"summary.json",summary);(results/"extra").write_text("bad");rejected(lambda:m.analysis_chain(args))
    return 4

def aggregate_aliases():
    # Author code is only compared on fabricated data; native/loss numerical oracles above stay independent.
    spec=importlib.util.spec_from_file_location("synthetic_author_comparison",ROOT/"analyze_diagnostics_closed_outputs.py")
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    total=0
    for append,fit in ((False,True),(True,True),(False,False)):
        r,e=row(append,fit)
        r.update(population="fixed_fit" if fit else "tune",kind=0,fixed400_member=fit,query_external_heldout=not fit,query_internal_group=False,
                 original_bound_obstructed=True if fit else None,observation_semantics="synthetic",public_coverage="unknown",
                 diagnostic_loss_only_positive_used=append,cached_outside_pool_rank=1000 if append else None,
                 original_other_candidate_index=0,structural_evidence=None)
        oldrow=dict(target_candidate_index=r["target_candidate_id"],original_bound_obstructed=True,original_coarse_gap=0.,
                    original_other_candidate_index=0)
        om={(193,m.qid(r)):oldrow} if fit else {}
        legacy={}
        for cap in (5,10):
            rr=r["roles"][cap]
            for role in m.ROLES:rr[role]["trace_sha256_natural512"]=m.trace_hash(rr[role])
            obj=r["caps"][cap];cost=obj["natural_cost"].tolist()
            legacy[a.key(193,cap,r["pool"],r)]=dict(cost_fingerprint=a.bm.cost_fingerprint(r["candidate_indices"].tolist(),r["coarse_cost"].tolist(),cost,obj["dtypes"],[rr[z]["trace_sha256_natural512"] for z in m.ROLES]),
                                                        diagnostic=dict(rank=1000 if append else a.bm.native_rank(cost,511,"torch.float64")))
        for _,f,_ in a.worker_facts(r,om,legacy,{(193,cap,m.qid(r)):list(range(10))for cap in (5,10)}):
            cap=f["cap"];rr=r["roles"][cap];obj=r["caps"][cap]
            expected=m.diagnostic(r["candidate_indices"].tolist(),r["coarse_cost"].numpy(),obj["natural_cost"].numpy(),
                r["target_position"],r["gamma"],[float(rr[z]["scale"][0])for z in m.ROLES],cap,obj["dtypes"],old=0 if fit else None,outside=r["cached_outside_pool_rank"],deltas=[rr[z]["delta"].numpy()for z in m.ROLES])
            m.compare_diag(f["diagnostic"],expected)
            m.fact_aliases(r,cap,f,oldrow if fit else None,list(range(10)))
            total+=1
    # Explicitly fabricated full external row: complete predefined groups, allunknown/n_a and seed union.
    f=dict(seed=193,cap=5,pool="dual",cohort="tune",query_qualification="external",
           query_identity=dict(query_order=0,variant="observed",source="synthetic",record_id="q",smiles="C",group="g"),
           fixed400_member=False,original87=False,population_source="tune",worker_probe=True,old_manifest=None,
           source="synthetic",observation_semantics="synthetic",gamma=.25,public_coverage="unknown",positive_membership="natural_miss_no_loss_append",
           diagnostic=dict(rank=None,bucket="P0_target_absent",target_in_pool=False,static_unbeatable_count=None,pairwise_bound=10),
           feature_evidence=None,loss_evidence=None,nn_evidence=[dict(status="unknown",role=z,is_target=False)for z in m.ROLES],
           structural_evidence=None,forward_matching_proxy=None,missing_measurement_mask_status="unknown",true_forward_RMSE_status="unknown")
    fs=[f,dict(copy.deepcopy(f),seed=211)]
    parts=a.population_summaries(fs,set())
    summary=dict(fixed400=400,original87=87,C10_released_old87=66,summaries=parts,equal_seed_summaries=a.equal_seed_summaries(parts))
    for part in parts:part.pop("_query_hashes",None);part.pop("_groups",None)
    m.summary_check(summary,{str(i):v for i,v in enumerate(fs)},set(range(66)))
    bad=copy.deepcopy(summary);bad["summaries"][0]["nn_availability"]["known"]=1
    rejected(lambda:m.summary_check(bad,{str(i):v for i,v in enumerate(fs)},set(range(66))))
    # Core counters only include released fixed old members, and unknowns are not negative constraints.
    f.update(old_manifest=dict(original_coarse_gap=0),query_qualification="fit",diagnostic=dict(f["diagnostic"],rank=20,bucket="S2_fail_static_open"))
    c=m.core_counts([f]);assert c["old_pair_released_but_actual_top10_failed"]==1 and c["S2_exact_collision_unknown"]==1 and c["S2_loss_evidence_unknown"]==1 and c["S2_crossquery_collision_reference_unknown"]==1 and c["S2_any_exact_collision_constraint"]==0
    return total+3

def native_jacobians():
    cases=0;ideal_false_rejections=0
    for input_dtype in (torch.float32,torch.float64):
        for logit_dtype in (torch.float32,torch.float64):
            for cap in (5,10):
                for saturated in (False,True):
                    gain=100000.;x=(.0006 if cap==5 else .0012) if saturated else .00001
                    inp=torch.zeros(512,36,dtype=input_dtype);inp[:,0]=x;inp.requires_grad_()
                    logits=(inp[:,0]*gain).to(logit_dtype)
                    positions=[0,511]
                    grad=torch.autograd.grad(logits[positions].sum(),inp,retain_graph=True)[0][positions]
                    delta=cap*torch.tanh(logits/cap)
                    local=1-torch.tanh(logits.detach()[positions]/cap).square()
                    dd=grad*local[:,None];weight=.25
                    scale=torch.full((33,),1.37,dtype=input_dtype)
                    tr=dict(raw_X=(inp[:,:33]*scale).detach(),raw_O=inp[:,33:].detach(),
                            input36=inp.detach(),logit=logits.detach(),delta=delta.detach(),scale=scale,
                            fixed_x_center=torch.zeros(33,dtype=input_dtype),fixed_o_center=torch.zeros(3,dtype=input_dtype),
                            inference_role_weight=weight,input_jacobians=dict(positions=positions,
                            dlogit_dinput36=grad.detach(),ddelta_dinput36=dd.detach(),
                            weighted_dcost_dinput36=(dd*weight*scale[0]).detach()))
                    m.trace_algebra(tr,512,cap,weight)
                    ideal=grad.numpy().astype(np.float64)*(1-np.tanh(logits.detach().numpy()[positions].astype(np.float64)/cap)**2)[:,None]
                    ideal_false_rejections+=int(not np.allclose(dd.numpy(),ideal,rtol=2e-5,atol=2e-6))
                    bad=copy.deepcopy(tr);bad["input_jacobians"]["ddelta_dinput36"][0,0]+=max(.01,abs(float(dd[0,0]))*.01)
                    rejected(lambda:m.trace_algebra(bad,512,cap,weight))
                    cases+=1
    assert ideal_false_rejections>=1
    return cases,ideal_false_rejections

def physical_dtype_prefix():
    # Integer-valued costs are equal across float32/64; torch.equal alone cannot certify dtype.
    r,e=row(append=True,fit=True)
    r["coarse_cost"]=torch.arange(512,dtype=torch.float64)
    r["loss_coarse_cost"]=torch.arange(513,dtype=torch.float32)
    for cap in (5,10):
        rr=r["roles"][cap]
        r["caps"][cap]["natural_cost"]=r["coarse_cost"]+.75*rr["no_aug"]["delta"]+.25*rr["aug"]["delta"]
        r["loss_roles"][cap]={role:trace(513,cap,None,"fit_current_frozen_loss_full513",True,r["loss_coarse_cost"],512,r["loss_strict_excluded"])for role in m.ROLES}
        for role in m.ROLES:
            m.CAP=cap
            m.loss_check(r["loss_roles"][cap][role],r["loss_coarse_cost"].numpy(),512,r["loss_strict_excluded"].numpy())
    assert torch.equal(r["loss_coarse_cost"][:512],r["coarse_cost"]) and r["loss_coarse_cost"].dtype!=r["coarse_cost"].dtype
    try:m.row_contract(r,e)
    except AssertionError as error:assert str(error)=="full513 physical dtype prefix"
    else:raise AssertionError("coarse dtype switch accepted")
    count=1
    for field in ("candidate_ids","raw_X","scale","fixed_x_center","fixed_o_center"):
        r,e=row(append=True,fit=True)
        if field=="candidate_ids":r["loss_candidate_indices"]=r["loss_candidate_indices"].to(torch.int32)
        else:
            tr=r["loss_roles"][5]["no_aug"];tr[field]=tr[field].double();tr["trace_sha256_loss513"]=m.trace_hash(tr)
        rejected(lambda:m.row_contract(r,e));count+=1
    return count

def saved_DD_weighted_chain():
    # Simulate an already-admitted local backend residual within the frozen absolute tolerance.
    # The weighted source chain is based on saved DD; re-amplifying the CPU local approximation is incorrect.
    n=512;cap=5;weight=.25;positions=[0,511]
    scale=torch.ones(33);scale[0]=1e6
    inp=torch.zeros(n,36);inp[:,0]=.0006
    grad=torch.zeros(2,36);grad[:,0]=10.
    dd=torch.zeros(2,36);dd[:,0]=1e-6
    logits=torch.full((n,),60.)
    weighted=dd*weight*scale[0]
    tr=dict(raw_X=inp[:,:33]*scale,raw_O=inp[:,33:],input36=inp,logit=logits,
            delta=cap*torch.tanh(logits/cap),scale=scale,fixed_x_center=torch.zeros(33),
            fixed_o_center=torch.zeros(3),inference_role_weight=weight,
            input_jacobians=dict(positions=positions,dlogit_dinput36=grad,ddelta_dinput36=dd,
                                weighted_dcost_dinput36=weighted))
    local=1-torch.tanh(logits[positions]/cap).square()
    assert torch.allclose(dd,grad*local[:,None],rtol=2e-5,atol=2e-6)
    assert not torch.allclose(weighted,grad*local[:,None]*weight*scale[0],rtol=2e-5,atol=2e-6)
    m.trace_algebra(tr,n,cap,weight)
    bad=copy.deepcopy(tr);bad["input_jacobians"]["weighted_dcost_dinput36"][0,0]+=.01
    rejected(lambda:m.trace_algebra(bad,n,cap,weight))
    return 1

def main():
    numeric();c=contracts();s=states();g=gates();a=aggregate_aliases();j,jf=native_jacobians();dp=physical_dtype_prefix();wc=saved_DD_weighted_chain()
    # The real entry point must gate before numeric imports or tensor decoding.
    tree=ast.parse((ROOT/"independent_closed_outputs_checker.py").read_text())
    mainnode=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=="main")
    text=ast.get_source_segment((ROOT/"independent_closed_outputs_checker.py").read_text(),mainnode)
    assert text.index("closed_output_provenance(")<text.index("import numpy")<text.index("torch.load(")
    receipt=dict(status="PASS",schema="independent_closed_checker_synthetic_v1",source_sha256=m.sha(ROOT/"independent_closed_outputs_checker.py"),
                 native_dtype_cases=16,contract_cases=c,state_cases=s,gate_cases=g,aggregate_alias_cases=a,native_jacobian_cases=j,ideal_FP64_false_rejection_cases=jf,physical_dtype_prefix_cases=dp,saved_DD_weighted_chain_cases=wc,
                 real_payload_reads=0,model_calls=0,NN_calls=0,API_calls=0)
    (ROOT/"independent_closed_outputs_checker_synthetic_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))
if __name__=="__main__":main()

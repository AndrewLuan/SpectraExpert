#!/usr/bin/env python3
"""功能概述：使用纯合成512/513张量验证 device-indexing 修订及相关索引路径。
主要职责：check_ast 核对新旧 worker 只变 native_top10 的 ID 设备迁移；run 核成本/ID
稳定排序、反向设备组合、513排除、structural排序、Python位置VJP、CPU bool mask迁移、
trace转CPU和 detached。可要求真实CUDA；没有CUDA时必须失败，不能伪报GPU通过。
代码库关系：只读 --worker 与 --baseline 两个精确SHA源码，调用其纯tensor helper；
不会调用 run_replay/capture_role/row_delta/load_checkpoint/read_feature_cache，不构造模型。
输入输出与副作用：仅导入运行时库和小源码，在 --out 写一个新JSON凭证；
所有输入由 arange、常量和整型公式现场生成，无真实cache/checkpoint/报告/测试输入。
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import torch


def need(value, reason):
    if not value:
        raise AssertionError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_ast(before, after):
    old, new = ast.parse(Path(before).read_text()), ast.parse(Path(after).read_text())
    old.body = old.body[1:]
    new.body = new.body[1:]
    f_old = next(node for node in old.body if isinstance(node,ast.FunctionDef) and node.name=="native_top10")
    f_new = next(node for node in new.body if isinstance(node,ast.FunctionDef) and node.name=="native_top10")
    expected = ast.parse("by_id = torch.argsort(ids[:512].to(costs.device), stable=True)").body[0]
    need(ast.dump(f_new.body[1],include_attributes=False)==ast.dump(expected,include_attributes=False),"device_fix_assignment")
    f_old.body[1] = expected
    need(ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False),"non_indexing_AST_change")


def patterns(n,dtype):
    x = torch.arange(n,dtype=torch.int64)
    cases = {
        "all_tied":torch.zeros(n,dtype=dtype),
        "groups":((x*17)%11-5).to(dtype),
        "descending":(-x).to(dtype),
        "sawtooth":((x*193)%509-254).to(dtype)/8,
        "native_ties":(torch.ones(n,dtype=dtype)+(x%3).to(dtype)*torch.finfo(dtype).eps),
        "signed_zero":torch.tensor([-0. if i%2 else 0. for i in range(n)],dtype=dtype),
    }
    if n==513:
        for cost in cases.values(): cost[512] = -1e12
    return cases


def structural_sort_source(worker_path):
    tree=ast.parse(Path(worker_path).read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="structural_evidence")
    loop=next(node for node in fn.body if isinstance(node,ast.For))
    body=ast.Module(body=loop.body[:2],type_ignores=[])
    return compile(ast.fix_missing_locations(body),str(worker_path),"exec")


def run(args):
    need(sha(args.worker)==args.worker_sha256 and sha(args.baseline)==args.baseline_sha256,"source_SHA")
    check_ast(args.baseline,args.worker)
    worker, baseline=load(args.worker,"_synthetic_device_worker"),load(args.baseline,"_synthetic_baseline_worker")
    cuda_available=torch.cuda.is_available()
    need(not args.require_cuda or cuda_available,"CUDA_required_no_fallback")
    devices=["cpu"]+(["cuda:0"] if args.require_cuda else [])
    structural_code=structural_sort_source(args.worker)
    counts=dict(native_top10_cases=0,old_cross_device_failure_cases=0,structural_cpu_index_cases=0,
                input_jacobian_cases=0,mask_and_loss_cases=0,trace_hash_cases=0,scoring_shape_cases=0)
    for n in (512,513):
        for idtype in (torch.int32,torch.int64):
            ids_cpu=((torch.arange(n,dtype=torch.int64)*193)%104729+1000).to(idtype)
            for dtype in (torch.float32,torch.float64):
                for label,cost_cpu in patterns(n,dtype).items():
                    expected=sorted(range(512),key=lambda p:(float(cost_cpu[p]),int(ids_cpu[p])))[:10]
                    for id_device in devices:
                        for cost_device in devices:
                            ids,cost=ids_cpu.to(id_device),cost_cpu.to(cost_device)
                            ids_before,cost_before=ids.clone(),cost.clone()
                            actual=worker.native_top10(ids,cost)
                            need(actual==expected,"native_sort:"+str((n,idtype,dtype,label,id_device,cost_device)))
                            need(torch.equal(ids,ids_before) and torch.equal(cost,cost_before),"input_mutation")
                            need(cost.dtype==dtype and len(actual)==10 and 512 not in actual,"dtype_or_513_leak")
                            counts["native_top10_cases"]+=1
                            if id_device!=cost_device:
                                try:
                                    baseline.native_top10(ids,cost)
                                except RuntimeError as error:
                                    need("device" in str(error) or "cpu" in str(error),"unexpected_old_error")
                                else:
                                    raise AssertionError("old_cross_device_bug_not_reproduced")
                                counts["old_cross_device_failure_cases"]+=1
                            namespace=dict(torch=torch,ids=ids_cpu[:512],caps={5:{"natural_cost":cost[:512]}},cap=5)
                            exec(structural_code,namespace)
                            expected_all=sorted(range(512),key=lambda p:(float(cost_cpu[p]),int(ids_cpu[p])))
                            need(namespace["order"]==expected_all,"structural_sort")
                            counts["structural_cpu_index_cases"]+=1
    for device in devices:
        for n in (512,513):
            for cap in (5,10):
                u=((torch.arange(n*36,dtype=torch.float32).reshape(n,36)%13-6)/4).to(device).requires_grad_(True)
                logit=u[:,0]*2+u[:,33]/2
                delta=cap*torch.tanh(logit/cap)
                trace=dict(raw_X=u[:,:33].detach(),raw_O=u[:,33:].detach(),input36=u,
                           logit=logit,delta=delta,scale=torch.full((33,),2.,device=device))
                positions=[0,17,255,n-1]
                jac=worker.selected_jacobians(trace,positions,.25,trace["scale"][0],cap)
                expected=torch.zeros((len(positions),36),device=device)
                expected[:,0]=2;expected[:,33]=.5
                torch.testing.assert_close(jac["dlogit_dinput36"],expected,atol=0,rtol=0)
                local=1-torch.tanh(logit.detach()[positions]/cap).square()
                torch.testing.assert_close(jac["ddelta_dinput36"],expected*local[:,None],atol=0,rtol=0)
                torch.testing.assert_close(jac["weighted_dcost_dinput36"],expected*local[:,None]*.5,atol=0,rtol=0)
                counts["input_jacobian_cases"]+=1
                before_hash=worker.tensor_trace_hashes(trace,natural_only=False)
                cpu_trace=worker.detached(trace)
                need(all(v.device.type=="cpu" for v in cpu_trace.values() if isinstance(v,torch.Tensor)),"detached_CPU")
                need(before_hash==worker.tensor_trace_hashes(cpu_trace,natural_only=False),"trace_hash_CPU_conversion")
                counts["trace_hash_cases"]+=1
                for coarse_dtype in (torch.float32,torch.float64):
                    coarse=torch.arange(n,dtype=coarse_dtype)/64
                    excluded=torch.arange(n)%5==0
                    target=n-1
                    values=delta.detach()
                    actual=worker.analytical_loss(coarse,values,logit.detach(),trace["scale"][0],
                                                  target,excluded,cap)
                    allowed=(~excluded).to(device);allowed[target]=True
                    need(actual["allowed"].device==values.device and torch.equal(actual["allowed"],allowed),"mask_device")
                    for key in ("probability","listwise_gradient_delta","anchor_gradient_delta",
                                "total_gradient_delta","total_gradient_logit"):
                        need(bool((actual[key][~allowed]==0).all()),"masked_nonzero:"+key)
                    d=values.clone().requires_grad_(True)
                    costs=coarse.to(device)/trace["scale"][0]+d
                    objective=torch.logsumexp(-costs[allowed]/.5,0)+costs[target]/.5+.1*d[allowed].square().mean()
                    grad=torch.autograd.grad(objective,d)[0]
                    torch.testing.assert_close(grad.double(),actual["total_gradient_delta"].double(),atol=2e-6,rtol=2e-5)
                    need(torch.equal(objective.detach(),actual["total_loss"]),"objective_value")
                    counts["mask_and_loss_cases"]+=1
                row=dict(candidate_indices=torch.arange(n),target_position=n-1,evaluation_pool_size=512,
                         kind=2,features=trace["raw_X"],coarse_scores=torch.arange(n,dtype=torch.float64),
                         predicted_ppm=torch.zeros((n,3),device=device),tags=torch.zeros((n,3),device=device),
                         H=torch.zeros(n,device=device),K=torch.zeros(n,device=device),
                         observation_input={"semantics":"synthetic"},group="no_scorer_identity",
                         negative_exclusion_mask=torch.zeros(n,dtype=torch.bool))
                for fit in (False,True):
                    scorer=worker.scoring_input(row,fit)
                    need(len(scorer["features"])==(n if fit else 512),"scoring_shape")
                    need(not {"candidate_indices","target_position","group","negative_exclusion_mask"}&scorer.keys(),"scorer_metadata")
                    counts["scoring_shape_cases"]+=1
    need(sha(args.worker)==args.worker_sha256 and sha(args.baseline)==args.baseline_sha256,"source_changed")
    receipt=dict(schema="runtime_device_indexing_synthetic_v1",status="PASS",
                 worker_sha256=args.worker_sha256,baseline_sha256=args.baseline_sha256,
                 synthetic_script_sha256=sha(__file__),torch_version=torch.__version__,
                 cuda_available=cuda_available,cuda_exercised=args.require_cuda,devices_exercised=devices,
                 counts=counts,real_model_calls=0,checkpoint_loads=0,cache_reads=0,training_steps=0,
                 external_API_calls=0,nn_success=False,
                 claim="Fabricated tensor indexing/device contract only; no scientific ranker/cache forward.")
    output=Path(args.out)
    need(not output.exists() and output.parent.is_dir(),"new_output_required")
    output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(status="PASS",counts=counts,cuda_exercised=args.require_cuda,receipt=str(output))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",required=True);parser.add_argument("--worker-sha256",required=True)
    parser.add_argument("--baseline",required=True);parser.add_argument("--baseline-sha256",required=True)
    parser.add_argument("--out",required=True);parser.add_argument("--require-cuda",action="store_true")
    run(parser.parse_args())


if __name__=="__main__":
    main()

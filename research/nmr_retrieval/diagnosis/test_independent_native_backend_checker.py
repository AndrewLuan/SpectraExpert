"""功能概述：用合成保存量检查独立核验器的原dtype代数、较严trace容差和逐文件传输拒绝行为。
所有fixture均在内存/临时目录生成；不导入作者分析器，不读取实验数据、模型或权重。
"""
import json
import ast
import copy
import inspect
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
import independent_native_backend_checker as c


class TestAlgebra(unittest.TestCase):
    def setUp(self):
        c.np=np;c.torch=torch;c.CAP=5;c._DEVICE=torch.device("cpu")

    def test_native_loss_support_and_saturation(self):
        for coarse_dtype in (torch.float32,torch.float64):
            for delta_dtype in (torch.float32,torch.float64):
                delta=torch.tensor([4.,-1.,3.,-.5],dtype=delta_dtype)
                logit=torch.tensor([40.,-2.,1.,-40.],dtype=delta_dtype)
                scale=torch.full((33,),3.,dtype=delta_dtype)
                coarse=torch.tensor([1e8,1e8+8,1e8+16,1e8+24],dtype=coarse_dtype)
                allowed=torch.tensor([True,False,True,True]);cost=coarse/scale[0]+delta
                p=torch.zeros_like(cost);p[allowed]=torch.softmax(-cost[allowed]/.5,0)
                ind=torch.zeros_like(p);ind[0]=1
                gl=torch.where(allowed,(ind-p)/.5,torch.zeros_like(p))
                ga=torch.where(allowed,2*.1*delta/3,torch.zeros_like(delta));local=1-torch.tanh(logit/5).square()
                ll=torch.logsumexp(-cost[allowed]/.5,0)+cost[0]/.5;anchor=.1*delta[allowed].square().mean()
                loss=dict(actual_normalized_costs=cost,probability=p,listwise_gradient_delta=gl,anchor_gradient_delta=ga,
                    total_gradient_delta=gl+ga,tanh_local_derivative=local,listwise_gradient_logit=gl*local,
                    anchor_gradient_logit=ga*local,total_gradient_logit=(gl+ga)*local,listwise_loss=ll,
                    anchor_loss=anchor,total_loss=ll+anchor,allowed=allowed,support_count=3,
                    normalized_cost_dtype=str(cost.dtype),probability_dtype=str(p.dtype))
                tr=dict(delta=delta,logit=logit,scale=scale,actual_train_loss=loss)
                c.loss_check(tr,coarse.numpy(),0,np.asarray([True,True,False,False]))
                loss["tanh_local_derivative"]=local.to(torch.float64 if local.dtype==torch.float32 else torch.float32)
                with self.assertRaisesRegex(AssertionError,"native dtype/shape"):
                    c.loss_check(tr,coarse.numpy(),0,np.asarray([True,True,False,False]))

    def fixture_trace(self):
        n=512;x=torch.arange(n*33,dtype=torch.float32).reshape(n,33)/100
        z=torch.linspace(-40,40,n);scale=torch.full((33,),3.);pos=[2,250,511]
        dx=torch.arange(108,dtype=torch.float32).reshape(3,36)/7;dd=dx*(1-torch.tanh(z[pos]/5).square())[:,None]
        return dict(raw_X=x,raw_O=torch.zeros(n,3),input36=torch.cat((((x-1)/scale).clamp(-20,20),torch.zeros(n,3)),1),
                    logit=z,delta=5*torch.tanh(z/5),scale=scale,fixed_x_center=torch.ones(33),fixed_o_center=torch.zeros(3),
                    inference_role_weight=.75,input_jacobians=dict(positions=pos,dlogit_dinput36=dx,
                    ddelta_dinput36=dd,weighted_dcost_dinput36=dd*.75*scale[0]))

    def test_trace_weighted_saved_DD_and_strict_center(self):
        for mutation,reason in (("center","centered36"),("weighted","weighted-cost"),("dtype","dtype/shape")):
            tr=self.fixture_trace();c.trace_algebra(tr,512,5,.75)
            if mutation=="center":tr["input36"][511,0]+=1e-3
            elif mutation=="weighted":tr["input_jacobians"]["weighted_dcost_dinput36"][1,0]+=1
            else:tr["input_jacobians"]["ddelta_dinput36"]=tr["input_jacobians"]["ddelta_dinput36"].double()
            with self.assertRaisesRegex(AssertionError,reason):c.trace_algebra(tr,512,5,.75)

    def test_center_tolerance_retains_2e6(self):
        with self.assertRaisesRegex(AssertionError,"native algebra"):
            c.native_close(torch.tensor([1.+1e-5]),torch.ones(1),"center",rtol=2e-6)


class TestTransport(unittest.TestCase):
    def setUp(self):c._MAPPING.clear();c._BOOTSTRAP.clear();c._CONSUMED.clear()

    def test_exact_mapping_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();p=root/"raw.json";p.write_text('{"x":1}')
            k="/logical/train/raw.json";c._MAPPING[k]=dict(physical_path=str(p),sha256=c.raw_sha(p),bytes=p.stat().st_size)
            self.assertEqual(c.read(dict(path=k,sha256=c.raw_sha(p))),dict(x=1))
            for bad in ("/logical/train/sub/../raw.json","/logical//train/raw.json","/logical/train/./raw.json",str(p)):
                with self.assertRaises(AssertionError):c.io_path(bad)
            p.write_text('{"x":2}')
            with self.assertRaisesRegex(AssertionError,"bytes/SHA"):c.io_path(k)

    def test_symlink_after_use_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t).resolve()/"a.json";p.write_text("1")
            k="/logical/train/a.json";c._MAPPING[k]=dict(physical_path=str(p),sha256=c.raw_sha(p),bytes=1)
            c.io_path(k);other=p.with_name("b.json");p.rename(other);p.symlink_to(other)
            with self.assertRaisesRegex(AssertionError,"physical alias"):c.io_path(k)

    def test_bootstrap_bool_size_and_nonhex_hash_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t).resolve()/"blank.log";p.write_bytes(b"")
            row=dict(physical_path=str(p),sha256=c.raw_sha(p),bytes=False)
            c._BOOTSTRAP[str(p)]=row
            with self.assertRaisesRegex(AssertionError,"transport row types"):c.io_path(p)
            row["bytes"]=0;row["sha256"]="z"*64
            with self.assertRaisesRegex(AssertionError,"transport row types"):c.io_path(p)

    def test_map_symlink_rejected_at_finish_without_prior_map_use(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t).resolve()/"map.json";p.write_text("{}")
            c._MAP_PIN=(str(p),c.raw_sha(p));c._BOOTSTRAP[str(p)]=dict(physical_path=str(p),sha256=c.raw_sha(p),bytes=2)
            other=p.with_name("same.json");p.rename(other);p.symlink_to(other)
            with self.assertRaisesRegex(AssertionError,"physical alias"):c.finish_transport()

    def test_actual_author_ledger_guards_reject_incomplete_and_wrong_hashes(self):
        # 直接执行analysis_chain实际ledger守卫的AST片段，read只返回合成元数据。
        fn=ast.parse(inspect.getsource(c.analysis_chain)).body[0]
        start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id=="request" for t in n.targets))
        end=next(i for i,n in enumerate(fn.body[start:],start) if isinstance(n,ast.For))
        code=compile(ast.Module(body=fn.body[start:end],type_ignores=[]),"<actual-checker-ledger-guards>","exec")
        def ref(k):return dict(path="/logical/train/"+k,sha256="a"*64)
        r={k:ref(k) for k in ("fixed_manifest","query_manifest","qualification_contract","qualification_bindings",
                            "backend_decision","backend_actual_review")}
        r.update(legacy_predictions=[ref("legacy"+str(i)) for i in range(20)],
                 worker_outputs=[dict(summary=ref("summary"),index=ref("index"))],
                 completed_output_provenance={str(i):ref("prov"+str(i)) for i in range(7)},
                 analysis_source_sha256={"/source/"+str(i):"b"*64 for i in range(3)},
                 transport_map=ref("map"),source_git_commit="c"*40)
        r["completed_output_provenance"]["sealed_manifest"]=r["completed_output_provenance"].pop("6")
        r["completed_output_provenance"]["source_stage_root"]="/logical/train/stage"
        sealed=dict(files={"worker/tensor_shards/"+str(i)+".pt":dict(sha256="d"*64) for i in range(1652)})
        direct=[r[k] for k in ("fixed_manifest","query_manifest","qualification_contract","qualification_bindings",
                              "backend_decision","backend_actual_review")]+r["legacy_predictions"]+list(r["worker_outputs"][0].values())
        direct += [x for x in r["completed_output_provenance"].values() if isinstance(x,dict)]
        hashes={x["path"]:x["sha256"] for x in direct}
        hashes.update({"/logical/train/stage/"+k:x["sha256"] for k,x in sealed["files"].items()})
        rows=[dict(logical_path=k,physical_path="/physical/"+str(i),sha256=v,bytes=1) for i,(k,v) in enumerate(hashes.items())]
        ledger={x["logical_path"]:dict(path=x["physical_path"],sha256=x["sha256"],bytes=1) for x in rows}
        args=types.SimpleNamespace(request="/physical/request",request_sha256="e"*64)
        refs=dict(input_admission=dict(path="/physical/A",sha256="f"*64))
        A=dict(reviewed_source_sha256=r["analysis_source_sha256"],transport_map_sha256=r["transport_map"]["sha256"])
        good=dict(transport_map_sha256="a"*64,algebra_device="cuda:0",runtime="original_JJ",source_git_commit="c"*40,
                  input_sha256=hashes,physical_input_transport=ledger,
                  bootstrap_sha256={args.request:args.request_sha256,refs["input_admission"]["path"]:"f"*64,**r["analysis_source_sha256"]})
        def read(x):
            if x["path"]==args.request:return r
            if x==r["transport_map"]:return dict(entries=rows)
            return sealed
        def argument(name):return r["transport_map"]["path" if name=="--transport-map" else "sha256"]
        env=dict(read=read,args=args,refs=refs,A=A,argument=argument,Path=Path,exact_logical=c.exact_logical,need=c.need)
        exec(code,dict(env,result=copy.deepcopy(good)))
        for mutation in ("empty","hash","bootstrap"):
            bad=copy.deepcopy(good)
            if mutation=="empty":bad["input_sha256"]={};bad["physical_input_transport"]={}
            elif mutation=="hash":bad["input_sha256"][next(iter(hashes))]="0"*64
            else:bad["bootstrap_sha256"]["/physical/extra"]="1"*64
            with self.assertRaisesRegex(AssertionError,"author .* (transport|bootstrap)"):
                exec(code,dict(env,result=bad))


if __name__=="__main__":unittest.main()

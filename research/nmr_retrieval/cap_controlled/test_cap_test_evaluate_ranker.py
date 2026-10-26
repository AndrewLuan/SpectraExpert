"""功能概述：sealed新路径的metadata准入与真实cap loader/人工tensor端到端合成测试。
Fixture仅使用人工512候选、6query和临时JSON；正式1371数量在生产常量固定，合成测试
临时patch数量以核完整入口与失败保留。禁真实test/cache/旧结果读取，不连接远端。
"""
import ast
import hashlib
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import cap_test_evaluation_common as c
import cap_test_evaluate_ranker as e
from test_cap_evaluation_entry import EntryFixture
from test_evaluate_cap_ranker import row,models,FakeFixed

def h(path,*_):return c.sha(path,True)

class SealedFixture:
    counts={'observed':4,'frequency_crop':1,'peak_dropout':1}
    def __init__(self,root,config='cap5_single',seed=193):
        self.root=Path(root).resolve();self.config=config;self.seed=seed
        (self.root/'synthetic_data').mkdir()
        self.f=EntryFixture(self.root/'synthetic_data',cap=10 if config.startswith('cap10_') else 5,seed=seed,cohort='test',pool='single',observed_count=4)
        self.module=self.f.module;self.rt=self.f.rt;self.protocol=self.f.protocol;self.order=self.f.order
        self.data=self.f.data
        # 独立按原canonical JSON规则生成真实人工行的内容SHA，非自报占位值。
        canonical=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.identity=dict(query_sha256=canonical([{k:r[k] for k in c.IDENTITY} for r in self.data['rows'] if r['variant']=='observed']),identity_content_sha256=canonical([{k:r[k] for k in c.IDENTITY} for r in self.data['rows']]),variant_mask_content_sha256=canonical([(tuple(r[k] for k in c.IDENTITY[:-1]),r['mask_content_sha256']) for r in self.data['rows']]))
        self.data['manifest']['appended_training_positives']=0
        self.data['manifest'].update(self.identity)
        self.sources={str(Path(e.__file__).resolve()):h(e.__file__),str(Path(c.__file__).resolve()):h(c.__file__),**json.loads(self.f.bindings.read_text())['source_sha256']}
        self.inputs=dict(self.sources);self.caches={};self.entries=[]
        self.roles={}
        for role,r in self.f.roles.items():
            ref=dict(path=r['checkpoint'],sha256=r['checkpoint_sha256'],epoch=64,alpha=1.,**{k:v for k,v in r.items() if k not in ('checkpoint','checkpoint_sha256')})
            self.roles[role]=ref
            for k in ('path','centers','center_rows','producer_summary'):
                self.inputs[ref[k]]=h(ref[k])
            exitpath=str(Path(ref['path']).parent/'EXIT_CODE');self.inputs[exitpath]=h(exitpath)
        for seedvalue in c.SEEDS:
            self.caches[str(seedvalue)]={'test':{}}
            for pool in c.POOLS:
                folder=self.root/f'cache_s{seedvalue}_{pool}'/'test';folder.mkdir(parents=True)
                cache=folder/'features.pt';cache.write_bytes(('synthetic payload '+str(seedvalue)+pool).encode())
                manifest={**self.data['manifest'],'recall':pool}
                producer=folder/'summary.json';c.write(producer,{**manifest,'feature_cache_sha256':h(cache)})
                exitpath=folder/'EXIT_CODE';exitpath.write_text('0\n')
                reg=dict(path=str(cache),sha256=h(cache),manifest=manifest,
                  producer=dict(path=str(producer),sha256=h(producer),exit_path=str(exitpath)))
                self.caches[str(seedvalue)]['test'][pool]=reg
                for p in (cache,producer,exitpath):self.inputs[str(p)]=h(p)
        for cfg in c.CONFIGS:
            for seedvalue in c.SEEDS:
                pool=cfg.rsplit('_',1)[1];reg=self.caches[str(seedvalue)]['test'][pool]
                self.entries.append(dict(config=cfg,seed=seedvalue,pool=pool,cache=reg['path'],cache_sha256=reg['sha256'],checkpoints=copy.deepcopy(self.roles),execution='evaluate',job_id=cfg+str(seedvalue),job_out=str(self.root/'outputs'/cfg/str(seedvalue))))
        self.bindings=self.root/'test_bindings.json';self.plan=self.root/'plan.json';self.selection=self.root/'selection.json'
        self.val=self.root/'validation_summary.json';self.vreview=self.root/'validation_review.json';self.planreview=self.root/'plan_review.json'
        self.frozen=self.root/'frozen.json';self.review=self.root/'final_review.json'
        self.binding=dict(schema='cap10_test_bindings_20261024',source_checkout=str(self.f.source),entries=self.entries,
          caches=self.caches,source_sha256=self.sources,input_sha256=self.inputs,cap_module=dict(path=str(Path(self.module.__file__).resolve()),sha256=h(self.module.__file__)),
          fixed_module=dict(path=str(self.f.fixed),sha256=h(self.f.fixed)))
        c.write(self.bindings,self.binding)
        c.write(self.plan,dict(entries=self.entries,preselected=c.BASELINE,selection_locked=True,promotion=False,selection_performed=False))
        c.write(self.val,dict(status='PASS',test_read=False,promotion=False,preselected=c.BASELINE))
        c.write(self.selection,dict(preselected=c.BASELINE,selection_locked=True))
        c.write(self.vreview,dict(status='PASS',independent=True,reviewer='synthetic independent',scope='cap_validation_results_and_implementation',reviewed_sha256={str(self.val):h(self.val)}))
        c.write(self.planreview,dict(status='PASS',independent=True,reviewer='synthetic independent',scope='cap_test_plan_implementation',reviewed_sha256={str(self.plan):h(self.plan)}))
        self.freeze=dict(schema='cap10_sealed_selection_20261024',selection_locked=True,promotion=False,preselected=c.BASELINE,
          entries=self.entries,fixed_test_identity=self.identity,input_sha256=self.inputs,source_sha256=self.sources)
        for key,path in [('test_plan',self.plan),('validation_summary',self.val),('validation_selection',self.selection),('validation_review',self.vreview),('freeze_review',self.planreview)]:self.freeze[key]=dict(path=str(path),sha256=h(path))
        c.write(self.frozen,self.freeze);self.repin_review()
        entry=next(x for x in self.entries if (x['config'],x['seed'])==(config,seed))
        self.out=Path(entry['job_out']);self.cache=Path(entry['cache'])
        self.args=SimpleNamespace(source=str(self.f.source),protocol=str(self.protocol),bindings=str(self.bindings),frozen_selection=str(self.frozen),freeze_review=str(self.review),cache=str(self.cache),out=str(self.out),config=config,seed=seed,device='cpu',threads=1)
    def repin_review(self):
        required={**self.inputs,**self.sources}
        for p in [Path(e.__file__).resolve(),Path(c.__file__).resolve(),self.protocol,self.bindings,self.plan,self.val,self.selection,self.vreview,self.planreview,self.frozen]:required[str(p)]=h(p)
        c.write(self.review,dict(status='PASS',independent=True,reviewer='synthetic final reviewer',scope='cap10_test_freeze_independent_review',reviewed_sha256=required))
    def make_native_baseline(self):
        """执行旧fixed loader真实函数AST，仅替换模型构造为人工CapRanker和临时payload。"""
        source=next(p for p in (Path(__file__).parent.parent/'orchestrator-261023-1335/h_pool_fixed_center_ranker.py',Path(__file__).parent.parent/'nmr_pool_ranking_20261023/h_pool_fixed_center_ranker.py') if p.is_file())
        tree=ast.parse(source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_checkpoint')
        budget=dict(epochs=64,samples=8192,batch_size=32,anchor=.1,lr=.001)
        namespace=dict(torch=torch,Path=Path,json=json,SCHEMA='nmr_pool_fixed_center_20261023',require=c.need,
          input_fingerprints=lambda paths:{str(Path(p).resolve()):h(p,True) for p in paths},
          FixedCenterRanker=lambda sigmas,x,o:self.module.CapRanker(sigmas,x,o,5,self.rt))
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
        native_roles={}
        for j,(role,ref) in enumerate(self.roles.items()):
            folder=self.root/'native_weights'/role;folder.mkdir(parents=True)
            for name in ('centers.json','center_rows.jsonl'):(folder/name).write_bytes((Path(ref['path']).parent/name).read_bytes())
            saved=torch.load(ref['path'],map_location='cpu',weights_only=False)
            saved.update(schema=namespace['SCHEMA'],epoch=7+j,alpha=.25*(j+1));saved['args'].update(budget)
            checkpoint=folder/'best.pt';torch.save(saved,checkpoint)
            producer=folder/'summary.json';c.write(producer,dict(best=dict(epoch=saved['epoch'],alpha=saved['alpha']),center_estimate_sha256=saved['center_estimate_sha256']))
            exitpath=folder/'EXIT_CODE';exitpath.write_text('0\n')
            new=dict(path=str(checkpoint),sha256=h(checkpoint,True),epoch=saved['epoch'],alpha=saved['alpha'],
              centers=str(folder/'centers.json'),centers_sha256=h(folder/'centers.json',True),center_rows=str(folder/'center_rows.jsonl'),
              center_rows_sha256=h(folder/'center_rows.jsonl',True),producer_summary=str(producer),producer_summary_sha256=h(producer,True))
            native_roles[role]=new
            for path in (checkpoint,producer,exitpath,folder/'centers.json',folder/'center_rows.jsonl'):self.inputs[str(path)]=h(path,True)
        self.rt.fixed=SimpleNamespace(SCHEMA=namespace['SCHEMA'],BUDGET=budget,load_checkpoint=namespace['load_checkpoint'],FixedCenterRanker=FakeFixed)
        for entry in self.entries:
            if entry['config'].startswith('fixed_'):entry['checkpoints']=copy.deepcopy(native_roles)
        c.write(self.bindings,self.binding);plan=json.loads(self.plan.read_text());plan['entries']=self.entries;c.write(self.plan,plan)
        self.freeze['test_plan']['sha256']=h(self.plan)
        c.write(self.planreview,dict(status='PASS',independent=True,reviewer='synthetic independent',scope='cap_test_plan_implementation',reviewed_sha256={str(self.plan):h(self.plan)}))
        self.freeze['freeze_review']['sha256']=h(self.planreview);c.write(self.frozen,self.freeze);self.repin_review()
    def patch_context(self):
        return patch.multiple(e,ORDER_SHA=self.order,load_module=lambda p:self.module)
    def gate(self):
        with patch.object(c,'ORDER_SHA',self.order),patch.dict(c.TEST_COUNTS,self.counts,clear=True):
            return c.sealed_gate(self.args,[e.__file__,c.__file__])
    def run(self):
        reg=self.caches[str(self.seed)]['test'][self.config.rsplit('_',1)[1]]
        def reader(path,known_sha=None):
            value=copy.deepcopy(self.data);value['manifest']=copy.deepcopy(reg['manifest']);return value,known_sha
        self.rt.read_feature_cache=reader
        argv=[str(Path(e.__file__).resolve())]
        for k,v in vars(self.args).items():argv.extend(['--'+k.replace('_','-'),str(v)])
        with self.patch_context(),patch.object(c,'ORDER_SHA',self.order),patch.dict(c.TEST_COUNTS,self.counts,clear=True),patch.object(self.module,'configure_runtime',return_value=self.rt),patch('sys.argv',argv):
            e.main()
        return json.loads((self.out/'summary.json').read_text())

class ColdIsolationTests(unittest.TestCase):
    def cold(self,root,forbidden=None):
        f=SealedFixture(root)
        # 只有当前cell缓存人工tensor化；真实wrapper冷进程仅准入该cache+两weights。
        reg=f.caches[str(f.seed)]['test']['single'];data=copy.deepcopy(f.data);data['manifest']=reg['manifest']
        torch.save(data,f.cache);reg['sha256']=h(f.cache);f.inputs[str(f.cache)]=reg['sha256']
        producer=Path(reg['producer']['path']);c.write(producer,{**reg['manifest'],'feature_cache_sha256':reg['sha256']})
        reg['producer']['sha256']=h(producer);f.inputs[str(producer)]=h(producer)
        for entry in f.entries:
            if entry['cache']==str(f.cache):entry['cache_sha256']=reg['sha256']
        c.write(f.bindings,f.binding);plan=json.loads(f.plan.read_text());plan['entries']=f.entries;c.write(f.plan,plan);f.freeze['test_plan']['sha256']=h(f.plan)
        c.write(f.planreview,dict(status='PASS',independent=True,reviewer='synthetic',scope='cap_test_plan',reviewed_sha256={str(f.plan):h(f.plan)}));f.freeze['freeze_review']['sha256']=h(f.planreview)
        c.write(f.frozen,f.freeze);f.repin_review()
        operator=f.root/'isolation_entry.py';operator.write_bytes((Path(__file__).parent/'isolation_entry.py').read_bytes())
        self.assertEqual(h(operator),h(Path(__file__).parent/'isolation_entry.py'))
        runner=f.root/'cold_runner.py'
        runner.write_text("""import sys,json,os
from pathlib import Path
sys.path.insert(0,__ROOT_SOURCE__)
import torch
from types import SimpleNamespace
import cap_test_evaluation_common as c
import cap_test_evaluate_ranker as e
from test_evaluate_cap_ranker import FakeFixed,cap_module
c.TEST_COUNTS.clear();c.TEST_COUNTS.update(__COUNTS__)
c.ORDER_SHA=__ORDER__;e.ORDER_SHA=__ORDER__
module=cap_module()
rt=SimpleNamespace(fixed=SimpleNamespace(FixedCenterRanker=FakeFixed),base=SimpleNamespace(validate_row=lambda *x:None,check_science=lambda *x:None),source_paths=lambda:[Path(module.__file__).resolve(),Path(__FIXED_SOURCE__)],read_feature_cache=lambda path,known_sha=None:(torch.load(path,map_location='cpu',weights_only=False),known_sha))
module.configure_runtime=lambda *x:rt
e.load_module=lambda path:module
if os.environ.get('SYNTHETIC_FORBIDDEN_READ'):
    Path(os.environ['SYNTHETIC_FORBIDDEN_READ']).read_bytes()
e.main()
""".replace('__ROOT_SOURCE__',repr(str(Path(e.__file__).parent))).replace('__COUNTS__',repr(f.counts)).replace('__ORDER__',repr(f.order)).replace('__FIXED_SOURCE__',repr(str(f.f.fixed))))
        # 上面替换仅处理专用占位常量，不改变被审生产源码。
        command=[str(runner)]
        for k,v in vars(f.args).items():command.extend(['--'+k.replace('_','-'),str(v)])
        allowed=[str(f.cache),*[r['path'] for r in f.roles.values()],str(producer),reg['producer']['exit_path']]
        access=f.root/'access_logs/cold.json'
        argv=[sys.executable,'-B',str(operator),'--bindings',str(f.bindings),'--phase','test','--access-log',str(access),'--output-root',str(f.out)]
        for path in allowed:argv.extend(['--allow-read',path])
        argv.extend(['--',*command]);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        if forbidden:env['SYNTHETIC_FORBIDDEN_READ']=str(f.caches['211']['test']['dual']['path'] if forbidden=='cache' else f.root/'old/predictions.jsonl')
        if forbidden=='pred':
            q=f.root/'old/predictions.jsonl';q.parent.mkdir();q.write_text('synthetic untouched sentinel')
        proc=subprocess.run(argv,capture_output=True,text=True,env=env,timeout=60)
        return f,proc,json.loads(access.read_text()),allowed
    def test_real_cold_isolation_five_exact_paths_three_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,proc,access,allowed=self.cold(tmp)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            self.assertEqual(access['internal_exit'],0);self.assertEqual(access['rejected_paths'],[])
            self.assertEqual(len(allowed),5);self.assertEqual(sum(Path(p).suffix=='.pt' for p in allowed),3)
            self.assertTrue(all(access['recorded_reads'].get(p,0)>0 for p in allowed))
            self.assertTrue(json.loads((f.out/'summary.json').read_text())['test_read'])
    def test_real_guard_refuses_other_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,proc,access,_=self.cold(tmp,'cache');self.assertNotEqual(proc.returncode,0)
            self.assertEqual(access['internal_exit'],1);self.assertTrue(access['rejected_paths'])
            self.assertFalse((f.out/'predictions.jsonl').exists())
    def test_real_guard_refuses_unregistered_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,proc,access,_=self.cold(tmp,'pred');self.assertNotEqual(proc.returncode,0)
            self.assertEqual(access['internal_exit'],1);self.assertTrue(access['rejected_paths'])
            self.assertFalse((f.out/'predictions.jsonl').exists())

class GateTests(unittest.TestCase):
    def fixture(self,tmp):return SealedFixture(tmp)
    def test_complete_gate_checks_metadata_without_opening_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);original=Path.open;opened=[]
            def audited(path,*args,**kwargs):
                if path.suffix=='.pt':opened.append(str(path));raise AssertionError('payload before gate')
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',audited):pins,*_=f.gate()
            self.assertEqual(opened,[]);self.assertNotIn(str(f.cache),pins.values)
    def test_result_gate_without_fake_cell_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);args=copy.copy(f.args)
            for key in ('config','seed','cache'):delattr(args,key)
            with patch.object(c,'ORDER_SHA',f.order),patch.dict(c.TEST_COUNTS,f.counts,clear=True):
                result=c.sealed_gate(args,[e.__file__,c.__file__],require_cell=False)
            self.assertIsNone(result[-1]);self.assertEqual(len(result[3]['entries']),12)
    def test_final_review_must_cover_actual_frozen_and_tool_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);v=json.loads(f.review.read_text());del v['reviewed_sha256'][str(f.frozen)];c.write(f.review,v)
            with self.assertRaisesRegex(ValueError,'complete_exact_SHA'):f.gate()
    def test_reselection_refused_before_any_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);f.freeze['preselected']='cap5_dual';c.write(f.frozen,f.freeze);f.repin_review()
            with self.assertRaisesRegex(ValueError,'registered_frozen_preselected'):f.gate()
    def test_validation_lock_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);c.write(f.selection,dict(preselected='cap10_dual',selection_locked=True));f.freeze['validation_selection']['sha256']=h(f.selection);c.write(f.frozen,f.freeze);f.repin_review()
            with self.assertRaisesRegex(ValueError,'validation_preselected'):f.gate()
    def test_missing_cell_even_if_all_three_documents_rewritten_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);f.freeze['entries'].pop();c.write(f.frozen,f.freeze);f.binding['entries']=f.freeze['entries'];c.write(f.bindings,f.binding)
            p=json.loads(f.plan.read_text());p['entries']=f.freeze['entries'];c.write(f.plan,p);f.freeze['test_plan']['sha256']=h(f.plan);c.write(f.frozen,f.freeze);f.repin_review()
            with self.assertRaisesRegex(ValueError,'twelve_logical'):f.gate()
    def test_single_dual_weight_refs_cannot_diverge(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);entry=next(x for x in f.entries if x['config']=='fixed_dual' and x['seed']==193)
            entry['checkpoints']['aug']['alpha']=.5
            c.write(f.bindings,f.binding);p=json.loads(f.plan.read_text());p['entries']=f.entries;c.write(f.plan,p);f.freeze['test_plan']['sha256']=h(f.plan);c.write(f.frozen,f.freeze);f.repin_review()
            with self.assertRaisesRegex(ValueError,'same_weights_across_single_dual'):f.gate()
    def test_partial_baseline_reuse_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);f.entries[0]['execution']='reuse';c.write(f.bindings,f.binding);p=json.loads(f.plan.read_text());p['entries']=f.entries;c.write(f.plan,p);f.freeze['test_plan']['sha256']=h(f.plan);c.write(f.frozen,f.freeze);f.repin_review()
            with self.assertRaisesRegex(ValueError,'all_four_baselines'):f.gate()
    def test_wrong_cache_identity_even_if_registry_frozen_rewritten_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);f.binding['caches']['211']['test']['dual']['manifest']['variant_mask_content_sha256']='f'*64;c.write(f.bindings,f.binding);f.repin_review()
            with self.assertRaisesRegex(ValueError,'cache_registry_test_identity'):f.gate()
    def test_cli_cache_and_out_exact_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);f.args.out=str(f.root/'different')
            with self.assertRaisesRegex(ValueError,'CLI_out'):f.gate()
    def test_false_final_review_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);v=json.loads(f.review.read_text());v['independent']=False;c.write(f.review,v)
            with self.assertRaisesRegex(ValueError,'independent_PASS'):f.gate()

class EvaluationTests(unittest.TestCase):
    def test_actual_new_cap_loader_through_sealed_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SealedFixture(tmp);s=f.run();self.assertEqual(s['records'],6)
            self.assertTrue(s['test_read']);self.assertFalse(s['selection_performed']);self.assertFalse(s['promotion'])
            self.assertEqual(s['model_state_before'],s['model_state_after']);self.assertEqual((f.out/'EXIT_CODE').read_text(),'0\n')
            rows=[json.loads(x) for x in (f.out/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual({tuple(x['query_identity']) for x in rows},{tuple(c.IDENTITY)})
            self.assertTrue(all(len(x['natural_candidate_costs'])==512 for x in rows))
    def test_actual_cap10_seed211_loader_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SealedFixture(tmp,'cap10_dual',211);s=f.run()
            self.assertEqual((s['cap'],s['seed'],s['pool']),(10,211,'dual'));self.assertEqual(s['records'],6)
    def test_native_fixed_loader_original_best_epoch_and_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SealedFixture(tmp,'fixed_single');f.make_native_baseline();s=f.run()
            self.assertEqual(s['config'],'fixed_single');self.assertEqual(s['records'],6)
            refs=f.entries[0]['checkpoints'];self.assertEqual([refs[k]['alpha'] for k in ('no_aug','aug')],[.25,.5])
            rows=[json.loads(x) for x in (f.out/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['role_absolute_correction_bounds'],[1.25,2.5])
    def test_payload_changed_after_gate_is_rejected_and_failure_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SealedFixture(tmp);f.cache.write_text('changed')
            with self.assertRaisesRegex(ValueError,'input_sha'):f.run()
            self.assertEqual((f.out/'EXIT_CODE').read_text(),'1\n');self.assertTrue((f.out/'failure.json').exists())
    def test_native_alpha_preserved_with_exact_float32_order(self):
        _,ms=models(5);r=row();n=e.natural_row(r)
        ms[0][2]['alpha']=.25;ms[1][2]['alpha']=.5
        actual=e.score_natural(ms,n,.75,'cpu',5)
        d=[saved['alpha']*model.row_delta(n,scale,'cpu')*scale[0] for model,scale,saved in ms]
        expected=n['coarse_scores']+.25*d[0]+.75*d[1]
        self.assertTrue(torch.equal(expected,actual['cost']))
        self.assertEqual(actual['role_bounds'],[1.25,2.5])
    def test_natural512_no_label_or_append_influence(self):
        _,ms=models(10);r=row();a=e.score_natural(ms,e.natural_row(r),.25,'cpu',10)
        r.update(source='other',record_id='other',target_position=512,negative_exclusion_mask=torch.ones(513,dtype=torch.bool));r['features'][-1]=1e15
        b=e.score_natural(ms,e.natural_row(r),.25,'cpu',10);self.assertTrue(torch.equal(a['cost'],b['cost']))
    def test_outside_native_fallback_retains_full_rank(self):
        _,ms=models(5);r=row();r.update(target_position=512,target_in_coarse_pool=False,outside_pool_rank=1005)
        result=e.outcome(r,e.score_natural(ms,e.natural_row(r),.25,'cpu',5))
        self.assertEqual(result['rank'],1005);self.assertEqual(result['RR'],1/1005);self.assertIsNone(result['in_pool_target_rank'])
if __name__=='__main__':unittest.main()

"""功能概述：两阶段test元数据工具的作者合成测试，所有权重/缓存只是不存在路径和SHA。
人工Reader保存JSON/源码/EXIT字节，独立构造已锁val、旧缓存producer登记、12角色与
plan独审。覆盖无锁首读边界、seed/forward/identity、all12fresh、source闭包和无环finalreview。
无目录扫描、真实cache/pred读取、模型调用、SSH或科学派发。
"""
import copy
import json
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
import prepare_cap_test_metadata as m


class MemoryReader:
    def __init__(self):self.data={};self.reads={}
    def add(self,path,value):
        path=str(path);raw=value if type(value) is bytes else m.encoded(value);self.data[path]=raw
        return dict(path=path,sha256=m.sha(raw))
    def raw(self,path,wanted,allow_registered_test_metadata=False):
        if Path(path).suffix in ('.pt','.pth','.ckpt') or 'prediction' in Path(path).name:raise AssertionError('scientific_payload_read')
        raw=self.data[path];m.need(m.sha(raw)==wanted,'actual_metadata_or_source_SHA');self.reads[path]=wanted;return raw
    def ref(self,ref,allow_registered_test_metadata=False):return json.loads(self.raw(ref['path'],ref['sha256'],allow_registered_test_metadata))


class Fixture:
    def __init__(self,root='/synthetic/cap10'):
        self.root=Path(root);r=self.root;self.reader=MemoryReader();read=self.reader;self.inputs={};self.refs={}
        def source(path):return read.add(path,b'# artificial source only\n')
        leaf={}
        for name in m.RUNTIME:
            ref=source(r/name if name in ('cap_ranker.py','train_cap_ranker.py') else r/'runtime'/name);leaf[ref['path']]=ref['sha256']
        tools={}
        for key,name in [('evaluator','cap_test_evaluate_ranker.py'),('common','cap_test_evaluation_common.py'),('statistics','summarize_sealed_cap_results.py'),('pairmath','sealed_cap_pairmath.py'),('matrix','sealed_cap_matrix.py')]:tools[key]=source(r/name if key in ('pairmath','matrix') else r/('resource_ready_analysis' if key=='statistics' else 'native_cost_evaluation')/name)
        leaf.update({tools[k]['path']:tools[k]['sha256'] for k in ('evaluator','common')})
        tools['cap_module']=next(dict(path=p,sha256=h) for p,h in leaf.items() if Path(p).name=='cap_ranker.py')
        tools['fixed_module']=next(dict(path=p,sha256=h) for p,h in leaf.items() if Path(p).name=='h_pool_fixed_center_ranker.py')
        tools['builder']=read.add(str(Path(m.__file__).resolve()),Path(m.__file__).read_bytes())
        downstream={tools[k]['path']:tools[k]['sha256'] for k in ('statistics','pairmath','matrix')}
        operators={}
        for name in ('isolation_entry.py','supervise_resource_ready_queue.py','launch_resource_ready_stage.py'):
            ref=source(r/name);operators[ref['path']]=ref['sha256']
        self.source=m.merge(leaf,downstream,operators,{tools['builder']['path']:tools['builder']['sha256']})
        self.registry=dict(tool_paths=tools,leaf_source_sha256=leaf,downstream_source_sha256=downstream,operator_source_sha256=operators)
        self.initial=dict(roles={},input_sha256={});self.validation=dict(roles={},caches={},input_sha256={})
        for seed in m.SEEDS:
            self.initial['roles'][str(seed)]={};self.validation['caches'][str(seed)]={'original_val':{}}
            for role in ('no_aug','aug'):self.initial['roles'][str(seed)][role]=self.role(r/f'initial/s{seed}/{role}',seed,role)
            for cap in (5,10):
                self.validation['roles'].setdefault(str(cap),{})[str(seed)]={role:self.role(r/f'rankers/C{cap}/s{seed}/{role}',seed,role,cap) for role in ('no_aug','aug')}
        self.identity={k:m.sha(k.encode()) for k in m.IDENTITY};metadata={};evaluations=[]
        for seed in m.SEEDS:
            for pool in m.POOLS:
                fwd=m.sha(str(seed).encode());folder=r/f'old/sealed/features/inherited_s{seed}/{pool}';cache=str(folder/'features.pt');ch=m.sha(cache.encode())
                producer=dict(schema='artificial_feature_cache',cohort='test',recall=pool,candidate_count=114982,pool_size=512,candidate_order_sha256=m.ORDER,
                  records=1371,base_queries=1000,variants=m.VARIANTS,checkpoint_sha256=fwd,feature_cache_sha256=ch,**self.identity)
                pr=read.add(folder/'summary.json',producer);er=read.add(folder/'EXIT_CODE',b'0\n')
                metadata[f'{seed}/{pool}']=dict(seed=seed,recall=pool,candidate_order_sha256=m.ORDER,cache=cache,cache_sha256=ch,producer_summary=pr,producer_exit=er,forward_checkpoint_sha256=fwd)
                evaluations.append(dict(config='fixedcenter_mixed_listwise_'+pool,seed=seed,cache=cache,cache_sha256=ch,
                  checkpoints={role:dict(path=rec['checkpoint'],sha256=rec['checkpoint_sha256']) for role,rec in self.initial['roles'][str(seed)].items()}))
                self.validation['caches'][str(seed)]['original_val'][pool]=dict(manifest=dict(checkpoint_sha256=fwd))
        self.legacy=dict(fixed_test_identity=self.identity,cache_metadata=metadata,evaluations=evaluations)
        baseline=dict(MRR=.58,top1=.51,frequency_crop_MRR=.26,peak_dropout_MRR=.21,macro_MRR=.43)
        self.protocol=dict(schema='cap10_controlled_protocol_20261024',remote_root=str(r),local_root=str(r),source_checkout='/synthetic/science',seeds=[193,211],caps=[5,10],final_epoch=64,alpha=1.,epoch_offset=0,only_eligible_new_config='cap10_dual',control_config='cap5_dual',preselected=m.BASELINE,test_allowed=False,promotion=False,pool_size=512,candidate_library_size=114982,validation=dict(baseline_signals=baseline))
        self.refs['protocol']=read.add(r/'protocol.json',self.protocol)
        self.initial['input_sha256']=dict(self.inputs);self.refs['initial_bindings']=read.add(r/'initial_bindings.json',self.initial)
        self.validation.update(input_sha256=dict(self.inputs),training_bindings=self.refs['initial_bindings'],cap_module=tools['cap_module'],fixed_module=tools['fixed_module'])
        self.refs['validation_bindings']=read.add(r/'validation_bindings.json',self.validation)
        self.summary=dict(status='PASS',test_read=False,promotion=False,preselected='cap10_dual',validation_run=str(r/'runs/validation_native_cost'),signals={'cap10_dual':dict(baseline,MRR=.60),'cap5_dual':dict(baseline,MRR=.59)},selection=dict(preselected='cap10_dual',gates=dict(MRR_gt_fixed_single=True,MRR_gt_matched_C5dual=True,**{k:True for k in ('top1','frequency_crop_MRR','peak_dropout_MRR','macro_MRR')})))
        self.summary.update(bindings=self.refs['validation_bindings'],input_sha256={**self.inputs,self.refs['protocol']['path']:self.refs['protocol']['sha256']},source_sha256=self.source)
        self.selection=dict(preselected='cap10_dual',selection_locked=False)
        self.lock=dict(schema='cap10_validation_lock_20261024',status='LOCKED',selection_locked=True,selection_scope='train_validation_only',test_reselection_allowed=False,promotion=False,protocol_sha256=self.refs['protocol']['sha256'],preselected='cap10_dual',validation_run=self.summary['validation_run'])
        for key,value in [('validation_summary',self.summary),('validation_selection',self.selection),('validation_lock',self.lock)]:self.refs[key]=read.add(r/(key+'.json'),value)
        valpins={**self.summary['input_sha256'],**{self.refs[k]['path']:self.refs[k]['sha256'] for k in ('validation_summary','validation_selection','validation_bindings','initial_bindings')}}
        self.valreview=dict(status='PASS',independent=True,reviewer='artificial nonauthor',scope='actual_validation_results',reviewed_sha256={**self.source,**valpins})
        self.initial['input_sha256']=dict(self.inputs);self.validation['input_sha256']=dict(self.inputs)
        qualifier=dict(schema='cap10_baseline_reuse_qualification_20261024',status='PASS',qualified=False,all_four_or_none=True,registered_baseline_jobs_count=4,new_evaluation_jobs_count=12,validation_preselected='cap10_dual',test_predictions_opened=False,test_tensors_opened=False)
        for key,value in [('initial_bindings',self.initial),('validation_bindings',self.validation),('validation_review',self.valreview),('baseline_qualification',qualifier),('source_registry',self.registry)]:self.refs[key]=read.add(r/(key+'.json'),value)
        self.refs['legacy_test_plan']=read.add(r/'old/sealed-old/extended24_test_plan.json',self.legacy)
        for key in ('test_implementation_review','statistics_implementation_review','builder_implementation_review'):
            self.refs[key]=read.add(r/(key+'.json'),dict(status='PASS',independent=True,reviewer='artificial nonauthor',scope='test_implementation',reviewed_sha256=self.source))
        self.lock.update({k:self.refs[k] for k in ('validation_summary','validation_selection','validation_review')});self.replace('validation_lock',self.lock)
        self.request=dict(schema='cap10_test_metadata_request_20261025',scope=dict(remote_root=str(r),evaluation_source_root=str(r/'native_cost_evaluation'),selection_scope='train_validation_only',all_12_fresh=True,test_reselection_allowed=False,promotion=False),refs=self.refs)

    def role(self,folder,seed,role,cap=None):
        cp=str(folder/('best.pt' if cap is None else 'final.pt'));h=m.sha(cp.encode());best=dict(epoch=7 if cap is None else 64,alpha=.5 if cap is None else 1.)
        record=dict(checkpoint=cp,checkpoint_sha256=h,best=best)
        for key,name in [('centers','centers.json'),('center_rows','center_rows.jsonl')]:
            path=str(folder/name);record[key]=path;record[key+'_sha256']=m.sha((str(seed)+role+name).encode());self.inputs[path]=record[key+'_sha256']
        producer=dict(best=best) if cap is None else dict(epoch=64,alpha=1.,delta_cap=cap,final_checkpoint_sha256=h)
        ref=self.reader.add(folder/'summary.json',producer);record.update(producer_summary=ref['path'],producer_summary_sha256=ref['sha256'])
        exitref=self.reader.add(folder/'EXIT_CODE',b'0\n');self.inputs.update({cp:h,ref['path']:ref['sha256'],exitref['path']:exitref['sha256']})
        return record
    def replace(self,key,value):self.refs[key]=self.reader.add(self.refs[key]['path'],value)
    def prepare(self):return m.prepare(self.reader,self.request)
    def freeze_request(self):
        bindings,plan,_=self.prepare();br=self.reader.add(self.root/'test_bindings.json',bindings);pr=self.reader.add(self.root/'test_plan.json',plan)
        required=m.merge(plan['evidence_input_sha256'],bindings['input_sha256'],plan['source_sha256'],{br['path']:br['sha256'],pr['path']:pr['sha256']})
        review=self.reader.add(self.root/'specific_plan_review.json',dict(status='PASS',independent=True,reviewer='artificial nonauthor',scope='test_plan_and_implementation',reviewed_sha256=required))
        self.request['freeze_stage_refs']=dict(bindings=br,plan=pr,plan_review=review);return self.request


class MetadataTests(unittest.TestCase):
    def setUp(self):self.f=Fixture()
    def test_complete_prepare_twelve_exact_cells_no_scientific_payload(self):
        bindings,plan,_=self.f.prepare();self.assertEqual(len(plan['entries']),12);self.assertTrue(all(e['execution']=='evaluate' for e in plan['entries']))
        self.assertEqual(set(bindings['source_sha256']),set(self.f.registry['leaf_source_sha256']))
        self.assertFalse(any(Path(p).suffix=='.pt' for p in self.f.reader.reads))
        self.assertTrue(set(self.f.registry["downstream_source_sha256"]).isdisjoint(bindings["input_sha256"]))
        self.assertNotIn(self.f.refs["statistics_implementation_review"]["path"],bindings["input_sha256"])
        self.assertIn(self.f.refs["statistics_implementation_review"]["path"],plan["evidence_input_sha256"])

    def test_extra_same_basename_source_alias_is_not_leaf_runtime(self):
        extra=self.f.reader.add(self.f.root/'extra/data.py',b'# artificial source only\n')
        self.f.registry['leaf_source_sha256'][extra['path']]=extra['sha256'];self.f.replace('source_registry',self.f.registry)
        with self.assertRaisesRegex(ValueError,'exact_runtime_leaf_source_names'):self.f.prepare()

    def test_missing_or_pending_lock_stops_before_old_test_metadata(self):
        self.f.lock['status']='PENDING';self.f.replace('validation_lock',self.f.lock)
        with self.assertRaisesRegex(ValueError,'complete_actual_validation_lock_required'):self.f.prepare()
        self.assertNotIn(self.f.refs['legacy_test_plan']['path'],self.f.reader.reads)
    def test_lock_must_bind_actual_prior_review_SHA(self):
        self.f.lock['validation_review']=dict(self.f.refs['validation_review'],sha256='a'*64);self.f.replace('validation_lock',self.f.lock)
        with self.assertRaisesRegex(ValueError,'lock_binds_prior_result_review_without_cycle'):self.f.prepare()
    def test_locked_guard_selection_cannot_be_changed(self):
        self.f.summary['selection']['gates']['top1']=False;self.f.replace('validation_summary',self.f.summary)
        self.f.valreview['reviewed_sha256'][self.f.refs['validation_summary']['path']]=self.f.refs['validation_summary']['sha256'];self.f.replace('validation_review',self.f.valreview)
        self.f.lock.update({k:self.f.refs[k] for k in ('validation_summary','validation_selection','validation_review')});self.f.replace('validation_lock',self.f.lock)
        with self.assertRaisesRegex(ValueError,'existing_locked_guard'):self.f.prepare()
    def test_reuse_is_not_an_accepted_policy_in_this_tool(self):
        ref=self.f.refs['baseline_qualification'];q=self.f.reader.ref(ref);q['qualified']=True;self.f.replace('baseline_qualification',q)
        with self.assertRaisesRegex(ValueError,'all_four_baselines_fresh_qualification'):self.f.prepare()
    def test_seed_cache_swapping_rejected_before_producer_read(self):
        self.f.legacy['cache_metadata']['193/single']['cache']=self.f.legacy['cache_metadata']['211/single']['cache'];self.f.replace('legacy_test_plan',self.f.legacy)
        with self.assertRaisesRegex(ValueError,'fixed_registered_cache_seed_path'):self.f.prepare()
    def test_forward_checkpoint_drift_rejected(self):
        self.f.legacy['cache_metadata']['193/single']['forward_checkpoint_sha256']='a'*64;self.f.replace('legacy_test_plan',self.f.legacy)
        with self.assertRaisesRegex(ValueError,'inherited_seed_forward_checkpoint'):self.f.prepare()
    def test_freeze_twelve_jobs_exact_five_paths_and_no_finalreview_cycle(self):
        req=self.f.freeze_request();frozen,queue,required=m.freeze(self.f.reader,req)
        self.assertEqual(len(queue['jobs']),12);self.assertEqual(queue['resource_wait'],dict(maximum_seconds=120,poll_seconds=.2))
        self.assertEqual(queue['operator_paths']['supervisor']['path'],str(self.f.root/'supervise_resource_ready_queue.py'))
        self.assertEqual(queue['source_sha256'][queue['operator_paths']['launcher']['path']],queue['operator_paths']['launcher']['sha256'])
        final=str(self.f.root/'test_freeze_independent_review.json')
        self.assertEqual(queue['review_paths'],[final]);self.assertNotIn(final,queue['source_sha256']);self.assertNotIn(final,queue['input_sha256']);self.assertNotIn(final,required)
        for job in queue['jobs']:self.assertEqual(len(job['allowed_reads']),5);self.assertEqual(sum(Path(p).suffix=='.pt' for p in job['allowed_reads']),3)
        self.assertIn(str(self.f.root/'test_queue.json'),required);self.assertIn(str(self.f.root/'frozen_test_selection.json'),required)
    def test_plan_review_needs_full_deferred_weight_SHA(self):
        req=self.f.freeze_request();ref=req['freeze_stage_refs']['plan_review'];review=self.f.reader.ref(ref);key=next(p for p in review['reviewed_sha256'] if p.endswith('final.pt'));review['reviewed_sha256'].pop(key)
        req['freeze_stage_refs']['plan_review']=self.f.reader.add(ref['path'],review)
        with self.assertRaisesRegex(ValueError,'specific_test_plan_review_exact_SHA'):m.freeze(self.f.reader,req)


class ActualReaderAndConsumerTests(unittest.TestCase):
    def test_real_reader_two_stages_and_native_sealed_gate_no_tensor_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();f=Fixture(root)
            request=f.freeze_request();frozen,queue,required=m.freeze(f.reader,request)
            f.reader.add(root/'frozen_test_selection.json',frozen);f.reader.add(root/'test_queue.json',queue)
            final=dict(status='PASS',independent=True,reviewer='artificial independent specific reviewer',scope='cap10_test_freeze_implementation',reviewed_sha256=required)
            f.reader.add(root/'test_freeze_independent_review.json',final)
            files={};hashes={}
            for path,raw in f.reader.data.items():
                p=Path(path)
                if p.is_relative_to(root):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
                else:self.assertEqual(p.read_bytes(),raw)
                files[path]=str(p);hashes[path]=m.sha(raw)
            mp=root/'metadata_map.json';mp.write_bytes(m.encoded(dict(files=files,sha256=hashes)))
            real=m.Reader(dict(path=str(mp),sha256=m.sha(mp.read_bytes())))
            bindings,plan,_=m.prepare(real,request);actual_frozen,actual_queue,_=m.freeze(real,request);real.finish()
            self.assertEqual(len(actual_queue['jobs']),12);self.assertEqual(bindings['entries'],frozen['entries']);self.assertEqual(plan['preselected'],'cap10_dual')
            self.assertFalse(any(Path(p).suffix=='.pt' for p in real.reads))
            # 真common消费新builder格式；只执行metadata gate，不加载模型或缓存。
            source=Path(m.__file__).parent/'native_cost_evaluation/cap_test_evaluation_common.py'
            spec=importlib.util.spec_from_file_location('_metadata_builder_native_common',source);common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
            args=SimpleNamespace(source=f.protocol['source_checkout'],protocol=str(root/'protocol.json'),bindings=str(root/'test_bindings.json'),frozen_selection=str(root/'frozen_test_selection.json'),freeze_review=str(root/'test_freeze_independent_review.json'))
            result=common.sealed_gate(args,[],require_cell=False)
            self.assertIsNone(result[-1]);self.assertEqual(len(result[3]['entries']),12)


    def test_cold_isolation_metadata_and_leaf_pins_exclude_downstream_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();f=Fixture(root);request=f.freeze_request()
            frozen,queue,required=m.freeze(f.reader,request)
            f.reader.add(root/'frozen_test_selection.json',frozen);f.reader.add(root/'test_queue.json',queue)
            f.reader.add(root/'test_freeze_independent_review.json',dict(status='PASS',independent=True,reviewer='artificial independent',scope='test_freeze',reviewed_sha256=required))
            for path,raw in f.reader.data.items():
                p=Path(path)
                if p.is_relative_to(root):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            job=queue['jobs'][0]
            for path in job['allowed_reads']:
                p=Path(path)
                if p.suffix=='.pt':p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(path.encode())
            guard=root/'cold_guard.py';guard.write_bytes((Path(m.__file__).parent/'isolation_entry.py').read_bytes())
            common=Path(m.__file__).parent/'native_cost_evaluation/cap_test_evaluation_common.py'
            program=root/'metadata_probe.py'
            body="import importlib.util,json\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
            body+=f"spec=importlib.util.spec_from_file_location('actual_native_common',{str(common)!r});c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)\n"
            values=dict(source=f.protocol['source_checkout'],protocol=str(root/'protocol.json'),bindings=str(root/'test_bindings.json'),frozen_selection=str(root/'frozen_test_selection.json'),freeze_review=str(root/'test_freeze_independent_review.json'))
            body+=f"args=SimpleNamespace(**{values!r});pins,proto,binding,frozen,_=c.sealed_gate(args,[],require_cell=False)\n"
            body+="for p,h in binding['source_sha256'].items():pins.add(p,h)\npins.finish()\n"
            program.write_text(body)
            log=root/'access_logs/test/cold_metadata.json'
            argv=[sys.executable,'-B',str(guard),'--bindings',str(root/'test_bindings.json'),'--phase','test','--access-log',str(log),'--output-root',job['job_out']]
            for path in job['allowed_reads']:argv+=['--allow-read',path]
            result=subprocess.run(argv+['--',str(program)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            receipt=json.loads(log.read_text());self.assertEqual(receipt['internal_exit'],0);self.assertEqual(receipt['rejected_paths'],[])
            self.assertTrue(set(f.registry['downstream_source_sha256']).isdisjoint(receipt['recorded_reads']))
            self.assertNotIn(f.refs['statistics_implementation_review']['path'],receipt['recorded_reads'])

    def test_reader_refuses_unregistered_test_summary_before_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();folder=root/'test';folder.mkdir();meta=folder/'summary.json';meta.write_text('{}')
            h=m.sha(meta.read_bytes());mp=root/'metadata_map.json';mp.write_bytes(m.encoded(dict(files={str(meta):str(meta)},sha256={str(meta):h})))
            reader=m.Reader(dict(path=str(mp),sha256=m.sha(mp.read_bytes())))
            with self.assertRaisesRegex(ValueError,'unregistered_test_metadata_forbidden'):reader.raw(str(meta),h)
            self.assertNotIn(str(meta),reader.reads)


if __name__=='__main__':unittest.main()

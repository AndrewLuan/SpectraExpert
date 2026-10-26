"""功能概述：准入checker的人工元数据测试，不消费任何实际未完成产物。
独立构造8已审final角色、成功run/独审声明、16val+4mechanism矩阵和始末inventory；
篡改同时更新SHA以确保拒绝来自科学注册约束，而不是旧摘要偶然失败。
MemoryBundle仅人工JSON/源码字符串，没有张量或预测；不导入训练/评价producer。
"""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import review_cap_evaluation_admission as c


class MemoryBundle:
    def __init__(self):self.docs={};self.hashes={};self.remote={};self.reads={};self.payload_reads=[]
    def put(self,path,value):
        path=str(path);self.docs[path]=copy.deepcopy(value);self.hashes[path]=c.digest(json.dumps(value,sort_keys=True).encode());self.remote[path]=self.hashes[path]
        return dict(path=path,sha256=self.hashes[path])
    def json(self,path,expected=None):
        path=str(path);c.need(expected is None or self.hashes[path]==expected,'actual_copied_SHA');self.reads[path]=self.hashes[path];return copy.deepcopy(self.docs[path])
    def raw(self,path,expected=None):
        path=str(path)
        if Path(path).suffix=='.pt' or 'predictions' in Path(path).name:self.payload_reads.append(path);raise AssertionError('payload forbidden')
        if expected:c.need(self.remote.get(path)==expected,'source_SHA')
        if type(self.docs.get(path)) is str:return self.docs[path].encode()
        return b'0\n' if Path(path).name in ('EXIT_CODE','EXTERNAL_EXIT_CODE') else b'artificial source'
    def ref(self,ref):return self.json(ref['path'],ref['sha256'])
    def pin(self,path,sha):c.need(self.remote.get(str(path))==sha,'current_inventory_pin')
    def exit0(self,path):c.need(self.raw(path)==b'0\n','actual_EXIT0_file')


class Fixture:
    def __init__(self):
        self.b=MemoryBundle();self.root=Path('/artificial/cap_controlled');r=self.root;b=self.b
        self.protocol=dict(remote_root=str(r),source_checkout='/artificial/science',caps=[5,10],seeds=[193,211],test_allowed=False,promotion=False,
          final_epoch=64,alpha=1.,epoch_offset=0,pool_size=512,candidate_library_size=114982,
          mechanism=dict(pairs=400,original_obstructed_pairs=87,saturation_threshold=.95))
        self.sources={str(r/name):c.digest(name.encode()) for name in (*c.EVAL,'cap_ranker.py','train_cap_ranker.py','isolation_entry.py','supervise_controlled_queue.py','launch_controlled_stage.py')}
        b.remote.update(self.sources);b.hashes.update(self.sources)
        self.binding=dict(source_sha256=self.sources,input_sha256=dict(self.sources),roles={},caches={},training_run=str(r/'runs/train_after_import_fix'))
        models=[];parentroles={}
        for cap,seed,role in sorted(c.MATRIX):
            folder=r/f'rankers/C{cap}/s{seed}/{role}';parentroles.setdefault(str(seed),{})[role]=dict(center33=[0.]*33,O_center3=[0.]*3,centers_sha256=c.digest((str(seed)+role+'center').encode()),center_rows_sha256=c.digest((str(seed)+role+'rows').encode()))
            parent=parentroles[str(seed)][role];ref=dict(center33=parent['center33'],O_center3=parent['O_center3'],best=dict(epoch=64,alpha=1.,selection='fixed_final_no_selection'))
            for key,name in [('checkpoint','final.pt'),('centers','centers.json'),('center_rows','center_rows.jsonl'),('producer_summary','summary.json')]:
                path=str(folder/name);h=parent[key+'_sha256'] if key in ('centers','center_rows') else c.digest(path.encode());ref[key]=path;ref[key+'_sha256']=h;self.binding['input_sha256'][path]=h;b.remote[path]=h
            self.binding['roles'].setdefault(str(cap),{}).setdefault(str(seed),{})[role]=ref
            models.append(dict(cap=cap,seed=seed,role=role,epochs=64,alpha=1.,draws=524288,optimizer_steps=16384,checkpoint=ref['checkpoint'],checkpoint_sha256=ref['checkpoint_sha256'],centers_sha256=ref['centers_sha256'],center_rows_sha256=ref['center_rows_sha256']))
        for seed,cohort,pool in itertools.product((193,211),('train','original_val','tune'),('single','dual')):
            path=f'/artificial/caches/s{seed}/{cohort}/{pool}/features.pt';h=c.digest(path.encode());b.remote[path]=h;self.binding['input_sha256'][path]=h
            producer=str(Path(path).parent/'summary.json');exitpath=str(Path(path).parent/'EXIT_CODE');ph=c.digest(producer.encode())
            self.binding['input_sha256'][producer]=ph;self.binding['input_sha256'][exitpath]=c.digest(exitpath.encode())
            b.remote[producer]=ph;b.remote[exitpath]=self.binding['input_sha256'][exitpath]
            self.binding['caches'].setdefault(str(seed),{}).setdefault(cohort,{})[pool]=dict(path=path,sha256=h,manifest=dict(cohort=cohort,recall=pool),producer=dict(path=producer,sha256=ph,exit_path=exitpath))
        fixed=b.put(r/'fixed_train_pairs.json',dict(schema='cap10_fixed_train_pairs_20261024',count=400,obstructed_count=87,selection_input=False));self.binding['fixed_train_pairs']=fixed
        self.binding['input_sha256'][fixed['path']]=fixed['sha256'];self.protocol['mechanism']['fixed_train_pairs']=fixed
        pref=b.put(r/'protocol.json',self.protocol)
        oldcaches=copy.deepcopy(self.binding['caches'])
        for pools in oldcaches.values():
            for routes in pools.values():
                for reg in routes.values():reg['producer']=reg['producer']['path']
        tb=b.put(r/'execution_bindings.json',dict(roles=parentroles,caches=oldcaches));tq=b.put(r/'train_after_import_fix_queue.json',dict(phase='train',run_name='train_after_import_fix',root=str(r)))
        self.binding.update(training_queue=tq,training_bindings=tb)
        audit=dict(schema='cap10_training_audit_20261024',status='PASS',independent=True,inputs_unchanged=True,training_run=self.binding['training_run'],checks={k:True for k in c.CHECKS},input_sha256={},input_end_sha256={},source_sha256={},queue_sha256=tq['sha256'],bindings_sha256=tb['sha256'],protocol_sha256=pref['sha256'],models=models)
        ar=b.put(r/'audits/training/summary.json',audit);ir=b.put(r/'audit_implementation.json',dict(status='PASS',independent=True,scope='cap10_training_audit_implementation',reviewed_sha256={}))
        self.binding['training_audit']=dict(ar,implementation_review=ir)
        run=r/'runs/train_after_import_fix'
        b.put(run/'completion.json',dict(external_exit=0,status='COMPLETE',inputs_unchanged=True,pid=103,queue_sha256=tq['sha256'],jobs=[dict(internal_exit=0,external_exit=0) for i in range(8)]))
        b.put(run/'launch_receipt.json',dict(internal_exit=0,external_exit=0,child_pid=103,wrapper_pid=102,queue_sha256=tq['sha256']))
        b.put(run/'driver_start.json',dict(pid=103,parent_pid=102,queue_sha256=tq['sha256']))
        reviewed=dict(self.sources);reviewed.update({ar['path']:ar['sha256'],tq['path']:tq['sha256'],tb['path']:tb['sha256'],pref['path']:pref['sha256']})
        tr=b.put(r/'actual_training_review.json',dict(status='PASS',independent=True,scope='actual_eight_training_results_and_reviewed_CPU_audit_provenance',reviewed_sha256=reviewed,models=8,total_history_rows=512,total_draws=4194304,total_optimizer_steps=131072,test_data_opened=False,execution=dict(training_run=str(run))))
        er=b.put(r/'eval_review.json',dict(status='PASS',independent=True,scope='cap10_validation_and_mechanism_implementation',reviewed_sha256=self.sources))
        jj=b.put(r/'JJ67_execution.json',dict(actual_exit=0,source_sha256=self.sources,source_end_sha256=self.sources))
        log=b.put(r/'JJ67_execution.log','Ran 67 tests in 0.123s\n\nOK\n')
        self.request=dict(schema='cap10_evaluation_admission_request_20261025',root=str(r),training_results_review=tr,evaluation_implementation_review=er,evaluation_JJ_execution=jj,evaluation_JJ_log=log)
        b.put(r/'validation_bindings.json',self.binding)
        for phase in ('validation','mechanism'):b.put(r/(phase+'_queue.json'),self.queue(phase))

    def queue(self,phase):
        r=self.root;b=self.b;jobs=[];comb=itertools.product((5,10),(193,211),('single','dual'),('original_val','tune')) if phase=='validation' else itertools.product((5,10),(193,211),('dual',),('train',))
        for n,(cap,seed,pool,cohort) in enumerate(comb):
            jid=f'{phase}_C{cap}_s{seed}_{pool}_{cohort}';out=str(r/phase/jid);access=str(r/'access_logs'/phase/(jid+'.json'));cache=self.binding['caches'][str(seed)][cohort][pool]['path']
            args={'--source':self.protocol['source_checkout'],'--protocol':str(r/'protocol.json'),'--bindings':str(r/'validation_bindings.json'),'--cache':cache,'--cap':str(cap),'--seed':str(seed),'--pool':pool,'--cohort':cohort,'--device':'cuda:0','--threads':'4','--out':out}
            if phase=='mechanism':args['--query-manifest']=str(r/'fixed_train_pairs.json')
            command=[str(r/'evaluate_cap_ranker.py')]+[v for pair in args.items() for v in pair]
            reads=[self.binding['roles'][str(cap)][str(seed)][role]['checkpoint'] for role in ('no_aug','aug')]+[cache]
            argv=['/artificial/python','-B',str(r/'isolation_entry.py'),'--bindings',str(r/'validation_bindings.json'),'--phase',phase,'--access-log',access,'--output-root',out]
            for p in reads:argv+=['--allow-read',p]
            jobs.append(dict(id=jid,command=command,argv=argv+['--']+command,cwd=self.protocol['source_checkout'],gpu=n%8,job_out=out,access_log=access,allowed_reads=reads))
        sources=dict(self.sources)
        for name in ('protocol.json','validation_bindings.json','isolation_entry.py','supervise_controlled_queue.py','launch_controlled_stage.py'):sources[str(r/name)]=b.hashes[str(r/name)]
        return dict(schema='cap10_controlled_queue_20261024',phase=phase,run_name=phase,root=str(r),bindings=str(r/'validation_bindings.json'),input_sha256=self.binding['input_sha256'],source_sha256=sources,maximum_seconds=10800,max_parallel=8,automatic_retry=False,promotion=False,external_api_calls=0,review_paths=[str(r/(phase+'_admission_independent_review.json'))],python='/artificial/python',jobs=jobs)

    def change(self,path,mutator):
        value=self.b.json(path);mutator(value);self.b.put(path,value)


class AdmissionTests(unittest.TestCase):
    def setUp(self):self.f=Fixture()
    def runcheck(self):return c.verify(self.f.b,self.f.request)
    def test_complete_16plus4_review_closure_without_payload(self):
        result=self.runcheck();self.assertEqual(result['validation']['jobs'],16);self.assertEqual(result['mechanism']['jobs'],4)
        for phase,doc in result.items():
            qpath=str(self.f.root/(phase+'_queue.json'));q=self.f.b.docs[qpath]
            self.assertEqual(doc['reviewed_sha256'],dict(q['source_sha256'],**{qpath:self.f.b.hashes[qpath]}))
        self.assertEqual(self.f.b.payload_reads,[])
    def test_wrong_failed_training_run_rejected(self):
        self.f.change(self.f.root/'validation_bindings.json',lambda v:v.update(training_run=str(self.f.root/'runs/train')))
        with self.assertRaisesRegex(ValueError,'successful_training_run_only'):self.runcheck()
    def test_bool_outer_exit_not_zero(self):
        self.f.change(self.f.root/'runs/train_after_import_fix/launch_receipt.json',lambda v:v.update(internal_exit=False))
        with self.assertRaisesRegex(ValueError,'strict_actual_exit0'):self.runcheck()
    def test_missing_or_false_audit_check_rejected(self):
        path=self.f.binding['training_audit']['path'];self.f.b.docs[path]['checks']['frozen_scale']=False
        with self.assertRaisesRegex(ValueError,'all12_audit_checks'):self.runcheck()
    def test_one_checkpoint_drift_against_cpu_audit_rejected(self):
        self.f.b.docs[str(self.f.root/'validation_bindings.json')]['roles']['10']['211']['aug']['checkpoint_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'audit_final_checkpoint_ref'):self.runcheck()
    def test_missing_val_cell_rejected(self):
        self.f.b.docs[str(self.f.root/'validation_queue.json')]['jobs'].pop()
        with self.assertRaisesRegex(ValueError,'complete_unique_phase_jobs'):self.runcheck()
    def test_mechanism_cannot_read_single_or_extra_tensor(self):
        self.f.b.docs[str(self.f.root/'mechanism_queue.json')]['jobs'][0]['allowed_reads'].append('/artificial/extra.pt')
        with self.assertRaisesRegex(ValueError,'three_exact_tensor_reads'):self.runcheck()
    def test_validation_cannot_apply_query_filter(self):
        self.f.b.docs[str(self.f.root/'validation_queue.json')]['jobs'][0]['command']+=['--query-manifest','/artificial/selection.json']
        with self.assertRaisesRegex(ValueError,'exact_registered_CLI'):self.runcheck()
    def test_unreviewed_source_or_current_inventory_drift_rejected(self):
        path=str(self.f.root/'evaluate_cap_ranker.py');self.f.b.remote[path]='0'*64
        with self.assertRaisesRegex(ValueError,'source_SHA|current_inventory_pin'):self.runcheck()
    def test_jj_bool_exit_or_66_tests_rejected(self):
        ref=self.f.request['evaluation_JJ_log'];self.f.b.docs[ref['path']]='Ran 66 tests in 0.123s\n\nOK\n'
        with self.assertRaisesRegex(ValueError,'JJ67_actual_execution'):self.runcheck()
    def test_fixed400_manifest_alteration_rejected(self):
        ref=self.f.binding['fixed_train_pairs'];self.f.b.docs[ref['path']]['obstructed_count']=88
        with self.assertRaisesRegex(ValueError,'actual_fixed_manifest_contract'):self.runcheck()

    def test_cache_same_cohort_from_other_seed_rejected(self):
        binding=self.f.b.docs[str(self.f.root/'validation_bindings.json')]
        binding['caches']['193']['tune']['single']=copy.deepcopy(binding['caches']['211']['tune']['single'])
        with self.assertRaisesRegex(ValueError,'inherited_seed_cache_identity'):self.runcheck()


class BundleTests(unittest.TestCase):
    def test_existing_copy_alias_only_through_exact_hashed_base_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();old=root/'old';old.mkdir();new=root/'new';new.mkdir()
            meta=old/'metadata.json';meta.write_text('{"artificial":true}')
            remote='/artificial/metadata.json';mh=c.digest(meta.read_bytes())
            base=old/'manifest.json';base.write_text(json.dumps(dict(files={remote:str(meta)},sha256={remote:mh})))
            value=dict(files={remote:str(meta)},sha256={remote:mh},remote_before_sha256={remote:mh},remote_after_sha256={remote:mh},hash_only_inputs={},base_manifest=dict(path=str(base),sha256=c.digest(base.read_bytes())))
            path=new/'artifact_map.json';path.write_text(json.dumps(value));bundle=c.Bundle(path)
            self.assertTrue(bundle.json(remote)['artificial']);bundle.finish()
            base.write_text('{}')
            with self.assertRaisesRegex(ValueError,'existing_manifest_or_inventory_drift'):bundle.finish()

    def test_actual_copy_metadata_and_hash_only_tensor_without_open_then_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();meta=root/'metadata.json';meta.write_text('{"artificial":true}')
            remote='/artificial/metadata.json';tensor='/artificial/features.pt';mh=c.digest(meta.read_bytes())
            before={remote:mh,tensor:'a'*64}
            value=dict(files={remote:str(meta)},sha256={remote:mh},remote_before_sha256=before,remote_after_sha256=before,hash_only_inputs={tensor:'a'*64})
            path=root/'artifact_map.json';path.write_text(json.dumps(value));bundle=c.Bundle(path)
            self.assertTrue(bundle.json(remote)['artificial']);bundle.pin(tensor,'a'*64);bundle.finish()
            meta.write_text('{"artificial":false}')
            with self.assertRaisesRegex(ValueError,'consumed_file_drift'):bundle.finish()

    def test_symlink_metadata_to_payload_rejected_before_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();tensor=root/'features.pt';tensor.write_text('untouched artificial bytes')
            link=root/'metadata.json';link.symlink_to(tensor)
            with self.assertRaisesRegex(ValueError,'metadata_or_source_only|symlink_forbidden'):c.safe(link)


if __name__=='__main__':unittest.main()

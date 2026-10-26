"""功能概述：人工metadata全16+4恢复准入fixture，不加载tensor或实际失败预测。
复用旧checker作者的已审训练元数据fixture，只新增隔离binding/source/queue映射与攻击。
常量SHA patch只用于人工原件，不改production常量；断言保留全部预算、角色、旧失败证据。
"""
import copy,itertools,unittest,json,tempfile
from unittest import mock
from pathlib import Path
import review_native_cost_evaluation_admission as c
from test_review_cap_evaluation_admission import Fixture

class NativeFixture:
    def __init__(self):
        f=Fixture();self.b=b=f.b;self.root=r=f.root;self.old=f
        # The real old registry has 206 admitted pins; use inert hash-only metadata here.
        while len(f.binding['input_sha256'])<206:
            name=f'/artificial/old_metadata/{len(f.binding["input_sha256"])}.json';h=c.digest(name.encode())
            f.binding['input_sha256'][name]=h;b.remote[name]=h
        b.put(r/'validation_bindings.json',f.binding)
        for phase in ('validation','mechanism'):b.put(r/(phase+'_queue.json'),f.queue(phase))
        self.old_binding_sha=b.hashes[str(r/'validation_bindings.json')]
        self.old_queues={s:b.hashes[str(r/(s+'_queue.json'))] for s in ('validation','mechanism')}
        b.put(r/'runs/validation/completion.json',dict(status='FAILED',queue_sha256=self.old_queues['validation'],pid=205))
        b.put(r/'runs/validation/launch_receipt.json',dict(internal_exit=1,external_exit=1,queue_sha256=self.old_queues['validation'],child_pid=205,wrapper_pid=204))
        self.binding=n=copy.deepcopy(f.binding);sources=dict(n['source_sha256'])
        for name in c.EVAL:
            sources.pop(str(r/name));new=str(r/'native_cost_evaluation'/name);h=c.digest(new.encode());sources[new]=h;b.remote[new]=h;b.hashes[new]=h
        er=b.put(r/'native_cost_evaluation/implementation_review.json',dict(status='PASS',independent=True,scope='cap10_validation_and_mechanism_implementation',reviewed_sha256=sources))
        jj=b.put(r/'JJ_native71.json',dict(actual_exit=0,source_sha256=sources,source_end_sha256=sources))
        log=b.put(r/'JJ_native71.log','Ran 71 tests in 0.5s\n\nOK\n')
        fr=b.put(r/'validation_failed_attempt_independent_review.json',dict(status='PASS',independent=True,
          scope='failed_validation_preservation_and_exposure_accounting_not_recovery_admission',queue_sha256=self.old_queues['validation'],
          copied_artifacts=74,actual_remote_input_pins=206,prior_training_audit_inputs_preserved=200,launched_C5_jobs=8,pending_C10_jobs=8,
          original_val_internal_failures=4,tune_peer_cleanup=4,completed_prediction_rows=0,usable_validation_metric_summaries=0,
          model_loading_and_validation_reading_occurred=True,no_zero_validation_exposure_claim=True,automatic_retry=False,recovery_admission=False,
          driver_pid=205,wrapper_pid=204))
        dr=b.put(r/'validation_cost_diagnostic_result_independent_review.json',dict(status='PASS',independent=True,
          scope='actual_single_validation_row_native_cost_contract_not_full_validation_or_recovery_admission',
          generalization_scope='C5_seed193_single_original_val_row0_only',no_NaN_or_shape_failure_in_this_row=True,
          validation_recovery_admission=False,actual_inner_exit=0,actual_external_exit=0,actual_ssh_exit=0))
        for rel,doc in (
          ('diagnostics/validation_cost_contract_row0/summary.json',dict(cap=5,seed=193,pool='single',cohort='original_val',row_index=0,
            diagnostic=dict(legacy_assertions=dict(shape512=True,dtype_float32=False,all_finite=True),native_cost=dict(shape=[512],dtype='torch.float64',finite=True,nonfinite_count=0)))),
          ('access_logs/diagnostics/validation_cost_contract_row0.json',dict(internal_exit=0))):b.put(r/rel,doc)
        b.put(r/'actual_validation_cost_diagnostic_execution_receipt.json',dict(actual_inner_exit=0,actual_external_exit=0,actual_ssh_exit=0,
          summary_sha256=b.hashes[str(r/'diagnostics/validation_cost_contract_row0/summary.json')]))
        frdoc=b.docs[fr['path']]
        frdoc['reviewed_sha256']={str(r/rel):b.hashes[str(r/rel)] for rel in ('validation_bindings.json','validation_queue.json','mechanism_queue.json','runs/validation/completion.json','runs/validation/launch_receipt.json')}
        fr=b.put(fr['path'],frdoc)
        drdoc=b.docs[dr['path']]
        drdoc['reviewed_sha256']={str(r/rel):b.hashes[str(r/rel)] for rel in ('diagnostics/validation_cost_contract_row0/summary.json','actual_validation_cost_diagnostic_execution_receipt.json','access_logs/diagnostics/validation_cost_contract_row0.json')}
        dr=b.put(dr['path'],drdoc)
        inputs=dict(n['input_sha256']);inputs.update({str(r/'native_cost_evaluation'/name):b.hashes[str(r/'native_cost_evaluation'/name)] for name in c.EVAL})
        inputs.update({ref['path']:ref['sha256'] for ref in (er,jj,fr)})
        n.update(source_sha256=sources,input_sha256=inputs,evaluation_implementation=er,evaluation_JJ_tests=jj,
          failure_recovery_review=fr,prior_validation_bindings=dict(path=str(r/'validation_bindings.json'),sha256=self.old_binding_sha),
          evaluation_source_root=str(r/'native_cost_evaluation'),created_utc='2026-10-25T00:00:00Z',input_scope='native recovery')
        self.bp=b.put(r/'native_cost_validation_bindings.json',n)
        self.qrefs={}
        for phase in ('validation','mechanism'):
            old=b.json(r/(phase+'_queue.json'));q=copy.deepcopy(old);name=phase+'_native_cost'
            q.update(run_name=name,bindings=self.bp['path'],input_sha256=inputs,source_sha256=dict(sources),
              review_paths=[str(r/(name+'_admission_independent_review.json'))],
              recovery_from=dict(queue=dict(path=str(r/(phase+'_queue.json')),sha256=self.old_queues[phase]),failed_validation_run=str(r/'runs/validation'),failure_review=fr,automatic_retry=False))
            for fn in ('protocol.json','isolation_entry.py','supervise_controlled_queue.py','launch_controlled_stage.py'):
                q['source_sha256'][str(r/fn)]=b.hashes[str(r/fn)]
            q['source_sha256'][self.bp['path']]=self.bp['sha256']
            for j in q['jobs']:
                mapping={str(r/fn):str(r/'native_cost_evaluation'/fn) for fn in c.EVAL}
                out=str(r/name/j['id']);access=str(r/'access_logs'/name/(j['id']+'.json'))
                mapping.update({str(r/'validation_bindings.json'):self.bp['path'],j['job_out']:out,j['access_log']:access})
                j['command']=[mapping.get(x,x) for x in j['command']];j['argv']=[mapping.get(x,x) for x in j['argv']]
                j.update(job_out=out,access_log=access)
            self.qrefs[phase]=b.put(r/(name+'_queue.json'),q)
        self.request=dict(f.request,schema='cap10_native_cost_evaluation_admission_request_20261025',bindings=self.bp,
          queues=self.qrefs,runs={s:str(r/'runs'/(s+'_native_cost')) for s in self.qrefs},
          review_paths={s:str(r/(s+'_native_cost_admission_independent_review.json')) for s in self.qrefs},
          evaluation_source_root=str(r/'native_cost_evaluation'),evaluation_implementation_review=er,
          evaluation_JJ_execution=jj,evaluation_JJ_log=log,expected_JJ_test_count=71,cost_diagnostic_result_review=dr)
    def check(self):
        with mock.patch.object(c,'OLD_BINDING_SHA',self.old_binding_sha),mock.patch.object(c,'OLD_QUEUES',self.old_queues):
            return c.verify(self.b,self.request)

class NativeAdmissionTests(unittest.TestCase):
    def setUp(self):self.f=NativeFixture()
    def test_full_16plus4_exact_new_source_and_review_closure(self):
        out=self.f.check()
        for phase in ('validation','mechanism'):
            q=self.f.b.docs[self.f.qrefs[phase]['path']]
            self.assertEqual(out[phase]['reviewed_sha256'],{**q['source_sha256'],self.f.qrefs[phase]['path']:self.f.qrefs[phase]['sha256']})
            self.assertEqual(out[phase]['original_input_pins_preserved'],206)
        self.assertEqual(self.f.b.payload_reads,[])
    def test_dropping_old_source_input_is_rejected(self):
        self.f.b.docs[self.f.bp['path']]['input_sha256'].pop(str(self.f.root/'evaluate_cap_ranker.py'))
        with self.assertRaisesRegex(ValueError,'preserve_all_original206'):self.f.check()
    def test_old_registry_or_centers_cannot_change(self):
        self.f.b.docs[self.f.bp['path']]['roles']['5']['193']['aug']['center33'][0]=.1
        with self.assertRaisesRegex(ValueError,'all_training_cache_role_metadata_unchanged'):self.f.check()
    def test_native_run_cannot_overwrite_failed_output(self):
        q=self.f.b.docs[self.f.qrefs['validation']['path']];q['jobs'][0]['job_out']=str(self.f.root/'validation'/q['jobs'][0]['id'])
        with self.assertRaisesRegex(ValueError,'exact_output_scope'):self.f.check()
    def test_old_evaluator_entry_is_rejected(self):
        q=self.f.b.docs[self.f.qrefs['validation']['path']];q['jobs'][0]['command'][0]=str(self.f.root/'evaluate_cap_ranker.py')
        with self.assertRaisesRegex(ValueError,'canonical_evaluator'):self.f.check()
    def test_source_drift_or_unreviewed_old_root_rejected(self):
        key=str(self.f.root/'native_cost_evaluation/evaluate_cap_ranker.py');self.f.b.remote[key]='0'*64
        with self.assertRaisesRegex(ValueError,'source_SHA|current_inventory_pin'):self.f.check()
    def test_jj_count_or_bool_exit_rejected(self):
        self.f.b.docs[self.f.request['evaluation_JJ_execution']['path']]['actual_exit']=False
        with self.assertRaisesRegex(ValueError,'strict_actual_exit0'):self.f.check()
    def test_actual_log_must_match_declared_full_count(self):
        self.f.b.docs[self.f.request['evaluation_JJ_log']['path']]='Ran 70 tests in 0.5s\n\nOK\n'
        with self.assertRaisesRegex(ValueError,'JJ_full_suite_actual_execution'):self.f.check()
    def test_failed_run_must_remain_failed(self):
        self.f.b.docs[str(self.f.root/'runs/validation/completion.json')]['status']='COMPLETE'
        with self.assertRaisesRegex(ValueError,'old_failure_stays_failed'):self.f.check()
    def test_probe_cannot_be_promoted_to_recovery_gate(self):
        self.f.b.docs[self.f.request['cost_diagnostic_result_review']['path']]['validation_recovery_admission']=True
        with self.assertRaisesRegex(ValueError,'diagnosis_limited_to_dtype_contract'):self.f.check()
    def test_no_zero_exposure_claim_allowed(self):
        self.f.b.docs[self.f.binding['failure_recovery_review']['path']]['model_loading_and_validation_reading_occurred']=False
        with self.assertRaisesRegex(ValueError,'no_zero_exposure_or_retry_claim'):self.f.check()
    def test_matrix_still_requires_all16(self):
        self.f.b.docs[self.f.qrefs['validation']['path']]['jobs'].pop()
        with self.assertRaisesRegex(ValueError,'complete_unique_phase_jobs'):self.f.check()
    def test_scientific_gpu_or_budget_change_rejected(self):
        q=self.f.b.docs[self.f.qrefs['mechanism']['path']];q['jobs'][0]['gpu']=7
        with self.assertRaisesRegex(ValueError,'no_scientific_job_argument_or_role_change'):self.f.check()
    def test_failure_and_probe_review_must_cover_actual_metadata_bytes(self):
        fref=self.f.binding['failure_recovery_review'];self.f.b.docs[fref['path']]['reviewed_sha256']={}
        with self.assertRaisesRegex(ValueError,'review_covers_actual_preserved_metadata'):self.f.check()
    def test_probe_summary_drift_after_prior_review_rejected(self):
        path=str(self.f.root/'diagnostics/validation_cost_contract_row0/summary.json')
        self.f.b.hashes[path]='e'*64
        with self.assertRaisesRegex(ValueError,'review_covers_actual_preserved_metadata'):self.f.check()

class NativeBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.meta=self.root/'metadata.json';self.meta.write_text('{}');h=c.digest(self.meta.read_bytes());self.remote='/artificial/metadata.json'
        self.inventory=dict(actual_exit=0,inputs_unchanged=True,test_read=False,model_calls=0,
          remote_before_sha256={self.remote:h},remote_after_sha256={self.remote:h})
        self.inv=self.root/'inventory.json';self.inv.write_text(json.dumps(self.inventory))
        self.value=dict(schema='new_native_schema_not_old_string',files={self.remote:str(self.meta)},sha256={self.remote:h},
          remote_before_sha256={self.remote:h},remote_after_sha256={self.remote:h},hash_only_inputs={},
          actual_inventory=dict(path=str(self.inv),sha256=c.digest(self.inv.read_bytes()),ssh_actual_exit=0))
        self.path=self.root/'artifact_map.json'
    def save(self):self.path.write_text(json.dumps(self.value))
    def test_actual_inventory_required_for_native_map(self):
        self.value.pop('actual_inventory');self.save()
        with self.assertRaisesRegex(ValueError,'native_actual_inventory_required'):c.Bundle(self.path)
    def test_new_schema_cannot_bypass_strict_actual_inventory_exit(self):
        self.inventory['actual_exit']=False;self.inv.write_text(json.dumps(self.inventory));self.value['actual_inventory']['sha256']=c.digest(self.inv.read_bytes());self.save()
        with self.assertRaisesRegex(ValueError,'strict_actual_exit0'):c.Bundle(self.path)
    def test_new_schema_actual_inventory_and_end_sha(self):
        self.save();b=c.Bundle(self.path);self.assertEqual(b.json(self.remote),{});b.finish()
        self.inv.write_text('{}')
        with self.assertRaisesRegex(ValueError,'existing_manifest_or_inventory_drift'):b.finish()

if __name__=='__main__':unittest.main()

"""功能概述：人工16+4资源恢复准入元数据及拒绝攻击，不加载任何科学张量/预测。
固定native原件fixture由原18项suite独立保留；本文件组合完整旧核验与新矩阵函数，
并攻击source别名、213输入、资源阈值、科学命令、失败曝光与JJ真实退出/log闭包。
"""
import copy
import json
from pathlib import Path
import unittest
from unittest import mock
import review_resource_ready_evaluation_admission as c
from test_review_native_cost_evaluation_admission import NativeFixture

class ResourceFixture:
    def __init__(self):
        self.old=NativeFixture();self.b=b=self.old.b;self.root=r=self.old.root
        self.protocol=dict(b.json(r/'protocol.json'),local_root='/artificial/local')
        self.binding=self.old.binding
        self.native_queues=dict(self.old.qrefs)
        # Inert registered files model real 20-source native envelope; metadata only.
        for phase,ref in self.native_queues.items():
            q=b.docs[ref['path']]
            for i in range(20-len(q['source_sha256'])):
                p=str(r/f'inert_science{i}.py');h=c.sha(p.encode());q['source_sha256'][p]=h;b.hashes[p]=b.remote[p]=h
            self.native_queues[phase]=b.put(ref['path'],q)
        self.operators={str(r/name):c.sha(name.encode()) for name in c.OPERATORS.values()}
        self.readers={str(r/'resource_ready_analysis'/name):c.sha(('reader'+name).encode()) for name in c.READERS}
        b.hashes.update(self.operators);b.remote.update(self.operators);b.hashes.update(self.readers);b.remote.update(self.readers)
        self.request=copy.deepcopy(self.old.request)
        self.request.update(schema='cap10_resource_ready_evaluation_admission_request_20261025',analysis_source_root=str(r/'resource_ready_analysis'))
        for prefix,sources,scope in (
            ('operator',self.operators,'cap10_resource_ready_operator_and_launcher_implementation'),
            ('reader',self.readers,'cap10_resource_ready_validation_and_mechanism_readers_implementation')):
            self.request[prefix+'_implementation_review']=b.put(r/(prefix+'_impl.json'),dict(status='PASS',independent=True,scope=scope,reviewed_sha256=sources))
            self.request[prefix+'_JJ_execution']=b.put(r/(prefix+'_JJ.json'),dict(actual_exit=0,actual_ssh_exit=0,source_sha256=sources,source_end_sha256=sources))
            self.request[prefix+'_JJ_log']=b.put(r/(prefix+'_JJ.log'),'Ran 43 tests in 1.25s\n\nOK\n')
            self.request['expected_'+prefix+'_JJ_test_count']=43
        strategy=dict(schema='cap10_resource_readiness_recovery_strategy_20261025',
          status='LOCKED_INFRASTRUCTURE_RECOVERY_STRATEGY_NOT_EXECUTION_ADMISSION',prior_run='validation_native_cost',
          recovery='all16_fresh_in_one_new_run_after_operator_and_actual_queue_independent_PASS',
          completed_cells=4,cancelled_cells=4,unlaunched_cells=8,logical_validation_matrix_cells=16,
          prior_completed_cells_retained=True,prior_completed_cells_used_for_selection=False,
          prior_partial_predictions_used_for_selection=False,controller_opened_prior_validation_metrics=False,
          rerun_choice_based_on_metrics=False,compare_all_four_prior_completed_cells_against_new_semantic_predictions=True,
          additional_completed_validation_evaluations_expected=4,training_draws_unchanged=4194304,
          optimizer_steps_unchanged=131072,new_training=False,test_read=False,test_reselection_allowed=False,
          promotion=False,official_system_changed=False,old_training_operator_source_bytes_unchanged=True,
          actual_recovery_launch_allowed=False,resources=dict(maximum_wait_seconds=120,poll_seconds=.2,
            stage_deadline_seconds=10800,memory_threshold_mib_exclusive=1024,utilization_threshold_percent_exclusive=5,
            foreign_processes='reject and never signal',owned_completed_PID_telemetry='bounded observation with lineage checks'))
        self.request['recovery_strategy']=b.put(r/'resource_readiness_recovery_strategy.json',strategy)
        success=[dict(id=f'validation_C5_s{s}_{p}_original_val',internal_exit=0,external_exit=0) for s in (193,211) for p in ('single','dual')]
        partial=[dict(id=f'validation_C5_s{s}_{p}_tune',internal_exit=None,external_exit=-15) for s in (193,211) for p in ('single','dual')]
        b.put(r/'runs/validation_native_cost/completion.json',dict(status='FAILED',error="AssertionError(('gpu_not_idle', 0))",pid=1692998,
          queue_sha256=self.native_queues['validation']['sha256'],external_exit=1,jobs=success+partial))
        b.put(r/'runs/validation_native_cost/launch_receipt.json',dict(child_pid=1692998,wrapper_pid=1692993,internal_exit=1,external_exit=1,
          queue_sha256=self.native_queues['validation']['sha256']))
        failure=dict(status='PASS',independent=True,scope='failed_validation_resource_execution_exposure_and_preservation_not_recovery_admission',
          queue_sha256=self.native_queues['validation']['sha256'],wrapper_pid=1692993,driver_pid=1692998,copied_artifacts=82,
          actual_input_pins_preserved=213,completed_validation_cells=4,completed_validation_records=3616,cancelled_partial_tune_cells=4,
          unlaunched_C10_cells=8,training_draws_added=0,training_optimizer_steps_added=0,training_total_draws_remains=4194304,
          training_total_steps_remains=131072,metrics_values_interpreted=False,actual_test_payload_read=False,
          measured_utilization_memory_apps_at_failed_guard_available=False,selection_performed=False,automatic_retry=False,
          scientific_recovery_admission=False,completed_cells=success)
        failure['reviewed_sha256']={str(r/rel):b.hashes[str(r/rel)] for rel in (
          'validation_native_cost_queue.json','runs/validation_native_cost/completion.json','runs/validation_native_cost/launch_receipt.json')}
        self.request['resource_failure_review']=b.put(r/'resource_failure.json',failure)
        self.qrefs={}
        for phase,ref in self.native_queues.items():
            q=b.json(ref['path']);name=phase+'_resource_ready';q['run_name']=name
            q['operator_paths']={role:dict(path=str(r/file),sha256=self.operators[str(r/file)]) for role,file in c.OPERATORS.items()}
            q['resource_wait']=dict(maximum_seconds=120,poll_seconds=.2)
            q['source_sha256'].update(self.operators);q['source_sha256'].update(self.readers)
            q['review_paths']=[str(r/(name+'_admission_independent_review.json'))]
            q['analysis_tools']=dict(source_root=str(r/'resource_ready_analysis'),implementation_review=self.request['reader_implementation_review'])
            q['operator_implementation_review']=self.request['operator_implementation_review']
            q['recovery_from']=dict(prior_queue=ref,failed_validation_run=str(r/'runs/validation_native_cost'),failure_review=self.request['resource_failure_review'],
              strategy=self.request['recovery_strategy'],automatic_retry=False,prior_completed_cells_used_for_selection=False)
            for j in q['jobs']:
                out=str(r/name/j['id']);access=str(r/'access_logs'/name/(j['id']+'.json'));mapping={j['job_out']:out,j['access_log']:access}
                j['command']=[mapping.get(x,x) for x in j['command']];j['argv']=[mapping.get(x,x) for x in j['argv']];j.update(job_out=out,access_log=access)
            self.qrefs[phase]=b.put(r/(name+'_queue.json'),q)
        self.request.update(bindings=self.old.bp,queues=self.qrefs,
          runs={s:str(r/'runs'/(s+'_resource_ready')) for s in self.qrefs},
          review_paths={s:str(r/(s+'_resource_ready_admission_independent_review.json')) for s in self.qrefs})
    def queue(self,phase='validation'):
        with mock.patch.object(c,'NATIVE_QUEUES',{s:r['sha256'] for s,r in self.native_queues.items()}):
            return c.verify_new_queue(self.b,self.root,self.binding,self.protocol,self.request,phase,self.operators,self.readers)
    def failed(self):return c.verify_second_failure(self.b,self.root,self.request)
    def check(self):
        with mock.patch.object(c,'NATIVE_QUEUES',{s:r['sha256'] for s,r in self.native_queues.items()}):
            self.failed();c.verify_strategy(self.b,self.request)
            for s in self.qrefs:self.queue(s)
        for prefix,sources in [('operator',self.operators),('reader',self.readers)]:
            c.reviewed_alias(self.b.ref(self.request[prefix+'_implementation_review']),sources,self.root,self.protocol,prefix)
            c.verify_JJ(self.b,self.request,prefix,sources,43)

class ResourceAdmissionTests(unittest.TestCase):
    def setUp(self):self.f=ResourceFixture()
    def test_complete16plus4_new_resource_metadata_only(self):
        self.f.check();self.assertEqual(self.f.b.payload_reads,[])
        self.assertEqual(len(self.f.queue()['source_sha256']),25)
    def test_old_native_training_validation_closure_still_full(self):
        f=NativeFixture();self.assertEqual(f.check()['validation']['jobs'],16)
    def test_no_partial_success_reuse_or_metric_based_rerun(self):
        self.f.b.docs[self.f.request['recovery_strategy']['path']]['prior_completed_cells_used_for_selection']=True
        with self.assertRaisesRegex(ValueError,'locked_result_independent'):self.f.check()
    def test_strategy_must_compare_prior_four_semantic_outputs(self):
        self.f.b.docs[self.f.request['recovery_strategy']['path']]['compare_all_four_prior_completed_cells_against_new_semantic_predictions']=False
        with self.assertRaisesRegex(ValueError,'locked_result_independent'):self.f.check()
    def test_source_extra_or_missing_rejected(self):
        self.f.b.docs[self.f.qrefs['validation']['path']]['source_sha256'].pop(next(iter(self.f.operators)))
        with self.assertRaisesRegex(ValueError,'exact25_source_union'):self.f.queue()
    def test_scientific_input_cannot_be_removed(self):
        self.f.b.docs[self.f.qrefs['validation']['path']]['input_sha256']={}
        with self.assertRaisesRegex(ValueError,'unchanged_213'):self.f.queue()
    def test_scientific_command_or_gpu_cannot_change(self):
        self.f.b.docs[self.f.qrefs['validation']['path']]['jobs'][0]['gpu']=7
        with self.assertRaisesRegex(ValueError,'only_output_access'):self.f.queue()
    def test_old_output_is_not_reused(self):
        q=self.f.b.docs[self.f.qrefs['validation']['path']];q['jobs'][0]['job_out']=str(self.f.root/'validation_native_cost'/q['jobs'][0]['id'])
        with self.assertRaisesRegex(ValueError,'only_output_access'):self.f.queue()
    def test_resource_policy_exact120_point2(self):
        self.f.b.docs[self.f.qrefs['validation']['path']]['resource_wait']['poll_seconds']=.5
        with self.assertRaisesRegex(ValueError,'exact_120'):self.f.queue()
    def test_old_operator_alias_same_digest_rejected(self):
        self.f.b.docs[self.f.qrefs['validation']['path']]['operator_paths']['supervisor']['path']=str(self.f.root/'supervise_controlled_queue.py')
        with self.assertRaisesRegex(ValueError,'exact_new_operator_paths'):self.f.queue()
    def test_missing_one_matrix_cell_rejected(self):
        self.f.b.docs[self.f.qrefs['validation']['path']]['jobs'].pop()
        with self.assertRaisesRegex(ValueError,'same_registered_matrix_size'):self.f.queue()
    def test_strict_bool_failed_exit_not_int1(self):
        self.f.b.docs[str(self.f.root/'runs/validation_native_cost/launch_receipt.json')]['internal_exit']=True
        with mock.patch.object(c,'NATIVE_QUEUES',{s:r['sha256'] for s,r in self.f.native_queues.items()}):
            with self.assertRaisesRegex(ValueError,'strict_double1'):self.f.failed()
    def test_second_failure_review_must_cover_actual_metadata(self):
        self.f.b.docs[self.f.request['resource_failure_review']['path']]['reviewed_sha256']={}
        with mock.patch.object(c,'NATIVE_QUEUES',{s:r['sha256'] for s,r in self.f.native_queues.items()}):
            with self.assertRaisesRegex(ValueError,'review_covers_actual'):self.f.failed()
    def test_second_failure_successes_cannot_include_tune(self):
        self.f.b.docs[str(self.f.root/'runs/validation_native_cost/completion.json')]['jobs'][0]['id']='validation_C10_s193_single_original_val'
        with mock.patch.object(c,'NATIVE_QUEUES',{s:r['sha256'] for s,r in self.f.native_queues.items()}):
            with self.assertRaisesRegex(ValueError,'four_completed_C5'):self.f.failed()
    def test_local_review_alias_accepted_and_dual_conflict_rejected(self):
        pins={str(Path(self.f.protocol['local_root'])/Path(p).relative_to(self.f.root)):h for p,h in self.f.readers.items()}
        doc=dict(status='PASS',independent=True,reviewed_sha256=pins)
        c.reviewed_alias(doc,self.f.readers,self.f.root,self.f.protocol,'reader')
        doc['reviewed_sha256'][next(iter(self.f.readers))]='0'*64
        with self.assertRaisesRegex(ValueError,'reader_current_SHA'):c.reviewed_alias(doc,self.f.readers,self.f.root,self.f.protocol,'reader')
    def test_jj_bool_exit_rejected(self):
        self.f.b.docs[self.f.request['operator_JJ_execution']['path']]['actual_exit']=False
        with self.assertRaisesRegex(ValueError,'strict_actual_exit0'):c.verify_JJ(self.f.b,self.f.request,'operator',self.f.operators,43)
    def test_jj_wrong_raw_count_rejected(self):
        self.f.b.docs[self.f.request['reader_JJ_log']['path']]='Ran 42 tests in 1.0s\nOK\n'
        with self.assertRaisesRegex(ValueError,'raw_JJ_suite'):c.verify_JJ(self.f.b,self.f.request,'reader',self.f.readers,43)
    def test_jj_source_drift_rejected(self):
        self.f.b.docs[self.f.request['operator_JJ_execution']['path']]['source_end_sha256']={}
        with self.assertRaisesRegex(ValueError,'JJ_actual_sources'):c.verify_JJ(self.f.b,self.f.request,'operator',self.f.operators,43)

if __name__=='__main__':unittest.main()

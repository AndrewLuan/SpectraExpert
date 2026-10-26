"""功能概述：机制具体准入人工组件fixture及拒绝攻击，不读真实结果/张量。
原c09完整链另由其40项已审suite覆盖；本fixture显式mock该delegate和val16metadata，
只检新增213/source16/q36精确差分、五源/JJ61/失败策略；不冒称全实际main测试。
"""
import copy,json,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import review_fixed_mechanism_admission as c
from test_prepare_mechanism_fixed_rows_recovery import Fixture as PreparationFixture

class MemoryBundle:
    def __init__(self,reader):
        self.data=reader.data;self.hashes={p:c.digest(raw) for p,raw in self.data.items()};self.remote=dict(self.hashes);self.reads={}
    def raw(self,p,expected=None):
        p=str(p);raw=self.data[p];h=c.digest(raw)
        c.need(expected is None or expected==h,'fixture_actual_byte_SHA');self.reads[p]=h;return raw
    def json(self,p,expected=None):return json.loads(self.raw(p,expected))
    def ref(self,r):return self.json(r['path'],r['sha256'])
    def pin(self,p,h):c.need(self.remote.get(str(p))==h,'fixture_current_pin')

class AdmissionFixture:
    def __init__(self):
        f=self.f=PreparationFixture();self.refs=f.refs;r=f.root;rd=f.reader
        protocol=rd.ref(f.refs['protocol']);protocol['local_root']='/artificial/local';f.replace('protocol',protocol)
        f.queue['source_sha256'].pop(str(r/'old24.py'))
        f.queue['source_sha256'][f.refs['protocol']['path']]=f.refs['protocol']['sha256']
        operators={}
        for index,role,name in [(22,'supervisor','supervise_resource_ready_queue.py'),(23,'launcher','launch_resource_ready_stage.py')]:
            f.queue['source_sha256'].pop(str(r/f'old{index}.py'))
            ref=rd.add(r/name,Path(name).stem.encode())
            f.queue['source_sha256'][ref['path']]=ref['sha256'];operators[role]=ref
        f.queue['operator_paths']=operators
        f.queue['resource_wait']={'seconds':120,'poll_seconds':0.2}
        f.replace('prior_queue',f.queue)
        done=dict(status='FAILED',pid=403,queue_sha256=f.refs['prior_queue']['sha256'],external_exit=1)
        outer=dict(child_pid=403,wrapper_pid=402,queue_sha256=f.refs['prior_queue']['sha256'],internal_exit=1,external_exit=1)
        dr=rd.add(r/'runs/mechanism_resource_ready/completion.json',done);lr=rd.add(r/'runs/mechanism_resource_ready/launch_receipt.json',outer)
        failure=dict(status='PASS',independent=True,scope='failed_mechanism_execution_preservation_and_exposure_not_recovery_admission',
          queue_sha256=f.refs['prior_queue']['sha256'],all_four_dispatched=True,completed_cells=0,complete_output_lines_lower_bound=12,
          actual_input_pins_preserved=213,driver_pid=403,wrapper_pid=402,
          reviewed_sha256={f.refs['prior_queue']['path']:f.refs['prior_queue']['sha256'],dr['path']:dr['sha256'],lr['path']:lr['sha256']})
        f.replace('failure_review',failure)
        strategy=rd.ref(f.refs['recovery_strategy']);strategy.update(recovery='all4_fresh_in_one_new_run_after_mechanism_only_adapter_and_actual_queue_independent_PASS',prior_partial_predictions_used_for_selection=False,test_read=False)
        f.replace('recovery_strategy',strategy)
        log=rd.add(r/'JJ_execution.log',b'Ran 18 tests in 1.0s\nOK\nRan 12 tests in 1.0s\nOK\nRan 31 tests in 1.0s\nOK\n');f.refs['JJ_log']=log
        jj=rd.ref(f.refs['JJ_execution']);jj.update(actual_ssh_exit=0,log_sha256=log['sha256'],all_deployed_sha256={'synthetic':'same'},
          all_deployed_end_sha256={'synthetic':'same'},dependencies_sha256={'synthetic':'same'},dependencies_end_sha256={'synthetic':'same'},all_sources_unchanged=True)
        f.replace('JJ_execution',jj);f.request['JJ_test_count']=61
        binding,q,_=f.prepare()
        self.bp=rd.add(r/'mechanism_fixed_rows_bindings.json',binding);self.qp=rd.add(r/'mechanism_fixed_rows_queue.json',q)
        for p,h in q['source_sha256'].items():
            if p not in rd.data:
                payload=Path(p).stem.encode();self.assert_sha(payload,h);rd.data[p]=payload
        self.b=MemoryBundle(rd);self.b.remote.update(q['input_sha256']);self.request=f.request;self.prior={'synthetic':'delegate contract separately tested'}
    @staticmethod
    def assert_sha(raw,h):
        if c.digest(raw)!=h:raise AssertionError(('fixture_source_bytes',raw,h))
    def check(self):
        with patch.object(c,'BINDING_SHA',self.f.refs['prior_bindings']['sha256']),patch.object(c,'QUEUE_SHA',self.f.refs['prior_queue']['sha256']),\
          patch.object(c,'FAILURE_SHA',self.f.refs['failure_review']['sha256']),patch.object(c,'STRATEGY_SHA',self.f.refs['recovery_strategy']['sha256']),\
          patch.object(c.resource,'verify',return_value={}) as delegated,patch.object(c.results,'actual_stage',return_value=({},dict(jobs=16,strict_double_exit0=True))) as validated:
            out=c.verify(self.b,self.request,self.prior,object());delegated.assert_called_once();validated.assert_called_once();return out
    def mutate(self,ref,change):
        doc=self.b.ref(ref);change(doc);raw=json.dumps(doc).encode();self.b.data[ref['path']]=raw;h=c.digest(raw);self.b.hashes[ref['path']]=self.b.remote[ref['path']]=h
        # Only mutable artifact simulations update request refs. Queue/binding are identified by canonicalpath.
        if ref not in (self.qp,self.bp):ref['sha256']=h

class FixedMechanismAdmissionTests(unittest.TestCase):
    def setUp(self):self.f=AdmissionFixture()
    def test_exact_new4_q36_binding16_and213(self):
        out=self.f.check();self.assertEqual(out['jobs'],4);self.assertEqual(len(out['reviewed_sha256']),37);self.assertFalse(out['validation_rerun'])
    def test_extra_scientific_argument_rejected(self):
        self.f.mutate(self.f.qp,lambda q:q['jobs'][0]['command'].extend(['--other','value']))
        with self.assertRaisesRegex(ValueError,'only_leaf_binding_output_access_change'):self.f.check()
    def test_changed_model_or_cache_registry_rejected(self):
        self.f.mutate(self.f.bp,lambda b:b['roles'].update(other='substituted'))
        with self.assertRaisesRegex(ValueError,'all_scientific_binding_fields_unchanged'):self.f.check()
    def test_dropped_scientific_input_rejected(self):
        self.f.mutate(self.f.qp,lambda q:q['input_sha256'].pop(next(iter(q['input_sha256']))))
        with self.assertRaisesRegex(ValueError,'unchanged213'):self.f.check()
    def test_old_output_reuse_rejected(self):
        self.f.mutate(self.f.qp,lambda q:q['jobs'][0].update(job_out='/old/failure/output'))
        with self.assertRaisesRegex(ValueError,'only_leaf_binding_output_access_change'):self.f.check()
    def test_rawlog_must_be_bound_to_JJ_receipt(self):
        self.f.mutate(self.f.refs['JJ_execution'],lambda jj:jj.update(log_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'receipt_bound_raw_log'):self.f.check()
    def test_full_base_failure_propagates(self):
        with patch.object(c.resource,'verify',side_effect=ValueError('original_training_missing')):
            with self.assertRaisesRegex(ValueError,'original_training_missing'):c.verify(self.f.b,self.f.request,self.f.prior,object())

if __name__=='__main__':unittest.main()

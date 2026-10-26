"""功能概述：机制恢复metadata准备器的人工4格/213input不变与拒绝边界测试。
旧绑定/队列SHA常量仅在人工fixture中patch；不读真实tensor/prediction，不派发。
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_mechanism_fixed_rows_recovery as m

class MemoryReader:
    def __init__(self):self.data={};self.reads={}
    def add(self,path,value):
        raw=value if type(value) is bytes else m.encode(value);path=str(path);self.data[path]=raw;return dict(path=path,sha256=m.digest(raw))
    def raw(self,ref):
        raw=self.data[ref['path']];m.need(m.digest(raw)==ref['sha256'],'fixture_SHA');self.reads[ref['path']]=ref['sha256'];return raw
    def ref(self,ref):return json.loads(self.raw(ref))

class Fixture:
    def __init__(self):
        self.root=r=Path('/artificial/mechanism');self.reader=rd=MemoryReader();refs={}
        refs['protocol']=rd.add(r/'protocol.json',dict(remote_root=str(r),test_allowed=False,promotion=False))
        self.binding=dict(schema='cap10_execution_bindings_20261024',source_sha256={str(r/'native_cost_evaluation'/('source'+str(i)+'.py')):m.digest(str(i).encode()) for i in range(15)},input_sha256={str(r/'science'/('input'+str(i)+'.pt')):m.digest(('input'+str(i)).encode()) for i in range(213)},roles={'original':'same'},caches={'original':'same'},fixed_train_pairs={'original':'same'},training_run='unchanged')
        refs['prior_bindings']=rd.add(r/'native_cost_validation_bindings.json',self.binding)
        jobs=[]
        for cap,seed in [(5,193),(5,211),(10,193),(10,211)]:
            jid=f'mechanism_C{cap}_s{seed}_dual_train';out=str(r/'mechanism_resource_ready'/jid);access=str(r/'access_logs/mechanism_resource_ready'/(jid+'.json'))
            command=[str(r/'native_cost_evaluation/evaluate_cap_ranker.py'),'--cap',str(cap),'--seed',str(seed),'--pool','dual','--cohort','train','--bindings',refs['prior_bindings']['path'],'--out',out]
            jobs.append(dict(id=jid,job_out=out,access_log=access,command=command,argv=['python','-B','isolation','--bindings',refs['prior_bindings']['path'],'--access-log',access,'--output-root',out,'--']+command,allowed_reads=list(self.binding['input_sha256'])[:3],gpu=len(jobs)))
        self.queue=dict(phase='mechanism',bindings=refs['prior_bindings']['path'],jobs=jobs,input_sha256=self.binding['input_sha256'],source_sha256={str(r/('old'+str(i)+'.py')):m.digest(('old'+str(i)).encode()) for i in range(25)},run_name='mechanism_resource_ready',analysis_tools={'old':'same'})
        self.queue['source_sha256'][refs['protocol']['path']]=refs['protocol']['sha256']
        refs['prior_queue']=rd.add(r/'mechanism_resource_ready_queue.json',self.queue)
        refs['failure_review']=rd.add(r/'mechanism_identity_failure_independent_review.json',dict(status='PASS',independent=True))
        refs['recovery_strategy']=rd.add(r/'mechanism_identity_recovery_strategy.json',dict(new_training=False,validation_rerun=False,fixed_train_pairs_unchanged=400,fixed_train_pairs_per_seed=200,original87_unchanged=87,compare_all_prior_complete_mechanism_prediction_lines_to_new_semantic_rows=True))
        refs['evaluator']=rd.add(r/'mechanism_fixed_rows/evaluate_cap_ranker.py',b'# synthetic leaf source')
        refs['preparer']=rd.add(r/'prepare_mechanism_fixed_rows_recovery.py',Path(m.__file__).read_bytes())
        reader_sources={name:rd.add(r/'mechanism_fixed_rows'/name,b'# synthetic reader') for name in m.READERS}
        sources={r['path']:r['sha256'] for r in [refs['evaluator'],refs['preparer'],*reader_sources.values()]}
        refs['implementation_review']=rd.add(r/'implementation_review.json',dict(status='PASS',independent=True,scope='cap10_mechanism_fixed_rows_recovery_implementation',reviewed_sha256=sources))
        refs['JJ_execution']=rd.add(r/'JJ_execution.json',dict(actual_exit=0,source_sha256=sources,source_end_sha256=sources))
        refs['JJ_log']=rd.add(r/'JJ_execution.log',b'Ran 17 tests in 1.0s\n\nOK\n')
        self.refs=refs;self.request=dict(schema='cap10_mechanism_fixed_rows_recovery_request_20261025',root=str(r),refs=refs,reader_sources=reader_sources,JJ_test_count=17)
    def prepare(self):
        with patch.object(m,'OLD_BINDING',self.refs['prior_bindings']['sha256']),patch.object(m,'OLD_QUEUE',self.refs['prior_queue']['sha256']),patch.object(m,'FAILURE',self.refs['failure_review']['sha256']),patch.object(m,'STRATEGY',self.refs['recovery_strategy']['sha256']):
            return m.prepare(self.reader,self.request)
    def replace(self,key,value):self.refs[key]=self.reader.add(self.refs[key]['path'],value)

class PreparationTests(unittest.TestCase):
    def setUp(self):self.f=Fixture()
    def test_complete4_only_metadata_original213_and_roles_exact(self):
        b,q,required=self.f.prepare();old=self.f.binding
        self.assertEqual(set(b)-set(old),{'mechanism_evaluator','mechanism_recovery'})
        self.assertEqual({k:v for k,v in b.items() if k not in ('source_sha256','mechanism_evaluator','mechanism_recovery')},{k:v for k,v in old.items() if k!='source_sha256'})
        self.assertEqual(len(b['input_sha256']),213);self.assertEqual(b['input_sha256'],q['input_sha256']);self.assertEqual(len(q['jobs']),4)
        self.assertFalse(any(Path(path).suffix=='.pt' for path in self.f.reader.reads));self.assertIn(str(self.f.root/'mechanism_fixed_rows_queue.json'),required)
        self.assertEqual({x['gpu'] for x in q['jobs']},{x['gpu'] for x in self.f.queue['jobs']})
    def test_old_implementation_source_path_cannot_be_leaf(self):
        self.f.refs['evaluator']['path']=str(self.f.root/'native_cost_evaluation/evaluate_cap_ranker.py')
        with self.assertRaisesRegex(ValueError,'exact_mechanism_leaf_path'):self.f.prepare()
    def test_missing_source_in_implementation_review_refused(self):
        value=self.f.reader.ref(self.f.refs['implementation_review']);value['reviewed_sha256']={};self.f.replace('implementation_review',value)
        with self.assertRaisesRegex(ValueError,'complete_implementation_review'):self.f.prepare()
    def test_bool_JJ_exit_and_log_failure_refused(self):
        value=self.f.reader.ref(self.f.refs['JJ_execution']);value['actual_exit']=False;self.f.replace('JJ_execution',value)
        with self.assertRaisesRegex(ValueError,'actual_JJ_same_sources'):self.f.prepare()
        self.f=Fixture();self.f.refs['JJ_log']=self.f.reader.add(self.f.refs['JJ_log']['path'],b'Ran 17 tests in 1.0s\nFAILED\n')
        with self.assertRaisesRegex(ValueError,'actual_JJ_suite'):self.f.prepare()
    def test_actual_fresh_process_suite_counts_sum_and_each_OK(self):
        self.f.refs['JJ_log']=self.f.reader.add(self.f.refs['JJ_log']['path'],b'Ran 8 tests in 1.0s\nOK\nRan 9 tests in 1.0s\nOK\n')
        self.f.prepare()
        for log in (b'Ran 8 tests in 1.0s\nOK\nRan 9 tests in 1.0s\nFAILED\n',b'Ran 8 tests in 1.0s\nOK\nRan 8 tests in 1.0s\nOK\n'):
            self.f.refs['JJ_log']=self.f.reader.add(self.f.refs['JJ_log']['path'],log)
            with self.assertRaisesRegex(ValueError,'actual_JJ_suite'):self.f.prepare()
    def test_validation_rerun_strategy_refused(self):
        value=self.f.reader.ref(self.f.refs['recovery_strategy']);value['validation_rerun']=True;self.f.replace('recovery_strategy',value)
        with self.assertRaisesRegex(ValueError,'mechanism_only_recovery_strategy'):self.f.prepare()
    def test_missing_duplicate_or_wrong_mechanism_matrix_refused(self):
        for change in ('missing','duplicate','validation'):
            self.f=Fixture();q=copy.deepcopy(self.f.queue)
            if change=='missing':q['jobs'].pop()
            elif change=='duplicate':q['jobs'][1]=copy.deepcopy(q['jobs'][0])
            else:q['jobs'][0]['command'][q['jobs'][0]['command'].index('--cohort')+1]='original_val'
            self.f.replace('prior_queue',q)
            with self.assertRaises(ValueError):self.f.prepare()
    def test_original_input_shortening_refused(self):
        b=copy.deepcopy(self.f.binding);b['input_sha256'].pop(next(iter(b['input_sha256'])));self.f.replace('prior_bindings',b)
        with self.assertRaisesRegex(ValueError,'unchanged_213_inputs'):self.f.prepare()
    def test_no_tensor_alias_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();tensor=root/'features.pt';tensor.write_bytes(b'explicit fake');alias=root/'alias.json';alias.symlink_to(tensor)
            mp=root/'map.json';mp.write_bytes(m.encode(dict(files={'/canonical/meta.json':str(alias)},sha256={'/canonical/meta.json':m.digest(tensor.read_bytes())})))
            reader=m.Reader(dict(path=str(mp),sha256=m.digest(mp.read_bytes())))
            with self.assertRaisesRegex(ValueError,'unsafe_alias'):reader.raw(dict(path='/canonical/meta.json',sha256=m.digest(b'explicit fake')))
            self.assertEqual(reader.reads,{})

if __name__=='__main__':unittest.main()

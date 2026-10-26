"""功能概述：用本地人工metadata检验独立结果reader的SHA/路径/类型边界，不接触科学产物。"""
import json,tempfile,unittest,copy,types
from pathlib import Path
import review_cap_validation_mechanism_results as r

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cap_result_reader_');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.file=self.root/'payload/validation/cell/predictions.jsonl'
        self.file.parent.mkdir(parents=True);self.file.write_text('{"rank":1}\n')
        self.remote='/research/controlled/validation/cell/predictions.jsonl';h=r.digest(self.file.read_bytes())
        self.m=dict(files={self.remote:str(self.file)},sha256={self.remote:h},hash_only_inputs={'/research/cache/train/features.pt':'a'*64},
          remote_before_sha256={self.remote:h,'/research/cache/train/features.pt':'a'*64},
          remote_after_sha256={self.remote:h,'/research/cache/train/features.pt':'a'*64},
          remote_inventory_before_exit=0,remote_inventory_after_exit=0,rsync_exit=0)
        self.path=self.root/'manifest.json';self.save()
    def save(self):self.path.write_text(json.dumps(self.m))
    def test_exact_copied_bytes_and_hash_only_not_opened(self):
        e=r.Evidence([self.path]);self.assertEqual(e.json(self.remote),{'rank':1});e.pin('/research/cache/train/features.pt','a'*64);e.finish()
    def test_read_race_and_finish_drift_reject(self):
        e=r.Evidence([self.path]);self.file.write_text('{"rank":2}\n')
        with self.assertRaisesRegex(ValueError,'consumed_bytes_SHA'):e.raw(self.remote)
        with self.assertRaisesRegex(ValueError,'copied_bytes_changed'):e.finish()
    def test_remote_before_after_disagreement_reject(self):
        self.m['remote_after_sha256']=dict(self.m['remote_after_sha256']);self.m['remote_after_sha256'][self.remote]='b'*64;self.save()
        with self.assertRaisesRegex(ValueError,'remote_begin_end'):r.Evidence([self.path])
    def test_boolean_exit_is_not_actual_zero(self):
        self.m['rsync_exit']=False;self.save()
        with self.assertRaisesRegex(ValueError,'strict_int_EXIT0'):r.Evidence([self.path])
    def test_hash_only_cannot_be_promoted_to_readable_input(self):
        e=r.Evidence([self.path])
        with self.assertRaisesRegex(ValueError,'missing_copied_metadata'):e.raw('/research/cache/train/features.pt')
    def test_symlink_to_forbidden_tensor_rejects_before_read(self):
        forbidden=self.root/'sealed/private.pt';forbidden.parent.mkdir();forbidden.write_text('not scientific')
        self.file.unlink();self.file.symlink_to(forbidden)
        with self.assertRaisesRegex(ValueError,'test_or_sealed_forbidden'):r.Evidence([self.path])
    def test_undeclared_union_missing_reject(self):
        self.m['hash_only_inputs']={};self.save()
        with self.assertRaisesRegex(ValueError,'complete_collector_union'):r.Evidence([self.path])
    def test_missing_or_boolean_numeric_compare_reject(self):
        with self.assertRaisesRegex(ValueError,'missing:'):r.compare({},dict(eligible=True),'gate')
        with self.assertRaisesRegex(ValueError,'exact:'):r.compare(False,0,'exit')
    def test_metadata_view_binds_original_without_opening_tensor_copy(self):
        original=self.root/'original.json';original.write_text(json.dumps(self.m));h=r.digest(original.read_bytes())
        self.m['source_manifest']=dict(path=str(original),sha256=h);self.save()
        e=r.Evidence([self.path]);self.assertEqual(e.localpins[str(original)],h);e.finish()
    def test_metadata_view_cannot_shrink_actual_remote_union(self):
        original=self.root/'original.json';original.write_text(json.dumps(self.m))
        self.m['source_manifest']=dict(path=str(original),sha256=r.digest(original.read_bytes()))
        self.m['remote_before_sha256']={self.remote:self.m['sha256'][self.remote]};self.m['remote_after_sha256']=dict(self.m['remote_before_sha256']);self.m['hash_only_inputs']={};self.save()
        with self.assertRaisesRegex(ValueError,'metadata_view_preserves_full_actual_union'):r.Evidence([self.path])
    def test_metadata_view_original_byte_drift_rejected(self):
        original=self.root/'original.json';original.write_text(json.dumps(self.m))
        self.m['source_manifest']=dict(path=str(original),sha256=r.digest(original.read_bytes()));self.save();original.write_text('{}')
        with self.assertRaisesRegex(ValueError,'metadata_view_original_manifest_SHA'):r.Evidence([self.path])

class PriorSuccessSemanticTests(unittest.TestCase):
    def setUp(self):
        self.root=Path('/artificial/cap');self.docs={};self.payload={};self.jobs=[];completed=[]
        fields=r.calc.FIELDS
        for seed in (193,211):
            for pool in ('single','dual'):
                jid=f'validation_C5_s{seed}_{pool}_original_val'
                rows=[dict(query_identity=dict(zip(fields,(i,'observed','source',str(i),'C',str(i)))),rank=1,
                  natural_candidate_costs=[float(i),0.],trace=dict(alpha=1.,pid=5),pid=4) for i in range(904)]
                raw=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
                old=self.root/'validation_native_cost'/jid/'predictions.jsonl';new=self.root/'validation_resource_ready'/jid/'predictions.jsonl'
                self.payload[str(old)]=raw;self.payload[str(new)]=raw
                self.jobs.append(dict(id=jid,job_out=str(new.parent)))
                completed.append(dict(id=jid,prediction_sha256=r.digest(raw)))
        self.args=types.SimpleNamespace(resource_failure_review='/artificial/failure.json',recovery_strategy='/artificial/strategy.json')
        self.docs[self.args.resource_failure_review]=dict(status='PASS',independent=True,completed_validation_cells=4,completed_cells=completed)
        self.docs[self.args.recovery_strategy]=dict(recovery='all16_fresh_in_one_new_run_after_operator_and_actual_queue_independent_PASS',
          prior_completed_cells_used_for_selection=False,prior_partial_predictions_used_for_selection=False,
          compare_all_four_prior_completed_cells_against_new_semantic_predictions=True)
        self.e=types.SimpleNamespace(json=lambda p:copy.deepcopy(self.docs[str(p)]),raw=lambda p:self.payload[str(p)])
    def test_all4_3616_semantics_exact(self):
        result=r.compare_prior_successes(self.e,self.root,self.args,self.jobs)
        self.assertEqual(result['rows_compared'],3616);self.assertFalse(result['prior_results_used_for_selection'])
    def test_only_process_time_fields_may_differ(self):
        path=str(Path(self.jobs[0]['job_out'])/'predictions.jsonl');rows=[json.loads(line) for line in self.payload[path].splitlines()]
        for row in rows:row.update(pid=100,finished_utc='new process');row['trace']['pid']=101
        self.payload[path]=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
        self.assertTrue(r.compare_prior_successes(self.e,self.root,self.args,self.jobs)['all_semantic_fields_exact'])
    def test_cost_rank_trace_identity_change_rejected(self):
        path=str(Path(self.jobs[0]['job_out'])/'predictions.jsonl');rows=[json.loads(line) for line in self.payload[path].splitlines()]
        rows[-1]['trace']['alpha']=.5;self.payload[path]=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
        with self.assertRaisesRegex(ValueError,'all_prior_completed_semantic_fields_exact'):r.compare_prior_successes(self.e,self.root,self.args,self.jobs)
    def test_prior_prediction_byteSHA_must_match_preservation(self):
        path=str(self.root/'validation_native_cost'/self.jobs[0]['id']/'predictions.jsonl');self.payload[path]+=b'\n'
        with self.assertRaisesRegex(ValueError,'preserved_prior_prediction_SHA'):r.compare_prior_successes(self.e,self.root,self.args,self.jobs)
    def test_no_result_dependent_partial_reuse(self):
        self.docs[self.args.recovery_strategy]['prior_partial_predictions_used_for_selection']=True
        with self.assertRaisesRegex(ValueError,'locked_all16_semantic_comparison'):r.compare_prior_successes(self.e,self.root,self.args,self.jobs)

class PriorMechanism12LineTests(unittest.TestCase):
    def setUp(self):
        self.root=Path('/artificial/cap');self.payload={};self.jobs=[];cells=[];self.hashes={}
        for cap in (5,10):
            for seed in (193,211):
                jid=f'mechanism_C{cap}_s{seed}_dual_train'
                rows=[dict(query_identity=dict(zip(r.calc.FIELDS,(i,'observed','source',str(i),'C',str(i)))),rank=1,trace=dict(alpha=1.,pid=5)) for i in range(200)]
                raw=b''.join(json.dumps(row).encode()+b'\n' for row in rows[:3]);fresh=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
                self.payload[str(self.root/'mechanism_resource_ready'/jid/'predictions.jsonl')]=raw
                new=self.root/'mechanism_fixed_manifest'/jid;self.payload[str(new/'predictions.jsonl')]=fresh;self.jobs.append(dict(id=jid,job_out=str(new)))
                cells.append(dict(id=jid,prediction_sha256=r.digest(raw),complete_output_lines=3,ends_in_newline=True))
        failure=dict(status='PASS',independent=True,completed_cells=0,complete_output_lines_lower_bound=12,cells=cells)
        fp='/artificial/mechanism_failure.json';sp='/artificial/mechanism_strategy.json';self.hashes[fp]='a'*64
        strategy=dict(schema='cap10_fixed_mechanism_identity_recovery_strategy_20261025',
          recovery='all4_fresh_in_one_new_run_after_mechanism_only_adapter_and_actual_queue_independent_PASS',
          prior_partial_predictions_used_for_selection=False,validation_rerun=False,
          compare_all_prior_complete_mechanism_prediction_lines_to_new_semantic_rows=True,
          failure_review=dict(path=fp,sha256='a'*64))
        self.docs={fp:failure,sp:strategy};self.args=types.SimpleNamespace(mechanism_failure_review=fp,mechanism_recovery_strategy=sp)
        self.e=types.SimpleNamespace(json=lambda p:copy.deepcopy(self.docs[str(p)]),raw=lambda p:self.payload[str(p)],hashes=self.hashes)
    def check(self):return r.compare_prior_mechanism_lines(self.e,self.root,self.args,self.jobs)
    def test_exact12_old_complete_lines_match_new800_rows(self):self.assertEqual(self.check()['complete_lines_compared'],12)
    def test_changed_semantic_cost_rejected(self):
        p=str(Path(self.jobs[0]['job_out'])/'predictions.jsonl');rows=[json.loads(line) for line in self.payload[p].splitlines()];rows[0]['rank']=2
        self.payload[p]=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
        with self.assertRaisesRegex(ValueError,'prior12_mechanism_semantics_exact'):self.check()
    def test_original_failed_bytes_not_replaced(self):
        p=next(k for k in self.payload if '/mechanism_resource_ready/' in k);self.payload[p]+=b'\n'
        with self.assertRaisesRegex(ValueError,'original3_complete_mechanism_lines_SHA'):self.check()
    def test_mechanism_lines_cannot_be_used_for_selection(self):
        self.docs[self.args.mechanism_recovery_strategy]['prior_partial_predictions_used_for_selection']=True
        with self.assertRaisesRegex(ValueError,'locked12_line_semantic_only_scope'):self.check()
    def test_failure_review_SHA_must_match_locked_strategy(self):
        self.hashes[self.args.mechanism_failure_review]='b'*64
        with self.assertRaisesRegex(ValueError,'mechanism_strategy_binds_actual_failure_review'):self.check()

if __name__=='__main__':unittest.main()

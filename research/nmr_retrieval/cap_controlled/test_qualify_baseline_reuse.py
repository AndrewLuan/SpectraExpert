"""功能概述：baseline 四格整体复用资格的纯 JSON 合成测试。

所有 checkpoint/cache/prediction 引用均指向不存在的合成路径；没有张量或预测文件。
Fixture 生成完整旧冻结映射/队列/四协议/四双退出收据/独审引用与始末 SHA 声明，
攻击会重新绑定相关元数据 SHA，避免仅依靠旧文件摘要意外失败。测试无 SSH、
无实际 qualifier inventory、无研究模块或正式科学任务。
"""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qualify_baseline_reuse as q


class Fixture:
    def __init__(self, directory):
        self.root=Path(directory).resolve()
        self.refs={}
        self.expected=[]
        identity={k:q.content_sha(k) for k in q.IDENTITIES}
        sources={'/science/nmr_transfer/'+name:q.content_sha(name) for name in q.SOURCE_NAMES}
        role_by_seed={}
        for seed in (193,211):
            role_by_seed[seed]={}
            for role in ('no_aug','aug'):
                base=f'/science/fixed/s{seed}/{role}'
                artifact=lambda name:dict(path=base+'/'+name,sha256=q.content_sha(base+'/'+name))
                role_by_seed[seed][role]=dict(checkpoint=artifact('best.pt'),centers=artifact('centers.json'),
                    center_rows=artifact('center_rows.jsonl'),alpha=1.,
                    tensor_sha256={name:q.content_sha(base+'/'+name) for name in q.ROLE_DIGESTS})
        for config,seed in sorted(q.CELLS):
            recall=config.removeprefix('fixed_')
            cache=f'/science/sealed/features/s{seed}/{recall}/features.pt'
            self.expected.append(dict(config=config,seed=seed,recall=recall,
                cache=dict(path=cache,sha256=q.content_sha(cache)),checkpoint_roles=copy.deepcopy(role_by_seed[seed]),
                fixed_test_identity=copy.deepcopy(identity),query_order_sha256=q.content_sha('query-order'),
                candidate_order_sha256=q.content_sha('candidate-order'),source_sha256=copy.deepcopy(sources),
                conditions=dict(cohort='test',query_count=1371,base_queries=1000,variant_counts=q.VARIANTS,
                    candidate_count=114982,natural_pool_size=512,appended_training_positives=0)))
        self.prior=copy.deepcopy(self.expected)
        self.protocol=dict(schema='cap10_controlled_protocol_20261024',
                           test_preregistration=dict(baseline_reuse_mode='all_four_or_none',matrix_cells=12))
        self.state_roles={str(seed):copy.deepcopy(role_by_seed[seed]) for seed in (193,211)}
        self.missing=set()
        self.old_cell_omissions=set()
        self.protocol_mutation=lambda value,index:None
        self.receipt_mutation=lambda value,index:None
        self.review_mutation=lambda value:None
        self.inventory_mutation=lambda value:None
        self.lock_mutation=lambda value:None
        self.admission_mutation=lambda value:None
        self.state_mutation=lambda value:None
        self.impl_mutation=lambda value:None

    def put(self,name,value):
        path=self.root/(name+'.json');path.write_text(json.dumps(value,sort_keys=True)+'\n')
        ref=dict(path='/registered_metadata/'+name+'.json',local_path=str(path),sha256=q.digest(path.read_bytes()))
        self.refs[name]=ref
        return ref

    def build(self):
        self.refs={}
        pref=self.put('current_protocol',self.protocol)
        lock=dict(schema='cap10_validation_lock_20261024',status='LOCKED',selection_scope='train_validation_only',
                  test_reselection_allowed=False,promotion=False,protocol_sha256=pref['sha256'],
                  preselected='fixedcenter_mixed_listwise_single')
        self.lock_mutation(lock);lref=self.put('validation_lock',lock)
        own=str(Path(q.__file__).resolve())
        impl=dict(status='PASS',independent=True,scope='baseline_reuse_metadata_implementation',
                  reviewed_sha256={own:q.digest(Path(own).read_bytes())})
        self.impl_mutation(impl);iref=self.put('implementation_review',impl)
        scientific={}
        for cell in self.prior:
            scientific.update(cell['source_sha256']);p,h=q.canonical_ref(cell['cache']);scientific[p]=h
            for role in cell['checkpoint_roles'].values():
                for name in ('checkpoint','centers','center_rows'):
                    p,h=q.canonical_ref(role[name]);scientific[p]=h
        fixed={}
        for cell in self.prior:
            roles=cell['checkpoint_roles']
            entry=dict(seed=cell['seed'],cohort='test',recall=cell['recall'],cache=cell['cache']['path'],
                       cache_sha256=cell['cache']['sha256'],center_artifacts=[])
            for role in ('no_aug','aug'):
                values=roles[role]
                entry[role+'_checkpoint']=values['checkpoint']['path']
                entry[role+'_sha256']=values['checkpoint']['sha256']
                entry['center_artifacts'].append(dict(centers=values['centers']['path'],centers_sha256=values['centers']['sha256'],
                    row_audit=values['center_rows']['path'],row_audit_sha256=values['center_rows']['sha256']))
            fixed.setdefault(q.LEGACY[cell['config']],{})[str(cell['seed'])]=entry
        f=dict(schema='nmr_pool_ranker_20261023',fixed_center_configurations=fixed,
            fixed_test_identity=self.prior[0]['fixed_test_identity'],candidate_count=114982,
            candidate_order_sha256=self.prior[0]['candidate_order_sha256'],test_variants=q.VARIANTS,
            evaluation_inputs_sha256=scientific)
        fref=self.put('prior_frozen',f)
        jobs=[];oldcells=[]
        for cell in self.prior:
            legacy=q.LEGACY[cell['config']];seed=cell['seed'];roles=cell['checkpoint_roles']
            out=f'/science/sealed/{legacy}/s{seed}/test'
            argv=['/venv/bin/python','/science/tools/evaluate_pool_fixed_center_sealed.py',
                '--source','/science','--config',legacy,'--cache',cell['cache']['path'],
                '--no-aug-checkpoint',roles['no_aug']['checkpoint']['path'],
                '--aug-checkpoint',roles['aug']['checkpoint']['path'],'--seed',str(seed),
                '--out',out,'--frozen-selection',fref['path']]
            pins={**scientific,fref['path']:fref['sha256']}
            jobs.append(dict(id=f'sealed_{legacy}_s{seed}',argv=argv,cwd='/science',out=out,inputs_sha256=pins))
            prediction=dict(path=out+'/predictions.jsonl',sha256=q.content_sha(out+'/historical-predictions'))
            oldcells.append(dict(config=cell['config'],seed=seed,predictions=prediction))
        queue=dict(schema='nmr_research_jobs_20261022',jobs=jobs)
        qref=self.put('prior_queue',queue)
        reviewed={fref['path']:fref['sha256'],qref['path']:qref['sha256'],**scientific}
        job_evidence=[]
        for index,(job,old) in enumerate(zip(jobs,oldcells)):
            protocol=dict(args=q.parse_legacy_argv(job['argv']),inputs_sha256=job['inputs_sha256'])
            self.protocol_mutation(protocol,index)
            protocol_ref=self.put('prior_protocol_'+str(index),protocol)
            receipt=dict(state='complete',internal_exit=0,external_exit=0,pid=12000+index,argv=job['argv'],cwd=job['cwd'],
                output=job['out'],started_utc='2026-10-24T11:39:00+00:00',finished_utc='2026-10-24T11:40:00+00:00',
                queue_source_sha256={qref['path']:qref['sha256']},requested_inputs_sha256=job['inputs_sha256'])
            self.receipt_mutation(receipt,index)
            receipt_ref=self.put('prior_receipt_'+str(index),receipt)
            old.update(protocol_ref=protocol_ref,receipt_ref=receipt_ref)
            for ref in (protocol_ref,receipt_ref,old['predictions']):reviewed[ref['path']]=ref['sha256']
            job_evidence.append(dict(config=q.LEGACY[old['config']],seed=old['seed'],pid=12000+index,
                internal_exit=0,external_exit=0,predictions_sha256=old['predictions']['sha256']))
        review=dict(status='PASS',independent=True,reviewed_sha256=reviewed,job_evidence=job_evidence)
        self.review_mutation(review);rref=self.put('prior_execution_review',review)
        states=dict(schema='fixed_baseline_role_state_proof_20261024',roles=self.state_roles,
            query_identity=dict(fixed_test_identity=self.prior[0]['fixed_test_identity'],
                query_order_sha256=self.prior[0]['query_order_sha256'],conditions=self.prior[0]['conditions']))
        self.state_mutation(states);sref=self.put('state_proof',states)
        srref=self.put('state_review',dict(status='PASS',independent=True,scope='fixed_baseline_state_and_identity_metadata',
            reviewed_sha256={sref['path']:sref['sha256']}))
        before=dict(scientific)
        before.update({old['predictions']['path']:old['predictions']['sha256'] for old in oldcells})
        inventory=dict(schema='cap10_baseline_reuse_hash_inventory_20261024',status='PASS',actual_exit=0,
            method='actual_sha256_before_and_after',before_sha256=before,after_sha256=copy.deepcopy(before),
            test_predictions_parsed=False,model_forward_calls=0)
        self.inventory_mutation(inventory);invref=self.put('current_hash_inventory',inventory)
        historical=dict(frozen_ref=fref,queue_ref=qref,execution_review_ref=rref,inventory_ref=invref,
                        state_proof_ref=sref,state_review_ref=srref,
                        cells=[c for c in oldcells if (c['config'],c['seed']) not in self.old_cell_omissions])
        for name in self.missing:historical.pop(name,None)
        metadata_sha={ref['path']:ref['sha256'] for ref in self.refs.values()}
        admission=dict(schema='cap10_baseline_metadata_admission_20261024',status='ADMITTED',explicit_root_authorization=True,
            phase='baseline_reuse_metadata_only',expected_cells_sha256=q.content_sha(self.expected),
            protocol_sha256=pref['sha256'],validation_lock_sha256=lref['sha256'],metadata_sha256=metadata_sha)
        self.admission_mutation(admission);aref=self.put('root_admission',admission)
        self.request=dict(schema=q.REQUEST_SCHEMA,expected_cells=self.expected,protocol_ref=pref,validation_lock_ref=lref,
                          implementation_review_ref=iref,admission_ref=aref,historical=historical)
        return self.request

    def execute(self):
        metadata=q.Metadata();result=q.qualify(self.build(),metadata);metadata.finish()
        return result


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.f=Fixture(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def assert_no_reuse(self,result,code=None):
        self.assertFalse(result['qualified']);self.assertEqual(result['registered_baseline_jobs_count'],4)
        self.assertEqual(result['new_evaluation_jobs_count'],12)
        self.assertEqual([row['qualified'] for row in result['baseline_cells']],[False]*4)
        if code:self.assertTrue(any(code in row['reason'] for row in result['findings']),result['findings'])
    def test_complete_four_cells_reuse_together(self):
        r=self.f.execute();self.assertTrue(r['qualified']);self.assertEqual(r['registered_baseline_jobs_count'],0)
        self.assertEqual(r['new_evaluation_jobs_count'],8);self.assertEqual(r['findings'],[])
    def test_absent_state_proof_explicit_all_four_rerun(self):
        self.f.missing={'state_proof_ref','state_review_ref'}
        self.assert_no_reuse(self.f.execute(),'missing_state_proof')
    def test_empty_historical_evidence_is_conservative_not_fabricated(self):
        request=self.f.build();request['historical']={}
        self.assert_no_reuse(q.qualify(request,q.Metadata()),'missing_frozen')
    def test_one_alpha_mismatch_cannot_reuse_other_three(self):
        self.f.state_roles['193']['no_aug']['alpha']=.75
        self.assert_no_reuse(self.f.execute(),'alpha_O_centers_full33scale')
    def test_full_scale_proof_missing_or_changed_fails(self):
        self.f.state_roles['211']['aug']['tensor_sha256'].pop('full33_scale_sha256')
        self.assert_no_reuse(self.f.execute(),'alpha_O_centers_full33scale')
    def test_single_new_adapter_source_SHA_change_fails_all_four(self):
        source=next(p for p in self.f.expected[0]['source_sha256'] if Path(p).name=='evaluate_pool_fixed_center_sealed.py')
        self.f.expected[0]['source_sha256'][source]='0'*64
        self.assert_no_reuse(self.f.execute(),'scientific_input_or_source_identity_changed')
    def test_source_set_cannot_omit_a_dependency(self):
        self.f.expected[0]['source_sha256'].pop(next(iter(self.f.expected[0]['source_sha256'])))
        self.assert_no_reuse(self.f.execute(),'source_set')
    def test_cache_seed_or_recall_identity_mismatch(self):
        self.f.expected[0]['cache']=copy.deepcopy(self.f.expected[1]['cache'])
        self.assert_no_reuse(self.f.execute(),'cache_identity')
    def test_swapped_checkpoint_role_sha_not_accepted(self):
        for cell in self.f.prior:
            if cell['seed']==193:
                cell['checkpoint_roles']['no_aug']['checkpoint']['sha256']='f'*64
        self.assert_no_reuse(self.f.execute(),'checkpoint_role_or_byteSHA')
    def test_query_mask_changed_not_intersection_fallback(self):
        for cell in self.f.expected:cell['fixed_test_identity']['variant_mask_content_sha256']='e'*64
        self.assert_no_reuse(self.f.execute(),'query_mask_variants_or_candidate_order')
    def test_explicit_query_order_proof_must_match(self):
        self.f.state_mutation=lambda value:value['query_identity'].update(query_order_sha256='d'*64)
        self.assert_no_reuse(self.f.execute(),'full1371_order_mask')
    def test_natural512_and_zero_injection_are_fixed_request_contract(self):
        self.f.expected[0]['conditions']['appended_training_positives']=1
        with self.assertRaisesRegex(ValueError,'natural1371'):self.f.execute()
    def test_false_is_not_real_integer_zero(self):
        self.f.receipt_mutation=lambda value,index:value.update(internal_exit=False) if index==0 else None
        self.assert_no_reuse(self.f.execute(),'prior_real_double0')
    def test_protocol_self_consistent_wrong_checkpoint_path_still_rejected(self):
        self.f.protocol_mutation=lambda value,index:value['args'].update(aug_checkpoint='/other/aug.pt') if index==0 else None
        self.assert_no_reuse(self.f.execute(),'protocol_matches_actual_argv')
    def test_wrong_independent_prior_prediction_hash_not_reused(self):
        def mutate(value):
            p=next(p for p in value['reviewed_sha256'] if p.endswith('/predictions.jsonl'))
            value['reviewed_sha256'][p]='a'*64
        self.f.review_mutation=mutate
        self.assert_no_reuse(self.f.execute(),'registered_prior_prediction')
    def test_current_prediction_inventory_mismatch_not_reused(self):
        def mutate(value):
            p=next(p for p in value['before_sha256'] if p.endswith('/predictions.jsonl'))
            value['before_sha256'][p]=value['after_sha256'][p]='b'*64
        self.f.inventory_mutation=mutate
        self.assert_no_reuse(self.f.execute(),'registered_prior_prediction')
    def test_inventory_must_be_actual_double_snapshot(self):
        self.f.inventory_mutation=lambda value:value['after_sha256'].pop(next(iter(value['after_sha256'])))
        with self.assertRaisesRegex(ValueError,'actual_stable_hash_inventory'):self.f.execute()
    def test_missing_one_prior_cell_registers_four(self):
        self.f.old_cell_omissions={('fixed_single',193)}
        self.assert_no_reuse(self.f.execute(),'missing_prior_cell')
    def test_validation_unlocked_rejects_before_any_old_metadata(self):
        self.f.lock_mutation=lambda value:value.update(status='PENDING')
        metadata=q.Metadata()
        with self.assertRaisesRegex(ValueError,'validation_must_be_locked'):q.qualify(self.f.build(),metadata)
        self.assertFalse(any('prior_' in p for p in metadata.before))
    def test_root_admission_is_explicit_and_expected_SHA_bound(self):
        self.f.admission_mutation=lambda value:value.update(expected_cells_sha256='0'*64)
        metadata=q.Metadata()
        with self.assertRaisesRegex(ValueError,'root_metadata_admission'):q.qualify(self.f.build(),metadata)
        self.assertFalse(any('prior_' in p for p in metadata.before))
    def test_unrelated_implementation_PASS_is_rejected(self):
        self.f.impl_mutation=lambda value:value.update(reviewed_sha256={})
        with self.assertRaisesRegex(ValueError,'implementation_review_required'):self.f.execute()
    def test_source_is_pinned_before_first_metadata_read(self):
        fake=self.f.root/'temporary_qualifier.py';fake.write_text('# initial source\n')
        changed=b'# distinct reviewed source after load starts\n'
        with patch.object(q,'__file__',str(fake)):
            self.f.impl_mutation=lambda value:value['reviewed_sha256'].update({str(fake):q.digest(changed)})
            request=self.f.build();metadata=q.Metadata();original=metadata.load
            def mutate_after_source_pin(ref):
                fake.write_bytes(changed)
                return original(ref)
            with patch.object(metadata,'load',side_effect=mutate_after_source_pin):
                with self.assertRaisesRegex(ValueError,'implementation_review_required'):
                    q.qualify(request,metadata)
    def test_never_opens_or_hashes_any_prediction_or_tensor_path(self):
        original=Path.read_bytes;attempts=[]
        def guarded(path):
            if path.suffix in ('.pt','.pth','.ckpt','.jsonl'):
                attempts.append(str(path));raise AssertionError('scientific bytes opened')
            return original(path)
        with patch.object(Path,'read_bytes',guarded):self.assertTrue(self.f.execute()['qualified'])
        self.assertEqual(attempts,[])
    def test_finish_rejects_metadata_symlink_before_read(self):
        metadata=q.Metadata();ref=self.f.put('one',{});metadata.load(ref)
        path=Path(ref['local_path']);target=self.f.root/'values.pt';target.write_text('synthetic-not-data')
        path.unlink();path.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'forbidden_non_metadata'):metadata.finish()
    def test_main_ineligible_result_has_real_EXIT0_and_all_four_jobs(self):
        self.f.missing={'state_proof_ref'};request=self.f.build()
        path=self.f.root/'request.json';path.write_text(json.dumps(request))
        out=self.f.root/'qualification'
        with contextlib.redirect_stdout(io.StringIO()):
            q.main(['--request',str(path),'--request-sha256',q.digest(path.read_bytes()),'--out',str(out)])
        self.assertEqual((out/'EXIT_CODE').read_text(),'0\n')
        value=json.loads((out/'summary.json').read_text());self.assert_no_reuse(value)
        self.assertEqual(value['input_sha256'],value['input_end_sha256'])
        self.assertEqual(value['source_sha256'],value['source_end_sha256'])
    def test_main_bad_request_sha_keeps_failure_and_EXIT1(self):
        path=self.f.root/'request.json';path.write_text(json.dumps(self.f.build()));out=self.f.root/'failed'
        with self.assertRaisesRegex(ValueError,'actual_SHA'):
            q.main(['--request',str(path),'--request-sha256','0'*64,'--out',str(out)])
        self.assertEqual((out/'EXIT_CODE').read_text(),'1\n');self.assertTrue((out/'failure.json').exists())


if __name__=='__main__':unittest.main()

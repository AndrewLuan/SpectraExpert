"""功能概述：合成验证sealed适配器的review先行、named allowlist和身份/自然池契约。
所有cache/权重用临时合成文件；评分入口用合成行与冻结小模型，不执行真实test或训练。
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import evaluate_pool_fixed_center_sealed as a
from test_pool_normalization_counterfactual import make_row,make_models


def data_fixture():
    rows=[]
    for variant,count in a.VARIANTS.items():
        template=make_row(variant=variant);template.update(coarse_rank=1,baseline_rank=1)
        for n in range(count):
            row=dict(template,query_order=n,record_id=str(n));rows.append(row)
    rows.sort(key=lambda r:(r['query_order'],r['variant']))
    identity=a.content_sha([{k:r[k] for k in ('query_order','variant','source','record_id','smiles','group')} for r in rows])
    masks=a.content_sha([(tuple(r[k] for k in ('query_order','variant','source','record_id','smiles')),r['mask_content_sha256']) for r in rows])
    manifest=dict(cohort='test',recall='single',candidate_count=114982,candidate_order_sha256=a.fixed.CANDIDATE_SHA,
                  pool_size=512,records=1371,base_queries=1000,appended_training_positives=0,
                  variants=a.VARIANTS.copy(),query_sha256='a'*64,identity_content_sha256=identity,
                  variant_mask_content_sha256=masks,source_inputs_sha256={})
    return dict(manifest=manifest,rows=rows)


def tree(root):
    source=root/'science.py';source.write_text('synthetic pinned source')
    cache=root/'cache/features.pt';cache.parent.mkdir();cache.write_bytes(b'synthetic cache')
    (cache.parent/'summary.json').write_text(json.dumps(dict(cohort='test',recall='single',feature_cache_sha256=a.sha(cache))));(cache.parent/'EXIT_CODE').write_text('0\n')
    models=[];artifacts=[]
    for role in ('no_aug','aug'):
        parent=root/role;parent.mkdir();cp=parent/'best.pt';cp.write_bytes(role.encode());models.append(str(cp))
        for name in ('centers.json','center_rows.jsonl','summary.json'):(parent/name).write_text('{}')
        (parent/'EXIT_CODE').write_text('0\n')
        artifacts.append(dict(centers=str(parent/'centers.json'),centers_sha256=a.sha(parent/'centers.json'),
                              row_audit=str(parent/'center_rows.jsonl'),row_audit_sha256=a.sha(parent/'center_rows.jsonl')))
    args=argparse.Namespace(source=str(root),config='fixedcenter_mixed_listwise_single',cache=str(cache),
        no_aug_checkpoint=models[0],aug_checkpoint=models[1],seed=193,frozen_selection=str(root/'frozen.json'),out=str(root/'out'))
    allowed=dict(no_aug_checkpoint=models[0],aug_checkpoint=models[1],seed=193,cohort='test',recall='single',cache_sha256=a.sha(cache))
    selected=dict(allowed,cache=str(cache),no_aug_sha256=a.sha(models[0]),aug_sha256=a.sha(models[1]),
                  center_estimate_sha256=[x['centers_sha256'] for x in artifacts],center_artifacts=artifacts)
    pinned=a.input_fingerprints([source,cache,cache.parent/'summary.json',cache.parent/'EXIT_CODE',
        *[p for cp in models for p in (cp,Path(cp).parent/'summary.json',Path(cp).parent/'EXIT_CODE',
                                      Path(cp).parent/'centers.json',Path(cp).parent/'center_rows.jsonl')]])
    review=dict(status='PASS',independent=True,reviewer='independent synthetic reviewer',
                scope='freeze_plan_and_implementation',reviewed_sha256={str(source):a.sha(source)})
    reviewpath=root/'review.json';reviewpath.write_text(json.dumps(review))
    data=data_fixture()
    frozen=dict(schema=a.SCHEMA,selection_scope='train_val_only',frozen_utc='2026-10-23T13:00:00Z',
                verify_eligibility='unavailable',promotion=False,freeze_review=dict(path=str(reviewpath),sha256=a.sha(reviewpath)),
                reviewed_inputs_sha256=review['reviewed_sha256'],evaluation_inputs_sha256=pinned,
                ranker_configurations={args.config:{'193':allowed}},allowed_ranker_configs=[allowed],
                fixed_center_configurations={args.config:{'193':selected}},
                candidate_order_sha256=a.fixed.CANDIDATE_SHA,
                fixed_test_identity={k:data['manifest'][k] for k in a.IDENTITY})
    evidence={}
    for key in a.EVIDENCE_KEYS:
        path=root/(key+'.json' if key not in a.TOOL_KEYS else key+'.py')
        value={}
        if key=='main_validation_summary':value=dict(schema='pool_validation_selection_20261023')
        elif key=='training_audit':value=dict(status='PASS')
        elif key=='additional_training_audit':value=dict(status='PASS',scope='fixedcenter_training_no_test')
        path.write_text(json.dumps(value));evidence[key]=str(path)
    expected={models[0]:a.sha(models[0]),models[1]:a.sha(models[1])}
    joint=dict(schema='pool_joint_validation_20261023',metadata={args.config:{'193':{
        c:dict(checkpoints=expected) for c in ('original_val','tune')}}})
    jointpath=root/'joint_summary.json';jointpath.write_text(json.dumps(joint))
    tools={evidence[k]:a.sha(evidence[k]) for k in a.TOOL_KEYS}
    plan=dict(schema='pool_extended_sealed_test_plan_20261023',selection_scope='train_val_only',
        purpose='sealed_test_ablation_only',promotion=False,preselected=args.config,
        fixed_test_identity=frozen['fixed_test_identity'],evaluations=[dict(config=args.config,seed=193,
        cache=str(cache),cache_sha256=a.sha(cache))],tool_sha256=tools,source_sha256={str(source):a.sha(source)})
    planpath=root/'test_plan.json';planpath.write_text(json.dumps(plan))
    frozen.update(preselected=args.config,sealed_evidence_paths=evidence,test_plan_path=str(planpath),
                  validation_summary_path=str(jointpath))
    pinned.update(a.input_fingerprints([planpath,jointpath,*evidence.values()]))
    frozen['reviewed_inputs_sha256']=dict(pinned)
    review['reviewed_sha256']=dict(pinned);reviewpath.write_text(json.dumps(review))
    frozen['freeze_review']['sha256']=a.sha(reviewpath)
    (root/'frozen.json').write_text(json.dumps(frozen))
    return args,frozen,selected,data,source


class PreflightTests(unittest.TestCase):
    def run_tree(self,mutation=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,frozen,selected,data,source=tree(root)
            if mutation:mutation(args,frozen,selected,data,root)
            (root/'frozen.json').write_text(json.dumps(frozen))
            with patch.object(a,'source_paths',return_value=[source]):
                return a.preflight(args)
    def test_exact_named_configuration_pass(self):
        f,s,p=self.run_tree();self.assertEqual(s['seed'],193);self.assertTrue(p)
    def test_review_fail_before_cache_read(self):
        def mutate(args,f,s,d,r):
            review=r/'review.json';v=a.read_json(review);v['status']='PENDING';review.write_text(json.dumps(v));f['freeze_review']['sha256']=a.sha(review)
            Path(args.cache).unlink()
        with self.assertRaisesRegex(ValueError,'sealed_review_PASS'):self.run_tree(mutate)
    def test_review_SHA_changed(self):
        def mutate(args,f,s,d,r):(r/'review.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'sealed_review_SHA'):self.run_tree(mutate)
    def test_wrong_named_seed(self):
        def mutate(args,f,s,d,r):args.seed=211
        with self.assertRaises(KeyError):self.run_tree(mutate)
    def test_wrong_config(self):
        def mutate(args,f,s,d,r):args.config='frozen_single'
        with self.assertRaisesRegex(ValueError,'sealed_named_config'):self.run_tree(mutate)
    def test_cache_path_swapped(self):
        def mutate(args,f,s,d,r):s['cache']=str(r/'other/features.pt')
        with self.assertRaisesRegex(ValueError,'sealed_fixed_allowlist'):self.run_tree(mutate)
    def test_checkpoint_bytes_changed(self):
        def mutate(args,f,s,d,r):Path(args.no_aug_checkpoint).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'sealed_input_not_frozen'):self.run_tree(mutate)
    def test_center_artifacts_swapped(self):
        def mutate(args,f,s,d,r):s['center_artifacts'].reverse()
        with self.assertRaisesRegex(ValueError,'sealed_center_path_roles'):self.run_tree(mutate)
    def test_center_digest_mismatch(self):
        def mutate(args,f,s,d,r):s['center_estimate_sha256'][0]='0'*64
        with self.assertRaisesRegex(ValueError,'sealed_center_SHA_roles'):self.run_tree(mutate)
    def test_unpinned_adapter_source(self):
        def mutate(args,f,s,d,r):del f['evaluation_inputs_sha256'][str(r/'science.py')]
        with self.assertRaisesRegex(ValueError,'sealed_required_review_binding'):self.run_tree(mutate)
    def test_verify_producer_rejected_before_loading(self):
        def mutate(args,f,s,d,r):
            p=Path(args.cache).parent/'summary.json';v=a.read_json(p);v['cohort']='verify';p.write_text(json.dumps(v))
            f['evaluation_inputs_sha256'][str(p)]=a.sha(p)
            f['reviewed_inputs_sha256'][str(p)]=a.sha(p)
            review=r/'review.json';v=a.read_json(review);v['reviewed_sha256'][str(p)]=a.sha(p);review.write_text(json.dumps(v))
            f['freeze_review']['sha256']=a.sha(review)
        with self.assertRaisesRegex(ValueError,'sealed_cache_producer_metadata'):self.run_tree(mutate)
    def test_unrelated_review_rejected_before_cache_read(self):
        def mutate(args,f,s,d,r):
            source=r/'science.py';review=r/'review.json';record=a.read_json(review)
            record['reviewed_sha256']={str(source):a.sha(source)};review.write_text(json.dumps(record))
            f['freeze_review']['sha256']=a.sha(review);f['reviewed_inputs_sha256']=record['reviewed_sha256']
            Path(args.cache).unlink()
        with self.assertRaisesRegex(ValueError,'sealed_required_review_binding'):self.run_tree(mutate)
    def test_subset_review_cannot_drop_test_plan(self):
        def mutate(args,f,s,d,r):
            del f['reviewed_inputs_sha256'][f['test_plan_path']];Path(args.cache).unlink()
        with self.assertRaisesRegex(ValueError,'sealed_required_review_binding'):self.run_tree(mutate)
    def test_fixed_evidence_keys_required(self):
        def mutate(args,f,s,d,r):del f['sealed_evidence_paths']['training_audit'];Path(args.cache).unlink()
        with self.assertRaisesRegex(ValueError,'sealed_mandatory_evidence_paths'):self.run_tree(mutate)
    def test_verify_never_opened(self):
        def mutate(args,f,s,d,r):f['ranker_configurations'][args.config]['193']['cohort']='verify'
        with self.assertRaisesRegex(ValueError,'sealed_named_allowlist'):self.run_tree(mutate)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.data=data_fixture();self.frozen=dict(candidate_order_sha256=a.fixed.CANDIDATE_SHA,
                   fixed_test_identity={k:self.data['manifest'][k] for k in a.IDENTITY})
        self.selected=dict(recall='single')
    def check(self):return a.validate_test(self.data,self.frozen,self.selected)
    def test_actual_1371_three_identity_and_natural_pool(self):self.assertEqual(len(self.check()),1371)
    def test_positive_injection_rejected(self):
        self.data['rows'][0]=copy.deepcopy(self.data['rows'][0])
        self.data['rows'][0]['candidate_indices']=torch.arange(513)
        with self.assertRaises(ValueError):self.check()
    def test_row_identity_changed(self):
        self.data['rows'][0]['record_id']='different'
        with self.assertRaisesRegex(ValueError,'sealed_row_identity'):self.check()
    def test_variant_mask_changed(self):
        self.data['rows'][0]['mask_content_sha256']='different'
        with self.assertRaisesRegex(ValueError,'sealed_row_identity'):self.check()
    def test_fixed_query_SHA_changed(self):
        self.data['manifest']['query_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'sealed_fixed_identity'):self.check()
    def test_short_query_count(self):
        self.data['rows'].pop()
        with self.assertRaisesRegex(ValueError,'sealed_natural1371'):self.check()


class EntryTests(unittest.TestCase):
    def test_actual_score_loop_and_roundtrip_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();args,frozen,selected,data,source=tree(root);out=Path(args.out);out.mkdir()
            models=make_models()
            for role,(model,scale,saved) in enumerate(models):
                saved['args'].update(a.fixed.BUDGET,training_pool='mixed',loss='listwise')
                saved.update(center_estimate_sha256=selected['center_estimate_sha256'][role],train_sha256='train',dual_train_sha256='dual')
            one=[data['rows'][0]]
            with patch.object(a,'source_paths',return_value=[source]),                 patch.object(a.fixed,'read_feature_cache',return_value=(data,selected['cache_sha256'])),                 patch.object(a,'validate_test',return_value=one),                 patch.object(a.fixed,'load_checkpoint',side_effect=models),                 patch.object(a.fixed.base,'check_science'):
                summary=a.evaluate(args,out,torch.device('cpu'),lambda *a,**k:None)
            self.assertEqual(summary['cohort'],'test');self.assertEqual(summary['config'],args.config)
            row=json.loads((out/'predictions.jsonl').read_text())
            self.assertEqual(len(row['natural_candidate_costs']),512)
            self.assertEqual(len(row['natural_candidate_indices']),512)
            self.assertTrue((out/'protocol.json').is_file())


if __name__=='__main__':unittest.main()

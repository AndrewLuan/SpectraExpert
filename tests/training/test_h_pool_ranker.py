"""排序研究的合成契约测试：检查负例隔离梯度、两种排序损失、相同曝光与平衡池、
缓存身份/观测掩码、自然候选与补正例边界，以及继承checkpoint冻结观测模型。
只用内存小样本和临时checkpoint，不读取真实科学评测数据，不执行正式训练。
"""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from nmr_transfer import h_pool_ranker as p
from nmr_transfer.h_research_ranker import ResearchRanker, ranking_loss


def row():
    n = 512
    return dict(query_order=0, variant='observed', source='synthetic', record_id='x',
        smiles='C', group='g', kind=0, observation_input={'x':torch.tensor([1.])}, query_observation={'windows':None}, candidate_indices=torch.arange(n), evaluation_pool_size=n,
        features=torch.zeros(n,33), coarse_scores=torch.arange(n).float()/100,
        predicted_ppm=torch.zeros(n,128), tags=torch.zeros(n,128), H=torch.ones(n),
        K=torch.ones(n), target_position=0, target_in_coarse_pool=True,
        negative_exclusion_mask=torch.zeros(n,dtype=torch.bool), mask_content_sha256='mask')


def cache(recall):
    return dict(manifest=dict(schema='nmr_research_20261022',cohort='train',
        candidate_order_sha256='c', checkpoint_sha256='w',pool_size=512,
        candidate_count=114982,recall=recall,query_sha256='q',identity_content_sha256='i',
        variant_mask_content_sha256='m',negative_excluded_structure_sha256='x',
        training_subset='train',source_inputs_sha256={'a':'b'}),rows=[row()])


class PoolRankerTests(unittest.TestCase):
    def test_pairwise_matches_old(self):
        r=row(); d=torch.randn(512,requires_grad=True); scale=torch.ones(33)
        self.assertTrue(torch.equal(p.pool_loss(r,d,scale,.1,'pairwise16'),
                                    ranking_loss(r,d,scale,.1)))

    def test_listwise_matches_cross_entropy(self):
        r=row(); d=torch.randn(512,requires_grad=True)
        costs=r['coarse_scores']+d
        expected=torch.nn.functional.cross_entropy((-costs/.5)[None],torch.tensor([0]))
        self.assertTrue(torch.allclose(p.pool_loss(r,d,torch.ones(33),0,'listwise'),expected))

    def test_excluded_negatives_have_no_direct_loss_gradient(self):
        for mode in ('pairwise16','listwise'):
            r=row(); r['negative_exclusion_mask'][1:20]=True
            d=torch.zeros(512,requires_grad=True)
            p.pool_loss(r,d,torch.ones(33),.1,mode).backward()
            self.assertTrue(torch.equal(d.grad[1:20],torch.zeros(19)))
            self.assertNotEqual(float(d.grad[0]),0.)

    def test_no_eligible_negatives_fails(self):
        r=row(); r['negative_exclusion_mask'][:]=True
        with self.assertRaisesRegex(ValueError,'no_eligible'):
            p.pool_loss(r,torch.zeros(512),torch.ones(33),.1,'listwise')

    def test_balanced_deterministic_schedule(self):
        groups={'a':[dict(row(),group='a')],'b':[dict(row(),group='b')]}
        a=p.training_schedule(['a','b'],groups,211,1,100)
        b=p.training_schedule(['a','b'],groups,211,1,100)
        self.assertEqual([(p.row_key(r),i) for r,i in a],[(p.row_key(r),i) for r,i in b])
        self.assertEqual(sum(i for _,i in a),50)
        self.assertNotEqual([p.row_key(r) for r,_ in a],
            [p.row_key(r) for r,_ in p.training_schedule(['a','b'],groups,193,1,100)])

    def test_valid_caches(self):
        self.assertEqual(len(p.validate_caches(cache('single'),cache('dual'))),1)

    def test_mask_mismatch_fails(self):
        a,b=cache('single'),cache('dual'); b['rows'][0]['mask_content_sha256']='other'
        with self.assertRaisesRegex(ValueError,'observation_mask'):
            p.validate_caches(a,b)

    def test_actual_observation_change_fails_despite_same_hash(self):
        a,b=cache('single'),cache('dual')
        b['rows'][0]['observation_input']['x'][0]=2
        with self.assertRaisesRegex(ValueError,'observation_content'):
            p.validate_caches(a,b)

    def test_training_presence_false_without_append_fails(self):
        r=row();r['target_in_coarse_pool']=False
        with self.assertRaisesRegex(ValueError,'training_presence'):
            p.validate_row(r,True)

    def test_exclusion_provenance_mismatch_fails(self):
        a,b=cache('single'),cache('dual')
        b['manifest']['negative_excluded_structure_sha256']='other'
        with self.assertRaisesRegex(ValueError,'cache_identity'):
            p.validate_caches(a,b)

    def test_eval_injected_positive_fails(self):
        r=row(); r['candidate_indices']=torch.arange(513);r['target_position']=512
        with self.assertRaisesRegex(ValueError,'positive_injection'):
            p.validate_row(r,False)

    def test_duplicate_natural_pool_fails(self):
        r=row();r['candidate_indices'][2]=1
        with self.assertRaisesRegex(ValueError,'natural_candidates'):
            p.validate_row(r,False)

    def test_load_preserves_values_and_freezes_observation(self):
        model=ResearchRanker(True,[.1]*9)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            saved=dict(schema=p.OLD_SCHEMA,observation=True,
                       initial_sigmas=model.observation_model.initial_sigmas.tolist(),
                       state_dict=model.state_dict(),scale=torch.arange(1,34).float())
            torch.save(saved,path)
            got,scale,_=p.load_checkpoint(path,torch.device('cpu'))
            for k,v in model.state_dict().items():
                self.assertTrue(torch.equal(v,got.state_dict()[k]))
            self.assertTrue(torch.equal(scale,saved['scale']))
            self.assertFalse(any(x.requires_grad for x in got.observation_model.parameters()))
            self.assertTrue(all(x.requires_grad for x in got.network.parameters()))

    def test_inherited_cost_exact_and_optimizer_freezes_observation(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        model=ResearchRanker(True,[.1]*9)
        with torch.no_grad():
            model.network[-1].weight.normal_(0,.1)
        r=row();r['features']=torch.randn(512,33)
        r['observation_input']={'semantics':'line_positions','xs':[1.,2.],'weights':[1.,1.]}
        scale=torch.ones(33)
        before={k:v.clone() for k,v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save(dict(schema=p.OLD_SCHEMA,observation=True,initial_sigmas=[.1]*9,
                            state_dict=model.state_dict(),scale=scale,alpha=.5),path)
            loaded,got_scale,saved=p.load_checkpoint(path,torch.device('cpu'))
            old_cost=r['coarse_scores']+.5*model.row_delta(r,scale,'cpu')*scale[0]
            new_cost=r['coarse_scores']+saved['alpha']*loaded.row_delta(r,got_scale,'cpu')*got_scale[0]
            self.assertTrue(torch.equal(old_cost,new_cost))
            opt=torch.optim.AdamW(loaded.network.parameters(),lr=.001)
            p.pool_loss(r,loaded.row_delta(r,got_scale,'cpu'),got_scale,.1,'listwise').backward()
            opt.step()
            for k,v in loaded.state_dict().items():
                if k.startswith('observation_model.'):
                    self.assertTrue(torch.equal(v,before[k]))
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in loaded.state_dict().items()
                                if k.startswith('network.')))

    def test_shared_negative_mask_disagreement_fails(self):
        a,b=cache('single'),cache('dual'); b['rows'][0]['negative_exclusion_mask'][3]=True
        with self.assertRaisesRegex(ValueError,'shared_negative_exclusion'):
            p.validate_caches(a,b)

    def test_test_config_cache_sha_must_be_frozen(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'freeze.json'
            frozen=dict(schema=p.SCHEMA,frozen_utc='now',selection_scope='train_val_only',
                        allowed_ranker_configs=[])
            path.write_text(json.dumps(frozen))
            args=SimpleNamespace(frozen_selection=str(path),no_aug_checkpoint='/a',
                                 aug_checkpoint='/b',seed=211)
            with self.assertRaisesRegex(ValueError,'config_not_frozen'):
                p.check_frozen(args,dict(manifest={'cohort':'test','recall':'single'}),'wrong')

    def test_verify_unavailable_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'freeze.json'
            config=dict(no_aug_checkpoint='/a',aug_checkpoint='/b',seed=211,
                        cohort='verify',recall='single',cache_sha256='cache')
            path.write_text(json.dumps(dict(schema=p.SCHEMA,frozen_utc='now',
                selection_scope='train_val_only',allowed_ranker_configs=[config],
                verify_eligibility='unavailable')))
            args=SimpleNamespace(frozen_selection=str(path),no_aug_checkpoint='/a',
                                 aug_checkpoint='/b',seed=211)
            with self.assertRaisesRegex(ValueError,'acceptance_unavailable'):
                p.check_frozen(args,dict(manifest={'cohort':'verify','recall':'single'}),'cache')

    def test_unfrozen_test_rejected(self):
        with self.assertRaisesRegex(ValueError,'frozen_selection_required'):
            p.check_frozen(SimpleNamespace(frozen_selection=None),
                           dict(manifest={'cohort':'test'}),'cache')

    def test_validation_needs_no_test_freeze(self):
        p.check_frozen(SimpleNamespace(frozen_selection=None),
                      dict(manifest={'cohort':'tune'}),'cache')


if __name__=='__main__':
    unittest.main()

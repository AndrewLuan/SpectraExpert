"""功能概述：用合成自然候选和真实ResearchRanker核验中心反事实诊断。
SelectionTests覆盖固定结构/base身份hash、排除internal、缺变体和跨seed观测失败关闭；
ScoringTests覆盖追加正例先剔除、无标签/身份评分输入、固定scale/权重/raw、单独中心替换、
真实row_delta一致性、非线性翻序、常量平移不翻序及epsilon tie；ReportingTests覆盖自然
池外RR、common/new/lost/neither、共同target竞争计数与RR分解；AggregationArithmeticTests
用不同共同候选数核对逐维加权均值、global max与固定尺度中心差。无真实数据或训练。
"""
import copy
import hashlib
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

import pool_normalization_counterfactual as c

for source_root in (Path('/Users/andrewluan/Desktop/Projects/IRfingerprints/SpectrExpert'),
                    Path('/home/user/IRfingerprints/SpectrExpert')):
    if (source_root/'nmr_transfer/h_research_ranker.py').is_file():
        sys.path.insert(0,str(source_root));break
else:
    raise FileNotFoundError('scientific_checkout_source_missing')
from nmr_transfer.h_research_ranker import ResearchRanker
from nmr_transfer.h_pool_ranker import validate_caches

torch.set_num_threads(1)


def observation():
    return dict(semantics='quantile_support',xs=[1.,2.],weights=[.5,.5],
                windows=[],partial_peaks=False,reported_count=None)


def make_row(pool='S',target=10,variant='observed'):
    ids=torch.arange(512) if pool=='S' else torch.cat((torch.arange(256),torch.arange(512,768)))
    positions=(ids==target).nonzero().flatten()
    present=bool(len(positions))
    if not present:ids=torch.cat((ids,torch.tensor([target])))
    t=ids.float();features=torch.stack([torch.sin(t/(2+j))+j*.05 for j in range(33)],1)
    coarse=t*.001
    features[:,0]=coarse
    row=dict(query_order=0,source='synthetic',record_id='id0',smiles='CC',group='train_group',
        variant=variant,mask_content_sha256='mask',candidate_indices=ids,evaluation_pool_size=512,
        features=features,coarse_scores=coarse,predicted_ppm=t[:,None]%11*.1+torch.tensor([[.1,1.,2.,3.]]),
        tags=torch.zeros((len(ids),4),dtype=torch.long),H=torch.full((len(ids),),10.),
        K=torch.full((len(ids),),10.),observation_input=observation(),kind=0,
        query_observation=dict(windows=[],partial_peaks=False),target_position=int(positions[0]) if present else 512,
        target_in_coarse_pool=present,outside_pool_rank=None if present else 1234)
    if variant=='frequency_crop':
        row['query_observation']['windows']=[[.5,2.5]]
        row['observation_input']['windows']=[[.5,2.5]]
    if variant=='peak_dropout':
        row['query_observation']['partial_peaks']=True;row['observation_input']['partial_peaks']=True
    return row


def make_models():
    result=[]
    for role in (False,True):
        torch.manual_seed(17+int(role))
        model=ResearchRanker(True,[.2]*9)
        with torch.no_grad():model.network[-1].weight.normal_(0,.15)
        for p in model.parameters():p.requires_grad_(False)
        model.eval()
        scale=torch.linspace(.5,2.,33)
        saved=dict(alpha=.5 if role else .25,args=dict(seed=193,augment=role),
                   state_dict={k:v.detach().clone() for k,v in model.state_dict().items()})
        result.append((model,scale,saved))
    return result


def selection_indexes(groups=5):
    names=[];i=0
    while len(names)<groups:
        g='g'+str(i);i+=1
        if not c.internal_group(g):names.append(g)
    rows=[]
    for n,g in enumerate(names):
        for record in (0,1):
            for variant in c.VARIANTS:
                rows.append(dict(query_order=n*2+record,source='synthetic',record_id=str(record),
                    smiles='C'+str(n),group=g,variant=variant,mask_content_sha256='mask'+variant,
                    kind=0,observation_input=observation(),query_observation={}))
    return {(seed,pool):{c.identity(r):copy.deepcopy(r) for r in rows}
            for seed in c.SEEDS for pool in ('single','dual')}


class SelectionTests(unittest.TestCase):
    def test_fixed_group_and_base_hash(self):
        indexes=selection_indexes()
        a=c.selection_manifest(indexes,count=3)
        reversed_indexes={k:dict(reversed(list(v.items()))) for k,v in reversed(list(indexes.items()))}
        b=c.selection_manifest(reversed_indexes,count=3)
        self.assertEqual(a,b);self.assertEqual(len(a['groups']),3);self.assertEqual(len(a['entries']),9)
        names={r['group'] for r in indexes[(193,'single')].values()}
        expected=sorted(names,key=lambda g:(hashlib.sha256(('pool_norm_diagnostic_20261023'+g).encode()).hexdigest(),g))[:3]
        self.assertEqual(a['groups'],expected)
        for group in a['groups']:
            selected={tuple(r['base_query']) for r in a['entries'] if r['group']==group}
            self.assertEqual(len(selected),1)

    def test_internal_groups_never_selected(self):
        indexes=selection_indexes();self.assertTrue(all(not c.internal_group(g) for g in c.selection_manifest(indexes,3)['groups']))

    def test_all_missing_variant_is_explicit_skip(self):
        indexes=selection_indexes(1)
        for ix in indexes.values():
            for k in list(ix):
                if k[-1]=='peak_dropout':del ix[k]
        manifest=c.selection_manifest(indexes,1)
        self.assertEqual(len(manifest['entries']),2)
        self.assertEqual(manifest['skips'][0]['variant'],'peak_dropout')

    def test_partial_missing_variant_rejected(self):
        indexes=selection_indexes(1)
        ix=indexes[(211,'dual')]
        for k in list(ix):
            if k[-1]=='peak_dropout':del ix[k]
        with self.assertRaisesRegex(ValueError,'cross_seed_pool_missing_variant'):c.selection_manifest(indexes,1)

    def test_observation_mask_mismatch_rejected(self):
        indexes=selection_indexes(1)
        for r in indexes[(211,'dual')].values():r['mask_content_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'query_observation_mismatch'):c.selection_manifest(indexes,1)

    def test_actual_observation_content_checked(self):
        indexes=selection_indexes(1)
        for r in indexes[(211,'dual')].values():r['observation_input']['xs']=[5.]
        with self.assertRaisesRegex(ValueError,'query_observation_mismatch'):c.selection_manifest(indexes,1)

    def test_too_few_groups_rejected(self):
        with self.assertRaisesRegex(ValueError,'insufficient_train_groups'):c.selection_manifest(selection_indexes(1),2)

    def test_duplicate_query_identity_rejected(self):
        row=make_row()
        with self.assertRaisesRegex(ValueError,'duplicate_query_identity'):c.index_rows(dict(rows=[row,row]))


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.models=make_models();self.s=make_row('S');self.d=make_row('D')

    def probes(self,s=None,d=None):
        return c.run_probes(self.models,c.natural_row(s or self.s),c.natural_row(d or self.d),torch.device('cpu'))

    def test_appended_positive_never_scored_or_centered(self):
        row=make_row('S',target=10000)
        natural=c.natural_row(row)
        row['features'][-1]=1e8;row['predicted_ppm'][-1]=1e8;row['coarse_scores'][-1]=1e8
        other=copy.deepcopy(row);other['features'][-1]=-1e8;other['predicted_ppm'][-1]=-1e8
        a,_,_=self.probes(s=row);b,_,_=self.probes(s=other)
        self.assertEqual(natural['features'].shape,(512,33))
        self.assertTrue(torch.equal(a['S_native']['cost'],b['S_native']['cost']))

    def test_no_identity_or_label_in_scoring_input(self):
        self.s['observation_input']['source']='bad';self.s['query_observation']['record_id']='bad'
        row=c.natural_row(self.s)
        for key in ('source','record_id','smiles','group','candidate_indices','outside_pool_rank'):
            self.assertNotIn(key,row)
        self.assertEqual(row['target_position'],-1);self.assertIs(row['target_in_coarse_pool'],False)
        self.assertNotIn('source',row['observation_input']);self.assertNotIn('record_id',row['query_observation'])

    def test_actual_native_path_and_state_raw_fixed(self):
        state=c.model_snapshot(self.models);rows=c.value_sha((self.s,self.d))
        outputs,_,checks=self.probes()
        self.assertEqual(tuple(outputs),c.PROBES)
        self.assertTrue(checks['raw_unchanged']);self.assertTrue(checks['model_scale_state_unchanged'])
        self.assertTrue(all(x<=1e-6 for x in checks['native_delta_max_abs_errors']))
        self.assertEqual(c.model_snapshot(self.models),state)
        self.assertEqual(c.value_sha((self.s,self.d)),rows)

    def test_only_X_center_swapped(self):
        outputs,raw,_=self.probes()
        for role in range(2):
            native=outputs['D_native']['traces'][role];changed=outputs['D_use_S_Xcenter']['traces'][role]
            self.assertTrue(torch.equal(native['zo'],changed['zo']))
            self.assertTrue(torch.equal(changed['xcenter'],raw['S'][role]['xcenter']))
            self.assertTrue(torch.equal(changed['ocenter'],raw['D'][role]['ocenter']))

    def test_only_O_center_swapped(self):
        outputs,raw,_=self.probes()
        for role in range(2):
            native=outputs['D_native']['traces'][role];changed=outputs['D_use_S_Ocenter']['traces'][role]
            self.assertTrue(torch.equal(native['zx'],changed['zx']))
            self.assertTrue(torch.equal(changed['xcenter'],raw['D'][role]['xcenter']))
            self.assertTrue(torch.equal(changed['ocenter'],raw['S'][role]['ocenter']))

    def test_scale_fixed_for_every_probe(self):
        outputs,raw,_=self.probes()
        for name,out in outputs.items():
            for role in range(2):
                trace=out['traces'][role];rr=raw[out['pool']][role]
                self.assertTrue(torch.equal(trace['zx_raw'],(rr['x']-trace['xcenter'])/self.models[role][1]))

    def test_coarse_feature_is_also_clamped(self):
        row=c.natural_row(self.s);row['features']=row['features'].clone();row['features'][0,0]=1000.
        raw=c.raw_trace(self.models[0][0],row,self.models[0][1],torch.device('cpu'))
        trace=c.instrument(self.models[0][0],raw,0)
        self.assertEqual(float(trace['zx'][0,0]),20.)

    def test_nonlinearity_can_reverse_order_under_center_shift(self):
        class Square(nn.Module):
            def forward(self,x):
                v=x[:,0].square()
                return torch.stack((v,v,v,v),1)
        class Fake:
            network=Square()
        raw=dict(x=torch.zeros((2,33)),o=torch.zeros((2,3)),xcenter=torch.zeros(33),
                 ocenter=torch.zeros(3),scale=torch.ones(33))
        raw['x'][:,0]=torch.tensor([-1.,1.])
        left=torch.zeros(33);left[0]=-.5;right=torch.zeros(33);right[0]=.5
        a=c.instrument(Fake(),raw,0,xcenter=left);b=c.instrument(Fake(),raw,0,xcenter=right)
        flips=c.pair_changes(a['delta'].tolist(),b['delta'].tolist())
        self.assertEqual(flips['strict_reversals'],1)

    def test_pure_constant_score_shift_has_no_flips(self):
        result=c.pair_changes([1.,3.,5.],[10.,12.,14.])
        self.assertEqual(result['strict_reversals'],0);self.assertEqual(result['tie_transitions'],0)

    def test_tie_transitions_distinct_from_strict_reversals(self):
        result=c.pair_changes([0.,0.,2.],[0.,1.,0.])
        self.assertEqual(result['tie_to_strict'],1)
        self.assertEqual(result['strict_to_tie'],1)
        self.assertEqual(result['strict_reversals'],1)

    def test_epsilon_fixed(self):
        result=c.pair_changes([0.,.5e-7],[0.,2e-7])
        self.assertEqual(result['tie_to_strict'],1);self.assertEqual(result['epsilon'],1e-7)

    def test_invalid_pool_or_scale_rejected(self):
        self.s['evaluation_pool_size']=511
        with self.assertRaisesRegex(ValueError,'natural_pool_must'):c.natural_row(self.s)
        self.models[0][1][0]=0
        with self.assertRaisesRegex(ValueError,'invalid_fixed_scale'):self.probes(s=make_row())

    def test_nonfinite_raw_features_rejected(self):
        self.s['features'][0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite_raw_X'):c.natural_row(self.s)

    def test_gamma_uses_explicit_missing_flags(self):
        _,_,normal=self.probes()
        _,_,missing=self.probes(s=make_row(variant='peak_dropout'),d=make_row('D',variant='peak_dropout'))
        self.assertEqual(normal['gamma'],.25);self.assertEqual(missing['gamma'],.75)


class ReportingTests(unittest.TestCase):
    def report(self,target):
        models=make_models();s=make_row('S',target=target);d=make_row('D',target=target)
        out,raw,checks=c.run_probes(models,c.natural_row(s),c.natural_row(d),torch.device('cpu'))
        return c.query_report(193,c.identity(s),s,d,out,raw,checks)

    def test_common_target_and_RR_decomposition(self):
        r=self.report(10)
        self.assertEqual(r['target_category'],'common')
        parts=r['RR_decomposition']
        self.assertAlmostEqual(parts['total'],parts['normalization']+parts['composition'],places=12)
        self.assertIsNotNone(r['common_only_RR_decomposition'])
        self.assertIn('D_native',r['common_target_competitors'])
        self.assertEqual(r['raw_common_differences']['no_aug']['X33']['max_abs'],[0.]*33)

    def test_new_target_natural_outside_single_uses_cache_rank(self):
        r=self.report(600)
        self.assertEqual(r['target_category'],'new')
        self.assertEqual(r['ranks']['S_native']['rank'],1234)
        self.assertEqual(r['ranks']['S_use_both_Dcenters']['rank'],1234)
        self.assertIsNone(r['ranks']['D_native']['common_only_RR'])

    def test_lost_target_natural_outside_dual_fixed_for_all_dual_probes(self):
        r=self.report(400);self.assertEqual(r['target_category'],'lost')
        for name in c.PROBES:
            if name.startswith('D_'):self.assertEqual(r['ranks'][name]['rank'],1234)
        self.assertEqual(r['RR_decomposition']['normalization'],0.)

    def test_neither_target_and_background_contract(self):
        r=self.report(10000);self.assertEqual(r['target_category'],'neither')
        self.assertEqual(r['RR_decomposition'],dict(total=0.,normalization=0.,composition=0.))
        with self.assertRaisesRegex(ValueError,'invalid_outside_pool_rank'):
            c.rank_in_pool(torch.arange(512),torch.zeros(512),10000,512)

    def test_raw_common_difference_not_assumed_zero(self):
        s=make_row();d=make_row('D');d['features'][0,2]+=.001
        models=make_models();out,raw,checks=c.run_probes(models,c.natural_row(s),c.natural_row(d),torch.device('cpu'))
        r=c.query_report(193,c.identity(s),s,d,out,raw,checks)
        self.assertFalse(r['raw_common_differences']['no_aug']['X33']['exact_equal'])
        self.assertGreater(r['raw_common_differences']['no_aug']['X33']['max_abs'][2],0)

    def test_target_identity_mismatch_rejected_after_scoring(self):
        s=make_row(target=10);d=make_row('D',target=11)
        out,raw,checks=c.run_probes(make_models(),c.natural_row(s),c.natural_row(d),torch.device('cpu'))
        with self.assertRaisesRegex(ValueError,'target_identity_mismatch'):
            c.query_report(193,c.identity(s),s,d,out,raw,checks)

    def test_pair_weighted_and_query_mean_differ(self):
        a=c.pair_changes([0.,1.],[1.,0.]);b=c.pair_changes([0.,1.,2.],[0.,1.,2.])
        result=c.aggregate_pairs([a,b])
        self.assertEqual(result['query_mean_strict_reversal_rate'],.5)
        self.assertEqual(result['pair_weighted_strict_reversal_rate'],.25)

    def test_aggregate_has_category_and_seed_variant(self):
        r=self.report(10);a=c.aggregate([r])
        self.assertEqual(a['by_target_category']['common']['queries'],1)
        self.assertEqual(a['by_seed_variant']['193']['observed']['queries'],1)
        self.assertEqual(a['by_seed']['211']['queries'],0)



class AggregationArithmeticTests(unittest.TestCase):
    def test_candidate_weighted_feature_mean_and_global_max(self):
        result=c.aggregate_differences([
            dict(count=1,mean_abs=[2.,4.],mean_signed=[2.,-4.],max_abs=[3.,5.]),
            dict(count=3,mean_abs=[6.,8.],mean_signed=[-6.,8.],max_abs=[10.,12.]),
            dict(count=0,mean_abs=None,mean_signed=None,max_abs=None)])
        self.assertEqual(result['common_candidate_observations'],4)
        self.assertEqual(result['queries_with_common_candidates'],2)
        self.assertEqual(result['candidate_weighted_mean_abs'],[5.,7.])
        self.assertEqual(result['candidate_weighted_mean_signed'],[-4.,5.])
        self.assertEqual(result['max_abs'],[10.,12.])

    def test_center_weighted_and_query_means_and_fixed_scale(self):
        rows=[
            dict(common_candidates=1,centers=dict(no_aug=dict(D_minus_S_Xcenter=[2.,-4.],fixed_scale=[2.,4.]))),
            dict(common_candidates=3,centers=dict(no_aug=dict(D_minus_S_Xcenter=[10.,8.],fixed_scale=[2.,4.]))),
            dict(common_candidates=0,centers={})]
        raw=c.aggregate_center_differences(rows,'no_aug','D_minus_S_Xcenter')
        scaled=c.aggregate_center_differences(rows,'no_aug','D_minus_S_Xcenter',True)
        self.assertEqual(raw['common_candidate_weighted_mean_abs'],[8.,7.])
        self.assertEqual(raw['query_mean_abs'],[6.,6.])
        self.assertEqual(raw['max_abs'],[10.,8.])
        self.assertEqual(scaled['common_candidate_weighted_mean_abs'],[4.,1.75])
        self.assertEqual(scaled['query_mean_abs'],[3.,1.5])
        self.assertEqual(scaled['max_abs'],[5.,2.])
        self.assertIsNone(c.aggregate_differences([dict(count=0)])['max_abs'])


if __name__=='__main__':
    unittest.main()

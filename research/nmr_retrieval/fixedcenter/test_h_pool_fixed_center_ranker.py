"""功能概述：合成验证固定中心追加模块的统计资格、query-row权重和训练接口。
测试采用小型bank身份侧表与真实ResearchRanker；只有测试常量/identity_fn使用小候选宇宙。
核心覆盖mask/heldout/internal/追加正例、空池fail、query等权池.5、固定36常量、10初始tensor
精确继承、一步网络训练不改O/scale/center、共同候选分数不依赖池及checkpoint证据SHA。
EntryFixture保留64epochs生产args，以小合成schedule/内部验证替身贯通实际fit→load→evaluate；
不执行真实训练、读取真实谱或修改科学源。
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import h_pool_fixed_center_ranker as m
from test_pool_normalization_counterfactual import make_row,make_models

torch.set_num_threads(1)


def group_name(internal=False,start=0):
    i=start
    while m.internal_group('fixed_group_'+str(i)) is not internal:i+=1
    return 'fixed_group_'+str(i)


def case_data():
    fit_a=group_name(False);fit_b=group_name(False,50);internal=group_name(True);heldout='historical_heldout_group'
    smiles=['synthetic_'+str(i) for i in range(1024)]
    groups=[fit_b]*1024;groups[1]=heldout;groups[2]=internal;groups[10]=fit_a;groups[20]=fit_b;groups[30]=internal
    rows_s=[];rows_d=[]
    for order,(group,target,variant) in enumerate(((fit_a,10,'observed'),(fit_a,10,'frequency_crop'),
                                                  (fit_b,20,'observed'),(internal,30,'observed'))):
        for pool,rows in (('S',rows_s),('D',rows_d)):
            row=make_row(pool,target,variant)
            row.update(query_order=order,group=group,record_id='fixed'+str(order),variant=variant)
            row['negative_exclusion_mask']=torch.tensor([groups[int(i)]==group or groups[int(i)]==heldout for i in row['candidate_indices']])
            rows.append(row)
    order=m.ordered_sha(smiles);excluded=m.ordered_sha([heldout])
    manifest=dict(schema='nmr_research_20261022',cohort='train',candidate_count=114982,
        pool_size=512,candidate_order_sha256=order,checkpoint_sha256='forward',source_inputs_sha256={},
        query_sha256='queries',identity_content_sha256='identity',variant_mask_content_sha256='masks',
        negative_excluded_structure_sha256=excluded,training_subset='train')
    single=dict(rows=rows_s,smiles=smiles,manifest=dict(manifest,recall='single'))
    dual=dict(rows=rows_d,smiles=smiles,manifest=dict(manifest,recall='dual'))
    contract=dict(candidate_order_sha256=order,training_excluded_groups=[heldout])
    return single,dual,contract,groups


def qualification(single,dual,contract,groups):
    with patch.object(m,'CANDIDATE_COUNT',1024),patch.object(m,'CANDIDATE_SHA',m.ordered_sha(single['smiles'])):
        return m.candidate_qualification(single,dual,contract,lambda s:groups[int(s.rsplit('_',1)[1])])


class CenterEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.s,self.d,self.contract,self.groups=case_data()
        self.q=qualification(self.s,self.d,self.contract,self.groups)

    def test_bank_candidate_order_and_heldout_hash(self):
        self.assertEqual(self.q['metadata']['candidate_count'],1024)
        self.assertEqual(self.q['metadata']['derived_candidate_groups_sha256'],m.ordered_sha(self.groups))
        self.assertEqual(self.q['metadata']['historical_heldout_sha256'],m.ordered_sha(self.contract['training_excluded_groups']))

    def test_natural_positive_mask_heldout_and_internal_excluded(self):
        row=self.s['rows'][0];positions=m.eligible_positions(row,self.q).tolist()
        self.assertNotIn(1,positions);self.assertNotIn(2,positions);self.assertNotIn(10,positions)
        self.assertIn(3,positions)

    def test_appended_positive_excluded_without_target_branch(self):
        row=copy.deepcopy(self.s['rows'][0]);target=1000
        self.q['groups'][target]=row['group']
        row['candidate_indices']=torch.cat((row['candidate_indices'],torch.tensor([target])))
        row['negative_exclusion_mask']=torch.cat((row['negative_exclusion_mask'],torch.tensor([True])))
        row['target_position']=512;row['target_in_coarse_pool']=False
        positions=m.eligible_positions(row,self.q)
        self.assertTrue(bool((positions<512).all()))

    def test_mask_cannot_unexclude_historical_heldout(self):
        row=copy.deepcopy(self.s['rows'][0]);row['negative_exclusion_mask'][1]=False
        with self.assertRaisesRegex(ValueError,'negative_mask_semantics'):m.eligible_positions(row,self.q)

    def test_mask_cannot_unexclude_query_structure(self):
        row=copy.deepcopy(self.s['rows'][0]);row['negative_exclusion_mask'][10]=False
        with self.assertRaisesRegex(ValueError,'negative_mask_semantics'):m.eligible_positions(row,self.q)

    def test_internal_query_excluded_from_estimator(self):
        with self.assertRaisesRegex(ValueError,'query_not_fit'):m.eligible_positions(self.s['rows'][-1],self.q)

    def test_empty_eligible_fails_no_fallback(self):
        row=copy.deepcopy(self.s['rows'][0])
        self.q['groups']=[row['group']]*1024
        row['negative_exclusion_mask']=torch.ones(512,dtype=torch.bool)
        with self.assertRaisesRegex(ValueError,'empty_eligible'):m.eligible_positions(row,self.q)

    def test_wrong_candidate_order_rejected(self):
        self.d['smiles']=list(reversed(self.d['smiles']))
        with self.assertRaisesRegex(ValueError,'fixed_candidate_identity'):qualification(self.s,self.d,self.contract,self.groups)

    def test_wrong_heldout_hash_rejected(self):
        self.contract['training_excluded_groups'].append('another')
        with self.assertRaisesRegex(ValueError,'negative_exclusion_contract_SHA'):qualification(self.s,self.d,self.contract,self.groups)

    def test_only_group_identity_is_used_not_identity_random_split(self):
        # 注入函数只返回group；生产代码lambda明确仅取identity(s)[1]。
        self.assertFalse(self.q['internal'][3])
        self.assertEqual(self.q['groups'][3],self.groups[3])


class CenterEstimatorTests(unittest.TestCase):
    def test_query_row_equal_not_group_weighted_and_pool_half(self):
        s,d,contract,groups=case_data();q=qualification(s,d,contract,groups)
        rows=s['rows'][:3]
        for n,(a,b) in enumerate(zip(rows,d['rows'][:3])):
            a['features']=torch.full_like(a['features'],float(n*4))
            b['features']=torch.full_like(b['features'],float(n*4+2))
            a['predicted_ppm']=torch.full_like(a['predicted_ppm'],float(n*4))
            b['predicted_ppm']=torch.full_like(b['predicted_ppm'],float(n*4+2))
        class O(torch.nn.Module):
            def forward(self,predicted,tags,obs,**kw):return predicted[:,:3]
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.observation_model=O()
        x,o,meta=m.estimate_centers(Model(),rows,{m.base.row_key(r):r for r in d['rows']},q,torch.device('cpu'))
        self.assertTrue(torch.equal(x,torch.full((33,),5.)))
        self.assertTrue(torch.equal(o,torch.full((3,),5.)))
        self.assertEqual(meta['query_rows'],3)
        self.assertNotEqual(float(x[0]),6.)  # group-equal would be6; fixed contract is query-row equal.

    def test_estimator_state_and_role_specific_O(self):
        s,d,contract,groups=case_data();q=qualification(s,d,contract,groups)
        models=make_models();results=[]
        for model,_,_ in models:
            before={k:v.clone() for k,v in model.state_dict().items()}
            x,o,meta=m.estimate_centers(model,s['rows'][:3],{m.base.row_key(r):r for r in d['rows']},q,torch.device('cpu'))
            self.assertTrue(m.base.same_value(before,model.state_dict()))
            self.assertEqual(x.dtype,torch.float32);self.assertEqual(o.dtype,torch.float32)
            self.assertTrue(meta['model_unchanged']);results.append((x,o))
        self.assertTrue(torch.equal(results[0][0],results[1][0]))


class ModelTests(unittest.TestCase):
    def setUp(self):
        initial,scale,saved=make_models()[0];saved['initial_sigmas']=initial.observation_model.initial_sigmas.tolist()
        self.initial,self.scale,self.saved=initial,scale,saved
        self.model=m.inherit_model(initial,saved,torch.ones(33)*.1,torch.ones(3)*.2,torch.device('cpu'))

    def test_initial_ten_tensors_exact_plus36_buffers(self):
        self.assertEqual(sum(self.model.state_dict()[k].numel() for k in m.CENTER_BUFFERS),36)
        self.assertEqual(len(self.initial.state_dict()),10)
        for key,value in self.initial.state_dict().items():self.assertTrue(torch.equal(value,self.model.state_dict()[key]))

    def test_fixed_rule_matches_manual_forward_all33clamped(self):
        row=make_row()
        with torch.no_grad():
            x=((row['features']-self.model.fixed_x_center)/self.scale).clamp(-20,20)
            o=self.model.observation_model(row['predicted_ppm'],row['tags'],row['observation_input'],H=row['H'],K=row['K'])
            o=(o-self.model.fixed_o_center).clamp(-20,20)
            expected=5*torch.tanh(self.model.network(torch.cat((x,o),1))[:,row['kind']]/5)
            actual=self.model.row_delta(row,self.scale,torch.device('cpu'))
        self.assertTrue(torch.equal(expected,actual))

    def test_common_candidates_pool_invariant_fixed_raw(self):
        a,b=make_row(),make_row('D')
        with torch.no_grad():
            da=self.model.row_delta(a,self.scale,torch.device('cpu'))
            db=self.model.row_delta(b,self.scale,torch.device('cpu'))
        self.assertTrue(torch.allclose(da[:256],db[:256],rtol=1e-6,atol=1e-7))

    def test_network_step_preserves_O_scale_and_centers(self):
        s,d,_,_=case_data();row=d['rows'][0]
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        scale=self.scale.clone();optimizer=torch.optim.AdamW(self.model.network.parameters(),lr=.001,weight_decay=.001)
        delta=self.model.row_delta(row,self.scale,torch.device('cpu'))
        loss=m.base.pool_loss(row,delta,self.scale,.1,'listwise')
        loss.backward();torch.nn.utils.clip_grad_norm_(self.model.network.parameters(),2.);optimizer.step()
        for key,value in before.items():
            if key.startswith('observation_model.') or key in m.CENTER_BUFFERS:self.assertTrue(torch.equal(value,self.model.state_dict()[key]))
        self.assertTrue(torch.equal(scale,self.scale))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in self.model.state_dict().items() if k.startswith('network.')))

    def test_no_candidate_metadata_affects_scores(self):
        a=make_row();b=copy.deepcopy(a)
        b.update(source='changed',record_id='changed',smiles='wrong',group='wrong',target_position=500)
        with torch.no_grad():self.assertTrue(torch.equal(self.model.row_delta(a,self.scale,'cpu'),self.model.row_delta(b,self.scale,'cpu')))

    def test_fixed_budget_rejects_short_training_or_other_arm(self):
        args=SimpleNamespace(seed=193,training_pool='mixed',loss='listwise',**m.BUDGET)
        m.validate_fit_budget(args)
        args.epochs=32
        with self.assertRaisesRegex(ValueError,'full_budget'):m.validate_fit_budget(args)
        args.epochs=64;args.loss='pairwise16'
        with self.assertRaisesRegex(ValueError,'only_registered'):m.validate_fit_budget(args)


class EntryFixtureTests(unittest.TestCase):
    def test_fit_checkpoint_load_and_validation_evaluate(self):
        s,d,contract,groups=case_data()
        # no_aug role uses the2 observed fit query rows; the internal row remains validation-only.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out=root/'fit';out.mkdir()
            train=root/'single/train/features.pt';dual=root/'dual/train/features.pt'
            for p in (train,dual):p.parent.mkdir(parents=True);p.write_bytes(b'synthetic train cache')
            contractdir=root/'cohorts';contractdir.mkdir()
            contract.update(cohorts_sha256='payload_sha')
            cp=contractdir/'cohort_contract.json';cp.write_text(json.dumps(contract))
            manifest=dict(contract_sha256=m.input_fingerprints([cp])[str(cp)],cohorts_sha256='payload_sha')
            (contractdir/'manifest.json').write_text(json.dumps(manifest));(contractdir/'EXIT_CODE').write_text('0\n')
            payload=str(contractdir/'cohorts.pt')
            for data in (s,d):data['manifest']['source_inputs_sha256']={payload:'payload_sha'}
            initial,scale,saved=make_models()[0]
            saved.update(scale=scale,schema=m.base.OLD_SCHEMA,observation=True,initial_sigmas=initial.observation_model.initial_sigmas.tolist(),
                cache_protocol={k:s['manifest'][k] for k in m.base.IDENTITY}|dict(recall='single'),
                source_inputs_sha256={payload:'payload_sha'},ranker_source_sha256=m.input_fingerprints(m.base.source_paths()),
                train_sha256=m.input_fingerprints([train])[str(train)],epoch=0)
            # Same inherited role args as production; no score or center tuning.
            saved['args']=dict(seed=193,augment=False,**m.BUDGET)
            initialpath=root/'initial/best.pt';initialpath.parent.mkdir();torch.save(saved,initialpath)
            (initialpath.parent/'summary.json').write_text(json.dumps(dict(best=dict(epoch=0,alpha=saved['alpha']),training_cache_sha256=saved['train_sha256'])))
            (initialpath.parent/'EXIT_CODE').write_text('0\n')
            args=SimpleNamespace(train=str(train),dual_train=str(dual),initial_checkpoint=str(initialpath),
                cohort_contract=str(cp),seed=193,augment=False,training_pool='mixed',loss='listwise',**m.BUDGET)
            pins=m.input_fingerprints([train,dual,initialpath,cp])
            def load(path):return (s if path==str(train) else d),'synthetic'
            def schedule(names,gg,seed,epoch,samples):
                row=next(iter(gg.values()))[0]
                return [(row,0),(row,1)]
            def internal_validation(model,rows,scale,device,alphas):
                return {},{str(a):dict(overall=dict(MRR=.5)) for a in alphas}
            with patch.object(m,'CANDIDATE_COUNT',1024),patch.object(m,'CANDIDATE_SHA',m.ordered_sha(s['smiles'])),\
                 patch.object(m.identity_source,'identity',lambda x:('unused',groups[int(x.rsplit('_',1)[1])],'unused')),\
                 patch.object(m,'read_feature_cache',side_effect=load),patch.object(m.base,'training_schedule',side_effect=schedule),\
                 patch.object(m,'validation',side_effect=internal_validation):
                m.fit(args,out,torch.device('cpu'),lambda *a,**kw:None,pins)
            self.assertTrue((out/'last.pt').is_file());self.assertEqual(len((out/'history.jsonl').read_text().splitlines()),9)
            (out/'EXIT_CODE').write_text('0\n')
            model,loaded_scale,new_saved=m.load_checkpoint(out/'best.pt',torch.device('cpu'))
            self.assertEqual(new_saved['epoch'],0)
            self.assertTrue(torch.equal(loaded_scale,scale))
            self.assertEqual(new_saved['schema'],m.SCHEMA)
            estimate=json.loads((out/'centers.json').read_text())
            self.assertEqual(estimate['query_rows'],2)
            self.assertEqual(estimate['constant_fp32'],36)
            self.assertTrue(torch.equal(model.fixed_x_center,torch.tensor(estimate['center33'])))
            # Tampering constants evidence must fail load before evaluation.
            before=(out/'centers.json').read_bytes()
            changed=dict(estimate,center33=[99.]*33);(out/'centers.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'fixed_center_estimate_SHA'):m.load_checkpoint(out/'best.pt','cpu')
            (out/'centers.json').write_bytes(before)
            # Evaluation pair needs both augmented roles; use the same tensors with explicit true role for synthetic only.
            augmented=copy.deepcopy(new_saved);augmented['args']['augment']=True
            augpath=root/'aug';augpath.mkdir()
            for name in ('centers.json','center_rows.jsonl','summary.json','EXIT_CODE'):(augpath/name).write_bytes((out/name).read_bytes())
            torch.save(augmented,augpath/'best.pt')
            val=copy.deepcopy(s);val['manifest'].update(cohort='original_val');val['rows']=[copy.deepcopy(s['rows'][0])]
            row=val['rows'][0];row.update(coarse_rank=1,baseline_rank=1);row['negative_exclusion_mask']=row['negative_exclusion_mask'].clone()
            evalargs=SimpleNamespace(cache='synthetic_validation_cache',no_aug_checkpoint=str(out/'best.pt'),
                                    aug_checkpoint=str(augpath/'best.pt'),seed=193)
            evalout=root/'evaluation';evalout.mkdir()
            with patch.object(m,'read_feature_cache',return_value=(val,'sha')):
                m.evaluate(evalargs,evalout,torch.device('cpu'),lambda *a,**kw:None,{})
            self.assertTrue((evalout/'summary.json').is_file())
            self.assertEqual(len((evalout/'predictions.jsonl').read_text().splitlines()),1)

    def test_test_cohort_is_not_opened_for_scoring(self):
        with patch.object(m,'read_feature_cache',return_value=(dict(manifest=dict(cohort='test')),'sha')):
            with self.assertRaisesRegex(ValueError,'validation_only_no_test'):
                m.evaluate(SimpleNamespace(cache='synthetic'),Path('unused'),torch.device('cpu'),lambda *a,**k:None,{})


if __name__=='__main__':unittest.main()

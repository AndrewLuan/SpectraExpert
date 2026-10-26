"""功能概述：freeze_tools独立审计本轮evaluation四工具的数学与执行绑定边界。
原14独审用例迁移到独立native子目录，补充真实mixed precision oracle。
仅生成临时合成JSON和人工tensor；不读取真实cache/checkpoint/test，不派发远端。
测试以独立fp32逐步成本oracle核固定400/87解释算术，核自然512/source-blind评分、
native hit/miss语义、先seed比值再等均值、唯一C10dual及四guard、固定other/95%阈值。
另复用作者临时回执fixture攻击未绑定audit、bool退出、源码起始pin和成功run_name。
fixture依赖文件一并由独立review记录SHA；本文件不改被审实现。
"""
import copy
import contextlib
import io
import itertools
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch

import analyze_cap_mechanism as a
import cap_evaluation_common as c
import evaluate_cap_ranker as e
import summarize_cap_validation as s
from test_analyze_cap_mechanism import pairs
from test_cap_evaluation_entry import EntryFixture
from test_evaluate_cap_ranker import models, row
from test_summarize_cap_validation import MatrixFixture, put


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def native_scalar_cost(coarse, no_aug_delta, aug_delta, gamma=.25, scales=(1.,1.)):
    # 独立重放torch float32各次乘/加，不能把实数margin当作fp32成本的逐位真值。
    ca = f32(f32(no_aug_delta) * f32(scales[0]))
    cb = f32(f32(aug_delta) * f32(scales[1]))
    return f32(f32(f32(coarse) + f32(f32(1-gamma)*ca)) + f32(f32(gamma)*cb))


def fp32_pairs(large_first=True, saturation95=False):
    original, per_cap = pairs()
    for cap in c.CAPS:
        for i, r in enumerate(per_cap[cap]):
            old = original[i]
            z = r['fixed_train_pair']
            td = f32(-.95*cap) if saturation95 else f32(-4.9 if cap == 5 else -5.1)
            od = 0. if saturation95 else f32(.5 if cap == 5 else 9.99)
            coarse = [50.]*512
            coarse[10] = 1e7 if large_first and i == 0 else 1.
            coarse[2] = coarse[10] + old['original_coarse_gap']
            costs = coarse.copy()
            for index, delta in ((2,td),(10,od)):
                costs[index] = native_scalar_cost(coarse[index], delta, delta)
            r['natural_candidate_coarse_costs'] = coarse
            r['natural_candidate_costs'] = costs
            r['target_cost'] = costs[2]
            r['rank'] = sum(x <= f32(costs[2]+c.EPS) for x in costs)
            r['RR'] = 1/r['rank']
            z['actual_fixed_pair_margin'] = costs[2]-costs[10]
            for t in z['role_saturation']:
                t.update(target_delta=td, original_other_delta=od,
                         target_saturation=abs(td)/cap, original_other_saturation=abs(od)/cap,
                         target_saturated95=abs(td)/cap>=.95, original_other_saturated95=abs(od)/cap>=.95)
            # 消费正式producer字段；成本与δ由独立oracle给定，不调用模型或正式评分函数。
            delta=torch.zeros(512);delta[2]=td;delta[10]=od
            synthetic_row=dict(candidate_indices=torch.arange(512),target_position=2,target_in_coarse_pool=True,
                               coarse_scores=torch.tensor(coarse,dtype=torch.float32))
            scored=dict(cost=torch.tensor(costs,dtype=torch.float32),traces=[dict(delta=delta),dict(delta=delta)],
                        total_bound=cap,role_bounds=[cap,cap],gamma=.25,coarse_dtype='torch.float32',correction_dtypes=['torch.float32']*2,final_cost_dtype='torch.float32')
            r['fixed_train_pair']=e.fixed_pair_evidence(synthetic_row,scored,old,cap)
    return original, per_cap


class ScoringBoundaryTests(unittest.TestCase):
    def test_mixed_native_formula_and_dtype_independent_of_scoring_helper(self):
        for cap,dtype,gamma in itertools.product((5,10),(torch.float32,torch.float64),(.25,.75)):
            _,pair=models(cap);r=row();r['coarse_scores']=r['coarse_scores'].to(dtype)+.123456789123
            natural=e.natural_row(r)
            correction=[saved['alpha']*model.row_delta(natural,scale,'cpu')*scale[0] for model,scale,saved in pair]
            expected=natural['coarse_scores']+(1-gamma)*correction[0]+gamma*correction[1]
            scored=e.score_natural(pair,natural,gamma,'cpu',cap)
            self.assertTrue(torch.equal(scored['cost'],expected));self.assertEqual(scored['cost'].dtype,dtype)

    def test_source_labels_masks_and_513_poison_cannot_change_score(self):
        _, model_pair = models(10)
        r = row()
        before = e.score_natural(model_pair,e.natural_row(r),.25,'cpu',10)
        changed = copy.deepcopy(r)
        changed.update(source='other_source',record_id='private',group='other_group',target_position=512,
                       target_in_coarse_pool=False,outside_pool_rank=1001)
        changed['negative_exclusion_mask'][:] = True
        for key in e.CANDIDATE_FIELDS:
            if key in changed:
                changed[key][-1] = 1e10 if changed[key].dtype.is_floating_point else 99
        after = e.score_natural(model_pair,e.natural_row(changed),.25,'cpu',10)
        self.assertTrue(torch.equal(before['cost'],after['cost']))
        self.assertEqual(e.trace_hashes(before),e.trace_hashes(after))

    def test_raw33_and_O3_are_clamped_and_fixed_state_unchanged(self):
        _, model_pair = models(5)
        r=row();r['features'][:512] = 1e4;r['predicted_ppm'][:512] = -1e4
        digest=e.model_digest(model_pair)
        scored=e.score_natural(model_pair,e.natural_row(r),.75,'cpu',5)
        for t in scored['traces']:
            self.assertTrue(torch.equal(t['centered_X'],torch.full((512,33),20.)))
            self.assertTrue(torch.equal(t['centered_O'],torch.full((512,3),-20.)))
        self.assertEqual(digest,e.model_digest(model_pair))

    def test_hit_unused_fallback_ignored_and_miss_strict(self):
        _, model_pair=models(5);r=row();scored=e.score_natural(model_pair,e.natural_row(r),.25,'cpu',5)
        for sentinel in (None,0,-1,'unused'):
            r['outside_pool_rank']=sentinel
            self.assertTrue(e.outcome(r,scored)['target_in_pool'])
        r.update(target_position=512,target_in_coarse_pool=False)
        for bad in (None,0,512,True,513.0):
            r['outside_pool_rank']=bad
            with self.assertRaises(ValueError):e.outcome(r,scored)
        for good in (513,114982):
            r['outside_pool_rank']=good
            self.assertEqual(e.outcome(r,scored)['rank'],good)

    def test_public_gamma_is_independent_of_source_and_observation_metadata(self):
        for obs,expected in (({},.25),({'windows':[[1,2]]},.75),({'partial_peaks':True},.75)):
            changed=dict(obs,source='private',record_id='private',target_position=0)
            self.assertEqual(e.route_gamma(changed),expected)


class ArithmeticTests(unittest.TestCase):
    def test_scalar_native_replay_matches_torch_mixed_operations_and_EPS(self):
        for dtype,gamma in itertools.product((torch.float32,torch.float64),(.25,.75)):
            coarse=torch.tensor([10000015.1234567,-1e-7],dtype=dtype)
            d0=torch.tensor([-4.9,.0123456]);d1=torch.tensor([.5,-.9876543]);scale0=torch.tensor(1.234567);scale1=torch.tensor(.3456789)
            native=coarse+(1-gamma)*(d0*scale0)+gamma*(d1*scale1)
            for i in range(2):
                actual=c.native_cost_replay(float(coarse[i]),float(d0[i]),float(d1[i]),float(scale0),float(scale1),gamma,str(dtype),['torch.float32']*2)
                self.assertEqual(actual,float(native[i]))
            self.assertEqual(c.native_threshold(float(coarse[1]),str(dtype)),float(coarse[1]+1e-7))
    def test_all_four_guards_independently_veto_and_no_single_replacement(self):
        b=dict(c.BASELINE_SIGNALS);control=dict(b,MRR=.6);candidate=dict(b,MRR=.7)
        self.assertEqual(c.select(candidate,control,b)['preselected'],'cap10_dual')
        for guard in c.GUARDS:
            bad=dict(candidate);bad[guard]=b[guard]-2e-12
            selection=c.select(bad,control,b)
            self.assertFalse(selection['eligible']);self.assertEqual(selection['preselected'],c.BASELINE)
            self.assertFalse(selection['alternate_selection_allowed'])
        tie=dict(candidate,MRR=control['MRR']+5e-13)
        self.assertFalse(c.select(tie,control,b)['eligible'])

    def test_query_weighted_merge_then_mean_of_seed_conditional_ratios(self):
        matrix={}
        for cap,seed,pool,cohort in itertools.product(c.CAPS,c.SEEDS,c.POOLS,('original_val','tune')):
            count,offset=(2,0) if cohort=='original_val' else (3,2)
            rr=[]
            for i in range(count):
                hit=seed==211 or cohort=='original_val';rank=2 if seed==211 else (1 if hit else 1000)
                q=dict(query_order=offset+i,variant='observed',source=c.SOURCES[offset+i],record_id=cohort+str(i),smiles='C',group='g'+str(offset+i))
                rr.append(dict(query_identity=q,rank=rank,RR=1/rank,target_in_pool=hit))
            for variant in ('frequency_crop','peak_dropout'):
                rank=1000 if variant=='peak_dropout' and seed==193 else 2
                rr.append(dict(query_identity=dict(q,variant=variant),rank=rank,RR=1/rank,target_in_pool=rank<=512))
            summary=dict(records=len(rr),reranking_seconds_mean=.1,reranking_seconds_p95=.1,elapsed_seconds=1.,cost_semantics='synthetic')
            matrix[(cap,seed,pool,cohort)]=(summary,{'args':{'out':'synthetic'}},rr)
        with patch.object(c,'MERGED_OBSERVED',5):
            signals,_,table,_=s.aggregate(matrix)
        self.assertAlmostEqual(signals['cap10_dual']['MRR'],((2+3*.001)/5+.5)/2)
        self.assertAlmostEqual(signals['cap10_dual']['macro_MRR'],((2+3*.001)/5+.5)/2)
        mean=next(x for x in table if x['config']=='cap10_dual' and x['seed']=='two_seed_mean' and x['cohort']=='merged_validation' and x['variant']=='observed' and x['source']=='all')
        self.assertEqual(mean['in_pool_MRR'],.75)
        self.assertNotEqual(mean['in_pool_MRR'],(2+5*.5)/7)
        empty=next(x for x in table if x['config']=='cap10_dual' and x['seed']=='two_seed_mean' and x['cohort']=='merged_validation' and x['variant']=='peak_dropout' and x['source']=='all')
        self.assertIsNone(empty['in_pool_MRR'])

    def test_native_fp32_large_coarse_margin_is_valid_not_false_failure(self):
        old,per=fp32_pairs()
        self.assertEqual(per[5][0]['fixed_train_pair']['actual_fixed_pair_margin'],10.)
        self.assertEqual(per[10][0]['fixed_train_pair']['actual_fixed_pair_margin'],1.)
        report=a.fixed_pair_analysis(old,per)
        self.assertEqual(report['all400']['query_seed_pairs'],400)
        self.assertEqual(report['original87']['query_seed_pairs'],87)

    def test_saturation95_inclusive_normalized_to_own_cap_and_original_other_fixed(self):
        old,per=fp32_pairs(large_first=False,saturation95=True)
        report=a.fixed_pair_analysis(old,per)
        for seed in c.SEEDS:
            for cap in c.CAPS:
                roles=report['all400']['per_seed'][str(seed)]['caps'][str(cap)]['saturation']
                for role in ('no_aug','aug'):
                    self.assertEqual(roles[role]['target']['fraction'],1.)
                    self.assertEqual(roles[role]['original_other']['fraction'],0.)
        per[10][0]['fixed_train_pair']['original_other_candidate_index']=11
        with self.assertRaisesRegex(ValueError,'original_pair_unchanged'):a.fixed_pair_analysis(old,per)


class ExecutionClosureTests(unittest.TestCase):
    def test_actual_entry_reader_preserves_each_row_native_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=EntryFixture(directory)
            fixture.rows[0]['coarse_scores']=fixture.rows[0]['coarse_scores'].double()+1.234567891234e-8
            with contextlib.redirect_stdout(io.StringIO()):fixture.run()
            _,_,rows=c.read_evaluation(c.Pins(),fixture.out,5,193,'single','original_val',fixture.bindings,fixture.protocol,True)
            self.assertEqual(rows[0]['final_cost_dtype'],'torch.float64');self.assertEqual(rows[-1]['final_cost_dtype'],'torch.float32')
            self.assertTrue(any(f32(x)!=x for x in rows[0]['natural_candidate_costs']))

    def test_real_producer_superset_manifest_supported_without_weakening_SHA(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=EntryFixture(directory)
            binding=json.loads(fixture.bindings.read_text())
            binding['caches']['193']['original_val']['single']['manifest']['feature_cache_sha256']=c.sha(fixture.cache)
            fixture.bindings.write_text(json.dumps(binding))
            with contextlib.redirect_stdout(io.StringIO()): result=fixture.run()
            self.assertEqual(result['records'],4)

    def test_unregistered_valid_shaped_training_audit_must_be_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            f=MatrixFixture(directory);binding=json.loads(Path(f.bindings).read_text())
            fake=Path(directory)/'unregistered_audit/summary.json';fake.parent.mkdir()
            fake.write_bytes(Path(f.audit).read_bytes());(fake.parent/'EXIT_CODE').write_text('0\n')
            pins=c.Pins();pins.json(f.protocol)
            with self.assertRaises(ValueError):s.training_audit(pins,fake,binding,f.protocol)

    def test_summary_self_and_common_pinned_before_matrix_reads(self):
        class Stop(Exception):pass
        with tempfile.TemporaryDirectory() as directory:
            f=MatrixFixture(directory)
            def stop(pins,*args,**kwargs):
                self.assertIn(str(Path(s.__file__).resolve()),pins.values)
                self.assertIn(str(Path(c.__file__).resolve()),pins.values)
                raise Stop()
            with patch.object(s,'load_matrix',side_effect=stop):
                with self.assertRaises(Stop):f.execute()

    def test_mechanism_self_common_and_load_matrix_dependency_pinned_at_start(self):
        class Stop(Exception):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);origin=put(root/'fixed.json',dict(schema='cap10_fixed_train_pairs_20261024',records=[]))
            protocol=dict(schema='cap10_controlled_protocol_20261024',seeds=[193,211],caps=[5,10],pool_size=512,
                candidate_library_size=114982,promotion=False,test_allowed=False,final_epoch=64,alpha=1.,epoch_offset=0,
                only_eligible_new_config='cap10_dual',control_config='cap5_dual',preselected=c.BASELINE,
                mechanism=dict(saturation_threshold=.95,selection_input=False,fixed_train_pairs=dict(sha256=c.sha(origin))))
            pp=put(root/'protocol.json',protocol);bp=put(root/'bindings.json',dict(fixed_train_pairs=dict(path=origin,sha256=c.sha(origin))))
            args=type('Args',(),dict(protocol=pp,bindings=bp,validation_queue='unused',validation_run='unused',mechanism_queue='unused',mechanism_run='unused'))()
            out=root/'out';out.mkdir()
            def stop(pins,*args,**kwargs):
                for module in (a,c,s):self.assertIn(str(Path(module.__file__).resolve()),pins.values)
                raise Stop()
            with patch.object(s,'load_matrix',side_effect=stop):
                with self.assertRaises(Stop):a.run(args,out)

    def test_bool_job_exit_and_changed_success_run_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            f=MatrixFixture(directory,run_name='validation_final_success')
            completion=json.loads((f.run/'completion.json').read_text())
            completion['jobs'][0]['internal_exit']=False
            put(f.run/'completion.json',completion)
            put(f.run/'jobs'/(f.jobs[0]['id']+'.receipt.json'),completion['jobs'][0])
            with self.assertRaisesRegex(ValueError,'real_double_exit'):f.execute()

    def test_safe_resolves_test_symlink_before_payload_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'test').mkdir();target=root/'test/private.json';target.write_text('{}')
            alias=root/'benign.json';alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'test_path_forbidden'):c.Pins().json(alias)


if __name__=='__main__':
    unittest.main()

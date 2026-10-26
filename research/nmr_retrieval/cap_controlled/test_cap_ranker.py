"""功能概述：新cap模型及续训入口的合成核验，不读取真实缓存或启动真实训练。
使用真实已pin FixedCenterRanker/O/network核C5逐位一致、C10解析式、共同候选不变性、
strict负例资格和loss-only追加target。EntryFixture只替换缓存/旧checkpoint加载与小型
candidate身份表，并在测试内把draw预算缩成每epoch2draw，实际贯通全部64epoch的
main→训练→initial/final保存→严格新loader。生产CLI没有减预算选项。
另核源/producer漂移、固定final/alpha/cap拒绝、中心与O/scale不变及两臂曝光SHA一致。
测试所有文件仅写临时目录；读取的科学文件只是模块源码，不含test数据或旧test报告。
"""
import argparse
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import cap_ranker as c
import train_cap_ranker as t

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
SOURCE = next(p for p in (Path('/Users/andrewluan/Desktop/Projects/IRfingerprints/SpectrExpert'),
                         Path('/home/user/IRfingerprints/SpectrExpert')) if (p/'nmr_transfer/h_pool_ranker.py').is_file())
FIXED = next(p for p in (HERE.parent/'orchestrator-261023-1335/h_pool_fixed_center_ranker.py',
                        SOURCE/'outputs/nmr_pool_ranking_20261023/h_pool_fixed_center_ranker.py') if p.is_file())
RT = c.configure_runtime(SOURCE, FIXED)


def group_name(internal, start=0):
    while RT.internal_group('cap_group_' + str(start)) is not internal:
        start += 1
    return 'cap_group_' + str(start)


def row(pool=0, target=10, variant='observed', order=0):
    ids = torch.arange(512) if pool == 0 else torch.cat((torch.arange(256), torch.arange(512, 768)))
    positions = (ids == target).nonzero().flatten()
    hit = len(positions) > 0
    if not hit:
        ids = torch.cat((ids, torch.tensor([target])))
    values = ids.float()
    features = torch.stack([torch.sin(values / (2 + j)) + j*.05 for j in range(33)], 1)
    coarse = values*.001
    features[:, 0] = coarse
    return dict(query_order=order, source='synthetic', record_id=str(order), smiles='CC', group=group_name(False),
        variant=variant, mask_content_sha256='mask_' + variant, candidate_indices=ids, evaluation_pool_size=512,
        features=features, coarse_scores=coarse, predicted_ppm=values[:, None]%11*.1 + torch.tensor([[.1,1.,2.,3.]]),
        tags=torch.zeros((len(ids), 4), dtype=torch.long), H=torch.full((len(ids),),10.), K=torch.full((len(ids),),10.),
        observation_input=dict(semantics='quantile_support', xs=[1.,2.], weights=[.5,.5], windows=[],
                               partial_peaks=False, reported_count=None), kind=0,
        query_observation=dict(windows=[], partial_peaks=False), target_position=int(positions[0]) if hit else 512,
        target_in_coarse_pool=hit, outside_pool_rank=None if hit else 1291)


def native_models(cap=5):
    torch.manual_seed(101)
    native = RT.fixed.FixedCenterRanker([.2]*9, torch.linspace(-.2,.2,33), [.1,.2,.3]).eval()
    with torch.no_grad():
        native.network[-1].weight.normal_(0,.3)
    model = c.CapRanker([.2]*9, native.fixed_x_center, native.fixed_o_center, cap, RT).eval()
    model.load_state_dict(native.state_dict())
    return native, model, torch.linspace(.5, 2., 33)


def qualification():
    query = group_name(False)
    groups = [group_name(False, 100)]*1024
    groups[10] = groups[900] = query
    groups[1] = 'external_heldout'
    groups[2] = group_name(True)
    return dict(groups=groups, internal=torch.tensor([RT.internal_group(g) for g in groups]),
                heldout={'external_heldout'}, metadata=dict(derived_candidate_groups_sha256='group_SHA',
                    historical_heldout_sha256='heldout_SHA'))


def with_mask(value, q=None):
    q = q or qualification()
    value['negative_exclusion_mask'] = torch.tensor([q['groups'][int(i)] == value['group'] or
        q['groups'][int(i)] in q['heldout'] for i in value['candidate_indices']])
    return value


def natural(value):
    keys = ('features','coarse_scores','predicted_ppm','tags','H','K')
    return dict({k:value[k][:512] for k in keys}, kind=value['kind'],
                observation_input=value['observation_input'], query_observation=value['query_observation'])


class ModelTests(unittest.TestCase):
    def test_C5_native_bitwise_all_heads_and_observation_routes(self):
        native, model, scale = native_models()
        for kind in range(4):
            for route in ({}, {'windows': [[.5,2.5]]}, {'partial_peaks': True}):
                r = row(); r['kind'] = kind
                r['observation_input'].update(route)
                self.assertTrue(torch.equal(native.row_delta(r,scale,'cpu'), model.row_delta(r,scale,'cpu')))
        self.assertEqual(c.tensor_hashes(native,scale), c.tensor_hashes(model,scale))

    def test_C10_loss_and_inference_use_same_cap_expression(self):
        _, model, scale = native_models(10)
        r = row()
        with torch.no_grad():
            model.network[0].weight.zero_()
            model.network[0].bias.fill_(1.)
            model.network[-1].weight.fill_(1.5)
            z = ((r['features']-model.fixed_x_center)/scale).clamp(-20,20)
            o = model.observation_model(r['predicted_ppm'],r['tags'],r['observation_input'],H=r['H'],K=r['K'])
            logits = model.network(torch.cat((z,(o-model.fixed_o_center).clamp(-20,20)),1))[:,0]
            expected = 10*torch.tanh(logits/10)
            self.assertTrue(torch.equal(model.row_delta(r,scale,'cpu'),expected))
            self.assertTrue((expected.abs() <= 10).all())
            self.assertTrue((expected.abs() > 5).any())

    def test_common_candidates_costs_invariant_without_pool_statistics(self):
        _, model, scale = native_models(10)
        s, d = natural(row(0)), natural(row(1))
        a, b = model.row_delta(s,scale,'cpu'), model.row_delta(d,scale,'cpu')
        self.assertTrue(torch.equal(a[:256],b[:256]))

    def test_score_pair_native_gamma_and_cost(self):
        _, model, scale = native_models()
        models = [(model,scale,dict(alpha=1.,delta_cap=5))]*2
        for obs, gamma in (({},.25),({'windows':[[1.,2.]]},.75),({'partial_peaks':True},.75)):
            r = natural(row()); r['query_observation'] = obs
            result, actual_gamma = c.score_pair(models,r,'cpu')
            delta = model.row_delta(r,scale,'cpu')*scale[0]
            expected = r['coarse_scores']+(1-gamma)*delta+gamma*delta
            self.assertTrue(torch.equal(result,expected)); self.assertEqual(gamma,actual_gamma)

    def test_score_rejects_injected_positive_and_identity(self):
        _, model, scale = native_models()
        models = [(model,scale,dict(alpha=1.,delta_cap=5))]*2
        r = natural(row()); r['source'] = 'hidden'
        with self.assertRaisesRegex(ValueError,'identity'): c.score_pair(models,r,'cpu')
        r = row(target=900)
        with self.assertRaisesRegex(ValueError,'natural512'): c.score_pair(models,r,'cpu')

    def test_only_network_updates_and_frozen_hash_exact(self):
        _, model, scale = native_models(10)
        before = c.frozen_digest(model,scale)
        r = with_mask(row(target=900)); q = qualification()
        mask = c.strict_negative_mask(r,q,RT)
        network = {k:v.clone() for k,v in model.network.state_dict().items()}
        opt = torch.optim.AdamW(model.network.parameters(),lr=.001,weight_decay=.001)
        loss = c.listwise_loss(r,model.row_delta(r,scale,'cpu'),scale,mask)
        loss.backward(); opt.step()
        self.assertEqual(c.frozen_digest(model,scale),before)
        self.assertTrue(any(not torch.equal(v,network[k]) for k,v in model.network.state_dict().items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.observation_model.parameters()))


class MaskTests(unittest.TestCase):
    def test_strict_external_internal_samegroup_and_loss_only_target(self):
        r = with_mask(row(target=900)); q = qualification()
        m = c.strict_negative_mask(r,q,RT)
        self.assertTrue(m[1] and m[2] and m[10] and m[512])
        self.assertFalse(m[3])
        delta = torch.zeros(513,requires_grad=True)
        loss = c.listwise_loss(r,delta,torch.ones(33),m)
        loss.backward()
        self.assertNotEqual(float(delta.grad[512]),0)
        self.assertEqual(float(delta.grad[1]),0); self.assertEqual(float(delta.grad[2]),0)

    def test_wrong_historical_mask_rejected(self):
        r = with_mask(row()); r['negative_exclusion_mask'][1] = False
        with self.assertRaisesRegex(ValueError,'historical_negative'): c.strict_negative_mask(r,qualification(),RT)

    def test_internal_or_external_query_never_fit(self):
        for g in (group_name(True),'external_heldout'):
            r = with_mask(row()); r['group'] = g
            with self.assertRaisesRegex(ValueError,'fit_query'): c.strict_negative_mask(r,qualification(),RT)

    def test_no_eligible_negative_fails(self):
        r = row(); q = qualification(); q['groups'] = [r['group']]*1024
        r = with_mask(r,q)
        with self.assertRaisesRegex(ValueError,'no_eligible'): c.strict_negative_mask(r,q,RT)

    def test_loss_excluded_poison_does_not_change_value(self):
        r = with_mask(row()); m = c.strict_negative_mask(r,qualification(),RT)
        a = torch.zeros(512); b = a.clone(); b[m] = 1e9; b[r['target_position']] = 0
        self.assertTrue(torch.equal(c.listwise_loss(r,a,torch.ones(33),m),c.listwise_loss(r,b,torch.ones(33),m)))


class ScheduleTests(unittest.TestCase):
    def test_paired_RNG_exact_half_all64epochs_and_role_variant_order(self):
        groups = {'a':[dict(id=0),dict(id=1)],'b':[dict(id=2)]}
        for seed in (193,211):
            for epoch in range(1,65):
                a = RT.base.training_schedule(sorted(groups),groups,seed,epoch,8192)
                b = RT.base.training_schedule(sorted(groups),groups,seed,epoch,8192)
                self.assertEqual(a,b)
                self.assertEqual(sum(pool for _,pool in a),4096)

    def test_protocol_rejects_selection_budget_or_permission_drift(self):
        p = dict(schema='cap10_controlled_protocol_20261024',training_allowed=True,test_allowed=False,
            seeds=[193,211],caps=[5,10],budget=dict(c.BUDGET),final_epoch=64,alpha=1,epoch_offset=0)
        args = argparse.Namespace(seed=193,cap=5,role='no_aug')
        t.validate_protocol(p,args)
        for field,value in (('training_allowed',False),('test_allowed',True),('alpha',.5),('final_epoch',32),('epoch_offset',64)):
            bad=copy.deepcopy(p); bad[field]=value
            with self.assertRaises(ValueError): t.validate_protocol(bad,args)
        bad=copy.deepcopy(p); bad['budget']['weight_decay']=.01
        with self.assertRaisesRegex(ValueError,'budget'): t.validate_protocol(bad,args)


class EntryFixture:
    """完整main fixture；仅CPU内存合成缓存与mock旧producer，生产代码预算不可由CLI缩减。"""
    def __init__(self, root, role='no_aug', cap_value=5):
        self.root=Path(root); self.role=role; self.cap=cap_value
        self.q=qualification(); self.native,self.model,self.scale=native_models()
        self.data={}
        for pool in (0,1):
            rows=[with_mask(row(pool,10,'observed',0),self.q),with_mask(row(pool,900,'observed',1),self.q),
                  with_mask(row(pool,10,'frequency_crop',2),self.q)]
            manifest=dict(schema='nmr_research_20261022',cohort='train',recall=('single','dual')[pool],
                candidate_count=114982,pool_size=512,candidate_order_sha256='order',query_sha256='queries',
                identity_content_sha256='identities',variant_mask_content_sha256='variantmasks',
                negative_excluded_structure_sha256='heldout_SHA',source_inputs_sha256={},training_subset='train')
            self.data[pool]=dict(schema=manifest['schema'],manifest=manifest,rows=rows,smiles=['unused']*1024)
        self.protocol=dict(schema='cap10_controlled_protocol_20261024',training_allowed=True,test_allowed=False,
            seeds=[193,211],caps=[5,10],budget=dict(c.BUDGET),final_epoch=64,alpha=1,epoch_offset=0,
            remote_root=str(self.root/'remote'))
        self.pp=self.root/'protocol.json';self.pp.write_text(json.dumps(self.protocol))
        self.bp=self.root/'bindings.json'; self.cache_entries={}
        for pool,name in ((0,'single'),(1,'dual')):
            directory=self.root/'train'/name;directory.mkdir(parents=True)
            cache=directory/'features.pt';cache.write_bytes(b'synthetic cache, not real tensor payload')
            producer_manifest=dict(self.data[pool]['manifest'],feature_cache_sha256=c.sha256(cache))
            producer=directory/'summary.json';producer.write_text(json.dumps(producer_manifest))
            (directory/'EXIT_CODE').write_text('0\n')
            self.cache_entries[name]=dict(path=str(cache),sha256=c.sha256(cache),producer=str(producer),manifest=producer_manifest)
        parent=self.root/'initial';parent.mkdir()
        ck=parent/'best.pt';ck.write_bytes(b'synthetic parent handled by explicit mock loader')
        rows=parent/'center_rows.jsonl';rows.write_text('{"synthetic":true}\n')
        centers=parent/'centers.json';centers.write_text(json.dumps(dict(center33=self.native.fixed_x_center.tolist(),
            O_center3=self.native.fixed_o_center.tolist(),row_audit_sha256=c.sha256(rows))))
        center_sha=c.sha256(centers)
        self.initial=dict(schema=RT.fixed.SCHEMA,alpha=1,epoch=24,initial_sigmas=[.2]*9,
            args=dict(seed=193,augment=role=='aug',training_pool='mixed',loss='listwise',**RT.fixed.BUDGET),
            train_sha256=self.cache_entries['single']['sha256'],dual_train_sha256=self.cache_entries['dual']['sha256'],
            center_estimate_sha256=center_sha,cache_protocol={k:self.data[0]['manifest'].get(k) for k in RT.base.IDENTITY},
            source_inputs_sha256={})
        producer=parent/'summary.json';producer.write_text(json.dumps(dict(best=dict(epoch=24,alpha=1),center_estimate_sha256=center_sha)))
        (parent/'EXIT_CODE').write_text('0\n')
        record=dict(checkpoint=str(ck),checkpoint_sha256=c.sha256(ck),centers=str(centers),centers_sha256=center_sha,
            center_rows=str(rows),producer_summary=str(producer),center33=self.native.fixed_x_center.tolist(),
            O_center3=self.native.fixed_o_center.tolist(),qualification=self.q['metadata'])
        contract=self.root/'cohorts/cohort_contract.json';contract.parent.mkdir()
        contract.write_text('{}');(contract.parent/'manifest.json').write_text('{}');(contract.parent/'EXIT_CODE').write_text('0\n')
        self.bindings=dict(schema='cap10_execution_bindings_20261024',source_checkout=str(SOURCE),
            caches={'193':{'train':self.cache_entries}},roles={'193':{role:record}},cohort_contract=str(contract),
            heldout_groups_sha256='heldout_SHA')
        paths=RT.source_paths()+[Path(t.__file__),self.pp]
        paths+=list(parent.iterdir())+list(contract.parent.iterdir())
        for entry in self.cache_entries.values():paths+=list(Path(entry['path']).parent.iterdir())
        self.bindings['input_sha256']=c.fingerprints(paths);self.bp.write_text(json.dumps(self.bindings))
        self.out=self.root/'remote/rankers'/('C'+str(cap_value))/'s193'/role
        self.args=['--source',str(SOURCE),'--fixed-module',str(FIXED),'--protocol',str(self.pp),'--bindings',str(self.bp),
                   '--cap',str(cap_value),'--seed','193','--role',role,'--device','cpu','--out',str(self.out)]

    def run(self):
        def read(path, **kwargs):
            pool=0 if str(path)==self.cache_entries['single']['path'] else 1
            return copy.deepcopy(self.data[pool]),None
        def initial_loader(path,device):
            return copy.deepcopy(self.native).to(device),self.scale.to(device),copy.deepcopy(self.initial)
        with patch.object(c,'configure_runtime',return_value=RT),patch.object(RT,'read_feature_cache',side_effect=read),\
             patch.object(RT.fixed,'candidate_qualification',return_value=self.q),\
             patch.object(RT.fixed,'validate_contract_producer',return_value={}),\
             patch.object(RT.fixed,'load_checkpoint',side_effect=initial_loader),patch.object(RT.base,'check_science'):
            with contextlib.redirect_stdout(io.StringIO()):
                t.main(self.args)
        return json.loads((self.out/'summary.json').read_text())


class EntryTests(unittest.TestCase):
    @staticmethod
    def tiny_budget():
        return dict(c.BUDGET,samples=2,batch_size=2,draws_per_fit=128,optimizer_steps=64)

    def test_actual_main64epochs_save_load_and_C5C10_paired_audit(self):
        results=[]
        for cap_value in (5,10):
            with tempfile.TemporaryDirectory() as directory,patch.object(c,'BUDGET',self.tiny_budget()):
                f=EntryFixture(directory,cap_value=cap_value)
                summary=f.run();results.append(summary)
                self.assertEqual((f.out/'EXIT_CODE').read_text(),'0\n')
                self.assertFalse((f.out/'best.pt').exists())
                self.assertEqual(summary['epoch'],64);self.assertEqual(summary['alpha'],1)
                self.assertEqual(len(summary['epoch_records']),64)
                self.assertEqual(summary['draws'],128);self.assertEqual(summary['optimizer_steps'],64)
                model,scale,saved=c.load_checkpoint(f.out/'final.pt','cpu',expected_cap=cap_value,runtime=RT)
                self.assertEqual(c.frozen_digest(model,scale),summary['frozen_tensor_sha256'])
                self.assertEqual(saved['initial_tensor_sha256'],c.tensor_hashes(f.native,f.scale))
                self.assertEqual(c.sha256(f.out/'centers.json'),c.sha256(Path(f.bindings['roles']['193']['no_aug']['centers'])))
                with self.assertRaisesRegex(ValueError,'expected_cap'):c.load_checkpoint(f.out/'final.pt','cpu',expected_cap=15,runtime=RT)
        for key in ('exposure_sha256','pool_exposure_sha256','negative_exposure_sha256','strict_negative_qualification_sha256',
                    'initial_tensor_sha256','frozen_tensor_sha256'):
            self.assertEqual(results[0][key],results[1][key])
        for a,b in zip(results[0]['epoch_records'],results[1]['epoch_records']):
            for key in ('exposure_sha256','pool_exposure_sha256','negative_exposure_sha256'):
                self.assertEqual(a[key],b[key])

    def test_aug_main_retains_cache_variant_order_and_full_loader(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(c,'BUDGET',self.tiny_budget()):
            f=EntryFixture(directory,role='aug',cap_value=10);summary=f.run()
            self.assertEqual(summary['training_rows'],3)
            model,scale,saved=c.load_checkpoint(f.out/'final.pt','cpu',runtime=RT)
            self.assertTrue(saved['args']['augment']);self.assertEqual(saved['role'],'aug')

    def test_missing_or_changed_declared_source_fails_before_out(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(c,'BUDGET',self.tiny_budget()):
            f=EntryFixture(directory);del f.bindings['input_sha256'][str(Path(t.__file__).resolve())]
            f.bp.write_text(json.dumps(f.bindings))
            with patch.object(c,'configure_runtime',return_value=RT):
                with self.assertRaisesRegex(ValueError,'binding_missing'):t.main(f.args)
            self.assertFalse(f.out.exists())

    def test_cache_producer_or_output_role_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(c,'BUDGET',self.tiny_budget()):
            f=EntryFixture(directory);f.args[-1]=str(f.root/'wrong_output')
            with patch.object(c,'configure_runtime',return_value=RT):
                with self.assertRaisesRegex(ValueError,'registered_new_fit_output'):t.main(f.args)
            self.assertFalse((f.root/'wrong_output').exists())

    def test_center_or_producer_final_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(c,'BUDGET',self.tiny_budget()):
            f=EntryFixture(directory);f.run()
            (f.out/'centers.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'center_artifact'):c.load_checkpoint(f.out/'final.pt','cpu',runtime=RT)

    def test_producer_pool_or_budget_metadata_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(c,'BUDGET',self.tiny_budget()):
            f=EntryFixture(directory);f.run()
            original=json.loads((f.out/'summary.json').read_text())
            for field,value in (('pool_exposure_sha256','wrong'),('strict_negative_qualification_sha256','wrong'),
                                ('center_rows_sha256','wrong'),('optimizer_steps',63)):
                altered=dict(original);altered[field]=value
                (f.out/'summary.json').write_text(json.dumps(altered))
                with self.assertRaises(ValueError):c.load_checkpoint(f.out/'final.pt','cpu',runtime=RT)

    def test_actual_old_producer_summary_additional_SHA_key_supported(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(c,'BUDGET',self.tiny_budget()):
            f=EntryFixture(directory)
            self.assertNotIn('feature_cache_sha256',f.data[0]['manifest'])
            self.assertIn('feature_cache_sha256',f.cache_entries['single']['manifest'])
            summary=f.run();self.assertTrue(summary['initial_tensors_inherited_exact'])

    def test_self_consistent_producer_manifest_wrong_cache_SHA_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(c,'BUDGET',self.tiny_budget()):
            f=EntryFixture(directory)
            entry=f.cache_entries['single'];entry['manifest']['feature_cache_sha256']='0'*64
            producer=Path(entry['producer']);producer.write_text(json.dumps(entry['manifest']))
            f.bindings['input_sha256'][str(producer.resolve())]=c.sha256(producer);f.bp.write_text(json.dumps(f.bindings))
            with patch.object(c,'configure_runtime',return_value=RT):
                with self.assertRaisesRegex(ValueError,'producer_payload_SHA'):t.main(f.args)
            self.assertFalse(f.out.exists())

    def test_nonfinal_or_alpha_selection_checkpoint_rejected(self):
        base=dict(schema=c.SCHEMA,observation=True,delta_cap=5,budget=c.BUDGET,negative_mask_policy=c.MASK_POLICY,
            seed=193,role='no_aug',alpha=1.,epoch=64,draws=524288,optimizer_steps=16384,
            args=dict(seed=193,augment=False,cap=5,training_pool='mixed',loss='listwise'),
            strict_negative_qualification_sha256='q',parent_checkpoint_sha256='p',center_estimate_sha256='c',center_rows_sha256='r')
        c.validate_saved(base)
        for key,value in (('epoch',32),('alpha',.5),('draws',10),('delta_cap',7),('schema',RT.fixed.SCHEMA)):
            bad=dict(base);bad[key]=value
            with self.assertRaises(ValueError):c.validate_saved(bad)


if __name__=='__main__':
    unittest.main()

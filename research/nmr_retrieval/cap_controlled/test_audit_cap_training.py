"""功能概述：训练审计器的临时合成回归，覆盖真实产物和对抗篡改。

复用训练作者 fixture 生成小 draws 的完整 64 epoch 产物；审计器自行重算
资格和抽样。driver 测试使用真实临时 wrapper/子进程，全部无科学输入。
不读取真实 cache/checkpoint/test，也不执行正式训练或远程调用。
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import audit_cap_training as a
import cap_ranker as cap
import test_cap_ranker as training
from test_controlled_operator_independent import SupervisorFixture


class ModelAuditFixture:
    def __init__(self, directory, cap_value=5, role='aug'):
        self.f = training.EntryFixture(directory, role=role, cap_value=cap_value)
        self.record = self.f.bindings['roles']['193'][role]
        parent = dict(self.f.initial, state_dict={k:v.detach().clone() for k,v in self.f.native.state_dict().items()},
                      scale=self.f.scale.clone())
        torch.save(parent, self.record['checkpoint'])
        self.record['checkpoint_sha256'] = cap.sha256(self.record['checkpoint'])
        self.f.bindings['input_sha256'][str(Path(self.record['checkpoint']).resolve())] = self.record['checkpoint_sha256']
        self.f.bp.write_text(json.dumps(self.f.bindings))
        self.f.run()
        self.expected = a.replay_training(self.f.data[0], self.f.data[1], self.f.q['groups'],
                                         self.f.q['heldout'], role, 193)
        self.role, self.cap = role, cap_value

    def check(self, inputs=None):
        return a.audit_model(self.f.out, self.record['checkpoint'], self.record, self.cap, 193,
                             self.role, self.expected, inputs or a.Inputs())

    def final(self, mutate):
        path = self.f.out / 'final.pt'
        value = torch.load(path, weights_only=False, map_location='cpu')
        mutate(value)
        torch.save(value, path)
        summary = json.loads((self.f.out / 'summary.json').read_text())
        summary.update({k:v for k,v in value.items() if k not in ('state_dict','scale','initial_sigmas')})
        summary['final_checkpoint_sha256'] = cap.sha256(path)
        (self.f.out / 'summary.json').write_text(json.dumps(summary))


class ModelAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tiny = dict(cap.BUDGET, samples=2, batch_size=2, draws_per_fit=128, optimizer_steps=64)
        self.patches = [patch.object(cap,'BUDGET',self.tiny), patch.object(a,'BUDGET',self.tiny)]
        for p in self.patches: p.start()
        self.fixture = ModelAuditFixture(self.tmp.name)

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def test_complete_model_actual64_replay_and_tensor_audit(self):
        row = self.fixture.check()
        self.assertEqual((row['cap'],row['seed'],row['role'],row['epochs'],row['alpha']), (5,193,'aug',64,1))
        self.assertEqual((row['draws'],row['optimizer_steps']), (128,64))
        self.assertEqual(row['exposure_sha256'],self.fixture.expected['exposure_sha256'])

    def test_final_state_shape_change_even_same_capacity_rejected(self):
        def mutate(value):
            key=next(k for k,v in value['state_dict'].items() if k.startswith('network.') and v.ndim==2)
            value['state_dict'][key]=value['state_dict'][key].t().contiguous()
        self.fixture.final(mutate)
        with self.assertRaisesRegex(ValueError,'shape_dtype'):
            self.fixture.check()

    def test_final_state_dtype_change_rejected(self):
        def mutate(value):
            key=next(k for k in value['state_dict'] if k.startswith('network.'))
            value['state_dict'][key]=value['state_dict'][key].double()
        self.fixture.final(mutate)
        with self.assertRaisesRegex(ValueError,'shape_dtype'):
            self.fixture.check()

    def test_O_or_full_scale_or_center_frozen_against_parent(self):
        original=torch.load(self.fixture.f.out/'final.pt',weights_only=False,map_location='cpu')
        for key in ('scale','fixed_x_center','observation'):
            torch.save(original,self.fixture.f.out/'final.pt')
            def mutate(value):
                if key=='scale':value['scale']=value['scale'].clone();value['scale'][32]+=.1
                else:
                    name='fixed_x_center' if key=='fixed_x_center' else next(k for k in value['state_dict'] if k.startswith('observation_model.') and value['state_dict'][k].numel())
                    value['state_dict'][name]=value['state_dict'][name].clone()
                    value['state_dict'][name].reshape(-1)[0]+=.1
            self.fixture.final(mutate)
            with self.assertRaisesRegex(ValueError,'frozen_state_changed|scale_exact_parent'):
                self.fixture.check()

    def test_initial_network_tensor_tamper_rejected(self):
        path=self.fixture.f.out/'initial.pt'
        value=torch.load(path,weights_only=False,map_location='cpu')
        key=next(k for k in value['state_dict'] if k.startswith('network.'))
        value['state_dict'][key].reshape(-1)[0]+=1
        torch.save(value,path)
        summary=json.loads((self.fixture.f.out/'summary.json').read_text())
        summary['initial_checkpoint_sha256']=cap.sha256(path)
        (self.fixture.f.out/'summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError,'initial_or_frozen_state'):
            self.fixture.check()

    def test_missing_history_epoch_not_undertraining_excuse(self):
        p=self.fixture.f.out/'history.jsonl';values=[json.loads(x) for x in p.read_text().splitlines()][:-1]
        p.write_text(''.join(json.dumps(x)+'\n' for x in values))
        summary=json.loads((self.fixture.f.out/'summary.json').read_text());summary['epoch_records']=values
        (self.fixture.f.out/'summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError,'complete_real64_history'):
            self.fixture.check()

    def test_final_budget_or_alpha_tamper_rejected(self):
        original=torch.load(self.fixture.f.out/'final.pt',weights_only=False,map_location='cpu')
        for key,value in [('epoch',63),('alpha',.5),('draws',127),('optimizer_steps',63)]:
            torch.save(original,self.fixture.f.out/'final.pt')
            self.fixture.final(lambda state:state.update({key:value}))
            with self.assertRaisesRegex(ValueError,'checkpoint_budget|fixed_contract'):
                self.fixture.check()

    def test_self_consistent_final_exposure_lie_rejected_by_replay(self):
        self.fixture.final(lambda state:state.update(exposure_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'independent_final_exposure'):
            self.fixture.check()

    def test_negative_qualification_artifact_tamper_rejected(self):
        p=self.fixture.f.out/'negative_qualification.json';value=json.loads(p.read_text())
        value['records'][0]['negatives']+=1;p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'all_negative_qualification'):
            self.fixture.check()

    def test_cache_oldmask_tamper_and_zero_natural_eligibility_rejected(self):
        single,dual=copy.deepcopy(self.fixture.f.data[0]),copy.deepcopy(self.fixture.f.data[1])
        single['rows'][0]['negative_exclusion_mask'][1]=False
        with self.assertRaisesRegex(ValueError,'historical_mask_semantics'):
            a.replay_training(single,dual,self.fixture.f.q['groups'],self.fixture.f.q['heldout'],'aug',193)

    def test_snapshot_detects_input_drift_at_end(self):
        inputs=a.Inputs();inputs.pin(self.fixture.f.out/'summary.json')
        (self.fixture.f.out/'summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'input_changed_at_finish'):
            inputs.finish()

    def test_nan_final_tensor_refused(self):
        def mutate(value):
            key=next(k for k in value['state_dict'] if k.startswith('network.'))
            value['state_dict'][key].reshape(-1)[0]=float('nan')
        self.fixture.final(mutate)
        with self.assertRaisesRegex(ValueError,'shape_dtype_finite'):
            self.fixture.check()

    def test_unadmitted_saved_input_rejected_before_open(self):
        path=Path(self.tmp.name)/'unadmitted.json';path.write_text('{"must_not_read":true}')
        original=torch.load(self.fixture.f.out/'initial.pt',weights_only=False,map_location='cpu')
        admitted=dict(original['input_sha256'],**original['ranker_source_sha256'])
        def mutate(value):value['input_sha256'][str(path)]=cap.sha256(path)
        self.fixture.final(mutate)
        real_sha=a.sha;calls=[]
        def spy(p):
            calls.append(str(Path(p).resolve()));return real_sha(p)
        with patch.object(a,'sha',side_effect=spy):
            with self.assertRaisesRegex(ValueError,'unadmitted_model_input'):
                a.audit_model(self.fixture.f.out,self.fixture.record['checkpoint'],self.fixture.record,5,193,'aug',self.fixture.expected,a.Inputs(),admitted)
        self.assertNotIn(str(path.resolve()),calls)

    def test_producer_and_saved_input_maps_cannot_jointly_omit_required_source(self):
        key=next(k for k in torch.load(self.fixture.f.out/'final.pt',weights_only=False)['input_sha256']
                 if Path(k).name=='h_global_features.py')
        self.fixture.final(lambda value:(value['input_sha256'].pop(key),value['ranker_source_sha256'].pop(key)))
        p=self.fixture.f.out/'protocol.json';value=json.loads(p.read_text());value['input_sha256'].pop(key)
        p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'producer_complete_input_closure'):
            self.fixture.check()

    def test_real_job_CLI_binding_rejects_self_consistent_wrong_device(self):
        p=self.fixture.f.out/'protocol.json';value=json.loads(p.read_text());wanted=copy.deepcopy(value['args'])
        value['args']['device']='cuda:7';p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'producer_CLI_exact_actual_job'):
            a.audit_model(self.fixture.f.out,self.fixture.record['checkpoint'],self.fixture.record,
                5,193,'aug',self.fixture.expected,a.Inputs(),expected_cli=wanted)

    def test_finish_refuses_redirect_to_even_an_allowed_metadata_file(self):
        root=Path(self.tmp.name);first=root/'first.json';second=root/'second.json'
        first.write_text('{}');second.write_text('{}');inputs=a.Inputs();inputs.pin(first)
        first.unlink();first.symlink_to(second)
        with self.assertRaisesRegex(ValueError,'canonical_input_redirected'):
            inputs.finish()

    def test_all_natural_candidates_internal_leaves_no_eligible_negatives(self):
        single,dual=copy.deepcopy(self.fixture.f.data[0]),copy.deepcopy(self.fixture.f.data[1])
        groups=[training.group_name(True)]*1024
        groups[10]=groups[900]=training.group_name(False)
        heldout=self.fixture.f.q['heldout']
        for data in (single,dual):
            for row in data['rows']:
                row['negative_exclusion_mask']=torch.tensor([groups[int(i)]==row['group'] or groups[int(i)] in heldout for i in row['candidate_indices']])
        with self.assertRaisesRegex(ValueError,'empty_natural_negatives'):
            a.replay_training(single,dual,groups,heldout,'aug',193)

    def test_both_caps_independently_audit_same_matched_streams(self):
        left=self.fixture.check()
        with tempfile.TemporaryDirectory() as directory:
            right=ModelAuditFixture(directory,cap_value=10).check()
        for key in ('initial_tensor_sha256','strict_negative_qualification_sha256','exposure_sha256','pool_exposure_sha256','negative_exposure_sha256'):
            self.assertEqual(left[key],right[key],key)

    def test_main_unrelated_review_fails_and_preserves_exit1(self):
        review=Path(self.tmp.name)/'unrelated_review.json';review.write_text(json.dumps({'status':'PASS','independent':True,'reviewed_sha256':{}}))
        out=Path(self.tmp.name)/'failed_audit'
        argv=[]
        for name,value in [('root',self.tmp.name),('run',self.tmp.name),('queue',str(review)),('bindings',str(review)),('protocol',str(review)),('source',self.tmp.name),('implementation-review',str(review)),('out',str(out))]:
            argv+=['--'+name,value]
        with self.assertRaisesRegex(ValueError,'implementation_review_required'):a.main(argv)
        self.assertEqual((out/'EXIT_CODE').read_text(),'1\n')
        self.assertTrue((out/'failure.json').is_file())


class DriverAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.f=SupervisorFixture(self.root)
        with self.f.child.open('a') as stream:stream.write('(out/"EXIT_CODE").write_text("0\\n")\n')
        self.f.write();result=self.f.run(wrapper=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def tearDown(self):
        self.tmp.cleanup()

    def check(self):
        inputs=a.Inputs();inputs.pin(self.f.queue)
        return a.verify_driver(self.root,self.f.queue,self.f.q,inputs)

    def test_eight_actual_tiny_children_provenance(self):
        self.assertEqual(self.check()['jobs'],8)

    def test_bool_internal_exit_is_not_real_zero(self):
        p=Path(self.f.q['jobs'][0]['access_log']);value=json.loads(p.read_text());value['internal_exit']=False;p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'strict_integer_exit0'):
            self.check()

    def test_wrong_child_PID_or_argv_or_timestamp_refused(self):
        p=Path(self.f.q['jobs'][0]['access_log']);original=json.loads(p.read_text())
        for field,value in [('pid',123),('argv',['unrelated']),('finished_utc','2020-01-01T00:00:00Z')]:
            modified=dict(original);modified[field]=value;p.write_text(json.dumps(modified))
            with self.assertRaises(ValueError):self.check()

    def test_missing_source_or_after_input_closure_refused(self):
        p=self.f.out/'input_fingerprints.after.json';p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'input_begin_end'):
            self.check()


if __name__=='__main__':
    unittest.main()

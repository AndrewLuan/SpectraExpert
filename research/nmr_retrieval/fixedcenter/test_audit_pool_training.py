"""功能概述：用合成tensor和身份行验证audit_pool_training的失败关闭行为。
fixture提供完整64轮预算与epoch0最佳权重，AuditModelTests覆盖继承/冻结/finite/
预算/曝光/源pin等独立破坏；ExposureTests独立按逐draw参考算法核验重放与增强。
所有测试使用临时内存数据，不读真实训练或测试产物，不执行训练与外部API。
"""
import copy
import hashlib
import json
import unittest

import numpy as np
import torch

import audit_pool_training as a


def fixture():
    args = dict(a.BUDGET, stage='fit', out='/formal', train='/train.pt',
                dual_train='/dual.pt', initial_checkpoint='/old.pt',
                training_pool='original',loss='pairwise16',seed=193,augment=False)
    pins={'/train.pt':'train','/dual.pt':'dual','/old.pt':'old','/source.py':'source'}
    protocol=dict(args=args,inputs_sha256=pins)
    old=dict(args=dict(seed=193,augment=False),state_dict={
        'network.weight':torch.tensor([[1.,2.]]),
        'observation_model.width_raw':torch.tensor([3.])},
        scale=torch.tensor([2.,3.]),alpha=.25,observation=True,
        initial_sigmas=[.1],source_inputs_sha256={'science':'fixed'},
        cache_protocol={'recall':'single'},train_sha256='train')
    exposure={e:dict(exposure='e'+str(e),original='o'+str(e),mixed='m'+str(e),
                         draws=e*8192,dual_draws=e*4096) for e in a.CHECK_EPOCHS}
    checkpoints={}
    for label,epoch in (('initial',0),('best',0),('last',64)):
        checkpoints[label]=dict(schema=a.SCHEMA,observation=True,args=copy.deepcopy(args),
            state_dict=copy.deepcopy(old['state_dict']),scale=old['scale'].clone(),initial_sigmas=[.1],
            alpha=.25,epoch=epoch,source_inputs_sha256={'science':'fixed'},
            ranker_source_sha256={'/source.py':'source'},train_sha256='train',
            dual_train_sha256='dual',initialization_sha256='old',
            exposure_sha256=exposure[epoch]['exposure'],pool_exposure_sha256=exposure[epoch]['original'])
    checkpoints['last']['state_dict']['network.weight'] += 1.
    best=dict(epoch=0,alpha=.25,MRR=.4)
    summary=dict(best=best,epochs=64,samples_per_epoch=8192,training_pool='original',
                 loss='pairwise16',input_sha256=pins,exposure_sha256='e64',
                 pool_exposure_sha256='o64',observation_parameter_max_changes={'width_raw':0.})
    history=[dict(epoch=e,best=best,training_loss=0.1) for e in a.CHECK_EPOCHS]
    expected=dict(training_pool='original',loss='pairwise16',seed=193,augment=False)
    return summary,protocol,old,checkpoints,history,expected,exposure


class AuditModelTests(unittest.TestCase):
    def setUp(self):
        self.v=fixture()

    def check(self):
        return a.audit_model(*self.v)

    def test_epoch_zero_best_and_full_budget_pass(self):
        result=self.check()
        self.assertEqual(result['best_epoch'],0)
        self.assertEqual(result['draws'],524288)
        self.assertEqual(result['optimizer_steps'],16384)

    def test_initial_network_change_rejected(self):
        self.v[3]['initial']['state_dict']['network.weight'] += 1
        with self.assertRaisesRegex(ValueError,'initial_tensors_changed'): self.check()

    def test_tensor_dtype_change_rejected(self):
        self.v[3]['initial']['state_dict']['network.weight']=self.v[3]['initial']['state_dict']['network.weight'].double()
        with self.assertRaisesRegex(ValueError,'initial_tensors_changed'): self.check()

    def test_initial_alpha_change_rejected(self):
        self.v[3]['initial']['alpha']=.5
        with self.assertRaisesRegex(ValueError,'initial_scale_alpha_changed'): self.check()

    def test_initial_scale_change_rejected(self):
        self.v[3]['initial']['scale'] += .1
        with self.assertRaisesRegex(ValueError,'initial_scale_alpha_changed'): self.check()

    def test_all_phases_observation_checked(self):
        for phase in ('initial','best','last'):
            with self.subTest(phase=phase):
                self.v=fixture()
                self.v[3][phase]['state_dict']['observation_model.width_raw'] += .1
                with self.assertRaises(ValueError): self.check()

    def test_nonfinite_state_rejected(self):
        self.v[3]['last']['state_dict']['network.weight'][0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite_checkpoint'): self.check()

    def test_nonfinite_history_rejected(self):
        self.v[4][-1]['training_loss']=float('inf')
        with self.assertRaisesRegex(ValueError,'nonfinite_summary_or_history'): self.check()

    def test_nonfinite_summary_rejected(self):
        self.v[0]['duration']=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite_summary_or_history'): self.check()

    def test_incomplete_history_rejected(self):
        del self.v[4][-1]
        with self.assertRaisesRegex(ValueError,'history_incomplete'): self.check()

    def test_short_budget_rejected(self):
        self.v[1]['args']['epochs']=32
        with self.assertRaisesRegex(ValueError,'protocol_budget'): self.check()

    def test_last_not_64_rejected(self):
        self.v[3]['last']['epoch']=56
        with self.assertRaises(ValueError): self.check()

    def test_exposure_mismatch_rejected(self):
        self.v[3]['last']['exposure_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'checkpoint_exposure'): self.check()

    def test_pool_exposure_mismatch_rejected(self):
        self.v[3]['last']['pool_exposure_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'checkpoint_pool_exposure'): self.check()

    def test_summary_exposure_mismatch_rejected(self):
        self.v[0]['exposure_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'summary_exposure'): self.check()

    def test_source_pin_mismatch_rejected(self):
        self.v[3]['best']['ranker_source_sha256']['/source.py']='wrong'
        with self.assertRaisesRegex(ValueError,'checkpoint_source_pins'): self.check()

    def test_input_pin_mismatch_rejected(self):
        self.v[3]['best']['initialization_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'checkpoint_input_pin'): self.check()

    def test_best_checkpoint_mismatch_rejected(self):
        self.v[3]['best']['alpha']=.5
        with self.assertRaisesRegex(ValueError,'best_checkpoint_mismatch'): self.check()

    def test_old_seed_mismatch_rejected(self):
        self.v[2]['args']['seed']=211
        with self.assertRaisesRegex(ValueError,'old_initial_role'): self.check()

    def test_declared_observation_changes_rejected(self):
        self.v[0]['observation_parameter_max_changes']['width_raw']=1.
        with self.assertRaisesRegex(ValueError,'observation_declared_changes'): self.check()



    def test_network_capacity_change_rejected(self):
        self.v[3]['last']['state_dict']['network.weight']=torch.ones((2,1))
        with self.assertRaisesRegex(ValueError,'state_tensor_capacity_changed'):self.check()

    def test_last_tensor_dtype_change_rejected(self):
        self.v[3]['last']['state_dict']['network.weight']=torch.ones((1,2),dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,'state_tensor_capacity_changed'):self.check()


class ExposureTests(unittest.TestCase):
    def rows(self):
        return [dict(query_order=i,variant=v,source='synthetic',record_id=str(i),smiles='C',group=str(i))
                for i in range(100) for v in ('observed','frequency_crop')]

    def test_reference_digest_and_half_pool(self):
        rows=self.rows()
        got,ng,ni=a.recompute_exposure(rows,193,True,epochs=8,samples=10)
        groups={}
        for row in rows:
            if int(hashlib.sha256(('ranker_internal_317'+row['group']).encode()).hexdigest()[:12],16)%5:
                groups.setdefault(row['group'],[]).append(row)
        names=sorted(groups); identity=hashlib.sha256(); mixed=hashlib.sha256()
        for epoch in range(1,9):
            rng=np.random.default_rng(193+2000+epoch); vrng=np.random.default_rng(193+4000+epoch)
            ids=rng.integers(0,len(names),size=10); pools=np.arange(10)%2
            np.random.default_rng(193+6000+epoch).shuffle(pools)
            for n,pool in zip(ids,pools):
                candidates=groups[names[int(n)]]
                row=candidates[int(vrng.integers(len(candidates)))]
                identity.update(json.dumps([row[k] for k in a.KEY_FIELDS],separators=(',',':')).encode()+b'\n')
                mixed.update(bytes([int(pool)]))
        self.assertEqual(got[8]['exposure'],identity.hexdigest())
        self.assertEqual(got[8]['mixed'],mixed.hexdigest())
        self.assertEqual(got[8]['dual_draws'],40)
        self.assertEqual(got[8]['draws'],80)
        self.assertEqual(ng,len(groups)); self.assertEqual(ng+ni,100)

    def test_aug_changes_variant_exposure(self):
        plain,_,_=a.recompute_exposure(self.rows(),193,False,epochs=8,samples=10)
        aug,_,_=a.recompute_exposure(self.rows(),193,True,epochs=8,samples=10)
        self.assertNotEqual(plain[8]['exposure'],aug[8]['exposure'])
        self.assertEqual(plain[8]['mixed'],aug[8]['mixed'])

    def test_internal_split_empty_rejected(self):
        with self.assertRaisesRegex(ValueError,'empty_internal'):
            a.recompute_exposure([],193,False,epochs=8,samples=10)


class FilesystemTests(unittest.TestCase):
    def make_tree(self, root):
        from pathlib import Path
        source=root/'science.py';source.write_text('frozen synthetic source\n')
        source_pins={str(source):a.sha(source)}
        jobs=[]
        for seed in (193,211):
            train=root/'old'/str(seed)/'single/train/features.pt'
            dual=root/'old'/str(seed)/'dual/train/features.pt'
            for cache,recall in ((train,'single'),(dual,'dual')):
                cache.parent.mkdir(parents=True)
                torch.save(dict(manifest=dict(cohort='train',recall=recall),
                                rows=ExposureTests().rows()),cache)
                (cache.parent/'summary.json').write_text(json.dumps(dict(feature_cache_sha256=a.sha(cache))))
                (cache.parent/'EXIT_CODE').write_text('0\n')
            for aug in (False,True):
                old_path=root/'old'/str(seed)/('aug' if aug else 'no_aug')/'best.pt'
                old_path.parent.mkdir(parents=True)
                old=fixture()[2];old['args']=dict(seed=seed,augment=aug);old['train_sha256']=a.sha(train)
                torch.save(old,old_path)
                for pool in ('original','mixed'):
                    for loss in ('pairwise16','listwise'):
                        label='aug' if aug else 'no_aug'
                        jobid=f'fit_{pool}_{loss}_s{seed}_{label}'
                        out=root/'formal/rankers'/f'{pool}_{loss}'/f's{seed}'/label
                        out.mkdir(parents=True)
                        summary,protocol,_,checkpoints,history,_,exposure=fixture()
                        args=protocol['args']
                        args.update(out=str(out),train=str(train),dual_train=str(dual),
                                    initial_checkpoint=str(old_path),seed=seed,augment=aug,training_pool=pool,loss=loss)
                        inputs={str(p):a.sha(p) for p in (train,dual,old_path)}
                        pins=dict(inputs,**source_pins)
                        for cache in (train,dual):
                            for name in ('summary.json','EXIT_CODE'):
                                p=cache.parent/name;pins[str(p)]=a.sha(p)
                        protocol['inputs_sha256']=pins
                        summary.update(training_pool=pool,loss=loss,input_sha256=pins,
                                       training_groups=1,internal_groups=1,pool_exposure_sha256=exposure[64][pool])
                        for cp in checkpoints.values():
                            cp.update(args=copy.deepcopy(args),state_dict=copy.deepcopy(old['state_dict']),
                                      ranker_source_sha256=source_pins,train_sha256=inputs[str(train)],
                                      dual_train_sha256=inputs[str(dual)],initialization_sha256=inputs[str(old_path)],
                                      pool_exposure_sha256=exposure[cp['epoch']][pool])
                        argv=['python','-m','nmr_transfer.h_pool_ranker']
                        for name in ('train','dual_train','initial_checkpoint','out','training_pool','loss','seed',
                                     'epochs','samples','batch_size','anchor','lr'):
                            argv += ['--'+name.replace('_','-'),str(args[name])]
                        if aug: argv.append('--augment')
                        job=dict(id=jobid,kind='training',out=str(out),argv=argv,inputs_sha256=inputs)
                        jobs.append(job)
                        receipt=dict(state='complete',external_exit=0,internal_exit=0,finished_utc='synthetic',
                                     argv=argv,output=str(out),requested_inputs_sha256=inputs,
                                     queue_source_sha256=source_pins)
                        run=root/'runs/formal';run.mkdir(parents=True,exist_ok=True)
                        (run/(jobid+'.receipt.json')).write_text(json.dumps(receipt))
                        (out/'summary.json').write_text(json.dumps(summary))
                        (out/'protocol.json').write_text(json.dumps(protocol))
                        (out/'history.jsonl').write_text(''.join(json.dumps(h)+'\n' for h in history))
                        (out/'EXIT_CODE').write_text('0\n')
                        for name,cp in checkpoints.items():torch.save(cp,out/(name+'.pt'))

        # 生产者审计要求48任务全部完成，并把receipt绑定到launcher和driver起止UTC。
        for index in range(32):
            out=root/'formal/validation'/str(index);out.mkdir(parents=True)
            (out/'EXIT_CODE').write_text('0\n')
            jobs.append(dict(id='val_'+str(index),kind='validation',out=str(out),
                             argv=['synthetic_validation',str(index)],inputs_sha256={}))
        for job in jobs:job['cwd']=str(root)
        queue_path=root/'formal_queue.json'
        queue_path.write_text(json.dumps(dict(jobs=jobs,source_inputs_sha256=source_pins)))
        queue_sha={str(queue_path):a.sha(queue_path)}
        bound={**source_pins,**queue_sha}
        run=root/'runs/formal'
        for job in jobs:
            receipt=dict(state='complete',external_exit=0,internal_exit=0,
                         started_utc='2026-10-22T06:00:02+00:00',finished_utc='2026-10-22T06:00:03+00:00',
                         argv=job['argv'],output=job['out'],cwd=job['cwd'],
                         requested_inputs_sha256=job['inputs_sha256'],queue_source_sha256=bound)
            (run/(job['id']+'.receipt.json')).write_text(json.dumps(receipt))
        launch=dict(external_exit=0,internal_exit=0,started_utc='2026-10-22T06:00:00+00:00',
                    finished_utc='2026-10-22T06:00:05+00:00',queue_sha256=queue_sha,
                    bundle_sha256=source_pins,source_sha256=source_pins,child_pid=123,
                    argv=['python','-m','nmr_transfer.h_research_driver','--queue',str(queue_path)])
        driver=dict(pid=123,argv=['driver.py','--queue',str(queue_path)],
                    utc='2026-10-22T06:00:01+00:00',inputs_sha256=bound)
        completion=dict(success=True,inputs_unchanged=True,utc='2026-10-22T06:00:04+00:00',
                        jobs={j['id']:json.loads((run/(j['id']+'.receipt.json')).read_text()) for j in jobs})
        for name,value in (('launch_receipt.json',launch),('driver_start.json',driver),('completion.json',completion)):
            (run/name).write_text(json.dumps(value))
        for name in ('EXIT_CODE','EXTERNAL_EXIT_CODE'):(run/name).write_text('0\n')

        return jobs

    def run_tree(self, mutation=None):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            jobs=self.make_tree(root)
            if mutation:mutation(root,jobs)
            fake=(fixture()[-1],1,1)
            with patch.object(a,'recompute_exposure',return_value=fake):
                return a.audit(root)

    def test_full_16_filesystem_pass_and_freeze_contract(self):
        result=self.run_tree()
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['training_jobs'],16)
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(len(result['models']),16)
        self.assertTrue(result['inputs_unchanged'])
        self.assertTrue(any(p.endswith('/last.pt') for p in result['input_sha256']))

    def test_pending_receipt_rejected(self):
        def mutate(root,jobs):
            p=root/'runs/formal'/(jobs[0]['id']+'.receipt.json')
            receipt=json.loads(p.read_text());receipt['external_exit']=None
            p.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'receipt_completion_binding'):self.run_tree(mutate)

    def test_missing_fit_rejected(self):
        def mutate(root,jobs):
            p=root/'formal_queue.json';queue=json.loads(p.read_text());queue['jobs'].pop()
            p.write_text(json.dumps(queue))
        with self.assertRaisesRegex(ValueError,'launch_queue_binding'):self.run_tree(mutate)

    def test_actual_source_hash_change_rejected(self):
        with self.assertRaisesRegex(ValueError,'input_pin_mismatch'):
            self.run_tree(lambda root,jobs:(root/'science.py').write_text('changed'))



    def mutate_receipt(self, root, jobs, field, value):
        run=root/'runs/formal';jobid=jobs[0]['id'];p=run/(jobid+'.receipt.json')
        receipt=json.loads(p.read_text());receipt[field]=value
        p.write_text(json.dumps(receipt))
        completion=json.loads((run/'completion.json').read_text())
        completion['jobs'][jobid]=receipt
        (run/'completion.json').write_text(json.dumps(completion))

    def test_bool_exit_rejected_even_when_completion_matches(self):
        with self.assertRaisesRegex(ValueError,'job_not_double_exit0'):
            self.run_tree(lambda r,j:self.mutate_receipt(r,j,'internal_exit',False))

    def test_invalid_finished_utc_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid_completion_utc'):
            self.run_tree(lambda r,j:self.mutate_receipt(r,j,'finished_utc','synthetic'))

    def test_naive_utc_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid_completion_utc'):
            self.run_tree(lambda r,j:self.mutate_receipt(r,j,'finished_utc','2026-10-22T06:00:03'))

    def test_receipt_source_unbound_rejected(self):
        with self.assertRaisesRegex(ValueError,'receipt_source_binding'):
            self.run_tree(lambda r,j:self.mutate_receipt(r,j,'queue_source_sha256',{}))

    def test_receipt_time_order_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid_job_time_order'):
            self.run_tree(lambda r,j:self.mutate_receipt(r,j,'finished_utc','2026-10-22T06:00:01+00:00'))

    def test_launch_pid_binding_rejected(self):
        def mutate(root,jobs):
            p=root/'runs/formal/launch_receipt.json';record=json.loads(p.read_text())
            record['child_pid']=124;p.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'driver_launch_identity'):self.run_tree(mutate)


if __name__=='__main__':
    unittest.main()

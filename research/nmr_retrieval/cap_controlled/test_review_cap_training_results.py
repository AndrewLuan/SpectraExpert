"""功能概述：训练结果只读复核器的纯元数据矩阵测试，不运行科学任务。

fixture 手工构造八格/512 history/真实格式的 driver、isolation、audit JSON 证据；
dummy .pt 仅为可哈希字节，绝不导入 torch 或声称完成真实 tensor audit。
这验证后验检查器的字段/闭包/攻击拒绝；真实 tensor 结论只来自受审 CPU audit。
"""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import review_cap_training_results as r


class Fixture:
    def __init__(self,directory):
        self.local=Path(directory).resolve();self.root=Path('/remote/cap_controlled')
        self.files={};self.hashes={};self.hash_only={};self.values={}
        self.qp=self.root/'train_after_import_fix_queue.json';self.bp=self.root/'execution_bindings.json'
        self.pp=self.root/'protocol.json';self.ap=self.root/'audits/training'
        self.ip=self.root/'cap_training_audit_implementation_review.json'
        self.fp=self.root/'zero_budget_failure_independent_review.json'
        self.run=self.root/'runs/train_after_import_fix';self.runname='train_after_import_fix'
        self.construct()

    def put(self,path,value):
        path=str(path);raw=value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True)+'\n').encode()
        local=self.local/'payload'/path.lstrip('/');local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(raw)
        self.files[path]=str(local);self.hashes[path]=r.sha(raw);self.values[path]=copy.deepcopy(value)
        return self.hashes[path]

    def blob(self,path):
        path=str(path);self.hash_only[path]=r.sha(('bound existing scientific bytes:'+path).encode());return self.hash_only[path]

    def construct(self):
        R=self.root
        for name in ('cap_ranker.py','train_cap_ranker.py','isolation_entry.py','supervise_controlled_queue.py',
                     'launch_controlled_stage.py','audit_cap_training.py'):
            self.put(R/name,b'# synthetic source only\n')
        protocol_budget=dict(r.BUDGET,fits=8,samples_per_epoch=8192,single_pool_draws_per_epoch=4096,dual_pool_draws_per_epoch=4096,
            total_draws=4194304,total_optimizer_steps=131072,new_forward_bank_extractions=0,maximum_stage_seconds=7200,
            loss='listwise',training_pool='mixed',checkpoint='final epoch64 only',alpha=1,early_stopping=False,cap_grid_allowed=False)
        proto=dict(budget=protocol_budget,caps=[5,10],seeds=[193,211],final_epoch=64,alpha=1,epoch_offset=0)
        self.put(self.pp,proto)
        contract=Path('/science/cohorts/cohort_contract.json')
        self.put(contract,dict(training_excluded_groups=['external_heldout']))
        self.bindings=dict(source_checkout='/science',cohort_contract=str(contract),heldout_groups=1,
            heldout_groups_sha256=r.sha(b'external_heldout'),roles={},caches={})
        for seed in (193,211):
            self.bindings['roles'][str(seed)]={};self.bindings['caches'][str(seed)]={'train':{}}
            for pool in ('single','dual'):
                path=f'/science/features/s{seed}/{pool}/features.pt'
                self.bindings['caches'][str(seed)]['train'][pool]=dict(path=path,sha256=self.blob(path))
            for role in ('no_aug','aug'):
                d=Path(f'/science/parent/s{seed}/{role}');rowdata=b'{"synthetic_role_row":true}\n'
                rh=self.put(d/'center_rows.jsonl',rowdata)
                center=dict(center33=[float(seed)]*33,O_center3=[.1,.2,.3],row_audit_sha256=rh)
                ch=self.put(d/'centers.json',center)
                self.bindings['roles'][str(seed)][role]=dict(checkpoint=str(d/'best.pt'),checkpoint_sha256=self.blob(d/'best.pt'),
                    centers=str(d/'centers.json'),centers_sha256=ch,center_rows=str(d/'center_rows.jsonl'),center_rows_sha256=rh,
                    center33=center['center33'],O_center3=center['O_center3'],qualification={'candidate_count':114982})
        inputs={**self.hashes,**self.hash_only};inputs.pop(str(R/'audit_cap_training.py'))
        self.bindings['input_sha256']=inputs
        self.put(self.bp,self.bindings)
        sources={p:h for p,h in self.hashes.items() if (Path(p).parent==R and p!=str(R/'audit_cap_training.py'))}
        self.q=dict(root=str(R),phase='train',run_name=self.runname,bindings=str(self.bp),python='/venv/python',
                    maximum_seconds=7200,input_sha256=inputs,source_sha256=sources,
                    review_paths=[str(R/'train_recovery_admission_independent_review.json')],jobs=[])
        self.q['recovery']=dict(prior_run=str(R/'runs/train'),prior_archive_manifest_sha256='a'*64,prior_draws=0,
            prior_optimizer_steps=0,scientific_configuration_changed=False,automatic_retry=False)
        models=[];rolehist={}
        for cap,seed,role in sorted(r.MATRIX):
            d=R/f'rankers/C{cap}/s{seed}/{role}';parent=self.bindings['roles'][str(seed)][role]
            command=[str(R/'train_cap_ranker.py'),'--source','/science','--fixed-module','/science/fixed.py',
                     '--protocol',str(self.pp),'--bindings',str(self.bp),'--cap',str(cap),'--seed',str(seed),'--role',role,
                     '--device','cuda:0','--out',str(d)]
            cli={command[i][2:].replace('-','_'):command[i+1] for i in range(1,len(command),2)}
            cli['cap']=cap;cli['seed']=seed
            reads=[self.bindings['caches'][str(seed)]['train'][pool]['path'] for pool in ('single','dual')]+[parent['checkpoint']]
            job=dict(id=f'fit_C{cap}_s{seed}_{role}',command=command,argv=['/venv/python',str(R/'isolation_entry.py')]+command,
                     cwd='/science',gpu=str(len(self.q['jobs'])),job_out=str(d),access_log=str(R/f'access_logs/{self.runname}/fit_C{cap}_s{seed}_{role}.json'),
                     allowed_reads=reads)
            self.q['jobs'].append(job)
            initial={name:r.canonical([seed,role,name]) for name in ('network.weight','observation.raw_sigmas','fixed_x_center','fixed_o_center','__scale__')}
            frozen=r.canonical({k:v for k,v in initial.items() if not k.startswith('network.')})
            group=next('fit_group_'+str(i) for i in range(20)
                if int(r.hashlib.sha256(('ranker_internal_317fit_group_'+str(i)).encode()).hexdigest()[:12],16)%5!=0)
            records=[]
            for variant in (['observed'] if role=='no_aug' else ['observed','frequency_crop']):
                for pool in (0,1):records.append(dict(query_key=[0,variant,'synthetic','record','C',group],pool=pool,negatives=400,
                    ordered_eligible_candidate_sha256=r.canonical([seed,role,variant,pool,'ids']),
                    excluded_mask_sha256=r.canonical([seed,role,variant,pool,'mask'])))
            qual=r.canonical(records);history=[]
            for epoch in range(1,65):
                item=dict(epoch=epoch,draws=8192,pool_draws={'0':4096,'1':4096},optimizer_steps=256,
                    cumulative_draws=8192*epoch,cumulative_optimizer_steps=256*epoch,mean_training_loss=.5,
                    frozen_tensor_sha256=frozen,seconds=epoch)
                for name in ('exposure_sha256','pool_exposure_sha256','negative_exposure_sha256'):
                    item[name]=r.canonical([seed,role,epoch,name]);item['cumulative_'+name]=r.canonical([seed,role,epoch,'cumulative',name])
                history.append(item)
            ih=self.put(d/'initial.pt',b'dummy initial -- hash only, never a model')
            fh=self.put(d/'final.pt',('dummy final cap'+str(cap)).encode())
            row=dict(cap=cap,seed=seed,role=role,checkpoint=str(d/'final.pt'),checkpoint_sha256=fh,
                centers_sha256=parent['centers_sha256'],center_rows_sha256=parent['center_rows_sha256'],epochs=64,alpha=1,
                draws=524288,optimizer_steps=16384,parent_checkpoint_sha256=parent['checkpoint_sha256'],
                initial_tensor_sha256=initial,strict_negative_qualification_sha256=qual)
            for name in ('exposure_sha256','pool_exposure_sha256','negative_exposure_sha256'):row[name]=history[-1]['cumulative_'+name]
            pins=copy.deepcopy(inputs)
            summary=dict(schema=r.MODEL_SCHEMA,delta_cap=cap,seed=seed,role=role,args=dict(cli,augment=role=='aug',training_pool='mixed',loss='listwise',**r.BUDGET),
                budget=r.BUDGET,epoch=64,alpha=1,draws=524288,optimizer_steps=16384,negative_mask_policy=r.MASK,
                optimizer_reset=True,continuation_epoch_offset=0,selection='fixed_final_epoch64_alpha1_no_selection',
                centers_frozen=True,scale_frozen=True,observation_frozen=True,initial_tensors_inherited_exact=True,
                parent_checkpoint=parent['checkpoint'],parent_checkpoint_sha256=parent['checkpoint_sha256'],
                train_sha256=self.bindings['caches'][str(seed)]['train']['single']['sha256'],
                dual_train_sha256=self.bindings['caches'][str(seed)]['train']['dual']['sha256'],
                initial_tensor_sha256=initial,frozen_tensor_sha256=frozen,initial_checkpoint_sha256=ih,final_checkpoint_sha256=fh,
                source_inputs_start_sha256=pins,source_inputs_end_sha256=pins,input_sha256=pins,epoch_records=history,
                strict_negative_qualification_sha256=qual,training_rows=len(records)//2,training_groups=1,
                **{name:row[name] for name in ('exposure_sha256','pool_exposure_sha256','negative_exposure_sha256')})
            self.put(d/'summary.json',summary)
            self.put(d/'history.jsonl',b''.join((json.dumps(v)+'\n').encode() for v in history))
            self.put(d/'initial_tensor_sha256.json',initial)
            self.put(d/'negative_qualification.json',dict(policy=r.MASK,records=records,sha256=qual,qualification=parent['qualification']))
            self.put(d/'protocol.json',dict(args=cli,protocol=proto,input_sha256=pins));self.put(d/'EXIT_CODE',b'0\n')
            self.put(d/'centers.json',self.values[parent['centers']]);self.put(d/'center_rows.jsonl',self.values[parent['center_rows']])
            models.append(row);rolehist[cap,seed,role]=history
        self.put(self.qp,self.q);qh=self.hashes[str(self.qp)]
        self.put(R/'train_recovery_admission_independent_review.json',dict(status='PASS',independent=True,admission_phase='train',
            reviewed_sha256={**sources,str(self.qp):qh}))
        small={**sources,str(self.qp):qh,str(R/'train_recovery_admission_independent_review.json'):self.hashes[str(R/'train_recovery_admission_independent_review.json')]}
        argv=['/venv/python','-B',str(R/'supervise_controlled_queue.py'),'--queue',str(self.qp),'--out',str(self.run),'--execute']
        start=dict(argv=argv[2:],queue_sha256=qh,pid=100,parent_pid=99,phase='train',run_name=self.runname,
                   source_sha256=sources,input_sha256=inputs,started_utc='2026-10-24T17:20:01+00:00')
        receipts=[]
        for index,job in enumerate(self.q['jobs']):
            started=dict(id=job['id'],pid=101+index,parent_pid=100,argv=job['argv'],cwd=job['cwd'],gpu=job['gpu'],job_out=job['job_out'],
                access_log=job['access_log'],queue_sha256=qh,requested_inputs_sha256={p:inputs[p] for p in job['allowed_reads']},
                started_utc='2026-10-24T17:20:02+00:00')
            receipt=dict(started,internal_exit=0,external_exit=0,finished_utc='2026-10-24T17:25:02+00:00')
            access=dict(pid=101+index,parent_pid=100,phase='train',argv=job['command'],allowed_reads=sorted(job['allowed_reads']),
                rejected_paths=[],error=None,internal_exit=0,recorded_reads={p:1 for p in job['allowed_reads']},
                started_utc='2026-10-24T17:20:03+00:00',finished_utc='2026-10-24T17:25:01+00:00')
            self.put(self.run/'jobs'/(job['id']+'.started.json'),started)
            self.put(self.run/'jobs'/(job['id']+'.receipt.json'),receipt);self.put(job['access_log'],access);receipts.append(receipt)
        done=dict(queue_sha256=qh,pid=100,phase='train',run_name=self.runname,status='COMPLETE',inputs_unchanged=True,error=None,
                  external_exit=0,finished_utc='2026-10-24T17:25:03+00:00',jobs=receipts)
        launch=dict(argv=argv,queue_sha256=qh,child_pid=100,wrapper_pid=99,phase='train',run_name=self.runname,internal_exit=0,external_exit=0,
                    started_utc='2026-10-24T17:20:00+00:00',finished_utc='2026-10-24T17:25:04+00:00')
        for name,value in (('driver_start.json',start),('completion.json',done),('launch_receipt.json',launch),
                           ('status.json',dict(phase='COMPLETE',completed=8,running=[],pending=0,error=None)),
                           ('input_fingerprints.before.json',inputs),('input_fingerprints.after.json',inputs),
                           ('operator_fingerprints.before.json',small),('operator_fingerprints.after.json',small)):
            self.put(self.run/name,value)
        self.put(self.run/'EXIT_CODE',b'0\n');self.put(self.run/'EXTERNAL_EXIT_CODE',b'0\n')
        self.put(R/(self.runname+'.operator_claim.json'),dict(queue_sha256=qh,pid=100))
        self.put(R/(self.runname+'.launch_claim.json'),dict(queue_sha256=qh,wrapper_pid=99))
        self.put(self.fp,dict(status='PASS',independent=True,training_completed=False,draws=0,optimizer_steps=0,history_rows=0,
            input_pins_unchanged=len(inputs),archive_manifest_sha256='a'*64))
        self.put(self.ip,dict(status='PASS',independent=True,scope='cap10_training_audit_implementation',
            reviewed_sha256={str(R/'audit_cap_training.py'):self.hashes[str(R/'audit_cap_training.py')]}))
        union={**self.hashes,**self.hash_only}
        audit=dict(schema=r.AUDIT_SCHEMA,status='PASS',independent=True,inputs_unchanged=True,checks={k:True for k in r.CHECKS},
            queue_sha256=qh,bindings_sha256=self.hashes[str(self.bp)],protocol_sha256=self.hashes[str(self.pp)],training_run=str(self.run),
            source_sha256={**sources,str(R/'audit_cap_training.py'):self.hashes[str(R/'audit_cap_training.py')]},
            input_sha256=union,input_end_sha256=union,models=models,test_data_opened=False,new_model_calls=0,new_training_jobs=0,
            execution=dict(wrapper_pid=99,driver_pid=100,jobs=8,queue_sha256=qh,started_utc=start['started_utc'],finished_utc=done['finished_utc']))
        self.put(self.ap/'summary.json',audit);self.put(self.ap/'EXIT_CODE',b'0\n')
        audit_argv=['/venv/python','-B',str(R/'audit_cap_training.py'),'--root',str(R),'--run',str(self.run),
            '--queue',str(self.qp),'--bindings',str(self.bp),'--protocol',str(self.pp),'--source','/science',
            '--implementation-review',str(self.ip),'--out',str(self.ap)]
        self.put(R/'actual_training_audit_execution_receipt.json',dict(remote_argv=audit_argv,argv=audit_argv,
            actual_inner_exit=0,actual_external_exit=0,actual_ssh_exit=0,
            source_sha256={str(R/'audit_cap_training.py'):self.hashes[str(R/'audit_cap_training.py')]}))

    def manifest(self):
        union={**self.hashes,**self.hash_only}
        value=dict(files=self.files,sha256=self.hashes,hash_only_inputs=self.hash_only,
                   remote_before_sha256=union,remote_after_sha256=union)
        self.map=self.local/'manifest.json';self.map.write_text(json.dumps(value))
        return self.map

    def rebind_fixture_audit_inputs(self,*paths):
        """只在合成攻击中同步声明，确保拒绝来自目标行为而非陈旧摘要。"""
        value=copy.deepcopy(self.values[str(self.ap/'summary.json')])
        for path in paths:
            value['input_sha256'][str(path)]=self.hashes[str(path)]
            value['input_end_sha256'][str(path)]=self.hashes[str(path)]
        self.put(self.ap/'summary.json',value)

    def check(self):
        bundle=r.Bundle(self.manifest(),self.root)
        result=r.verify(bundle,self.root,self.qp,self.bp,self.pp,self.ap,self.ip,self.fp);bundle.finish()
        return result


class ResultTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.f=Fixture(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_full_eight_matrix_512_history_metadata(self):
        value=self.f.check();self.assertEqual(value['models'],8);self.assertEqual(value['total_history_rows'],512)
        self.assertEqual(value['total_draws'],4194304);self.assertIn('no second tensor read',value['tensor_verification'])
    def test_wrong_actual_run_is_not_old_failure(self):
        value=copy.deepcopy(self.f.values[str(self.f.ap/'summary.json')]);value['training_run']=str(self.f.root/'runs/train')
        self.f.put(self.f.ap/'summary.json',value)
        with self.assertRaisesRegex(ValueError,'exact_successful_run'):self.f.check()
    def test_CPU_audit_outer_actual_exit_must_be_real_zero(self):
        path=self.f.root/'actual_training_audit_execution_receipt.json'
        value=copy.deepcopy(self.f.values[str(path)]);value['actual_ssh_exit']=False;self.f.put(path,value)
        with self.assertRaisesRegex(ValueError,'strict_int_exit0'):self.f.check()
    def test_bool_false_is_not_driver_exit_zero(self):
        p=self.f.run/'launch_receipt.json';v=copy.deepcopy(self.f.values[str(p)]);v['internal_exit']=False;self.f.put(p,v)
        self.f.rebind_fixture_audit_inputs(p)
        with self.assertRaisesRegex(ValueError,'strict_int_exit0'):self.f.check()
    def test_history_epoch64_missing_rejected(self):
        p=self.f.root/'rankers/C5/s193/aug/history.jsonl';raw=self.f.values[str(p)]
        self.f.put(p,b'\n'.join(raw.splitlines()[:-1])+b'\n')
        self.f.rebind_fixture_audit_inputs(p)
        with self.assertRaisesRegex(ValueError,'actual64_history'):self.f.check()
    def test_missing_actual_initial_or_final_hash_closure(self):
        self.f.manifest();value=json.loads(self.f.map.read_text())
        p=str(self.f.root/'rankers/C5/s193/aug/final.pt');value['remote_before_sha256'].pop(p);value['remote_after_sha256'].pop(p)
        self.f.map.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'copied_matches_remote'):r.Bundle(self.f.map,self.f.root)
    def test_tensor_can_only_be_hashed_at_exact_sixteen_registered_outputs(self):
        p=self.f.root/'rankers/unregistered.pt';self.f.put(p,b'not data')
        with self.assertRaisesRegex(ValueError,'forbidden_review_input'):r.Bundle(self.f.manifest(),self.f.root)
    def test_explicit_json_supplement_keeps_base_map_bytes_and_matches_actual_remote_SHA(self):
        base=self.f.manifest();value=json.loads(base.read_text());remote=self.f.bindings['cohort_contract']
        local=value['files'].pop(remote);digest=value['sha256'].pop(remote);value['hash_only_inputs'][remote]=digest
        base.write_text(json.dumps(value));before=r.sha(base.read_bytes())
        extra=self.f.local/'supplement.json';extra.write_text(json.dumps(dict(
            schema='cap10_training_review_metadata_supplement_20261024',base_manifest_sha256=before,
            files={remote:local},sha256={remote:digest})))
        bundle=r.Bundle(base,self.f.root,extra)
        r.verify(bundle,self.f.root,self.f.qp,self.f.bp,self.f.pp,self.f.ap,self.f.ip,self.f.fp);bundle.finish()
        self.assertEqual(bundle.supplement_copied_files,1);self.assertEqual(r.sha(base.read_bytes()),before)
    def test_prior_import_failure_cannot_be_reported_as_completed_training(self):
        value=copy.deepcopy(self.f.values[str(self.f.fp)]);value['training_completed']=True;self.f.put(self.f.fp,value)
        self.f.rebind_fixture_audit_inputs(self.f.fp)
        with self.assertRaisesRegex(ValueError,'first_import_failure'):self.f.check()
    def test_finished_copied_input_symlink_is_rechecked_before_read(self):
        bundle=r.Bundle(self.f.manifest(),self.f.root)
        remote=str(self.f.pp);local=Path(self.f.files[remote]);target=self.f.local/'sealed_forbidden/summary.json'
        target.parent.mkdir();target.write_text('{}');local.unlink();local.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'forbidden_review_input'):bundle.finish()
    def test_main_keeps_actual_zero_and_no_selection(self):
        out=self.f.local/'review'
        argv=['--artifact-map',str(self.f.manifest()),'--root',str(self.f.root),'--queue',str(self.f.qp),'--bindings',str(self.f.bp),
              '--protocol',str(self.f.pp),'--audit',str(self.f.ap),'--implementation-review',str(self.f.ip),'--failure-review',str(self.f.fp),
              '--audit-execution-sha256',self.f.hashes[str(self.f.root/'actual_training_audit_execution_receipt.json')],'--out',str(out)]
        with contextlib.redirect_stdout(io.StringIO()):r.main(argv)
        value=json.loads((out/'review.json').read_text());self.assertEqual(value['status'],'PASS')
        self.assertFalse(value['selection_performed']);self.assertFalse(value['promotion']);self.assertEqual((out/'EXIT_CODE').read_text(),'0\n')


if __name__=='__main__':unittest.main()

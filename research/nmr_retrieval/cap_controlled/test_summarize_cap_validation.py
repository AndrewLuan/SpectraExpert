"""功能概述：完整16job synthetic operator receipts→真实prediction reader→648表→选择入口。
MatrixFixture使用完整512候选/cost/trace字段，但把observed320/1024计数常量patch为2/3，
保留五source、三variant、两seed、两cap、两pool及所有回执。纯合成metadata无真实谱。
攻击覆盖bool退出、缺任务、source/argv/权重、成本伪rank与audit缺项；保留空source层null。
"""
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import cap_evaluation_common as c
import summarize_cap_validation as s

T0='2026-10-24T17:00:00+00:00';T1='2026-10-24T17:00:01+00:00';T2='2026-10-24T17:00:02+00:00';T3='2026-10-24T17:00:03+00:00';T4='2026-10-24T17:00:04+00:00'
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return str(p)
def plain(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x);return str(p)
class MatrixFixture:
    def __init__(self,root,fixed_pairs=None,run_name='validation'):
        self.root=Path(root).resolve();self.run=self.root/'runs'/run_name;(self.run/'jobs').mkdir(parents=True)
        self.source=plain(self.root/'dummy_source.py','# explicit synthetic source\n')
        self.sourcepins={str(Path(__file__).with_name('evaluate_cap_ranker.py').resolve()):c.sha(Path(__file__).with_name('evaluate_cap_ranker.py')),str(Path(__file__).with_name('train_cap_ranker.py').resolve()):c.sha(Path(__file__).with_name('train_cap_ranker.py')),str(Path(s.__file__).resolve()):c.sha(s.__file__),str(Path(c.__file__).resolve()):c.sha(c.__file__),self.source:c.sha(self.source)}
        self.cap_module=dict(path=self.source,sha256=c.sha(self.source))
        self.roles={};self.caches={}
        for cap,seed in itertools.product(c.CAPS,c.SEEDS):
            roles={}
            for role in ('no_aug','aug'):
                folder=self.root/'weights'/str(cap)/str(seed)/role
                ck=plain(folder/'final.pt',f'synthetic {cap} {seed} {role}');centers=put(folder/'centers.json',{'synthetic':True})
                rows=plain(folder/'center_rows.jsonl','{"synthetic":true}\n');producer=put(folder/'summary.json',{'synthetic':True})
                plain(folder/'EXIT_CODE','0\n')
                roles[role]=dict(checkpoint=ck,checkpoint_sha256=c.sha(ck),centers=centers,centers_sha256=c.sha(centers),
                  center_rows=rows,center_rows_sha256=c.sha(rows),producer_summary=producer,producer_summary_sha256=c.sha(producer))
            self.roles.setdefault(str(cap),{})[str(seed)]=roles
        for seed,cohort,pool in itertools.product(c.SEEDS,('original_val','tune'),c.POOLS):
            folder=self.root/'caches'/str(seed)/cohort/pool;path=plain(folder/'features.pt','synthetic cache file')
            count=2 if cohort=='original_val' else 3
            manifest=dict(schema='nmr_research_20261022',candidate_count=114982,pool_size=512,candidate_order_sha256=c.ORDER_SHA,query_sha256=cohort,checkpoint_sha256='f'*64,source_inputs_sha256={},identity_content_sha256='i'*64,variant_mask_content_sha256='v'*64,cohort=cohort,recall=pool,records=count+2,base_queries=count,variants={'observed':count,'frequency_crop':1,'peak_dropout':1})
            producer={**manifest,'feature_cache_sha256':c.sha(path)};pp=put(folder/'summary.json',producer);ep=plain(folder/'EXIT_CODE','0\n')
            self.caches.setdefault(str(seed),{}).setdefault(cohort,{})[pool]=dict(path=path,sha256=c.sha(path),manifest=producer,producer=dict(path=pp,sha256=c.sha(pp),exit_path=ep))
        self.fixed_pairs=None
        if fixed_pairs is not None:
            self.fixed_pairs=put(self.root/'fixed_train_pairs.json',dict(schema='cap10_fixed_train_pairs_20261024',records=fixed_pairs))
            for seed in c.SEEDS:
                path=plain(self.root/'caches'/str(seed)/'train/dual/features.pt','synthetic fixed train cache')
                manifest=dict(schema='nmr_research_20261022',candidate_count=114982,pool_size=512,candidate_order_sha256=c.ORDER_SHA,
                    query_sha256='train',checkpoint_sha256='f'*64,source_inputs_sha256={},identity_content_sha256='i'*64,
                    variant_mask_content_sha256='v'*64,cohort='train',recall='dual',records=200,base_queries=200,variants={'observed':200})
                producer={**manifest,'feature_cache_sha256':c.sha(path)};pp=put(Path(path).parent/'summary.json',producer);ep=plain(Path(path).parent/'EXIT_CODE','0\n')
                self.caches[str(seed)]['train']={'dual':dict(path=path,sha256=c.sha(path),manifest=producer,producer=dict(path=pp,sha256=c.sha(pp),exit_path=ep))}
        self.baseline=put(self.root/'baseline.json',dict(config=c.BASELINE,test_read=False,signals={k:dict(value=v) for k,v in c.BASELINE_SIGNALS.items()}))
        protocol=dict(schema='cap10_controlled_protocol_20261024',seeds=[193,211],caps=[5,10],pool_size=512,candidate_library_size=114982,
          promotion=False,test_allowed=False,final_epoch=64,alpha=1.,epoch_offset=0,only_eligible_new_config='cap10_dual',control_config='cap5_dual',preselected=c.BASELINE,
          mechanism=dict(saturation_threshold=.95,selection_input=False,fixed_train_pairs={'sha256':c.sha(self.fixed_pairs)} if self.fixed_pairs else {}),validation=dict(baseline_signals=c.BASELINE_SIGNALS))
        self.protocol=put(self.root/'protocol.json',protocol)
        self.trainingbinding=put(self.root/'execution_bindings.json',{'synthetic_training':True});self.trainingqueue=put(self.root/'train_queue.json',{'synthetic_training':True})
        self.bindings=put(self.root/'validation_bindings.json',dict(schema='cap10_execution_bindings_20261024',source_checkout=str(self.root),
          training_run=str(self.root/'runs/train_after_import_fix'),training_bindings=dict(path=self.trainingbinding,sha256=c.sha(self.trainingbinding)),training_queue=dict(path=self.trainingqueue,sha256=c.sha(self.trainingqueue)),source_sha256=self.sourcepins,cap_module=self.cap_module,caches=self.caches,roles=self.roles,baseline_reference=dict(path=self.baseline,sha256=c.sha(self.baseline))))
        if self.fixed_pairs is not None:
            binding=json.loads(Path(self.bindings).read_text());binding['fixed_train_pairs']=dict(path=self.fixed_pairs,sha256=c.sha(self.fixed_pairs));put(self.bindings,binding)
        self.jobs=[];self.receipts=[];self.inputpins={**self.sourcepins,self.protocol:c.sha(self.protocol),self.bindings:c.sha(self.bindings)}
        for cap,seed,pool,cohort in itertools.product(c.CAPS,c.SEEDS,c.POOLS,('original_val','tune')):
            self.make_eval(cap,seed,pool,cohort,len(self.jobs))
        self.inputpins.pop(self.bindings)
        if self.fixed_pairs is not None:
            self.inputpins[self.fixed_pairs]=c.sha(self.fixed_pairs)
            for seed in c.SEEDS:
                info=self.caches[str(seed)]['train']['dual'];self.inputpins[info['path']]=info['sha256']
        binding=json.loads(Path(self.bindings).read_text());binding['input_sha256']=self.inputpins;put(self.bindings,binding)
        for job in self.jobs:
            for name in ('summary.json','protocol.json'):
                pp=Path(job['job_out'])/name;v=json.loads(pp.read_text());v['input_sha256'][self.bindings]=c.sha(self.bindings);put(pp,v)
            job['allowed_reads']=[x for x in job['allowed_reads'] if x!=self.bindings]
        self.stage_sources={**self.sourcepins,self.bindings:c.sha(self.bindings),self.protocol:c.sha(self.protocol)}
        review=str(self.root/'validation_admission.json')
        self.queue=put(self.root/'validation_queue.json',dict(schema='cap10_controlled_queue_20261024',phase='validation',run_name=run_name,python=sys.executable,local_root=str(self.root),bindings=self.bindings,review_paths=[review],source_sha256=self.stage_sources,input_sha256=self.inputpins,jobs=self.jobs))
        qsha=c.sha(self.queue)
        put(review,dict(status='PASS',independent=True,admission_phase='validation',reviewed_sha256={**self.stage_sources,self.queue:qsha}))
        self.wrapper_argv=[sys.executable,'-B',str(self.root/'supervise_controlled_queue.py'),'--queue',self.queue,'--out',str(self.run),'--execute']
        for job,rec in zip(self.jobs,self.receipts):
            rec.update(queue_sha256=qsha,requested_inputs_sha256={p:self.inputpins[p] for p in job['allowed_reads']})
            put(self.run/'jobs'/(job['id']+'.receipt.json'),rec)
        put(self.run/'driver_start.json',dict(pid=101,parent_pid=100,argv=self.wrapper_argv[2:],started_utc=T0,queue_sha256=qsha,source_sha256=self.stage_sources,input_sha256=self.inputpins))
        put(self.run/'completion.json',dict(status='COMPLETE',pid=101,started_utc=T0,finished_utc=T3,jobs=self.receipts,queue_sha256=qsha,external_exit=0,inputs_unchanged=True))
        put(self.run/'launch_receipt.json',dict(wrapper_pid=100,child_pid=101,argv=self.wrapper_argv,queue_sha256=qsha,source_sha256=self.stage_sources,started_utc=T0,finished_utc=T4,internal_exit=0,external_exit=0))
        plain(self.run/'EXIT_CODE','0\n');plain(self.run/'EXTERNAL_EXIT_CODE','0\n')
        put(self.run/'input_fingerprints.before.json',self.inputpins);put(self.run/'input_fingerprints.after.json',self.inputpins)
        small={**self.stage_sources,self.queue:qsha,review:c.sha(review)}
        put(self.run/'operator_fingerprints.before.json',small);put(self.run/'operator_fingerprints.after.json',small)
        models=[]
        for cap,seed,role in itertools.product(c.CAPS,c.SEEDS,('no_aug','aug')):
            ref=self.roles[str(cap)][str(seed)][role]
            models.append(dict(cap=cap,seed=seed,role=role,checkpoint=ref['checkpoint'],checkpoint_sha256=ref['checkpoint_sha256'],
              parent_checkpoint_sha256='a'*64,strict_negative_qualification_sha256='b'*64,exposure_sha256='c'*64,pool_exposure_sha256='d'*64,negative_exposure_sha256='e'*64,centers_sha256=ref['centers_sha256'],center_rows_sha256=ref['center_rows_sha256'],epochs=64,alpha=1.,draws=524288,optimizer_steps=16384))
        checks={k:True for k in ('same_initial_tensors','frozen_observation','frozen_centers','frozen_scale','matched_exposure','matched_pool_exposure','matched_negative_exposure','strict_negative_qualification','finite','full_budget','real_driver_exit','source_input_unchanged')}
        self.audit=put(self.root/'training_audit/summary.json',dict(schema='cap10_training_audit_20261024',status='PASS',independent=True,
          training_run=str(self.root/'runs/train_after_import_fix'),protocol_sha256=c.sha(self.protocol),bindings_sha256=c.sha(self.trainingbinding),queue_sha256=c.sha(self.trainingqueue),inputs_unchanged=True,models=models,checks=checks,input_sha256=self.sourcepins,source_sha256=self.sourcepins,
          negative_mask_policy='external_heldout_OR_ranker_internal_OR_same_query_group; target_loss_only'))
        plain(Path(self.audit).parent/'EXIT_CODE','0\n')
        self.register_audit()
        self.out=self.root/'selection'
    def register_audit(self):
        trainrun=self.root/'runs/train_after_import_fix';(trainrun/'jobs').mkdir(parents=True)
        sources={**self.sourcepins,self.protocol:c.sha(self.protocol),self.trainingbinding:c.sha(self.trainingbinding)}
        jobs=[];records=[]
        for cap,seed,role in itertools.product(c.CAPS,c.SEEDS,('no_aug','aug')):
            ref=self.roles[str(cap)][str(seed)][role];jid=f'fit{len(jobs)}'
            command=[str(Path(__file__).with_name('train_cap_ranker.py')),'--cap',str(cap),'--seed',str(seed),'--role',role]
            access=str(self.root/'access_logs/train_after_import_fix'/(jid+'.json'))
            argv=[sys.executable,'-B',self.source,'--']+command;pid=400+len(jobs)
            job=dict(id=jid,command=command,argv=argv,cwd=str(self.root),gpu=None,job_out=str(Path(ref['checkpoint']).parent),access_log=access,allowed_reads=list(sources))
            rec=dict(id=jid,argv=argv,cwd=str(self.root),gpu=None,pid=pid,parent_pid=91,started_utc=T1,finished_utc=T2,
                internal_exit=0,external_exit=0,job_out=job['job_out'],access_log=access)
            put(access,dict(pid=pid,parent_pid=91,argv=command,started_utc=T1,finished_utc=T2,internal_exit=0,rejected_paths=[]))
            jobs.append(job);records.append(rec)
        review=str(self.root/'train_admission.json')
        put(self.trainingqueue,dict(schema='cap10_controlled_queue_20261024',phase='train',run_name='train_after_import_fix',
            python=sys.executable,local_root=str(self.root),bindings=self.trainingbinding,review_paths=[review],
            source_sha256=sources,input_sha256=sources,jobs=jobs))
        qsha=c.sha(self.trainingqueue)
        put(review,dict(status='PASS',independent=True,admission_phase='train',reviewed_sha256={**sources,self.trainingqueue:qsha}))
        argv=[sys.executable,'-B',str(self.root/'supervise_controlled_queue.py'),'--queue',self.trainingqueue,'--out',str(trainrun),'--execute']
        for job,rec in zip(jobs,records):
            rec.update(queue_sha256=qsha,requested_inputs_sha256=sources)
            put(trainrun/'jobs'/(job['id']+'.receipt.json'),rec)
        put(trainrun/'driver_start.json',dict(pid=91,parent_pid=90,argv=argv[2:],started_utc=T0,queue_sha256=qsha,source_sha256=sources,input_sha256=sources))
        put(trainrun/'completion.json',dict(status='COMPLETE',pid=91,started_utc=T0,finished_utc=T3,jobs=records,queue_sha256=qsha,external_exit=0,inputs_unchanged=True))
        put(trainrun/'launch_receipt.json',dict(wrapper_pid=90,child_pid=91,argv=argv,queue_sha256=qsha,source_sha256=sources,started_utc=T0,finished_utc=T4,internal_exit=0,external_exit=0))
        plain(trainrun/'EXIT_CODE','0\n');plain(trainrun/'EXTERNAL_EXIT_CODE','0\n')
        for when in ('before','after'):
            put(trainrun/('input_fingerprints.'+when+'.json'),sources)
            put(trainrun/('operator_fingerprints.'+when+'.json'),{**sources,self.trainingqueue:qsha,review:c.sha(review)})
        audit=json.loads(Path(self.audit).read_text())
        evidence=list(trainrun.glob('*.json'))+list((trainrun/'jobs').glob('*.receipt.json'))
        evidence += [Path(j['access_log']) for j in jobs]+[trainrun/'EXIT_CODE',trainrun/'EXTERNAL_EXIT_CODE',Path(review),Path(self.trainingqueue)]
        audit['input_sha256']={**sources,**{str(p):c.sha(p) for p in evidence}}
        audit['queue_sha256']=qsha;put(self.audit,audit)
        self.auditreview=put(self.root/'audit_implementation_review.json',dict(status='PASS',independent=True,reviewer='synthetic_independent_reviewer',
           scope='synthetic_audit_implementation',reviewed_sha256=audit['source_sha256']))
        binding=json.loads(Path(self.bindings).read_text())
        binding['training_queue']['sha256']=qsha
        binding['training_audit']=dict(path=self.audit,sha256=c.sha(self.audit),implementation_review=dict(path=self.auditreview,
           sha256=c.sha(self.auditreview),required_scope='synthetic_audit_implementation'))
        put(self.bindings,binding)
        self.inputpins.update(audit['input_sha256']);self.inputpins[self.auditreview]=c.sha(self.auditreview)
        self.refresh_validation()
    def refresh_validation(self):
        binding=json.loads(Path(self.bindings).read_text())
        binding['training_audit']['sha256']=c.sha(self.audit)
        self.inputpins[self.audit]=c.sha(self.audit);self.inputpins[self.auditreview]=c.sha(self.auditreview)
        binding['input_sha256']=self.inputpins;put(self.bindings,binding)
        self.stage_sources={**self.sourcepins,self.bindings:c.sha(self.bindings),self.protocol:c.sha(self.protocol)}
        for job in self.jobs:
            for name in ('protocol.json','summary.json'):
                p=Path(job['job_out'])/name;x=json.loads(p.read_text());x['input_sha256'][self.bindings]=c.sha(self.bindings);put(p,x)
        q=json.loads(Path(self.queue).read_text());q.update(source_sha256=self.stage_sources,input_sha256=self.inputpins);put(self.queue,q)
        qsha=c.sha(self.queue);review=q['review_paths'][0]
        put(review,dict(status='PASS',independent=True,admission_phase='validation',reviewed_sha256={**self.stage_sources,self.queue:qsha}))
        for job,rec in zip(self.jobs,self.receipts):
            rec.update(queue_sha256=qsha,requested_inputs_sha256={p:self.inputpins[p] for p in job['allowed_reads']})
            put(self.run/'jobs'/(job['id']+'.receipt.json'),rec)
        for name in ('driver_start.json','completion.json','launch_receipt.json'):
            p=self.run/name;x=json.loads(p.read_text());x['queue_sha256']=qsha
            if 'source_sha256' in x:x['source_sha256']=self.stage_sources
            if 'input_sha256' in x:x['input_sha256']=self.inputpins
            if 'jobs' in x:x['jobs']=self.receipts
            put(p,x)
        for when in ('before','after'):
            put(self.run/('input_fingerprints.'+when+'.json'),self.inputpins)
            put(self.run/('operator_fingerprints.'+when+'.json'),{**self.stage_sources,self.queue:qsha,review:c.sha(review)})

    def make_eval(self,cap,seed,pool,cohort,number):
        count=2 if cohort=='original_val' else 3;base=0 if cohort=='original_val' else 2
        rows=[]
        for i,variant in enumerate(['observed']*count+['frequency_crop','peak_dropout']):
            costs=list(map(float,range(512)));costs[1]=-1. if cap==10 else 1.
            rank=1 if cap==10 else 2;ids=list(range(512))
            q=dict(query_order=i,variant=variant,source=c.SOURCES[(base+i)%5],record_id=cohort+str(i),smiles='C',group=cohort+'g'+str(i))
            rows.append(dict(query_identity=q,target_in_pool=True,rank=rank,RR=1/rank,in_pool_target_rank=rank,cached_outside_pool_rank=None,
              target_candidate_index=1,target_cost=costs[1],reranking_seconds=.1,appended_training_positive=False,top1=rank==1,top5=True,top10=True,gamma=.25,mask_content_sha256='a'*64,
              natural_candidate_indices=ids,natural_candidate_costs=costs,natural_candidate_coarse_costs=list(map(float,range(512))),
              natural_candidate_trace_sha256=[[hashlib.sha256(f'{cap}/{role}/{k}'.encode()).hexdigest() for k in ids] for role in range(2)]))
        out=self.root/'evaluations'/f'C{cap}_s{seed}_{pool}_{cohort}';out.mkdir(parents=True)
        pred=out/'predictions.jsonl';pred.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        cache=self.caches[str(seed)][cohort][pool];roles=self.roles[str(cap)][str(seed)]
        inputs={**self.sourcepins,self.protocol:c.sha(self.protocol),self.bindings:c.sha(self.bindings),cache['path']:cache['sha256']}
        for r in roles.values():
            inputs.update({r['checkpoint']:r['checkpoint_sha256'],r['centers']:r['centers_sha256'],r['center_rows']:r['center_rows_sha256'],r['producer_summary']:r['producer_summary_sha256']})
        self.inputpins.update(inputs)
        args=dict(cap=cap,seed=seed,pool=pool,cohort=cohort,out=str(out),source=str(self.root),bindings=self.bindings,protocol=self.protocol,cache=cache['path'])
        command=[str(Path(__file__).with_name('evaluate_cap_ranker.py'))]
        for k,v in args.items():command+=['--'+k.replace('_','-'),str(v)]
        pid=200+number;argv=command
        put(out/'protocol.json',dict(args=args,argv=argv,pid=pid,ppid=101,started_utc=T1,input_sha256=inputs))
        summary=dict(schema='cap_ranker_evaluation_20261024',scope='complete_train_validation_evaluation',status='PASS',cap=cap,seed=seed,pool=pool,cohort=cohort,inputs_unchanged=True,
          model_state_before='fixed',model_state_after='fixed',test_read=False,training=False,selection_performed=False,promotion=False,
          forward_bank_calls=0,external_api_calls=0,pid=pid,ppid=101,started_utc=T1,finished_utc=T2,input_sha256=inputs,source_sha256=self.sourcepins,
          predictions_sha256=c.sha(pred),records=len(rows),identity_sha256=c.ordered_sha(sorted(c.identity_sha(r) for r in rows)),
          metrics={v:c.metrics([r for r in rows if r['query_identity']['variant']==v]) for v in c.VARIANTS},
          reranking_seconds_mean=.1,reranking_seconds_p95=.1,elapsed_seconds=.4,cost_semantics='synthetic raw timing',
          cache_sha256=cache['sha256'],cache_manifest={k:v for k,v in cache['manifest'].items() if k!='feature_cache_sha256'},producer_manifest=cache['manifest'],checkpoints={v['checkpoint']:v['checkpoint_sha256'] for v in roles.values()})
        put(out/'summary.json',summary);plain(out/'EXIT_CODE','0\n')
        jid=f'eval{number}';access=self.root/'access_logs'/(jid+'.json')
        job=dict(id=jid,argv=[sys.executable,'-B',self.source,'--']+command,command=command,cwd=str(self.root),job_out=str(out),gpu=None,access_log=str(access),allowed_reads=list(inputs))
        rec=dict(id=jid,argv=job['argv'],cwd=job['cwd'],gpu=None,pid=pid,parent_pid=101,started_utc=T1,finished_utc=T2,internal_exit=0,external_exit=0,job_out=str(out),access_log=str(access))
        put(access,dict(pid=pid,parent_pid=101,argv=command,internal_exit=0,rejected_paths=[],started_utc=T1,finished_utc=T2))
        self.jobs.append(job);self.receipts.append(rec)
    def execute(self):
        args=type('Args',(),dict(protocol=self.protocol,bindings=self.bindings,training_audit=self.audit,queue=self.queue,run=str(self.run),out=str(self.out)))()
        self.out.mkdir()
        with patch.object(c,'MERGED_OBSERVED',5),patch.dict(s.COHORT_OBSERVED,{'original_val':2,'tune':3}):
            return s.run(args,self.out)
class MatrixTests(unittest.TestCase):
    def test_complete16_real_reader_648_table_and_select_only_C10dual(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);r=f.execute()
            self.assertEqual(r['table_rows'],648);self.assertEqual(r['preselected'],'cap10_dual')
            self.assertEqual(r['signals']['cap5_dual']['MRR'],.5);self.assertEqual(r['signals']['cap10_dual']['MRR'],1.)
            self.assertTrue(r['deferred_tensor_sha256']);self.assertEqual(len(r['cost_records']),16)
    def test_bool_exit_not_real_int_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);p=f.run/'launch_receipt.json';x=json.loads(p.read_text());x['external_exit']=False;put(p,x)
            with self.assertRaisesRegex(ValueError,'real_double_exit'):f.execute()
    def test_one_missing_named_eval_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);p=f.run/'completion.json';x=json.loads(p.read_text());x['jobs'].pop();put(p,x)
            with self.assertRaisesRegex(ValueError,'complete_job_set'):f.execute()
    def test_missing_required_audit_check_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);x=json.loads(Path(f.audit).read_text());del x['checks']['matched_negative_exposure'];put(f.audit,x);f.refresh_validation()
            with self.assertRaisesRegex(ValueError,'all_training_checks'):f.execute()
    def test_false_model_center_binding_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);x=json.loads(Path(f.audit).read_text());x['models'][0]['centers_sha256']='0'*64;put(f.audit,x);f.refresh_validation()
            with self.assertRaisesRegex(ValueError,'audited_weights_centers'):f.execute()

    def test_explicit_successful_run_name_is_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp,run_name='validation_after_import_fix');r=f.execute()
            self.assertEqual(r['validation_run'],str(f.root/'runs/validation_after_import_fix'))
    def test_failed_original_training_run_cannot_replace_successful_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);x=json.loads(Path(f.audit).read_text());x['training_run']=str(f.root/'runs/train');put(f.audit,x);f.refresh_validation()
            with self.assertRaisesRegex(ValueError,'actual_successful_training_run'):f.execute()

    def test_unregistered_duplicate_PASS_audit_is_refused_before_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);other=f.root/'unregistered_audit.json';other.write_bytes(Path(f.audit).read_bytes())
            with self.assertRaisesRegex(ValueError,'registered_training_audit_path'):
                s.training_audit(c.Pins(),str(other),json.loads(Path(f.bindings).read_text()),f.protocol)
    def test_registered_review_must_cover_all_audit_source_SHA(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MatrixFixture(tmp);review=json.loads(Path(f.auditreview).read_text());review['reviewed_sha256'].pop(next(iter(review['reviewed_sha256'])));put(f.auditreview,review)
            b=json.loads(Path(f.bindings).read_text());b['training_audit']['implementation_review']['sha256']=c.sha(f.auditreview);put(f.bindings,b);f.refresh_validation()
            with self.assertRaisesRegex(ValueError,'complete_source_review'):f.execute()

class EstimandTests(unittest.TestCase):
    def test_query_merge_source_macro_and_mean_of_conditional_ratios(self):
        matrix={}
        for cap,seed,pool,cohort in itertools.product(c.CAPS,c.SEEDS,c.POOLS,('original_val','tune')):
            n=320 if cohort=='original_val' else 1024;rows=[]
            for i in range(n):
                source='HMDB' if cohort=='original_val' else c.SOURCES[1+i%4]
                hit=(seed==211 or cohort=='original_val')
                rank=2 if seed==211 else (1 if hit else 1000)
                q=dict(query_order=i,variant='observed',source=source,record_id=cohort+str(i),smiles='C',group=cohort+str(i))
                rows.append(dict(query_identity=q,rank=rank,RR=1/rank,target_in_pool=hit))
            for variant in ('frequency_crop','peak_dropout'):
                rows.append(dict(query_identity=dict(q,variant=variant),rank=1,RR=1.,target_in_pool=True))
            summary=dict(records=len(rows),reranking_seconds_mean=.1,reranking_seconds_p95=.1,elapsed_seconds=1.,cost_semantics='synthetic')
            matrix[(cap,seed,pool,cohort)]=(summary,{'args':{'out':'synthetic'}},rows)
        values,seeds,table,cost=s.aggregate(matrix)
        self.assertAlmostEqual(values['cap10_dual']['MRR'],((320+1024*.001)/1344+.5)/2)
        self.assertAlmostEqual(values['cap10_dual']['macro_MRR'],((1+4*.001)/5+.5)/2)
        allrow=next(r for r in table if r['config']=='cap10_dual' and r['cohort']=='merged_validation' and r['variant']=='observed' and r['source']=='all' and r['seed']=='two_seed_mean')
        self.assertEqual(allrow['in_pool_MRR'],.75)
        self.assertNotEqual(allrow['in_pool_MRR'],(320+1344*.5)/(320+1344))
        sparse=next(r for r in table if r['config']=='cap10_dual' and r['cohort']=='original_val' and r['variant']=='observed' and r['source']=='WanJuan' and r['seed']=='two_seed_mean')
        self.assertEqual(sparse['queries'],0);self.assertIsNone(sparse['MRR'])
    def test_missing_source_is_not_dropped(self):
        rows=[dict(query_identity=dict(query_order=i,variant='observed',source='HMDB',record_id=str(i),smiles='C',group=str(i)),rank=1,RR=1.,target_in_pool=True) for i in range(1344)]
        with self.assertRaisesRegex(ValueError,'five_sources'):c.signal_for_seed(rows)

if __name__=='__main__':unittest.main()

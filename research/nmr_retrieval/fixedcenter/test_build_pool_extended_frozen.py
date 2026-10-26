"""功能概述：以合成完成证据测试扩展冻结器的24任务与metadata-only边界。
复用旧冻结fixture并追加4fit/8val摘要、中心证据和严格driver来源；不执行任何模型。
测试覆盖完整矩阵、独立review全SHA、唯一validation选择、中心/seed/producer绑定、
输入漂移及实际check_frozen/driver的兼容；临时测试cache字节只由fixture建立。
"""
import copy
import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import build_pool_extended_frozen as m
import test_build_pool_frozen as legacy

write=legacy.write


class Fixture:
    def __init__(self,tmp):
        self.f=legacy.Fixture(tmp);f=self.f;self.root=f.root
        self.fixed=Path(write(f.root/'h_pool_fixed_center_ranker.py','# synthetic fixed source'))
        self.adapter=Path(write(f.base/'evaluate_pool_fixed_center_sealed.py','# synthetic adapter'))
        self.joint_tool=Path(write(f.base/'summarize_fixed_center_validation.py','# synthetic joint tool'))
        self.registration=Path(write(f.base/'additional_plan.json',dict(schema='fixed_train_center_conditional_preregistration_20261023',
            single_additional_family='fixedcenter_mixed_listwise',configurations=list(m.ADDED_CONFIGS),seeds=[193,211],roles=['no_aug','aug'],fits=4)))
        write(f.source/'nmr_transfer/data.py','# synthetic identity source')
        write(f.audit_path.parent/'EXIT_CODE','0\n')
        sourcepins={str(self.fixed):m.sha(self.fixed)}
        self.models=[];jobs=[];extra_inputs={str(Path(m.additional_audit.__file__).resolve()):m.sha(m.additional_audit.__file__),
                                          str(Path(m.additional_audit.old.__file__).resolve()):m.sha(m.additional_audit.old.__file__)}
        for seed in m.validation.SEEDS:
            for aug in (False,True):
                role='aug' if aug else 'no_aug';out=f.root/'additional/rankers/fixedcenter_mixed_listwise'/f's{seed}'/role
                estimate=dict(seed=seed,augment=aug,selection_scope='fit_train_only',center33=[float(seed)]*33,O_center3=[float(aug)]*3)
                write(out/'center_rows.jsonl','{"synthetic":true}\n')
                estimate['row_audit_sha256']=m.sha(out/'center_rows.jsonl')
                write(out/'centers.json',estimate)
                for name in ('initial.pt','best.pt','last.pt'):write(out/name,f'fixed{seed}{role}{name}')
                write(out/'summary.json',dict(center_estimate_sha256=m.sha(out/'centers.json')))
                write(out/'protocol.json',dict(args=dict(seed=seed,augment=aug)))
                write(out/'history.jsonl','{}\n');write(out/'EXIT_CODE','0\n')
                job=dict(id=f'fit_fixed_s{seed}_{role}',kind='training',out=str(out),argv=['python',str(self.fixed)],
                         cwd=str(f.source),deps=[],inputs_sha256={})
                jobs.append(job);self.models.append(dict(job_id=job['id'],role=[seed,aug],output=str(out),
                    center_estimate_sha256=m.sha(out/'centers.json'),center_rows_sha256=m.sha(out/'center_rows.jsonl')))
                extra_inputs.update({str(out/name):m.sha(out/name) for name in
                    ('summary.json','protocol.json','history.jsonl','EXIT_CODE','initial.pt','best.pt','last.pt','centers.json','center_rows.jsonl')})
        self.evaluations=[];added_meta={}
        for config in m.ADDED_CONFIGS:
            pool=config.rsplit('_',1)[1];added_meta[config]={}
            for seed in m.validation.SEEDS:
                weights={str(Path(next(x['output'] for x in self.models if x['role']==[seed,aug]))/'best.pt'):
                         m.sha(Path(next(x['output'] for x in self.models if x['role']==[seed,aug]))/'best.pt') for aug in (False,True)}
                added_meta[config][str(seed)]={}
                centers=[next(x['center_estimate_sha256'] for x in self.models if x['role']==[seed,aug]) for aug in (False,True)]
                for cohort in m.validation.COHORTS:
                    cache=f.old/'features'/f'inherited_s{seed}'/pool/cohort/'features.pt';write(cache,'synthetic validation tensor')
                    producer=json.loads((cache.parent/'summary.json').read_text())
                    producer.update(feature_cache_sha256=m.sha(cache),query_sha256='a'*64,candidate_order_sha256=f.candidate)
                    write(cache.parent/'summary.json',producer)
                    f.summary['input_sha256'][str(cache.parent/'summary.json')]=m.sha(cache.parent/'summary.json')
                    out=f.root/'additional/fixedcenter_mixed_listwise'/f's{seed}'/pool/cohort
                    args=dict(stage='evaluate',source=str(f.source),cache=str(cache),seed=seed,out=str(out),
                        no_aug_checkpoint=next(p for p in weights if Path(p).parent.name=='no_aug'),
                        aug_checkpoint=next(p for p in weights if Path(p).parent.name=='aug'))
                    inputs=dict(sourcepins,**weights);inputs[str(cache)]=m.sha(cache)
                    write(out/'protocol.json',dict(args=args,inputs_sha256=inputs))
                    write(out/'summary.json',dict(input_sha256=inputs,cohort=cohort,recall=pool,query_sha256='a'*64,
                        candidate_order_sha256=f.candidate,checkpoints=weights,center_estimate_sha256=centers))
                    write(out/'EXIT_CODE','0\n')
                    argv=['python',str(self.fixed)]
                    for k,v in args.items():argv+=['--'+k.replace('_','-'),str(v)]
                    job=dict(id=f'eval_fixed_s{seed}_{pool}_{cohort}',kind='validation',argv=argv,cwd=str(f.source),
                             out=str(out),deps=[],inputs_sha256={str(cache):m.sha(cache)})
                    jobs.append(job);self.evaluations.append(dict(job_id=job['id'],role=[seed,pool,cohort],
                        output=str(out),checkpoints=weights))
                    added_meta[config][str(seed)][cohort]=dict(candidate_order_sha256=f.candidate,checkpoints=weights)
                    extra_inputs.update({str(out/name):m.sha(out/name) for name in ('summary.json','protocol.json','EXIT_CODE')})
                    extra_inputs.update(inputs);extra_inputs[str(cache.parent/'summary.json')]=m.sha(cache.parent/'summary.json')
                    write(cache.parent/'EXIT_CODE','0\n');extra_inputs[str(cache.parent/'EXIT_CODE')]=m.sha(cache.parent/'EXIT_CODE')
        for config in m.validation.CONFIGS:f.summary['signals'][config]=dict(f.summary['signals']['frozen_single'])
        f.summary['selection']=m.validation.choose(f.summary['signals'])
        write(f.summary_path,f.summary)
        self.queue_path=Path(write(f.root/'additional_queue.json',dict(jobs=jobs,source_inputs_sha256=sourcepins)))
        self.run=f.root/'runs/additional'
        start='2026-10-23T05:00:'
        bound=dict(sourcepins,**{str(self.queue_path):m.sha(self.queue_path)});receipts={}
        for job in jobs:
            r=dict(state='complete',external_exit=0,internal_exit=0,started_utc=start+'02Z',finished_utc=start+'03Z',
                argv=job['argv'],output=job['out'],cwd=job['cwd'],requested_inputs_sha256=job['inputs_sha256'],
                queue_source_sha256=bound)
            receipts[job['id']]=r;write(self.run/(job['id']+'.receipt.json'),r)
        write(self.run/'launch_receipt.json',dict(external_exit=0,internal_exit=0,started_utc=start+'00Z',
            finished_utc=start+'05Z',queue_sha256={str(self.queue_path):m.sha(self.queue_path)},
            bundle_sha256=sourcepins,source_sha256=sourcepins,child_pid=999,argv=['python','-m','driver','--queue',str(self.queue_path)]))
        write(self.run/'driver_start.json',dict(pid=999,argv=['driver','--queue',str(self.queue_path)],
            utc=start+'01Z',inputs_sha256=bound))
        write(self.run/'completion.json',dict(success=True,inputs_unchanged=True,utc=start+'04Z',jobs=receipts))
        write(self.run/'EXIT_CODE','0\n');write(self.run/'EXTERNAL_EXIT_CODE','0\n')
        self.audit=dict(status='PASS',scope='fixedcenter_training_no_test',queue_sha256=m.sha(self.queue_path),
            training_jobs=4,validation_jobs=8,inputs_unchanged=True,draws_per_fit=524288,steps_per_fit=16384,
            checks=dict.fromkeys(m.ADDITIONAL_CHECKS,True),models=self.models,evaluations=self.evaluations,input_sha256=extra_inputs)
        self.audit_path=Path(write(f.base/'extra_audit/summary.json',self.audit));write(self.audit_path.parent/'EXIT_CODE','0\n')
        self.summary=copy.deepcopy(f.summary)
        self.summary.update(schema=m.JOINT_SCHEMA,main_validation_summary=dict(path=str(f.summary_path),sha256=m.sha(f.summary_path)),
            fixed_center_training_audit=dict(path=str(self.audit_path),sha256=m.sha(self.audit_path)),
            additional_queue=dict(path=str(self.queue_path),sha256=m.sha(self.queue_path)),additional_run=str(self.run))
        self.summary['source_sha256'][str(self.joint_tool)]=m.sha(self.joint_tool)
        self.summary['input_sha256'].update(extra_inputs)
        self.summary['metadata'].update(added_meta)
        signal=dict(next(iter(f.summary['signals'].values())))
        for config in m.ADDED_CONFIGS:self.summary['signals'][config]=dict(signal)
        self.summary['signals'][m.ADDED_CONFIGS[1]]['MRR']=.55
        self.summary['selection']=m.choose(self.summary['signals'])
        self.summary_path=Path(write(f.base/'joint/summary.json',self.summary));write(self.summary_path.parent/'EXIT_CODE','0\n')
        self.validation_review=Path(f.base/'joint_review.json')
        self.plan=copy.deepcopy(f.plan);self.plan['schema']=m.PLAN_SCHEMA
        self.plan['preselected']=self.summary['selection']['preselected']
        for config in ('frozen_dual',)+m.ADDED_CONFIGS:
            for seed in m.validation.SEEDS:
                e=next(e for e in f.plan['evaluations'] if e['config']=='mixed_listwise_'+config.rsplit('_',1)[1] and e['seed']==seed)
                self.plan['evaluations'].append(dict(e,config=config))
        self.plan['source_sha256']={str(f.source/'nmr_transfer'/name):m.sha(f.source/'nmr_transfer'/name) for name in m.SOURCE_NAMES}
        self.plan['tool_sha256']=self.tools()
        self.plan_path=Path(write(f.base/'extended_plan.json',self.plan));self.review_path=f.base/'extended_review.json'
        self.approve()

    def tools(self):
        return {str(Path(p).resolve()):m.sha(p) for p in [m.__file__,m.old.__file__,m.validation.__file__,
            m.additional_audit.__file__,m.additional_audit.old.__file__,self.joint_tool,self.fixed,self.adapter,self.f.test_summarizer]}

    def args(self):
        f=self.f
        return SimpleNamespace(root=str(f.root),source=str(f.source),validation_summary=str(self.summary_path),
            summary_sha256=m.sha(self.summary_path),main_validation_summary=str(f.summary_path),
            validation_review=str(self.validation_review),joint_summarizer=str(self.joint_tool),test_plan=str(self.plan_path),
            review=str(self.review_path),train_audit=str(f.audit_path),additional_train_audit=str(self.audit_path),
            additional_queue=str(self.queue_path),additional_run=str(self.run),eligibility=str(f.eligibility_path),
            preregistration=str(f.registration_path),additional_preregistration=str(self.registration),
            fixed_center_ranker=str(self.fixed),sealed_adapter=str(self.adapter),test_summarizer=str(f.test_summarizer),
            formal_run=None,diagnostics_run=None)

    def approve(self):
        vr={str(self.summary_path):m.sha(self.summary_path),str(self.joint_tool):m.sha(self.joint_tool)}
        vr.update(self.summary['input_sha256']);vr.update(self.summary['source_sha256'])
        write(self.validation_review,dict(status='PASS',independent=True,reviewer='independent_synthetic',
            scope='joint_validation_results_and_implementation',reviewed_sha256=vr))
        # Capture exact required map from the real build until final review; no scientific operation is mocked.
        captured={}
        write(self.review_path,{})
        def collect(review,required):captured.update(required)
        out=self.f.base/'approval_probe';out.mkdir(exist_ok=True)
        with patch.object(m.old,'review',side_effect=collect):
            self._build(out)
        for p in out.iterdir():p.unlink()
        out.rmdir()
        write(self.review_path,dict(status='PASS',independent=True,reviewer='independent_synthetic',
            scope='freeze_plan_and_implementation',reviewed_sha256=captured))

    def _build(self,out):
        class Clock(datetime.datetime):
            @classmethod
            def now(cls,tz=None):return cls(2026,10,23,12,0,0,tzinfo=datetime.timezone.utc)
        with patch.object(m.dt,'datetime',Clock):return m.build(self.args(),out)

    def build(self):
        out=self.f.base/'freeze_extended';out.mkdir()
        result=self._build(out)
        return result,json.loads((out/'frozen_selection.json').read_text()),json.loads((out/'sealed_queue.json').read_text())


class ExtendedFrozenTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Fixture(self.tmp.name)

    def test_real_full_build24_and_metadata_only(self):
        original=Path.open
        def guard(path,*a,**kw):
            if '/sealed/features/' in str(path) and path.name=='features.pt':raise AssertionError('test_tensor_read')
            if path.name=='predictions.jsonl':raise AssertionError('predictions_read')
            return original(path,*a,**kw)
        with patch.object(Path,'open',guard):receipt,frozen,queue=self.f.build()
        self.assertEqual(receipt['jobs'],24);self.assertEqual(len(queue['jobs']),24)
        self.assertEqual(len(frozen['ranker_configurations']),12)
        self.assertEqual(frozen['preselected'],m.ADDED_CONFIGS[1]);self.assertFalse(frozen['promotion'])
        self.assertEqual(set(frozen['fixed_center_configurations']),set(m.ADDED_CONFIGS))
        for config in m.ADDED_CONFIGS:
            for seed in m.validation.SEEDS:
                fixed=frozen['fixed_center_configurations'][config][str(seed)]
                self.assertEqual(fixed['cohort'],'test');self.assertEqual(len(fixed['center_artifacts']),2)
                self.assertIn('--config',next(j['argv'] for j in queue['jobs'] if j['id']==f'sealed_{config}_s{seed}'))
        self.assertTrue(all(str(self.f.audit_path) not in j['inputs_sha256'] for j in queue['jobs']))
        self.assertIn(str(self.f.audit_path),queue['source_inputs_sha256'])
        import ast
        tree=ast.parse(legacy.scientific_source('h_research_driver.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_queue')
        ns={'Path':Path};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actualdriver','exec'),ns)
        ns['validate_queue'](queue)

    def test_missing_added_configuration(self):
        self.f.plan['evaluations']=[e for e in self.f.plan['evaluations'] if e['config']!=m.ADDED_CONFIGS[0]]
        write(self.f.plan_path,self.f.plan)
        with self.assertRaisesRegex(ValueError,'extended24'):self.f.build()

    def test_missing_dual_control(self):
        self.f.plan['evaluations']=[e for e in self.f.plan['evaluations'] if e['config']!='frozen_dual']
        write(self.f.plan_path,self.f.plan)
        with self.assertRaisesRegex(ValueError,'extended24'):self.f.build()

    def test_plan_cannot_reselect(self):
        self.f.plan['preselected']=None;write(self.f.plan_path,self.f.plan)
        with self.assertRaisesRegex(ValueError,'explicit_extended'):self.f.build()

    def test_failed_new_audit(self):
        self.f.audit['checks']['full_budget']=False;write(self.f.audit_path,self.f.audit)
        with self.assertRaisesRegex(ValueError,'complete_additional'):self.f.build()

    def test_missing_fit_checkpoint_audit_pin(self):
        for name in ('initial.pt','last.pt'):
            path=str(Path(self.f.models[0]['output'])/name)
            self.f.audit['input_sha256'].pop(path)
        write(self.f.audit_path,self.f.audit)
        with self.assertRaisesRegex(ValueError,'additional_audit_missing_artifact'):self.f.build()

    def test_missing_new_eval_metadata(self):
        out=Path(self.f.evaluations[0]['output']);(out/'summary.json').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):self.f.build()

    def test_center_artifact_changed(self):
        path=Path(self.f.models[0]['output'])/'centers.json';write(path,dict(center33=[99.]*33))
        with self.assertRaisesRegex(ValueError,'input_sha_changed'):self.f.build()

    def test_review_must_bind_adapter(self):
        review=json.loads(self.f.review_path.read_text());del review['reviewed_sha256'][str(self.f.adapter)]
        write(self.f.review_path,review)
        with self.assertRaisesRegex(ValueError,'review_does_not_bind'):self.f.build()

    def test_joint_review_must_bind_summary(self):
        review=json.loads(self.f.validation_review.read_text());del review['reviewed_sha256'][str(self.f.summary_path)]
        write(self.f.validation_review,review)
        with self.assertRaisesRegex(ValueError,'joint_review_missing'):self.f.build()

    def test_seed_swapped_cache(self):
        e=next(e for e in self.f.plan['evaluations'] if e['config']==m.ADDED_CONFIGS[0] and e['seed']==193)
        other=next(e for e in self.f.plan['evaluations'] if e['config']==m.ADDED_CONFIGS[0] and e['seed']==211)
        e.update(cache=other['cache'],cache_sha256=other['cache_sha256']);write(self.f.plan_path,self.f.plan)
        with self.assertRaisesRegex(ValueError,'seed_or_inherited'):self.f.build()

    def test_old_selection_gates_preserved(self):
        old=m.validation.choose({c:self.f.summary['signals'][c] for c in tuple(m.validation.CONFIGS)+('frozen_single','frozen_dual')})
        joint=m.choose(self.f.summary['signals'])
        for c in m.validation.CONFIGS:self.assertEqual(old['configurations'][c],joint['configurations'][c])

    def test_actual_joint_helper_selection_exact_contract(self):
        import summarize_fixed_center_validation as joint
        additional={c:self.f.summary['signals'][c] for c in m.ADDED_CONFIGS}
        self.assertEqual(m.choose(self.f.summary['signals']),joint.extend_selection(self.f.f.summary,additional))

    def test_unregistered_incumbent_rejected(self):
        self.f.plan['evaluations'][0]['config']='incumbent_single';write(self.f.plan_path,self.f.plan)
        with self.assertRaisesRegex(ValueError,'extended24'):self.f.build()


if __name__=='__main__':unittest.main()

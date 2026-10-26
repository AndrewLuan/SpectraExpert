"""功能概述：以临时合成产物检验封存测试汇总的身份、模型、退出及固定选择契约。
fixture生成1000正常、200裁窗、171缺峰记录及真实JSON/JSONL/SHA，不读研究数据。
TestReadTest覆盖完整producer输入和多类损坏；TestAggregate确认先算逐seed指标再平均，
缺seed失败、历史召回证据不补造。全部文件位于TemporaryDirectory，退出后清理。
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import summarize_pool_test as m


def fixture(base):
    config=m.validation.CONFIGS[0];seed=193
    folder=base/'sealed'/config/'s193/test';folder.mkdir(parents=True)
    cache=base/'cache/features.pt';cache.parent.mkdir();cache.write_bytes(b'synthetic-feature-cache')
    checkpoints={}
    for role in ('no_aug','aug'):
        p=base/'rankers'/role/'best.pt';p.parent.mkdir(parents=True);p.write_bytes(role.encode());checkpoints[role]=str(p)
    frozen_path=base/'frozen.json';frozen_path.write_text('{}')
    args=dict(stage='evaluate',cache=str(cache),no_aug_checkpoint=checkpoints['no_aug'],aug_checkpoint=checkpoints['aug'],
              seed=seed,out=str(folder),frozen_selection=str(frozen_path))
    inputs={str(p):m.validation.sha(p) for p in (cache,Path(checkpoints['no_aug']),Path(checkpoints['aug']),frozen_path)}
    argv=['/python','-m','nmr_transfer.h_pool_ranker']
    for name,value in args.items():argv.extend(['--'+name.replace('_','-'),str(value)])
    job=dict(out=str(folder),argv=argv,inputs_sha256=inputs)
    rows=[];indices=list(range(512));costs=[float(i)/512 for i in indices]
    for i in range(1000):
        for variant in m.VARIANTS:
            if variant=='frequency_crop' and i>=200 or variant=='peak_dropout' and i>=171:continue
            rows.append(dict(query_order=i,source=sorted(m.validation.SOURCES)[i%5],record_id='synthetic_'+str(i),
                smiles='C',group='synthetic_group_'+str(i),variant=variant,mask_content_sha256='mask_'+str(i)+variant,
                rank=1+i%700,target_in_coarse_pool=i%700<512,evaluation_pool_size=512,
                natural_candidate_indices=indices,natural_candidate_costs=costs))
    rows.sort(key=lambda r:(r['query_order'],r['variant']))
    fields=('query_order','variant','source','record_id','smiles','group')
    identity=dict(query_sha256='a'*64,
        identity_content_sha256=m.validation.content_sha([{k:r[k] for k in fields} for r in rows]),
        variant_mask_content_sha256=m.validation.content_sha([(tuple(r[k] for k in fields[:-1]),r['mask_content_sha256']) for r in rows]))
    manifest=dict(cohort='test',recall='single',candidate_count=114982,pool_size=512,variants=m.VARIANTS,
        base_queries=1000,records=1371,appended_training_positives=0,candidate_order_sha256=m.validation.CANDIDATE_SHA,
        feature_cache_sha256=inputs[str(cache)],matching_seconds_mean=.01,**identity)
    normal=[r for r in rows if r['variant']=='observed']
    metrics=dict(overall=m.validation.metrics(normal),by_source={s:m.validation.metrics([r for r in normal if r['source']==s])
        for s in sorted(m.validation.SOURCES)},stress={v:m.validation.metrics([r for r in rows if r['variant']==v]) for v in list(m.VARIANTS)[1:]})
    summary=dict(input_sha256=inputs,checkpoints={p:inputs[p] for p in checkpoints.values()},metrics=metrics,
        cohort='test',recall='single',candidate_order_sha256=m.validation.CANDIDATE_SHA,query_sha256=identity['query_sha256'],
        reranking_seconds_mean=.02)
    (cache.parent/'summary.json').write_text(json.dumps(manifest))
    (folder/'protocol.json').write_text(json.dumps(dict(args=args,inputs_sha256=inputs)))
    (folder/'summary.json').write_text(json.dumps(summary))
    (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    allowed=dict(no_aug_checkpoint=args['no_aug_checkpoint'],aug_checkpoint=args['aug_checkpoint'],seed=seed,
                 cohort='test',recall='single',cache_sha256=inputs[str(cache)])
    frozen=dict(allowed_ranker_configs=[allowed],fixed_test_identity=identity,ranker_configurations={config:{str(seed):copy.deepcopy(allowed)}})
    return job,frozen,rows,args


def sealed_fixture(base):
    folder=base/'freeze';folder.mkdir();run=base/'runs/sealed';run.mkdir(parents=True)
    frozen_path=folder/'frozen_selection.json';queue_path=folder/'sealed_queue.json'
    configs=sorted(set(m.validation.CONFIGS)|{'frozen_single'})
    review_path=folder/'independent_review.json';reviewed={str(Path(m.__file__).resolve()):m.validation.sha(m.__file__)}
    review_path.write_text(json.dumps(dict(status='PASS',independent=True,reviewer='synthetic_reviewer',
        scope='freeze_plan_and_implementation',reviewed_sha256=reviewed)))
    frozen=dict(schema='nmr_pool_ranker_20261023',selection_scope='train_val_only',promotion=False,
        verify_eligibility='unavailable',candidate_count=114982,candidate_order_sha256=m.validation.CANDIDATE_SHA,
        test_variants=m.VARIANTS,preselected=m.validation.CONFIGS[0],configurations=configs,
        reviewed_inputs_sha256=reviewed,evaluation_inputs_sha256={},freeze_review=dict(path=str(review_path),sha256=m.validation.sha(review_path)),
        fixed_test_identity=dict(query_sha256='a'*64,identity_content_sha256='b'*64,variant_mask_content_sha256='c'*64))
    named={c:{str(s):dict(no_aug_checkpoint=str(base/'rankers'/c/f's{s}/no_aug/best.pt'),
        aug_checkpoint=str(base/'rankers'/c/f's{s}/aug/best.pt'),seed=s,cohort='test',recall=c.rsplit('_',1)[1],cache_sha256='c'*64)
        for s in m.validation.SEEDS} for c in configs}
    frozen.update(ranker_configurations=named,allowed_ranker_configs=[v for seeds in named.values() for v in seeds.values()])
    frozen_path.write_text(json.dumps(frozen));(folder/'EXIT_CODE').write_text('0')
    driver_source=base/'nmr_transfer/h_research_driver.py';driver_source.parent.mkdir();driver_source.write_text('# synthetic producer\n')
    sources={str(p):m.validation.sha(p) for p in (frozen_path,driver_source)};jobs=[]
    for config in configs:
        for seed in m.validation.SEEDS:
            out=base/'sealed'/config/f's{seed}/test';out.mkdir(parents=True);(out/'EXIT_CODE').write_text('0')
            jobs.append(dict(id=f'sealed_{config}_s{seed}',kind='sealed',gpu=True,argv=['/python','-m','nmr_transfer.h_pool_ranker'],
                cwd=str(base),out=str(out),deps=[],inputs_sha256=sources))
    queue=dict(schema='nmr_research_jobs_20261022',source_inputs_sha256=sources,jobs=jobs)
    queue_path.write_text(json.dumps(queue))
    (folder/'build_receipt.json').write_text(json.dumps(dict(status='BUILT_NOT_EXECUTED',promotion=False,
        output_sha256={str(p):m.validation.sha(p) for p in (frozen_path,queue_path)})))
    argv=['/python','-m','nmr_transfer.h_research_driver','--queue',str(queue_path),'--out',str(run),'--gpus','0']
    launch=dict(external_exit=0,internal_exit=0,started_utc='2026-10-23T05:00:00Z',finished_utc='2026-10-23T05:05:00Z',
        queue_sha256={str(queue_path):m.validation.sha(queue_path)},bundle_sha256=sources,
        source_sha256={str(driver_source):m.validation.sha(driver_source)},child_pid=123,argv=argv)
    source_pins={**sources,**launch['queue_sha256']}
    start=dict(pid=123,argv=['/source/h_research_driver.py']+argv[3:],utc='2026-10-23T05:01:00Z',inputs_sha256=source_pins)
    records={}
    for job in jobs:
        record=dict(state='complete',external_exit=0,internal_exit=0,started_utc='2026-10-23T05:02:00Z',
            finished_utc='2026-10-23T05:03:00Z',argv=job['argv'],output=job['out'],cwd=job['cwd'],
            requested_inputs_sha256=job['inputs_sha256'],queue_source_sha256=source_pins)
        records[job['id']]=record;(run/(job['id']+'.receipt.json')).write_text(json.dumps(record))
    completion=dict(success=True,inputs_unchanged=True,utc='2026-10-23T05:04:00Z',jobs=records)
    for name,data in [('launch_receipt.json',launch),('driver_start.json',start),('completion.json',completion)]:
        (run/name).write_text(json.dumps(data))
    for name in ('EXIT_CODE','EXTERNAL_EXIT_CODE'):(run/name).write_text('0')
    return frozen_path,queue_path,run


class TestReadTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name).resolve()
        self.job,self.frozen,self.rows,self.args=fixture(self.base)

    def tearDown(self):self.temp.cleanup()

    def run_read(self):return m.read_test(self.job,self.frozen,{}, {},self.base/'frozen.json')

    def test_complete_fixed_test_producer(self):
        config,seed,rows,cost=self.run_read()
        self.assertEqual((config,seed),(m.validation.CONFIGS[0],193));self.assertEqual(len(rows),1371)
        self.assertAlmostEqual(cost,.03)

    def test_missing_stress_rejected(self):
        p=Path(self.job['out'])/'predictions.jsonl'
        p.write_text(''.join(json.dumps(r)+'\n' for r in self.rows if r['variant']!='peak_dropout'))
        with self.assertRaisesRegex(ValueError,'identity_or_variant'):self.run_read()

    def test_mask_content_mismatch_rejected(self):
        self.rows[0]['mask_content_sha256']='wrong'
        (Path(self.job['out'])/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        with self.assertRaisesRegex(ValueError,'identity_or_variant'):self.run_read()

    def test_checkpoint_substitution_rejected(self):
        self.frozen['allowed_ranker_configs'][0]['no_aug_checkpoint']='/wrong/best.pt'
        with self.assertRaisesRegex(ValueError,'not_allowlisted'):self.run_read()

    def test_another_allowlisted_family_cannot_be_relabelled(self):
        self.frozen['ranker_configurations'][m.validation.CONFIGS[0]]['193']['no_aug_checkpoint']='/different/allowed/best.pt'
        with self.assertRaisesRegex(ValueError,'not_allowlisted_for_config'):self.run_read()

    def test_another_freeze_file_cannot_be_substituted(self):
        with self.assertRaisesRegex(ValueError,'frozen_path_mismatch'):
            m.read_test(self.job,self.frozen,{}, {},self.base/'other-freeze.json')

    def test_protocol_and_driver_seed_must_match(self):
        p=Path(self.job['out'])/'protocol.json';data=json.loads(p.read_text());data['args']['seed']=211;p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'protocol_argv'):self.run_read()

    def test_metric_forgery_rejected(self):
        p=Path(self.job['out'])/'summary.json';data=json.loads(p.read_text());data['metrics']['overall']['MRR']=.9;p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'metric_mismatch'):self.run_read()

    def test_fixed_test_identity_changed_rejected(self):
        self.frozen['fixed_test_identity']['query_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'fixed_identity_required'):self.run_read()

    def test_empty_fixed_identity_rejected(self):
        self.frozen['fixed_test_identity']={}
        with self.assertRaisesRegex(ValueError,'fixed_identity_required'):self.run_read()

    def test_cache_input_drift_rejected(self):
        Path(self.args['cache']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'sha_changed'):self.run_read()

    def test_false_rank_membership_rejected(self):
        self.rows[0]['target_in_coarse_pool']=False
        (Path(self.job['out'])/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        with self.assertRaisesRegex(ValueError,'natural_pool_membership'):self.run_read()


class TestAggregate(unittest.TestCase):
    def test_metric_mean_does_not_average_ranks(self):
        row=dict(source='HMDB',variant='observed',rank=1,target_in_coarse_pool=True)
        data={'chosen':{193:[row],211:[dict(row,rank=3)]}}
        result=m.aggregate(data)
        mean=next(r for r in result if r['seed']=='two_seed_mean' and r['variant']=='observed' and r['source']=='all')
        self.assertAlmostEqual(mean['MRR'],2/3)
        self.assertEqual(mean['queries'],1)

    def test_missing_seed_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing_seed'):m.aggregate({'x':{193:[]}})

    def test_bool_is_not_actual_process_exit(self):
        self.assertTrue(m.double_exit({'external_exit':0,'internal_exit':0}))
        for key in ('external_exit','internal_exit'):
            bad={'external_exit':0,'internal_exit':0};bad[key]=False;self.assertFalse(m.double_exit(bad))


class TestSealedEnvelope(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name).resolve()
        self.frozen,self.queue,self.run=sealed_fixture(self.base)

    def tearDown(self):self.temp.cleanup()

    def verify(self):return m.verify_run(self.frozen,self.queue,self.run,{}, {})

    def test_complete18_job_envelope_preserves_preselected_without_selecting(self):
        with mock.patch.object(m.validation,'choose',side_effect=AssertionError('test cannot select')):
            frozen,jobs=self.verify()
        self.assertEqual(len(jobs),18);self.assertEqual(frozen['preselected'],m.validation.CONFIGS[0])

    def test_outer_exit_missing_rejected(self):
        (self.run/'EXTERNAL_EXIT_CODE').unlink()
        with self.assertRaises(FileNotFoundError):self.verify()

    def test_boolean_job_exit_rejected(self):
        p=self.run/'completion.json';completion=json.loads(p.read_text());key=next(iter(completion['jobs']))
        completion['jobs'][key]['internal_exit']=False;p.write_text(json.dumps(completion))
        (self.run/(key+'.receipt.json')).write_text(json.dumps(completion['jobs'][key]))
        with self.assertRaisesRegex(ValueError,'double_exit0'):self.verify()

    def test_finish_before_start_rejected(self):
        p=self.run/'launch_receipt.json';data=json.loads(p.read_text());data['finished_utc']='2026-10-23T04:00:00Z';p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'time_order'):self.verify()

    def test_unknown_job_cannot_enter_ablation(self):
        p=self.run/'completion.json';data=json.loads(p.read_text());data['jobs']['unregistered']={};p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'job_matrix'):self.verify()

    def test_frozen_content_changed_after_builder_rejected(self):
        data=json.loads(self.frozen.read_text());data['preselected']=None;self.frozen.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'input_sha_changed'):self.verify()


if __name__=='__main__':unittest.main()

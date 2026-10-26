"""功能概述：独立复核 summarize_pool_validation 的真实文件门禁与 producer 兼容性。
Fixture 创建完整 original_val 合成预测、缓存摘要、模型文件、protocol 和队列 argv；
DriverFixture 创建 48 项合成真实格式 receipt，用于拒绝伪 EXIT 与未完成证据。
Tests 覆盖身份、stress、source、seed、checkpoint、池合同和科学输入始末 SHA。
LegacyTests 调用旧 build_validation_manifest 的真实 legacy_entries 与 Snapshot，只把
冻结 SHA 参数替换为临时合成材料的 SHA，验证历史控制的 schema 和 None 召回证据。
所有数据均在临时目录，测试不读取真实谱值、不训练、不启动远程汇总、不修改主工具。
"""
import collections
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

D=Path(__file__).parent
LEGACY=next(p for p in (D.parent/'nmr_global_retrieval_20261022/build_validation_manifest.py',
    D.parent/'orchestrator-261022-1544/build_validation_manifest.py') if p.is_file())

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

m=load('review_summary',D/'summarize_pool_validation.py')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')

def make_rows(cohort='original_val',count=320):
    return sorted([dict(query_order=i,variant=v,source=sorted(m.SOURCES)[i%5],
        record_id=cohort+str(i),smiles='C'+str(i),group='g'+cohort+str(i),rank=1,
        target_in_coarse_pool=True,evaluation_pool_size=512,
        natural_candidate_indices=list(range(512)),natural_candidate_costs=[float(j) for j in range(512)],
        mask_content_sha256=digest([cohort,i,v]))
        for i in range(count) for v in m.VARIANTS],key=lambda r:(r['query_order'],r['variant']))

def row_summary(rows):
    normal=[r for r in rows if r['variant']=='observed']
    return dict(overall=m.metrics(normal),by_source={s:m.metrics([r for r in normal if r['source']==s])
        for s in sorted({r['source'] for r in normal})},stress={v:m.metrics([r for r in rows if r['variant']==v]) for v in m.VARIANTS[1:]})

class Fixture:
    def __init__(self,tmp,config='original_pairwise16_single'):
        self.root=Path(tmp).resolve()/'outputs/nmr_pool_ranking_20261023'
        self.expected=m.evaluation_args(self.root,config,193,'original_val')
        self.out=Path(self.expected['out']);self.out.mkdir(parents=True)
        self.cache=Path(self.expected['cache']);self.cache.parent.mkdir(parents=True);self.cache.write_text('synthetic features only')
        for role in ('no_aug_checkpoint','aug_checkpoint'):
            p=Path(self.expected[role]);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(role)
        self.inputs={str(self.cache):m.sha(self.cache),**{self.expected[k]:m.sha(self.expected[k]) for k in ('no_aug_checkpoint','aug_checkpoint')}}
        self.args=dict(self.expected)
        self.protocol=dict(args=self.args,inputs_sha256=self.inputs)
        self.rows=make_rows()
        self.cache_summary=dict(cohort='original_val',recall='single',candidate_count=114982,pool_size=512,
            query_sha256=digest('synthetic-query'),candidate_order_sha256=m.CANDIDATE_SHA,
            feature_cache_sha256=m.sha(self.cache),matching_seconds_mean=.01)
        self.refresh_identity()
        self.summary=dict(cohort='original_val',recall='single',query_sha256=self.cache_summary['query_sha256'],
            candidate_order_sha256=m.CANDIDATE_SHA,input_sha256=self.inputs,
            checkpoints={self.expected[k]:self.inputs[self.expected[k]] for k in ('no_aug_checkpoint','aug_checkpoint')},
            reranking_seconds_mean=.02)
        self.job=dict(kind='validation',argv=self.argv(self.expected),inputs_sha256={str(self.cache):m.sha(self.cache)})
        self.save()
    @staticmethod
    def argv(args):
        return ['python','-m','nmr_transfer.h_pool_ranker']+sum((['--'+k.replace('_','-'),str(v)] for k,v in args.items()),[])
    def refresh_identity(self):
        fields=('query_order','variant','source','record_id','smiles','group')
        self.cache_summary.update(records=len(self.rows),variants=dict(collections.Counter(r['variant'] for r in self.rows)),
            identity_content_sha256=digest([{k:r[k] for k in fields} for r in self.rows]),
            variant_mask_content_sha256=digest([(tuple(r[k] for k in fields[:-1]),r['mask_content_sha256']) for r in self.rows]))
    def save(self):
        self.summary['metrics']=row_summary(self.rows)
        write(self.out/'protocol.json',self.protocol);write(self.out/'summary.json',self.summary)
        write(self.cache.parent/'summary.json',self.cache_summary)
        (self.out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
    def read(self):
        return m.read_eval(self.out,193,'single','original_val',{str(self.out):self.job},{},{},self.expected)

class ReadEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.f=Fixture(self.temp.name)
    def test_complete_real_file_fixture_accepted(self):
        rows,meta=self.f.read();self.assertEqual(len(rows),960);self.assertAlmostEqual(meta['cost_seconds'],.03)
    def test_frozen_family_paths_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            f=Fixture(d,'frozen_single');self.assertEqual(len(f.read()[0]),960)
    def test_wrong_model_family_rejected(self):
        f=self.f;f.args['no_aug_checkpoint']=str(Path(self.temp.name)/'other.pt');f.save()
        with self.assertRaisesRegex(ValueError,'family_or_argv'):f.read()
    def test_argv_and_protocol_joint_wrong_family_rejected(self):
        f=self.f;f.args['aug_checkpoint']=str(Path(self.temp.name)/'other.pt');f.job['argv']=f.argv(f.args);f.save()
        with self.assertRaisesRegex(ValueError,'family_or_argv'):f.read()
    def test_seed_changed_rejected(self):
        self.f.args['seed']=211;self.f.save()
        with self.assertRaisesRegex(ValueError,'role_mismatch'):self.f.read()
    def test_shared_stress_omission_rejected(self):
        f=self.f;f.rows=[r for r in f.rows if r['variant']=='observed' or r['query_order']==0];f.save()
        with self.assertRaisesRegex(ValueError,'identity_or_variant'):f.read()
    def test_mask_changed_rejected(self):
        self.f.rows[0]['mask_content_sha256']='wrong';self.f.save()
        with self.assertRaisesRegex(ValueError,'identity_or_variant'):self.f.read()
    def test_identity_changed_rejected(self):
        self.f.rows[0]['record_id']='replacement';self.f.save()
        with self.assertRaisesRegex(ValueError,'identity_or_variant'):self.f.read()
    def test_unknown_source_with_consistent_hashes_rejected(self):
        f=self.f
        for r in f.rows:
            if r['source']==sorted(m.SOURCES)[0]:r['source']='unknown'
        f.refresh_identity();f.save()
        with self.assertRaisesRegex(ValueError,'five_sources'):f.read()
    def test_wrong_candidate_library_rejected(self):
        f=self.f;f.cache_summary['candidate_order_sha256']='wrong';f.summary['candidate_order_sha256']='wrong';f.save()
        with self.assertRaisesRegex(ValueError,'library_changed'):f.read()
    def test_checkpoint_summary_sha_mismatch_rejected(self):
        self.f.summary['checkpoints'][self.f.expected['aug_checkpoint']]='wrong';self.f.save()
        with self.assertRaisesRegex(ValueError,'checkpoint_summary'):self.f.read()
    def test_cache_sha_changed_rejected(self):
        self.f.cache.write_text('changed')
        with self.assertRaisesRegex(ValueError,'input_sha_changed'):self.f.read()
    def test_order_changed_rejected(self):
        self.f.rows.reverse();self.f.save()
        with self.assertRaisesRegex(ValueError,'order_changed'):self.f.read()
    def test_duplicate_candidate_rejected(self):
        self.f.rows[0]['natural_candidate_indices'][1]=0;self.f.save()
        with self.assertRaisesRegex(ValueError,'natural_pool'):self.f.read()

class DriverFixture:
    def __init__(self,tmp):
        self.root=Path(tmp).resolve();self.run=self.root/'run';self.run.mkdir()
        self.queue_path=self.root/'queue.json';self.source=self.root/'source.py';self.source.write_text('# synthetic source')
        self.sources={str(self.source):m.sha(self.source)}
        self.jobs=[]
        for i in range(48):
            out=self.root/('job'+str(i));out.mkdir();(out/'EXIT_CODE').write_text('0\n')
            self.jobs.append(dict(id='j'+str(i),kind='training' if i<16 else 'validation',
                argv=['synthetic-job',str(i)],cwd=str(self.root),out=str(out),inputs_sha256={}))
        write(self.queue_path,dict(source_inputs_sha256=self.sources,jobs=self.jobs))
        self.qpin={str(self.queue_path):m.sha(self.queue_path)}
        self.launch=dict(child_pid=99,argv=['python','-m','nmr_transfer.h_research_driver','--queue',str(self.queue_path)],external_exit=0,internal_exit=0,started_utc='2026-10-23T05:50:00Z',
            finished_utc='2026-10-23T06:00:00Z',queue_sha256=self.qpin,source_sha256=self.sources,bundle_sha256=self.sources)
        self.receipts={j['id']:dict(state='complete',external_exit=0,internal_exit=0,started_utc='2026-10-23T05:51:00Z',
            finished_utc='2026-10-23T05:59:00Z',pid=100+i,argv=j['argv'],output=j['out'],cwd=j['cwd'],
            queue_source_sha256={**self.sources,**self.qpin},requested_inputs_sha256={}) for i,j in enumerate(self.jobs)}
        self.save()
    def save(self):
        (self.run/'EXIT_CODE').write_text('0\n');(self.run/'EXTERNAL_EXIT_CODE').write_text('0\n')
        write(self.run/'launch_receipt.json',self.launch)
        write(self.run/'driver_start.json',dict(pid=99,argv=['driver.py','--queue',str(self.queue_path)],utc='2026-10-23T05:50:30Z',inputs_sha256={**self.sources,**self.qpin}))
        for jid,r in self.receipts.items():write(self.run/(jid+'.receipt.json'),r)
        write(self.run/'completion.json',dict(success=True,inputs_unchanged=True,utc='2026-10-23T05:59:30Z',jobs=self.receipts))
    def verify(self):return m.verify_run(self.run,'formal',{},{})

class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.f=DriverFixture(self.temp.name)
    def test_complete_driver_schema_accepted(self):self.assertEqual(len(self.f.verify()),48)
    def test_nonzero_external_file_rejected(self):
        (self.f.run/'EXTERNAL_EXIT_CODE').write_text('1')
        with self.assertRaisesRegex(ValueError,'nonzero'):self.f.verify()
    def test_pending_receipt_rejected(self):
        self.f.receipts['j0']['state']='running';self.f.save()
        with self.assertRaisesRegex(ValueError,'dual_exit0'):self.f.verify()
    def test_job_source_mismatch_rejected(self):
        self.f.receipts['j0']['queue_source_sha256']={};self.f.save()
        with self.assertRaisesRegex(ValueError,'source_or_cwd'):self.f.verify()
    def test_boolean_outer_exit_rejected(self):
        self.f.launch['external_exit']=False;self.f.save()
        with self.assertRaises(ValueError):self.f.verify()
    def test_boolean_job_exit_rejected(self):
        self.f.receipts['j0']['internal_exit']=False;self.f.save()
        with self.assertRaises(ValueError):self.f.verify()
    def test_invalid_finished_time_rejected(self):
        self.f.receipts['j0']['finished_utc']='not-a-UTC-time';self.f.save()
        with self.assertRaises(ValueError):self.f.verify()

    def test_finish_before_start_rejected(self):
        self.f.receipts['j0']['finished_utc']='2026-10-23T05:50:45Z';self.f.save()
        with self.assertRaisesRegex(ValueError,'time_order'):self.f.verify()
    def test_naive_timestamp_rejected(self):
        self.f.receipts['j0']['finished_utc']='2026-10-23T05:59:00';self.f.save()
        with self.assertRaisesRegex(ValueError,'completion_utc'):self.f.verify()
    def test_outer_future_time_rejected(self):
        self.f.launch['finished_utc']='2099-10-23T06:00:00Z';self.f.save()
        with self.assertRaisesRegex(ValueError,'time_order'):self.f.verify()
    def test_driver_pid_mismatch_rejected(self):
        self.f.launch['child_pid']=100;self.f.save()
        with self.assertRaisesRegex(ValueError,'launch_identity'):self.f.verify()

class ScientificInputTests(unittest.TestCase):
    def test_end_rehash_detects_drift(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input';p.write_text('begin');expected={str(p):m.sha(p)};verified={}
            m.require_sha(expected,verified);p.write_text('end')
            with self.assertRaisesRegex(ValueError,'input_sha_changed'):m.require_sha(verified,{})

class LegacyTests(unittest.TestCase):
    def test_actual_legacy_schema_and_unknown_recall(self):
        legacy=load('review_actual_legacy',LEGACY)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/'outputs/nmr_pool_ranking_20261023'
            old=root.parent/'nmr_global_retrieval_20261022';old.mkdir(parents=True)
            (old/'build_validation_manifest.py').write_bytes(LEGACY.read_bytes())
            freeze=root.parent/'nmr_global_retrieval_20261020/freeze_20261020_2000';freeze.mkdir(parents=True)
            now=dt.datetime.now(dt.timezone.utc)
            started=(now-dt.timedelta(hours=3)).isoformat();frozen=(now-dt.timedelta(hours=2)).isoformat();ended=(now-dt.timedelta(hours=1)).isoformat()
            reference={c:make_rows(c,320 if c=='original_val' else 1024) for c in m.COHORTS}
            pins={};pairs=[];owners=[]
            for seed in m.SEEDS:
                owner='s'+str(seed);owners.append(dict(id=owner,seed=seed));parent=old/owner;parent.mkdir()
                pair=dict(owner=owner,dual_route_dir=str(parent))
                for role in ('no_aug','aug'):
                    ck=parent/(role+'.pt');ck.write_text(role);pair[role+'_checkpoint']=str(ck);pins[str(ck)]=legacy.sha(ck)
                for cohort in m.COHORTS:
                    out=parent/cohort;out.mkdir();blend=out/'blend'
                    (out/'EXIT_CODE').write_text('0');pins[str(out/'EXIT_CODE')]=legacy.sha(out/'EXIT_CODE')
                    endpoints={}
                    for filename,gamma,variant in [('predictions.jsonl',.25,'observed'),('missing_predictions.jsonl',.75,'missing')]:
                        part=[dict(r) for r in reference[cohort] if (r['variant']=='observed')==(variant=='observed')]
                        for row in part:
                            row.pop('target_in_coarse_pool',None)
                        path=out/filename;path.write_text(''.join(json.dumps(r)+'\n' for r in part))
                        endpoint=blend/('gamma_'+str(gamma))/filename;endpoint.parent.mkdir(parents=True,exist_ok=True);endpoint.write_bytes(path.read_bytes())
                        pins[str(endpoint)]=legacy.sha(endpoint);endpoints[str(endpoint)]=pins[str(endpoint)]
                    summary=dict(cohort=cohort,normal_gamma=.25,missing_gamma=.75,candidate_order_sha256=m.CANDIDATE_SHA,
                        blend_dir=str(blend),inputs_sha256=endpoints)
                    for role in ('no_aug','aug'):
                        summary[role+'_checkpoint']=pair[role+'_checkpoint'];summary[role+'_sha256']=pins[pair[role+'_checkpoint']]
                    write(out/'summary.json',summary);pins[str(out/'summary.json')]=legacy.sha(out/'summary.json')
                pairs.append(pair)
            baseline=dict(fragment=dict(primary_kind='dual_route',pairs=pairs,constant_gamma=.25,gamma=.75),owners=owners)
            write(old/'baseline_inputs.json',baseline)
            write(freeze/'freeze_bundle.json',dict(selection_scope='train_val_only',frozen_utc=frozen))
            bundle_sha=legacy.sha(freeze/'freeze_bundle.json')
            write(freeze/'builder_receipt.json',dict(audit='PASS',bundle_sha256=bundle_sha,start_and_end_inputs_sha256=pins))
            receipt_sha=legacy.sha(freeze/'builder_receipt.json')
            (freeze/'EXIT_CODE').write_text('0')
            write(freeze.parent/'finalize_actual_receipt.json',dict(actual_exit_code=0,bundle_sha256=bundle_sha,started_utc=started,ended_utc=ended))
            finalize_sha=legacy.sha(freeze.parent/'finalize_actual_receipt.json');(freeze.parent/'finalize_actual_EXIT_CODE').write_text('0')
            real=legacy.legacy_entries
            def substitute(baseline,freeze_dir,cutoff,snapshot,expected_bundle,expected_receipt,expected_finalize):
                self.assertEqual((expected_bundle,expected_receipt,expected_finalize),(legacy.LEGACY_BUNDLE_SHA,legacy.LEGACY_RECEIPT_SHA,legacy.LEGACY_FINALIZE_SHA))
                return real(baseline,freeze_dir,cutoff,snapshot,bundle_sha,receipt_sha,finalize_sha)
            fake_spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None))
            with patch.object(legacy,'legacy_entries',substitute),patch.object(m.importlib.util,'spec_from_file_location',return_value=fake_spec),patch.object(m.importlib.util,'module_from_spec',return_value=legacy):
                table,result=m.historical_incumbent(root,reference,{},{} )
            self.assertFalse(result['selection_eligible']);self.assertIsNone(result['signal']['cost_seconds'])
            self.assertEqual(result['signal']['MRR'],1.)
            self.assertTrue(all(r.get('recall512') is None and r.get('in_pool_MRR') is None for r in table))

if __name__=='__main__':unittest.main()

"""功能概述：生产sealed统计gate的独立执行阶段合成fixture，不读取任何实际test。
ExecutionFixture把真实operator/isolation/launcher源码复制到临时根，启动12个纯JSON
人工child，每个只读3个假tensor及/test/producer summary和EXIT，再产生人工1371行。
真实OS PID、内外0、actualargv、UTC、始末SHA和访问日志均由真实临时进程生成。
metadata-only Pins禁止gate execution读任何.pt或预测，用于验证首payload前全12收据门禁。
本文件仅审execution；完整sealed_gate另与V临时metadata fixture联调，不伪称冻结已批准。
所有stage路径均在TemporaryDirectory内，不调用模型/训练/远端，不修改已锁科学源。
"""
import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_sealed_cap_results as g

HERE=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));return str(path)


CHILD='''"""功能概述：临时测试child，只读5个人工注册输入并输出人工JSON，不调用科学模型。"""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
p=argparse.ArgumentParser()
for name in ('source','protocol','bindings','frozen-selection','freeze-review','config','seed','cache','out','device','threads'):p.add_argument('--'+name,required=True)
a=p.parse_args();a.seed=int(a.seed);a.threads=int(a.threads)
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
b=json.loads(Path(a.bindings).read_text());f=json.loads(Path(a.frozen_selection).read_text())
e=next(x for x in b['entries'] if (x['config'],x['seed'])==(a.config,a.seed))
cache=b['caches'][str(a.seed)]['test'][e['pool']]
for path in [e['cache'],*[x['path'] for x in e['checkpoints'].values()],cache['producer']['path'],cache['producer']['exit_path']]:Path(path).read_bytes()
started=now();meta=dict(args=vars(a),argv=sys.argv,pid=os.getpid(),ppid=os.getppid(),started_utc=started)
(out/'protocol.json').write_text(json.dumps(meta))
pred=out/'predictions.jsonl'
with pred.open('x') as stream:
 for i in range(1371):
  variant='observed' if i<1000 else ('frequency_crop' if i<1200 else 'peak_dropout')
  q=dict(query_order=i,variant=variant,source='HMDB',record_id=str(i),smiles='C',group='g'+str(i%2))
  rank=512
  stream.write(json.dumps(dict(query_identity=q,rank=rank,RR=1/rank,target_in_pool=True,in_pool_target_rank=rank,cached_outside_pool_rank=None,top1=False,top5=False,top10=False,natural_candidate_indices=list(range(512)),natural_candidate_costs=[0.]*512,winner_candidate_index=0,target_candidate_index=1,target_cost=0.,margin=0.,mask_content_sha256='a'*64))+'\\n')
checks={x['path']:x['sha256'] for x in e['checkpoints'].values()}
inputs=dict(b['input_sha256'],**b['source_sha256'])
for path in (a.protocol,a.bindings,a.frozen_selection,a.freeze_review):inputs[str(Path(path).resolve())]=sha(path)
s=dict(schema='cap10_sealed_evaluation_20261024',status='PASS',inputs_unchanged=True,model_state_before='unchanged',model_state_after='unchanged',config=a.config,seed=a.seed,pool=e['pool'],cohort='test',records=1371,pid=os.getpid(),ppid=os.getppid(),started_utc=started,finished_utc=now(),test_read=True,training=False,selection_performed=False,promotion=False,external_api_calls=0,source_sha256=b['source_sha256'],cache_sha256=e['cache_sha256'],candidate_order_sha256=b['candidate_order_sha256'],query_sha256=f['fixed_test_identity']['query_sha256'],checkpoints=checks,input_sha256=inputs,predictions_sha256=sha(pred))
rows=[json.loads(line) for line in pred.read_text().splitlines()]
ident=lambda r:hashlib.sha256(json.dumps(tuple(r['query_identity'][k] for k in ('query_order','variant','source','record_id','smiles','group')),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
s['identity_sha256']=hashlib.sha256('\\n'.join(sorted(ident(r) for r in rows)).encode()).hexdigest()
s['metrics']={v:dict(queries=n,MRR=1/512,top1=0.,top5=0.,top10=0.,recall512=1.,in_pool_MRR=1/512) for v,n in (('observed',1000),('frequency_crop',200),('peak_dropout',171))}
(out/'summary.json').write_text(json.dumps(s));(out/'EXIT_CODE').write_text('0\\n')
'''


class MetadataOnlyPins(g.c.Pins):
    def __init__(self):super().__init__(allow_test=True);self.payload_reads=[]
    def raw(self,path,expected=None):
        path=Path(path)
        if path.suffix in ('.pt','.pth','.ckpt') or path.name=='predictions.jsonl':
            self.payload_reads.append(str(path));raise AssertionError('execution_read_payload_before_gate')
        return super().raw(path,expected)
    def add(self,path,expected=None):
        path=Path(path)
        if path.suffix in ('.pt','.pth','.ckpt') or path.name=='predictions.jsonl':
            self.payload_reads.append(str(path));raise AssertionError('execution_hash_payload_before_gate')
        return super().add(path,expected)


class ExecutionFixture:
    def __init__(self,root):
        self.root=Path(root).resolve();self.source=self.root/'source';self.source.mkdir()
        for name in ('supervise_controlled_queue.py','isolation_entry.py','launch_controlled_stage.py'):
            shutil.copyfile(HERE/name,self.root/name)
        (self.root/'cap_ranker.py').write_text('# 人工测试占位；不导入科学模型\n')
        self.script=self.root/'cap_test_evaluate_ranker.py';self.script.write_text(CHILD)
        self.protocol=Path(put(self.root/'protocol.json',dict(source_checkout=str(self.source),budget=dict(maximum_stage_seconds=60))))
        self.caches={};inputs={}
        for seed in (193,211):
            self.caches[str(seed)]={'test':{}}
            for pool in ('single','dual'):
                parent=self.root/'caches'/str(seed)/pool/'test';parent.mkdir(parents=True)
                cache=parent/'features.pt';cache.write_bytes(b'SYNTHETIC_NOT_A_TORCH_TENSOR')
                producer=parent/'summary.json';producer.write_text(json.dumps(dict(feature_cache_sha256=digest(cache),cohort='test',recall=pool)))
                exitcode=parent/'EXIT_CODE';exitcode.write_text('0\n')
                self.caches[str(seed)]['test'][pool]=dict(path=str(cache),sha256=digest(cache),producer=dict(path=str(producer),sha256=digest(producer),exit_path=str(exitcode)))
                inputs.update({str(p):digest(p) for p in (cache,producer,exitcode)})
        self.entries=[]
        for config in g.matrix.CONFIGS:
            for seed in (193,211):
                pool=config.rsplit('_',1)[1];family=config.split('_')[0];weights={}
                for role in ('no_aug','aug'):
                    ck=self.root/'weights'/family/str(seed)/role/'final.pt';ck.parent.mkdir(parents=True,exist_ok=True)
                    if not ck.exists():ck.write_bytes(b'SYNTHETIC_NOT_A_CHECKPOINT')
                    weights[role]=dict(path=str(ck),sha256=digest(ck));inputs[str(ck)]=digest(ck)
                cache=self.caches[str(seed)]['test'][pool]
                self.entries.append(dict(config=config,seed=seed,pool=pool,execution='evaluate',cache=cache['path'],cache_sha256=cache['sha256'],checkpoints=weights,job_out=str(self.root/'evaluations'/config/('s'+str(seed))/'test')))
        self.frozen=dict(entries=self.entries,fixed_test_identity=dict(query_sha256='a'*64,identity_content_sha256='b'*64,variant_mask_content_sha256='c'*64))
        self.fp=Path(put(self.root/'frozen_selection.json',self.frozen));self.fr=Path(put(self.root/'freeze_review.json',dict(synthetic=True)))
        inputs.update({str(p):digest(p) for p in (self.protocol,self.fp,self.fr)})
        scientific_sources={str(self.root/n):digest(self.root/n) for n in ('supervise_controlled_queue.py','isolation_entry.py','launch_controlled_stage.py','cap_ranker.py','cap_test_evaluate_ranker.py')}
        self.bindings=dict(source_checkout=str(self.source),input_sha256=inputs,source_sha256=scientific_sources,entries=self.entries,caches=self.caches,candidate_order_sha256=g.c.ORDER_SHA)
        self.bp=Path(put(self.root/'test_bindings.json',self.bindings))
        sources=dict(scientific_sources);sources.update({str(p):digest(p) for p in (self.protocol,self.bp)})
        self.run=self.root/'runs'/'test_synthetic_success';jobs=[]
        for number,entry in enumerate(self.entries):
            cache=self.caches[str(entry['seed'])]['test'][entry['pool']]
            reads=[entry['cache'],*[x['path'] for x in entry['checkpoints'].values()],cache['producer']['path'],cache['producer']['exit_path']]
            command=[str(self.script)]
            for name,value in dict(source=str(self.source),protocol=str(self.protocol),bindings=str(self.bp),frozen_selection=str(self.fp),freeze_review=str(self.fr),config=entry['config'],seed=entry['seed'],cache=entry['cache'],out=entry['job_out'],device='cpu',threads=1).items():command+=['--'+name.replace('_','-'),str(value)]
            access=str(self.root/'access_logs'/('job'+str(number)+'.json'))
            argv=[sys.executable,'-B',str(self.root/'isolation_entry.py'),'--bindings',str(self.bp),'--phase','test','--access-log',access,'--output-root',entry['job_out']]
            for path in reads:argv+=['--allow-read',path]
            jobs.append(dict(id='job'+str(number),command=command,argv=argv+['--']+command,allowed_reads=reads,access_log=access,job_out=entry['job_out'],cwd=str(self.source),gpu=None))
        self.qp=self.root/'test_queue.json';review=self.root/'test_admission.json'
        self.queue=dict(schema='cap10_controlled_queue_20261024',phase='test',root=str(self.root),local_root=str(self.root),run_name=self.run.name,python=sys.executable,bindings=str(self.bp),selection_locked_before_test=True,input_sha256=inputs,source_sha256=sources,review_paths=[str(review)],jobs=jobs,max_parallel=4,maximum_seconds=60)
        put(self.qp,self.queue);put(review,dict(status='PASS',independent=True,admission_phase='test',reviewed_sha256={**sources,str(self.qp):digest(self.qp)}))
        self.args=argparse.Namespace(queue=str(self.qp),run=str(self.run),source=str(self.source),bindings=str(self.bp),protocol=str(self.protocol),frozen_selection=str(self.fp),freeze_review=str(self.fr))

    def launch(self):
        process=subprocess.run([sys.executable,'-B',str(self.root/'launch_controlled_stage.py'),'--queue',str(self.qp)],capture_output=True,text=True,timeout=30)
        if process.returncode:raise AssertionError('synthetic_operator_failed:'+process.stdout[-2000:]+process.stderr[-2000:])
        self.snapshot={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        return self

    def restore(self):
        for path,data in self.snapshot.items():path.write_bytes(data)

    def execute(self):
        pins=MetadataOnlyPins()
        value=g.execution(pins,self.args,self.bindings,self.frozen)
        return value,pins


class ExecutionIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.f=ExecutionFixture(cls.temp.name).launch()
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def tearDown(self):self.f.restore()

    def test_real12_tmp_children_and_test_producer5_reads_admit_without_payload_read(self):
        (queue,values,refs),pins=self.f.execute()
        self.assertEqual(len(values),12);self.assertEqual(len(refs),12);self.assertEqual(pins.payload_reads,[])
        for job in queue['jobs']:
            log=json.loads(Path(job['access_log']).read_text())
            self.assertEqual(set(log['allowed_reads']),set(job['allowed_reads']))
            self.assertEqual(len(log['allowed_reads']),5)
            self.assertTrue(any('/test/summary.json' in x for x in log['recorded_reads']))
            self.assertTrue(any('/test/EXIT_CODE' in x for x in log['recorded_reads']))

    def test_bool_outer_exit_rejected_before_payload(self):
        p=self.f.run/'launch_receipt.json';x=json.loads(p.read_text());x['internal_exit']=False;put(p,x)
        with self.assertRaisesRegex(ValueError,'strict_int0'):self.f.execute()

    def test_unrelated_driver_start_argv_rejected_before_payload(self):
        p=self.f.run/'driver_start.json';x=json.loads(p.read_text());x['argv']=['unrelated.py'];put(p,x)
        with self.assertRaises(ValueError):self.f.execute()

    def test_changed_job_started_command_or_pin_rejected_before_payload(self):
        job=self.f.queue['jobs'][0];p=self.f.run/'jobs'/(job['id']+'.started.json');x=json.loads(p.read_text());x['argv']=['unrelated.py'];put(p,x)
        with self.assertRaises(ValueError):self.f.execute()

    def test_access_UTC_outside_actual_job_rejected_before_payload(self):
        job=self.f.queue['jobs'][0];p=Path(job['access_log']);x=json.loads(p.read_text());x['started_utc']='2000-01-01T00:00:00+00:00';put(p,x)
        with self.assertRaises(ValueError):self.f.execute()

    def test_unregistered_recorded_tensor_rejected_before_payload(self):
        p=Path(self.f.queue['jobs'][0]['access_log']);x=json.loads(p.read_text());x['recorded_reads'][str(self.f.root/'unregistered.pt')]=1;put(p,x)
        with self.assertRaises(ValueError):self.f.execute()

    def test_same_basename_unregistered_script_with_synchronized_receipts_rejected(self):
        q=copy.deepcopy(self.f.queue);job=q['jobs'][0]
        alternate=self.f.root/'unregistered'/'cap_test_evaluate_ranker.py';alternate.parent.mkdir(exist_ok=True);alternate.write_text('# unrelated same-name implementation\n')
        old=job['command'][0];job['command'][0]=str(alternate)
        job['argv']=[str(alternate) if a==old else a for a in job['argv']]
        q['source_sha256'][str(alternate)]=digest(alternate);put(self.f.qp,q);qh=digest(self.f.qp)
        admission=q['review_paths'][0];put(admission,dict(status='PASS',independent=True,admission_phase='test',reviewed_sha256={**q['source_sha256'],str(self.f.qp):qh}))
        for name in ('driver_start.json','completion.json','launch_receipt.json'):
            path=self.f.run/name;value=json.loads(path.read_text());value['queue_sha256']=qh
            if 'source_sha256' in value:value['source_sha256']=q['source_sha256']
            put(path,value)
        done=json.loads((self.f.run/'completion.json').read_text())
        for j in q['jobs']:
            for suffix in ('.started.json','.receipt.json'):
                path=self.f.run/'jobs'/(j['id']+suffix);value=json.loads(path.read_text());value['queue_sha256']=qh;value['argv']=j['argv'];put(path,value)
            rec=json.loads((self.f.run/'jobs'/(j['id']+'.receipt.json')).read_text())
            done['jobs']=[rec if r['id']==j['id'] else r for r in done['jobs']]
        put(self.f.run/'completion.json',done)
        access=Path(job['access_log']);value=json.loads(access.read_text());value['argv']=job['command'];put(access,value)
        fingerprint={**q['source_sha256'],str(self.f.qp):qh,str(Path(admission).resolve()):digest(admission)}
        for name in ('operator_fingerprints.before.json','operator_fingerprints.after.json'):put(self.f.run/name,fingerprint)
        with self.assertRaisesRegex(ValueError,'registered_native_evaluator|evaluator.*source|exact.*evaluator'):self.f.execute()


def full_metadata(f):
    """把同一真实12child fixture补成完整冻结metadata，不构造任何真实模型。"""
    proto=json.loads(f.protocol.read_text())
    proto.update(schema='cap10_controlled_protocol_20261024',seeds=[193,211],caps=[5,10],pool_size=512,
      candidate_library_size=114982,promotion=False,test_allowed=False,final_epoch=64,alpha=1.,epoch_offset=0,
      only_eligible_new_config='cap10_dual',control_config='cap5_dual',preselected=g.c.BASELINE,
      test_preregistration={'synthetic_math_already_independently_reviewed':True})
    put(f.protocol,proto)
    for seed,pools in f.caches.items():
        for pool,reg in pools['test'].items():
            manifest=dict(cohort='test',recall=pool,candidate_count=114982,pool_size=512,
              candidate_order_sha256=g.c.ORDER_SHA,records=1371,base_queries=1000,variants=g.c.TEST_COUNTS,
              **f.frozen['fixed_test_identity'])
            reg['manifest']=manifest;put(reg['producer']['path'],dict(manifest,feature_cache_sha256=reg['sha256']))
            reg['producer']['sha256']=digest(reg['producer']['path'])
    for entry in f.entries:
        for ref in entry['checkpoints'].values():
            folder=Path(ref['path']).parent;ref.update(epoch=64,alpha=1.)
            for key,name in [('centers','centers.json'),('center_rows','center_rows.jsonl'),('producer_summary','summary.json')]:
                path=folder/name;path.write_text('{}\n');ref[key]=str(path);ref[key+'_sha256']=digest(path)
                f.bindings['input_sha256'][str(path)]=digest(path)
            path=folder/'EXIT_CODE';path.write_text('0\n');f.bindings['input_sha256'][str(path)]=digest(path)
    for pools in f.caches.values():
        for reg in pools['test'].values():f.bindings['input_sha256'][reg['producer']['path']]=digest(reg['producer']['path'])
    f.bindings['input_sha256'][str(f.protocol)]=digest(f.protocol)
    f.bindings['schema']='cap10_test_bindings_20261024'
    fake_module=f.root/'cap_ranker.py'
    for key in ('cap_module','fixed_module'):f.bindings[key]=dict(path=str(fake_module),sha256=digest(fake_module))
    # 自身frozen/review不能循环放进bindings；专项review单独覆盖最终字节。
    for path in (f.fp,f.fr):f.bindings['input_sha256'].pop(str(path),None)
    f.frozen.update(schema='cap10_sealed_selection_20261024',selection_locked=True,promotion=False,preselected=g.c.BASELINE)
    refs={
      'test_plan':dict(entries=f.entries,preselected=g.c.BASELINE,selection_locked=True,promotion=False,selection_performed=False),
      'validation_summary':dict(status='PASS',test_read=False,promotion=False,preselected=g.c.BASELINE),
      'validation_selection':dict(preselected=g.c.BASELINE,selection_locked=True)}
    for key,value in refs.items():
        path=f.root/(key+'.json');put(path,value);f.frozen[key]=dict(path=str(path),sha256=digest(path))
    for key,target,scope in [('validation_review','validation_summary','validation_results'),('freeze_review','test_plan','test_plan_implementation')]:
        path=f.root/('plan_review.json' if key=='freeze_review' else key+'.json');ref=f.frozen[target]
        put(path,dict(status='PASS',independent=True,reviewer='synthetic independent',scope=scope,reviewed_sha256={ref['path']:ref['sha256']}))
        f.frozen[key]=dict(path=str(path),sha256=digest(path))
    own=[Path(g.__file__),Path(g.c.__file__),Path(g.pm.__file__),Path(g.matrix.__file__)]
    f.frozen['input_sha256']=dict(f.bindings['input_sha256'])
    f.frozen['source_sha256']={**f.bindings['source_sha256'],**{str(p.resolve()):digest(p) for p in own}}
    put(f.fp,f.frozen);put(f.bp,f.bindings)
    required={**f.frozen['input_sha256'],**f.frozen['source_sha256'],str(f.protocol):digest(f.protocol),str(f.bp):digest(f.bp),str(f.fp):digest(f.fp)}
    required.update({r['path']:r['sha256'] for r in (f.frozen[k] for k in refs.keys()|{'validation_review','freeze_review'})})
    put(f.fr,dict(status='PASS',independent=True,reviewer='synthetic final independent',scope='test_freeze_implementation',reviewed_sha256=required))
    ar=f.root/'analysis_review.json';put(ar,dict(status='PASS',independent=True,reviewer='synthetic analysis independent',scope='analysis_implementation',reviewed_sha256={str(p.resolve()):digest(p) for p in own}))
    f.args.analysis_review=str(ar)
    f.queue['input_sha256']={**f.bindings['input_sha256']}
    f.queue['source_sha256']={**f.frozen['source_sha256'],str(f.protocol):digest(f.protocol),str(f.bp):digest(f.bp)}
    put(f.qp,f.queue)
    put(f.queue['review_paths'][0],dict(status='PASS',independent=True,admission_phase='test',reviewed_sha256={**f.queue['source_sha256'],str(f.qp):digest(f.qp)}))
    return f


class FullEntryIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.f=full_metadata(ExecutionFixture(cls.temp.name)).launch()
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def tearDown(self):self.f.restore()

    def test_complete_frozen_gate_actual12_receipts_and_raw_oracle(self):
        seen=[];original=g.c.Pins.raw
        def read(pins,path,expected=None):
            if Path(path).name=='predictions.jsonl':seen.append(str(path))
            return original(pins,path,expected)
        def math(raw,prereg):
            self.assertEqual(set(raw),set(g.matrix.CONFIGS))
            self.assertTrue(all(len(rows)==1371 for seeds in raw.values() for rows in seeds.values()))
            return {'metrics':[dict(synthetic=True)],'synthetic_only':True}
        out=self.f.root/'stats_out';out.mkdir(exist_ok=True)
        with patch.object(g.c.Pins,'raw',read),patch.object(g.matrix,'analyze_matrix',side_effect=math):
            result=g.run(self.f.args,out)
        self.assertEqual(len(seen),12);self.assertFalse(result['selection_performed']);self.assertTrue(result['inputs_unchanged'])

    def test_last_job_bad_receipt_prevents_first_prediction_read(self):
        job=self.f.queue['jobs'][-1];path=Path(job['access_log']);v=json.loads(path.read_text());v['internal_exit']=False;put(path,v)
        original=g.c.Pins.raw;seen=[]
        def read(pins,path,expected=None):
            if Path(path).name=='predictions.jsonl':seen.append(str(path));raise AssertionError('payload_before_all12')
            return original(pins,path,expected)
        with patch.object(g.c.Pins,'raw',read),self.assertRaisesRegex(ValueError,'strict_int0'):
            g.run(self.f.args,self.f.root/'unused')
        self.assertEqual(seen,[])

    def test_final_review_missing_actual_statistics_source_refused_before_payload(self):
        value=json.loads(self.f.fr.read_text());value['reviewed_sha256'].pop(str(Path(g.__file__).resolve()));put(self.f.fr,value)
        with self.assertRaisesRegex(ValueError,'complete_exact_SHA'):
            g.run(self.f.args,self.f.root/'unused')

    def test_payload_selfconsistent_rank_lie_rejected_by_cost_oracle(self):
        out=Path(self.f.entries[0]['job_out']);rows=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
        rows[0].update(rank=511,RR=1/511,in_pool_target_rank=511)
        with self.assertRaisesRegex(ValueError,'native_rank_from_fp32_costs'):
            g.verify_leaf_values(rows,json.loads((out/'summary.json').read_text()))

    def test_summary_mrr_lie_rejected_by_independent_query_reduction(self):
        out=Path(self.f.entries[0]['job_out']);rows=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
        summary=json.loads((out/'summary.json').read_text());summary['metrics']['observed']['MRR']=.5
        with self.assertRaisesRegex(ValueError,'leaf_metric_from_actual_predictions_observed_MRR'):g.verify_leaf_values(rows,summary)

    def test_query_identity_summary_sha_mismatch_rejected(self):
        out=Path(self.f.entries[0]['job_out']);rows=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
        rows[0]['query_identity']['record_id']='changed_identity'
        with self.assertRaisesRegex(ValueError,'leaf_actual_identity_SHA'):
            g.verify_leaf_values(rows,json.loads((out/'summary.json').read_text()))


class ActualSyntheticProducerInteroperationTests(unittest.TestCase):
    def test_actual_admitted_evaluator_artificial_loader_rows_pass_cost_oracle(self):
        from test_cap_test_evaluate_ranker import SealedFixture
        with tempfile.TemporaryDirectory() as tmp:
            fixture=SealedFixture(tmp);summary=fixture.run()
            rows=[json.loads(line) for line in (fixture.out/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),6)
            for row in rows:g.validate_payload_row(row)
            self.assertEqual(summary['records'],6)


if __name__=='__main__':unittest.main()

"""功能概述：完整16validation+4fixed-train机制job的真实metadata reader入口合成贯通。
复用MatrixFixture与固定400/87标量，不读取真实cache/谱，不调用模型；每条保留512候选、
cost/trace、fixedother/role delta/scale，验证最终run确实重新核全固定pair和共享候选。
"""
import copy
import hashlib
import itertools
import json
import struct
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cap_evaluation_common as c
import summarize_cap_validation as s
import analyze_cap_mechanism as a
from test_analyze_cap_mechanism import pairs
from test_summarize_cap_validation import MatrixFixture,put,plain,T0,T1,T2,T3,T4

class MechanismFixture:
    def __init__(self,root):
        original,bycap=pairs();self.f=MatrixFixture(root,fixed_pairs=original);f=self.f
        self.run=f.root/'runs/mechanism';(self.run/'jobs').mkdir(parents=True)
        jobs=[];receipts=[]
        for cap,seed in itertools.product(c.CAPS,c.SEEDS):
            values=[]
            for entry in bycap[cap]:
                if entry['seed']!=seed:continue
                r=copy.deepcopy(entry);r.pop('seed');cost=[struct.unpack('f',struct.pack('f',v))[0] for v in r['natural_candidate_costs']];r['natural_candidate_costs']=cost;fp=r['fixed_train_pair'];fp['actual_fixed_pair_margin']=cost[2]-cost[10]
                coarse=[0.]*512;coarse[2]=fp['original_coarse_gap']
                r.update(in_pool_target_rank=r['rank'],cached_outside_pool_rank=None,target_cost=cost[2],
                    top1=r['rank']<=1,top5=r['rank']<=5,top10=r['rank']<=10,mask_content_sha256='a'*64,
                    reranking_seconds=.1,appended_training_positive=False,natural_candidate_coarse_costs=coarse,
                    natural_candidate_trace_sha256=[[hashlib.sha256(f'{cap}/{role}/{k}'.encode()).hexdigest() for k in range(512)] for role in range(2)])
                values.append(r)
            out=f.root/'mechanism_evals'/f'C{cap}_s{seed}';out.mkdir(parents=True)
            pred=out/'predictions.jsonl';pred.write_text(''.join(json.dumps(r)+'\n' for r in values))
            roles=f.roles[str(cap)][str(seed)];cache=f.caches[str(seed)]['train']['dual']
            inputs={**f.sourcepins,f.protocol:c.sha(f.protocol),f.bindings:c.sha(f.bindings),
                cache['path']:cache['sha256'],f.fixed_pairs:c.sha(f.fixed_pairs)}
            for role in roles.values():
                inputs.update({role['checkpoint']:role['checkpoint_sha256'],role['centers']:role['centers_sha256'],
                   role['center_rows']:role['center_rows_sha256'],role['producer_summary']:role['producer_summary_sha256']})
            args=dict(cap=cap,seed=seed,pool='dual',cohort='train',out=str(out),source=str(f.root),
                bindings=f.bindings,protocol=f.protocol,cache=cache['path'],query_manifest=f.fixed_pairs)
            command=[str((Path(__file__).parent.parent/'native_cost_evaluation/evaluate_cap_ranker.py'))]
            for k,v in args.items():command+=['--'+k.replace('_','-'),str(v)]
            pid=300+len(jobs);jid=f'mechanism{len(jobs)}';access=str(f.root/'access_logs'/(jid+'.json'))
            put(out/'protocol.json',dict(args=args,argv=command,pid=pid,ppid=111,started_utc=T1,input_sha256=inputs))
            put(out/'summary.json',dict(schema='cap_ranker_evaluation_20261024',status='PASS',scope='fixed_train_pairs_mechanism_only',
                cap=cap,seed=seed,pool='dual',cohort='train',inputs_unchanged=True,model_state_before='fixed',model_state_after='fixed',
                test_read=False,training=False,selection_performed=False,promotion=False,forward_bank_calls=0,external_api_calls=0,
                pid=pid,ppid=111,started_utc=T1,finished_utc=T2,input_sha256=inputs,source_sha256=f.sourcepins,
                predictions_sha256=c.sha(pred),records=200,identity_sha256=c.ordered_sha(sorted(c.identity_sha(r) for r in values)),
                metrics={v:c.metrics([r for r in values if r['query_identity']['variant']==v]) for v in c.VARIANTS},
                reranking_seconds_mean=.1,reranking_seconds_p95=.1,elapsed_seconds=20.1,cost_semantics='synthetic timing',
                cache_sha256=cache['sha256'],cache_manifest={k:v for k,v in cache['manifest'].items() if k!='feature_cache_sha256'},producer_manifest=cache['manifest'],checkpoints={r['checkpoint']:r['checkpoint_sha256'] for r in roles.values()}))
            plain(out/'EXIT_CODE','0\n')
            job=dict(id=jid,argv=[sys.executable,'-B',f.source,'--']+command,command=command,cwd=str(f.root),gpu=None,
                job_out=str(out),access_log=access,allowed_reads=[x for x in inputs if x!=f.bindings])
            record=dict(id=jid,argv=job['argv'],cwd=job['cwd'],gpu=None,pid=pid,parent_pid=111,started_utc=T1,finished_utc=T2,
                internal_exit=0,external_exit=0,job_out=str(out),access_log=access)
            put(access,dict(pid=pid,parent_pid=111,argv=command,internal_exit=0,rejected_paths=[],started_utc=T1,finished_utc=T2))
            jobs.append(job);receipts.append(record)
        review=str(f.root/'mechanism_admission.json')
        self.queue=put(f.root/'mechanism_queue.json',dict(schema='cap10_controlled_queue_20261024',phase='mechanism',python=sys.executable,
            local_root=str(f.root),bindings=f.bindings,review_paths=[review],source_sha256=f.stage_sources,input_sha256=f.inputpins,jobs=jobs))
        qsha=c.sha(self.queue)
        put(review,dict(status='PASS',independent=True,admission_phase='mechanism',reviewed_sha256={**f.stage_sources,self.queue:qsha}))
        for job,rec in zip(jobs,receipts):
            rec.update(queue_sha256=qsha,requested_inputs_sha256={p:f.inputpins[p] for p in job['allowed_reads']})
            put(self.run/'jobs'/(job['id']+'.receipt.json'),rec)
        argv=[sys.executable,'-B',str(f.root/'supervise_controlled_queue.py'),'--queue',self.queue,'--out',str(self.run),'--execute']
        put(self.run/'driver_start.json',dict(pid=111,parent_pid=110,argv=argv[2:],started_utc=T0,queue_sha256=qsha,source_sha256=f.stage_sources,input_sha256=f.inputpins))
        put(self.run/'completion.json',dict(status='COMPLETE',pid=111,started_utc=T0,finished_utc=T3,jobs=receipts,queue_sha256=qsha,external_exit=0,inputs_unchanged=True))
        put(self.run/'launch_receipt.json',dict(wrapper_pid=110,child_pid=111,argv=argv,started_utc=T0,finished_utc=T4,queue_sha256=qsha,
            source_sha256=f.stage_sources,internal_exit=0,external_exit=0))
        plain(self.run/'EXIT_CODE','0\n');plain(self.run/'EXTERNAL_EXIT_CODE','0\n')
        for when in ('before','after'):
            put(self.run/('input_fingerprints.'+when+'.json'),f.inputpins)
            put(self.run/('operator_fingerprints.'+when+'.json'),{**f.stage_sources,self.queue:qsha,review:c.sha(review)})
        self.out=f.root/'mechanism_analysis'
    def execute(self):
        f=self.f;self.out.mkdir()
        args=SimpleNamespace(protocol=f.protocol,bindings=f.bindings,validation_queue=f.queue,validation_run=str(f.run),
            mechanism_queue=self.queue,mechanism_run=str(self.run),out=str(self.out))
        with patch.dict(s.COHORT_OBSERVED,{'original_val':2,'tune':3}):
            return a.run(args,self.out)
class EntryTests(unittest.TestCase):
    def test_full_actual_reader_16plus4_entry_and_400pair_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MechanismFixture(tmp);r=f.execute()
            self.assertEqual(len(r['common_candidate_invariance']),8)
            self.assertEqual(r['fixed_train_pair_analysis']['original87']['total_bound_release_count'],{'5':0,'10':87})
            self.assertEqual(r['fixed_train_pair_analysis']['all400']['query_seed_pairs'],400)
            self.assertFalse(r['selection_input']);self.assertEqual(r['model_calls'],0)
    def test_incomplete_mechanism_stage_refused_before_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=MechanismFixture(tmp);p=f.run/'completion.json';x=json.loads(p.read_text());x['jobs'].pop();put(p,x)
            with self.assertRaisesRegex(ValueError,'complete_job_set'):f.execute()
if __name__=='__main__':unittest.main()

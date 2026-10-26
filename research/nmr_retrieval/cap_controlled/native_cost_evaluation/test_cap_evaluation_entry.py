"""功能概述：真实cap checkpoint loader→evaluate.main→read_evaluation的合成入口fixture。
仅把旧FixedCenterRanker runtime替换为明确FakeFixed，将人工tensor写入临时目录；
不训练、不读取真实缓存，不减弱生产final64/schema/role/center/source/producer检查。
EntryFixture可供独立review复用；所有输入和输出均标记为synthetic。
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import cap_evaluation_common as c
import evaluate_cap_ranker as e
from test_evaluate_cap_ranker import FakeFixed,cap_module,row

class EntryFixture:
    def __init__(self,root,cap=5,seed=193,cohort='original_val',pool='single',observed_count=2):
        self.root=Path(root).resolve();self.cap=cap;self.seed=seed;self.cohort=cohort;self.pool=pool
        self.module=cap_module();self.fixed=self.root/'fixed_source.py';self.fixed.write_text('# synthetic fixed runtime source\n')
        self.source=self.root/'source';self.source.mkdir();self.smiles=['synthetic'+str(i) for i in range(512)]
        self.order=c.ordered_sha(self.smiles)
        self.cache=self.root/'cache/features.pt';self.cache.parent.mkdir();self.cache.write_bytes(b'explicit synthetic cache bytes; fake reader')
        self.rows=[]
        base=0 if cohort=='original_val' else 2
        for i,variant in enumerate(['observed']*observed_count+['frequency_crop','peak_dropout']):
            r=row()
            r.update(query_order=i,variant=variant,source=c.SOURCES[(base+i)%5],record_id=cohort+str(i),group=cohort+'group'+str(i),
                candidate_indices=r['candidate_indices'][:512],features=r['features'][:512],coarse_scores=r['coarse_scores'][:512],
                predicted_ppm=r['predicted_ppm'][:512],tags=r['tags'][:512],H=r['H'][:512],K=r['K'][:512],
                negative_exclusion_mask=r['negative_exclusion_mask'][:512])
            self.rows.append(r)
        manifest=dict(schema='nmr_research_20261022',cohort=cohort,recall=pool,candidate_count=114982,pool_size=512,
          candidate_order_sha256=self.order,query_sha256='synthetic_query',checkpoint_sha256='f'*64,source_inputs_sha256={},identity_content_sha256='i'*64,variant_mask_content_sha256='v'*64,records=len(self.rows),base_queries=observed_count,
          variants=dict(observed=observed_count,frequency_crop=1,peak_dropout=1))
        self.data=dict(manifest=manifest,rows=self.rows,smiles=self.smiles)
        self.cacheproducer=self.cache.parent/'summary.json';self.cacheproducer.write_text(json.dumps({**manifest,'feature_cache_sha256':c.sha(self.cache)}))
        self.cacheexit=self.cache.parent/'EXIT_CODE';self.cacheexit.write_text('0\n')
        self.rt=SimpleNamespace(fixed=SimpleNamespace(FixedCenterRanker=FakeFixed),
          base=SimpleNamespace(validate_row=lambda r,training:None,check_science=lambda saved,data:None),
          source_paths=lambda:[Path(self.module.__file__).resolve(),self.fixed],
          read_feature_cache=lambda path,known_sha=None:(copy.deepcopy(self.data),known_sha))
        self.protocol=self.root/'protocol.json';self.protocol.write_text(json.dumps(dict(schema='cap10_controlled_protocol_20261024',
          seeds=[193,211],caps=[5,10],pool_size=512,candidate_library_size=114982,promotion=False,test_allowed=False,
          final_epoch=64,alpha=1.,epoch_offset=0,only_eligible_new_config='cap10_dual',control_config='cap5_dual',preselected=c.BASELINE)))
        sources=c.Pins()
        for q in self.rt.source_paths():sources.add(q)
        self.train={pool:dict(sha256=hashlib.sha256(('synthetic train '+pool).encode()).hexdigest()) for pool in ('single','dual')}
        self.roles={}
        for role in ('no_aug','aug'):
            folder=self.root/'weights'/role;folder.mkdir(parents=True)
            rowaudit=folder/'center_rows.jsonl';rowaudit.write_text('{"synthetic":true}\n')
            center=folder/'centers.json';center.write_text(json.dumps(dict(center33=[0.]*33,O_center3=[0.]*3,row_audit_sha256=c.sha(rowaudit))))
            model=self.module.CapRanker([.1],torch.zeros(33),torch.zeros(3),cap,self.rt)
            scale=torch.ones(33);checkpoint=folder/'final.pt'
            saved=dict(schema=self.module.SCHEMA,observation=True,delta_cap=cap,budget=self.module.BUDGET,
              negative_mask_policy=self.module.MASK_POLICY,seed=seed,role=role,args=dict(seed=seed,augment=role=='aug',cap=cap,training_pool='mixed',loss='listwise'),
              epoch=64,alpha=1.,draws=self.module.BUDGET['draws_per_fit'],optimizer_steps=self.module.BUDGET['optimizer_steps'],
              strict_negative_qualification_sha256='synthetic_qualification',parent_checkpoint_sha256='synthetic_parent',
              center_estimate_sha256=c.sha(center),center_rows_sha256=c.sha(rowaudit),
              ranker_source_sha256=sources.values,state_dict=model.state_dict(),scale=scale,initial_sigmas=[.1],
              train_sha256=self.train['single']['sha256'],dual_train_sha256=self.train['dual']['sha256'],cache_protocol={'synthetic':'training'},source_inputs_sha256={},
              frozen_tensor_sha256=self.module.frozen_digest(model,scale),exposure_sha256='exposure',
              negative_exposure_sha256='negative_exposure',pool_exposure_sha256='pool_exposure',input_sha256=sources.values)
            torch.save(saved,checkpoint)
            producer={k:v for k,v in saved.items() if k not in ('state_dict','scale','initial_sigmas')}
            producer.update(final_checkpoint_sha256=c.sha(checkpoint),source_inputs_start_sha256=sources.values,source_inputs_end_sha256=sources.values)
            summary=folder/'summary.json';summary.write_text(json.dumps(producer));(folder/'EXIT_CODE').write_text('0\n')
            self.roles[role]=dict(checkpoint=str(checkpoint),checkpoint_sha256=c.sha(checkpoint),centers=str(center),centers_sha256=c.sha(center),
               center_rows=str(rowaudit),center_rows_sha256=c.sha(rowaudit),producer_summary=str(summary),producer_summary_sha256=c.sha(summary))
        binding=dict(schema='cap10_execution_bindings_20261024',source_checkout=str(self.source),evidence_root=str(self.root),
          cap_module=dict(path=str(Path(self.module.__file__).resolve()),sha256=c.sha(self.module.__file__)),
          fixed_module=dict(path=str(self.fixed),sha256=c.sha(self.fixed)),source_sha256=sources.values,
          caches={str(seed):{'train':self.train,cohort:{pool:dict(path=str(self.cache),sha256=c.sha(self.cache),manifest=manifest,
             producer=dict(path=str(self.cacheproducer),sha256=c.sha(self.cacheproducer),exit_path=str(self.cacheexit)))}}},
          roles={str(cap):{str(seed):self.roles}})
        self.bindings=self.root/'validation_bindings.json';self.bindings.write_text(json.dumps(binding))
        self.out=self.root/'output'
        self.argv=[str(Path(e.__file__).resolve()),'--source',str(self.source),'--protocol',str(self.protocol),
          '--bindings',str(self.bindings),'--cache',str(self.cache),'--cap',str(cap),'--seed',str(seed),'--pool',pool,
          '--cohort',cohort,'--out',str(self.out),'--device','cpu','--threads','1']
    def run(self):
        with patch.object(e,'ORDER_SHA',self.order),patch.dict(e.COHORT_OBSERVED,{self.cohort:self.data['manifest']['base_queries']}),\
             patch.object(e,'load_module',return_value=self.module),patch.object(self.module,'configure_runtime',return_value=self.rt),\
             patch('sys.argv',self.argv):
            e.main()
        return json.loads((self.out/'summary.json').read_text())
class EntryTests(unittest.TestCase):
    def test_actual_cap_loader_to_main_to_read_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=EntryFixture(tmp);summary=f.run()
            value,protocol,rows=c.read_evaluation(c.Pins(),f.out,5,193,'single','original_val',f.bindings,f.protocol,True)
            self.assertEqual(summary['records'],4);self.assertEqual(len(rows),4)
            self.assertEqual((f.out/'EXIT_CODE').read_text(),'0\n')
            self.assertEqual(value['model_state_before'],value['model_state_after'])
    def test_wrong_actual_checkpoint_role_refused_and_failure_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=EntryFixture(tmp);b=json.loads(f.bindings.read_text());b['roles']['5']['193']['aug']=b['roles']['5']['193']['no_aug'];f.bindings.write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,'seed_role'):f.run()
            self.assertEqual((f.out/'EXIT_CODE').read_text(),'1\n');self.assertTrue((f.out/'failure.json').exists())
    def test_cache_producer_byte_sha_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=EntryFixture(tmp);f.cacheproducer.write_text('{}')
            with self.assertRaisesRegex(ValueError,'input_sha'):f.run()
            self.assertEqual((f.out/'EXIT_CODE').read_text(),'1\n')
    def test_rankmetrics_and_RR_tamper_refused_after_repin(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=EntryFixture(tmp);f.run();pred=f.out/'predictions.jsonl';rows=[json.loads(x) for x in pred.read_text().splitlines()]
            rows[0]['rank']+=1;rows[0]['RR']=1/rows[0]['rank'];rows[0]['in_pool_target_rank']=rows[0]['rank']
            pred.write_text(''.join(json.dumps(x)+'\n' for x in rows));s=json.loads((f.out/'summary.json').read_text());s['predictions_sha256']=c.sha(pred);(f.out/'summary.json').write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'rank_cost'):c.read_evaluation(c.Pins(),f.out,5,193,'single','original_val',f.bindings,f.protocol)
    def test_real_legacy_postsave_producer_SHA_superset_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=EntryFixture(tmp);b=json.loads(f.bindings.read_text())
            b['caches']['193']['original_val']['single']['manifest']['feature_cache_sha256']=c.sha(f.cache)
            f.bindings.write_text(json.dumps(b));result=f.run()
            self.assertNotIn('feature_cache_sha256',result['cache_manifest'])
            self.assertEqual(result['producer_manifest']['feature_cache_sha256'],c.sha(f.cache))

if __name__=='__main__':unittest.main()

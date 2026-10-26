"""验证汇总工具的合成测试：覆盖完整家族门禁、两seed合并权重、晋升守门、
成本并列规则、跨cohort重复、driver真实外部退出证据和测试cohort拒绝。
仅操作临时目录与合成数字，不读取实际验证/测试谱或启动训练。
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('summarizer',Path(__file__).with_name('summarize_pool_validation.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def signals():
    base=dict(MRR=.5,top1=.4,frequency_crop_MRR=.3,peak_dropout_MRR=.2,
              macro_MRR=.5,cost_seconds=1.)
    return {k:dict(base) for k in list(m.CONFIGS)+['frozen_single','frozen_dual']}


def rows(cohort,rank,count=5):
    return [dict(source='s'+str(i%5),record_id=cohort+str(i),smiles='C'+str(i),group='g'+str(i),
                 query_order=i,variant=v,rank=rank,target_in_coarse_pool=True)
            for i in range(count) for v in m.VARIANTS]


class SelectionTests(unittest.TestCase):
    def test_cache_identity_and_mask_digests_bind_all_variants(self):
        rr=rows('o',1)
        for i,row in enumerate(rr):row.update(source=sorted(m.SOURCES)[i//3],mask_content_sha256='mask'+str(i))
        rr.sort(key=lambda r:(r['query_order'],r['variant']))
        fields=('query_order','variant','source','record_id','smiles','group')
        cache=dict(records=len(rr),variants=dict(m.collections.Counter(r['variant'] for r in rr)),
            identity_content_sha256=m.content_sha([{k:r[k] for k in fields} for r in rr]),
            variant_mask_content_sha256=m.content_sha([(tuple(r[k] for k in fields[:-1]),r['mask_content_sha256']) for r in rr]))
        m.validate_rows_identity(rr,cache)
        for mutation in ('remove_stress','mask','source','identity','order'):
            bad=copy.deepcopy(rr)
            if mutation=='remove_stress':bad=[r for r in bad if r['variant']!='peak_dropout']
            elif mutation=='mask':bad[0]['mask_content_sha256']='altered'
            elif mutation=='source':bad[0]['source']='new_source'
            elif mutation=='identity':bad[0]['record_id']='wrong_record'
            else:bad.reverse()
            with self.assertRaises(ValueError,msg=mutation):m.validate_rows_identity(bad,cache)

    def test_queue_protocol_and_expected_family_must_match(self):
        expected=m.evaluation_args(Path('/example/outputs/nmr_pool_ranking_20261023'),m.CONFIGS[0],193,'tune')
        argv=['/python','-m','nmr_transfer.h_pool_ranker']
        for key,value in expected.items():argv.extend(['--'+key.replace('_','-'),str(value)])
        m.bind_evaluation(expected,{'argv':argv},expected)
        for key in ('cache','no_aug_checkpoint','aug_checkpoint','seed'):
            wrong=dict(expected);wrong[key]='wrong'
            with self.assertRaisesRegex(ValueError,'family_or_argv'):m.bind_evaluation(wrong,{'argv':argv},expected)
        wrong_argv=argv+['--seed','193']
        with self.assertRaisesRegex(ValueError,'duplicate'):m.bind_evaluation(expected,{'argv':wrong_argv},expected)

    def test_frozen_baseline_paths_are_direct_R0O1F0(self):
        args=m.evaluation_args(Path('/x/outputs/nmr_pool_ranking_20261023'),'frozen_single',211,'original_val')
        self.assertIn('/rankers/factor_R0O1F0/s211/no_aug/best.pt',args['no_aug_checkpoint'])
        self.assertIn('/features/inherited_s211/single/original_val/features.pt',args['cache'])

    def test_legacy_recall_remains_unknown(self):
        got=m.metrics([{'rank':1},{'rank':600}])
        self.assertIsNone(got['recall512']);self.assertIsNone(got['in_pool_MRR'])
        self.assertAlmostEqual(got['MRR'],(1+1/600)/2)

    def test_repeated_read_rejects_changed_content(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'value.json';p.write_text('{"value":1}');pins={}
            m.read_json(p,pins);p.write_text('{"value":2}')
            with self.assertRaisesRegex(ValueError,'changed_between_reads'):m.read_json(p,pins)

    def test_all_gates_required(self):
        s=signals();c=m.CONFIGS[0];s[c]['MRR']=.51
        self.assertEqual(m.choose(s)['preselected'],c)
        for key in ('top1','frequency_crop_MRR','peak_dropout_MRR','macro_MRR'):
            bad=copy.deepcopy(s);bad[c][key]-=.01
            self.assertIsNone(m.choose(bad)['preselected'])

    def test_equal_baseline_not_improvement(self):
        self.assertIsNone(m.choose(signals())['preselected'])

    def test_partial_family_rejected(self):
        s=signals();del s[m.CONFIGS[-1]]
        with self.assertRaisesRegex(ValueError,'incomplete'):m.choose(s)

    def test_unregistered_addition_rejected(self):
        s=signals();s['extra']=dict(s['frozen_single'])
        with self.assertRaisesRegex(ValueError,'unregistered'):m.choose(s)

    def test_tie_cost_then_name(self):
        s=signals()
        for c in m.CONFIGS:s[c]['MRR']=.51
        s[m.CONFIGS[2]]['cost_seconds']=.5
        self.assertEqual(m.choose(s)['preselected'],m.CONFIGS[2])
        s[m.CONFIGS[3]]['cost_seconds']=.5
        self.assertEqual(m.choose(s)['preselected'],min(m.CONFIGS[2:4]))

    def test_close_tie_uses_cost(self):
        s=signals();a,b=m.CONFIGS[:2]
        s[a]['MRR']=.51;s[b]['MRR']=.51+5e-13;s[a]['cost_seconds']=.5
        self.assertEqual(m.choose(s)['preselected'],a)

    def test_merge_by_query_then_equal_seed_average(self):
        d={'x':{seed:{'original_val':rows('o',1),'tune':rows('t',2,10)} for seed in m.SEEDS}}
        meta={'x':{seed:{c:{'cost_seconds':1} for c in m.COHORTS} for seed in m.SEEDS}}
        table,s=m.aggregate(d,meta)
        self.assertAlmostEqual(s['x']['MRR'],2/3)
        self.assertEqual(len([r for r in table if r['source']=='all']),18)

    def test_duplicate_across_cohort_rejected(self):
        d={'x':{seed:{'original_val':rows('o',1),'tune':rows('o',2)} for seed in m.SEEDS}}
        meta={'x':{seed:{c:{'cost_seconds':1} for c in m.COHORTS} for seed in m.SEEDS}}
        with self.assertRaisesRegex(ValueError,'duplicate'):m.aggregate(d,meta)

    def test_missing_seed_rejected(self):
        d={'x':{193:{'original_val':rows('o',1),'tune':rows('t',2)}}}
        meta={'x':{193:{c:{'cost_seconds':1} for c in m.COHORTS}}}
        with self.assertRaisesRegex(ValueError,'seed_family'):m.aggregate(d,meta)

    def test_pool_metrics_are_conditional(self):
        rr=[dict(rank=1,target_in_coarse_pool=True),dict(rank=600,target_in_coarse_pool=False)]
        got=m.metrics(rr)
        self.assertEqual(got['recall512'],.5);self.assertEqual(got['in_pool_MRR'],1)
        self.assertAlmostEqual(got['MRR'],(1+1/600)/2)

    def test_outer_exit_failure_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'EXIT_CODE').write_text('0');(p/'EXTERNAL_EXIT_CODE').write_text('1')
            with self.assertRaisesRegex(ValueError,'nonzero'):
                m.verify_run(p,'formal',{},{})

    def test_outer_receipt_pending_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'EXIT_CODE').write_text('0');(p/'EXTERNAL_EXIT_CODE').write_text('0')
            (p/'launch_receipt.json').write_text(json.dumps({'external_exit':None,'internal_exit':0}))
            with self.assertRaisesRegex(ValueError,'outer_not_complete'):
                m.verify_run(p,'formal',{},{})

    def test_test_cohort_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve()
            args={'stage':'evaluate','seed':211,'out':str(p)}
            (p/'protocol.json').write_text(json.dumps({'args':args,'inputs_sha256':{}}))
            (p/'summary.json').write_text(json.dumps({'cohort':'test','recall':'single'}))
            with self.assertRaisesRegex(ValueError,'test_or_wrong'):
                m.read_eval(p,211,'single','test',{str(p):{'kind':'validation'}},{},{})

    def test_conflicting_sha_rejected(self):
        with self.assertRaisesRegex(ValueError,'conflicting'):
            m.require_sha({'file':'a'},{'file':'b'})


if __name__=='__main__':unittest.main()

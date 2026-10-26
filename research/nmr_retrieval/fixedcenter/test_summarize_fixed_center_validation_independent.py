"""功能概述：联合验证汇总的独立算术与审计证据拒绝测试；仅临时合成数据。"""
import json
from pathlib import Path
import tempfile
import unittest
import summarize_fixed_center_validation as m
import test_summarize_fixed_center_validation as t


class IndependentTests(unittest.TestCase):
    def test_query_weight_then_equal_seed_and_five_source_macro(self):
        ranks={193:{'original_val':2,'tune':4},211:{'original_val':5,'tune':10}}
        data={c:{s:{cohort:t.rows(cohort,rank) for cohort,rank in ranks[s].items()} for s in m.v.SEEDS} for c in m.CONFIGS}
        meta={c:{s:{cohort:dict(cost_seconds=.02 if cohort=='original_val' else .04)
                    for cohort in m.v.COHORTS} for s in m.v.SEEDS} for c in m.CONFIGS}
        table,signals=m.v.aggregate(data,meta)
        expected=sum((320/ranks[s]['original_val']+1024/ranks[s]['tune'])/1344 for s in m.v.SEEDS)/2
        for c in m.CONFIGS:
            self.assertAlmostEqual(signals[c]['MRR'],expected,14)
            source_means=[]
            for source in sorted(m.v.SOURCES):
                means=[]
                for seed in m.v.SEEDS:
                    rr=[1/r['rank'] for cohort in m.v.COHORTS for r in data[c][seed][cohort]
                        if r['variant']=='observed' and r['source']==source]
                    means.append(sum(rr)/len(rr))
                source_means.append(sum(means)/2)
            self.assertAlmostEqual(signals[c]['macro_MRR'],sum(source_means)/5,14)
        self.assertEqual(len(table),216)
    def test_unreviewed_audit_implementation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=t.Fixture(tmp);review=json.loads(f.review.read_text())
            key=next(k for k in review['reviewed_sha256'] if Path(k).name=='audit_pool_fixed_center_training.py')
            review['reviewed_sha256'][key]='0'*64;f.review.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError,'audit_reviewed_source_binding'):f.run_summary()
    def test_missing_audited_center_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=t.Fixture(tmp);path=f.audit/'summary.json';audit=json.loads(path.read_text())
            del audit['input_sha256'][str(Path(audit['models'][0]['output'])/'centers.json')]
            path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError,'audit_center_binding'):f.run_summary()


if __name__=='__main__':unittest.main()

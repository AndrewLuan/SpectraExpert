"""功能概述：新reader只排顶层运行计时；原完整metadata/3616+12严格拒绝测试承接。
所有数据均为原fixture的人造记录。成本的最小可表示漂移、rank/trace/identity仍必须拒绝。
不改变原6a9源码或旧测试文件；只将原测试module的内存引用指向本轮新reader。
"""
import json,math,unittest
from pathlib import Path
import review_cap_validation_mechanism_results_timing as r
import test_review_cap_validation_mechanism_results as old
old.r=r
EvidenceTests=old.EvidenceTests

def alter(rows,fn):
    data=[json.loads(line) for line in rows.splitlines() if line]
    for row in data:fn(row)
    return b''.join(json.dumps(row).encode()+b'\n' for row in data)

class TimingTests(old.PriorSuccessSemanticTests):
    def test_top_level_wall_clock_measurement_may_differ_all3616(self):
        for path,raw in list(self.payload.items()):
            self.payload[path]=alter(raw,lambda row:row.update(reranking_seconds=1. if '/validation_native_cost/' in path else 2.))
        for cell in self.docs[self.args.resource_failure_review]['completed_cells']:
            cell['prediction_sha256']=r.digest(self.payload[str(self.root/'validation_native_cost'/cell['id']/'predictions.jsonl')])
        self.assertTrue(r.compare_prior_successes(self.e,self.root,self.args,self.jobs)['all_semantic_fields_exact'])
    def test_minimal_actual_cost_drift_still_rejected(self):
        path=str(Path(self.jobs[0]['job_out'])/'predictions.jsonl')
        self.payload[path]=alter(self.payload[path],lambda row:row['natural_candidate_costs'].__setitem__(0,math.nextafter(row['natural_candidate_costs'][0],math.inf)))
        with self.assertRaisesRegex(ValueError,'all_prior_completed_semantic_fields_exact'):r.compare_prior_successes(self.e,self.root,self.args,self.jobs)
    def test_nested_same_name_not_silently_excluded(self):
        path=str(Path(self.jobs[0]['job_out'])/'predictions.jsonl');self.payload[path]=alter(self.payload[path],lambda row:row['trace'].update(reranking_seconds=2.))
        with self.assertRaisesRegex(ValueError,'all_prior_completed_semantic_fields_exact'):r.compare_prior_successes(self.e,self.root,self.args,self.jobs)

class MechanismTimingTests(old.PriorMechanism12LineTests):
    def test_top_level_timing_diff_allowed_for_authorized12_lines(self):
        for path,raw in list(self.payload.items()):self.payload[path]=alter(raw,lambda row:row.update(reranking_seconds=1. if '/mechanism_resource_ready/' in path else 2.))
        for cell in self.docs[self.args.mechanism_failure_review]['cells']:
            cell['prediction_sha256']=r.digest(self.payload[str(self.root/'mechanism_resource_ready'/cell['id']/'predictions.jsonl')])
        self.assertEqual(self.check()['complete_lines_compared'],12)
    def test_mechanism_trace_science_change_remains_exact(self):
        path=str(Path(self.jobs[0]['job_out'])/'predictions.jsonl');self.payload[path]=alter(self.payload[path],lambda row:row['trace'].update(alpha=.999999999999))
        with self.assertRaisesRegex(ValueError,'prior12_mechanism_semantics_exact'):self.check()

if __name__=='__main__':unittest.main()

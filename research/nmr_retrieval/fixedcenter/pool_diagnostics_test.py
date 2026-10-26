"""功能：以合成验证行测试池迁移代数、身份门禁和自然512候选保护；不接触真实谱。"""
import unittest
from pool_diagnostics import decompose, optional_pool

class DiagnosticsTest(unittest.TestCase):
    def row(self,i,rank):
        return dict(query_order=i,variant='observed',source='s',record_id=i,group=str(i),rank=rank,
                    target_in_coarse_pool=rank<=512,evaluation_pool_size=512)
    def test_four_categories_sum(self):
        a=[self.row(i,r) for i,r in enumerate([2,600,3,700])]
        b=[self.row(i,r) for i,r in enumerate([1,5,800,900])]
        pairs,result=decompose(a,b)
        m=result['by_variant']['observed']
        self.assertEqual({p['category'] for p in pairs},{'common','new','lost','neither'})
        self.assertAlmostEqual(m['delta_MRR'],sum(1/y['rank']-1/x['rank'] for x,y in zip(a,b))/4)
        self.assertEqual(m['delta_recall512'],0)
    def test_reject_identity(self):
        with self.assertRaisesRegex(ValueError,'identity'):
            decompose([self.row(1,1)],[self.row(2,1)])
    def test_reject_duplicate(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            decompose([self.row(1,1)]*2,[self.row(1,1)]*2)
    def test_reject_appended_positive(self):
        with self.assertRaisesRegex(ValueError,'nonnatural'):
            optional_pool({'candidate_indices':list(range(513))},{'candidate_indices':list(range(512))})
    def test_common_rank_change(self):
        a={'candidate_indices':list(range(512)),'candidate_costs':list(range(512))}
        b={'candidate_indices':list(range(512)),'candidate_costs':list(reversed(range(512)))}
        self.assertEqual(optional_pool(a,b)['common_candidate_mean_abs_rank_change'],256)
    def test_mask_drift(self):
        a=dict(self.row(1,1),mask_content_sha256='a');b=dict(self.row(1,1),mask_content_sha256='b')
        with self.assertRaisesRegex(ValueError,'content'):decompose([a],[b])
if __name__=='__main__':unittest.main()

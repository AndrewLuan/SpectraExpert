"""功能概述：6配置固定test矩阵统计的作者纯合成测试，无真实数据读取。
fixture为1371个伪身份、两connectivity簇，检查全180比较/324指标、空strata、
seed条件比值与拒绝丢cell/身份/掩码/协议扩展；通过unittest执行，无模型调用。
"""
import unittest,copy
import sealed_cap_matrix as m
from test_sealed_cap_pairmath import row

def registration():
    return dict(matrix_configs=list(m.CONFIGS),seeds=[193,211],matrix_cells=12,
      variant_counts=dict(m.pm.VARIANTS),queries_per_cell=1371,candidate_library_size=114982,
      strata=['all',*m.pm.SOURCES],comparisons=[dict(id=i,left=l,right=r) for i,l,r in m.COMPARISONS],
      comparison_strata=180,uncertainty=dict(bootstrap_seed=317,replicates=5000,confidence_level=.95),
      test_reselection_allowed=False,additional_comparisons_after_results_allowed=False)

def fixture():
    matrix={}
    for c in m.CONFIGS:
        matrix[c]={}
        for seed in m.pm.SEEDS:
            rs=[]
            for i in range(1371):
                hit=i%(2 if seed==193 else 4)==0
                x=row(i,1 if seed==193 and hit else (2 if hit else 700+i),hit,'g'+str(i%2))
                x['query_identity']['variant']='observed' if i<1000 else ('frequency_crop' if i<1200 else 'peak_dropout')
                rs.append(x)
            matrix[c][seed]=rs
    return matrix

class MatrixTests(unittest.TestCase):
    def test_full180_and324_including_empty_strata(self):
        x=m.analyze_matrix(fixture(),registration())
        self.assertEqual(len(x['metrics']),324);self.assertEqual(len(x['comparisons']),180)
        self.assertEqual(sum(r['status']=='EMPTY_STRATUM' for r in x['comparisons']),120)
        self.assertTrue(all(r['two_seed']['delta']['MRR']==0 for r in x['comparisons'] if r['status']=='COMPLETE'))
    def test_primary_equal_seed_ratio_in_table(self):
        rows=m.matrix_metrics(m.validate_matrix(fixture()))
        x=next(r for r in rows if r['config']=='fixed_single' and r['seed']=='two_seed_mean' and r['variant']=='observed' and r['source']=='all')
        self.assertEqual(x['in_pool_MRR'],.75)
    def test_missing_cells_and_partial_identities_rejected(self):
        x=fixture();del x['cap10_dual'][193]
        with self.assertRaisesRegex(ValueError,'all2_seeds'):m.validate_matrix(x)
        x=fixture();x['fixed_single'][193].pop()
        with self.assertRaisesRegex(ValueError,'exact1371'):m.validate_matrix(x)
    def test_mask_drift_rejected(self):
        x=fixture();x['fixed_dual'][211][0]['mask_content_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'complete_matrix_exact'):m.validate_matrix(x)
    def test_registration_or_reselection_expansion_rejected(self):
        r=registration();r['test_reselection_allowed']=True
        with self.assertRaisesRegex(ValueError,'no_reselection'):m.validate_registration(r)
        r=registration();r['comparisons'].pop()
        with self.assertRaisesRegex(ValueError,'exact10'):m.validate_registration(r)
    def test_empty_table_null_and_RR_partition(self):
        rows=m.matrix_metrics(m.validate_matrix(fixture()))
        for x in rows:
            if x['queries']==0:self.assertIsNone(x['MRR']);self.assertIsNone(x['in_pool_MRR'])
            else:self.assertAlmostEqual(x['in_pool_RR_contribution']+x['out_of_pool_RR_contribution'],x['MRR'])
if __name__=='__main__':unittest.main()

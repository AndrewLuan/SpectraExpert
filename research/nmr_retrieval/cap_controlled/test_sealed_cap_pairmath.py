"""功能概述：sealed_cap_pairmath纯数学作者合成测试，无真实train/val/test读取。
rows/paired构造两seed同身份的稀疏池命中；测试条件比值的非线性、整簇重采样、
null分母、完整strata、精确配对与RR贡献恒等式。由unittest入口执行，不调用模型。
"""
import copy,unittest,math
import sealed_cap_pairmath as m
MASK='a'*64

def row(i,rank,hit,group=None):
    q=dict(query_order=i,variant='observed',source='HMDB',record_id=str(i),smiles='C',group=group or 'g'+str(i))
    return dict(query_identity=q,rank=rank,RR=1/rank,target_in_pool=hit,mask_content_sha256=MASK)
def pair():
    left={193:[row(0,1,True),row(1,2,True),row(2,600,False),row(3,700,False)],
          211:[row(0,4,True),row(1,600,False),row(2,700,False),row(3,800,False)]}
    right={s:[row(i,2 if i==0 else 700+i,i==0) for i in range(4)] for s in m.SEEDS}
    return left,right
def observed(left,right,reps=37):
    return m.analyze_pair(left,right,replicates=reps)[0]

class PairMathTests(unittest.TestCase):
    def test_new_and_flat_schema_equivalent(self):
        x=row(0,2,True);flat=dict(x['query_identity'],rank=2,RR=.5,target_in_coarse_pool=True,mask_content_sha256=MASK)
        self.assertEqual(m.identity(m.normalize(x)),m.identity(m.normalize(flat)))
        self.assertEqual(m.outcomes(m.normalize(x)).tolist(),m.outcomes(m.normalize(flat)).tolist())
    def test_duplicate_identity_rejected(self):
        l,r=pair();l[193].append(copy.deepcopy(l[193][0]))
        with self.assertRaisesRegex(ValueError,'duplicate_query_identity'):observed(l,r)
    def test_mask_mismatch_or_missing_rejected(self):
        for value in (None,'b'*64):
            l,r=pair();r[211][0]['mask_content_sha256']=value
            with self.assertRaisesRegex(ValueError,'paired_mask_mismatch'):observed(l,r)
    def test_equal_seed_conditional_ratios_are_primary(self):
        l,r=pair();x=observed(l,r)
        self.assertAlmostEqual(x['two_seed']['left']['in_pool_MRR'],.5)
        self.assertAlmostEqual(x['two_seed']['left']['in_pool_MRR_pooled'],1.75/3)
        self.assertNotEqual(x['two_seed']['left']['in_pool_MRR'],x['two_seed']['left']['in_pool_MRR_pooled'])
    def test_one_cluster_resamples_whole_query_set(self):
        l,r=pair()
        for s in m.SEEDS:
            for x in l[s]+r[s]:x['query_identity']['group']='same'
        x=observed(l,r,43);p=x['two_seed']['delta']['MRR'];ci=x['paired95CI']['MRR']
        self.assertEqual(x['connectivity_clusters'],1)
        self.assertAlmostEqual(ci['lower'],p);self.assertAlmostEqual(ci['upper'],p)
        self.assertEqual(ci['valid_replicates'],43)
    def test_zero_denominator_preserves_null_replicates(self):
        l,r=pair()
        for s in m.SEEDS:
            for x in l[s]:x['rank']=700;x['RR']=1/700;x['target_in_pool']=False
        x=observed(l,r,47)
        self.assertIsNone(x['two_seed']['left']['in_pool_MRR'])
        self.assertIsNone(x['two_seed']['delta']['in_pool_MRR'])
        self.assertEqual(x['paired95CI']['in_pool_MRR']['undefined_replicates'],47)
        self.assertEqual(x['paired95CI']['in_pool_MRR']['valid_replicates'],0)
    def test_empty_strata_retained(self):
        l,r=pair();xs=m.analyze_pair(l,r,replicates=31)
        self.assertEqual(len(xs),18)
        self.assertEqual(sum(x['status']=='EMPTY_STRATUM' for x in xs),16)
        for x in xs:
            if x['status']=='EMPTY_STRATUM':self.assertEqual(x['queries_per_seed'],0);self.assertIsNone(x['CI'])
    def test_RR_partition_and_category_delta(self):
        l,r=pair();x=observed(l,r)
        for side in ('left','right'):
            a=x['two_seed'][side];self.assertAlmostEqual(a['in_pool_RR_contribution']+a['out_of_pool_RR_contribution'],a['MRR'])
        self.assertAlmostEqual(sum(v['contribution_to_total_delta_MRR'] for v in x['categories'].values()),x['two_seed']['delta']['MRR'])
        self.assertAlmostEqual(sum(v['fraction'] for v in x['categories'].values()),1.)
    def test_seed_reproducible_and_invalid_RR_rejected(self):
        l,r=pair();self.assertEqual(observed(l,r,53),observed(l,r,53))
        l[193][0]['RR']=.9
        with self.assertRaisesRegex(ValueError,'RR_matches_rank'):observed(l,r)
    def test_common_candidate_order_displacement(self):
        a=dict(natural_candidate_indices=list(range(512)),natural_candidate_costs=list(map(float,range(512))))
        b=copy.deepcopy(a);self.assertEqual(m.pool_change(a,b)['common_candidate_mean_abs_rank_change'],0)
        b['natural_candidate_costs'][0],b['natural_candidate_costs'][1]=1.,0.
        self.assertAlmostEqual(m.pool_change(a,b)['common_candidate_mean_abs_rank_change'],2/512)
if __name__=='__main__':unittest.main()

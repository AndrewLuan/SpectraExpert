"""功能概述：生产统计gate的作者纯合成row/CLI边界测试，无真实预测读取。
row构造512个精确fp32候选；测试native tie/rank/TopK/outside语义和篡改rank后的拒绝。
zero及argv_value检查严格退出类型和唯一CLI参数；真实12child整体验证由非作者独审完成。
"""
import copy,unittest
import summarize_sealed_cap_results as s

def row(target=1,hit=True):
    rank=target+1 if hit else 700
    return dict(query_identity=dict(query_order=0,variant='observed',source='HMDB',record_id='0',smiles='C',group='g'),
      natural_candidate_indices=list(range(512)),natural_candidate_costs=list(map(float,range(512))),
      winner_candidate_index=0,target_candidate_index=target if hit else None,target_cost=float(target) if hit else None,
      margin=float(target) if hit else None,target_in_pool=hit,rank=rank,RR=1/rank,
      in_pool_target_rank=rank if hit else None,cached_outside_pool_rank=None if hit else rank,
      top1=rank<=1,top5=rank<=5,top10=rank<=10,mask_content_sha256='a'*64)

class ResultGateTests(unittest.TestCase):
    def test_native_row_rank_and_ties(self):
        s.validate_payload_row(row())
        x=row(17);x.update(natural_candidate_costs=[1e10]*512,rank=512,RR=1/512,in_pool_target_rank=512,
          target_cost=1e10,margin=0.,top1=False,top5=False,top10=False)
        s.validate_payload_row(x)
    def test_rank_RR_selfconsistent_lie_rejected(self):
        x=row();x.update(rank=3,RR=1/3,in_pool_target_rank=3)
        with self.assertRaisesRegex(ValueError,'native_rank_from_fp32'):s.validate_payload_row(x)
    def test_winner_and_TopK_lie_rejected(self):
        x=row();x['winner_candidate_index']=7
        with self.assertRaisesRegex(ValueError,'native_winner'):s.validate_payload_row(x)
        x=row();x['top1']=True
        with self.assertRaisesRegex(ValueError,'native_TopK'):s.validate_payload_row(x)
    def test_outside_semantics_preserved(self):
        s.validate_payload_row(row(hit=False))
        x=row(hit=False);x['target_candidate_index']=5
        with self.assertRaisesRegex(ValueError,'native_outside'):s.validate_payload_row(x)
    def test_nonfp32_cost_or_duplicate_ID_rejected(self):
        x=row();x['natural_candidate_costs'][0]=1/3
        with self.assertRaisesRegex(ValueError,'exact_native_fp32'):s.validate_payload_row(x)
        x=row();x['natural_candidate_indices'][1]=0
        with self.assertRaisesRegex(ValueError,'native512_unique'):s.validate_payload_row(x)
    def test_strict_exit_and_unique_CLI(self):
        s.zero(0,'real')
        for value in (False,True,None,'0',1):
            with self.assertRaises(ValueError):s.zero(value,'bad')
        self.assertEqual(s.argv_value(['--seed','193'],'--seed'),'193')
        with self.assertRaisesRegex(ValueError,'single_CLI'):s.argv_value(['--seed','193','--seed','211'],'--seed')
if __name__=='__main__':unittest.main()

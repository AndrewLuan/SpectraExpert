"""功能概述：检查组合评测按真实查询身份对齐，且核种/压力路径不混用。
合成查询乱序时仍只用HMDB报告分支；缺查询必须报错。聚类bootstrap对同结构多谱
保留整组重采样，在全组固定差值时得到退化区间，且不接受错配cohort。无GPU或源数据。
"""
import unittest
from .h_count_finalization import compose,cluster_bootstrap,row_key


class FinalizationTests(unittest.TestCase):
    def rows(self):return [dict(source=s,record_id=str(i),smiles='CC',rank=2,top10=['CC'],top1_tanimoto=1.) for i,s in enumerate(['HMDB','NMRexp'])]
    def test_routing_joins_identity_not_file_order(self):
        w=self.rows();r=[dict(x,rank=9 if x['source']=='NMRexp' else 1) for x in reversed(w)];result=compose(w,r);self.assertEqual([x['rank'] for x in result],[1,2])
        with self.assertRaises(ValueError):compose(w,r[:1])
    def test_joint_identity_and_stress_keep_nuclei_separate(self):
        w=[dict(H_source='HMDB',H_record_id='h',smiles='CC',**{'joint':{'rank':2}})];r=[dict(w[0],joint={'rank':1})];self.assertEqual(compose(w,r)[0]['joint']['rank'],1);self.assertEqual(row_key(w[0]),('HMDB','h','CC'));self.assertNotEqual(row_key(dict(source='HMDB',record_id='h',smiles='CC',variant='1H_peak_dropout'),True),row_key(dict(source='HMDB',record_id='h',smiles='CC',variant='13C_peak_dropout'),True))
    def test_cluster_bootstrap_aligns_repeated_structure(self):
        old=self.rows();new=[dict(x,rank=1) for x in reversed(old)];result=cluster_bootstrap(new,old,repeats=50);self.assertEqual(result['connectivity_clusters'],1);self.assertEqual(result['metrics']['MRR']['difference'],.5);self.assertEqual(result['metrics']['MRR']['cluster_95_percentile_interval'],[.5,.5])
        with self.assertRaises(ValueError):cluster_bootstrap(new,old[:1],repeats=10)


if __name__=='__main__':unittest.main()

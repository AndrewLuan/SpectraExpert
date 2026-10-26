"""功能概述：验证多源分组合并、冲突回退、理论侧表隔离与冻结计数缺失掩码。
全部输入为合成结构和数组；不读取研究数据，不调用网络或GPU。
"""
import unittest
import numpy as np
from .h_count_expand import merge_registries,feature_counts
from .h_count_sidecars import combine,augment_theory_rows
from .data import identity


class SidecarTests(unittest.TestCase):
    def entry(self):return dict(counts={'1H':1},groups=[dict(parents={0:4},mass=4,complete=True)])
    def side(self):
        v={m:dict(q=np.ones(128,np.float32),annotation={}) for m in ['raw','calibrated','physical']};return dict(variants={('theory','QM9-NMR','record','C'):v},theory_validation=[])
    def test_merge_C_missing_is_not_conflict(self):
        merged,conflicts=merge_registries({'C':dict(counts={'13C':1},groups=None)},{'C':self.entry()});self.assertFalse(conflicts);self.assertEqual(merged['C']['counts'],{'1H':1,'13C':1})
    def test_H_conflict_unknown(self):
        old=self.entry();old['counts']['1H']=2;merged,conflicts=merge_registries({'C':old},{'C':self.entry()});self.assertNotIn('C',merged);self.assertEqual(conflicts,['C'])
    def test_sidecar_and_masks(self):
        side,excluded=combine([self.side()],{'C':self.entry()});self.assertFalse(excluded);rows,added=augment_theory_rows([],side,{});self.assertEqual(added,1);self.assertEqual(rows[0]['physics'][5],0);self.assertEqual(rows[0]['physics'][7],0);self.assertEqual(rows[0]['split'],'train');self.assertEqual(augment_theory_rows(rows,side,{})[1],0)
        p=np.zeros(8);p[6]=1/30;p[7]=1;self.assertEqual(feature_counts(p)['counts'],{'13C':1})
    def test_duplicates_and_test_rejected(self):
        with self.assertRaises(ValueError):combine([self.side(),self.side()],{'C':self.entry()})
        s=self.side();s['variants']={('theory','QM9-NMR','record','CCO'):next(iter(s['variants'].values()))};self.assertEqual(identity('CCO')[2],'test')
        with self.assertRaises(ValueError):combine([s],{})


if __name__=='__main__':unittest.main()

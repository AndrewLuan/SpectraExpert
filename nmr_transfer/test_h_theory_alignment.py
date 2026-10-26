"""功能概述：验证理论源正仿射不修改原谱、缺峰状态或由独立原子侧表管理的目标。
使用合成谱和拟合参数，验证窗口同域转换及train/theory边界，无外部数据或服务。
"""
import unittest
import numpy as np
from .h_theory_alignment import align_unassigned
from .h_labels import make_key


class AlignmentTests(unittest.TestCase):
    def row(self):return dict(domain='theory',source='FG26',split='train',record_id='a',smiles='CC',q=np.array([1.,2.,3.],np.float32),observation=dict(kind='partial_window',windows=[[0.,4.]],partial_peaks=True))
    def test_affine_and_missing(self):
        r=self.row();fit={'FG26|1H':dict(molecules=30,affine_scale=.5,affine_offset_ppm=1.)};rows,a=align_unassigned([r],{},fit);self.assertTrue(np.array_equal(rows[0]['q'],[1.5,2.,2.5]));self.assertTrue(np.array_equal(r['q'],[1.,2.,3.]));self.assertEqual(rows[0]['observation'],dict(kind='partial_window',windows=[[1.,3.]],partial_peaks=True));self.assertEqual(r['observation']['windows'],[[0.,4.]]);self.assertEqual(a['records'],{'FG26':1})
    def test_sidecar_is_not_double_calibrated(self):
        r=self.row();rows,a=align_unassigned([r],{make_key(r):{}},{'FG26|1H':dict(molecules=30,affine_scale=.5,affine_offset_ppm=1.)});self.assertIs(rows[0],r);self.assertFalse(a['records'])
    def test_scope(self):
        for domain,split in [('experimental','train'),('theory','test'),('theory','val')]:
            with self.assertRaises(ValueError):align_unassigned([dict(self.row(),domain=domain,split=split)],{}, {})


if __name__=='__main__':unittest.main()

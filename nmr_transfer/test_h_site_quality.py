"""功能概述：验证实验位点质量标记仅关闭母节点监督，保留原整谱与缺失标签证据。
使用合成标签，检查部分观测、理论隔离和两种对照；无研究数据或外部服务。
"""
import copy,unittest
import numpy as np
from .h_site_quality import conflicts,apply_quality
from .h_labels import make_key


class QualityTests(unittest.TestCase):
    def record(self):return dict(domain='experimental',source='synthetic',record_id='a',split='train',smiles='CC',q=np.arange(128,dtype=np.float32))
    def test_retains_spectrum_and_completeness(self):
        r=self.record();q=r['q'].copy();key=make_key(r);ann=dict(groups={0:[1.,1.,3.]},observed_H=3,structural_H=6,full_atom_labels=False);labels={key:ann};a=apply_quality(labels,[r],'methyl_record_mask');self.assertEqual(a['flagged_records'],1);self.assertEqual(labels[key]['groups'],{});self.assertEqual(labels[key]['observed_H'],3);self.assertFalse(labels[key]['full_atom_labels']);self.assertTrue(np.array_equal(r['q'],q));self.assertEqual(ann['groups'],{0:[1.,1.,3.]})
    def test_original_and_missing(self):
        r=self.record();key=make_key(r);ann=dict(groups={0:[1.,1.,3.]});labels={key:ann};apply_quality(labels,[r],'original');self.assertIs(labels[key],ann);self.assertFalse(conflicts('CC',dict(groups={0:[1.]})));self.assertFalse(conflicts('CC',dict(groups={0:[1.,1.,1.]})))
    def test_training_scope(self):
        for domain,split in [('theory','train'),('experimental','test'),('experimental','val')]:
            r=dict(self.record(),domain=domain,split=split)
            with self.assertRaises(ValueError):apply_quality({},[r],'methyl_record_mask')


if __name__=='__main__':unittest.main()

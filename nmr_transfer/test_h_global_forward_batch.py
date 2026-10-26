"""功能概述：验证同一实验查询的多种观测在排名训练批次中互不覆盖。

合成 HMDB 样例同时放入原谱与裁窗变体，检查元数据侧表不把完整峰恢复到裁窗谱。
正/负分子共享查询溶剂与pH，但保留各自结构核数。样例只构造小分子 CPU 图，
不读取实际训练/验证/测试数据、模型权重或网络。
"""
import unittest

import numpy as np
import torch

from .data import graph,identity,quantiles
from .h_conditions import encode_conditions,local_environment
from .h_global_forward_train import make_rank_batch
from .h_labels import make_key


class GlobalForwardBatchTests(unittest.TestCase):
    def test_hmdb_variants_do_not_overwrite_each_other(self):
        positive='CCO';negative='CCCC';_,group,split=identity(positive)
        p=np.zeros(8,np.float32);p[0]=6/40
        n=np.zeros(8,np.float32);n[0]=10/40
        meta=encode_conditions(dict(solvent='D2O',experimental_conditions=dict(pH='7')))
        meta['line_centers']=[1.,2.,7.]
        row=dict(smiles=positive,source='HMDB',record_id='same_record',domain='experimental',nucleus='1H',
            group=group,split=split,physics=p,q=quantiles([1.,2.,7.]),cond=meta['old_cond'],
            observation=dict(kind='unknown',windows=[]),measure='peak_height_not_integral')
        neg=dict(smiles=negative,group=identity(negative)[1],physics=n)
        cases=[dict(positive=row,variant=variant,negative=[neg],group=group) for variant in ('frequency_crop','observed')]
        graphs={s:graph(s) for s in (positive,negative)}
        for s,g in graphs.items():g.local_environment=torch.tensor(local_environment(s))
        raw={'same_record':dict(value=dict(x=[1.,2.,7.],y=[100.,1.,50.]))}
        batch,count,stride=make_rank_batch(cases,graphs,{make_key(row):meta},{},raw)
        self.assertEqual((count,stride),(2,2))
        self.assertLessEqual(float(batch.q[0].max()),.200001)
        self.assertGreater(float(batch.q[2].max()),.69999)
        self.assertTrue(torch.equal(batch.q[0],batch.q[1]))
        self.assertTrue(torch.equal(batch.q[2],batch.q[3]))
        self.assertTrue(torch.equal(batch.condition_vector[0],batch.condition_vector[1]))
        self.assertAlmostEqual(float(batch.physics[0,0]),6/40,places=6)
        self.assertAlmostEqual(float(batch.physics[1,0]),10/40,places=6)


if __name__=='__main__':
    unittest.main()

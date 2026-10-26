"""验证SpecGNN科学数据契约：结构置换、观测裁剪、未知缺失及源目标只读。
TestContract运行独立数学反例；不访问实验测试集，不启动网络和GPU训练。
覆盖窗口外峰不应受罚、未知缺失只匹配观测、空观测窗口仍有梯度，以及完整原子证据。
"""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from torch_geometric.data import Batch,Data
from .data import graph,identity,quantiles,physical_features
from .model import SpecGNN
from .observation import coverage,observation_loss,KINDS
from .train import Spectra
from .evaluate import distances_for_query
from .prepare import atom_aligned_audit


class TestContract(unittest.TestCase):
    def batch(self,kind='window',window=(0.,.5),target=.2):
        windows=torch.zeros(1,8,2);mask=torch.zeros(1,8,dtype=torch.bool)
        if window is not None:windows[0,0]=torch.tensor(window);mask[0,0]=True
        return SimpleNamespace(q=torch.full((1,128),target),windows=windows,window_mask=mask,
                               coverage_kind=torch.tensor([KINDS[kind]]))

    def test_unmeasured_signal_not_negative(self):
        b=self.batch();p=torch.cat([torch.full((1,64),.2),torch.full((1,64),2.)],1)
        self.assertAlmostEqual(float(observation_loss(p,b)),0.,places=6)
        p[:,64:]=20.
        self.assertAlmostEqual(float(observation_loss(p,b)),0.,places=6)

    def test_unknown_coverage_uses_presence(self):
        b=self.batch('unknown',None);p=torch.linspace(-10,10,128).reshape(1,-1);p[0,64]=.2;p=p.sort(1).values
        self.assertAlmostEqual(float(observation_loss(p,b)),0.,places=6)
        self.assertGreater(float(observation_loss(p,self.batch('complete_atoms',None))),1.)

    def test_empty_window_has_finite_gradient(self):
        p=torch.full((1,128),2.,requires_grad=True);loss=observation_loss(p,self.batch()).mean();loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0.)

    def test_no_window_inferred_from_peaks(self):
        self.assertEqual(coverage({'value':{'x':[1.,2.]}})['kind'],'unknown')
        self.assertEqual(coverage({'atom_target_mask':[1,0]},2)['kind'],'unknown')
        self.assertEqual(coverage({'atom_target_mask':[1,1]},2)['kind'],'complete_atoms')
        self.assertEqual(coverage({'observed_ppm_ranges':[[0.,3.],[6.,9.]]})['windows'],[[0.,3.],[6.,9.]])

    def test_graph_and_permutation(self):
        g=graph('CCO');self.assertEqual(g.x.shape[1],99);self.assertEqual(g.edge_attr.shape[1],24)
        g.physics=torch.tensor(physical_features('CCO')).reshape(1,8);g.cond=torch.zeros(1,4)
        m=SpecGNN(width=32,layers=2).eval();y=m(Batch.from_data_list([g]))
        perm=torch.randperm(g.num_nodes);inv=torch.argsort(perm)
        p=Data(x=g.x[perm],edge_index=inv[g.edge_index],edge_attr=g.edge_attr,physics=g.physics,cond=g.cond)
        torch.testing.assert_close(y,m(Batch.from_data_list([p])),rtol=1e-4,atol=1e-5)
        self.assertTrue((y[:,1:]>=y[:,:-1]).all())
        self.assertEqual(identity('CCO')[1],identity('OCC')[1])

    def test_targets_unchanged_and_ranking_matches_loss(self):
        s='CCO';q=quantiles([1.,2.,3.]);row=dict(smiles=s,domain='experimental',source='example',nucleus='1H',q=q.copy(),
            cond=np.zeros(4,np.float32),physics=physical_features(s),observation={'kind':'unknown','windows':[]})
        item=Spectra([row],{s:graph(s)},'1H','domain')[0]
        np.testing.assert_array_equal(q,row['q'])
        pred=torch.linspace(0.,.6,128).reshape(1,-1);b=Batch.from_data_list([Spectra([row],{s:graph(s)},'1H')[0]])
        torch.testing.assert_close(distances_for_query(pred,row,'1H'),observation_loss(pred,b))

    def test_atom_aligned_fit_recovers_known_transform(self):
        pairs={'1H':[(float(x),1.1*float(x)+.2,'aromatic_H',str(i)) for i,x in enumerate(np.linspace(0,12,200))]}
        result=atom_aligned_audit(pairs)['recalculated|1H']
        self.assertAlmostEqual(result['affine_scale'],1.1,places=4)
        self.assertAlmostEqual(result['affine_offset_ppm'],.2,places=4)
        self.assertLess(result['fitted_atom_MAE_ppm'],1e-5)

    def test_dropout_removes_whole_peak_without_interpolation(self):
        s='CCO';q=quantiles([1.,3.]);row=dict(smiles=s,domain='experimental',source='fixture',nucleus='1H',q=q.copy(),
            cond=np.zeros(4,np.float32),physics=physical_features(s),observation={'kind':'complete_atoms','windows':[]})
        def zero_normal(loc=0.,scale=1.,size=None):
            return np.zeros(size) if size is not None else 0.
        with patch('nmr_transfer.train.np.random.random',side_effect=[.1,np.array([0.,1.]),.9]),patch('nmr_transfer.train.np.random.normal',side_effect=zero_normal):
            item=Spectra([row],{s:graph(s)},'1H','augment')[0]
        torch.testing.assert_close(item.q,torch.full((1,128),.3))
        self.assertEqual(int(item.coverage_kind),KINDS['unknown'])
        np.testing.assert_array_equal(row['q'],q)


if __name__=='__main__':unittest.main()

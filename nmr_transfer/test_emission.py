"""核发射模型的物理与数学契约：核质量守恒、甲基绑定、酰胺双环境及重排不变。
只用乙醇、甲烷、DMF等人工结构检查；不读取任何实验test标签或拟合阈值。
"""
import unittest
import torch
from torch_geometric.data import Batch,Data
from .data import graph,physical_features
from .emission_model import NuclearSpecGNN


def fixture(s):
    g=graph(s);g.physics=torch.tensor(physical_features(s)).reshape(1,8);g.cond=torch.zeros(1,4)
    return g


class TestEmission(unittest.TestCase):
    def test_nuclear_mass_and_amide_components(self):
        b=Batch.from_data_list([fixture('CCO'),fixture('CN(C)C=O')])
        for n,expected,components in [('1H',[6.,7.],13),('13C',[2.,3.],5)]:
            m=NuclearSpecGNN(n,32,2);v,w,g=m.components(b)
            mass=torch.zeros(2).index_add_(0,g,w)
            torch.testing.assert_close(mass,torch.tensor(expected));self.assertEqual(int((g==1).sum()),components)
            y=m(b);self.assertEqual(tuple(y.shape),(2,128));self.assertTrue(torch.isfinite(y).all());self.assertTrue((y[:,1:]>=y[:,:-1]).all())
            y.square().mean().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_methane_hydrogens_share_shift(self):
        m=NuclearSpecGNN('1H',32,2);v,w,g=m.components(Batch.from_data_list([fixture('C')]))
        self.assertEqual(len(v),4);self.assertAlmostEqual(float(v.max()-v.min()),0.,places=6)

    def test_homonuclear_hydrogen_mass(self):
        m=NuclearSpecGNN('1H',32,2);b=Batch.from_data_list([fixture('[H][H]')])
        v,w,g=m.components(b)
        self.assertEqual(len(v),2);self.assertEqual(float(w.sum()),2.)
        self.assertTrue(torch.isfinite(m(b)).all());self.assertLess(float(m(b).detach().abs().max()),10.)

    def test_permutation_invariance(self):
        g=fixture('CCO');m=NuclearSpecGNN('13C',32,2).eval();y=m(Batch.from_data_list([g]))
        p=torch.randperm(g.num_nodes);inv=torch.argsort(p)
        other=Data(x=g.x[p],edge_index=inv[g.edge_index],edge_attr=g.edge_attr,physics=g.physics,cond=g.cond)
        torch.testing.assert_close(y,m(Batch.from_data_list([other])),rtol=1e-4,atol=1e-5)


if __name__=='__main__':unittest.main()

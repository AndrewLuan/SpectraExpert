"""功能概述：验证固定环境槽的质量、潜在容量、未知回退和结构排列契约。
只用合成结构分组/随机模型，检查同次分位标签、可微中心和registry SHA拒绝；
物理分组是测试输入假设，不读取实验/理论标签或外部服务。
"""
import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch
from .data import graph,physical_features
from .h_conditions import local_environment,CONDITION_DIM
from .h_acyl import acyl_environment
from .h_equivalence_model import equivalence_layout,CertifiedGroupSpecGNN
from .h_solvent_message import SolventMessageSpecGNN
from .emission_model import weighted_quantiles
from .prepare import sha


class EquivalenceTests(unittest.TestCase):
    def registry(self):return {'CCO':dict(groups=[dict(parents={0:3},mass=3),dict(parents={1:2},mass=2),dict(parents={2:1},mass=1)],counts={'1H':3})}
    def batch(self,names):
        graphs=[]
        for s in names:
            g=graph(s);g.structure_smiles=s;g.physics=torch.tensor(physical_features(s)).reshape(1,8);g.cond=torch.zeros(1,4);g.condition_vector=torch.zeros(1,CONDITION_DIM);g.local_environment=torch.tensor(local_environment(s));g.acyl_environment=torch.tensor(acyl_environment(s).copy());graphs.append(g)
        return Batch.from_data_list(graphs)
    def model(self,folder,registry=None,mode='physical'):
        p=Path(folder)/'registry.pt';torch.save(self.registry() if registry is None else registry,p);return CertifiedGroupSpecGNN('1H',16,1,'local','solvent',0,'late',str(p),sha(p),mode).eval()
    def test_layout_and_capacity(self):
        r=self.registry()['CCO'];ids,offset,dups,known=equivalence_layout('CCO',r);self.assertTrue(known);self.assertEqual(sum(ids.flatten()>=0),6);self.assertEqual(len(np.unique(ids[ids>=0])),3)
        ids,offset,dups,known=equivalence_layout('CCO',r,'local');self.assertEqual(len(np.unique(ids[ids>=0])),4)
        r=dict(groups=[dict(parents={0:3,1:3},mass=6)]);ids,_,_,_=equivalence_layout('CC',r);self.assertEqual(len(np.unique(ids[ids>=0])),1)
    def test_forward_and_gradient(self):
        with tempfile.TemporaryDirectory() as folder:
            m=self.model(folder);b=self.batch(['CCO','CCC']);d=m(b,details=True);self.assertEqual(d['weights'][d['groups']==0].sum(),6);self.assertEqual(d['weights'][d['groups']==1].sum(),8);self.assertLessEqual(len(d['values'][d['groups']==0].unique()),3);self.assertTrue(torch.equal(d['quantiles'],weighted_quantiles(d['values'],d['weights'],d['groups'],2)));d['quantiles'].sum().backward();self.assertTrue(torch.isfinite(m.equivalence_head[-1].weight.grad).all());self.assertGreater(float(m.equivalence_head[-1].weight.grad.abs().sum()),0)
    def test_unknown_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            m=self.model(folder,{});b=self.batch(['CCO']);expected=SolventMessageSpecGNN.forward(m,b,details=True);actual=m(b,details=True)
            for key in expected:self.assertTrue(torch.equal(expected[key],actual[key]))
    def test_group_order(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(1);a=self.model(folder);r=self.registry();r['CCO']['groups'].reverse();torch.manual_seed(1);b=self.model(folder,r);self.assertTrue(torch.allclose(a(self.batch(['CCO'])),b(self.batch(['CCO'])),atol=1e-7,rtol=0))
    def test_SHA(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'registry.pt';torch.save({},p)
            with self.assertRaises(ValueError):CertifiedGroupSpecGNN('1H',16,1,'local','solvent',0,'late',str(p),'wrong')


if __name__=='__main__':torch.set_num_threads(1);unittest.main()

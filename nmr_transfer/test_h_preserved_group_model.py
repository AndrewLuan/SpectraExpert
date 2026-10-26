"""功能概述：验证固定组残差解码保留初始谱、必要合并、核质量和可微性。
纯张量测试覆盖不等间距支持、同质量区间均值和交错owner重排。分子测试用合成
registry检查局部控制与原发射逐点相同、甲基酰胺既有间距保留、物理组容量及未知
结构精确回退；分区重排不改变谱，残差梯度有限。无真实谱标签、GPU或外部服务。
"""
import tempfile,unittest
from pathlib import Path
import torch
from .h_preserved_group_model import PreservedGroupSpecGNN,orbit_barycentres
from .h_solvent_message import SolventMessageSpecGNN
from . import test_h_equivalence_model as equivalence_fixtures
from .prepare import sha


class PreservedGroupTests(unittest.TestCase):
    batch=equivalence_fixtures.EquivalenceTests.batch
    def registry(self):return {'CCO':dict(groups=[dict(parents={0:3},mass=3),dict(parents={1:2},mass=2),dict(parents={2:1},mass=1)]),'CN(C)C=O':dict(groups=[dict(parents={0:3},mass=3),dict(parents={2:3},mass=3),dict(parents={3:1},mass=1)]),'CC':dict(groups=[dict(parents={0:3,1:3},mass=6)])}
    def model(self,folder,mode='physical',registry=None):
        path=Path(folder)/'registry.pt';torch.save(self.registry() if registry is None else registry,path);model=PreservedGroupSpecGNN('1H',16,1,'local','solvent',0,'late',str(path),sha(path),mode).eval();model.amide_gap_scale=.132;return model
    def test_unequal_spacing_is_preserved_and_merging_uses_bin_mean(self):
        values=torch.tensor([0.,1.,5.,20.],requires_grad=True);d=dict(values=values,weights=torch.ones(4),owners=torch.zeros(4,dtype=torch.long));parents=torch.zeros(4,dtype=torch.long);groups=torch.arange(4);orbits=torch.zeros(4,dtype=torch.long);_,base=orbit_barycentres(d,parents,groups,orbits,torch.arange(4),torch.full((4,),4),4);self.assertTrue(torch.equal(base,values));base.sum().backward();self.assertTrue(torch.equal(values.grad,torch.ones(4)))
        _,merged=orbit_barycentres(d,parents,torch.tensor([0,0,1,1]),torch.zeros(2,dtype=torch.long),torch.arange(2),torch.full((2,),2),2);self.assertTrue(torch.equal(merged,torch.tensor([.5,12.5])))
    def test_interleaved_owner_components(self):
        d=dict(values=torch.tensor([1.,10.,3.,30.]),weights=torch.ones(4),owners=torch.tensor([0,1,0,1]));parents=torch.tensor([0,0,1,1]);groups=torch.tensor([0,1,2,3]);orbits=torch.tensor([0,0,2,2]);_,base=orbit_barycentres(d,parents,groups,orbits,torch.tensor([0,1,0,1]),torch.full((4,),2),4);self.assertTrue(torch.equal(base,d['values'][torch.tensor([0,2,1,3])]));order=torch.tensor([3,1,0,2]);shuffled={k:v[order] for k,v in d.items()};_,again=orbit_barycentres(shuffled,parents,groups,orbits,torch.tensor([0,1,0,1]),torch.full((4,),2),4);self.assertTrue(torch.equal(base,again))
    def test_local_initial_spectrum_and_amide_gap_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            m=self.model(folder,'local');b=self.batch(['CCO','CN(C)C=O','CCC','CC']);old=SolventMessageSpecGNN.forward(m,b,details=True);new=m(b,details=True);self.assertTrue(torch.equal(old['quantiles'],new['quantiles']));self.assertEqual(new['weights'][new['groups']==1].sum(),7);self.assertGreater(len(new['values'][new['groups']==1].unique()),1)
    def test_physical_capacity_gradient_and_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            m=self.model(folder);b=self.batch(['CCO','CCC','CN(C)C=O']);old=SolventMessageSpecGNN.forward(m,b,details=True);d=m(b,details=True);self.assertLessEqual(len(d['values'][d['groups']==0].unique()),3);self.assertTrue(torch.equal(d['quantiles'][1],old['quantiles'][1]));self.assertEqual(d['weights'][d['groups']==0].sum(),6);d['quantiles'].sum().backward();self.assertTrue(torch.isfinite(m.equivalence_head[-1].weight.grad).all());self.assertGreater(float(m.equivalence_head[-1].weight.grad.abs().sum()),0)
    def test_group_reordering_keeps_spectrum(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(2);a=self.model(folder);r=self.registry()
            for entry in r.values():entry['groups'].reverse()
            torch.manual_seed(2);b=self.model(folder,registry=r);names=['CCO','CN(C)C=O','CC'];self.assertTrue(torch.equal(a(self.batch(names)),b(self.batch(names))))


if __name__=='__main__':torch.set_num_threads(1);unittest.main()

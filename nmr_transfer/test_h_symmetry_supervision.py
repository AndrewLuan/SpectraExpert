"""功能概述：核验理论对称母节点监督对源原子编号置换不变，且保留缺失位点。
合成NMe2完整标签交换两个母节点后联合误差应相同；默认旧监督仍能区分任意
定向标签。局部真实H质量、整谱q和源标签保持原值，未知位点不填充。无源数据或服务。
"""
import unittest
import torch
import numpy as np
from .data import graph,physical_features,quantiles
from .h_conditions import local_environment
from . import test_h_replay as replay_fixtures
from .h_replaytrain import ReplaySpectra,grouped_objective
from .h_labels import make_key
from .train import collate
from .emission_model import weighted_quantiles


class SymmetrySupervisionTests(unittest.TestCase):
    def fixture(self,swap=False,partial=False,enabled=True):
        r,g,n,parents,_=replay_fixtures.TestReplay().fixtures();r=dict(r,domain='theory',split='train');xs=[3.,2.] if swap else [2.,3.];labels={p:[x]*3 for p,x in zip(parents,xs)}
        if partial:labels.pop(parents[1])
        labels[3]=[8.];annotations={make_key(r):dict(groups=labels)};b=collate([ReplaySpectra([r],{r['smiles']:g},'1H',{},annotations=annotations,grouped=True,theory_orbits=enabled)[0]])
        values=torch.tensor([.2,.3,.8],requires_grad=True);weights=torch.tensor([3.,3.,1.]);owners=torch.tensor([*parents,3]);groups=torch.zeros(3,dtype=torch.long);details=dict(values=values,weights=weights,owners=owners,groups=groups,quantiles=weighted_quantiles(values,weights,groups,1))
        return b,details,parents,labels
    def test_orbit_loss_ignores_arbitrary_symmetric_parent_order(self):
        losses=[]
        for swap in [False,True]:
            b,d,parents,labels=self.fixture(swap=swap);loss=grouped_objective(d,b)['site'];losses.append(float(loss.detach()));loss.backward();self.assertTrue(torch.isfinite(d['values'].grad).all());self.assertEqual(int(b.site_expected[parents].sum()),6);self.assertAlmostEqual(b.site_targets[parents[0]][0].item(),labels[parents[0]][0]/10,places=6)
        self.assertAlmostEqual(losses[0],0,places=6);self.assertAlmostEqual(losses[1],0,places=6)
        b,d,_,_=self.fixture(swap=True,enabled=False);self.assertGreater(float(grouped_objective(d,b)['site'].detach()),.01)
    def test_partial_orbit_keeps_parent_observation(self):
        b,d,parents,_=self.fixture(partial=True);self.assertTrue(torch.equal(b.group_owner[parents],torch.tensor(parents)));self.assertEqual(int(b.group_mask[parents[1]].sum()),0)
    def test_orbit_merge_preserves_other_site_relative_weight(self):
        losses=[]
        for enabled in [False,True]:
            b,d,_,_=self.fixture(enabled=enabled)
            with torch.no_grad():d['values'][-1]=.9
            d['quantiles']=weighted_quantiles(d['values'],d['weights'],d['groups'],1);losses.append(grouped_objective(d,b)['site'].detach())
        torch.testing.assert_close(losses[0],losses[1],atol=1e-7,rtol=0)
    def test_interleaved_orbit_owners_and_component_order(self):
        s='CCC';g=graph(s);g.local_environment=torch.tensor(local_environment(s));r=dict(smiles=s,source='synthetic',record_id='interleaved',nucleus='1H',domain='theory',split='train',cond=np.zeros(4,np.float32),physics=physical_features(s),q=quantiles([1.]*6+[2.]*2),observation=dict(kind='complete_atoms',windows=[]));labels={make_key(r):dict(groups={0:[1.]*3,1:[2.]*2,2:[1.]*3},full_atom_labels=True)};ds=ReplaySpectra([r],{s:g},'1H',{},annotations=labels,grouped=True,theory_orbits=True);b=collate([ds[0],ds[0]]);owners=torch.tensor([0,1,2,0,1,2])+torch.tensor([0]*3+[len(g.x)]*3);mapped=(b.group_owner+b.ptr[b.batch])[owners];self.assertTrue(bool((mapped[1:]<mapped[:-1]).any()))
        for permutation in [torch.arange(6),torch.tensor([5,1,3,2,0,4])]:
            values=torch.tensor([.1,.2,.1]*2,requires_grad=True);weights=torch.tensor([3.,2.,3.]*2);groups=torch.tensor([0]*3+[1]*3);q=weighted_quantiles(values,weights,groups,2);d=dict(values=values[permutation],weights=weights[permutation],owners=owners[permutation],groups=groups[permutation],quantiles=q);parts=grouped_objective(d,b);torch.testing.assert_close(parts['site'],torch.tensor(0.),atol=1e-7,rtol=0);parts['loss'].backward();self.assertTrue(torch.isfinite(values.grad).all())


if __name__=='__main__':unittest.main()

"""功能概述：验证计数投影的最优性、梯度、质量、未知回退和同次发射契约。
小规模穷举核对DP，重排/幂等和退化输入验证不伪造位置；只用合成谱和结构计数。
依赖h_count_constraint，不读训练或测试谱，不调用GPU或外部服务。
"""
import itertools,unittest
from types import SimpleNamespace
import numpy as np
import torch
from .h_count_constraint import optimal_partition,project_details,count_penalty
from .emission_model import weighted_quantiles,make_model


class CountTests(unittest.TestCase):
    def detail(self,x,w=None):
        v=torch.tensor(x,dtype=torch.float64,requires_grad=True);w=torch.tensor(w or [1.]*len(x),dtype=torch.float64);g=torch.zeros(len(x),dtype=torch.long)
        return dict(values=v,weights=w,groups=g,owners=torch.arange(len(x)),exchangeable=torch.zeros(len(x),dtype=torch.bool),quantiles=weighted_quantiles(v,w,g,1))
    def batch(self,k):return SimpleNamespace(physics=torch.tensor([[0,0,0,0,0,k/30,0,1.]]),num_graphs=1)
    def test_optimum(self):
        x=np.array([-.2,.1,.9,1.1,2.3,7.]);w=np.array([1.,3.,2.,1.,.5,2.])
        for k in [1,2,3,4]:
            ids=optimal_partition(x,w,k)
            def cost(groups):return sum(np.sum(w[g]*(x[g]-np.average(x[g],weights=w[g]))**2) for g in groups)
            actual=cost([np.flatnonzero(ids==j) for j in range(k)])
            expected=min(cost(np.split(np.arange(len(x)),cuts)) for cuts in itertools.combinations(range(1,len(x)),k-1));self.assertAlmostEqual(actual,expected)
    def test_contracts_and_gradient(self):
        d=self.detail([.1,.12,.7,.72],[1.,3.,1.,3.]);b=self.batch(2);p=project_details(d,b)
        self.assertEqual(len(p['values'].unique()),2)
        for key in ['weights','groups','owners','exchangeable']:self.assertTrue(torch.equal(d[key],p[key]))
        self.assertTrue(torch.equal(p['quantiles'],weighted_quantiles(p['values'],p['weights'],p['groups'],1)))
        self.assertTrue(torch.allclose(project_details(p,b)['values'],p['values']))
        p['values'].square().sum().backward();self.assertTrue(torch.isfinite(d['values'].grad).all());self.assertGreater(float(count_penalty(d,b).detach()),0)
    def test_unknown_and_under_count(self):
        for k in [0,2,3]:
            d=self.detail([.1,.1,.7]);self.assertTrue(torch.equal(project_details(d,self.batch(k))['values'],d['values']))
    def test_permutation(self):
        x=np.array([.1,.12,.7,.72]);w=np.array([1.,3.,1.,3.]);p=np.array([3,0,2,1]);a=project_details(self.detail(x.tolist(),w.tolist()),self.batch(2))['values'];b=project_details(self.detail(x[p].tolist(),w[p].tolist()),self.batch(2))['values'];self.assertTrue(torch.allclose(a[p],b))
    def test_nucleus_guard(self):
        with self.assertRaises(ValueError):make_model(dict(model='atom',nucleus='13C',width=16,layers=1,count_decode=True))
    def test_mixed_batch(self):
        d=self.detail([.1,.3,.9,.2,.4]);d['groups']=torch.tensor([0,0,0,1,1]);b=SimpleNamespace(num_graphs=2,physics=torch.tensor([[0,0,0,0,0,1/30,0,1],[0,0,0,0,0,0,1/30,1]]));p=project_details(d,b);self.assertTrue(torch.equal(p['values'][3:],d['values'][3:]));self.assertEqual(len(p['values'][:3].unique()),1);self.assertEqual(p['quantiles'].shape,(2,128))
    def test_invalid(self):
        for x,w,k in [([1],[0],1),([float('nan')],[1],1),([1],[1],0)]:
            with self.assertRaises(ValueError):optimal_partition(x,w,k)
        with self.assertRaises(ValueError):project_details(self.detail([.1,.2]),self.batch(3))


if __name__=='__main__':unittest.main()

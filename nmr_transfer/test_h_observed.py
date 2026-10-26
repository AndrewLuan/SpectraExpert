"""¹H观测过程的概率和数据边界检查，使用合成数据，不读真实训练/测试谱。
核对峰簇不会跨大范围链化、峰高不作为积分、计数概率与SciPy相符、未知公共偏移
边缘化对候选对称，以及训练计数拟合严格排除val/test。依赖h_observed的公开函数，
由unittest执行；不生成模型、不访问服务器、不修改历史评测结果。
"""
import unittest
import numpy as np
import torch
from scipy.stats import binom,betabinom,poisson
from .data import identity
from .h_observed import cluster_lines,count_components,count_nll,fit_dropout_prior,marginal_shift
from .h_observation_finalize import visible_lines,blend
from .h_nonexchange import cached_nonexchange
from .h_partial_audit import project_ordered


class TestObserved(unittest.TestCase):
    def test_cluster_diameter_prevents_chaining(self):
        centers,groups=cluster_lines([1.,1.04,1.08,1.12,1.12,3.],.08)
        self.assertEqual([len(g) for g in groups],[3,1,1])
        self.assertTrue(all(max(g)-min(g)<=.080001 for g in groups))
        np.testing.assert_allclose(centers,[1.04,1.12,3.])

    def test_binomial_and_overreport_probability_matches_scipy(self):
        n=np.array([3.,7.,12.]);k=np.array([2.,4.,8.]);rates=[.5,1.5]
        expected=np.stack([binom.logpmf(k,n,.5),poisson.logpmf(k,n*1.5)],1)
        np.testing.assert_allclose(count_components(n,k,rates),expected)
        counts=torch.tensor(n,dtype=torch.float64);kk=2
        entry=dict(rates=rates,weights=[.4,.6],concentration=None)
        target=-np.log(.4*binom.pmf(kk,n,.5)+.6*poisson.pmf(kk,n*1.5))/kk
        np.testing.assert_allclose(count_nll(counts,kk,entry).numpy(),target,rtol=1e-6)

    def test_correlated_mass_beta_binomial_matches_scipy(self):
        n=np.array([3.,7.,12.]);k=2;rate=.75;c=2.
        expected=betabinom.logpmf(k,n,rate*c,(1-rate)*c)
        np.testing.assert_allclose(count_components(n,np.ones(3)*k,[rate],c)[:,0],expected)
        entry=dict(rates=[rate],weights=[1.],concentration=c)
        np.testing.assert_allclose(count_nll(torch.tensor(n),k,entry).numpy(),-expected/k,rtol=1e-6)

    def test_reference_marginal_is_candidate_symmetric(self):
        shifts=[-.1,0.,.1];cost=torch.tensor([[0.,2.],[1.,1.],[2.,0.]])
        score=marginal_shift(cost,shifts,.1,3)
        torch.testing.assert_close(score[0],score[1]);torch.testing.assert_close(marginal_shift(cost,shifts,0),cost[1])

    def test_hmdb_crop_and_dropout_do_not_restore_hidden_lines(self):
        raw={'a':dict(value=dict(x=[1.,1.01,3.,4.,9.]))}
        query=dict(record_id='a',q=[1.,4.],observation=dict(windows=[[0.,6.]],partial_peaks=True))
        np.testing.assert_array_equal(visible_lines(query,raw),[1.,4.])
        query['observation']=dict(windows=[[0.,6.]])
        np.testing.assert_array_equal(visible_lines(query,raw),[1.,1.01,3.,4.])

    def test_visibility_mixture_preserves_probability_mass(self):
        full=dict(logp=torch.log(torch.tensor([[.2,.8]])),cdf=torch.tensor([[0.,.2,1.]]))
        fixed=dict(logp=torch.log(torch.tensor([[.8,.2]])),cdf=torch.tensor([[0.,.8,1.]]))
        mixed=blend(full,fixed,.25)
        torch.testing.assert_close(mixed['logp'].exp(),torch.tensor([[.65,.35]]))
        torch.testing.assert_close(mixed['cdf'],torch.tensor([[0.,.65,1.]]))

    def test_exchange_visibility_uses_original_cached_positions(self):
        d=dict(values=torch.tensor([1.001,4.001]),weights=torch.tensor([3.,1.]),groups=torch.tensor([0,0]),exchangeable=torch.tensor([False,True]))
        cached=torch.tensor([[1.]*96+[4.]*32])
        q=cached_nonexchange(d,cached,1)
        torch.testing.assert_close(q,torch.ones(1,128))
        d['exchangeable'].fill_(False)
        torch.testing.assert_close(cached_nonexchange(d,cached,1),cached)

    def test_truth_projection_keeps_distinct_nuclear_mass(self):
        projected,error=project_ordered([1.,1.,4.,8.],[.8,1.2,8.3])
        np.testing.assert_array_equal(projected,[1.,1.,8.]);self.assertAlmostEqual(error,.7/3)
        with self.assertRaisesRegex(ValueError,'more_observed_H'):project_ordered([1.],[1.,2.])

    def test_count_fit_excludes_nontraining_identity(self):
        train=next(s for s in ['CC','CO','CCC','CCO','CCCC','CCCO'] if identity(s)[2]=='train')
        held=next('C'*i for i in range(1,80) if identity('C'*i)[2]!='train')
        meta=dict(xs=[1.,2.,4.],weights=[3.,2.,1.],structural_H=6)
        a={('experimental','synthetic','a',train):meta}
        b=dict(a);b['experimental','synthetic','bad',held]=dict(xs=[999.],weights=[999.],structural_H=1)
        self.assertEqual(fit_dropout_prior(a),fit_dropout_prior(b))


if __name__=='__main__':unittest.main()

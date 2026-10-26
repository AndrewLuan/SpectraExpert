"""缺失条件评分的概率与边界检查，使用合成候选密度，不读取真实训练/测试数据。
核对重复分位点加权与原公式一致、零条件假设不会借用真实结构H数、已知条件不做
未知溶剂边缘化、显式窗口与积分谱不误入原子报告评分。所有参数只在人工数据上检查。
依赖h_latent_score及原likelihood，无网络、文件或真实模型副作用。
"""
import unittest
import numpy as np
import torch
from .data import quantiles
from .likelihood import density,log_score
from .h_latent_score import weighted_log_score,features,score,CONFIGS,eligible,PROTOTYPES


class TestLatent(unittest.TestCase):
    def setup_query(self):
        p=torch.tensor([[.1,.1,.3,.8],[.12,.2,.31,.9]],dtype=torch.float32)
        kernel=density(p,'1H',.2)
        q=dict(q=quantiles([1.,3.,8.],[3.,2.,1.]),observation=dict(kind='unknown',windows=[]),cond=np.zeros(4,np.float32),source='synthetic',measure='observed_assigned_atom_equal_partial_possible',observed_xs=[1.,3.,8.],observed_weights=[3.,2.,1.])
        return kernel,q

    def test_weighted_repeated_points_match_original_window_and_unknown_likelihood(self):
        kernel,q=self.setup_query()
        torch.testing.assert_close(weighted_log_score(kernel,q),log_score(kernel,q,'1H'),rtol=1e-6,atol=1e-6)
        q['observation']=dict(kind='partial_window',windows=[[0.,6.]])
        torch.testing.assert_close(weighted_log_score(kernel,q),log_score(kernel,q,'1H'),rtol=1e-6,atol=1e-6)

    def test_candidate_counts_only_and_baseline_exact(self):
        kernel,q=self.setup_query();kernels=dict(given=kernel,**{name:kernel for name in PROTOTYPES});reporting=dict(solvent_prior=[.7,.1,.2],dropout={});counts=torch.tensor([10.,12.]);prior={}
        ff=features(q,kernels,counts,prior,reporting);torch.testing.assert_close(score(ff,CONFIGS[0]),log_score(kernel,q,'1H'),rtol=0,atol=0)
        q.update(smiles='arbitrary_true_identity',structural_H=999,full_atom_labels=True)
        other=features(q,kernels,counts,prior,reporting)
        for cfg in CONFIGS:torch.testing.assert_close(score(ff,cfg),score(other,cfg),rtol=0,atol=0)

    def test_known_condition_does_not_use_unknown_solvent_prototypes(self):
        kernel,q=self.setup_query();q['_condition_known']=True;bad=density(torch.tensor([[2.,2.],[2.,2.]]),'1H',.2);ff=features(q,dict(given=kernel,**{name:bad for name in PROTOTYPES}),torch.tensor([10.,12.]),{},dict(solvent_prior=[.7,.1,.2],dropout={}))
        for sigma in [0.,.1,.2]:
            for name in ['train_mixture','uniform_prototypes','train_prototypes']:torch.testing.assert_close(ff['nll'][name,sigma],ff['nll']['given',sigma],rtol=0,atol=0)

    def test_zero_prior_condition_is_exactly_excluded(self):
        kernel,q=self.setup_query();bad=density(torch.tensor([[2.,2.],[2.,2.]]),'1H',.2)
        ff=features(q,dict(given=kernel,**{name:bad for name in PROTOTYPES}),torch.tensor([10.,12.]),{},dict(solvent_prior=[.7,.1,.2],dropout={}))
        torch.testing.assert_close(ff['nll']['train_prototypes',0.],weighted_log_score(bad,q),rtol=1e-6,atol=1e-6)

    def test_windows_and_integrals_keep_original_scoring_route(self):
        _,q=self.setup_query();self.assertTrue(eligible(q));q['observation']['windows']=[[0.,5.]];self.assertFalse(eligible(q));q['observation']['windows']=[];q['measure']='integral_weighted';self.assertFalse(eligible(q))


if __name__=='__main__':unittest.main()

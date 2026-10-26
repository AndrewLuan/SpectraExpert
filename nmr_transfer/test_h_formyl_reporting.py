"""甲酰/醛CH潜在报告过程的同步数学测试，不使用实测峰或外部服务。
检查整谱状态边缘化、空支持/混合环境回退、训练重复结构均权和空支持收缩。
"""
import unittest
import torch
from .h_formyl_reporting import candidate_families,counterfactual_quantiles,mixture_score,fit_probability
from .h_formyl_workflow import reporting_kernel,nearest_tags,solvent_group
from .h_conditions import encode_conditions
from .likelihood import density


class TestFormylReporting(unittest.TestCase):
    def test_families_and_no_empty_support(self):
        fs=candidate_families(['CN(C)C=O','CCC=O','CCO','O=CCNC=O'])
        self.assertEqual(fs,['formamide','aldehyde','other','other'])
        p=torch.tensor([[.1,.1,.2,.8],[.8,.8,.8,.8],[.1,.2,.3,.4],[.1,.2,.3,.8]])
        tags=torch.tensor([[0,0,0,4],[4,4,4,4],[0,0,0,0],[0,0,0,4]])
        q,active=counterfactual_quantiles(p,tags,fs);self.assertEqual(active.tolist(),[True,False,False,False]);self.assertLess(float(q[0].max()),.8);torch.testing.assert_close(q[1:],p[1:])

    def test_integrates_one_state_for_whole_observation_and_zero_strength_is_identity(self):
        a=torch.tensor([2.,4.]);b=torch.tensor([1.,5.]);p=.8;k=3
        expected=-torch.log(p*torch.exp(-k*a)+(1-p)*torch.exp(-k*b))/k
        torch.testing.assert_close(mixture_score(a,b,p,k),expected)
        torch.testing.assert_close(mixture_score(a,b,p,k,0),a,rtol=0,atol=0)
        torch.testing.assert_close(mixture_score(a,b,1.,k),a,rtol=0,atol=0)
        self.assertTrue(bool((mixture_score(a,a,p,k)-a).abs().max()<1e-6))

    def test_training_structure_weight_and_empty_prior(self):
        a=dict(group='a',full_loglike=-1.,missing_loglike=-10.);b=dict(group='b',full_loglike=-10.,missing_loglike=-1.)
        one=fit_probability([a,b]);repeat=fit_probability([a]*10+[b]);self.assertAlmostEqual(one['probability'],repeat['probability'],places=10)
        self.assertFalse(fit_probability([])['identified']);self.assertEqual(fit_probability([],prior=.7)['probability'],.7)

    def test_reporting_kernel_normalizes_and_keeps_unaffected_candidates(self):
        p=torch.tensor([[.1,.1,.2,.8],[.1,.2,.3,.4]]);tags=torch.tensor([[0,0,0,4],[0,0,0,0]])
        kernel,active=reporting_kernel(p,p,tags,['formamide','other'],dict(kernel='mass'))
        self.assertEqual(active.tolist(),[True,False]);self.assertTrue(bool((kernel['logp'].exp().sum(1)*kernel['step']-1).abs().max()<.001))
        torch.testing.assert_close(kernel['logp'][1],density(p,'1H',.2)['logp'][1],rtol=0,atol=0)
        torch.testing.assert_close(nearest_tags(p,p,tags),tags)
        key=(0.,0.,0.,0.)+tuple(encode_conditions(dict(solvent='C6D6'))['vector']);self.assertEqual(solvent_group(key),'benzene')


if __name__=='__main__':unittest.main()

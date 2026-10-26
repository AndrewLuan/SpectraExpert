"""候选局部异方差密度的同步合成测试，不读取真实数据或训练权重。
核对质量归一化、相同宽度退化为原KDE、环境标签CDF与原模型分位数一致，以及
训练似然只读取已标注目标。依赖h_uncertainty和原发射接口，无外部副作用。
"""
import unittest
import torch
from .h_uncertainty import class_density,quantile_classes,parent_grid_nll,select_widths,CLASSES,WIDTH_GRID
from .test_h_conditions import sample
from .h_condmodel import ConditionSpecGNN
from .likelihood import density


class TestUncertainty(unittest.TestCase):
    def test_same_width_matches_original_normalized_density(self):
        p=torch.tensor([[.1,.2,.5,.8],[.2,.21,.22,.4]]);tags=torch.tensor([[0,1,2,3],[4,5,6,7]])
        old=density(p,'1H',.2);new=class_density(p,tags,[.2]*len(CLASSES));torch.testing.assert_close(new['logp'],old['logp'],atol=3e-6,rtol=2e-6);torch.testing.assert_close(new['cdf'],old['cdf'],atol=1e-6,rtol=2e-6)
        mass=new['cdf'][:,-1];torch.testing.assert_close(mass,torch.ones_like(mass),atol=1e-4,rtol=0)

    def test_class_cdf_preserves_original_quantiles(self):
        b=sample();m=ConditionSpecGNN('1H',32,2,'local').eval();d=m(b,details=True);tags,q=quantile_classes(d,b);torch.testing.assert_close(q,d['quantiles'],rtol=0,atol=0);self.assertEqual(tags.shape,q.shape);self.assertTrue(bool((tags<9).all()))

    def test_frozen_values_are_classified_from_emission_support_without_resorting(self):
        b=sample();m=ConditionSpecGNN('1H',32,2,'local').eval();d=m(b,details=True);tags,q=quantile_classes(d,b);reference=q.flip(1);fixed,support=quantile_classes(d,b,reference=reference);torch.testing.assert_close(support,reference,rtol=0,atol=0);torch.testing.assert_close(fixed,tags.flip(1),rtol=0,atol=0)

    def test_empty_classes_shrink_to_original_width_and_unobserved_targets_ignored(self):
        b=sample();b.site_targets=torch.zeros(len(b.x),6);b.site_mask=torch.zeros(len(b.x),6,dtype=torch.bool);m=ConditionSpecGNN('1H',32,2,'local').eval();d=m(b,details=True);totals,counts=parent_grid_nll(d,b);self.assertEqual(int(counts.sum()),0)
        torch.testing.assert_close(torch.tensor(select_widths(totals,counts)),torch.full((len(CLASSES),),.2))
        b.site_targets.fill_(999);other,_=parent_grid_nll(d,b);torch.testing.assert_close(other,totals)


if __name__=='__main__':unittest.main()

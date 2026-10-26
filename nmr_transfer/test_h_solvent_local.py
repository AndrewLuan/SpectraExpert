"""功能概述：同步检查54参数苯/酰胺局部残差的作用域、恒等初始化及核质量。
使用合成分子和人工参数，不访问真实谱。确认未知/水/氯仿、非NMe2酰胺
严格保持原预测，明确苯中NMe2能更新；不承诺单个甲基的cis/trans归属。
"""
import unittest
import torch
from .h_solvent_local import SolventLocalSpecGNN,dimethyl_amide_mask
from .h_deep_context import DeepConditionSpecGNN
from .test_h_conditions import sample


class TestSolventLocal(unittest.TestCase):
    def test_initial_identity_gate_and_mass(self):
        base=DeepConditionSpecGNN('1H',32,2,'local','solvent',3).eval()
        model=SolventLocalSpecGNN('1H',32,2,'local','solvent',3).eval()
        missing,extra=model.load_state_dict(base.state_dict(),strict=False)
        self.assertFalse(extra);self.assertTrue(all(n.startswith('solvent_local_head.') for n in missing))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),54)
        b=sample();torch.testing.assert_close(model(b),base(b),rtol=0,atol=0)
        self.assertEqual(int(dimethyl_amide_mask(b).sum()),2)
        with torch.no_grad():model.solvent_local_head.bias.fill_(.3)
        self.assertFalse(torch.equal(model(b),base(b)))
        self.assertEqual(float(model(b,details=True)['weights'].sum()),7.)
        for solvent in ['', 'Water', 'CDCl3']:
            b=sample(solvent=solvent);torch.testing.assert_close(model(b),base(b),rtol=0,atol=0)
        b=sample('CNC=O');self.assertFalse(bool(dimethyl_amide_mask(b).any()))
        torch.testing.assert_close(model(b),base(b),rtol=0,atol=0)


if __name__=='__main__':unittest.main()

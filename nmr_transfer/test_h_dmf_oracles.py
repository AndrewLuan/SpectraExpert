"""DMF事后反事实转换的合成算术检查，不使用真实测试谱或做任何模型选择。
验证甲基中心/间距转换保留条目质量和甲酰位置，并避免原地更新覆盖另一个甲基。
依赖h_dmf_oracles.methyl_oracle，无真实推断、文件或网络副作用。
"""
import unittest
import torch
from .h_dmf_oracles import methyl_oracle


class TestDMFOracle(unittest.TestCase):
    def test_center_update_preserves_mass_formyl_and_independent_masks(self):
        p=torch.tensor([.2,.2,.2,.3,.3,.3,.8]);updated=methyl_oracle(p,[3.,4.],center=True);torch.testing.assert_close(updated,torch.tensor([.3,.3,.3,.4,.4,.4,.8]));torch.testing.assert_close(p,torch.tensor([.2,.2,.2,.3,.3,.3,.8]));self.assertEqual(updated.shape,p.shape)


if __name__=='__main__':unittest.main()

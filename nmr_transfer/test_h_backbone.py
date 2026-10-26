"""逐级解冻容量的同步契约测试，无真实数据、训练任务或网络副作用。
核对控制/原发射头/末层/全GINE的参数边界，以及更新骨干时显式禁用旧缓存。
依赖h_backbonetrain.configure和ConditionSpecGNN，不能用来替代真实训练/检索验证。
"""
import unittest
from .h_condmodel import ConditionSpecGNN
from .h_backbonetrain import configure


class TestBackbone(unittest.TestCase):
    def test_progressive_unfreezing_and_cache_contract(self):
        sizes=[]
        for mode in ['control','heads','last','all']:
            m=ConditionSpecGNN('1H',32,2,'local');groups=configure(m,mode);enabled={n for n,p in m.named_parameters() if p.requires_grad};sizes.append(sum(p.numel() for p in m.parameters() if p.requires_grad));self.assertEqual(m.use_frozen_embedding,mode in ['control','heads'])
            self.assertEqual(sum(len(g['params']) for g in groups),len(enabled))
            if mode=='control':self.assertTrue(all(n.startswith('condition_head.') for n in enabled))
            if mode=='heads':self.assertTrue(any(n.startswith('shift_head.') for n in enabled));self.assertFalse(any(n.startswith('bond_emb_list.') for n in enabled))
            if mode=='last':self.assertTrue(any(n.startswith('bond_emb_list.1.') for n in enabled));self.assertFalse(any(n.startswith('bond_emb_list.0.') for n in enabled));self.assertFalse(any(n.startswith('init_atom_emb.') for n in enabled))
            if mode=='all':self.assertTrue(all(p.requires_grad for p in m.parameters()))
        self.assertTrue(all(a<b for a,b in zip(sizes,sizes[1:])))


if __name__=='__main__':unittest.main()

"""新增上下文块的同步合成检查，无真实谱或GPU任务。
核对各深度零门控精确初始化、门控学习梯度、最终缓存不重复传播与旧骨干所有权。
依赖h_deep_context/原条件模型，不改变数据或调用外部服务。
"""
import unittest
import torch
from .h_deep_context import DeepConditionSpecGNN,verify_cached_backbone,DEPTHS
from .h_response import AnchoredSpecGNN
from .test_h_conditions import sample


class TestDeepContext(unittest.TestCase):
    def test_zero_gates_exactly_preserve_initial_model_and_capacity_layout(self):
        b=sample();base=AnchoredSpecGNN('1H',32,2,'local','solvent').eval();counts=[]
        for depth in DEPTHS:
            m=DeepConditionSpecGNN('1H',32,2,'local','solvent',depth).eval();missing,extra=m.load_state_dict(base.state_dict(),strict=False);self.assertFalse(extra);self.assertTrue(all(n.startswith(('context_bonds.','context_blocks.','context_gates')) for n in missing));torch.testing.assert_close(m(b),base(b),rtol=0,atol=0);counts.append(sum(p.numel() for p in m.parameters()))
        self.assertEqual(len(set(counts)),1)

    def test_context_gates_learn_and_final_cache_is_not_applied_twice(self):
        b=sample();m=DeepConditionSpecGNN('1H',32,2,'local','solvent',3).eval();m(b).sum().backward();self.assertGreater(float(m.context_gates.grad.abs().sum()),0);self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if n.startswith(('init_atom_emb.','shift_head.','split_head.'))))
        with torch.no_grad():m.context_gates.fill_(.2);h=m.structure_embedding(b);expected=m(b)
        b.final_context_embedding=h;torch.testing.assert_close(m(b),expected,rtol=0,atol=0)

    def test_old_GINE_cache_rejects_changed_weights(self):
        m=DeepConditionSpecGNN('1H',32,2);state=m.state_dict();reference={k:v.clone() for k,v in state.items()};verify_cached_backbone(state,reference);key=next(k for k in reference if k.startswith('init_atom_emb.'));reference[key].add_(.1)
        with self.assertRaisesRegex(ValueError,'embedding_owner'):verify_cached_backbone(state,reference)


if __name__=='__main__':unittest.main()

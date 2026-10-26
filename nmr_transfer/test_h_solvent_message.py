"""溶剂消息及酰基表征的同步契约测试，不读取训练谱或外部服务。
检查结构分类、同布局消融、暖启动恒等、H质量、未知溶剂掩码、条件敏感消息
和跨条件缓存正确性；另核对新增支持的固定曝光质量及结构均权。
所有数值是随机初始化合成输入，不是DMF实验位移。
"""
import unittest
import torch
from unittest.mock import patch
from .h_acyl import ACYL_NAMES,acyl_environment,carbonyl_h_sites
from .h_solvent_message import SolventMessageSpecGNN,MESSAGE_MODES
from .h_deep_context import DeepConditionSpecGNN
from .h_conditions import encode_conditions
from .test_h_conditions import sample
from .h_condinfer import predict
from .h_message_support import extra_weights


class TestSolventMessage(unittest.TestCase):
    def test_extra_support_preserves_base_distribution_and_connectivity_mass(self):
        w=extra_weights(torch.tensor([1.,3.]),[dict(group='a'),dict(group='a'),dict(group='b')],.1)
        torch.testing.assert_close(w[:2],torch.tensor([.225,.675],dtype=torch.float64));self.assertAlmostEqual(float(w[2]+w[3]),float(w[4]));self.assertAlmostEqual(float(w.sum()),1.)

    def test_acyl_classes_reach_N_methyl_and_formyl_is_distinct(self):
        for s,tag in [('CN(C)C=O','formyl'),('CC(=O)N(C)C','alkyl'),('CN(C)C(=O)c1ccccc1','aryl'),('C=CC(=O)N(C)C','vinyl'),('C#CC(=O)N(C)C','alkynyl')]:
            env=acyl_environment(s);self.assertGreater(float(env[:,ACYL_NAMES.index(tag)].sum()),0)
            self.assertEqual(bool(carbonyl_h_sites(s)),tag=='formyl')
        self.assertEqual(set(carbonyl_h_sites('CCC=O').values()),{'aldehyde'})
        self.assertEqual(set(carbonyl_h_sites('CCNC=O').values()),{'formamide'})

    def test_identity_equal_capacity_mass_and_frozen_body(self):
        b=sample();base=DeepConditionSpecGNN('1H',32,2,'local','solvent',3).eval();counts=[]
        for mode in MESSAGE_MODES:
            m=SolventMessageSpecGNN('1H',32,2,'local','solvent',3,mode).eval();missing,extra=m.load_state_dict(base.state_dict(),strict=False)
            self.assertFalse(extra);self.assertTrue(all(k.startswith('solvent_message_') for k in missing))
            torch.testing.assert_close(m(b),base(b),rtol=0,atol=0);self.assertAlmostEqual(float(m(b,details=True)['weights'].sum()),7.)
            counts.append(sum(p.numel() for p in m.parameters() if p.requires_grad));m(b).sum().backward()
            self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if not n.startswith('solvent_message_')))
        self.assertEqual(len(set(counts)),1)

    def test_condition_messages_recompute_from_shared_base_and_reject_final_cache(self):
        b=sample();m=SolventMessageSpecGNN('1H',32,2,'local','solvent',3).eval()
        with torch.no_grad():
            m.solvent_message_gates.fill_(.3);base=m.unconditioned_embedding(b).clone();benz=m.structure_embedding(b).clone()
            b.unconditioned_context_embedding=base;torch.testing.assert_close(m.structure_embedding(b),benz,rtol=0,atol=0)
            b.condition_vector=torch.tensor(encode_conditions(dict(solvent='CDCl3'))['vector']).reshape(1,-1)
            chloro=m.structure_embedding(b).clone();self.assertGreater(float((chloro-benz).abs().max()),1e-6)
            del b.unconditioned_context_embedding;torch.testing.assert_close(m.structure_embedding(b),chloro,rtol=0,atol=0)
            b.condition_vector=torch.tensor(encode_conditions({})['vector']).reshape(1,-1);self.assertEqual(float(m.message_condition(b).abs().sum()),0.)
            b.final_context_embedding=base
            with self.assertRaisesRegex(ValueError,'must_not_reuse'):m.structure_embedding(b)

    def test_candidate_inference_keeps_only_unconditioned_cache(self):
        b=sample();m=SolventMessageSpecGNN('1H',32,2,'local','solvent',3).eval()
        with torch.no_grad():m.solvent_message_gates.fill_(.3)
        keys=[tuple([0.]*4+encode_conditions(dict(solvent=s))['vector'].tolist()) for s in ['benzene','CDCl3']]
        bank=dict(val=[dict(key=k) for k in keys],test=[],paired=[])
        with patch('nmr_transfer.h_condinfer.candidate_batches',return_value=[b]),patch('nmr_transfer.h_condinfer.condition_key',side_effect=lambda r,*args:r['key']):
            pp,_,tags=predict(m,bank,{},None,'recovered_local',torch.device('cpu'),lambda *args,**kw:None)
        self.assertFalse(hasattr(b,'final_context_embedding'));self.assertTrue(hasattr(b,'unconditioned_context_embedding'))
        self.assertGreater(float((pp[keys[0]]-pp[keys[1]]).abs().max()),1e-7);self.assertEqual(tags.keys(),pp.keys())


if __name__=='__main__':unittest.main()

"""锚定条件响应的同步合成边界测试，不读取真实谱或训练数据。
核对未知条件零响应、pH7锚定、已知pH梯度、等容量消融及旧参数冻结。
依赖h_response和条件特征契约，由unittest执行，无持久化或网络副作用。
"""
import unittest
import torch
from .h_response import AnchoredSpecGNN,RESPONSE_MODES
from .h_conditions import encode_conditions
from .test_h_conditions import sample


class TestResponse(unittest.TestCase):
    def test_unknown_condition_and_ph7_exactly_preserve_base_after_learning(self):
        for mode in RESPONSE_MODES:
            m=AnchoredSpecGNN('1H',32,2,'local',mode).eval();torch.nn.init.normal_(m.response_head[-1].weight,std=.1);torch.nn.init.normal_(m.response_head[-1].bias,std=.1)
            for record in [{},dict(solvent='Water',experimental_conditions=dict(pH='7'))]:
                b=sample();b.condition_vector=torch.tensor(encode_conditions(record)['vector']).reshape(1,-1)
                context=torch.randn(len(b.x),76);shifts=torch.randn(len(b.x),6);logits=torch.randn(len(b.x),1)
                parent=super(AnchoredSpecGNN,m).adjust_conditions(shifts,logits,context,b);actual=m.adjust_conditions(shifts,logits,context,b)
                for x,y in zip(parent,actual):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_known_water_ph_response_has_gradient_and_factory_restores(self):
        m=AnchoredSpecGNN('1H',32,2,'local','local');b=sample();b.condition_vector=torch.tensor(encode_conditions(dict(solvent='Water',experimental_conditions=dict(pH='10')))['vector']).reshape(1,-1)
        loss=m(b).mean();loss.backward();self.assertGreater(float(m.response_head[-1].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if not n.startswith('response_head.')))
        from .emission_model import make_model
        restored=make_model(dict(model='anchored_atom',nucleus='1H',width=32,layers=2,condition_mode='local',response_mode='local'));restored.load_state_dict(m.state_dict());torch.testing.assert_close(restored(b),m(b))

    def test_response_modes_have_equal_capacity(self):
        counts=[sum(p.numel() for p in AnchoredSpecGNN('1H',32,2,'local',mode).parameters() if p.requires_grad) for mode in RESPONSE_MODES];self.assertEqual(len(set(counts)),1)


if __name__=='__main__':unittest.main()

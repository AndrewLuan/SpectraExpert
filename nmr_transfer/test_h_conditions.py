"""条件响应模型的独立合成契约测试，不读取真实谱或服务器数据。
核对溶剂名称、缺失/非水pH掩码、局部酰胺甲基，以及冻结节点表示和零初始化残差
保持原模型发射。所有消融参数量一致，只允许条件残差头训练；未知字段不引入真值。
依赖h_conditions、h_condmodel和旧图/发射接口，由unittest执行，无文件副作用。
"""
import unittest
import numpy as np
import torch
from torch_geometric.data import Batch
from .data import graph,physical_features,quantiles
from .h_conditions import encode_conditions,solvent_features,local_environment,ConditionsSpectra,CONDITION_DIM,SOLVENT_DIM,LOCAL_DIM,LOCAL_NAMES
from .h_condmodel import ConditionSpecGNN,MODES,training_details
from .emission_model import NuclearSpecGNN
from .h_condpairs import condition_pairs,pair_response_loss
from .h_labels import make_key


def sample(smiles='CN(C)C=O',solvent='benzene'):
    r=dict(smiles=smiles,source='synthetic',record_id='a',nucleus='1H',domain='experimental',cond=np.zeros(4,np.float32),physics=physical_features(smiles),q=quantiles([2.,3.,8.]),observation=dict(kind='unknown',windows=[]))
    g=graph(smiles);g.local_environment=torch.tensor(local_environment(smiles));r['condition_vector']=encode_conditions(dict(solvent=solvent))['vector']
    return Batch.from_data_list([ConditionsSpectra([r],{smiles:g},'1H',metadata={})[0]])


class TestConditions(unittest.TestCase):
    def test_methanol_is_not_ethanol_and_mixture_is_multihot(self):
        _,tags=solvent_features('methanol');self.assertEqual(tags,['methanol'])
        _,tags=solvent_features('acetone+DMSO');self.assertEqual(set(tags),{'acetone','dmso'})

    def test_ph_requires_explicit_water_and_value(self):
        water=encode_conditions(dict(solvent='D2O',experimental_conditions=dict(pH='7.00')))
        absent=encode_conditions(dict(solvent='D2O'))
        chloro=encode_conditions(dict(solvent='CDCl3',experimental_conditions=dict(pH='7.00')))
        self.assertTrue(water['pH_used']);self.assertFalse(absent['pH_used']);self.assertFalse(chloro['pH_used'])
        self.assertEqual(float(water['vector'][SOLVENT_DIM]),1.);self.assertTrue(np.all(absent['vector'][SOLVENT_DIM:]==0))

    def test_explicit_amide_methyl_environment(self):
        env=local_environment('CN(C)C=O');self.assertEqual(env.shape[1],LOCAL_DIM)
        self.assertEqual(int(env[:,LOCAL_NAMES.index('amide_N_methyl')].sum()),2)
        self.assertEqual(int(local_environment('CN(C)C')[:,LOCAL_NAMES.index('amide_N_methyl')].sum()),0)

    def test_zero_adapter_and_frozen_embeddings_preserve_emissions(self):
        torch.manual_seed(1);base=NuclearSpecGNN('1H',32,2).eval();m=ConditionSpecGNN('1H',32,2,'local').eval()
        missing,unexpected=m.load_state_dict(base.state_dict(),strict=False);self.assertFalse(unexpected);self.assertTrue(all(k.startswith('condition_head.') for k in missing))
        b=sample();expected=base(b);torch.testing.assert_close(m(b),expected)
        b.frozen_embedding=base.structure_embedding(b).detach();torch.testing.assert_close(m(b),expected)
        self.assertTrue(all(p.requires_grad==name.startswith('condition_head.') for name,p in m.named_parameters()))

    def test_observed_all_exchange_water_spectrum_is_not_made_empty(self):
        b=sample('NC(N)=O','Water');b.line_water.fill_(True)
        m=ConditionSpecGNN('1H',32,2,'local').eval();d=m(b,details=True)
        visible=training_details(d,b)
        self.assertTrue(bool(d['exchangeable'].all()))
        torch.testing.assert_close(visible['quantiles'],d['quantiles'])
        self.assertTrue(bool((visible['quantiles'].abs()<100).all()))

    def test_pair_construction_excludes_test_identity_and_isotope_confounding(self):
        rows=[];meta={}
        for rid,solvent,ph,split in [('HMDB:X:1','Water','7','train'),('HMDB:X:2','Water','9','train'),('HMDB:X:3','D2O','11','train'),('HMDB:X:4','Water','12','test'),('HMDB:Y:5','Water','10','train')]:
            r=dict(smiles='CCN',source='HMDB',domain='experimental',record_id=rid,split=split);m=encode_conditions(dict(solvent=solvent,experimental_conditions=dict(pH=ph)));m['line_centers']=[1.,3.];rows.append(r);meta[make_key(r)]=m
        pairs=condition_pairs(rows,meta)
        self.assertEqual(len(pairs),1);self.assertEqual(pairs[0]['kind'],'pH')
        self.assertEqual({pairs[0]['a']['record_id'],pairs[0]['b']['record_id']},{'HMDB:X:1','HMDB:X:2'})

    def test_pair_loss_backpropagates_condition_response_only(self):
        rows=[];meta={};s='CCN';g=graph(s);g.local_environment=torch.tensor(local_environment(s))
        for i,ph in enumerate([7.,9.]):
            r=dict(smiles=s,source='HMDB',domain='experimental',record_id='HMDB:X:'+str(i),split='train',physics=physical_features(s),cond=np.array([0.,1.,0.,0.],np.float32),q=quantiles([1.,3.]),observation=dict(kind='unknown',windows=[]))
            m=encode_conditions(dict(solvent='Water',experimental_conditions=dict(pH=str(ph))));m['line_centers']=[1.+i*.3,3.+i*.3];rows.append(r);meta[make_key(r)]=m
        m=ConditionSpecGNN('1H',32,2,'ph');loss=pair_response_loss(m,condition_pairs(rows,meta),{s:g},meta,torch.device('cpu'));self.assertGreater(float(loss),0);loss.backward()
        self.assertTrue(torch.isfinite(m.condition_head[-1].weight.grad).all());self.assertGreater(float(m.condition_head[-1].weight.grad.abs().sum()),0)

    def test_amide_gap_calibration_preserves_center_and_nuclear_mass(self):
        b=sample();m=NuclearSpecGNN('1H',32,2).eval();d=m(b,details=True)
        parents=b.local_environment[:,LOCAL_NAMES.index('amide_N_methyl')]>0;active=parents[d['owners']];before=d['values'][active]
        m.amide_gap_scale=.5;after=m(b,details=True)
        torch.testing.assert_close(after['weights'],d['weights'])
        self.assertAlmostEqual(float(after['weights'].sum()),7.,places=6)
        self.assertAlmostEqual(float(after['values'][active].mean()),float(before.mean()),places=6)
        self.assertAlmostEqual(float(after['values'][active].max()-after['values'][active].min()),float(before.max()-before.min())*.5,places=5)
        torch.testing.assert_close(after['values'][~active],d['values'][~active])

    def test_equal_capacity_and_control_ignores_extra_metadata(self):
        b=sample();models=[ConditionSpecGNN('1H',32,2,mode).eval() for mode in MODES]
        self.assertEqual(len({sum(p.numel() for p in m.parameters() if p.requires_grad) for m in models}),1)
        m=models[0];torch.nn.init.normal_(m.condition_head[-1].weight,std=.01)
        before=m(b);b.condition_vector.fill_(123);b.local_environment.fill_(99);torch.testing.assert_close(m(b),before)
        output=m(b).sum();output.backward();self.assertTrue(all(p.grad is None for name,p in m.named_parameters() if not name.startswith('condition_head.')))

    def test_updated_GINE_must_ignore_old_structure_cache(self):
        torch.manual_seed(2);b=sample();m=ConditionSpecGNN('1H',32,2,'local').eval();b.frozen_embedding=m.structure_embedding(b).detach();old=m(b).detach()
        weight=next(p for p in m.init_atom_emb.parameters() if p.ndim==2)
        with torch.no_grad():weight.add_(torch.randn_like(weight)*.1)
        torch.testing.assert_close(m(b),old,rtol=0,atol=0);m.use_frozen_embedding=False;updated=m(b).detach();self.assertGreater(float((updated-old).abs().max()),1e-7)
        b.frozen_embedding=m.structure_embedding(b).detach();m.use_frozen_embedding=True;torch.testing.assert_close(m(b),updated,rtol=1e-6,atol=1e-7)


if __name__=='__main__':unittest.main()

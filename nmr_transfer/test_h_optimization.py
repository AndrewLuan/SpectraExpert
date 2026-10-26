"""¹H优化的独立契约检查：源索引映射、位点归属约束及部分标注不产生缺失负证据。
只用人工甲醇/乙醇与合成位移，不读取真实训练/val/test；还核对扩展接口的默认预测不变。
测试h_labels/source_groups、train.Spectra、h_objective.site_loss与emission_model接口。
"""
import unittest
import itertools
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch
from .data import graph,physical_features,quantiles
from .train import Spectra
from .h_labels import make_key,source_groups
from .h_objective import site_loss,objective
from .emission_model import NuclearSpecGNN
from .h_score import partial_transport,count_score,attach_observation,query_config
from .evaluate import missing_queries
from .observation import coverage


def example(groups):
    s='CO';r=dict(smiles=s,nucleus='1H',domain='experimental',source='synthetic',record_id='0',
        q=quantiles([1.,1.,1.,4.]),physics=physical_features(s),cond=np.zeros(4,np.float32),observation=dict(kind='unknown',windows=[]))
    meta={make_key(r):dict(groups=groups,full_atom_labels=False)}
    return Batch.from_data_list([Spectra([r],{s:graph(s)},'1H',annotations=meta)[0]])


class TestHOptimization(unittest.TestCase):
    def test_sparse_annotations_are_not_integral_mass_in_routing(self):
        c=dict(kind='observation_router',full_or_unknown=dict(kind='partial_transport',beta=4),frequency_crop=dict(kind='partial_transport',beta=1),peak_dropout=dict(kind='likelihood',beta=0))
        sparse=dict(measure='observed_assigned_atom_equal_partial_possible',observation=dict(kind='unknown',windows=[]))
        self.assertEqual(query_config(c,sparse)['kind'],'likelihood')
        integral=dict(measure='integral_weighted',observation=coverage(dict(peak_coverage='partial')))
        self.assertEqual(query_config(c,integral)['kind'],'likelihood')
        integral['observation']=coverage({})
        self.assertEqual(query_config(c,integral)['beta'],4)

    def test_partial_transport_matches_exhaustive_injective_assignment(self):
        atoms=torch.tensor([[1.,3.,4.,8.],[0.,2.,5.,9.]])
        observed=torch.tensor([4.1,1.2]);mask=torch.ones_like(atoms,dtype=torch.bool)
        expected=[min(sum(abs(a-x) for a,x in zip(sub,sorted(observed.tolist())))/2 for sub in itertools.combinations(row,2)) for row in atoms.tolist()]
        torch.testing.assert_close(partial_transport(atoms,mask,observed),torch.tensor(expected))

    def test_known_window_does_not_count_unmeasured_nuclei(self):
        atoms=torch.tensor([[1.,3.,9.],[1.,3.,4.]])
        query=dict(source='synthetic',observation=dict(kind='partial_window',windows=[[0.,3.5]]))
        score=count_score(atoms,torch.ones_like(atoms,dtype=torch.bool),query,2,{})
        torch.testing.assert_close(score[0],score[1])

    def test_full_report_count_component_is_deterministic(self):
        atoms=torch.tensor([[1.,2.,3.],[1.,2.,3.]])
        mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
        query=dict(source='synthetic',observation=dict(kind='unknown',windows=[]))
        prior={'synthetic':dict(rates=[1.],weights=[1.])}
        score=count_score(atoms,mask,query,2,prior)
        self.assertAlmostEqual(float(score[0]),0.,places=6);self.assertGreater(float(score[1]),100.)

    def test_query_observation_does_not_use_true_structure_counts(self):
        query=dict(source='synthetic')
        a=attach_observation(query,dict(xs=[1.,4.],weights=[3.,1.],structural_H=4,full_atom_labels=True))
        b=attach_observation(query,dict(xs=[1.,4.],weights=[3.,1.],structural_H=99,full_atom_labels=False))
        self.assertEqual(a,b)

    def test_missing_stress_preserves_observed_mass(self):
        r=dict(record_id='0',smiles='CO',q=quantiles([1.,4.,9.],[3.,2.,1.]),observed_xs=[1.,4.,9.],observed_weights=[3.,2.,1.],observation=dict(kind='unknown',windows=[]))
        crop=[v for name,v in missing_queries(r,'1H') if name=='frequency_crop'][0]
        self.assertEqual(crop['observed_xs'],[1.,4.]);self.assertEqual(sum(crop['observed_weights']),5.)

    def test_raw_to_canonical_parent_mapping(self):
        s='OCC';m=Chem.AddHs(Chem.MolFromSmiles(s));r=dict(smiles=s,atom_order=dict(source_to_canonical=list(range(len(m.GetAtoms()))),atomic_numbers=[a.GetAtomicNum() for a in m.GetAtoms()]))
        canonical,groups=source_groups(r)
        self.assertEqual(canonical,'CCO');counts={i:sum(parent==i for parent,_ in groups.values()) for i in range(3)}
        self.assertEqual(counts,{0:3,1:2,2:1})
        r['atom_order']['atomic_numbers'][0]=6
        with self.assertRaisesRegex(ValueError,'source_order_mismatch'):source_groups(r)

    def test_aromatic_nitrogen_mapping_preserves_attached_hydrogen(self):
        s='CCCCOC(C)c1nn[nH]n1';m=Chem.AddHs(Chem.MolFromSmiles(s))
        r=dict(smiles=s,atom_order=dict(source_to_canonical=list(range(m.GetNumAtoms())),atomic_numbers=[a.GetAtomicNum() for a in m.GetAtoms()]))
        canonical,groups=source_groups(r);target=Chem.AddHs(Chem.MolFromSmiles(canonical))
        for parent,_ in groups.values():self.assertTrue(any(a.GetAtomicNum()==1 for a in target.GetAtomWithIdx(parent).GetNeighbors()))

    def test_identical_global_spectrum_wrong_parent_is_penalized(self):
        b=example({0:[1.,1.,1.],1:[4.]});owner=torch.tensor([0,0,0,1]);w=torch.ones(4)
        def d(v):return dict(values=torch.tensor(v,requires_grad=True),weights=w,owners=owner)
        correct,_=site_loss(d([.1,.1,.1,.4]),b);wrong,_=site_loss(d([.4,.1,.1,.1]),b)
        self.assertLess(float(correct[0]),1e-6);self.assertGreater(float(wrong[0]),.1)

    def test_partial_labels_do_not_punish_unobserved_components(self):
        b=example({0:[1.]});v=torch.tensor([.1,.3,.5,.4],requires_grad=True)
        loss,eligible=site_loss(dict(values=v,weights=torch.ones(4),owners=torch.tensor([0,0,0,1])),b)
        self.assertTrue(bool(eligible[0]));self.assertLess(float(loss[0]),1e-6)
        loss.sum().backward();self.assertTrue(torch.isfinite(v.grad).all())

    def test_details_are_prediction_equivalent_and_ignore_annotations(self):
        b=example({0:[1.,1.,1.],1:[4.]});model=NuclearSpecGNN('1H',32,2).eval()
        normal=model(b);details=model(b,details=True)
        torch.testing.assert_close(normal,details['quantiles'])
        b.site_targets.fill_(99);torch.testing.assert_close(normal,model(b))
        loss=objective(details,b)['loss'];loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__':unittest.main()

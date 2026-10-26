"""功能概述：验证¹H质量审计和跨版本置换更正的证据边界。
合成分子由RDKit生成坐标及源索引，测试真实母节点错误、CH2非等价标签、缺失掩码、
理论整体偏差、不同谱/编号/测试分区和伪造补丁。断言更正保留位移多重集和原对象，
并拒绝仅靠模型偏差、标签平均或跨测试泄漏进行更正；不访问研究数据或网络。
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from .data import identity
from .h_assignment_audit import (H_bond_upper_bound, apply_repair, apply_reviewed_repair,
                                 copy_and_verify, find_repairs, inspect_record,
                                 spectrum_multiset)


class AssignmentAuditTests(unittest.TestCase):
    def pair(self, split='train'):
        smi = next('C'*n+'Oc1ccccc1' for n in range(2, 30)
                   if identity('C'*n+'Oc1ccccc1')[2] == split)
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=173), 0)
        vals = []; mask = []
        methyl = aromatic = None
        for a in mol.GetAtoms():
            if a.GetAtomicNum() != 1:
                vals.append(None); mask.append(0); continue
            parent = a.GetNeighbors()[0]
            mh = sum(x.GetAtomicNum() == 1 for x in parent.GetNeighbors())
            value = 7. if parent.GetIsAromatic() else 1. if mh == 3 else 3.
            vals.append(value); mask.append(1)
            if mh == 3:
                methyl = a.GetIdx()
            if parent.GetIsAromatic():
                aromatic = a.GetIdx()
        donor = dict(record_id='new', source_ref='nmrshiftdb2_2024', nucleus='1H',
            source_database_id='synthetic', smiles=smi, conditions={},
            atoms=[a.GetSymbol() for a in mol.GetAtoms()],
            atom_order=dict(source_to_canonical=list(range(mol.GetNumAtoms())),
                            atomic_numbers=[a.GetAtomicNum() for a in mol.GetAtoms()]),
            coordinates=mol.GetConformer().GetPositions().tolist(),
            atom_targets=vals, atom_target_mask=mask,
            spectrum=[dict(atom_index=i, shift=v) for i,v in enumerate(vals) if mask[i]])
        old = copy.deepcopy(donor)
        old.update(record_id='old', source_ref='nmrshiftdb2_2018')
        old['atom_targets'][methyl], old['atom_targets'][aromatic] = (
            old['atom_targets'][aromatic], old['atom_targets'][methyl])
        self.sync(old)
        return old, donor

    @staticmethod
    def sync(r):
        r['spectrum'] = [dict(atom_index=i, shift=v) for i,(v,m) in
                         enumerate(zip(r['atom_targets'],r['atom_target_mask'])) if m]

    @staticmethod
    def proposals(*records):
        return find_repairs(records, {r['record_id']:inspect_record(r) for r in records})

    def test_evidenced_swap_preserves_source_and_spectrum(self):
        old, donor = self.pair(); snapshot = copy.deepcopy(old)
        patches = self.proposals(old,donor); self.assertEqual(len(patches),1)
        fixed = apply_repair(old,patches[0],donor)
        self.assertEqual(fixed['atom_targets'],donor['atom_targets'])
        self.assertEqual(spectrum_multiset(fixed),spectrum_multiset(old))
        self.assertEqual(old,snapshot)
        self.assertEqual(old['atom_target_mask'],fixed['atom_target_mask'])
        self.assertEqual(old['atom_order'],fixed['atom_order'])
        self.assertEqual(inspect_record(fixed)['classification'],'no_detected_local_conflict')

    def test_different_spectra_or_order_are_not_evidence(self):
        for mode in ('shift','order','mask','experimental_conditions','solvent'):
            old, donor = self.pair()
            if mode == 'shift':
                i=next(i for i,m in enumerate(donor['atom_target_mask']) if m)
                donor['atom_targets'][i] += .1; self.sync(donor)
            elif mode == 'order':
                donor['atom_order']['basis']='different_contract'
            elif mode == 'mask':
                i=next(i for i,m in enumerate(donor['atom_target_mask']) if m)
                donor['atom_target_mask'][i]=0; donor['atom_targets'][i]=None; self.sync(donor)
            else:
                donor[mode]={'solvent':'different'} if mode=='experimental_conditions' else 'different'
            self.assertEqual(self.proposals(old,donor),[])

    def test_test_labels_and_validation_corrections_are_blocked(self):
        for split in ('val','test'):
            old, donor = self.pair(split)
            self.assertEqual(self.proposals(old,donor),[])
            if split == 'test':
                old['atom_targets']=['intentionally_unreadable']
                self.assertEqual(inspect_record(old)['classification'],'held_out_not_label_audited')

    def test_ch2_splitting_and_theory_domain_offset_do_not_relabel(self):
        _, donor = self.pair()
        a=inspect_record(donor)
        ch2=next(g for g in a['groups'] if len(g['source_H'])==2)
        i,j=ch2['source_H']; donor['atom_targets'][i]=2.; donor['atom_targets'][j]=4.
        self.sync(donor)
        theory=dict(smiles=donor['smiles'],value=dict(atom_index=[i+1 for i,m in enumerate(donor['atom_target_mask']) if m],
            x=[v+3 for v,m in zip(donor['atom_targets'],donor['atom_target_mask']) if m]))
        a=inspect_record(donor,theory)
        self.assertEqual(a['classification'],'no_detected_local_conflict')
        self.assertGreater(a['raw_theory_parent_mean_MAE_ppm'],2.9)
        self.assertEqual(a['methyl_conflicts'],[])
        theory['smiles']='C'
        a=inspect_record(donor,theory)
        self.assertEqual(a['classification'],'no_detected_local_conflict')
        self.assertEqual(a['theory_diagnostic_issue'],'theory_structure_mismatch')

    def test_theory_diagnostic_never_changes_patch_eligibility(self):
        old,donor=self.pair()
        expected=self.proposals(old,donor)
        wrong=dict(smiles='C',value=dict(atom_index=[1],x=[999.]))
        audits={r['record_id']:inspect_record(r,wrong) for r in (old,donor)}
        self.assertEqual(find_repairs([old,donor],audits),expected)

    def test_missing_labels_are_not_filled_or_zeroed(self):
        _, donor = self.pair(); a=inspect_record(donor)
        i=next(g for g in a['groups'] if g['ordinary_methyl'])['source_H'][0]
        donor['atom_target_mask'][i]=0; donor['atom_targets'][i]=None; self.sync(donor)
        a=inspect_record(donor)
        self.assertEqual(a['observed_H'],a['structural_H']-1)
        self.assertFalse(next(g for g in a['groups'] if i in g['source_H'])['full_parent_observed'])
        self.assertEqual(a['classification'],'no_detected_local_conflict')

    def test_partial_parent_does_not_create_false_theory_residual(self):
        _,r=self.pair(); a=inspect_record(r)
        hs=next(g for g in a['groups'] if len(g['source_H'])==2)['source_H']
        i,j=hs; r['atom_targets'][i]=2.; r['atom_targets'][j]=None; r['atom_target_mask'][j]=0
        self.sync(r)
        t=dict(smiles=r['smiles'],value=dict(atom_index=[i+1,j+1],x=[2.,4.]))
        a=inspect_record(r,t); g=next(g for g in a['groups'] if i in g['source_H'])
        self.assertNotIn('raw_theory_parent_mean_MAE_ppm',a)
        self.assertEqual(g['theory_diagnostic_skipped'],'partial_parent_observation')

    def test_copy_boundary_includes_nucleus_and_preserves_crlf(self):
        old,donor=self.pair(); patch=self.proposals(old,donor)[0]
        corrected=apply_repair(old,patch,donor)
        other=dict(old,nucleus='13C')
        raw=(json.dumps(other)+'\r\n').encode()
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source'; output=Path(d)/'output'
            source.write_bytes(raw+(json.dumps(old)+'\n').encode())
            counts=copy_and_verify(source,output,{'old':corrected},{'old':inspect_record(old)})
            self.assertEqual(counts['changed'],1); self.assertEqual(counts['unchanged'],1)
            self.assertTrue(output.read_bytes().startswith(raw))

    def test_geometry_upper_bound_respects_parent_element(self):
        self.assertEqual(H_bond_upper_bound(6),1.5)
        self.assertGreater(H_bond_upper_bound(34),1.54)
        self.assertGreater(H_bond_upper_bound(16),H_bond_upper_bound(6))

    def reviewed_case(self):
        smi='COc1cccc(OC)c1C=O'; mol=Chem.AddHs(Chem.MolFromSmiles(smi))
        self.assertEqual(identity(smi)[2],'train')
        self.assertEqual(AllChem.EmbedMolecule(mol,randomSeed=193),0)
        ranks=Chem.CanonicalRankAtoms(mol,breakTies=False,includeChirality=True)
        aromatic={}; formyl=None
        values=[]; mask=[]
        for a in mol.GetAtoms():
            if a.GetAtomicNum()!=1:
                values.append(None);mask.append(0);continue
            values.append(1.);mask.append(1);p=a.GetNeighbors()[0]
            if p.GetIsAromatic():aromatic.setdefault(ranks[p.GetIdx()],[]).append(a.GetIdx())
            elif any(b.GetBondType()==Chem.BondType.DOUBLE and b.GetOtherAtom(p).GetAtomicNum()==8 for b in p.GetBonds()):formyl=a.GetIdx()
        eq=next(v for v in aromatic.values() if len(v)==2)
        single=next(v[0] for v in aromatic.values() if len(v)==1)
        indices=[formyl,*eq,single]
        for i,v in zip(indices,[4.,7.,4.,9.]):values[i]=v
        r=dict(record_id='review',source_ref='synthetic',source_database_id='review-id',
            smiles=smi,nucleus='1H',conditions={},atoms=[a.GetSymbol() for a in mol.GetAtoms()],
            atom_order=dict(source_to_canonical=list(range(mol.GetNumAtoms())),
                atomic_numbers=[a.GetAtomicNum() for a in mol.GetAtoms()]),
            coordinates=mol.GetConformer().GetPositions().tolist(),atom_targets=values,atom_target_mask=mask)
        self.sync(r)
        review=dict(record_id=r['record_id'],source_database_id=r['source_database_id'],
            source_indices=indices,before=[4.,7.,4.,9.],confidence='source_report_structurally_supported',
            independent_measurement_confirmation=False,reviewed_by=['a','b'],
            evidence_uri='https://example.invalid/test-fixture',
            constraints=dict(formyl_source_H=formyl,reported_formyl_ppm=9.,equal_source_H_groups=[eq]))
        return r,review

    def test_reviewed_constraints_produce_unique_local_permutation(self):
        r,review=self.reviewed_case();old=copy.deepcopy(r)
        u=apply_reviewed_repair(r,review);proof=u['assignment_correction']
        self.assertEqual(proof['after'],[9.,4.,4.,7.])
        self.assertEqual(proof['enumerated_distinct_assignments'],12)
        self.assertEqual(len(proof['changed_source_indices']),3)
        self.assertEqual(spectrum_multiset(r),spectrum_multiset(u))
        self.assertEqual(r,old)
        for i in range(len(r['atom_targets'])):
            if i not in review['source_indices']:self.assertEqual(r['atom_targets'][i],u['atom_targets'][i])

    def test_reviewed_constraints_reject_nonunique_or_structurally_wrong_claim(self):
        for mode in ['nonunique','wrong_formyl','wrong_source','missing_review','claimed_independent']:
            r,review=self.reviewed_case()
            if mode=='nonunique':
                review['before']=[4.,7.,5.,9.]
                for i,v in zip(review['source_indices'],review['before']):r['atom_targets'][i]=v
                self.sync(r)
            elif mode=='wrong_formyl':review['constraints']['formyl_source_H']=review['source_indices'][-1]
            elif mode=='wrong_source':review['source_database_id']='other'
            elif mode=='missing_review':review['reviewed_by']=['one']
            else:review['independent_measurement_confirmation']=True
            with self.assertRaises(ValueError):apply_reviewed_repair(r,review)

    def test_tampered_patch_and_wrong_spectrum_are_rejected(self):
        old, donor = self.pair(); patch=self.proposals(old,donor)[0]
        patch['after'][0]=999.
        with self.assertRaisesRegex(ValueError,'repair_evidence_mismatch'):
            apply_repair(old,patch,donor)
        donor['spectrum'][0]['shift'] += 1.
        self.assertEqual(inspect_record(donor)['classification'],'formal_contract_error')


if __name__ == '__main__':
    unittest.main()

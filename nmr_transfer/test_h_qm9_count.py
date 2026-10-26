"""功能概述：用新生成的合成坐标验证QM9无标签结构对齐、原子重排与错配拒绝。
只验证图/坐标契约，不调用真实QM9标签、物理求解器或外部服务。
"""
import unittest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from .h_qm9_count import structure_only


class QM9AlignmentTests(unittest.TestCase):
    def fixture(self):
        m=Chem.AddHs(Chem.MolFromSmiles('CCO'));self.assertEqual(AllChem.EmbedMolecule(m,randomSeed=37),0);AllChem.UFFOptimizeMolecule(m);return dict(smiles='CCO',record_id='synthetic',atoms=[a.GetSymbol() for a in m.GetAtoms()],coordinates=m.GetConformer().GetPositions().tolist())
    def test_mapping_permutation(self):
        r=self.fixture();permutation=np.random.default_rng(37).permutation(len(r['atoms']));shuffled=dict(r,atoms=[r['atoms'][i] for i in permutation],coordinates=[r['coordinates'][i] for i in permutation]);s=structure_only(shuffled);self.assertEqual(sorted(s['source_to_canonical']),list(range(9)));self.assertEqual(s['nuclei'],['1H']);self.assertNotIn('atom_targets',s);full=Chem.AddHs(Chem.MolFromSmiles(r['smiles']));self.assertEqual([full.GetAtomWithIdx(i).GetAtomicNum() for i in s['source_to_canonical']],s['tokens'])
    def test_wrong_structure(self):
        r=self.fixture();r['smiles']='COC'
        with self.assertRaises(ValueError):structure_only(r)
    def test_missing_coordinate(self):
        r=self.fixture();r['coordinates'].pop()
        with self.assertRaises(ValueError):structure_only(r)


if __name__=='__main__':unittest.main()

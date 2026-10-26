"""功能概述：合成乱序H坐标契约下验证候选标签母节点解释与原观测多重集隔离。
无实验标签或真实分子位移常数；映射必须与位移数值无关，缺失掩码保持原样。
"""
import copy,unittest
from rdkit import Chem
from .h_hydrogen_order import label_parents,annotation


class HydrogenOrderTests(unittest.TestCase):
    def record(self):
        m=Chem.AddHs(Chem.MolFromSmiles('CCO'));order=[0,1,2,8,3,4,5,6,7];return dict(smiles='CCO',atom_order=dict(source_to_canonical=order,atomic_numbers=[m.GetAtomWithIdx(i).GetAtomicNum() for i in order]),atom_targets=[None,None,None,1.,1.,1.,3.,3.,5.],atom_target_mask=[0,0,0,1,1,1,1,1,1])
    def test_parent_order_and_spectrum(self):
        r=self.record();original=copy.deepcopy(r);_,a=annotation(r,'coordinate_order');_,b=annotation(r,'heavy_parent_order');self.assertEqual(b['groups'],{0:[1.,1.,1.],1:[3.,3.],2:[5.]});self.assertNotEqual(a['groups'],b['groups']);self.assertEqual(sorted(x for xs in a['groups'].values() for x in xs),sorted(x for xs in b['groups'].values() for x in xs));self.assertEqual(r,original)
    def test_mapping_ignores_labels(self):
        r=self.record();mapping=label_parents(r,'heavy_parent_order');r['atom_targets']=[999.]*9;self.assertEqual(mapping,label_parents(r,'heavy_parent_order'))
    def test_missing_mask_preserved(self):
        r=self.record();r['atom_target_mask'][-1]=0;r['atom_targets'][-1]=None
        for mode in ['coordinate_order','heavy_parent_order']:
            _,a=annotation(r,mode);self.assertEqual(a['observed_H'],5);self.assertFalse(a['full_atom_labels'])


if __name__=='__main__':unittest.main()

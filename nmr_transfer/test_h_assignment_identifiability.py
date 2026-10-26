"""功能概述：以小规模穷举检验甲基标签可识别性DP和缺失/语义边界。
将allocation_support的可行性及每个值的用量上下界与独立暴力赋值对照；覆盖混合2/3H
母组、可识别单甲基、多母节点无序目标、无解和无约束。测试不调用API或研究数据。
"""
import collections
import copy
import itertools
import random
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from .data import identity
from .h_assignment_identifiability import allocation_support, identify


def brute(counts,n3,n2):
    values=list(counts);sizes=[3]*n3+[2]*n2;solutions=[]
    for targets in itertools.product(values,repeat=len(sizes)):
        used=collections.Counter()
        for v,k in zip(targets,sizes):used[v]+=k
        if all(used[v]<=counts[v] for v in used):solutions.append(targets)
    return solutions


class IdentifiabilityTests(unittest.TestCase):
    def source_record(self):
        smi=next('C'*n+'O' for n in range(2,12) if identity('C'*n+'O')[2]=='train')
        m=Chem.AddHs(Chem.MolFromSmiles(smi));self.assertEqual(AllChem.EmbedMolecule(m,randomSeed=173),0)
        mask=[int(a.GetAtomicNum()==1) for a in m.GetAtoms()]
        targets=[1. if v else None for v in mask]
        return dict(data_nature='experimental',nucleus='1H',record_id='synthetic',source_ref='synthetic',
            smiles=smi,atoms=[a.GetSymbol() for a in m.GetAtoms()],
            coordinates=m.GetConformer().GetPositions().tolist(),atom_targets=targets,
            atom_target_mask=mask,atom_order=dict(source_to_canonical=list(range(m.GetNumAtoms())),
                atomic_numbers=[a.GetAtomicNum() for a in m.GetAtoms()]),
            spectrum=[dict(atom_index=i,shift=v) for i,v in enumerate(targets) if mask[i]])

    def test_unique_single_and_unordered_multiple_groups(self):
        a=allocation_support({1.:3,6.:2,7.:2,9.:1,10.:1},1,0)
        self.assertTrue(a['feasible']);self.assertEqual(a['group_size_support'][3],[1.])
        b=allocation_support({1.:3,2.:3,4.:2,5.:1},2,0)
        self.assertEqual(b['group_size_support'][3],[1.,2.])
        self.assertTrue(all(v['group3_min']==v['group3_max']==1 for v in b['value_support']))

    def test_no_permutation_can_create_missing_repeated_values(self):
        self.assertFalse(allocation_support({1.:1,1.1:1,1.2:1},1,0)['feasible'])
        self.assertTrue(allocation_support({1.:1,1.1:1,1.2:1},0,0)['feasible'])

    def test_partial_group_competes_with_complete_group(self):
        a=allocation_support({1.:4,2.:2},1,1)
        self.assertEqual(a['group_size_support'][3],[1.])
        self.assertEqual(a['group_size_support'][2],[2.])

    def test_exact_support_matches_independent_exhaustive_assignments(self):
        rng=random.Random(173)
        for _ in range(100):
            counts={float(i):rng.randint(1,7) for i in range(rng.randint(1,4))}
            n3=rng.randint(0,2);n2=rng.randint(0,2)
            reference=brute(counts,n3,n2);actual=allocation_support(counts,n3,n2)
            self.assertEqual(actual['feasible'],bool(reference))
            if not reference:continue
            for row in actual['value_support']:
                v=row['ppm'];xs=[sum(x==v for x in r[:n3]) for r in reference]
                ys=[sum(x==v for x in r[n3:]) for r in reference]
                self.assertEqual((row['group3_min'],row['group3_max']),(min(xs),max(xs)))
                self.assertEqual((row['group2_min'],row['group2_max']),(min(ys),max(ys)))

    def test_experimental_scope_and_missing_mask_are_preserved(self):
        r=self.source_record();original=copy.deepcopy(r)
        a=identify(r);self.assertEqual(r,original)
        i=a['constraints'][0]['source_H'][0]
        r['atom_target_mask'][i]=0;r['atom_targets'][i]=None
        r['spectrum']=[x for x in r['spectrum'] if x['atom_index']!=i]
        a=identify(r);self.assertEqual(a['constraints'][0]['observed_size'],2)
        self.assertFalse(a['missing_labels_imputed']);self.assertIsNone(r['atom_targets'][i])
        r['data_nature']='theoretical'
        with self.assertRaises(ValueError):identify(r)


if __name__=='__main__':unittest.main()

"""功能概述：用合成逐H谱验证原始/校准/物理平均三口径的母节点映射与质量守恒。
缺失H、错原子顺序和结构错配必须拒绝；测试不读取任何研究标签或外部服务。
"""
import copy,unittest
import numpy as np
from rdkit import Chem
from .h_count_targets import target_variants
from .h_count_train import apply_variants
from .data import identity
from .h_labels import make_key


class TargetTests(unittest.TestCase):
    def fixture(self):
        mol=Chem.AddHs(Chem.MolFromSmiles('CCO'));tokens=[a.GetAtomicNum() for a in mol.GetAtoms()];s=dict(source_smiles='CCO',source_to_canonical=list(range(len(tokens))),tokens=tokens);p=dict(nuclei={'1H':dict(status='predicted',count=3,groups=[[3,4,5],[6,7],[8]])});r=dict(smiles='CCO',value=dict(atom_index=[4,5,6,7,8,9],x=[1.,2.,3.,4.,6.,8.]));fit=dict(affine_scale=2.,affine_offset_ppm=1.,environment_residuals={});return s,p,r,fit
    def test_variants(self):
        s,p,r,f=self.fixture();original=copy.deepcopy((s,p,r,f));canonical,v=target_variants(s,p,r,f);self.assertEqual(canonical,'CCO');self.assertEqual((s,p,r,f),original)
        self.assertEqual(v['raw']['annotation']['groups'],{0:[1.,2.,3.],1:[4.,6.],2:[8.]})
        self.assertEqual(v['physical']['annotation']['groups'],{0:[5.,5.,5.],1:[11.,11.],2:[17.]})
        self.assertEqual(sum(map(len,v['physical']['annotation']['groups'].values())),6)
        self.assertAlmostEqual(sum(sum(xs) for xs in v['physical']['annotation']['groups'].values()),sum(sum(xs) for xs in v['calibrated']['annotation']['groups'].values()))
        self.assertTrue(np.array_equal(np.unique(v['physical']['q']),[5.,11.,17.]))
    def test_sidecar_isolation(self):
        s,p,r,f=self.fixture();smi,v=target_variants(s,p,r,f);smi='CC';_,group,split=identity(smi);row=dict(domain='theory',source='recalculated',record_id='example',smiles=smi,group=group,split=split,q=v['raw']['q']);labels={};side=dict(variants={make_key(row):v});rows,n=apply_variants([row],labels,side,'physical');self.assertEqual(n,1);self.assertTrue(np.array_equal(row['q'],v['raw']['q']));self.assertTrue(np.array_equal(rows[0]['q'],v['physical']['q']));self.assertEqual(labels[make_key(row)]['calibration'],'physical')
        bad=dict(row,split='test')
        with self.assertRaises(ValueError):apply_variants([bad],{},side,'raw')
    def test_rejections(self):
        for kind in ['missing','order','structure','group']:
            s,p,r,f=self.fixture()
            if kind=='missing':r['value']['atom_index'].pop();r['value']['x'].pop()
            elif kind=='order':s['source_to_canonical'][0],s['source_to_canonical'][3]=s['source_to_canonical'][3],s['source_to_canonical'][0]
            elif kind=='structure':r['smiles']='CCN'
            else:p['nuclei']['1H']['groups']=[[3,4],[6,7],[8]]
            with self.assertRaises(ValueError):target_variants(s,p,r,f)


if __name__=='__main__':unittest.main()

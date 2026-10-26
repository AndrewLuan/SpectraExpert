"""检查双核组合的独立数学例子：单核歧义可互补，并列不靠索引获胜。
使用三个小分子和人工距离，不读取真实实验或测试标签，不启动训练。
"""
import unittest
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from .joint import rank_entry


class TestJoint(unittest.TestCase):
    def test_complementary_nuclei(self):
        names=['CC','CCO','CCC'];gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        fps={s:gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in names}
        h=torch.tensor([.2,.1,.9]);c=torch.tensor([.2,.9,.1])
        self.assertEqual(rank_entry(h,0,names,fps)['rank'],2)
        self.assertEqual(rank_entry(c,0,names,fps)['rank'],2)
        result=rank_entry(.5*(h+c),0,names,fps)
        self.assertEqual(result['rank'],1);self.assertEqual(result['top1_tanimoto'],1.)

    def test_ties_are_pessimistic(self):
        names=['CC','CCO','CCC'];gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        fps={s:gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in names}
        self.assertEqual(rank_entry(torch.tensor([.2,.2,.9]),0,names,fps)['rank'],2)


if __name__=='__main__':unittest.main()

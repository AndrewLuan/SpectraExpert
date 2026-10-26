"""训练来源先验的同步算术检查，不读取真实验证或测试数据。
核对只用experimental训练计数、加权MRR以及来源缺失拒绝；不改变任何选模文件。
依赖h_sourceprior，由unittest执行，无网络或持久化副作用。
"""
import unittest
from .h_sourceprior import training_prior,prior_score


class TestSourcePrior(unittest.TestCase):
    def test_theory_counts_cannot_change_experimental_prior(self):
        counts,p=training_prior(dict(training_rows={'experimental|A':9,'experimental|B':1,'theory|X':10000}));self.assertEqual(counts,{'A':9,'B':1});self.assertEqual(p,{'A':.9,'B':.1});m=dict(by_source={'A':dict(queries=64,MRR=.2),'B':dict(queries=64,MRR=.9)});self.assertAlmostEqual(prior_score(m,p),.27)
        with self.assertRaisesRegex(ValueError,'coverage'):prior_score(dict(by_source={'A':dict(queries=64,MRR=.2)}),p)


if __name__=='__main__':main()

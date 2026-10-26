"""功能概述：验证条件联合研究的可微观测语义和训练边界。
用合成谱检查整谱边缘化、缺峰分支梯度、物理谱不变、报告头的未知条件和混合类别回退；
用小分子图检查显式频窗/审核隔离不启用报告似然，无积分和不完整配对不进入跨溶剂监督。
测试仅运行CPU，不读取真实实验数据、服务器、候选库或外部服务。
"""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from .data import graph, physical_features
from .h_conditions import encode_conditions, local_environment, CONDITION_DIM
from .h_formyl_reporting import counterfactual_quantiles
from .h_joint_conditions import ReportingHead, JointSpectra, spectrum_nll, state_nll, build_pairs
from .h_labels import make_key
from .h_joint_evaluate import rank
from .h_condinfer import condition_key
from .h_metadata import RECOVERED_MODE


class JointConditionsTests(unittest.TestCase):
    def test_state_likelihood_and_shift_gradients(self):
        full = torch.tensor([2., 4.], requires_grad=True)
        missing = torch.tensor([1., 5.], requires_grad=True)
        p = torch.tensor([.8, .8], requires_grad=True)
        k = torch.tensor([3., 2.])
        value = state_nll(full, missing, p, k)
        expected = -torch.log(.8*torch.exp(-k*full) + .2*torch.exp(-k*missing))/k
        torch.testing.assert_close(value, expected)
        value.sum().backward()
        for grad in (full.grad, missing.grad, p.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.)
        torch.testing.assert_close(state_nll(full, missing, torch.ones(2), k), full, rtol=0, atol=0)

    def test_removing_observed_state_keeps_physical_peak_and_gradients(self):
        physical = torch.tensor([[.2, .2, .3, .8]], requires_grad=True)
        original = physical.detach().clone()
        altered, active = counterfactual_quantiles(physical, torch.tensor([[0,0,0,4]]), ['formamide'])
        loss = spectrum_nll(torch.tensor([[.21, .29]]), altered).sum()
        loss.backward()
        self.assertTrue(bool(active[0]))
        torch.testing.assert_close(physical.detach(), original, rtol=0, atol=0)
        self.assertEqual(float(physical.grad[0,3]), 0.)
        self.assertGreater(float(physical.grad[0,:3].abs().sum()), 0.)

    def test_head_uses_condition_and_process_only(self):
        torch.manual_seed(317)
        head = ReportingHead()
        cond = torch.zeros(3, CONDITION_DIM)
        p = head(['formamide', 'aldehyde', 'other'], cond)
        torch.testing.assert_close(p, torch.tensor([.9, .9, 1.]))
        p[:2].sum().backward()
        self.assertTrue(torch.isfinite(head.layers[-1].weight.grad).all())

    @staticmethod
    def fixtures():
        smiles = 'CCN(C)C=O'
        g = graph(smiles)
        g.local_environment = torch.tensor(local_environment(smiles))
        rows = [dict(domain='experimental', split='train', source='synthetic', record_id=str(i),
                     smiles=smiles, group='same_structure', physics=physical_features(smiles),
                     cond=np.zeros(4, np.float32), q=np.tile([1., 2., 3., 8.], 32).astype(np.float32),
                     observation=dict(kind='unknown', windows=[])) for i in range(3)]
        meta = {make_key(r): encode_conditions(dict(solvent=s))
                for r, s in zip(rows, ['C6D6', 'CDCl3', ''])}
        return rows, {smiles:g}, meta

    def test_pair_filter_requires_observation_support_and_known_conditions(self):
        rows, graphs, meta = self.fixtures()
        labels = {make_key(r):dict(groups={}, integral_consistent=True) for r in rows}
        pairs, audit = build_pairs(rows, labels, graphs, meta)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]['complete_distribution'])
        labels[make_key(rows[1])]['integral_consistent'] = False
        self.assertEqual(build_pairs(rows, labels, graphs, meta)[0], [])
        rows[1]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'require_experimental_train'):
            build_pairs(rows, labels, graphs, meta)

    def test_explicit_window_and_review_holdout_disable_reporting(self):
        rows, graphs, _ = self.fixtures()
        for r in rows:r['source'] = 'HMDB'
        rows[1]['observation']['windows'] = [[0., 4.]]
        rows[2]['review_spectral_weight'] = 0.
        labels = {make_key(r):dict(groups={}) for r in rows}
        dataset = JointSpectra(rows, graphs, '1H', {}, annotations=labels, grouped=True)
        self.assertEqual([bool(dataset[i].reporting_eligible[0]) for i in range(3)], [True, False, False])
        np.testing.assert_array_equal(rows[0]['q'], np.tile([1.,2.,3.,8.], 32))

    def test_ranking_keeps_candidate_order_and_identity_at_zero_width(self):
        rows, _, meta = self.fixtures()
        rows = rows[:1]
        rows[0]['measure'] = 'integral_weighted'
        key = condition_key(rows[0], meta, RECOVERED_MODE)
        pred = torch.stack([torch.linspace(.1,.8,128),torch.linspace(.9,1.5,128)])
        bank = dict(single_indices=torch.tensor([0,1]), smiles=[rows[0]['smiles'],'CCO'],
                    physics=torch.ones(2,8), predictions={key:pred}, fixed_predictions={},
                    candidate_quantile_classes={key:torch.zeros(2,128,dtype=torch.uint8)},
                    condition_key_mode=RECOVERED_MODE,checkpoint_sha256='synthetic')
        # 纯likelihood配置是原query_score支持的冻结路由，积分谱不受宽度干预。
        inputs = (dict(HMDB=None,water_HMDB=None,dropout=None),{},dict(kind='likelihood'),{}, {})
        with tempfile.TemporaryDirectory() as folder:
            result = rank(bank,rows,meta,{},SimpleNamespace(reporting_enabled=False),[.2]*9,
                          inputs,torch.device('cpu'),Path(folder),lambda *a,**k:None)
            self.assertEqual(result['0.0']['query_sha256'],result['1.0']['query_sha256'])
            self.assertEqual(result['0.0']['candidate_order_sha256'],result['1.0']['candidate_order_sha256'])
            self.assertEqual(result['0.0']['metrics'],result['1.0']['metrics'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

"""功能概述：验证整体 NMR 研究新增评分的概率和冻结边界。

测试污染核零强度精确回退、密度与 CDF 一致、离群观测的稳健增益及有限性；
同时核对预声明消融包含基线、计数与积分独立干预且名称唯一。
使用 CPU 小谱，不访问实验原始数据、网络或 GPU；真实候选基线一致性另由 smoke 验证。
"""
import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from .h_global_scoring import contaminated, configs, latent_feature_inputs, frozen_choices
from .h_latent_score import features as latent_features, prototype_keys
from .h_conditions import CONDITION_DIM
from .likelihood import density, log_score
from .prepare import sha


class GlobalScoringTests(unittest.TestCase):
    def test_zero_contamination_preserves_probability(self):
        kernel=density(torch.tensor([[.1,.1,.7,.7]]),'1H',.2)
        with np.errstate(divide='ignore'):
            result=contaminated(kernel,0.)
        self.assertTrue(torch.equal(kernel['logp'],result['logp']))
        self.assertTrue(torch.equal(kernel['cdf'],result['cdf']))

    def test_density_and_cdf_share_same_mixture(self):
        kernel=density(torch.tensor([[.1,.1,.7,.7]]),'1H',.2)
        result=contaminated(kernel,.03)
        expected=.97*kernel['logp'].exp()+.03/(kernel['step']*kernel['logp'].shape[1])
        self.assertTrue(torch.allclose(result['logp'].exp(),expected,atol=1e-7))
        self.assertAlmostEqual(float(result['cdf'][0,-1]),1.,places=4)
        self.assertTrue(torch.all(torch.diff(result['cdf'])>=-1e-8))

    def test_contamination_bounds_outlier_penalty(self):
        kernel=density(torch.tensor([[.1,.1,.2,.2]]),'1H',.2)
        query=dict(q=np.array([1.,2.,13.]),observation=dict(kind='unknown',windows=[]))
        old=log_score(kernel,query,'1H')
        new=log_score(contaminated(kernel,.03),query,'1H')
        self.assertTrue(torch.isfinite(new).all())
        self.assertLess(float(new),float(old))

    def test_registered_interventions_are_independent(self):
        choices=configs(); names=[c['name'] for c in choices]
        self.assertEqual(len(names),len(set(names)))
        self.assertEqual(choices[0],dict(name='baseline',kind='baseline'))
        self.assertTrue(any(c['kind']=='atom' and c['beta']>.0 for c in choices))
        self.assertTrue(any(c['kind']=='integral_count' and c['beta']==0 for c in choices))
        self.assertTrue(any(c['kind']=='outlier' for c in choices))

    def test_given_only_frozen_subset_handles_unknown_solvent(self):
        kernel=density(torch.tensor([[.1,.1,.2,.2],[.1,.2,.3,.3]]),'1H',.2)
        query=dict(q=np.repeat(1.,128),cond=np.zeros(4),source='nmrshiftdb2_2024',
            observed_xs=[1.],observed_weights=[1.],_condition_known=False,observation=dict(kind='unknown',windows=[]))
        adapted,kernels=latent_feature_inputs(query,kernel,{},False)
        result=latent_features(adapted,kernels,torch.tensor([4.,6.]),{},dict(solvent_prior=[.3,.3,.4],dropout={}))
        self.assertTrue(torch.isfinite(result['nll']['given',.1]).all())
        self.assertFalse(query['_condition_known'])

    def test_new_bank_does_not_require_legacy_condition_mode(self):
        key=tuple([0.]*(4+CONDITION_DIM))
        bank=dict(predictions={key:torch.zeros(1,128)},condition_key_mode='recovered_solvent')
        keys=prototype_keys(bank)
        self.assertEqual(set(keys),{'Water','CDCl3','DMSO'})
        self.assertTrue(all(len(k)==len(key) for k in keys.values()))

    def test_heldout_rejects_changed_scoring_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); bank=root/'bank.pt'; bank.write_text('original')
            selection=root/'frozen.json'
            selection.write_text(json.dumps(dict(selection_scope='train_val_only',frozen_utc='2026-10-20T00:00:00Z',
                evaluation_inputs_sha256={str(bank.resolve()):sha(bank)},scoring_configs=[configs()[0]])))
            self.assertEqual(frozen_choices(selection,[bank]),[configs()[0]])
            bank.write_text('replacement')
            with self.assertRaisesRegex(ValueError,'global_frozen_input_changed'):
                frozen_choices(selection,[bank])


if __name__=='__main__':
    unittest.main()

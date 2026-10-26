"""功能概述：核验双通道候选的完整排序与新轮封存门禁。

测试构造非连续短名单，检查池外名次不会与前置短名单碰撞、训练补正例不可参与
评估、ties稳定且保守；临时文件仅用于检验冻结源改变和旧schema拒绝，无远程副作用。
"""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from .h_research_contract import check_locked, input_fingerprints, pool_background_rank, rank_shortlist


class ResearchContractTests(unittest.TestCase):
    def test_noncontiguous_pool_has_consistent_outside_ranks(self):
        coarse = torch.arange(8, dtype=torch.float32)
        pool = torch.tensor([0, 6, 7])
        self.assertEqual([pool_background_rank(coarse, pool, i) for i in range(1, 6)], [4, 5, 6, 7, 8])
        self.assertIsNone(pool_background_rank(coarse, pool, 6))

    def test_training_positive_does_not_bypass_recall(self):
        row = dict(candidate_indices=torch.tensor([0, 6, 7, 3]), evaluation_pool_size=3,
                   target_in_coarse_pool=False, target_position=3, outside_pool_rank=6)
        rank, top = rank_shortlist(row, torch.tensor([2., 1., 0., -999.]))
        self.assertEqual(rank, 6)
        self.assertEqual(top, [7, 6, 0])

    def test_ties_are_stable_and_conservative(self):
        row = dict(candidate_indices=torch.tensor([7, 2, 5]), target_in_coarse_pool=True, target_position=0)
        self.assertEqual(rank_shortlist(row, torch.ones(3)), (3, [2, 5, 7]))

    def test_freeze_requires_new_schema_and_exact_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder/'source'; source.write_text('original')
            pins = input_fingerprints([source])
            config = {'R': True}
            frozen = folder/'freeze.json'
            value = dict(schema='nmr_research_20261022', selection_scope='train_val_only',
                         frozen_utc='2026-10-23T02:44:36Z', allowed_extract_configs=[config],
                         evaluation_inputs_sha256=pins, verify_eligibility='unavailable')
            frozen.write_text(json.dumps(value))
            check_locked(frozen, pins, config, 'test')
            with self.assertRaisesRegex(ValueError, 'not_eligible'):
                check_locked(frozen, pins, config, 'verify')
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'input_changed'):
                check_locked(frozen, input_fingerprints([source]), config, 'test')
            value['schema'] = 'old'; frozen.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'invalid_frozen'):
                check_locked(frozen, pins, config, 'test')


if __name__ == '__main__':
    unittest.main()

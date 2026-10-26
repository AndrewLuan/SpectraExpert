"""功能概述：用纯 CPU 合成缓存核验配对重排残差混合，不读取真实数据。

fixtures 构造同预测/粗排/缓存但 scale 与权重不同的 no_aug/aug checkpoint。
端点测试逐条比较原 evaluate 单模型；其余检查 alpha=0 唯一计数、原单位凸组合、
窗口/删峰名称不能选择权重、名单外召回保持以及模型/候选协议拒绝。冻结测试用
临时文件核对精确 pair/gamma、权重/源码/缓存来源 SHA；不运行 GPU、训练或网络。
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .h_global_features import FEATURE_NAMES
from .h_global_ranker import ResidualRanker, evaluate, normalized_rows
from .h_global_ranker_blend import (GAMMAS, combined_residual, evaluate_blend,
                                     frozen_gammas, locked_inputs, main, validate_pair)
from .prepare import sha


def fixtures():
    smiles = ['CCO', 'CCCC', 'CCN', 'CCCO']
    order = hashlib.sha256('\n'.join(smiles).encode()).hexdigest()
    protocol = dict(coarse_config=dict(name='atom_count_1.0'), pool_size=3,
                    feature_code_sha256='feature-code', feature_names=FEATURE_NAMES,
                    candidate_order_sha256=order)
    checkpoints = []
    for augmented, scale0, scale1, alpha, coefficient in (
            (False, .05, .25, .5, .7), (True, .2, .5, 1., -.9)):
        model = ResidualRanker(list(range(len(FEATURE_NAMES))), typed=True)
        with torch.no_grad():
            model.network.weight[:, 1] = coefficient
            model.network.weight[:, 2] = -.3
        scale = torch.ones(len(FEATURE_NAMES))
        scale[0], scale[1] = scale0, scale1
        checkpoints.append(dict(protocol, state_dict=model.state_dict(), scale=scale,
            columns=model.columns, architecture='linear', typed=True, alpha=alpha,
            args=dict(augment=augmented), train_sha256='common-train',
            prediction_checkpoint_sha256='same-prediction'))
    rows = []
    for i, variant, target in ((0, 'observed', 0), (1, 'observed', -1),
                               (0, 'frequency_crop', 1), (1, 'peak_dropout', 2)):
        x = torch.zeros(3, len(FEATURE_NAMES))
        x[:, 0] = torch.tensor([.10, .13, .16])
        x[:, 1] = torch.tensor([-.5, .4, 1.2])
        x[:, 2] = torch.tensor([.5, .1, -.4])
        rows.append(dict(features=x, candidate_indices=torch.tensor([0, 1, 2]),
            coarse_scores=x[:, 0].clone(), target_position=target,
            target_in_coarse_pool=target >= 0, coarse_rank=target + 1 if target >= 0 else 4,
            baseline_rank=target + 1 if target >= 0 else 4, coarse_top10=[0, 1, 2],
            rerank_eligible=True, source='synthetic', record_id=str(i), smiles=smiles[i],
            group=str(i), variant=variant, query_order=i, kind=1,
            query_observation=dict(kind='unknown', windows=[])))
    manifest = dict(protocol, cohort='original_val', queries=len(rows), candidate_count=len(smiles),
                    checkpoint_sha256='same-prediction', source_inputs_sha256={},
                    appended_training_positives=0)
    return checkpoints[0], checkpoints[1], dict(rows=rows, smiles=smiles, manifest=manifest)


class RankerBlendTests(unittest.TestCase):
    def test_endpoints_exactly_equal_independent_single_models(self):
        left, right, data = fixtures()
        blended, _ = evaluate_blend(left, right, data)
        for gamma, saved in ((0., left), (1., right)):
            model = ResidualRanker(saved['columns'], saved['architecture'], saved['typed'])
            model.load_state_dict(saved['state_dict'])
            single, _ = evaluate(model, normalized_rows(data['rows'], saved['scale']),
                                 torch.device('cpu'), [saved['alpha']])
            self.assertEqual(blended[str(gamma)], single[str(saved['alpha'])])

    def test_zero_saved_alphas_never_duplicate_queries(self):
        left, right, data = fixtures()
        left['alpha'] = right['alpha'] = 0.
        outputs, reports = evaluate_blend(left, right, data, [0., 0., .5, 1.])
        self.assertEqual(list(outputs), ['0.0', '0.5', '1.0'])
        for key, rows in outputs.items():
            self.assertEqual(len(rows), len(data['rows']))
            self.assertEqual(reports[key]['overall']['queries'], 2)
            self.assertEqual([r['rank'] for r in rows], [r['coarse_rank'] for r in data['rows']])

    def test_residuals_use_each_branch_scale_in_raw_score_units(self):
        left, right, _ = fixtures()
        ld, rd = torch.tensor([1., -2.]), torch.tensor([-3., 4.])
        expected = .75 * left['alpha'] * ld * left['scale'][0] + .25 * right['alpha'] * rd * right['scale'][0]
        actual = combined_residual(ld, rd, left, right, .25)
        self.assertTrue(torch.equal(actual, expected))

    def test_prediction_coarse_and_candidate_order_must_match(self):
        for field, value in (('prediction_checkpoint_sha256', 'other-prediction'),
                             ('coarse_config', dict(name='different-coarse')),
                             ('candidate_order_sha256', 'different-order')):
            with self.subTest(field=field):
                left, right, data = fixtures()
                right[field] = value
                with self.assertRaises(ValueError):
                    validate_pair(left, right, data)

    def test_training_cache_and_augmentation_pair_must_match(self):
        left, right, data = fixtures()
        right['train_sha256'] = 'other-cache'
        with self.assertRaisesRegex(ValueError, 'training_cache_mismatch'):
            validate_pair(left, right, data)
        left, right, data = fixtures()
        right['args']['augment'] = False
        with self.assertRaisesRegex(ValueError, 'no_aug_and_aug'):
            validate_pair(left, right, data)

    def test_windows_dropout_names_and_identity_do_not_route_weights(self):
        left, right, data = fixtures()
        baseline, _ = evaluate_blend(left, right, data, [.5])
        altered = copy.deepcopy(data)
        for row in altered['rows']:
            row['source'] = 'arbitrary'
            row['record_id'] = 'changed'
            row['smiles'] = 'C'
            row['variant'] = 'observed' if row['variant'] != 'observed' else 'peak_dropout'
            row['query_observation'] = dict(kind='partial_window', windows=[[0., 4.]], partial_peaks=True)
        changed, _ = evaluate_blend(left, right, altered, [.5])
        self.assertEqual([(r['rank'], r['top10_indices']) for r in baseline['0.5']],
                         [(r['rank'], r['top10_indices']) for r in changed['0.5']])

    def test_outside_shortlist_rank_and_tie_fallback_are_preserved(self):
        left, right, data = fixtures()
        data['rows'][2]['rerank_eligible'] = False
        outputs, _ = evaluate_blend(left, right, data)
        for rows in outputs.values():
            self.assertEqual(rows[1]['rank'], 4)
            self.assertEqual(rows[2]['rank'], data['rows'][2]['coarse_rank'])
            self.assertEqual(rows[2]['top10_indices'], data['rows'][2]['coarse_top10'])

    def test_evaluation_positive_injection_is_rejected(self):
        left, right, data = fixtures()
        data['rows'][1]['target_position'] = 2
        with self.assertRaisesRegex(ValueError, 'positive_injection'):
            evaluate_blend(left, right, data)

    def test_frozen_pair_and_weight_are_exact_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / 'no.pt', Path(directory) / 'aug.pt'
            configs = [dict(no_aug_checkpoint=str(left), aug_checkpoint=str(right), gamma=.25)]
            frozen = dict(blend_configs=configs)
            self.assertEqual(frozen_gammas(frozen, left, right), [.25])
            with self.assertRaisesRegex(ValueError, 'pair_not_frozen'):
                frozen_gammas(frozen, right, left)
            frozen['blend_configs'][0]['gamma'] = .3
            with self.assertRaisesRegex(ValueError, 'unregistered'):
                frozen_gammas(frozen, left, right)
            frozen['blend_configs'][0]['gamma'] = .25
            frozen['blend_configs'] *= 2
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                frozen_gammas(frozen, left, right)

    def test_locked_inputs_pin_both_weights_sources_and_cache_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            left, right, origin = root / 'no.pt', root / 'aug.pt', root / 'origin'
            for p in (left, right, origin):
                p.write_text(p.name)
            here = Path(__file__).resolve().with_name('h_global_ranker_blend.py')
            sources = [here, here.with_name('h_global_ranker.py'), here.with_name('h_global_features.py')]
            frozen = dict(selection_scope='train_val_only',
                ranker_checkpoints_sha256={str(p): sha(p) for p in (left, right)},
                evaluation_inputs_sha256={str(p): sha(p) for p in sources + [origin]})
            manifest = dict(source_inputs_sha256={str(origin): sha(origin)})
            checks = locked_inputs(frozen, left, right, manifest)
            self.assertIn(str(left), checks)
            self.assertIn(str(right), checks)
            origin.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source_or_cache_input'):
                locked_inputs(frozen, left, right, manifest)

    def test_train_val_cli_outputs_unique_normal_and_stress_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            left, right, data = fixtures()
            lp, rp, cp = root / 'no.pt', root / 'aug.pt', root / 'cache.pt'
            torch.save(left, lp)
            torch.save(right, rp)
            torch.save(data, cp)
            out = root / 'output'
            argv = ['blend', '--stage', 'train_val', '--no-aug-checkpoint', str(lp),
                    '--aug-checkpoint', str(rp), '--cache', str(cp), '--out', str(out)]
            with patch('sys.argv', argv):
                main()
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['gammas'], list(GAMMAS))
            self.assertEqual(summary['device'], 'cpu')
            self.assertEqual((out / 'EXIT_CODE').read_text(), '0\n')
            for gamma in GAMMAS:
                folder = out / ('gamma_' + str(gamma))
                self.assertEqual(len((folder / 'predictions.jsonl').read_text().splitlines()), 2)
                self.assertEqual(len((folder / 'missing_predictions.jsonl').read_text().splitlines()), 2)

    def test_train_val_cli_rejects_synthetic_sealed_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            left, right, data = fixtures()
            data['manifest']['cohort'] = 'verify'
            lp, rp, cp = root / 'no.pt', root / 'aug.pt', root / 'cache.pt'
            torch.save(left, lp)
            torch.save(right, rp)
            torch.save(data, cp)
            out = root / 'rejected'
            argv = ['blend', '--stage', 'train_val', '--no-aug-checkpoint', str(lp),
                    '--aug-checkpoint', str(rp), '--cache', str(cp), '--out', str(out)]
            with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'access_violation'):
                main()
            self.assertEqual((out / 'EXIT_CODE').read_text(), '1\n')


if __name__ == '__main__':
    unittest.main()

"""功能概述：用 CPU 合成缓存验证公开观测标签路由，不读取真实 verify/test。

fixtures 沿用 blend 的合成 pair/全库缓存并添加显式窗口/缺峰标签；materialize
生成本地临时 blend 文件，验证路由只复用预测而不运行模型。测试覆盖两个字段的
读取边界、缺字段默认、normal/缺测端点、独立 scale 的既有 blend、身份/完整库/
补正例拒绝、冻结 pair/policy/gamma、输入始末 SHA，以及 train_val/locked 门禁。
所有临时产物随 fixture 清理，不调用网络、API、GPU或真实研究缓存。
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .h_global_metadata_route import (POLICY, explicit_missing, frozen_routes, main,
    route_predictions, snapshot_inputs, validate_origin, validate_predictions, verify_unchanged)
from .h_global_ranker import ResidualRanker
from .h_global_ranker_blend import GAMMAS, evaluate_blend, main as blend_main, validate_pair
from .test_h_global_ranker_blend import fixtures as blend_fixtures
from .prepare import sha


def fixtures():
    left, right, data = blend_fixtures()
    # 实际 shortlist>=10；合成全库 11 个、shortlist 10 个，另保留名单外真值。
    data['smiles'] += ['C' * i for i in range(5, 12)]
    order = hashlib.sha256('\n'.join(data['smiles']).encode()).hexdigest()
    for saved in (left, right, data['manifest']):
        saved.update(candidate_order_sha256=order, pool_size=10)
    data['manifest']['candidate_count'] = len(data['smiles'])
    for row in data['rows']:
        row['features'] = row['features'].repeat(4, 1)[:10].clone()
        row['features'][:, 0] = torch.linspace(.1, .37, 10)
        row['coarse_scores'] = row['features'][:, 0].clone()
        row['candidate_indices'] = torch.arange(10)
        row['coarse_top10'] = list(range(10))
        if not row['target_in_coarse_pool']:
            row['coarse_rank'] = row['baseline_rank'] = 11
    data['rows'][2]['query_observation'] = dict(windows=[[0., 4.]])
    data['rows'][3]['query_observation'] = dict(partial_peaks=True)
    predictions, _ = evaluate_blend(left, right, data)
    for rows in predictions.values():
        for row in rows:
            row['top10'] = [data['smiles'][i] for i in row.pop('top10_indices')]
    return left, right, data, predictions


def materialize(root, cohort='original_val'):
    left, right, data, _ = fixtures()
    data['manifest']['cohort'] = cohort
    lp, rp, cp = root / 'no.pt', root / 'aug.pt', root / 'cache.pt'
    torch.save(left, lp)
    torch.save(right, rp)
    torch.save(data, cp)
    blend = root / 'blend'
    frozen_path = None
    stage = 'train_val' if cohort in ('original_val', 'tune') else 'locked'
    argv = ['blend', '--stage', stage, '--no-aug-checkpoint', str(lp),
            '--aug-checkpoint', str(rp), '--cache', str(cp), '--out', str(blend)]
    if stage == 'locked':
        here = Path(__file__).with_name('h_global_metadata_route.py').resolve()
        sources = [here, here.with_name('h_global_ranker_blend.py'), here.with_name('h_global_ranker.py'),
                   here.with_name('h_global_features.py'), here.with_name('prepare.py')]
        configs = [dict(no_aug_checkpoint=str(lp), aug_checkpoint=str(rp), gamma=g) for g in (0., .5)]
        frozen = dict(selection_scope='train_val_only', blend_configs=configs,
            metadata_route_configs=[dict(configs[1], policy=POLICY)],
            ranker_checkpoints_sha256={str(p): sha(p) for p in (lp, rp)},
            evaluation_inputs_sha256={str(p): sha(p) for p in sources})
        frozen_path = root / 'frozen.json'
        frozen_path.write_text(json.dumps(frozen))
        argv += ['--frozen-selection', str(frozen_path)]
    with patch('sys.argv', argv):
        blend_main()
    return lp, rp, cp, blend, frozen_path


def route_argv(paths, out, stage='train_val'):
    lp, rp, cp, blend, frozen = paths
    argv = ['route', '--stage', stage, '--no-aug-checkpoint', str(lp),
            '--aug-checkpoint', str(rp), '--cache', str(cp), '--blend-dir', str(blend), '--out', str(out)]
    if frozen:
        argv += ['--frozen-selection', str(frozen)]
    return argv


class MetadataRouteTests(unittest.TestCase):
    def test_predicate_reads_only_public_fields_and_defaults_false(self):
        class PublicOnly(dict):
            def get(self, key, default=None):
                if key not in ('windows', 'partial_peaks'):
                    raise AssertionError('forbidden predicate field: ' + key)
                return super().get(key, default)
        self.assertFalse(explicit_missing(PublicOnly()))
        self.assertFalse(explicit_missing(PublicOnly(windows=[], partial_peaks=False)))
        self.assertTrue(explicit_missing(PublicOnly(windows=[[1., 3.]])))
        self.assertTrue(explicit_missing(PublicOnly(partial_peaks=True)))
        for invalid in (dict(windows='window'), dict(windows=[[3., 1.]]),
                        dict(windows=[[float('nan'), 4.]]), dict(partial_peaks='False')):
            with self.assertRaises(ValueError):
                explicit_missing(invalid)

    def test_normal_uses_no_aug_and_explicit_missing_uses_registered_blend(self):
        _, _, data, branches = fixtures()
        outputs, reports, counts = route_predictions(data, branches, GAMMAS)
        self.assertEqual(counts, dict(undeclared_missing=2, explicit_missing=2))
        for gamma, rows in outputs.items():
            for i, row in enumerate(rows):
                expected = branches['0.0' if i < 2 else gamma][i]
                self.assertEqual((row['rank'], row['top10']), (expected['rank'], expected['top10']))
            self.assertEqual(reports[gamma]['overall'], reports['0.0']['overall'])
        self.assertEqual(len(outputs['0.0']), len(data['rows']))

    def test_endpoints_and_alpha_zero_preserve_exact_rows_without_duplicates(self):
        left, right, data, branches = fixtures()
        for row in data['rows']:
            row['query_observation'] = dict(partial_peaks=True)
        outputs, _, _ = route_predictions(data, branches, [0., 0., 1.])
        self.assertEqual(set(outputs), {'0.0', '1.0'})
        for key in outputs:
            for routed, original in zip(outputs[key], branches[key]):
                self.assertEqual({k: v for k, v in routed.items() if k not in ('route_gamma', 'explicit_missing')}, original)
        left['alpha'] = right['alpha'] = 0.
        raw, _ = evaluate_blend(left, right, data)
        self.assertTrue(all(len(rows) == len(data['rows']) for rows in raw.values()))

    def test_variant_source_id_smiles_do_not_choose_branch(self):
        _, _, data, branches = fixtures()
        before, _, _ = route_predictions(data, branches, [.5])
        for row in data['rows']:
            row.update(source='changed', record_id='changed', smiles='C', group='changed',
                       variant='renamed_' + row['variant'])
        for rows in branches.values():
            for row in rows:
                row.update(source='changed', record_id='changed', smiles='C', group='changed',
                           variant='renamed_' + row['variant'])
        after, _, _ = route_predictions(data, branches, [.5])
        self.assertEqual([(r['rank'], r['top10'], r['route_gamma']) for r in before['0.5']],
                         [(r['rank'], r['top10'], r['route_gamma']) for r in after['0.5']])

    def test_metadata_can_route_observed_but_variant_alone_cannot(self):
        _, _, data, branches = fixtures()
        data['rows'][0]['query_observation'] = dict(windows=[[0., 4.]])
        data['rows'][2]['query_observation'] = {}
        out, _, _ = route_predictions(data, branches, [1.])
        self.assertEqual(out['1.0'][0]['route_gamma'], 1.)
        self.assertEqual(out['1.0'][2]['route_gamma'], 0.)

    def test_missing_cache_metadata_fails_instead_of_inventing_from_variant(self):
        _, _, data, branches = fixtures()
        del data['rows'][2]['query_observation']
        with self.assertRaisesRegex(ValueError, 'cached_observation_missing'):
            route_predictions(data, branches, [.5])

    def test_duplicate_missing_and_changed_identities_are_rejected(self):
        for mode in ('duplicate', 'missing', 'changed'):
            with self.subTest(mode=mode):
                _, _, data, branches = fixtures()
                rows = branches['0.0']
                if mode == 'duplicate':
                    rows.append(copy.deepcopy(rows[0]))
                elif mode == 'missing':
                    rows.pop()
                else:
                    rows[0]['record_id'] = 'other'
                with self.assertRaises(ValueError):
                    validate_predictions(rows, data)

    def test_full_library_outside_pool_rank_and_top10_are_checked(self):
        _, _, data, branches = fixtures()
        self.assertEqual(branches['0.0'][1]['rank'], 11)
        for mutation in ('outside_rank', 'too_large_rank', 'new_candidate', 'short_top10'):
            rows = copy.deepcopy(branches['0.0'])
            if mutation == 'outside_rank':
                rows[1]['rank'] = 1
            elif mutation == 'too_large_rank':
                rows[0]['rank'] = 12
            elif mutation == 'new_candidate':
                rows[0]['top10'] = ['unknown candidate']
            else:
                rows[0]['top10'].pop()
            with self.assertRaises(ValueError):
                validate_predictions(rows, data)

    def test_positive_injection_and_mismatched_pair_are_rejected(self):
        left, right, data, _ = fixtures()
        data['rows'][1]['target_position'] = 2
        with self.assertRaisesRegex(ValueError, 'positive_injection'):
            validate_pair(left, right, data)
        for field in ('coarse_config', 'candidate_order_sha256', 'prediction_checkpoint_sha256'):
            left, right, data, _ = fixtures()
            right[field] = 'changed'
            with self.assertRaises(ValueError):
                validate_pair(left, right, data)

    def test_frozen_pair_policy_gamma_are_exact_unique_and_no_runtime_search(self):
        with tempfile.TemporaryDirectory() as directory:
            lp, rp = Path(directory) / 'no.pt', Path(directory) / 'aug.pt'
            config = dict(no_aug_checkpoint=str(lp), aug_checkpoint=str(rp), policy=POLICY, gamma=.5)
            frozen = dict(metadata_route_configs=[config])
            self.assertEqual(frozen_routes(frozen, lp, rp), [.5])
            with self.assertRaisesRegex(ValueError, 'pair_not_frozen'):
                frozen_routes(frozen, rp, lp)
            for field, value in (('policy', 'variant_router'), ('gamma', .3), ('gamma', True)):
                invalid = copy.deepcopy(frozen)
                invalid['metadata_route_configs'][0][field] = value
                with self.assertRaises(ValueError):
                    frozen_routes(invalid, lp, rp)
            frozen['metadata_route_configs'] *= 2
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                frozen_routes(frozen, lp, rp)

    def test_inputs_are_checked_at_start_and_end(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input'
            path.write_text('original')
            checks = snapshot_inputs([path], {str(path): sha(path)})
            verify_unchanged(checks)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed_during_run'):
                verify_unchanged(checks)
            with self.assertRaisesRegex(ValueError, 'source_input_changed'):
                snapshot_inputs([], checks)

    def test_cli_reuses_predictions_without_model_forward_and_records_all_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = materialize(root)
            out = root / 'route'
            with patch('sys.argv', route_argv(paths, out)), patch.object(
                    ResidualRanker, 'forward', side_effect=AssertionError('model forward forbidden')):
                main()
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['gammas'], list(GAMMAS))
            self.assertEqual(summary['model_forward_calls'], 0)
            self.assertEqual((out / 'EXIT_CODE').read_text(), '0\n')
            self.assertEqual(summary['routing_counts']['explicit_missing'], 2)
            self.assertIn(str(paths[2]), summary['inputs_sha256'])
            for gamma in GAMMAS:
                folder = out / ('gamma_' + str(gamma))
                self.assertEqual(len((folder / 'predictions.jsonl').read_text().splitlines()), 2)
                self.assertEqual(len((folder / 'missing_predictions.jsonl').read_text().splitlines()), 2)

    def test_cli_refuses_stale_blend_origin_and_incomplete_result(self):
        for field, value in (('cache_sha256', 'stale'), ('coarse_config', 'different'),
                             ('candidate_order_sha256', 'different'), ('source_sha256', 'old-code')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                paths = materialize(root)
                summary_path = paths[3] / 'summary.json'
                summary = json.loads(summary_path.read_text())
                summary[field] = value
                summary_path.write_text(json.dumps(summary))
                with patch('sys.argv', route_argv(paths, root / 'bad')), self.assertRaisesRegex(ValueError, 'origin_mismatch'):
                    main()
                self.assertEqual((root / 'bad' / 'EXIT_CODE').read_text(), '1\n')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = materialize(root)
            (paths[3] / 'EXIT_CODE').write_text('1\n')
            with patch('sys.argv', route_argv(paths, root / 'bad')), self.assertRaisesRegex(ValueError, 'not_complete'):
                main()

    def test_train_val_refuses_synthetic_sealed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = materialize(root, 'verify')
            with patch('sys.argv', route_argv(paths, root / 'bad')), self.assertRaisesRegex(ValueError, 'access_violation'):
                main()

    def test_locked_reuses_only_exact_frozen_route_and_all_source_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = materialize(root, 'verify')
            with patch('sys.argv', route_argv(paths, root / 'locked', 'locked')):
                main()
            summary = json.loads((root / 'locked' / 'summary.json').read_text())
            self.assertEqual(summary['gammas'], [.5])
            self.assertEqual(summary['selection_scope'], 'frozen_metadata_route_configs')
            frozen = json.loads(paths[4].read_text())
            own = str(Path(__file__).with_name('h_global_metadata_route.py').resolve())
            del frozen['evaluation_inputs_sha256'][own]
            paths[4].write_text(json.dumps(frozen))
            with patch('sys.argv', route_argv(paths, root / 'bad', 'locked')), self.assertRaisesRegex(ValueError, 'source_not_frozen'):
                main()


if __name__ == '__main__':
    unittest.main()

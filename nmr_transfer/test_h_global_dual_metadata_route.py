"""功能概述：以 CPU 合成缓存检查双 gamma 公开观测路由及封存边界。

fixtures 沿用既有 blend/metadata-route 的自然短名单与名单外目标，materialize
只在临时目录生成 pair/cache/blend。测试明确检查 normal/缺测分支来自各自既有
排名，未知默认 normal，variant/source/ID 不选分支；同时验证完整库排名、Top10、
pair/origin、精确冻结 gamma 与新旧 helper 源码 SHA，以及运行中输入改写拒绝。
CLI 测试用合成 verify 字符串检查 locked 门禁，不读取真实 verify/test 数据；
不调用模型前向、训练、GPU、网络或外部 API，不改任何既有源码或研究产物。
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import h_global_dual_metadata_route as dual
from .h_global_ranker import ResidualRanker
from .h_global_ranker_blend import main as blend_main
from .test_h_global_metadata_route import fixtures
from .prepare import sha


def materialize(root, cohort='original_val'):
    left, right, data, _ = fixtures()
    data['manifest']['cohort'] = cohort
    lp, rp, cp = root / 'no.pt', root / 'aug.pt', root / 'cache.pt'
    for payload, path in ((left, lp), (right, rp), (data, cp)):
        torch.save(payload, path)
    blend, frozen_path = root / 'blend', None
    stage = 'train_val' if cohort in ('original_val', 'tune') else 'locked'
    argv = ['blend', '--stage', stage, '--no-aug-checkpoint', str(lp), '--aug-checkpoint', str(rp),
            '--cache', str(cp), '--out', str(blend)]
    if stage == 'locked':
        frozen = dict(selection_scope='train_val_only', frozen_utc='2026-10-20T20:01:00+00:00',
            blend_configs=[dict(no_aug_checkpoint=str(lp), aug_checkpoint=str(rp), gamma=g) for g in (.25, .75)],
            dual_metadata_route_configs=[dict(no_aug_checkpoint=str(lp), aug_checkpoint=str(rp), policy=dual.POLICY,
                normal_gamma=.25, missing_gamma=.75)],
            ranker_checkpoints_sha256={str(path): sha(path) for path in (lp, rp)},
            evaluation_inputs_sha256={str(path.resolve()): sha(path) for path in dual.sources()})
        frozen_path = root / 'frozen.json'
        frozen_path.write_text(json.dumps(frozen))
        argv += ['--frozen-selection', str(frozen_path)]
    with patch('sys.argv', argv):
        blend_main()
    return lp, rp, cp, blend, frozen_path


def argv(paths, out, stage='train_val', normal=.25, missing=.75):
    lp, rp, cp, blend, frozen = paths
    result = ['dual', '--stage', stage, '--no-aug-checkpoint', str(lp), '--aug-checkpoint', str(rp),
        '--cache', str(cp), '--blend-dir', str(blend), '--out', str(out),
        '--normal-gamma', str(normal), '--missing-gamma', str(missing)]
    if frozen is not None and stage == 'locked':
        result += ['--frozen-selection', str(frozen)]
    return result


class DualMetadataRouteTests(unittest.TestCase):
    def test_distinct_normal_missing_branches_and_no_mutation(self):
        _, _, data, branches = fixtures()
        for gamma, rank in (('0.25', 2), ('0.75', 1)):
            for index in (0, 2):
                branches[gamma][index]['rank'] = rank
        before = copy.deepcopy(branches)
        rows, report, counts = dual.route_predictions(data, branches, .25, .75)
        self.assertEqual([r['route_gamma'] for r in rows], [.25, .25, .75, .75])
        self.assertEqual(rows[0]['rank'], 2)
        self.assertEqual(rows[2]['rank'], 1)
        self.assertEqual(rows[1]['rank'], 11)
        self.assertEqual(counts, dict(undeclared_missing=2, explicit_missing=2))
        self.assertEqual(branches, before)
        self.assertIn('stress', report)
        self.assertEqual(len(rows), len(data['rows']))

    def test_unknown_defaults_normal_and_variant_alone_never_routes(self):
        _, _, data, branches = fixtures()
        data['rows'][0]['query_observation'] = dict(partial_peaks=True)
        data['rows'][2]['query_observation'] = {}
        data['rows'][3]['query_observation'] = None
        rows, _, _ = dual.route_predictions(data, branches, .25, .75)
        self.assertEqual([r['route_gamma'] for r in rows], [.75, .25, .25, .25])
        del data['rows'][3]['query_observation']
        again, _, _ = dual.route_predictions(data, branches, .25, .75)
        self.assertEqual([r['route_gamma'] for r in again], [r['route_gamma'] for r in rows])

    def test_only_two_public_fields_can_choose_branch(self):
        class PublicOnly(dict):
            def get(self, key, default=None):
                if key not in ('windows', 'partial_peaks'):
                    raise AssertionError('forbidden routing field:' + key)
                return super().get(key, default)
        _, _, data, branches = fixtures()
        data['rows'][0]['query_observation'] = PublicOnly(windows=[[0., 4.]])
        for index in (1, 2, 3):
            data['rows'][index]['query_observation'] = PublicOnly()
        before, _, _ = dual.route_predictions(data, branches, .25, .75)
        for rows in [data['rows']] + list(branches.values()):
            for row in rows:
                row.update(source='renamed', record_id='renamed', smiles='C', group='renamed', variant='new_' + row['variant'])
        after, _, _ = dual.route_predictions(data, branches, .25, .75)
        self.assertEqual([(r['rank'],r['top10'],r['route_gamma']) for r in before],
                         [(r['rank'],r['top10'],r['route_gamma']) for r in after])

    def test_equal_gammas_preserve_all_rows_without_duplicates(self):
        _, _, data, branches = fixtures()
        rows, _, _ = dual.route_predictions(data, branches, .25, .25)
        self.assertEqual(len(rows), len(branches['0.25']))
        self.assertEqual([(r['rank'], r['top10']) for r in rows], [(r['rank'], r['top10']) for r in branches['0.25']])

    def test_invalid_observation_and_unregistered_gamma_rejected(self):
        for value in ('unknown', dict(partial_peaks='False'), dict(windows=[[3., 1.]])):
            _, _, data, branches = fixtures()
            data['rows'][0]['query_observation'] = value
            with self.assertRaises(ValueError):
                dual.route_predictions(data, branches, .25, .75)
        for value in (True, .3, float('nan')):
            with self.assertRaisesRegex(ValueError, 'unregistered_gamma'):
                dual.validate_gamma(value)

    def test_prediction_identity_full_library_and_no_target_injection_guards(self):
        for failure in ('outside_rank', 'duplicate', 'identity', 'short_top10'):
            _, _, data, branches = fixtures()
            if failure == 'outside_rank':
                branches['0.25'][1]['rank'] = 1
            elif failure == 'duplicate':
                branches['0.75'].append(copy.deepcopy(branches['0.75'][0]))
            elif failure == 'identity':
                branches['0.25'][0]['record_id'] = 'other'
            else:
                branches['0.75'][2]['top10'].pop()
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                dual.route_predictions(data, branches, .25, .75)

    def test_frozen_pair_policy_both_gammas_and_unique_config_are_exact(self):
        lp, rp = str(Path('/tmp/no.pt').resolve()), str(Path('/tmp/aug.pt').resolve())
        config = dict(no_aug_checkpoint=lp, aug_checkpoint=rp, policy=dual.POLICY,
                      normal_gamma=.25, missing_gamma=.75)
        frozen = dict(dual_metadata_route_configs=[config])
        self.assertEqual(dual.frozen_route(frozen, lp, rp, .25, .75), (.25,.75))
        for args in ((rp,lp,.25,.75),(lp,rp,.75,.25)):
            with self.assertRaisesRegex(ValueError, 'not_frozen'):
                dual.frozen_route(frozen, *args)
        for field, value in (('policy','variant'),('normal_gamma',.3),('missing_gamma',True),('no_aug_checkpoint','relative.pt')):
            invalid = copy.deepcopy(frozen)
            invalid['dual_metadata_route_configs'][0][field] = value
            with self.assertRaises(ValueError):
                dual.frozen_route(invalid, lp, rp, .25,.75)
        frozen['dual_metadata_route_configs'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            dual.frozen_route(frozen, lp, rp, .25,.75)

    def test_train_val_cli_has_direct_outputs_full_report_and_no_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = materialize(root)
            out = root / 'dual'
            with patch('sys.argv', argv(paths, out)), patch.object(ResidualRanker, 'forward', side_effect=AssertionError('forward forbidden')):
                dual.main()
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual((summary['normal_gamma'],summary['missing_gamma']),(.25,.75))
            self.assertEqual(summary['selection_scope'],'train_val_only')
            self.assertEqual(summary['model_forward_calls'],0)
            self.assertEqual(len((out/'predictions.jsonl').read_text().splitlines()),2)
            self.assertEqual(len((out/'missing_predictions.jsonl').read_text().splitlines()),2)
            self.assertEqual(summary['cache_sha256'],sha(paths[2]))
            self.assertEqual(summary['no_aug_sha256'],sha(paths[0]))
            self.assertIn('overall',summary['metrics'])
            self.assertIn(str(Path(dual.__file__).with_name('h_global_metadata_route.py').resolve()),summary['inputs_sha256'])
            self.assertEqual((out/'EXIT_CODE').read_text(),'0\n')

    def test_locked_cli_requires_real_finalization_source_pins_and_exact_gammas(self):
        for failure in (None,'gamma','own_source','helper_source','draft'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                paths = materialize(root,'verify')
                frozen = json.loads(paths[4].read_text())
                if failure in ('own_source','helper_source'):
                    source = Path(dual.__file__) if failure=='own_source' else Path(dual.__file__).with_name('h_global_metadata_route.py')
                    del frozen['evaluation_inputs_sha256'][str(source.resolve())]
                elif failure=='draft':
                    frozen['frozen_utc']=None
                paths[4].write_text(json.dumps(frozen))
                arguments = argv(paths,root/'dual','locked',normal=.5 if failure=='gamma' else .25)
                if failure is None:
                    with patch('sys.argv',arguments):
                        dual.main()
                    self.assertEqual(json.loads((root/'dual/summary.json').read_text())['selection_scope'],'frozen_dual_metadata_route_configs')
                else:
                    with patch('sys.argv',arguments), self.assertRaises(ValueError):
                        dual.main()
                    self.assertEqual((root/'dual/EXIT_CODE').read_text(),'1\n')

    def test_train_val_does_not_accept_synthetic_sealed_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = materialize(root,'verify')
            with patch('sys.argv',argv(paths,root/'bad')), self.assertRaisesRegex(ValueError,'access_violation'):
                dual.main()

    def test_stale_origin_incomplete_blend_and_pair_mismatch_rejected(self):
        for failure in ('origin','exit','pair'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve()
                paths=materialize(root)
                if failure=='origin':
                    path=paths[3]/'summary.json'; summary=json.loads(path.read_text()); summary['cache_sha256']='changed'; path.write_text(json.dumps(summary))
                elif failure=='exit':
                    (paths[3]/'EXIT_CODE').write_text('1\n')
                else:
                    saved=torch.load(paths[1],weights_only=False); saved['prediction_checkpoint_sha256']='other'; torch.save(saved,paths[1])
                with patch('sys.argv',argv(paths,root/'bad')), self.assertRaises(ValueError):
                    dual.main()

    def test_input_changed_during_route_is_rejected_with_actual_failure_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            paths=materialize(root)
            original=dual.route_predictions
            def mutate(*args):
                result=original(*args)
                with paths[2].open('ab') as stream:
                    stream.write(b'changed')
                return result
            with patch('sys.argv',argv(paths,root/'bad')), patch.object(dual,'route_predictions',side_effect=mutate), self.assertRaisesRegex(ValueError,'changed_during_run'):
                dual.main()
            self.assertEqual((root/'bad/EXIT_CODE').read_text(),'1\n')


if __name__ == '__main__':
    unittest.main()

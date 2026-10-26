"""功能概述：以 CPU 合成证据验证前向诊断的单位、缺失口径与封存边界。

测试覆盖谱线/标签/积分区分、频窗、未知元数据、显式 MHz 换算、加权尾部、
最近环境歧义、同次 sigma、描述性 W1、正常/压力分离，以及真实文件读写的
main 成功和输入变化失败。只构造临时小 bank，不读真实 verify/test 或调用模型。
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from nmr_transfer import h_research_forward_diagnostics as d
from nmr_transfer.h_condinfer import condition_key
from nmr_transfer.h_labels import make_key
from nmr_transfer.h_research_observation import prepare_observation


def observation(xs, weights=None, windows=()):
    weights = np.ones(len(xs)) if weights is None else np.asarray(weights, dtype=float)
    return dict(xs=list(xs), weights=(weights/weights.sum()).tolist(), raw_weights=weights.tolist(),
                windows=list(windows), semantics='assigned_labels', coverage='unknown', partial_peaks=False)


def record(residuals, variant='observed', **kw):
    return dict(source='synthetic', semantics='assigned_labels', coverage='unknown',
                solvent_status='unknown', solvent_family='unknown', pH_known=False, pH=None,
                pH_used_by_model=False, frequency_known=False, frequency_MHz=None, variant=variant, residuals=residuals, **kw)


class ForwardDiagnosticsTests(unittest.TestCase):
    def test_ppm_conversion_once_and_sigma_already_ppm(self):
        out = d.residual_diagnostics(observation([1.2]), [.1], np.array([0]), [.1])
        self.assertAlmostEqual(out['nearest_prediction_ppm'][0], 1.)
        self.assertAlmostEqual(out['absolute_residual_ppm'][0], .2)
        self.assertEqual(out['nearest_sigma_ppm'], [.1])
        summary = d.summarize([record(out)])
        width = summary['overall']['width_coverage']
        self.assertEqual(width['fraction_within_1sigma'], 0.)
        self.assertAlmostEqual(width['mean_absolute_standardized_residual'], 2.)

    def test_explicit_window_filters_observed_and_descriptive_reverse(self):
        query = dict(source='synthetic', measure='integral_weighted', observed_xs=[-1., .2, 9.],
                     observed_weights=[1., 2., 1.], observation=dict(windows=[[0., 1.]]))
        prepared = prepare_observation(query)
        self.assertEqual(prepared['xs'], [.2])
        self.assertEqual(prepared['coverage'], 'explicit_window')
        out = d.residual_diagnostics(prepared, [-.001, .08, .9])
        self.assertEqual(out['visible_predicted_quantiles'], 1)
        self.assertAlmostEqual(out['absolute_residual_ppm'][0], .21)
        self.assertAlmostEqual(out['descriptive_support_distances']['reverse_nearest_mean_ppm'], .6)
        self.assertFalse(out['descriptive_support_distances']['missing_peak_penalty'])

    def test_unknown_coverage_and_manual_labels_are_not_integrals(self):
        query = dict(source='nmrshiftdb2_2024', measure='observed_assigned_atom_equal_partial_possible',
                     observed_xs=[1., 1., 2.], observed_weights=[1., 1., 1.],
                     full_atom_labels=True, observation=dict(kind='complete_atoms', windows=[]))
        obs = prepare_observation(query)
        self.assertEqual(obs['coverage'], 'unknown')
        self.assertEqual(obs['semantics'], 'assigned_labels')
        self.assertEqual(obs['raw_weights'], [2., 1.])
        self.assertEqual(obs['reported_count'], 3)
        query['measure'] = 'integral_weighted'
        integral = prepare_observation(query)
        self.assertEqual(integral['semantics'], 'integral')
        self.assertIsNone(integral['reported_count'])

    def test_hmdb_lines_do_not_use_height_as_integral(self):
        query = dict(source='HMDB', record_id='h', q=[1., 4.], observation=dict(windows=[]))
        raw = {'h': dict(value=dict(x=[1., 1.04, 4.], y=[1., 900., 2.]))}
        obs = prepare_observation(query, raw)
        self.assertEqual(obs['semantics'], 'line_positions')
        np.testing.assert_allclose(obs['xs'], [1.02, 4.])
        self.assertEqual(obs['raw_weights'], [1., 1.])
        self.assertIsNone(obs['reported_count'])

    def test_metadata_masks_and_frequency_require_actual_explicit_fields(self):
        unknown = d.condition_evidence({'source': 'Water_400MHz_pH7'}, {})
        self.assertEqual(unknown['solvent_status'], 'unknown')
        self.assertFalse(unknown['pH_known'])
        self.assertFalse(unknown['frequency_known'])
        meta = dict(solvent='CDCl3', family='chloroform', raw_pH='7.0', pH_used=False)
        raw = dict(experimental_conditions=dict(frequency='400 MHz'))
        known = d.condition_evidence({}, meta, raw)
        self.assertEqual(known['pH'], 7.)
        self.assertFalse(known['pH_used_by_model'])
        self.assertEqual(known['frequency_MHz'], 400.)
        self.assertEqual(.1 * known['frequency_MHz'], 40.)
        for value in (400, '400', '0 MHz', '400 Hz', 'unknown'):
            self.assertIsNone(d.condition_evidence({}, {}, {'experimental_conditions': {'frequency': value}})['frequency_MHz'])
        explicit = d.condition_evidence({}, {}, {'solvent': 'declared rare solvent'})
        self.assertEqual(explicit['solvent_status'], 'reported')
        self.assertEqual(explicit['solvent_family'], 'reported_unclassified')

    def test_nearest_class_and_exchangeability_are_model_matched(self):
        out = d.residual_diagnostics(observation([1., 3.]), [.09, .11, .3],
                                     np.array([0, 5, 6]), [.1, .2, .3])
        self.assertEqual(out['model_matched_local_class'], ['nearest_class_ambiguous', 'OH'])
        self.assertEqual(out['model_matched_exchangeability'], ['ambiguous', 'exchangeable'])
        self.assertIsNone(out['nearest_sigma_ppm'][0])
        self.assertEqual(out['nearest_sigma_ppm'][1], .3)
        self.assertFalse(out['atom_assignment_metric'])
        self.assertEqual(d.summarize([record(out)])['overall']['width_coverage']['positions'], 1)
        other = d.residual_diagnostics(observation([1.]), [.1], np.array([8]))
        self.assertEqual(other['model_matched_exchangeability'], ['unclassified'])

    def test_root_nested_and_metadata_condition_priority(self):
        raw = dict(frequency='400 MHz', pH='5',
                   experimental_conditions=dict(frequency='90 MHz', pH='7'))
        query = dict(frequency='800 MHz', ph='3')
        meta = dict(raw_frequency='600 MHz', frequency='700 MHz', raw_pH='6')
        found = d.condition_evidence(query, meta, raw)
        self.assertEqual(found['frequency_MHz'], 400.)
        self.assertEqual(found['frequency_field'], 'raw.frequency')
        self.assertEqual(found['pH'], 6.)
        self.assertEqual(found['pH_field'], 'metadata.raw_pH')
        self.assertFalse(found['pH_used_by_model'])
        no_prior = d.condition_evidence(query, {}, raw)
        self.assertEqual(no_prior['pH'], 5.)
        self.assertEqual(no_prior['pH_field'], 'raw.pH')
        raw['frequency'] = 400
        nested = d.condition_evidence(query, {}, raw)
        self.assertEqual(nested['frequency_MHz'], 90.)
        self.assertEqual(nested['frequency_field'], 'raw.experimental_conditions.frequency')
        query_only = d.condition_evidence(query, {})
        self.assertEqual(query_only['frequency_MHz'], 800.)
        self.assertEqual(query_only['frequency_field'], 'query.frequency')
        self.assertEqual(query_only['pH'], 3.)
        self.assertEqual(query_only['pH_field'], 'query.ph')
        metadata_only = d.condition_evidence({}, meta)
        self.assertEqual(metadata_only['frequency_MHz'], 600.)
        self.assertEqual(metadata_only['frequency_field'], 'metadata.raw_frequency')
        direct_meta = d.condition_evidence({}, {'frequency': '500 MHz'})
        self.assertEqual(direct_meta['frequency_MHz'], 500.)
        self.assertEqual(direct_meta['frequency_field'], 'metadata.frequency')
        for value in (400, '400', '400 Hz', 'unknown'):
            unknown = d.condition_evidence({}, {}, {'frequency': value})
            self.assertFalse(unknown['frequency_known'])
            self.assertIsNone(unknown['frequency_MHz'])
        zero_ph = d.condition_evidence({}, {}, {'ph': 0})
        self.assertEqual(zero_ph['pH'], 0.)
        self.assertEqual(zero_ph['pH_field'], 'raw.ph')

    def test_weighted_tail_statistics_and_strict_thresholds(self):
        out = d.weighted_stats([0., .5, 1., 2.], [.7, .1, .1, .1])
        self.assertEqual(out['median_ppm'], 0.)
        self.assertEqual(out['P95_ppm'], 2.)
        self.assertAlmostEqual(out['fraction_gt_0_5ppm'], .2)
        self.assertAlmostEqual(out['fraction_gt_1ppm'], .1)
        with self.assertRaisesRegex(ValueError, 'invalid_residuals'):
            d.weighted_stats([float('nan')])

    def test_query_equal_aggregation_ignores_arbitrary_integral_scale(self):
        a = d.residual_diagnostics(observation([1.], [100000.]), [0.])
        b = d.residual_diagnostics(observation([0.], [1.]), [0.])
        out = d.summarize([record(a), record(b)])['overall']
        self.assertAlmostEqual(out['query_equal_weighted_positions']['mean_ppm'], .5)
        self.assertAlmostEqual(out['query_mean_absolute_residual_ppm'], .5)

    def test_normal_and_stress_never_share_primary_average(self):
        good = record(d.residual_diagnostics(observation([1.]), [.1]))
        bad = record(d.residual_diagnostics(observation([10.]), [0.]), variant='frequency_crop')
        out = d.summarize([good, bad])
        self.assertEqual(out['overall']['queries'], 1)
        self.assertEqual(out['overall']['query_mean_absolute_residual_ppm'], 0.)
        self.assertEqual(out['stress']['frequency_crop']['query_mean_absolute_residual_ppm'], 10.)
        self.assertEqual(out['normal_records'], 1)

    def test_w1_is_support_distribution_distance_not_atom_error(self):
        out = d.residual_diagnostics(observation([2.]), [.1, .3])
        self.assertEqual(out['descriptive_support_distances']['W1_ppm'], 1.)
        self.assertEqual(out['descriptive_support_distances']['reverse_nearest_mean_ppm'], 1.)
        self.assertFalse(out['descriptive_support_distances']['atom_assignment_metric'])
        no_visible = d.residual_diagnostics(observation([2.], windows=[[1.9, 2.1]]), [.1, .3])
        self.assertIsNone(no_visible['descriptive_support_distances']['W1_ppm'])
        self.assertEqual(no_visible['visible_predicted_quantiles'], 0)

    def test_sealed_guard_requires_new_config_inputs_and_verify_qualification(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'freeze.json'
            config, inputs = {'metric_version': d.METRIC_VERSION}, {'/synthetic': 'abc'}
            with self.assertRaisesRegex(ValueError, 'new_frozen_selection'):
                d.guard_locked(None, inputs, config, 'test')
            frozen = dict(schema='nmr_research_20261022', selection_scope='train_val_only', frozen_utc='fixed',
                          evaluation_inputs_sha256=inputs, allowed_extract_configs=[config])
            p.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, 'config_not_frozen'):
                d.guard_locked(p, inputs, config, 'test')
            frozen['allowed_forward_diagnostics_configs'] = [config]
            p.write_text(json.dumps(frozen))
            d.guard_locked(p, inputs, config, 'test')
            with self.assertRaisesRegex(ValueError, 'input_changed'):
                d.guard_locked(p, {'/synthetic': 'changed'}, config, 'test')
            with self.assertRaisesRegex(ValueError, 'verify_not_eligible'):
                d.guard_locked(p, inputs, config, 'verify')

    def test_sealed_guard_precedes_bank_loading(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(d, 'required_inputs', return_value=[]), \
                mock.patch.object(d, 'load_bank', side_effect=AssertionError('must not parse sealed bank')) as loader:
            with self.assertRaisesRegex(ValueError, 'new_frozen_selection'):
                d.main(['--base', td, '--bank', 'absent', '--cohorts', td, '--fixed-identities', 'absent',
                        '--cohort', 'test', '--out', str(Path(td)/'out')])
            loader.assert_not_called()

    def _fixture(self, root):
        checkpoint, bankfile = root/'checkpoint.pt', root/'bank.pt'
        checkpoint.write_bytes(b'fixed checkpoint'); bankfile.write_bytes(b'mmap substitute')
        (root/'cohorts.pt').write_bytes(b'cohorts fixture'); (root/'manifest.json').write_text('{}')
        rows = [dict(domain='experimental', source='synthetic', record_id=str(i), smiles=s, group='group'+str(i),
                     cond=np.zeros(4, dtype=np.float32), q=np.full(128, float(i+1)),
                     measure='observed_assigned_atom_equal_partial_possible',
                     observation=dict(kind='unknown', windows=[])) for i, s in enumerate(['CC', 'CO'])]
        payload = dict(cohorts={'original_val': rows}, key_mode='recovered_local', metadata={},
                       annotations={make_key(r): dict(groups={0: [float(i+1)]}) for i, r in enumerate(rows)})
        key = condition_key(rows[0], {}, payload['key_mode'])
        bank = dict(single_indices=[1, 0], smiles=['CO', 'CC'], test=object(),
                    predictions={key: torch.tensor([[.2]*128, [.1]*128])},
                    candidate_quantile_classes={key: torch.zeros((2, 128), dtype=torch.uint8)},
                    candidate_quantile_sigma={key: torch.full((2, 128), .2)},
                    checkpoint_path=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        fixed = root/'fixed.json'
        fixed.write_text(json.dumps(dict(candidate_order_sha256=hashlib.sha256(b'CC\nCO').hexdigest())))
        (root/'cohort_contract.json').write_text(json.dumps(dict(
            cohorts_sha256=hashlib.sha256((root/'cohorts.pt').read_bytes()).hexdigest(),
            candidate_order_sha256=hashlib.sha256(b'CC\nCO').hexdigest(),
            cohort_query_sha256={'original_val': d.stamp_queries(rows)})))
        argv = ['--base', str(root), '--bank', str(bankfile), '--cohorts', str(root),
                '--fixed-identities', str(fixed), '--out', str(root/'out')]
        return bank, payload, argv, [bankfile, root/'cohorts.pt', root/'manifest.json', root/'cohort_contract.json', fixed]

    def test_hz_output_only_with_explicit_frequency_and_val_hash_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); bank, payload, argv, paths = self._fixture(root)
            fixed = json.loads((root/'fixed.json').read_text())
            fixed.update(cohort_query_sha256=json.loads((root/'cohort_contract.json').read_text())['cohort_query_sha256'])
            payload['annotations'][make_key(payload['cohorts']['original_val'][0])] = {'groups': {0: [1.1]}}
            args = SimpleNamespace(cohort='original_val', augment=False)
            raw = {'0': {'frequency': '400 MHz'}}
            with mock.patch.object(d, 'CANDIDATE_COUNT', 2), mock.patch.dict(d.QUERY_COUNTS, original_val=2):
                rows, _ = d.diagnose(bank, payload, raw, args, fixed)
                self.assertAlmostEqual(rows[0]['residuals']['absolute_residual_Hz'][0], 40., places=4)
                self.assertIsNone(rows[1]['residuals']['absolute_residual_Hz'])
                payload['cohorts']['original_val'][0]['q'] = np.full(128, 8.)
                with self.assertRaisesRegex(ValueError, 'fixed_validation_query_changed'):
                    d.diagnose(bank, payload, raw, args, fixed)

    def test_real_main_files_success_and_truth_global_bank_index(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); bank, payload, argv, paths = self._fixture(root)
            with mock.patch.object(d, 'CANDIDATE_COUNT', 2), mock.patch.dict(d.QUERY_COUNTS, original_val=2), \
                    mock.patch.object(d, 'required_inputs', return_value=paths), mock.patch.object(d, 'load_bank', return_value=bank), \
                    mock.patch.object(d.torch, 'load', return_value=payload), \
                    mock.patch.object(d, 'scoring_inputs', return_value=({}, {}, {}, {}, {})):
                d.main(argv)
            self.assertEqual((root/'out/EXIT_CODE').read_text(), '0\n')
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(summary['inputs_start_sha256'], summary['inputs_end_sha256'])
            self.assertLess(summary['overall']['query_mean_absolute_residual_ppm'], 1e-6)
            perquery = [json.loads(line) for line in (root/'out/per_query.jsonl').read_text().splitlines()]
            self.assertEqual(len(perquery), 2)
            self.assertFalse(summary['model_forward_executed'])
            self.assertFalse(summary['retrieval_ranking_executed'])

    def test_bank_prediction_dimension_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); bank, payload, argv, paths = self._fixture(root)
            fixed = json.loads((root/'fixed.json').read_text())
            fixed.update(cohort_query_sha256=json.loads((root/'cohort_contract.json').read_text())['cohort_query_sha256'])
            key = next(iter(bank['predictions']))
            bank['predictions'][key] = bank['predictions'][key][:, :127]
            with mock.patch.object(d, 'CANDIDATE_COUNT', 2), mock.patch.dict(d.QUERY_COUNTS, original_val=2):
                with self.assertRaisesRegex(ValueError, 'expected_128_quantiles'):
                    d.diagnose(bank, payload, {}, SimpleNamespace(cohort='original_val', augment=False), fixed)

    def test_real_main_input_change_has_failure_exit_no_complete_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); bank, payload, argv, paths = self._fixture(root)
            original = d.diagnose
            def changing(*args):
                result = original(*args)
                (root/'bank.pt').write_bytes(b'changed after actual diagnostics')
                return result
            with mock.patch.object(d, 'CANDIDATE_COUNT', 2), mock.patch.dict(d.QUERY_COUNTS, original_val=2), \
                    mock.patch.object(d, 'required_inputs', return_value=paths), mock.patch.object(d, 'load_bank', return_value=bank), \
                    mock.patch.object(d.torch, 'load', return_value=payload), \
                    mock.patch.object(d, 'scoring_inputs', return_value=({}, {}, {}, {}, {})), \
                    mock.patch.object(d, 'diagnose', side_effect=changing):
                with self.assertRaisesRegex(ValueError, 'input_changed_during_run'):
                    d.main(argv)
            self.assertEqual((root/'out/EXIT_CODE').read_text(), '1\n')
            self.assertFalse((root/'out/summary.json').exists())


if __name__ == '__main__':
    unittest.main()

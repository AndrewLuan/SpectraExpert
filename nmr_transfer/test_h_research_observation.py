"""功能概述：用 CPU 合成谱检查来源无关可训练观测模型的科学输入边界。

ObservationTests 覆盖原子标签/积分/线位置区分、HMDB 峰高与簇口径、未知覆盖和明确
删峰计数回退、F0/F1 同接口、重复质量守恒、宽度/报告/尾部梯度与有界正则，以及
纯参数 pack 往返和 train-only 拟合门禁。没有真实训练/验证/测试数据、网络或 GPU。
这些同步测试不比较检索效果，也不把 synthetic 拟合解释为真实 domain shift 收益。
"""
import copy
import unittest

import torch

from .h_research_observation import (O_FEATURE_NAMES, TrainableObservationModel,
                                     fit_observation_model, prepare_observation,
                                     training_environment_variant, training_line_variants)


class ObservationTests(unittest.TestCase):
    def query(self, xs=(1., 2.), weights=(3., 1.), measure=None, **state):
        return dict(source='nmrshiftdb2_2024', record_id='unused', smiles='unused',
                    observed_xs=list(xs), observed_weights=list(weights),
                    measure=measure or 'observed_assigned_atom_equal_partial_possible',
                    observation=dict(windows=state.get('windows', []), **{k: v for k, v in state.items() if k != 'windows'}))

    def test_integer_atomic_labels_never_become_integrals(self):
        labels = prepare_observation(self.query())
        integrals = prepare_observation(self.query(measure='integral_weighted'))
        unknown = prepare_observation(self.query(measure='arbitrary'))
        self.assertEqual(labels['semantics'], 'assigned_labels')
        self.assertEqual(labels['reported_count'], 4)
        self.assertEqual(integrals['semantics'], 'integral')
        self.assertIsNone(integrals['reported_count'])
        self.assertEqual(unknown['semantics'], 'quantile_support')
        self.assertIsNone(unknown['reported_count'])
        fractional = prepare_observation(self.query(xs=(1., 1.), weights=(.5, .5)))
        self.assertIsNone(fractional['reported_count'])

    def test_identity_fields_are_not_observation_or_network_inputs(self):
        query = self.query()
        a = prepare_observation(query)
        query.update(source='different_format_compatible_source', record_id='B', smiles='other', group='G')
        self.assertEqual(a, prepare_observation(query))
        self.assertFalse({'source', 'record_id', 'smiles', 'group'} & set(a))

    def test_hmdb_peak_heights_not_integrals_and_clusters_not_single_link(self):
        query = dict(source='HMDB', record_id='A', q=[1., 1.05, 1.1, 7.], observation=dict(windows=[]))
        raw = {'A': dict(value=dict(x=[1., 1.05, 1.1, 7.], y=[1., 1000., 500., 1.]))}
        obs = prepare_observation(query, raw)
        self.assertEqual(obs['xs'], [1.025, 1.1, 7.])
        self.assertEqual(obs['raw_weights'], [1., 1., 1.])
        self.assertIsNone(obs['reported_count'])
        raw['A']['value']['y'] = [999., 0., 3., 7.]
        self.assertEqual(obs, prepare_observation(query, raw))

    def test_unknown_is_preserved_and_window_is_explicit(self):
        obs = prepare_observation(self.query())
        self.assertEqual(obs['coverage'], 'unknown')
        self.assertEqual(obs['windows'], [])
        window = prepare_observation(self.query(windows=[[.5, 1.5]]))
        self.assertEqual(window['xs'], [1.])
        self.assertEqual(window['coverage'], 'explicit_window')

    def test_duplicate_labels_and_grouped_mass_are_equivalent(self):
        model = TrainableObservationModel()
        expanded = prepare_observation(self.query(xs=(1., 1., 1., 2.), weights=(1., 1., 1., 1.)))
        grouped = prepare_observation(self.query())
        full = model(torch.tensor([[1., 1., 1., 2.]]), torch.tensor([[0, 0, 0, 1]]), expanded,
                     H=torch.tensor([4.]), K=torch.tensor([2.]))
        compact = model(torch.tensor([[1., 2.]]), torch.tensor([[0, 1]]), grouped,
                        H=torch.tensor([4.]), K=torch.tensor([2.]), prediction_mass=torch.tensor([[3., 1.]]))
        self.assertTrue(torch.allclose(full, compact, atol=1e-6))

    def test_count_only_applies_to_atomic_normal_unknown_coverage(self):
        model = TrainableObservationModel()
        centers = torch.tensor([[1., 2.], [1., 2.]])
        tags = torch.zeros(2, 2, dtype=torch.long)
        obs = prepare_observation(self.query())
        normal = model(centers, tags, obs, H=torch.tensor([4., 8.]), K=torch.tensor([2., 4.]))
        self.assertTrue((normal[:, 2] > 0).all())
        for changed in (dict(obs, partial_peaks=True), dict(obs, windows=[[0., 3.]]),
                        dict(obs, semantics='integral', kind=1), dict(obs, semantics='line_positions', kind=2)):
            self.assertTrue(torch.equal(model(centers, tags, changed, H=torch.tensor([4., 8.]))[:, 2], torch.zeros(2)))

    def test_unknown_K_exactly_falls_back_to_H(self):
        model = TrainableObservationModel()
        args = (torch.tensor([[1., 2.]]), torch.tensor([[0, 1]]), prepare_observation(self.query()))
        a = model(*args, H=torch.tensor([4.]))
        b = model(*args, H=torch.tensor([4.]), K=torch.tensor([-1.]))
        self.assertTrue(torch.equal(a, b))

    def test_robust_tail_and_overreport_have_finite_backward(self):
        model = TrainableObservationModel()
        obs = prepare_observation(self.query(xs=(14.,), weights=(9.,)))
        output = model(torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[0, 1], [2, 3]]), obs,
                       H=torch.tensor([4., 6.]), K=torch.tensor([2., 2.]))
        self.assertEqual(output.shape, (2, len(O_FEATURE_NAMES)))
        (output.sum() + .01 * model.regularization()).backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        self.assertGreater(float(model.width_raw.grad.abs().sum()), 0.)
        self.assertGreater(float(model.tail_logit.grad.abs().sum()), 0.)
        self.assertGreater(float(model.report_logits.grad.abs().sum()), 0.)

    def test_F1_sigma_and_trust_optional_and_pack_roundtrip(self):
        model = TrainableObservationModel([.1] * 9)
        args = (torch.tensor([[1., 2.]], dtype=torch.float16), torch.tensor([[0, 1]]),
                prepare_observation(self.query()))
        output = model(*args, per_quantile_sigma=torch.tensor([[.3, .4]]), trust=torch.tensor([[.8, .9]]))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.equal(output, model(*args, per_quantile_sigma=torch.tensor([[.3, .4]]),
                                                 trust=torch.tensor([[0., 0.]]))))
        restored = TrainableObservationModel.load_pack(model.pack())
        self.assertTrue(torch.equal(output, restored(*args, per_quantile_sigma=torch.tensor([[.3, .4]]), trust=torch.tensor([[.8, .9]]))))

    def test_window_pdf_and_cdf_use_same_mixture_and_overlap_merges(self):
        model = TrainableObservationModel()
        obs = prepare_observation(self.query(xs=(1.,), weights=(1.,), windows=[[.5, 1.5]]))
        args = (torch.tensor([[1., 2.]]), torch.tensor([[0, 1]]))
        a = model(*args, obs)
        b = model(*args, dict(obs, windows=[[.5, 1.2], [1., 1.5]]))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.isfinite(a).all())

    def test_train_only_fit_decreases_synthetic_loss_without_mutating_inputs(self):
        record = dict(split='train', domain='experimental', source='ignored', record_id='ignored',
                      predicted_ppm=[.5, 2.], tags=[0, 1], H=4., K=2.,
                      observation=prepare_observation(self.query(xs=(1., 2.), weights=(3., 1.))))
        original = copy.deepcopy(record)
        model, audit = fit_observation_model([record], epochs=8, lr=.02)
        self.assertEqual(record, original)
        self.assertLess(audit['losses'][-1], audit['losses'][0])
        self.assertEqual(len(audit['training_numeric_sha256']), 64)
        self.assertFalse(audit['source_or_id_model_input'])
        self.assertGreater(float(model.regularization().detach()), 0.)
        with self.assertRaisesRegex(ValueError, 'train_only'):
            fit_observation_model([dict(record, split='val')])

    def test_invalid_weights_sigmas_and_candidate_shapes_rejected(self):
        with self.assertRaises(ValueError):
            prepare_observation(self.query(weights=(-1., 1.)))
        with self.assertRaises(ValueError):
            TrainableObservationModel([.2, .3])
        model = TrainableObservationModel()
        with self.assertRaises(ValueError):
            model(torch.ones(2, 3), torch.zeros(2, 3, dtype=torch.long), prepare_observation(self.query()),
                  prediction_mass=torch.ones(1, 3))

    def test_semantics_select_parameters_and_missing_semantics_rejected(self):
        model = TrainableObservationModel()
        obs = prepare_observation(self.query())
        args = (torch.tensor([[1., 2.]]), torch.tensor([[0, 1]]))
        before = model(*args, obs)
        with torch.no_grad():
            model.width_raw[1].add_(1.)
        self.assertTrue(torch.equal(before, model(*args, obs)))
        integral = dict(obs, semantics='integral', kind=1)
        self.assertFalse(torch.equal(before[:, 0], model(*args, integral)[:, 0]))
        for changed in (dict(obs, semantics='unknown'), {k: v for k, v in obs.items() if k != 'semantics'},
                        dict(obs, kind=3)):
            with self.assertRaises(ValueError):
                model(*args, changed)

    def test_window_density_integrates_to_one_with_robust_tail(self):
        model = TrainableObservationModel()
        grid = torch.linspace(-1., 1., 20001, dtype=torch.float64)
        logp = model._profile(torch.tensor([[.3, 1.4]], dtype=torch.float64),
                              torch.tensor([[.2, .4]], dtype=torch.float64),
                              torch.tensor([[.4, .6]], dtype=torch.float64),
                              grid, [[-1., 1.]], torch.tensor(.1, dtype=torch.float64))
        self.assertAlmostEqual(float(torch.trapezoid(logp.exp()[0], grid)), 1., places=6)

    def test_report_propensity_changes_relative_matching_not_uniform_density_scale(self):
        model = TrainableObservationModel()
        obs = prepare_observation(self.query(xs=(2.,), weights=(1.,)))
        args = (torch.tensor([[1., 2.]]), torch.tensor([[0, 1]]), obs)
        initial = model(*args)
        with torch.no_grad():
            model.report_logits[0].add_(1.)
        self.assertTrue(torch.allclose(initial[:, :2], model(*args)[:, :2], atol=1e-6))
        with torch.no_grad():
            model.report_logits[0, 1].sub_(3.)
        changed = model(*args)
        self.assertGreater(float(changed[0, 0].detach()), float(initial[0, 0].detach()))
        self.assertGreater(float(changed[0, 1].detach()), 0.)

    def test_invalid_certified_K_cannot_silently_become_unknown(self):
        model = TrainableObservationModel()
        args = (torch.tensor([[1., 2.]]), torch.tensor([[0, 1]]), prepare_observation(self.query()))
        for k in (5., 1.5):
            with self.assertRaisesRegex(ValueError, 'invalid_certified_K'):
                model(*args, H=torch.tensor([4.]), K=torch.tensor([k]))

    def test_line_augmentation_requires_actual_MHz_and_preserves_line_semantics(self):
        query = dict(source='HMDB', record_id='A', split='train', domain='experimental',
                     q=[1., 7.], observation=dict(windows=[]))
        raw = {'A': dict(value=dict(x=[1., 7.], y=[100., 1.]),
                         experimental_conditions=dict(frequency='600 MHz'))}
        original = copy.deepcopy(raw)
        result = training_line_variants(query, raw, seed=211)
        obs = result['variants'][0]['observation']
        self.assertEqual(obs['semantics'], 'line_positions')
        self.assertIsNone(obs['reported_count'])
        spacing = obs['xs'][1] - obs['xs'][0]
        self.assertAlmostEqual(spacing * 600, result['audit']['synthetic_spacing_Hz'], places=9)
        self.assertEqual(result, training_line_variants(query, raw, seed=211))
        self.assertEqual(raw, original)
        raw['A']['experimental_conditions']['frequency'] = '600'
        self.assertEqual(training_line_variants(query, raw)['audit']['skipped'], 'explicit_MHz_unavailable')
        with self.assertRaisesRegex(ValueError, 'train_only'):
            training_line_variants(dict(query, split='val'), original)

    def test_environment_augmentation_needs_certified_map_and_drops_whole_group(self):
        query = self.query(xs=(1., 2., 3.), weights=(1., 1., 1.))
        query.update(split='train', domain='experimental')
        self.assertEqual(training_environment_variant(query)['variants'], [])
        mapping = dict(certified=True, evidence_sha256='a' * 64, groups=['A', 'A', 'B'],
                       positions_ppm=[1., 2., 3.])
        result = training_environment_variant(query, mapping=mapping, seed=193)
        obs = result['variants'][0]['observation']
        self.assertIn(obs['xs'], ([1., 2.], [3.]))
        self.assertTrue(obs['partial_peaks'])
        self.assertEqual(obs['coverage'], 'declared_partial')
        malformed = dict(mapping, groups=[['A', 'B'], 'A', 'B'])
        self.assertEqual(training_environment_variant(query, mapping=malformed)['audit']['skipped'],
                         'ambiguous_or_misaligned_environment_mapping')
        self.assertEqual(training_environment_variant(query, mapping=dict(mapping, positions_ppm=[1., 2., 4.]))['variants'], [])
        with self.assertRaisesRegex(ValueError, 'train_only'):
            training_environment_variant(dict(query, split='test'), mapping=mapping)


if __name__ == '__main__':
    unittest.main()

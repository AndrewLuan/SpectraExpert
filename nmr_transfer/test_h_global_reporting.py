"""功能概述：以 CPU 小分子验证 H/等价组报告模型的质量和观测边界。

测试覆盖已知组大小、非交换环境、错误分区拒绝、未知 K 的旧概率精确回退、
代表氢与逐 H 数据的训练识别、概率归一化、重复结构均权，以及拒绝 val 拟合和
显式窗口误用。测试不读取真实实验数据、冻结测试集或远端文件，不使用网络/GPU。
"""
import unittest

import torch

from .h_global_reporting import fit_reporting, reporting_nll, structure_summary, training_records
from .h_observed import count_nll


PRIOR = dict(rates=[.25, .5, 1., 1.5], weights=[.1, .3, .5, .1])


class ReportingTests(unittest.TestCase):
    def test_high_symmetry_groups_preserve_H_mass(self):
        entry = dict(counts={'1H': 2}, groups=[dict(parents={0: 3, 7: 3}, mass=6),
                                             dict(parents={3: 1, 4: 1, 8: 1, 9: 1}, mass=4)])
        result = structure_summary('COc1ccc(OC)cc1', entry)
        self.assertEqual((result['H'], result['K']), (10, 2))
        self.assertEqual(result['group_sizes'], [6, 4])
        self.assertEqual((result['nonexchange_H'], result['nonexchange_K']), (10, 2))

    def test_exchangeable_environment_is_separate(self):
        result = structure_summary('CO', dict(groups=[dict(parents={0: 3}, mass=3), dict(parents={1: 1}, mass=1)]))
        self.assertEqual((result['H'], result['K']), (4, 2))
        self.assertEqual((result['nonexchange_H'], result['nonexchange_K']), (3, 1))

    def test_broken_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'partition_mismatch'):
            structure_summary('CO', dict(groups=[dict(parents={0: 2}, mass=2), dict(parents={1: 1}, mass=1)]))

    def test_unknown_K_keeps_original_probability_exactly(self):
        model = dict(atomic_prior=PRIOR, representative_rates=[1.], mixture_weights=[.1, .9])
        counts = torch.tensor([4., 10., 20.])
        result = reporting_nll(counts, torch.tensor([-1., -1., -1.]), 2, model)
        self.assertTrue(torch.equal(result, count_nll(counts, 2, PRIOR)))

    def test_mixture_handles_representatives_without_calling_them_missing_peaks(self):
        records = [dict(split='train', source='atomic', group=str(i), H=10, K=2, observed_count=2) for i in range(100)]
        model = fit_reporting(records, {'atomic': PRIOR}, prior_strength=5.)['atomic']
        self.assertGreater(sum(model['mixture_weights'][1:]), .8)
        n = torch.tensor([10.])
        new = reporting_nll(n, torch.tensor([2.]), 2, model)
        self.assertLess(float(new), float(count_nll(n, 2, PRIOR)))

    def test_full_H_training_retains_atom_component(self):
        records = [dict(split='train', source='atomic', group=str(i), H=10, K=2, observed_count=10) for i in range(60)]
        model = fit_reporting(records, {'atomic': PRIOR})['atomic']
        self.assertGreater(model['mixture_weights'][0], .97)

    def test_no_known_K_training_keeps_original_model(self):
        records = [dict(split='train', source='atomic', group='unknown', H=10, K=None, observed_count=2)]
        model = fit_reporting(records, {'atomic': PRIOR})['atomic']
        self.assertEqual(model['mixture_weights'], [1., 0., 0., 0., 0., 0.])
        counts = torch.tensor([10.])
        self.assertTrue(torch.equal(reporting_nll(counts, torch.tensor([2.]), 2, model), count_nll(counts, 2, PRIOR)))

    def test_noninteger_parent_mass_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'invalid_group_parent'):
            structure_summary('C', dict(groups=[dict(parents={0: 3.5}, mass=4)]))

    def test_mixture_is_a_normalized_count_distribution(self):
        model = dict(atomic_prior=PRIOR, representative_rates=[.5, 1.], mixture_weights=[.4, .3, .3])
        n = torch.tensor([4.], dtype=torch.float64)
        # count_nll 除以 k，零计数质量直接按各生成过程计算。
        zero = .4 * (.1 * .75**4 + .3 * .5**4 + .1 * torch.exp(torch.tensor(-6.)).item()) + .3 * .5**2
        mass = zero + sum(torch.exp(-k * reporting_nll(n, torch.tensor([2.]), k, model)).item() for k in range(1, 60))
        self.assertAlmostEqual(mass, 1., places=7)

    def test_duplicate_training_records_do_not_increase_structure_weight(self):
        a = dict(split='train', source='atomic', group='representative', H=10, K=2, observed_count=2)
        b = dict(split='train', source='atomic', group='full', H=10, K=2, observed_count=10)
        first = fit_reporting([a, b], {'atomic': PRIOR})['atomic']
        second = fit_reporting([a] * 8 + [b], {'atomic': PRIOR})['atomic']
        self.assertTrue(torch.allclose(torch.tensor(first['mixture_weights']), torch.tensor(second['mixture_weights']), atol=1e-7))

    def test_validation_cannot_fit(self):
        record = dict(split='val', source='atomic', group='x', H=10, K=2, observed_count=2)
        with self.assertRaisesRegex(ValueError, 'train_only'):
            fit_reporting([record], {'atomic': PRIOR})
        with self.assertRaisesRegex(ValueError, 'train_only'):
            training_records([dict(split='val', domain='experimental')], {}, {})

    def test_window_and_noninteger_labels_are_rejected(self):
        model = dict(atomic_prior=PRIOR, representative_rates=[1.], mixture_weights=[.1, .9])
        with self.assertRaisesRegex(ValueError, 'explicit_window'):
            reporting_nll(torch.tensor([10.]), torch.tensor([2.]), 2, model, {'windows': [[0., 5.]]})
        with self.assertRaisesRegex(ValueError, 'integer_labels'):
            reporting_nll(torch.tensor([10.]), torch.tensor([2.]), 2.5, model)


if __name__ == '__main__':
    unittest.main()

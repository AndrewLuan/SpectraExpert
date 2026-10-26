"""功能概述：用合成 CPU 数据核验固定 Hz 容量的单位、回退和观测边界。

FieldCapacityTests 检查预声明网格、显式 MHz 解析、Hz/ppm 换算、同分辨率
聚簇的容量效应、未知频率/非 HMDB/窗口回退、删峰后不恢复原细线，以及仅解析
指定原始 ID 的数据入口。CLI 封存集拒绝发生在任何文件或设备访问之前。
依赖 h_global_field_capacity 与已审核容量 DP；不读取真实数据、不训练、不访问
GPU、verify/test、服务器或外部服务，不修改生产文件。
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import h_global_field_capacity as field


class FieldCapacityTests(unittest.TestCase):
    def setUp(self):
        self.query = dict(source='HMDB', record_id='synthetic', smiles='CC',
                          q=np.array([.9, 1.1]), observation=dict(kind='unknown', windows=[]))
        self.record = dict(source='HMDB', nucleus='1H', record_id='synthetic',
                           experimental_conditions=dict(frequency='600 MHz'), value=dict(x=[.9, 1.1]))
        self.predictions = torch.stack([torch.full((128,), .1),
                                        torch.tensor([.09] * 64 + [.11] * 64)])

    def test_predeclared_grid_has_control_and_six_uniform_interventions(self):
        configurations = field.configs()
        self.assertEqual(len(configurations), 7)
        self.assertEqual(len({row['name'] for row in configurations}), 7)
        self.assertEqual({(row['width_Hz'], row['gamma']) for row in configurations if row['gamma']},
                         {(width, gamma) for width in (16., 24., 32.) for gamma in (.25, 1.)})

    def test_frequency_requires_explicit_positive_MHz_in_public_field(self):
        for value, expected in [('90 MHz', 90.), (' 400 mhz ', 400.), ('600.0 MHz', 600.)]:
            self.assertEqual(field.frequency_mhz(dict(experimental_conditions=dict(frequency=value))), expected)
        for value in [None, 600, '600', '600 Hz', '0 MHz', '-90 MHz', 'nan MHz', 'inf MHz']:
            self.assertIsNone(field.frequency_mhz(dict(experimental_conditions=dict(frequency=value))))
        self.assertIsNone(field.frequency_mhz(dict(frequency=600)))

    def test_Hz_to_ppm_uses_query_frequency_for_all_fields(self):
        self.assertAlmostEqual(field.resolution_ppm(self.record, 24.), .04)
        for frequency in [90., 300., 500.]:
            record = dict(experimental_conditions=dict(frequency=f'{frequency} MHz'))
            self.assertAlmostEqual(field.resolution_ppm(record, 24.) * frequency, 24.)
        with self.assertRaisesRegex(ValueError, 'invalid_width_Hz'):
            field.resolution_ppm(self.record, 0.)

    def test_single_capacity_penalizes_two_resolved_observations(self):
        result = field.field_capacity(self.predictions, self.query, self.record, 24.)
        self.assertGreater(float(result[0]), 1.)
        self.assertAlmostEqual(float(result[1]), 0., 6)
        low_field = dict(self.record, experimental_conditions=dict(frequency='90 MHz'))
        merged = field.field_capacity(self.predictions, self.query, low_field, 24.)
        self.assertTrue(torch.allclose(merged, torch.zeros_like(merged), atol=1e-6))

    def test_unknown_frequency_non_HMDB_and_window_fall_back_to_zero(self):
        unknown = dict(self.record, experimental_conditions=dict(frequency=None))
        window = dict(self.query, observation=dict(kind='window', windows=[[.8, 1.2]]))
        other = dict(self.query, source='NMRexp')
        for query, record in [(self.query, unknown), (window, self.record), (other, self.record)]:
            result = field.field_capacity(self.predictions, query, record, 24.)
            self.assertTrue(torch.equal(result, torch.zeros(2)))

    def test_partial_peaks_do_not_reintroduce_deleted_raw_lines(self):
        partial = dict(self.query, q=np.array([.9]),
                       observation=dict(kind='unknown', windows=[], partial_peaks=True))
        result = field.field_capacity(self.predictions, partial, self.record, 24.)
        self.assertTrue(torch.allclose(result, torch.zeros(2), atol=1e-6))

    def test_candidate_truth_identity_does_not_change_capacity(self):
        first = field.field_capacity(self.predictions, self.query, self.record, 24.)
        second = field.field_capacity(self.predictions, dict(self.query, smiles='CCCCCC'), self.record, 24.)
        self.assertTrue(torch.equal(first, second))

    def test_raw_reader_only_parses_requested_development_IDs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.jsonl'
            path.write_text('{"record_id":"not_requested",malformed\n' + json.dumps(self.record) + '\n')
            result = field.read_hmdb_records(path, [self.query])
            self.assertEqual(result, {'synthetic': self.record})
            path.write_text(json.dumps(self.record) + '\n' + json.dumps(self.record) + '\n')
            with self.assertRaisesRegex(ValueError, 'duplicate_raw_record'):
                field.read_hmdb_records(path, [self.query])

    def test_reference_requires_corresponding_report1_and_model_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'report_g1.0'
            folder.mkdir()
            path = folder / 'predictions.jsonl'
            row = dict(source='HMDB', record_id='synthetic', smiles='CC', rank=1,
                       top10=['CC'], variant='observed')
            path.write_text(json.dumps(row) + '\n')
            summary = dict(cohort='original_val', candidate_count=114982,
                           candidate_order_sha256='order', checkpoint_sha256='model',
                           query_sha256=field.stamp_queries([self.query]))
            (root / 'summary.json').write_text(json.dumps(summary))
            result = field.load_reference(path, [self.query], 'original_val', 'order', 'model')
            self.assertEqual(result[('HMDB', 'synthetic', 'CC')], row)
            with self.assertRaisesRegex(ValueError, 'provenance_changed'):
                field.load_reference(path, [self.query], 'original_val', 'order', 'other_model')
            with self.assertRaisesRegex(ValueError, 'requires_report1_reference'):
                field.load_reference(root / 'report_g0.25/predictions.jsonl', [self.query],
                                     'original_val', 'order', 'model')

    def test_CLI_refuses_sealed_cohorts_before_reading_any_files(self):
        for cohort in ['verify', 'test']:
            args = ['synthetic'] + [value for name in ['base', 'bank', 'widths', 'cohorts', 'reference', 'out']
                                    for value in ['--' + name, 'never_read']] + ['--cohort', cohort]
            with patch.object(sys, 'argv', args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    field.main()
            self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()

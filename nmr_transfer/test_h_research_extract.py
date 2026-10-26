"""功能概述：核验训练观测重建和缓存消费的科学边界。

测试细线/环境删除后粗排使用同一派生观测且原raw不变，未知频窗不被补全。
缓存必须具有实际producer EXIT0、对应内容SHA和一致manifest，损坏或半成品拒绝。
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .h_research_contract import input_fingerprints
from .h_research_extract import materialize_training_variant, content_sha
from .h_research_ranker import read_feature_cache


class ExtractionBoundaryTests(unittest.TestCase):
    def test_fine_lines_rebuild_all_numeric_support_without_mutating_source(self):
        query = dict(record_id='line', source='HMDB', q=np.array([1., 2.]),
                     observation=dict(kind='unknown', windows=[]))
        raw = {'line': dict(value=dict(x=[1., 2.], y=[10., 100.]), frequency='400 MHz')}
        before = copy.deepcopy(raw)
        support = dict(xs=[.99, 1.01, 1.99, 2.01], raw_weights=[1., 1., 1., 1.],
                       semantics='line_positions', windows=[], partial_peaks=False)
        observed, derived_raw = materialize_training_variant(query, dict(observation=support), raw)
        self.assertEqual(raw, before)
        self.assertEqual(derived_raw['line']['value']['x'], support['xs'])
        self.assertEqual(derived_raw['line']['value']['y'], [1., 1., 1., 1.])
        self.assertEqual(sorted(set(observed['q'])), support['xs'])
        self.assertEqual(observed['observation']['windows'], [])

    def test_environment_deletion_keeps_only_observed_mass(self):
        query = dict(record_id='assigned', q=[1., 2., 3.], observation=dict(kind='unknown', windows=[]))
        support = dict(xs=[1., 3.], raw_weights=[3., 1.], semantics='assigned_labels',
                       windows=[], partial_peaks=True)
        observed, _ = materialize_training_variant(query, dict(observation=support), {})
        self.assertEqual(int((observed['q'] == 1.).sum()), 96)
        self.assertEqual(int((observed['q'] == 3.).sum()), 32)
        self.assertTrue(observed['observation']['partial_peaks'])
        self.assertNotIn(2., observed['q'])

    def test_mask_hash_uses_actual_support(self):
        first = dict(q=np.array([1., 2.]), observation=dict(kind='unknown', windows=[]))
        same = dict(q=[1., 2.], observation=dict(kind='unknown', windows=[]))
        self.assertEqual(content_sha(first), content_sha(same))
        self.assertNotEqual(content_sha(first), content_sha(dict(same, q=[1., 3.])))

    def test_real_cache_exit_and_sha_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cache = root/'features.pt'
            torch.save(dict(rows=[], manifest=dict(schema='test', records=0)), cache)
            digest = input_fingerprints([cache])[str(cache.resolve())]
            (root/'summary.json').write_text(json.dumps(dict(schema='test', records=0, feature_cache_sha256=digest)))
            (root/'EXIT_CODE').write_text('1\n')
            with self.assertRaisesRegex(ValueError, 'not_exit0'):
                read_feature_cache(cache)
            (root/'EXIT_CODE').write_text('0\n')
            self.assertEqual(read_feature_cache(cache)[1], digest)
            with cache.open('ab') as stream:
                stream.write(b'altered')
            with self.assertRaisesRegex(ValueError, 'SHA_mismatch'):
                read_feature_cache(cache)


if __name__ == '__main__':
    unittest.main()

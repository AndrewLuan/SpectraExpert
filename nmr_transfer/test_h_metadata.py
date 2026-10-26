"""功能概述：用合成谱表验证冻结ID元数据恢复、哈希和旧/新条件路由。
TestMetadata覆盖恢复名称而不改原字段、已缺失字符串、来源/结构冲突与原文件
校验失败；确认新添test元数据不改变旧训练先验，额外被过滤训练行不参与新先验。
调用h_metadata和h_condinfer纯函数，临时文件仅含合成标签；不访问真实数据或GPU。
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from .data import identity
from .h_conditions import condition_from_old, solvent_features
from .h_condinfer import condition_key
from .h_labels import make_key
from .h_metadata import RECOVERED_MODE, metadata_view, raw_conditions, recover, balance_local_support, reencode_recovery
import torch


class TestMetadata(unittest.TestCase):
    def fixture(self):
        smi, _, split = identity('CCNC(C)=O')
        row = dict(smiles=smi, split=split, record_id='synthetic:1', source='NMRexp',
                   cond=np.zeros(4), xs=[2., 7.], weights=[3., 1.], domain='experimental')
        raw = dict(smiles=smi, record_id=row['record_id'], source=row['source'], nucleus='1H',
                   representation='integral_peak_spectrum', value=dict(x=row['xs'], y=row['weights']),
                   experimental_conditions=dict(solvent='C6D6', pH='7'))
        return row, raw

    def run_recovery(self, row, raw, metadata, expected=None):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'data.jsonl'; path.write_text(json.dumps(raw) + '\n')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return recover(path, [row], metadata, expected or digest, lambda *a, **kw: None)

    def test_exact_recovery_retains_legacy_and_routes_new_key(self):
        row, raw = self.fixture(); key = make_key(row)
        metadata = {key: condition_from_old(row['cond'])}
        old = condition_key(row, metadata, 'local')
        report = self.run_recovery(row, raw, metadata)
        self.assertEqual(report['matched'], 1)
        self.assertEqual(condition_key(row, metadata, 'local'), old)
        self.assertNotEqual(condition_key(row, metadata, RECOVERED_MODE), old)
        recovered = metadata_view(metadata, RECOVERED_MODE)[key]
        self.assertEqual(recovered['family'], 'benzene')
        self.assertTrue(recovered['metadata_known']); self.assertFalse(recovered['pH_used'])
        self.assertFalse(metadata_view(metadata, 'local')[key]['metadata_known'])

    def test_missing_sentinel_uses_evidence_fallback(self):
        meta, field = raw_conditions(dict(solvent='NaN', experimental_conditions=dict(solvent='None'),
                                          source_metadata=dict(NMR_solvent='CD3OD')))
        self.assertEqual(meta['family'], 'methanol')
        self.assertEqual(field, 'source_metadata.NMR_solvent')
        meta, _ = raw_conditions(dict(solvent='unknown'))
        self.assertFalse(meta['metadata_known'])
        meta, _ = raw_conditions(dict(solvent='not_known'))
        self.assertFalse(meta['metadata_known'])
        meta, _ = raw_conditions(dict(solvent='not_known', experimental_conditions=dict(solvent='not_known')))
        self.assertFalse(meta['metadata_known']); self.assertEqual(meta['family'], 'unknown')

    def test_explicit_aliases_and_ambiguous_spellings(self):
        for value, family in [('MeOD', 'methanol'), ('MeOH-d4', 'methanol'),
                              ('(CD3)2SO', 'dmso'), ('CD3COCD3', 'acetone'),
                              ('CHCl3-d', 'chloroform'), ('C₆D₆', 'benzene')]:
            self.assertEqual(solvent_features(value)[1], [family])
        for value in ['CDC13', 'CDCI3', 'CD3Cl']:
            self.assertEqual(solvent_features(value)[1], ['other'])

    def test_reencode_retains_original_vector_and_nested_missing_evidence(self):
        row, _ = self.fixture(); key = make_key(row); legacy = condition_from_old(row['cond'])
        original = legacy['vector'].copy()
        meta = dict(legacy, recovered_family='other', recovery_split='train',
                    solvent_recovery=dict(raw_solvent='not_known',
                                          experimental_conditions=dict(solvent='not_known'), NMR_solvent=None))
        audit = reencode_recovery({key: meta})
        self.assertEqual(audit['changed_family_records'], 1)
        np.testing.assert_array_equal(meta['vector'], original)
        self.assertEqual(meta['recovered_family'], 'unknown')
        self.assertFalse(meta['recovered_metadata_known'])
        self.assertEqual(meta['solvent_recovery']['experimental_conditions']['solvent'], 'not_known')

    def test_hash_failure_does_not_promote_metadata(self):
        row, raw = self.fixture(); metadata = {}
        with self.assertRaisesRegex(ValueError, 'raw_input_sha256_mismatch'):
            self.run_recovery(row, raw, metadata, 'bad')
        self.assertEqual(metadata, {})

    def test_identity_mismatch_rejects_same_record_id(self):
        row, raw = self.fixture(); raw['smiles'] = 'CO'
        with self.assertRaisesRegex(ValueError, 'frozen_identity_mismatch'):
            self.run_recovery(row, raw, {})

    def test_recovered_only_records_do_not_change_training_cohort(self):
        original = ('experimental', 'NMRexp', 'a', 'CC')
        extra = ('experimental', 'NMRexp', 'b', 'CO')
        metadata = {original: dict(vector=[0]), extra: dict(vector=[0], recovered_vector=[1],
                                                         recovery_only=True, recovery_split='train')}
        self.assertEqual(set(metadata_view(metadata, 'local')), {original})
        self.assertEqual(set(metadata_view(metadata, RECOVERED_MODE)), {original})
        metadata[extra]['recovery_split'] = 'test'
        self.assertEqual(set(metadata_view(metadata, RECOVERED_MODE)), {original, extra})

    def test_local_balancing_preserves_each_source_solvent_mass(self):
        rows = []; metadata = {}
        for i, smi in enumerate(['CN(C)C=O', 'CC', 'CCC', 'CCCC', 'CO']):
            row = dict(domain='experimental', source='synthetic', record_id=str(i), smiles=smi)
            rows.append(row); metadata[make_key(row)] = dict(family='benzene' if i < 4 else 'water')
        weights = torch.tensor([.1, .1, .1, .1, .6], dtype=torch.float64)
        result = balance_local_support(rows, metadata, weights)
        self.assertAlmostEqual(float(result[:4].sum()), .4)
        self.assertAlmostEqual(float(result[4]), .6)
        self.assertGreater(float(result[0]), .1)
        torch.testing.assert_close(result[1:4], result[1].expand(3))


if __name__ == '__main__':
    unittest.main()

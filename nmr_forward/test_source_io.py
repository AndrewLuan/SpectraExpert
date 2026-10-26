"""功能概述：检验 source_io 的跨来源语义规范化与证据保留。
主要职责：SourceTests 覆盖 WanJuan 嵌套核种、区间、频率、缺失 J、积分冲突及
立体结构恢复的身份边界。测试不访问服务器，不修改源数据。
关系与输入输出：使用最小峰表调用 normalize_record，确保返回 prepare.convert
所需字段且不改变原输入；可用 python -m unittest nmr_forward.test_source_io 运行。
"""
import copy
import unittest
from nmr_forward.source_io import normalize_record


def record():
    return {'source': 'WanJuan', 'record_id': 'abc', 'smiles': 'CC(O)C(=O)O',
            'representation': 'integral_peak_spectrum',
            'experimental_conditions': {'nucleus': '1H', 'frequency': '400 MHz', 'solvent': 'CDCl3'},
            'annotations': [{'shift': 3.1, 'shift_end': 3.3, 'integration_h': 2,
                             'j_hz': [], 'multiplicity': 'm', 'raw': '3.1-3.3 (m, 2H)'}],
            'value': {'x': [3.1], 'y': [2]}, 'provenance': {'smiles_std': 'C[C@H](O)C(=O)O'}}


class SourceTests(unittest.TestCase):
    def test_wanjuan_semantics_and_no_mutation(self):
        r = record(); before = copy.deepcopy(r)
        out, reason = normalize_record(r)
        self.assertIsNone(reason); self.assertEqual(r, before)
        self.assertEqual(out['nucleus'], '1H'); self.assertEqual(out['frequency'], 400)
        self.assertAlmostEqual(out['value']['x'][0], 3.2)
        self.assertEqual(out['value']['y'], [2]); self.assertIn('@', out['smiles'])
        self.assertEqual(out['source_locator']['provenance'], r['provenance'])
        self.assertEqual(out['normalization_evidence']['original_value'], r['value'])

    def test_nmr_exp(self):
        r = record(); r['nucleus'] = '1H'; r['frequency'] = 600
        r['annotations'] = [{'shift_min': 3.1, 'shift_max': 3.3, 'shift_mid': 3.2,
                             'integration_h': 2, 'multiplicity': 'dd', 'coupling_hz': [7, 2]}]
        out, reason = normalize_record(r)
        self.assertIsNone(reason); self.assertEqual(out['annotations'][0]['coupling_hz'], [7, 2])
        self.assertEqual(out['frequency'], 600)

    def test_incomplete_metadata_is_flagged_not_invented(self):
        r = record(); r['experimental_conditions']['frequency'] = 'MHz'
        r['annotations'][0]['raw'] = '3.1 (ddd, J=11.2, 7.4, 5.2 Hz, 2H)'
        out, reason = normalize_record(r)
        self.assertIsNone(reason); self.assertIsNone(out['frequency'])
        self.assertIn('raw_J_present_but_unparsed', out['normalization_flags'])
        self.assertEqual(out['annotations'][0]['coupling_hz'], [])

    def test_salt_change_not_applied(self):
        r = record(); r['provenance']['smiles_std'] += '.Cl'
        out, _ = normalize_record(r)
        self.assertEqual(out['smiles'], r['smiles'])
        self.assertIn('provenance_structure_differs', out['normalization_flags'])

    def test_conflicting_labels_rejected(self):
        r = record(); r['annotations'][0]['integration_h'] = 1
        self.assertEqual(normalize_record(r)[1], 'annotation_integral_mismatch')
        r = record(); r['nucleus'] = '13C'
        self.assertEqual(normalize_record(r)[1], 'conflicting_nucleus')


if __name__ == '__main__':
    unittest.main()

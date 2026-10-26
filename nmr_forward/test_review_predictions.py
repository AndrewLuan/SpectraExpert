# 功能概述：验证预测复查只生成线索、不确诊、不删除记录，且保持测试分母与原文件。
# 主要职责：用人工乙醇峰表测试交换氢容许量、超量/缺失、来源片段证据和输出保护。
# 代码库关系：调用review_predictions，不读取服务器数据或真实测试标签。
# 副作用：仅使用自动清理的临时目录；错误分母、非法阈值及覆写请求必须拒绝。
import copy
import json
import pathlib
import sys
import tempfile
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from review_predictions import annotate_record,review_file


def row(integral=6,error=.1):
    return dict(record_id='synthetic',smiles='CCO',structure_protons=6,
                reported_integral=integral,error_w1_ppm=error,flags=[],source='fixture',
                source_locator={},group='group',predicted_shifts=[1,2,3])


class ReviewTests(unittest.TestCase):
    def test_exchangeable_deficit_does_not_establish_error(self):
        source=row(5,4);before=copy.deepcopy(source);result=annotate_record(source,3)
        self.assertEqual(source,before)
        self.assertIn('deficit_compatible_with_exchangeable_H',result['review_categories'])
        self.assertEqual(result['review_priority'],3)
        self.assertFalse(result['confirmed_data_error'])

    def test_structural_disagreement_plus_high_error_prioritized(self):
        for integral in (3,8):
            result=annotate_record(row(integral,4),3)
            self.assertEqual(result['review_priority'],1)
            self.assertFalse(result['confirmed_data_error'])

    def test_source_fragment_evidence_preserved(self):
        source=row();source['source_locator']={'provenance':{'smiles_std':'CCO.C'}}
        result=annotate_record(source,3)
        self.assertIn('source_structure_fragment_omitted',result['review_categories'])
        self.assertTrue(result['diagnostic_evidence']['source_fragment_candidate']['dropped_has_carbon'])

    def test_all_records_retained_and_denominator_guarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);pred=root/'pred.jsonl';metric=root/'metrics.json'
            pred.write_text('\n'.join(json.dumps(r) for r in (row(),row(8,4))))
            metric.write_text(json.dumps({'n':3,'review_threshold_from_validation_p99':3}))
            with self.assertRaisesRegex(ValueError,'denominator'):
                review_file(pred,metric,root/'out.jsonl',root/'summary.json')
            self.assertFalse((root/'out.jsonl').exists())
            metric.write_text(json.dumps({'n':2,'review_threshold_from_validation_p99':3}))
            report=review_file(pred,metric,root/'out.jsonl',root/'summary.json')
            self.assertEqual(report['n'],2)
            self.assertEqual(len((root/'out.jsonl').read_text().splitlines()),2)
            with self.assertRaises(FileExistsError):
                review_file(pred,metric,root/'out.jsonl',root/'summary.json')

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):annotate_record(row(),float('nan'))


if __name__=='__main__':unittest.main()

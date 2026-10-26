# 功能概述：验证最终测试前候选冻结的身份校验、验证集唯一选择和后续变更检测。
# 主要职责：FreezeSelectionTests构造临时小型完成run，检查primary/auxiliary、同分规则、
# 缓存/分母/epoch不一致、缺文件、NaN、拒绝覆盖及冻结后任意受保护文件变更。
# 代码库关系：只调用freeze_selection.py，不导入train，不创建或读取真实test数据。
# 输入输出与副作用：所有文件与随机权重均在自动清理临时目录；失败不产生manifest。
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

try:
    from .freeze_selection import freeze_selection, verify_manifest
except ImportError:
    from freeze_selection import freeze_selection, verify_manifest


class FreezeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()

    def tearDown(self):
        self.directory.cleanup()

    def run_artifact(self, name, w1=0.2, auxiliary=0, n=20, cache='a' * 64):
        run = self.root / name
        run.mkdir()
        validation = {'n': n, 'integral_weighted_W1_ppm': w1, 'median_W1_ppm': w1,
                      'matched_known_J_MAE_Hz': None}
        provenance = {'cache_sha256': 'b' * 64, 'validation_cache_sha256': cache}
        (run / 'validation.json').write_text(json.dumps(validation))
        (run / 'provenance.json').write_text(json.dumps(provenance))
        (run / 'complete.json').write_text(json.dumps({'epochs': 2, 'best_validation_W1_ppm': w1}))
        (run / 'history.jsonl').write_text('\n'.join(json.dumps({'epoch': epoch, 'validation': validation}) for epoch in (1, 2)) + '\n')
        torch.save({'epoch': 1, 'aux_classes': auxiliary, 'model': {'weight': torch.tensor([1.])},
                    'provenance': provenance}, run / 'best.pt')
        return run

    def test_select_primary_and_auxiliary_only_from_validation(self):
        base = self.run_artifact('base', .1)
        aux = self.run_artifact('aux', .2, 9)
        worse = self.run_artifact('worse', .3, 9)
        # 无效test文件若被读取会出错；选择必须完全忽略其内容。
        (base / 'test_metrics.json').write_text('not valid JSON')
        out = self.root / 'selection.json'
        result = freeze_selection([worse, base, aux], out)
        self.assertEqual(result['selected'], {'primary': str(base), 'auxiliary': str(aux)})
        self.assertFalse(result['test_access']['accessed_by_selection_tool'])
        self.assertEqual(verify_manifest(out), result)
        for candidate in result['candidates']:
            self.assertEqual(set(candidate['files']), {'best.pt', 'validation.json', 'provenance.json', 'complete.json', 'history.jsonl'})
        before = out.read_bytes()
        with self.assertRaises(FileExistsError):freeze_selection([base], out)
        self.assertEqual(before, out.read_bytes())

    def test_tie_is_independent_of_argument_order_and_aux_can_be_absent(self):
        a = self.run_artifact('a');b = self.run_artifact('b')
        first = freeze_selection([b, a], self.root / 'first.json')
        second = freeze_selection([a, b], self.root / 'second.json')
        self.assertEqual(first['selected'], second['selected'])
        self.assertEqual(first['selected'], {'primary': str(a), 'auxiliary': None})

    def test_mismatched_cache_and_denominator_rejected(self):
        a = self.run_artifact('a')
        for name, kwargs, pattern in [('cache', {'cache': 'c' * 64}, 'cache SHA'),
                                      ('denominator', {'n': 21}, 'validation n')]:
            with self.subTest(name=name):
                b = self.run_artifact(name, **kwargs)
                out = self.root / (name + '.json')
                with self.assertRaisesRegex(ValueError, pattern):freeze_selection([a, b], out)
                self.assertFalse(out.exists())

    def test_missing_complete_or_invalid_best_epoch_rejected(self):
        missing = self.run_artifact('missing');(missing / 'complete.json').unlink()
        with self.assertRaises(FileNotFoundError):freeze_selection([missing], self.root / 'missing.json')
        bad = self.run_artifact('bad');ck = torch.load(bad / 'best.pt', weights_only=False)
        ck['epoch'] = 3;torch.save(ck, bad / 'best.pt')
        with self.assertRaisesRegex(ValueError, 'epoch'):freeze_selection([bad], self.root / 'bad.json')

    def test_best_epoch_metric_and_embedded_cache_are_checked(self):
        epoch = self.run_artifact('epoch')
        history = [json.loads(line) for line in (epoch / 'history.jsonl').read_text().splitlines()]
        history[0]['validation']['integral_weighted_W1_ppm'] = .3
        (epoch / 'history.jsonl').write_text('\n'.join(json.dumps(row) for row in history))
        with self.assertRaisesRegex(ValueError, 'epoch does not match'):
            freeze_selection([epoch], self.root / 'epoch.json')
        cache = self.run_artifact('cache')
        ck = torch.load(cache / 'best.pt', weights_only=False);ck['provenance']['cache_sha256'] = 'd' * 64
        torch.save(ck, cache / 'best.pt')
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            freeze_selection([cache], self.root / 'cache.json')

    def test_nonfinite_metric_and_parameter_rejected(self):
        for idx, invalid in enumerate((float('nan'), float('inf'))):
            run = self.run_artifact('metric' + str(idx), invalid)
            with self.assertRaises(ValueError):freeze_selection([run], self.root / f'nan{idx}.json')
        run = self.run_artifact('tensor')
        ck = torch.load(run / 'best.pt', weights_only=False);ck['model']['weight'][0] = float('nan')
        torch.save(ck, run / 'best.pt')
        with self.assertRaisesRegex(ValueError, 'nonfinite model'):
            freeze_selection([run], self.root / 'tensor.json')

    def test_any_frozen_file_change_is_detected(self):
        for idx, name in enumerate(('best.pt', 'validation.json', 'provenance.json', 'complete.json', 'history.jsonl')):
            with self.subTest(file=name):
                run = self.run_artifact('run' + str(idx));out = self.root / f'select{idx}.json'
                freeze_selection([run], out)
                with (run / name).open('ab') as stream:stream.write(b' ')
                with self.assertRaises(ValueError):verify_manifest(out)

    def test_manifest_selection_tampering_and_duplicate_runs_rejected(self):
        a = self.run_artifact('a');b = self.run_artifact('b', .3)
        manifest = freeze_selection([a, b], self.root / 'manifest.json')
        changed = copy.deepcopy(manifest);changed['selected']['primary'] = str(b)
        with self.assertRaisesRegex(ValueError, 'summary mismatch'):verify_manifest(changed)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            freeze_selection([a, a], self.root / 'duplicate.json')


if __name__ == '__main__':unittest.main()

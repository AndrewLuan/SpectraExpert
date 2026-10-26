# 功能概述：验证 train.py 的 warmstart、固定验证集、缓存身份和 epoch 边界恢复集成。
# 主要职责：TrainRestartTests 使用少量人工分子执行同步 CPU 训练并注入明确的保存后中断；
# 比较连续/恢复结果、优化器步数、history 和最佳模型，验证数据泄漏及完成状态拒绝。
# 代码库关系：调用 prepare.convert、train.main 与 training_state；不访问服务器或实测数据。
# 输入输出与副作用：仅写自动清理的临时目录、临时覆盖 argv/stdout；没有时间等待或异步线程。
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import train
from prepare import convert


def fixture_rows():
    result=[]
    for idx,(smiles,split) in enumerate([('CCO','train'),('CCC','train'),('CCN','val'),('CCF','test')]):
        row,error=convert(dict(nucleus='1H',representation='integral_peak_spectrum',
            smiles=smiles,value={'x':[1.,3.],'y':[3.,3.]},frequency=400,solvent='CDCl3',record_id=str(idx)))
        if error:raise AssertionError(error)
        row['split']=split
        result.append(row)
    return result


class TrainRestartTests(unittest.TestCase):
    def run_training(self,cache,out,*extra):
        arguments=['train.py','--cache',str(cache),'--out',str(out),'--device','cpu',
            '--width','8','--depth','1','--epochs','3','--batch-size','1',*extra]
        with patch.object(sys,'argv',arguments),contextlib.redirect_stdout(io.StringIO()):
            train.main()

    def assert_model_equal(self,left,right):
        self.assertEqual(left.keys(),right.keys())
        for name in left:self.assertTrue(torch.equal(left[name],right[name]),name)

    def test_main_resume_matches_continuous_and_appends_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache.pt';torch.save(fixture_rows(),cache)
            full=root/'full';resumed=root/'resumed'
            self.run_training(cache,full)
            original_save=train.save_training_state
            def interrupt_after_commit(*args,**kwargs):
                original_save(*args,**kwargs)
                if kwargs['epoch']==1:raise RuntimeError('injected completed epoch stop')
            with patch.object(train,'save_training_state',side_effect=interrupt_after_commit):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    self.run_training(cache,resumed)
            # 模拟 checkpoint 后多出的、甚至截断的未提交 history 尾部。
            with (resumed/'history.jsonl').open('a') as handle:handle.write('{partial')
            self.run_training(cache,resumed,'--resume')
            a=torch.load(full/'last.pt',weights_only=False);b=torch.load(resumed/'last.pt',weights_only=False)
            self.assert_model_equal(a['model'],b['model'])
            self.assertEqual(a['scheduler'],b['scheduler'])
            self.assertEqual(a['best'],b['best'])
            for key in a['optimizer']['state']:
                self.assert_model_equal(a['optimizer']['state'][key],b['optimizer']['state'][key])
            histories=[]
            for out in (full,resumed):
                rows=[json.loads(line) for line in (out/'history.jsonl').read_text().splitlines()]
                self.assertEqual([r['epoch'] for r in rows],[1,2,3])
                for row in rows:row.pop('elapsed_seconds')
                histories.append(rows)
            self.assertEqual(*histories)
            with self.assertRaisesRegex(ValueError,'completed run'):
                self.run_training(cache,resumed,'--resume')

    def test_warmstart_fresh_optimizer_and_registered_test_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache.pt';rows=fixture_rows();torch.save(rows,cache)
            original=root/'original';self.run_training(cache,original)
            expanded=root/'expanded.pt';torch.save(rows[:2],expanded)
            warm=root/'warm'
            self.run_training(expanded,warm,'--validation-cache',str(cache),'--init-from',str(original/'best.pt'))
            provenance=json.loads((warm/'provenance.json').read_text())
            self.assertEqual(provenance['init_from_sha256'],train.filehash(original/'best.pt'))
            self.assertEqual(json.loads((warm/'baseline.json').read_text())['val_n'],1)
            last=torch.load(warm/'last.pt',weights_only=False)
            self.assertTrue(all(float(state['step'])==6 for state in last['optimizer']['state'].values()))
            train.verify_test_cache(warm,cache)
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):train.verify_test_cache(warm,expanded)
            self.run_training(cache,warm,'--evaluate-test')
            self.assertEqual(json.loads((warm/'test_metrics.json').read_text())['n'],1)
            bad=root/'bad'
            with self.assertRaisesRegex(ValueError,'architecture mismatch'):
                self.run_training(cache,bad,'--init-from',str(original/'best.pt'),'--tie-methyl')
            self.assertFalse(bad.exists())

    def test_unlabelled_auxiliary_metrics_are_not_zero_accuracy(self):
        model=train.NMRForwardModel(feature_dim=18,hidden_dim=8,depth=1,
                                    aux_classes=len(train.MULTIPLICITY_CLASSES))
        loader=train.DataLoader(fixture_rows(),batch_size=4,collate_fn=train.collate)
        metrics,_=train.score(model,loader,'cpu')
        self.assertIsNone(metrics['matched_proton_multiplicity_accuracy'])
        self.assertIsNone(metrics['matched_known_J_MAE_Hz'])

    def test_fixed_validation_rejects_group_leakage_before_smoke(self):
        rows=fixture_rows();evaluation=copy.deepcopy(rows)
        evaluation[2]['group']=rows[0]['group']
        with self.assertRaisesRegex(ValueError,'group overlap'):
            train.resolve_splits(rows,evaluation,smoke=True)

    def test_visibility_warmstart_freeze_and_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache.pt';torch.save(fixture_rows(),cache)
            original=root/'original';self.run_training(cache,original)
            visible=root/'visible'
            self.run_training(cache,visible,'--init-from',str(original/'best.pt'),
                              '--visibility','--visibility-freeze-epochs','1')
            checkpoint=torch.load(visible/'last.pt',weights_only=False)
            self.assertTrue(any(key.startswith('visibility_head.') for key in checkpoint['model']))
            metrics=json.loads((visible/'validation.json').read_text())
            self.assertIn('reported_total_integral_MAE_H_all_records',metrics)
            self.assertIn('visibility_supervision_counts',metrics)
            self.run_training(cache,visible,'--evaluate-test')
            record=json.loads((visible/'test_predictions.jsonl').read_text().splitlines()[0])
            self.assertEqual(len(record['predicted_reported_weights']),record['structure_protons'])

    def test_committed_best_rejects_future_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            torch.save({'epoch':2},root/'best.pt')
            (root/'validation.json').write_text(json.dumps({'integral_weighted_W1_ppm':.2}))
            with self.assertRaisesRegex(ValueError,'outside committed'):
                train.verify_committed_best(root,1,.2)


if __name__=='__main__':unittest.main()

"""功能概述：CPU 合成测试 pool1024 工作流的科学边界与有界收尾。

cache_pair 构造同完整库/查询/粗排的两种 shortlist；测试仅允许 pool 改变，拒绝
验证补正例、候选/来源/原位移评分及训练曝光漂移。其余核验固定 CLI、对应初始化/
训练参数、两seed压力共同 gamma、UTC上限/90分钟准入、真实退出与必要产物，以及
只清理自己进程组并等待收敛。子进程和 GPU 查询均 mock，不启动真实作业、GPU、
网络、外部 API，也不读取真实 train/val/verify/test。
"""
import copy
import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from .h_global_pool_workflow import (Budget, COHORT_COUNTS, DeadlineReached, FIT_FIELDS,
    GPUS, HARD_DEADLINE, Workflow, aggregate, arms, extract_command, fit_command,
    read_reference, reference_paths, sampler_signature, terminate_owned, validate_cache_pair,
    validate_fit, main)
from .h_global_ranker_blend import GAMMAS


SMILES = [str(i) for i in range(114982)]


def cache_pair(cohort='original_val'):
    protocol = dict(cohort=cohort, feature_names=['score'], coarse_config=dict(name='atom_count_1.0'),
        candidate_count=114982, checkpoint_sha256='same-model', candidate_order_sha256='same-order',
        source_inputs_sha256={'same-input': 'same-SHA'}, feature_code_sha256='same-source',
        query_sha256='same-queries', variants={'observed': 4}, queries=4, appended_training_positives=0)
    old = dict(smiles=SMILES, manifest=dict(protocol, pool_size=512), rows=[])
    new = dict(smiles=SMILES, manifest=dict(protocol, pool_size=1024), rows=[])
    for order in range(4):
        target = 600 if order == 0 else order
        common = dict(query_order=order, variant='observed', source='synthetic', record_id=str(order),
            smiles=str(target), group=str(order), coarse_rank=target + 1, baseline_rank=target + 1,
            coarse_top10=list(range(10)), baseline_top10=list(range(10)), query_observation={},
            rerank_eligible=True)
        old_ids = list(range(512)) + ([target] if cohort == 'train' and target >= 512 else [])
        old['rows'].append(dict(common, candidate_indices=torch.tensor(old_ids),
            coarse_scores=torch.tensor(old_ids).float() / 1000,
            target_position=target if target < 512 else 512 if cohort == 'train' else -1,
            target_in_coarse_pool=target < 512))
        new['rows'].append(dict(common, candidate_indices=torch.arange(1024),
            coarse_scores=torch.arange(1024).float() / 1000,
            target_position=target, target_in_coarse_pool=True))
    if cohort == 'train':
        old['manifest']['appended_training_positives'] = 1
    return old, new


def reports():
    result = {}
    for seed in (193, 211):
        result[str(seed)] = {}
        for cohort, count in (('original_val', 320), ('tune', 1024)):
            metrics = {}
            for gamma in GAMMAS:
                # 两类别各偏好不同端点；总查询 microMRR 在 gamma=.5 最优。
                gain = .1 if gamma == .5 else 0.
                metrics[str(gamma)] = dict(overall=dict(queries=count, MRR=.5), stress={
                    'frequency_crop': dict(queries=count, MRR=.2 + gain + gamma * .02),
                    'peak_dropout': dict(queries=count // 2, MRR=.3 + gain - gamma * .02)})
            result[str(seed)][cohort] = dict(metrics=metrics)
    return result


class PoolWorkflowTests(unittest.TestCase):
    def test_fixed_arms_partition_and_extraction_scope(self):
        self.assertEqual(GPUS, (2, 3))
        self.assertEqual(len(arms()), 4)
        args = SimpleNamespace(root='/research', previous='/previous', base='/base')
        command = extract_command(args, 'physical_base_s193', 'tune', Path('/out'))
        self.assertEqual(command[command.index('--pool-size') + 1], '1024')
        self.assertEqual(command[command.index('--coarse') + 1], 'atom_count_1.0')
        self.assertIn('--augment', command)
        self.assertNotIn('--limit', command)
        smoke = extract_command(args, 'physical_base_s211', 'train', Path('/smoke'), 4)
        self.assertEqual(smoke[-2:], ['--limit', '4'])
        for sealed in ('verify', 'test'):
            with self.assertRaises(ValueError):
                extract_command(args, 'physical_base_s193', sealed, Path('/bad'))

    def test_deadline_aware_hard_cap_and_admission(self):
        maximum = Budget(HARD_DEADLINE, clock=lambda: 0).deadline
        self.assertGreater(Budget(clock=lambda: maximum - 6000).check(admission=True), 5400)
        with self.assertRaises(DeadlineReached):
            Budget(clock=lambda: maximum - 5000).check(admission=True)
        with self.assertRaises(DeadlineReached):
            Budget(clock=lambda: maximum - 10).check()
        for invalid in ('2026-10-20T19:51:00+00:00', '2026-10-20T19:49:00'):
            with self.assertRaises(ValueError):
                Budget(invalid)

    def test_validation_recovery_keeps_full_library_scores(self):
        old, new = cache_pair()
        result = validate_cache_pair(old, new, 'original_val', limit=4)
        self.assertEqual(result['observed'], dict(queries=4, recall512=.75, recall1024=1., recovered=1))
        self.assertEqual(new['rows'][0]['coarse_rank'], 601)

    def test_cache_candidate_mean_width_query_and_source_drift_rejected(self):
        for field in ('candidate_order_sha256', 'checkpoint_sha256', 'coarse_config',
                      'source_inputs_sha256', 'feature_code_sha256'):
            old, new = cache_pair()
            new['manifest'][field] = 'changed'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'protocol_changed'):
                validate_cache_pair(old, new, 'original_val', limit=4)
        for field, value in (('record_id', 'other'), ('coarse_rank', 2), ('query_observation', {'partial_peaks': True})):
            old, new = cache_pair()
            new['rows'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_cache_pair(old, new, 'original_val', limit=4)

    def test_validation_positive_injection_and_common_score_changes_rejected(self):
        old, new = cache_pair()
        new['rows'][0]['target_in_coarse_pool'] = False
        with self.assertRaisesRegex(ValueError, 'positive_injection'):
            validate_cache_pair(old, new, 'original_val', limit=4)
        old, new = cache_pair()
        new['rows'][0]['coarse_scores'][1] += .1
        with self.assertRaisesRegex(ValueError, 'common_candidate_score_changed'):
            validate_cache_pair(old, new, 'original_val', limit=4)

    def test_formal_training_signature_preserves_groups_and_variant_options(self):
        old, new = cache_pair('train')
        with patch.dict(COHORT_COUNTS, train=4):
            validate_cache_pair(old, new, 'train')
            new['rows'][0]['rerank_eligible'] = False
            with self.assertRaisesRegex(ValueError, 'exposure_changed'):
                validate_cache_pair(old, new, 'train')
        self.assertNotEqual(sampler_signature(old['rows'], True), sampler_signature(new['rows'], True))

    def test_shared_gamma_uses_both_seeds_cohorts_and_all_stress_queries(self):
        data = reports()
        result = aggregate(data)
        self.assertEqual(result['shared_stress_gamma'], .5)
        self.assertEqual(result['stress_queries'], 4032)
        self.assertTrue(all(all(v == .5 for v in row.values()) for row in result['normal1344_MRR'].values()))
        del data['193']
        with self.assertRaisesRegex(ValueError, 'two_seed'):
            aggregate(data)

    def test_reference_and_fit_match_exact_parameters_initialization_and_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = arms()[0]
            checkpoint, initial, summary = reference_paths(root, arm)
            checkpoint.parent.mkdir(parents=True)
            settings = dict(features='all', architecture='mlp', shared=False, augment=False,
                            seed=193, anchor=.1, epochs=64, samples=2048)
            state = {'network.weight': torch.tensor([1.])}
            report = dict(group_exposure_sha256='same', training_groups=3, internal_groups=1,
                          training_rows=3, internal_rows=1)
            torch.save(dict(args=settings, pool_size=512), checkpoint)
            torch.save(dict(state_dict=state), initial)
            summary.write_text(json.dumps(report))
            (checkpoint.parent / 'EXIT_CODE').write_text('0\n')
            _, _, actual = read_reference(root, arm)
            self.assertEqual(actual, settings)
            output = root / 'new'
            output.mkdir()
            torch.save(dict(args=settings, pool_size=1024, epoch=8, alpha=.5), output / 'best.pt')
            torch.save(dict(state_dict=state), output / 'initial.pt')
            (output / 'summary.json').write_text(json.dumps(report))
            validate_fit(root, output, arm, settings)
            command = fit_command(Path('/train'), output, settings)
            for key in ('epochs', 'samples', 'seed', 'anchor'):
                self.assertEqual(command[command.index('--' + key) + 1], str(settings[key]))
            torch.save(dict(state_dict={'network.weight': torch.tensor([2.])}), output / 'initial.pt')
            with self.assertRaisesRegex(ValueError, 'initial_model_changed'):
                validate_fit(root, output, arm, settings)

    def test_cleanup_only_own_groups_waits_and_escalates(self):
        first, second = Mock(), Mock()
        first.pid, second.pid = 101, 202
        first.poll.return_value = None
        second.poll.return_value = 0
        first.wait.side_effect = [subprocess.TimeoutExpired('own', 5), 0]
        second.wait.return_value = 0
        with patch('os.killpg') as kill:
            terminate_owned([('own1', first, Path('/one')), ('own2', second, Path('/two'))])
        self.assertEqual(kill.call_args_list[0].args, (101, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (101, signal.SIGKILL))
        self.assertEqual(first.wait.call_count, 2)
        self.assertEqual(second.wait.call_count, 2)

    def test_runner_requires_real_zero_exit_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'child'
            folder.mkdir()
            (folder / 'EXIT_CODE').write_text('0\n')
            (folder / 'summary.json').write_text('{}')
            (folder / 'features.pt').write_text('synthetic')
            workflow = Workflow.__new__(Workflow)
            workflow.out, workflow.active = root, []
            workflow.budget = Mock()
            workflow.budget.check.return_value = 100
            workflow.guard = Mock()
            workflow.screen = Mock()
            workflow.status = Mock()
            workflow.resources = Mock()
            child = Mock()
            child.pid = 123
            child.poll.return_value = 0
            child.wait.return_value = 0
            job = ('child', ['python', 'synthetic'], folder, ('features.pt',))
            with patch('subprocess.Popen', return_value=child) as popen:
                workflow.run([job], 'synthetic', gpu=True)
            self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            (folder / 'features.pt').unlink()
            with patch('subprocess.Popen', return_value=child), self.assertRaisesRegex(ValueError, 'missing_output'):
                workflow.run([job], 'missing')
            (folder / 'features.pt').write_text('synthetic')
            (folder / 'EXIT_CODE').write_text('1\n')
            with patch('subprocess.Popen', return_value=child), self.assertRaisesRegex(ValueError, 'artifact_exit_not_zero'):
                workflow.run([job], 'contradictory')

    def test_parent_failure_is_diagnosed_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (('forward', '1'), ('observation_calibration', '0')):
                folder = root / name
                folder.mkdir()
                (folder / 'EXIT_CODE').write_text(value)
            workflow = Workflow.__new__(Workflow)
            workflow.args = SimpleNamespace(root=str(root))
            workflow.guard, workflow.status = Mock(), Mock()
            workflow.resources = Mock(side_effect=AssertionError('GPU forbidden'))
            with self.assertRaisesRegex(RuntimeError, 'parent_failed_explicit_diagnosis'):
                workflow.dependencies()
            workflow.resources.assert_not_called()

    def test_expired_main_preserves_explicit_pending_without_any_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ['pool', '--root', str(root), '--base', '/base', '--previous', '/previous', '--gate', '/gate']
            fake = Mock()
            fake.check.side_effect = DeadlineReached('expired')
            with patch('sys.argv', argv), patch('nmr_transfer.h_global_pool_workflow.Budget', return_value=fake), patch('subprocess.Popen') as popen:
                self.assertEqual(main(), 124)
            popen.assert_not_called()
            self.assertEqual((root / 'pool1024/EXIT_CODE').read_text(), '124\n')
            partial = json.loads((root / 'pool1024/partial_summary.json').read_text())
            self.assertEqual(partial['state'], 'pending')
            self.assertFalse(partial['comparison_complete'])


if __name__ == '__main__':
    unittest.main()

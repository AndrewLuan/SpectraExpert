"""功能概述：独立审查 C5/C10 数学、资格、冻结与真实续训入口的合成测试。

直接重算损失和梯度；复用作者的临时数据 fixture 仅提供输入文件和 mock I/O，
不复用作者断言或验收数学。入口仍遍历 64 epoch，测试内仅缩小 draws。
无真实谱/张量读取，无远程访问，无科学训练或 test 读取。
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import cap_ranker as cap
import train_cap_ranker as train
import test_cap_ranker as fixture


class IndependentCapTests(unittest.TestCase):
    def test_native_C5_and_C10_analytic_all_heads_at_large_logits(self):
        native, c5, scale = fixture.native_models(5)
        c10 = cap.CapRanker([.2] * 9, native.fixed_x_center, native.fixed_o_center, 10, fixture.RT)
        c10.load_state_dict(native.state_dict())
        for value in (-50., -10., 0., 10., 50.):
            with torch.no_grad():
                for model in (native, c5, c10):
                    model.network[0].weight.zero_()
                    model.network[0].bias.fill_(1.)
                    hidden = torch.nn.functional.silu(torch.tensor(1.))
                    for head, logit in enumerate([value, value/2, -value/2, -value]):
                        model.network[-1].weight[head].fill_(logit / (32 * hidden))
            row = fixture.natural(fixture.row())
            for kind in range(4):
                row['kind'] = kind
                a = c5.row_delta(row, scale, 'cpu')
                b = c10.row_delta(row, scale, 'cpu')
                self.assertTrue(torch.equal(a, native.row_delta(row, scale, 'cpu')))
                logits = c10.network(torch.zeros(512, 36))[:, kind]
                self.assertTrue(torch.equal(b, 10 * torch.tanh(logits/10)))
        self.assertEqual(cap.tensor_hash(c5.fixed_x_center), cap.tensor_hash(c10.fixed_x_center))

    def test_strict_mask_and_loss_gradient_match_independent_formula(self):
        row = fixture.with_mask(fixture.row(target=900))
        q = fixture.qualification()
        actual = cap.strict_negative_mask(row, q, fixture.RT)
        indices = row['candidate_indices'].tolist()
        expected = torch.tensor([q['groups'][i] == row['group'] or q['groups'][i] in q['heldout'] or
            fixture.RT.internal_group(q['groups'][i]) for i in indices])
        expected[row['target_position']] = True
        self.assertTrue(torch.equal(actual, expected))
        a = torch.linspace(-2, 2, 513, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        scale = torch.ones(33) * 2
        loss_a = cap.listwise_loss(row, a, scale, actual)
        keep = [i for i, value in enumerate(expected.tolist()) if not value] + [512]
        costs = row['coarse_scores'] / 2 + b
        loss_b = torch.logsumexp(-costs[keep] * 2, 0) + 2 * costs[512] + .1 * (b[keep] ** 2).mean()
        loss_a.backward(); loss_b.backward()
        self.assertTrue(torch.allclose(loss_a, loss_b, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-7, rtol=0))
        self.assertEqual(float(a.grad[1]), 0)
        self.assertEqual(float(a.grad[2]), 0)
        self.assertNotEqual(float(a.grad[512]), 0)

    def test_all_internal_natural_negatives_fail_even_with_appended_target(self):
        row = fixture.row(target=900)
        q = fixture.qualification()
        for i in row['candidate_indices'][:512].tolist():
            q['groups'][i] = fixture.group_name(True)
        q['groups'][900] = row['group']
        q['internal'] = torch.tensor([fixture.RT.internal_group(g) for g in q['groups']])
        fixture.with_mask(row, q)
        with self.assertRaisesRegex(ValueError, 'no_eligible'):
            cap.strict_negative_mask(row, q, fixture.RT)

    def test_common_candidate_permutation_is_exact_same_cost(self):
        _, model, scale = fixture.native_models(10)
        s = fixture.natural(fixture.row(0)); d = fixture.natural(fixture.row(1))
        reverse = torch.arange(511, -1, -1)
        for key in ('features', 'coarse_scores', 'predicted_ppm', 'tags', 'H', 'K'):
            d[key] = d[key][reverse]
        models = [(model, scale, {'alpha': 1, 'delta_cap': 10})] * 2
        left, _ = cap.score_pair(models, s, 'cpu')
        right, _ = cap.score_pair(models, d, 'cpu')
        self.assertTrue(torch.equal(left[:256], right[reverse][:256]))
        poisoned = dict(s, target_position=512)
        with self.assertRaisesRegex(ValueError, 'identity'):
            cap.score_pair(models, poisoned, 'cpu')

    def test_production_budget_and_final_policy_cannot_be_shortened(self):
        protocol = dict(schema='cap10_controlled_protocol_20261024', training_allowed=True,
            test_allowed=False, seeds=[193,211], caps=[5,10], budget=dict(cap.BUDGET),
            final_epoch=64, alpha=1., epoch_offset=0)
        args = type('Args', (), {'seed': 193, 'cap': 10, 'role': 'aug'})()
        for key, value in [('epochs', 63), ('samples', 1024), ('batch_size', 64), ('draws_per_fit', 1),
                           ('optimizer_steps', True), ('weight_decay', .0001)]:
            bad = copy.deepcopy(protocol); bad['budget'][key] = value
            with self.assertRaises(ValueError):
                train.validate_protocol(bad, args)

    def test_aug_both_caps_actual64_entry_and_loaded_outputs(self):
        tiny = dict(cap.BUDGET, samples=2, batch_size=2, draws_per_fit=128, optimizer_steps=64)
        summaries = []
        with patch.object(cap, 'BUDGET', tiny):
            for c in (5, 10):
                with tempfile.TemporaryDirectory() as directory:
                    f = fixture.EntryFixture(directory, role='aug', cap_value=c)
                    original_hashes = cap.tensor_hashes(f.native, f.scale)
                    summary = f.run(); summaries.append(summary)
                    initial = torch.load(f.out / 'initial.pt', weights_only=False, map_location='cpu')
                    final = torch.load(f.out / 'final.pt', weights_only=False, map_location='cpu')
                    history = [json.loads(line) for line in (f.out / 'history.jsonl').read_text().splitlines()]
                    self.assertEqual([r['epoch'] for r in history], list(range(1,65)))
                    self.assertEqual((final['epoch'], final['alpha'], final['draws'], final['optimizer_steps']), (64,1.,128,64))
                    self.assertEqual(initial['initial_tensor_sha256'], original_hashes)
                    self.assertEqual({k:cap.tensor_hash(v) for k,v in initial['state_dict'].items()}, {k:v for k,v in original_hashes.items() if k != '__scale__'})
                    self.assertEqual(cap.tensor_hash(initial['scale']), original_hashes['__scale__'])
                    for key, value in final['state_dict'].items():
                        if not key.startswith('network.'):
                            self.assertTrue(torch.equal(value, initial['state_dict'][key]))
                    model, scale, loaded = cap.load_checkpoint(f.out / 'final.pt', 'cpu', expected_cap=c, runtime=fixture.RT)
                    natural = fixture.natural(fixture.row(target=900))
                    self.assertEqual(len(model.row_delta(natural, scale, 'cpu')), 512)
                    self.assertEqual(loaded['delta_cap'], c)
                    self.assertFalse((f.out / 'best.pt').exists())
                    with self.assertRaisesRegex(ValueError, 'expected_cap'):
                        cap.load_checkpoint(f.out / 'final.pt', 'cpu', expected_cap=15-c, runtime=fixture.RT)
        for key in ('initial_tensor_sha256', 'frozen_tensor_sha256', 'exposure_sha256',
                    'pool_exposure_sha256', 'negative_exposure_sha256', 'strict_negative_qualification_sha256'):
            self.assertEqual(summaries[0][key], summaries[1][key], key)
        for left, right in zip(summaries[0]['epoch_records'], summaries[1]['epoch_records']):
            for key in ('draws','optimizer_steps','exposure_sha256','pool_exposure_sha256','negative_exposure_sha256'):
                self.assertEqual(left[key], right[key], key)


if __name__ == '__main__':
    unittest.main()

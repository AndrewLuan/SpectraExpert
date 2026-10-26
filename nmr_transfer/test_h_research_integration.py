"""功能概述：独立核验新 R/O/F 接口与父级检索训练、完整排序的集成边界。

ResearchIntegrationTests 用 CPU 合成查询检查 ranking_loss 的隔离候选梯度、补入训练
正例的评估隔离、O 关闭/开启的相同 MLP 容量及 zero-init 后的实际观测模型梯度。
validate_model_pair 用临时源指纹检验 noaug/aug 顺序、seed、训练预算及科学输入绑定。
bank 测试替换 I/O 和模型，在第一次实际候选调用处检查旧 embedding 确实已从传入
batch 清除，防止调用 clone helper 却丢弃返回值。所有 mock 与临时文件只用于本测试，
不改父模块或冻结科学源码，不读真实训练/验证/测试谱，不联网、不启用 GPU。
"""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from .h_global_features import FEATURE_NAMES
from .h_research_contract import input_fingerprints, pool_background_rank, rank_shortlist
from .h_research_observation import prepare_observation
from .h_research_ranker import ResearchRanker, ranking_loss, validate_model_pair
from .h_research_recall import combine_shortlists, directional_kernel_scores


class _FirstPredictionReached(Exception):
    pass


class _FakeBatch:
    def __init__(self):
        self.num_graphs = 1
        self.frozen_embedding = torch.tensor([123.])
        self.final_context_embedding = torch.tensor([456.])
        self.unconditioned_context_embedding = torch.tensor([789.])

    def clone(self):
        return copy.deepcopy(self)


class ResearchIntegrationTests(unittest.TestCase):
    def synthetic_row(self):
        query = dict(source='synthetic', observed_xs=[1., 2.], observed_weights=[2., 1.],
                     measure='observed_assigned_atom_equal_partial_possible',
                     observation=dict(windows=[]))
        return dict(features=torch.zeros(4, len(FEATURE_NAMES)),
                    coarse_scores=torch.tensor([0., .1, .2, .3]),
                    candidate_indices=torch.arange(4), evaluation_pool_size=4,
                    predicted_ppm=torch.tensor([[1., 2.], [2., 3.], [4., 5.], [7., 8.]]),
                    tags=torch.tensor([[0, 1], [0, 1], [2, 3], [4, 5]]),
                    H=torch.tensor([4., 5., 6., 7.]), K=torch.tensor([2., 3., 4., 5.]),
                    observation_input=prepare_observation(query), kind=2,
                    target_in_coarse_pool=True, target_position=0,
                    negative_exclusion_mask=torch.tensor([True, False, False, False]))

    def test_heldout_candidate_gets_zero_ranking_and_anchor_gradient(self):
        row = dict(target_position=0, coarse_scores=torch.tensor([0., -100., 1., 2.]),
                   negative_exclusion_mask=torch.tensor([True, True, False, False]))
        delta = torch.tensor([.3, -20., .5, .7], requires_grad=True)
        ranking_loss(row, delta, torch.ones(len(FEATURE_NAMES)), anchor=.2).backward()
        self.assertEqual(float(delta.grad[1]), 0.)
        self.assertNotEqual(float(delta.grad[0]), 0.)
        self.assertGreater(float(delta.grad[2:].abs().sum()), 0.)

    def test_appended_training_positive_has_loss_gradient_but_no_eval_shortcut(self):
        row = dict(target_position=3, target_in_coarse_pool=False,
                   coarse_scores=torch.tensor([1., 2., 3., 99.]),
                   negative_exclusion_mask=torch.tensor([False, False, False, True]),
                   candidate_indices=torch.tensor([0, 4, 5, 3]), evaluation_pool_size=3,
                   outside_pool_rank=5)
        delta = torch.tensor([0., 0., 0., -999.], requires_grad=True)
        # 用有限正例值检查训练梯度，再以极端优分验证评估剔除。
        finite = torch.zeros(4, requires_grad=True)
        ranking_loss(row, finite, torch.ones(len(FEATURE_NAMES)), anchor=.1).backward()
        self.assertGreater(float(finite.grad[3]), 0.)
        rank, top = rank_shortlist(row, row['coarse_scores'] + delta)
        self.assertEqual(rank, 5)
        self.assertNotIn(3, top)

    def test_observation_off_and_on_keep_same_mlp_capacity_and_initial_head(self):
        torch.manual_seed(317); off = ResearchRanker(False)
        torch.manual_seed(317); on = ResearchRanker(True)
        self.assertEqual(sum(p.numel() for p in off.network.parameters()),
                         sum(p.numel() for p in on.network.parameters()))
        for name, value in off.network.state_dict().items():
            self.assertTrue(torch.equal(value, on.network.state_dict()[name]), name)
        self.assertTrue(all(not p.requires_grad for p in off.observation_model.parameters()))
        row = self.synthetic_row(); scale = torch.ones(len(FEATURE_NAMES))
        self.assertTrue(torch.equal(off.row_delta(row, scale, 'cpu'), on.row_delta(row, scale, 'cpu')))

    def test_joint_observation_receives_nonzero_gradient_on_second_step(self):
        torch.manual_seed(317)
        model = ResearchRanker(True)
        row = self.synthetic_row(); scale = torch.ones(len(FEATURE_NAMES))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
        initial = {k: v.detach().clone() for k, v in model.observation_model.named_parameters()}
        second_norm = None
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            delta = model.row_delta(row, scale, 'cpu')
            loss = ranking_loss(row, delta, scale, anchor=.1) + .01 * model.observation_model.regularization()
            loss.backward()
            gradients = [p.grad for p in model.observation_model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            if step == 1:
                second_norm = sum(float(g.abs().sum()) for g in gradients)
            optimizer.step()
        self.assertGreater(second_norm, 0.)
        self.assertTrue(any(not torch.equal(initial[k], p.detach())
                            for k, p in model.observation_model.named_parameters()))

    def pair_fixture(self, source):
        science = {'/synthetic/widths.json': 'width', '/synthetic/bank.pt': 'bank'}
        protocol = dict(schema='nmr_research_20261022', recall='dual', pool_size=512,
                        candidate_order_sha256='candidate_order', checkpoint_sha256='forward')
        data = dict(manifest=dict(protocol, source_inputs_sha256=science))
        args = dict(seed=211, epochs=8, samples=16, batch_size=2, anchor=.1,
                    observation_regularization=.01)
        def saved(augment):
            return dict(args=dict(args, augment=augment), cache_protocol=copy.deepcopy(protocol),
                        source_inputs_sha256=copy.deepcopy(science),
                        ranker_source_sha256=input_fingerprints([source]),
                        train_sha256='shared_training_cache', observation=True)
        return [(None, None, saved(False)), (None, None, saved(True))], data

    def test_model_pair_roles_seed_budget_and_shared_science_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.py'; source.write_text('constant')
            models, data = self.pair_fixture(source)
            validate_model_pair(models, data, expected_seed=211)
            cases = [
                ('role', lambda pair, cache: pair.reverse()),
                ('seed', lambda pair, cache: pair[1][2]['args'].update(seed=193)),
                ('training', lambda pair, cache: pair[1][2].update(observation=False)),
                ('training', lambda pair, cache: pair[1][2].update(train_sha256='different')),
                ('budget', lambda pair, cache: pair[1][2]['args'].update(samples=32)),
                ('shared_science', lambda pair, cache: cache['manifest']['source_inputs_sha256'].update({'/synthetic/widths.json': 'changed'})),
                ('cache_identity', lambda pair, cache: cache['manifest'].update(checkpoint_sha256='other_forward')),
            ]
            for message, mutate in cases:
                with self.subTest(message=message):
                    pair, cache = copy.deepcopy(models), copy.deepcopy(data)
                    mutate(pair, cache)
                    with self.assertRaisesRegex(ValueError, message):
                        validate_model_pair(pair, cache, expected_seed=211)
            source.write_text('changed_after_fit')
            with self.assertRaisesRegex(ValueError, 'source_changed'):
                validate_model_pair(models, data, expected_seed=211)

    def test_bank_passes_stripped_batch_to_actual_first_prediction(self):
        from . import h_research_bank as bank_module
        class FakeModel:
            def eval(self):
                return self
            def __call__(self, batch, details):
                self_outer.assertFalse(hasattr(batch, 'frozen_embedding'))
                self_outer.assertFalse(hasattr(batch, 'final_context_embedding'))
                self_outer.assertFalse(hasattr(batch, 'unconditioned_context_embedding'))
                raise _FirstPredictionReached()
        self_outer = self
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root/'checkpoint.pt'; checkpoint.write_bytes(b'checkpoint')
            args = SimpleNamespace(device='cpu', reference_bank='mock_reference', reference_extension=None,
                                   cohorts=str(root/'cohorts'), prepared=str(root/'prepared'),
                                   checkpoint=str(checkpoint), research_forward=True)
            key = (0.,) * (4 + bank_module.CONDITION_DIM)
            reference = dict(checkpoint_sha256='old', predictions={key: torch.zeros(1, 128)},
                             fixed_predictions={}, candidate_quantile_classes={})
            payload = dict(cohorts={'train': []}, metadata={}, key_mode='control')
            prepared = dict(graphs={})
            def fake_load(path, **kwargs):
                return payload if Path(path).name == 'cohorts.pt' else prepared
            with mock.patch.object(bank_module, 'load_bank', return_value=reference), \
                    mock.patch.object(bank_module.torch, 'load', side_effect=fake_load), \
                    mock.patch.object(bank_module, 'candidate_batches', return_value=[_FakeBatch()]), \
                    mock.patch('nmr_transfer.h_research_forward.load_checkpoint_model', return_value=FakeModel()):
                with self.assertRaises(_FirstPredictionReached):
                    bank_module.build_bank(args, root, lambda *a, **k: None)

    def test_dual_channel_is_identity_blind_and_never_injects_unrecalled_truth(self):
        kernel = dict(logp=torch.full((8, 4), .25).log(),
                      cdf=torch.arange(5).float()[None].expand(8, -1)/4, low=0., step=1.)
        coarse = torch.arange(8).float()
        scores = []
        for target in (0, 7):
            observation = dict(xs=[1.], weights=[1.], windows=[], source=str(target), true_index=target)
            positive = directional_kernel_scores(kernel, observation)
            scores.append(combine_shortlists(coarse, positive, 4, 2, expected_candidates=None))
        self.assertTrue(torch.equal(scores[0], scores[1]))
        self.assertNotIn(7, scores[1].tolist())
        self.assertEqual(pool_background_rank(coarse, scores[1], target=7), 8)


if __name__ == '__main__':
    unittest.main()

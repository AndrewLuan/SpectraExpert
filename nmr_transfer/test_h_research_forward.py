"""功能概述：验证新F确实训练末层GINE，且固定组输出仍遵守旧算法契约。

测试只用合成registry、随机小checkpoint和CPU图。覆盖physical/local同参数初始化、
GINE梯度/冻结前层、旧embedding拒绝、同Batch布局隔离、H质量、未知精确回退及
新checkpoint重载。它不把既有Preserved固定组视作新发现，不读真实谱或调用API。
"""
from pathlib import Path
import tempfile
import unittest

import torch

from .h_research_forward import (ResearchForwardSpecGNN, configure_forward_training,
    load_checkpoint_model, strip_embedding_caches, quantile_sigmas, EMBEDDING_CACHES)
from .h_preserved_group_model import PreservedGroupSpecGNN
from .h_equivalence_model import MAX_GROUPS
from .emission_model import weighted_quantiles
from .prepare import sha
from . import test_h_equivalence_model as fixtures


class ResearchForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def checkpoint(self, folder):
        registry = Path(folder) / 'registry.pt'
        torch.save({'CC': dict(groups=[dict(parents={0: 3, 1: 3}, mass=6)]),
            'CCO': dict(groups=[dict(parents={0: 3}, mass=3), dict(parents={1: 2}, mass=2),
                               dict(parents={2: 1}, mass=1)])}, registry)
        cfg = dict(model='preserved_group_atom', nucleus='1H', width=16, layers=2,
            condition_mode='local', response_mode='solvent', context_depth=0, message_mode='late',
            equivalence_registry_path=str(registry), equivalence_registry_sha256=sha(registry),
            equivalence_mode='physical', amide_gap_scale=.132)
        torch.manual_seed(11)
        old = PreservedGroupSpecGNN('1H', 16, 2, 'local', 'solvent', 0, 'late', str(registry), sha(registry), 'physical')
        old.amide_gap_scale = .132
        path = Path(folder) / 'initial.pt'
        torch.save(dict(args=cfg, state_dict=old.state_dict()), path)
        return path

    def batch(self, names=('CC', 'CCO', 'CCC')):
        return fixtures.EquivalenceTests().batch(list(names))

    def test_pair_has_identical_weights_capacity_and_different_structural_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.checkpoint(folder)
            a = load_checkpoint_model(path, equivalence_mode='physical', trainable=True)
            b = load_checkpoint_model(path, equivalence_mode='local', trainable=True)
            self.assertEqual(a.state_dict().keys(), b.state_dict().keys())
            self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
            self.assertEqual(sum(p.numel() for p in a.parameters() if p.requires_grad),
                             sum(p.numel() for p in b.parameters() if p.requires_grad))
            batch = self.batch()
            physical = a(batch, details=True)
            local = b(batch, details=True)  # 故意共享已被physical附加结构memo的Batch。
            self.assertEqual(tuple(physical['group_centers'].shape), (3, MAX_GROUPS))
            self.assertEqual(int(physical['group_mask'][0].sum()), 1)
            self.assertEqual(int(local['group_mask'][0].sum()), 2)
            self.assertTrue(torch.equal(physical['group_Hmass'].sum(1), torch.tensor([6., 6., 0.])))
            self.assertFalse(physical['group_mask'][2].any())
            self.assertEqual(float(physical['group_trust'][2].sum()), 0.)
            self.assertTrue(torch.equal(physical['total_Hmass'], torch.tensor([6., 6., 8.])))
            self.assertEqual(float(physical['ungrouped_Hmass'][2]), 8.)
            self.assertEqual(tuple(physical['quantiles'].shape), (3, 128))
            self.assertEqual(tuple(physical['quantile_sigma_ppm'].shape), (3, 128))
            self.assertTrue(torch.equal(physical['quantile_sigma_ppm'], 10 * physical['quantile_sigma']))
            self.assertTrue(torch.equal(physical['quantiles'], weighted_quantiles(
                physical['values'], physical['weights'], physical['groups'], 3)))

    def test_live_GINE_is_reused_only_inside_one_forward_and_cleared_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            model = load_checkpoint_model(self.checkpoint(folder), trainable=True)
            calls = []
            handle = model.atom_bond_block_list[-1].register_forward_hook(lambda *_: calls.append(1))
            try:
                model(self.batch(), details=True)
                self.assertEqual(len(calls), 1)
                self.assertIsNone(model._research_live_embedding)
                model(self.batch(), details=True)
                self.assertEqual(len(calls), 2)
                bad = self.batch()
                bad.frozen_embedding = torch.zeros(len(bad.x), 16)
                with self.assertRaisesRegex(ValueError, 'cached_embedding_forbidden'):
                    model(bad)
                self.assertIsNone(model._research_active_batch)
                self.assertIsNone(model._research_live_embedding)
            finally:
                handle.remove()

    def test_nonconstant_sigma_follows_exact_mass_quantile_index(self):
        details = dict(values=torch.tensor([2., .5, .5, 1.]), weights=torch.tensor([1., 1., 2., 4.]),
            groups=torch.tensor([0, 0, 0, 1]), component_sigma=torch.tensor([.1, .02, .03, .04]))
        details['quantiles'] = weighted_quantiles(details['values'], details['weights'], details['groups'], 2)
        actual = quantile_sigmas(details, 2)
        expected = torch.tensor([.02] * 32 + [.03] * 64 + [.1] * 32)
        self.assertTrue(torch.equal(actual[0], expected))
        self.assertTrue(torch.equal(actual[1], torch.full((128,), .04)))

    def test_real_last_GINE_has_gradient_and_earlier_layers_stay_frozen(self):
        with tempfile.TemporaryDirectory() as folder:
            model = load_checkpoint_model(self.checkpoint(folder), trainable=True)
            prefixes, names = configure_forward_training(model)
            self.assertTrue(all(name.startswith(prefixes) for name in names))
            self.assertIn('atom_bond_block_list.1.conv.nn.0.weight', names)
            self.assertIn('bond_emb_list.1.0.weight', names)
            details = model(self.batch(), details=True)
            (details['quantiles'].square().mean() + details['component_sigma'].sum()).backward()
            parameter = dict(model.named_parameters())['atom_bond_block_list.1.conv.nn.0.weight']
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertIsNone(dict(model.named_parameters())['atom_bond_block_list.0.conv.nn.0.weight'].grad)
            self.assertGreater(float(model.research_sigma_head[-1].bias.grad.abs().sum()), 0.)

    def test_cached_embedding_is_rejected_and_explicit_strip_does_not_change_source(self):
        with tempfile.TemporaryDirectory() as folder:
            model = load_checkpoint_model(self.checkpoint(folder))
            for name in EMBEDDING_CACHES:
                batch = self.batch()
                setattr(batch, name, torch.ones(len(batch.x), 16))
                with self.assertRaisesRegex(ValueError, 'cached_embedding_forbidden'):
                    model(batch)
                clean = strip_embedding_caches(batch)
                self.assertTrue(hasattr(batch, name))
                self.assertFalse(hasattr(clean, name))
                self.assertTrue(torch.equal(clean.x, batch.x))
                self.assertTrue(torch.isfinite(model(clean)).all())

    def test_unknown_fallback_and_new_checkpoint_reload_are_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            model = load_checkpoint_model(self.checkpoint(folder))
            original = PreservedGroupSpecGNN.forward(model, self.batch(('CCC',)), details=True)
            output = model(self.batch(('CCC',)), details=True)
            for name in ('quantiles', 'values', 'weights', 'groups', 'owners', 'exchangeable'):
                self.assertTrue(torch.equal(original[name], output[name]))
            self.assertTrue(torch.all(output['component_sigma'] > model.sigma_min))
            self.assertTrue(torch.all(output['component_group'] == -1))
            path = Path(folder) / 'research.pt'
            torch.save(dict(args=model.research_config, state_dict=model.state_dict()), path)
            other = load_checkpoint_model(path)
            again = other(self.batch(), details=True)
            before = model(self.batch(), details=True)
            self.assertTrue(torch.equal(before['quantiles'], again['quantiles']))
            self.assertTrue(torch.equal(before['component_sigma'], again['component_sigma']))


if __name__ == '__main__':
    unittest.main()

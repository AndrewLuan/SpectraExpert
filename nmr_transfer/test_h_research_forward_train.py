"""功能概述：用CPU合成数据验证新F的实际监督、可信归属和曝光边界。

覆盖Gaussian sigma梯度/峰高口径、窗口并集及缺峰不恢复、逐母节点可信mask在无序
合并前生效、三源/两域确定性等曝光、显式保留结构排除、一次真实优化步骤和冻结体。
所有checkpoint/谱/条件均在临时目录构造，不访问真实数据、GPU或外部API。
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from .data import graph, identity, quantiles, physical_features
from .h_conditions import local_environment, encode_conditions
from .h_labels import make_key
from .h_research_forward import load_checkpoint_model, configure_forward_training
from .h_research_forward_train import (ResearchSpectra, THEORY_SOURCES, gaussian_observed_nll,
    forward_objective, frozen_core_check, make_exposure, select_training_rows, exposure_key)
from .train import collate
from . import test_h_research_forward as model_fixtures
from . import h_research_forward_train as trainer
from .prepare import sha


class ResearchForwardTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def row(self, smiles='CCO', source='HMDB', domain='experimental', rid='row'):
        smiles, group, split = identity(smiles)
        return dict(smiles=smiles, group=group, split=split, source=source, domain=domain, record_id=rid,
            nucleus='1H', q=quantiles([1., 2., 7.]), cond=np.zeros(4, np.float32),
            physics=physical_features(smiles), measure='peak_equal_intensity_unused',
            observation=dict(kind='unknown', windows=[]))

    def dataset(self, row, annotations=None, meta=None):
        item = graph(row['smiles'])
        item.local_environment = torch.tensor(local_environment(row['smiles']))
        item.frozen_embedding = torch.zeros(len(item.x), 16)
        graphs = {row['smiles']: item}
        metadata = {make_key(row): meta or encode_conditions(dict(solvent='D2O'))}
        return ResearchSpectra([row], graphs, '1H', metadata, annotations=annotations or {},
            grouped=True, theory_orbits=True), graphs

    def likelihood_batch(self, windows=None, line=False):
        bounds = torch.zeros(1, 8, 2)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        for index, interval in enumerate(windows or []):
            bounds[0, index] = torch.tensor(interval)
            mask[0, index] = True
        return SimpleNamespace(num_graphs=1, q=torch.full((1, 128), .12), windows=bounds,
            window_mask=mask, research_line_measure=torch.tensor([line]))

    def likelihood_details(self, second=.7, masses=(3., 1.)):
        return dict(values=torch.tensor([.1, second], requires_grad=True),
            weights=torch.tensor(masses), groups=torch.zeros(2, dtype=torch.long),
            component_sigma=torch.tensor([.025, .04], requires_grad=True))

    def test_sigma_has_real_likelihood_gradient_and_line_measure_does_not_use_H_mass(self):
        first = self.likelihood_details()
        loss = gaussian_observed_nll(first, self.likelihood_batch(line=True)).sum()
        loss.backward()
        self.assertGreater(float(first['component_sigma'].grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(first['component_sigma'].grad).all())
        altered = self.likelihood_details(masses=(30., 1.))
        self.assertTrue(torch.allclose(gaussian_observed_nll(first, self.likelihood_batch(line=True)),
                                      gaussian_observed_nll(altered, self.likelihood_batch(line=True))))
        self.assertFalse(torch.allclose(gaussian_observed_nll(first, self.likelihood_batch()),
                                       gaussian_observed_nll(altered, self.likelihood_batch())))

    def test_window_union_and_unreported_outside_peak_are_conditioned_without_completion(self):
        overlap = self.likelihood_batch([[0., .2], [.1, .3]])
        united = self.likelihood_batch([[0., .3]])
        details = self.likelihood_details()
        self.assertTrue(torch.allclose(gaussian_observed_nll(details, overlap),
                                      gaussian_observed_nll(details, united), atol=1e-6))
        self.assertTrue(torch.allclose(gaussian_observed_nll(details, united),
                                      gaussian_observed_nll(self.likelihood_details(second=1.2), united), atol=1e-6))

    def test_partial_HMDB_does_not_restore_metadata_peaks_and_trusted_groups_survive(self):
        row = self.row()
        row.update(q=quantiles([1.]), observation=dict(kind='partial_window', windows=[[0., 2.]], partial_peaks=True))
        annotation = dict(groups={0: [1., 1., 1.], 1: [2., 3.], 2: [7.]}, full_atom_labels=True,
            trusted_site_mask={0: [True, True, True], 1: [True, False], 2: False})
        labels = {make_key(row): copy.deepcopy(annotation)}
        meta = encode_conditions(dict(solvent='D2O'))
        meta['line_centers'] = [1., 7.]
        data, graphs = self.dataset(row, labels, meta)
        item = data[0]
        self.assertLessEqual(float(item.q.max()), .100001)
        self.assertEqual(item.site_mask.sum(1)[:3].tolist(), [3, 1, 0])
        self.assertGreater(float(item.research_assignment_weight), 0.)
        self.assertFalse(hasattr(item, 'frozen_embedding'))
        self.assertTrue(hasattr(graphs[row['smiles']], 'frozen_embedding'))
        self.assertEqual(labels[make_key(row)], annotation)
        blocked = dict(row, research_disable_assignment=True)
        disabled, _ = self.dataset(blocked, labels, meta)
        self.assertFalse(disabled[0].site_mask.any())
        no_evidence = dict(annotation, trusted_site_mask={})
        unknown, _ = self.dataset(row, {make_key(row): no_evidence}, meta)
        self.assertFalse(unknown[0].site_mask.any())

    def test_atomic_completeness_does_not_become_measurement_window(self):
        row = self.row(source='QM9-NMR', domain='theory')
        row['measure'] = 'atom_equal'
        row['observation'] = dict(kind='complete_atoms', windows=[])
        data, _ = self.dataset(row)
        item = data[0]
        self.assertEqual(int(item.coverage_kind), 2)
        self.assertFalse(item.window_mask.any())
        self.assertFalse(bool(item.research_full_distribution))

    def selection_fixture(self):
        train = [identity('C' * n)[0] for n in range(1, 35) if identity('C' * n)[2] == 'train']
        val = next(identity('C' * n)[0] for n in range(1, 35) if identity('C' * n)[2] == 'val')
        rows = [self.row(train[0], rid='exp'), self.row(val, rid='val')]
        rows += [self.row(train[0], source=source, domain='theory', rid=source) for source in THEORY_SOURCES]
        rows += [self.row(train[1], source='FG26', domain='theory', rid='forbidden')]
        audit = dict(selected_training_groups=[rows[0]['group']],
            forbidden_groups=[rows[1]['group'], rows[5]['group']], validation_keys=[list(make_key(rows[1]))],
            theory_indices_by_source={source: [index + 2] for index, source in enumerate(THEORY_SOURCES)})
        return rows, audit

    def test_data_boundaries_and_equal_source_exposure_are_deterministic(self):
        rows, audit = self.selection_fixture()
        exp, theory, val = select_training_rows(rows, audit)
        self.assertEqual((len(exp), len(theory), len(val)), (1, 3, 1))
        mapped = copy.deepcopy(audit)
        mapped['validation_keys'] = dict(original_val=mapped['validation_keys'], tune=[],
            verify=[['experimental', 'sealed_source', 'never_read', 'C']])
        self.assertEqual(len(select_training_rows(rows, mapped)[2]), 1)
        left = make_exposure(exp, theory, 97, 193, 1)
        self.assertEqual(left, make_exposure(exp, theory, 97, 193, 1))
        counts = {source: sum(theory[i]['source'] == source for i in left[1]) for source in THEORY_SOURCES}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(len(left[0]), len(left[1]))
        leaked = copy.deepcopy(audit)
        leaked['theory_indices_by_source']['FG26'].append(5)
        with self.assertRaisesRegex(ValueError, 'theory_index_scope_mismatch'):
            select_training_rows(rows, leaked)
        changed = copy.deepcopy(audit)
        changed['selected_training_groups'] += [rows[5]['group']]
        with self.assertRaisesRegex(ValueError, 'selected_forbidden_overlap'):
            select_training_rows(rows, changed)

    def test_real_optimizer_updates_GINE_sigma_and_preserves_frozen_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = model_fixtures.ResearchForwardTests().checkpoint(folder)
            model = load_checkpoint_model(checkpoint, trainable=True)
            prefixes, _ = configure_forward_training(model)
            initial = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            data, _ = self.dataset(self.row())
            batch = collate([data[0]])
            details = model(batch, details=True)
            parts = forward_objective(details, batch)
            parts['loss'].backward()
            self.assertGreater(float(model.research_sigma_head[-1].bias.grad.abs().sum()), 0.)
            self.assertGreater(float(model.atom_bond_block_list[-1].conv.nn[0].weight.grad.abs().sum()), 0.)
            optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
            optimizer.step()
            self.assertGreater(frozen_core_check(initial, model.state_dict(), prefixes), 0)
            with torch.no_grad():
                model.atom_bond_block_list[0].conv.nn[0].weight.add_(1.)
            with self.assertRaisesRegex(ValueError, 'frozen_tensor_changed'):
                frozen_core_check(initial, model.state_dict(), prefixes)

    def test_duplicate_theory_keys_keep_each_cache_row_and_never_share_hard_assignment(self):
        rows, audit = self.selection_fixture()
        other = dict(rows[2], q=quantiles([3., 4.]))
        rows.append(other)
        audit['theory_indices_by_source']['QME14S'].append(6)
        _, theory, _ = select_training_rows(rows, audit)
        matched = [row for row in theory if row['source'] == 'QME14S']
        self.assertEqual(len(matched), 2)
        self.assertEqual({row['research_cache_row_index'] for row in matched}, {2, 6})
        self.assertNotEqual(exposure_key(matched[0]), exposure_key(matched[1]))
        self.assertFalse(np.array_equal(matched[0]['q'], matched[1]['q']))
        audit['quality_defaults'] = {'theory|QME14S': dict(assignment_weight=1., assignment_evidence='unresolved shared key')}
        with self.assertRaisesRegex(ValueError, 'ambiguous_theory_assignment_key'):
            select_training_rows(rows, audit)

    def test_small_actual_pilot_main_never_computes_validation_and_records_true_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = model_fixtures.ResearchForwardTests().checkpoint(folder)
            rows, audit = self.selection_fixture()
            prepared = root / 'prepared'
            prepared.mkdir()
            graphs = {}
            for smiles in {row['smiles'] for row in rows}:
                item = graph(smiles)
                item.local_environment = torch.tensor(local_environment(smiles))
                item.frozen_embedding = torch.zeros(len(item.x), 16)
                graphs[smiles] = item
            torch.save(dict(rows=rows, graphs=graphs), prepared / 'cache.pt')
            cache_sha = sha(prepared / 'cache.pt')
            (prepared / 'manifest.json').write_text(json.dumps(dict(cache_sha256=cache_sha)))
            metadata = {make_key(row): encode_conditions({}) for row in rows}
            torch.save(metadata, prepared / 'metadata.pt')
            torch.save({}, root / 'labels.pt')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            saved.update(prepared_sha256=cache_sha, baseline_sha256='synthetic')
            torch.save(saved, checkpoint)
            paths = [prepared / 'cache.pt', prepared / 'manifest.json', prepared / 'metadata.pt',
                     root / 'labels.pt', checkpoint, Path(saved['args']['equivalence_registry_path'])]
            audit['input_sha256'] = {str(path.resolve()): sha(path) for path in paths}
            (root / 'audit.json').write_text(json.dumps(audit))
            out = root / 'pilot'
            arguments = ['h_research_forward_train', '--prepared', str(prepared), '--metadata', str(prepared / 'metadata.pt'),
                '--labels', str(root / 'labels.pt'), '--init', str(checkpoint), '--audit', str(root / 'audit.json'),
                '--out', str(out), '--mode', 'physical', '--seed', '193', '--stage', 'pilot',
                '--epochs', '1', '--samples', '6', '--domain-batch', '2', '--workers', '0']
            with mock.patch('sys.argv', arguments), mock.patch.object(trainer, 'validate_forward',
                    side_effect=AssertionError('pilot touched validation objective')):
                trainer.main(device='cpu')
            self.assertEqual((out / 'EXIT_CODE').read_text().strip(), '0')
            summary = json.loads((out / 'summary.json').read_text())
            self.assertTrue(all(summary['gradient_seen'].values()))
            self.assertIsNone(summary['best_macro'])
            self.assertEqual(summary['actual_theory_source_exposures'], {source: 2 for source in THEORY_SOURCES})
            history = [json.loads(line) for line in (out / 'history.jsonl').read_text().splitlines()]
            self.assertTrue(all(not row['validation']['validation_objectives_computed'] for row in history))
            self.assertEqual(history[-1]['steps'], 3)


if __name__ == '__main__':
    unittest.main()

"""功能概述：用合成排名核验本轮汇总、完整性门禁及结构聚类配对统计。

只创建临时JSON/JSONL及微型CPU checkpoint；不读真实test、verify或研究结果。
覆盖同连接结构多谱、跨seed逐query指标平均、排名乱序、归属/seed混淆、真实EXIT、
不完整家族、五项晋升guard、低成本并列和封存前禁读test等关键行为。
同时核验实际checkpoint的seed/有序角色、跨家族输出复用及显式Recall@512证据。
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import h_research_results as results


FAMILIES = ['incumbent_frozen', 'matched_budget_2048'] + [f'factor_R{r}O{o}F{f}'
    for r in (0, 1) for o in (0, 1) for f in (0, 1)] + ['forward_local_slots']


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path.resolve())


def row(identifier, smiles='C', rank=2, variant='observed', cohort='original_val', source='S0'):
    return dict(record_id=identifier, smiles=smiles, source=source, rank=rank, variant=variant,
                cohort=cohort, group=results.connectivity(smiles), query_order=identifier)


def pin(entry):
    entry['inputs_sha256'] = {path: results.sha(path) for path in results._entry_paths(entry)}


def fixture(directory, candidates=('matched_budget_2048',)):
    root = Path(directory)
    arms = write(root/'arms.json', dict(seeds=[211, 193], families=[dict(name=n) for n in FAMILIES]))
    manifest = dict(schema=results.SCHEMA, scope='train_val_only', seeds=[211, 193],
        preregistered_arms=arms, preregistered_arms_sha256=results.sha(arms),
        candidate_count=114982, candidate_order_sha256='a'*64, expected_sources=[f'S{i}' for i in range(5)],
        expected_counts={cohort: dict(observed=count, frequency_crop=1, peak_dropout=1)
                         for cohort, count in results.VAL_COUNTS.items()}, families=[])
    for family in ['incumbent_frozen'] + list(candidates):
        output = dict(name=family, inference_cost=2 if family == 'matched_budget_2048' else 1, entries=[])
        for seed in results.SEEDS:
            for cohort, count in results.VAL_COUNTS.items():
                folder = root/family/str(seed)/cohort; folder.mkdir(parents=True)
                rank = 2 if family == 'incumbent_frozen' else 1
                rows = [row(i, ['C', 'CC', 'CCC', 'CO', 'CN'][i % 5], rank, cohort=cohort, source=f'S{i%5}')
                        for i in range(count)]
                rows += [row(count+i, 'CCO', rank, variant=v, cohort=cohort) for i, v in enumerate(results.VARIANTS[1:])]
                predictions = folder/'predictions.jsonl'
                predictions.write_text(''.join(json.dumps(item)+'\n' for item in reversed(rows)))
                checkpoint_paths = [folder/(role+'.pt') for role in ('no_aug', 'aug')]
                for role, checkpoint in zip(('no_aug', 'aug'), checkpoint_paths):
                    torch.save(dict(args=dict(seed=seed, augment=role == 'aug')), checkpoint)
                checkpoints = {str(p.resolve()): results.sha(p) for p in checkpoint_paths}
                protocol = write(folder/'protocol.json', dict(args=dict(seed=seed, stage='evaluate',
                    no_aug_checkpoint=str(checkpoint_paths[0].resolve()), aug_checkpoint=str(checkpoint_paths[1].resolve())),
                    inputs_sha256=checkpoints))
                summary = write(folder/'summary.json', dict(metrics=results.summaries(rows), cohort=cohort,
                    candidate_order_sha256=manifest['candidate_order_sha256'], checkpoints=checkpoints))
                exit_code = folder/'EXIT_CODE'; exit_code.write_text('0\n')
                entry = dict(seed=seed, cohort=cohort, adapter='research', protocol=protocol,
                    predictions=[str(predictions.resolve())], summary=summary, exit_code=str(exit_code.resolve()))
                pin(entry); output['entries'].append(entry)
        manifest['families'].append(output)
    return Path(write(root/'manifest.json', manifest)), manifest


class ResearchResultsTests(unittest.TestCase):
    def test_legacy_config_cannot_override_actual_checkpoint_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); summary = {}; evidence = []
            for role in ('no_aug', 'aug'):
                checkpoint = root/(role+'.pt')
                torch.save(dict(args=dict(seed=193, augment=role == 'aug')), checkpoint)
                digest = results.sha(checkpoint)
                config = write(root/(role+'.json'), dict(checkpoint=str(checkpoint.resolve()),
                    checkpoint_sha256=digest, args=dict(seed=211, augment=role == 'aug')))
                evidence.append(dict(role=role, checkpoint=str(checkpoint.resolve()), config=config))
                summary[role+'_checkpoint'] = str(checkpoint.resolve()); summary[role+'_sha256'] = digest
            with self.assertRaisesRegex(ValueError, 'config_actual_checkpoint_mismatch'):
                results._seed_proof(dict(seed=211, adapter='legacy', seed_evidence=evidence), summary)

    def test_cluster_duplicate_structure_is_one_unit_and_crop_interval_crosses_zero(self):
        before = [row(0, 'C', 2, 'frequency_crop'), row(1, 'C', 2, 'frequency_crop'), row(2, 'CC', 1, 'frequency_crop')]
        after = [dict(r, rank=1 if r['smiles'] == 'C' else 2) for r in before]
        output = results.paired_cluster_bootstrap({193: after}, {193: list(reversed(before))})
        self.assertEqual(output['connectivity_clusters'], 2)
        interval = output['metrics']['193']['MRR']['cluster_95_percentile_interval']
        self.assertLess(interval[0], 0); self.assertGreater(interval[1], 0)
        self.assertAlmostEqual(output['metrics']['193']['MRR']['difference'], 1/6)
        same = [row(0, 'C[C@H](O)F'), row(1, 'C[C@@H](O)F')]
        self.assertEqual(results.paired_cluster_bootstrap({193: same}, {193: same})['connectivity_clusters'], 1)

    def test_two_seed_average_is_per_query_metric_and_not_reciprocal_mean_rank(self):
        old = {seed: [row(0, rank=2)] for seed in results.SEEDS}
        new = {193: [row(0, rank=1)], 211: [row(0, rank=4)]}
        output = results.paired_cluster_bootstrap(new, old)
        self.assertEqual(output['metrics']['two_seed_query_mean']['MRR']['difference'], .125)
        self.assertNotAlmostEqual(.125, 1/2.5-.5)
        with self.assertRaisesRegex(ValueError, '5000'):
            results.paired_cluster_bootstrap(new, old, repeats=4999)

    def test_paired_alignment_rejects_missing_duplicate_and_changed_query_order(self):
        original = [row(0), row(1, 'CC')]
        results.paired_cluster_bootstrap({193: original}, {193: list(reversed(original))})
        for altered in (original[:1], [original[0], original[0]], [dict(original[0], query_order=99), original[1]]):
            with self.assertRaises(ValueError):
                results.paired_cluster_bootstrap({193: altered}, {193: original})

    def test_complete_validation_micro_then_seed_mean_selects_and_preserves_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            output = results.aggregate_manifest(path)
            self.assertEqual(output['selected_family'], 'matched_budget_2048')
            self.assertEqual(output['complete_families'], 2)
            self.assertEqual(len(output['pending_families']), 9)
            self.assertTrue(output['selection_provisional'])
            candidate = output['families']['matched_budget_2048']
            self.assertEqual(candidate['mean']['overall']['queries'], 1344)
            self.assertEqual(candidate['mean']['overall']['MRR'], 1)
            self.assertEqual(candidate['by_seed']['193']['overall']['queries'], 1344)
            self.assertEqual(candidate['by_cohort']['tune']['overall']['queries'], 1024)
            self.assertTrue(all(candidate['promotion_guards'].values()))
            self.assertEqual(set(candidate['paired_vs_incumbent']['observed']['metrics']), {'193', '211', 'two_seed_query_mean'})

    def test_only_all_eleven_complete_families_have_nonprovisional_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = fixture(temporary, candidates=FAMILIES[1:])
            output = results.aggregate_manifest(path)
            self.assertEqual(output['complete_families'], 11)
            self.assertTrue(output['all_families_complete'])
            self.assertFalse(output['selection_provisional'])

    def test_partial_family_cannot_promote_and_unknown_exit_is_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            family = manifest['families'][1]; family['entries'].pop()
            Path(family['entries'][0]['exit_code']).write_text('unknown\n')
            write(path, manifest)
            output = results.aggregate_manifest(path)
            self.assertEqual(output['selected_family'], 'incumbent_frozen')
            self.assertEqual(output['families'][family['name']]['state'], 'pending')
            self.assertNotIn('promotion_guards', output['families'][family['name']])
            Path(family['entries'][0]['exit_code']).write_text('7\n')
            output = results.aggregate_manifest(path)
            self.assertEqual(output['families'][family['name']]['state'], 'failed')

    def test_cohorts_are_micro_merged_before_two_seed_average(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            for entry in manifest['families'][1]['entries']:
                rows = [json.loads(line) for line in Path(entry['predictions'][0]).read_text().splitlines()]
                for item in rows:
                    if item['variant'] == 'observed' and item['cohort'] == 'tune':
                        item['rank'] = 4 if entry['seed'] == 193 else 2
                Path(entry['predictions'][0]).write_text(''.join(json.dumps(item)+'\n' for item in rows))
                summary = results.read(entry['summary']); summary['metrics'] = results.summaries(rows)
                write(entry['summary'], summary); pin(entry)
            write(path, manifest)
            candidate = results.aggregate_manifest(path)['families']['matched_budget_2048']
            self.assertAlmostEqual(candidate['mean']['overall']['MRR'], (320+1024*.375)/1344)
            self.assertAlmostEqual(candidate['by_cohort']['tune']['overall']['MRR'], .375)
            self.assertNotAlmostEqual(candidate['mean']['overall']['MRR'], (1+.375)/2)

    def test_each_nondegradation_guard_can_block_and_tie_prefers_lower_cost(self):
        base = dict(overall=dict(MRR=.5, top1=.4), stress={v: dict(MRR=.3) for v in results.VARIANTS[1:]}, five_source_macro_MRR=.5)
        improved = copy.deepcopy(base); improved['overall']['MRR'] = .6
        for location, key in [(('overall',), 'top1'), (('stress', 'frequency_crop'), 'MRR'),
                              (('stress', 'peak_dropout'), 'MRR'), ((), 'five_source_macro_MRR')]:
            bad = copy.deepcopy(improved); parent = bad
            for item in location: parent = parent[item]
            parent[key] -= .2
            self.assertFalse(all(results.promotion_guards(bad, base).values()))
        self.assertFalse(results.promotion_guards(base, base)['normal_MRR'])
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = fixture(temporary, candidates=('matched_budget_2048', 'factor_R0O0F0'))
            output = results.aggregate_manifest(path)
            self.assertEqual(output['selected_family'], 'factor_R0O0F0')

    def test_seed_confusion_and_summary_forgery_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            entry = manifest['families'][1]['entries'][0]
            protocol = results.read(entry['protocol']); protocol['args']['seed'] = 211
            write(entry['protocol'], protocol); pin(entry); write(path, manifest)
            with self.assertRaisesRegex(ValueError, 'seed_or_stage'):
                results.aggregate_manifest(path)
            protocol['args']['seed'] = 193; write(entry['protocol'], protocol)
            summary = results.read(entry['summary']); summary['metrics']['overall']['MRR'] = .99
            write(entry['summary'], summary); pin(entry); write(path, manifest)
            with self.assertRaisesRegex(ValueError, 'metric_mismatch'):
                results.aggregate_manifest(path)

    def test_duplicate_seed_declaration_or_shared_prediction_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            original = copy.deepcopy(manifest)
            manifest['families'][1]['entries'].append(copy.deepcopy(manifest['families'][1]['entries'][0]))
            write(path, manifest)
            with self.assertRaisesRegex(ValueError, 'duplicate_seed'):
                results.aggregate_manifest(path)
            manifest = original
            manifest['families'][1]['entries'][2]['predictions'] = manifest['families'][1]['entries'][0]['predictions']
            write(path, manifest)
            with self.assertRaisesRegex(ValueError, 'reuses_predictions'):
                results.aggregate_manifest(path)

    def test_actual_checkpoint_seed_and_ordered_roles_cannot_be_relabelled(self):
        for fault in ('actual_seed', 'ordered_roles', 'bool_seed'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                path, manifest = fixture(temporary)
                entry = next(e for e in manifest['families'][1]['entries'] if e['seed'] == 211)
                protocol = results.read(entry['protocol'])
                summary = results.read(entry['summary'])
                if fault == 'ordered_roles':
                    args = protocol['args']
                    args['no_aug_checkpoint'], args['aug_checkpoint'] = args['aug_checkpoint'], args['no_aug_checkpoint']
                else:
                    checkpoint = protocol['args']['no_aug_checkpoint']
                    torch.save(dict(args=dict(seed=193 if fault == 'actual_seed' else True, augment=False)), checkpoint)
                    digest = results.sha(checkpoint)
                    protocol['inputs_sha256'][checkpoint] = digest
                    summary['checkpoints'][checkpoint] = digest
                write(entry['protocol'], protocol); write(entry['summary'], summary)
                pin(entry); write(path, manifest)
                with self.assertRaisesRegex(ValueError, 'checkpoint_seed_or_role'):
                    results.aggregate_manifest(path)

    def test_prediction_files_cannot_count_as_two_different_complete_families(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            duplicate = copy.deepcopy(manifest['families'][1])
            duplicate['name'] = 'factor_R0O0F0'
            manifest['families'].append(duplicate); write(path, manifest)
            with self.assertRaisesRegex(ValueError, 'family_reuses_predictions'):
                results.aggregate_manifest(path)

    def test_recall_uses_explicit_pool_evidence_and_conditional_in_pool_mrr(self):
        rows = [dict(row(0, rank=2), target_in_coarse_pool=True, evaluation_pool_size=512),
                dict(row(1, rank=600), target_in_coarse_pool=False, evaluation_pool_size=512),
                dict(row(2, rank=1), target_in_coarse_pool=True, evaluation_pool_size=512)]
        metrics = results.metrics(rows)
        self.assertEqual(metrics['recall512'], 2/3)
        self.assertEqual(metrics['in_pool_MRR'], .75)
        self.assertAlmostEqual(metrics['final_MRR'], (1.5+1/600)/3)
        self.assertEqual(metrics['final_MRR'], metrics['MRR'])
        self.assertEqual(metrics['recall_diagnostics']['known_in_pool_queries'], 2)
        outside = results.metrics(rows[1:2])
        self.assertEqual(outside['recall512'], 0)
        self.assertIsNone(outside['in_pool_MRR'])
        self.assertEqual(outside['recall_diagnostics']['state'], 'supported')
        seed_means = results._mean_summaries({s: results.summaries(rows) for s in results.SEEDS}, ['S0'])
        self.assertEqual(seed_means['overall']['recall512'], 2/3)
        self.assertEqual(seed_means['overall']['in_pool_MRR'], .75)

    def test_unknown_or_partial_recall_never_uses_rank_as_membership_or_fills_zero(self):
        for item in (row(0, rank=1), dict(row(0, rank=1), target_in_coarse_pool=True),
                     dict(row(0, rank=900), evaluation_pool_size=512)):
            with self.subTest(item=item):
                computed = results.metrics([item])
                self.assertEqual(computed['recall_diagnostics']['state'], 'unsupported')
                self.assertIsNone(computed['recall512']); self.assertIsNone(computed['in_pool_MRR'])
                self.assertEqual(computed['final_MRR'], 1/item['rank'])
        known = dict(row(1), target_in_coarse_pool=True, evaluation_pool_size=512)
        partial = results.metrics([known, row(0)])
        self.assertEqual(partial['recall_diagnostics']['state'], 'partial')
        self.assertEqual(partial['recall_diagnostics']['known_queries'], 1)
        self.assertIsNone(partial['recall512']); self.assertIsNone(partial['in_pool_MRR'])
        for changes in (dict(target_in_coarse_pool=1), dict(evaluation_pool_size=True),
                        dict(evaluation_pool_size=1024), dict(target_in_coarse_pool=False)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'pool'):
                results.metrics([dict(known, **changes)])

    def test_cli_exports_per_query_recall_without_mutating_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            entry = manifest['families'][1]['entries'][0]
            rows = [json.loads(line) for line in Path(entry['predictions'][0]).read_text().splitlines()]
            rows = [dict(item, target_in_coarse_pool=True, evaluation_pool_size=512) for item in rows]
            Path(entry['predictions'][0]).write_text(''.join(json.dumps(item)+'\n' for item in rows))
            summary = results.read(entry['summary']); summary['metrics'] = results.summaries(rows)
            write(entry['summary'], summary); pin(entry); write(path, manifest)
            inputs = {p: results.sha(p) for family in manifest['families'] for e in family['entries'] for p in results._entry_paths(e)}
            inputs[str(path)] = results.sha(path)
            out = Path(temporary)/'report'
            with patch('sys.argv', ['results', '--manifest', str(path), '--out', str(out)]):
                results.main()
            report = results.read(out/'summary.json')
            self.assertEqual((out/'EXIT_CODE').read_text().strip(), '0')
            table = report['recall_table']
            actual = [json.loads(line) for line in Path(table['path']).read_text().splitlines()]
            self.assertEqual(table['rows'], 2*2*(320+1024+4))
            self.assertEqual(len(actual), table['rows'])
            self.assertEqual(table['sha256'], results.sha(table['path']))
            known = [r for r in actual if r['recall_status'] == 'known']
            self.assertEqual(len(known), 322)
            self.assertTrue(all(r['family'] == 'matched_budget_2048' and r['seed'] == 193 and
                                r['cohort'] == 'original_val' for r in known))
            self.assertEqual(len({(r['family'], r['seed'], r['query_uid']) for r in actual}), len(actual))
            self.assertTrue(all(r['target_in_coarse_pool'] is None for r in actual if r['recall_status'] == 'unknown'))
            self.assertNotIn('recall_query_rows', report)
            self.assertEqual(inputs, {p: results.sha(p) for p in inputs})

    def test_legacy_seed_and_ordered_checkpoint_proof_are_bound_to_summary(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            entry = manifest['families'][0]['entries'][0]
            summary = results.read(entry['summary']); entry.pop('protocol'); entry['adapter'] = 'legacy'
            entry['seed_evidence'] = []
            for role in ('no_aug', 'aug'):
                checkpoint = Path(entry['summary']).parent/(role+'.pt')
                torch.save(dict(args=dict(seed=193, augment=role == 'aug')), checkpoint)
                summary[role+'_checkpoint'] = str(checkpoint.resolve())
                summary[role+'_sha256'] = results.sha(checkpoint)
                entry['seed_evidence'].append(dict(role=role, checkpoint=str(checkpoint.resolve())))
            write(entry['summary'], summary); pin(entry); write(path, manifest)
            output = results.aggregate_manifest(path)
            self.assertEqual(output['complete_families'], 2)
            evidence = entry['seed_evidence'][0]
            torch.save(dict(args=dict(seed=211, augment=False)), evidence['checkpoint'])
            summary['no_aug_sha256'] = results.sha(evidence['checkpoint']); write(entry['summary'], summary)
            pin(entry); write(path, manifest)
            with self.assertRaisesRegex(ValueError, 'legacy_seed_or_role'):
                results.aggregate_manifest(path)

    def test_after_read_prediction_change_is_rejected_and_cli_records_real_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            target = Path(manifest['families'][1]['entries'][0]['predictions'][0])
            real = results.paired_cluster_bootstrap
            changed = False
            def mutate(*args, **kwargs):
                nonlocal changed
                output = real(*args, **kwargs)
                if not changed:
                    target.write_text(target.read_text()+'\n'); changed = True
                return output
            with patch.object(results, 'paired_cluster_bootstrap', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'input_changed_during_read'):
                    results.aggregate_manifest(path)
            out = Path(temporary)/'failed_output'
            with patch('sys.argv', ['results', '--manifest', str(path), '--out', str(out)]):
                with self.assertRaisesRegex(ValueError, 'input_SHA'):
                    results.main()
            self.assertEqual((out/'EXIT_CODE').read_text().strip(), '1')
            self.assertFalse((out/'summary.json').exists())

    def test_test_scope_requires_current_freeze_and_never_selects(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = fixture(temporary)
            manifest['scope'] = 'frozen_evaluation'
            manifest['expected_counts'] = dict(test=dict(observed=1000, frequency_crop=200, peak_dropout=171))
            manifest['expected_query_identity_sha256'] = {'test': 'b'*64}
            with self.assertRaisesRegex(ValueError, 'frozen_selection_required'):
                results._scope_guard(manifest, 'frozen_evaluation', None)
            frozen = dict(schema='nmr_research_20261022', frozen_utc='2020-01-01T00:00:00Z',
                selection_scope='train_val_only', results_plan_sha256=results.result_plan_sha256(manifest),
                evaluation_inputs_sha256={str(Path(results.__file__).resolve()): results.sha(results.__file__)})
            frozen_path = write(Path(temporary)/'freeze.json', frozen)
            results._scope_guard(manifest, 'frozen_evaluation', frozen_path)
            altered = copy.deepcopy(manifest); altered['families'][0]['inference_cost'] = 999
            with self.assertRaisesRegex(ValueError, 'invalid_freeze'):
                results._scope_guard(altered, 'frozen_evaluation', frozen_path)
            values = copy.deepcopy(manifest); values['families'][0]['entries'][0]['inputs_sha256'] = {'late_output': 'newsha'}
            self.assertEqual(results.result_plan_sha256(values), results.result_plan_sha256(manifest))
            with self.assertRaisesRegex(ValueError, 'scope'):
                results._scope_guard(manifest, 'train_val_only', None)

    def test_synthetic_frozen_aggregation_is_descriptive_and_checks_exact_query_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            manifest['scope'] = 'frozen_evaluation'
            manifest['expected_counts'] = dict(test=dict(observed=1000, frequency_crop=200, peak_dropout=171))
            frozen_path = Path(temporary)/'freeze.json'
            weights = {}
            for family in manifest['families']:
                family['entries'] = [entry for entry in family['entries'] if entry['cohort'] == 'original_val']
                for entry in family['entries']:
                    entry['cohort'] = 'test'
                    rows = []
                    for variant, count in manifest['expected_counts']['test'].items():
                        rows += [row(i, ['C', 'CC', 'CCC', 'CO', 'CN'][i%5], 2 if family['name'] == 'incumbent_frozen' else 1,
                                     variant=variant, cohort='test', source=f'S{i%5}') for i in range(count)]
                    manifest['expected_query_identity_sha256'] = {'test': results.query_identity_sha256(rows)}
                    Path(entry['predictions'][0]).write_text(''.join(json.dumps(item)+'\n' for item in rows))
                    summary = results.read(entry['summary']); summary.update(cohort='test', metrics=results.summaries(rows))
                    write(entry['summary'], summary)
                    protocol = results.read(entry['protocol']); protocol['args']['frozen_selection'] = str(frozen_path.resolve())
                    write(entry['protocol'], protocol); weights.update(summary['checkpoints']); pin(entry)
            write(path, manifest)
            frozen = dict(schema='nmr_research_20261022', frozen_utc='2020-01-01T00:00:00Z',
                selection_scope='train_val_only', results_plan_sha256=results.result_plan_sha256(manifest),
                evaluation_inputs_sha256={str(Path(results.__file__).resolve()): results.sha(results.__file__),
                    manifest['preregistered_arms']: manifest['preregistered_arms_sha256'], **weights})
            write(frozen_path, frozen)
            output = results.aggregate_manifest(path, scope='frozen_evaluation', frozen_selection=frozen_path)
            self.assertTrue(output['selection_disabled']); self.assertIsNone(output['selected_family'])
            self.assertNotIn('promotion_guards', output['families']['matched_budget_2048'])
            manifest['expected_query_identity_sha256']['test'] = 'f'*64
            write(path, manifest); frozen['results_plan_sha256'] = results.result_plan_sha256(manifest); write(frozen_path, frozen)
            with self.assertRaisesRegex(ValueError, 'query_identity_changed'):
                results.aggregate_manifest(path, scope='frozen_evaluation', frozen_selection=frozen_path)


if __name__ == '__main__':
    unittest.main()

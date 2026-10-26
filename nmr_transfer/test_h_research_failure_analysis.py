"""功能概述：用临时合成验证产物测试失败归因、完整性和固定配对，不读真实数据。

覆盖自然池召回与排序边界、未知频窗、逐query两seed指标平均、固定哈希样例、
ROF配对、结果/诊断SHA及真实CLI成功失败生命周期。所有模型文件均为微型CPU字典。
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import h_research_failure_analysis as failure
from . import h_research_results as results
from .test_h_research_results import fixture, pin, row, write


def record(identifier=0, rank=2, member=True, **kwargs):
    item = dict(row(identifier, rank=rank, **kwargs), evaluation_pool_size=512,
                target_in_coarse_pool=member, analysis_metadata=failure.metadata())
    item['query_uid'] = results.json_digest(list(results.query_key(item)))
    return item


def diagnostic(item):
    return dict({key: item[key] for key in ('source', 'record_id', 'smiles', 'group', 'variant', 'query_order')},
        semantics='assigned_labels', windows_ppm=[], partial_peaks=False, coverage='unknown',
        solvent_status='reported', solvent='water', solvent_family='water', pH_known=True, pH=0.,
        pH_used_by_model=False, frequency_known=False, frequency_MHz=None,
        mask_content_sha256=results.json_digest(['synthetic_observation_mask', item['query_order'], item['variant']]),
        residuals=dict(atom_assignment_metric=False, per_query_weighted=dict(mean_ppm=.2)))


def completed(directory):
    path, manifest = fixture(directory, candidates=('factor_R0O0F0',))
    bank = Path(directory)/'shared_inputs'/'bank.pt'; bank.parent.mkdir()
    bank.write_bytes(b'opaque synthetic bank; never unpickled')
    cohorts = bank.parent/'cohorts'/'cohorts.pt'; cohorts.parent.mkdir()
    cohorts.write_bytes(b'opaque synthetic cohort; never unpickled')
    input_sha = {str(p.resolve()): results.sha(p) for p in (bank, cohorts)}
    for family in manifest['families']:
        for entry in family['entries']:
            protocol = results.read(entry['protocol']); summary = results.read(entry['summary'])
            for role in ('no_aug', 'aug'):
                checkpoint = protocol['args'][role+'_checkpoint']
                torch.save(dict(args=dict(seed=entry['seed'], augment=role == 'aug'),
                    cache_protocol=dict(checkpoint_sha256='f'*64, candidate_order_sha256='a'*64)), checkpoint)
                digest = results.sha(checkpoint)
                protocol['inputs_sha256'][checkpoint] = digest; summary['checkpoints'][checkpoint] = digest
            items = [dict(json.loads(line), target_in_coarse_pool=True, evaluation_pool_size=512)
                     for line in Path(entry['predictions'][0]).read_text().splitlines()]
            cache_path = Path(entry['summary']).parent/'cache'/'features.pt'
            digests = failure.diagnostic_digests([diagnostic(item) for item in items])
            cache_summary = write(cache_path.parent/'summary.json', dict(cohort=entry['cohort'],
                candidate_order_sha256='a'*64, checkpoint_sha256='f'*64, source_inputs_sha256=input_sha, **digests))
            protocol['args']['cache'] = str(cache_path)
            protocol['inputs_sha256'][cache_summary] = results.sha(cache_summary)
            Path(entry['predictions'][0]).write_text(''.join(json.dumps(item)+'\n' for item in items))
            summary['metrics'] = results.summaries(items)
            write(entry['protocol'], protocol); write(entry['summary'], summary); pin(entry)
            manifest.setdefault('expected_query_identity_sha256', {})[entry['cohort']] = results.query_identity_sha256(items)
    write(path, manifest)
    out = Path(directory)/'results'
    with patch('sys.argv', ['results', '--manifest', str(path), '--out', str(out)]):
        results.main()
    summary = results.read(out/'summary.json'); table = summary['recall_table']
    spec = dict(schema=failure.SCHEMA, scope='train_val_only', results_manifest=str(path),
        results_manifest_sha256=results.sha(path), results_summary=str(out/'summary.json'),
        results_summary_sha256=results.sha(out/'summary.json'), recall_queries=table['path'],
        recall_queries_sha256=table['sha256'], results_exit_code=str(out/'EXIT_CODE'),
        results_exit_code_sha256=results.sha(out/'EXIT_CODE'), diagnostics=[])
    return Path(write(Path(directory)/'analysis_inputs.json', spec)), spec, manifest


def add_diagnostic(directory, spec, manifest, *, bound=False, checkpoint_sha='f'*64, incomplete=False):
    family = 'factor_R0O0F0'
    entry = next(e for f in manifest['families'] if f['name'] == family for e in f['entries']
                 if e['seed'] == 193 and e['cohort'] == 'original_val')
    items = [diagnostic(json.loads(line)) for line in Path(entry['predictions'][0]).read_text().splitlines()]
    folder = Path(directory)/'diagnostic'/'original_val'; folder.mkdir(parents=True)
    per_query = folder/'per_query.jsonl'
    items = items[:-1] if incomplete else items
    per_query.write_text(''.join(json.dumps(item)+'\n' for item in items))
    cache = results.read(Path(results.read(entry['protocol'])['args']['cache']).parent/'summary.json')
    inputs = cache['source_inputs_sha256']
    protocol = write(folder/'protocol.json', dict(args=dict(cohort='original_val',
        bank=next(p for p in inputs if Path(p).name == 'bank.pt'),
        cohorts=str(Path(next(p for p in inputs if Path(p).name == 'cohorts.pt')).parent), extension=None), inputs_sha256=inputs))
    summary = write(folder/'summary.json', dict(schema='research_forward_diagnostics_v1', cohort='original_val',
        candidate_order_sha256='a'*64, checkpoint_sha256=checkpoint_sha, inputs_start_sha256=inputs, inputs_end_sha256=inputs,
        per_query_sha256=results.sha(per_query), **failure.diagnostic_digests(items)))
    exit_code = folder/'EXIT_CODE'; exit_code.write_text('0\n')
    output = dict(family=family, seed=193, cohort='original_val', summary=summary, summary_sha256=results.sha(summary),
        protocol=protocol, protocol_sha256=results.sha(protocol), per_query=str(per_query),
        per_query_sha256=results.sha(per_query), exit_code=str(exit_code), exit_code_sha256=results.sha(exit_code))
    if bound:
        checkpoint = results.read(entry['protocol'])['args']['no_aug_checkpoint']
        output.update(forward_checkpoint_sha256=checkpoint_sha,
                      checkpoint_binding=dict(path=checkpoint, sha256=results.sha(checkpoint)))
    spec['diagnostics'].append(output)
    return output


class FailureAnalysisTests(unittest.TestCase):
    def test_recall_and_ranking_failures_use_explicit_membership_not_missing_assignment(self):
        rows = [record(0, 600, False), record(1, 20, True), record(2, 30, None),
                record(3, 1), record(4, 5), record(5, 10)]
        self.assertEqual([failure.failure_category(r) for r in rows], list(failure.FAILURES))
        output = failure.single_bucket(rows)
        self.assertEqual(output['failure_counts'], dict.fromkeys(failure.FAILURES, 1))
        self.assertEqual(output['Topk_failure_decomposition']['10']['total_failures'], 3)
        self.assertEqual(output['Topk_failure_decomposition']['10']['recall_failures'], 1)
        self.assertEqual(output['Topk_failure_decomposition']['10']['recalled_ranking_failures'], 1)
        self.assertEqual(output['Topk_failure_decomposition']['10']['unknown_membership_failures'], 1)
        self.assertEqual(output['Topk_failure_decomposition']['1']['total_failures'], 5)
        self.assertIsNone(output['metrics']['recall512'])

    def test_missingness_is_only_explicit_window_and_partial_metadata(self):
        item = diagnostic(record(0, variant='frequency_crop'))
        self.assertEqual(failure.metadata(item)['missingness'], 'unknown_coverage')
        item.update(windows_ppm=[[0., 5.]], coverage='explicit_window')
        self.assertEqual(failure.metadata(item)['missingness'], 'explicit_window')
        item['partial_peaks'] = True
        self.assertEqual(failure.metadata(item)['missingness'], 'window_and_partial')
        item['coverage'] = 'complete'
        with self.assertRaisesRegex(ValueError, 'coverage'):
            failure.metadata(item)

    def test_numeric_zero_ph_known_frequency_unknown_and_residual_binding(self):
        item = diagnostic(record())
        values = failure.metadata(item)
        self.assertEqual(values['pH_known'], 'known'); self.assertEqual(values['frequency_known'], 'unknown')
        self.assertEqual(values['residual_bin'], 'unknown')
        self.assertEqual(failure.metadata(item, residual_bound=True)['residual_bin'], '(0.1,0.25]ppm')
        item['residuals']['atom_assignment_metric'] = True
        with self.assertRaisesRegex(ValueError, 'assignment'):
            failure.metadata(item, residual_bound=True)

    def test_two_seed_means_metrics_before_aggregation_and_retains_discordance(self):
        rows = {193: [record(rank=1)], 211: [record(rank=4)]}
        combined = failure.mean_query_rows(rows)
        self.assertNotIn('rank', combined[0])
        output = failure.mean_bucket(combined)
        self.assertEqual(output['metrics']['MRR'], .625)
        self.assertNotEqual(output['metrics']['MRR'], 1/2.5)
        self.assertEqual(output['failure_pattern_counts'], {'top1|rank2_5': 1})
        self.assertEqual(output['membership_pattern_counts'], {'both_in': 1})

    def test_one_seed_metadata_missing_is_unknown_and_conflicting_known_metadata_rejected(self):
        a, b = record(), record()
        a['analysis_metadata'] = failure.metadata(diagnostic(a))
        combined = failure.mean_query_rows({193: [a], 211: [b]})
        self.assertEqual(combined[0]['analysis_metadata']['semantics'], 'unknown')
        b['analysis_metadata'] = dict(a['analysis_metadata'], semantics='line_positions')
        with self.assertRaisesRegex(ValueError, 'metadata_mismatch'):
            failure.mean_query_rows({193: [a], 211: [b]})

    def test_fixed_factorial_plan_has_four_contrasts_per_axis_and_separate_controls(self):
        plan = failure.contrast_plan()
        self.assertEqual(len(plan), 14)
        for axis in 'ROF':
            parts = [p for p in plan if p['axis'] == axis]
            self.assertEqual(len(parts), 4)
            for part in parts:
                before, after = part['before'][-6:], part['after'][-6:]
                self.assertEqual(sum(a != b for a, b in zip(before, after)), 1)
        self.assertEqual(len(failure.CORE_CI), 5)

    def test_contrast_retains_opposite_seed_directions_and_true_per_query_delta(self):
        before = {seed: [record(rank=2)] for seed in results.SEEDS}
        after = {193: [record(rank=1)], 211: [record(rank=4)]}
        output, rows = failure.compare_families(before, after, ci=True)
        self.assertEqual(output['two_seed_query_metric_mean_delta']['MRR'], .125)
        self.assertEqual(output['per_seed']['193']['metric_delta']['MRR'], .5)
        self.assertEqual(output['per_seed']['211']['metric_delta']['MRR'], -.25)
        self.assertEqual(output['paired_cluster_CI']['metrics']['two_seed_query_mean']['MRR']['difference'], .125)
        self.assertEqual(output['samples']['improved'][0]['before_ranks'], {'193': 2, '211': 2})
        self.assertEqual(rows[0]['after_ranks'], {'193': 1, '211': 4})
        self.assertEqual(rows[0]['after_membership'], {'193': True, '211': True})
        self.assertNotIn('final_reciprocal_rank', rows[0])

    def test_samples_are_hash_stable_not_sorted_by_rank_or_input_order(self):
        rows = [record(i, i+1) for i in range(20)]
        a = failure.samples(rows); b = failure.samples(list(reversed(rows)))
        self.assertEqual(a, b)
        altered = [dict(item, rank=500-item['rank']) for item in rows]
        self.assertEqual([v['record_id'] for v in a], [v['record_id'] for v in failure.samples(altered)])

    def test_all_fixed_contrasts_and_only_fixed_core_ci_no_selection(self):
        names = {p[k] for p in failure.contrast_plan() for k in ('before', 'after')}
        by_family = {name: {seed: [record(0), record(1, variant='frequency_crop'), record(2, variant='peak_dropout')]
                            for seed in results.SEEDS} for name in names}
        with patch.object(results, 'paired_cluster_bootstrap') as bootstrap:
            report = failure.analyze(by_family)
            bootstrap.assert_not_called()
            self.assertFalse(report['selection_performed'])
            self.assertTrue(all(v['state'] == 'complete' for v in report['comparisons'].values()))
            failure.analyze(by_family, core_ci=True)
            self.assertEqual(bootstrap.call_count, 15)
            self.assertTrue(all(call.kwargs == dict(repeats=5000, seed=317) for call in bootstrap.call_args_list))

    def test_completed_results_cli_inputs_and_outputs_are_bound_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, spec, _ = completed(temporary)
            before = {p: results.sha(p) for p in Path(temporary).rglob('*') if p.is_file()}
            out = Path(temporary)/'failure_analysis'
            failure.main(['--manifest', str(path), '--out', str(out)])
            report = results.read(out/'summary.json')
            self.assertEqual((out/'EXIT_CODE').read_text().strip(), '0')
            self.assertEqual(report['inputs_start_sha256'], report['inputs_end_sha256'])
            family = report['families']['factor_R0O0F0']['two_seed_query_metric_mean']['observed']
            self.assertEqual(family['overall']['metrics']['queries'], 1344)
            self.assertEqual(family['overall']['metrics']['MRR'], 1)
            self.assertEqual(len(family['strata']['source']), 5)
            self.assertEqual(family['strata']['semantics']['unknown']['metrics']['queries'], 1344)
            self.assertEqual(report['paired_query_table']['rows'], 0)
            self.assertTrue(report['selection_provisional'])
            self.assertEqual(before, {p: results.sha(p) for p in before})

    def test_nonvalidation_scope_rejected_before_any_prediction_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary)/'inputs.json', dict(schema=failure.SCHEMA, scope='frozen_evaluation'))
            with patch.object(failure.Snapshot, 'rows') as rows:
                with self.assertRaisesRegex(ValueError, 'validation_only'):
                    failure.run(path)
                rows.assert_not_called()

    def test_wrong_sha_or_unknown_real_exit_rejected_and_cli_marks_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, spec, _ = completed(temporary)
            spec['results_summary_sha256'] = '0'*64; write(path, spec)
            out = Path(temporary)/'failed'
            with self.assertRaisesRegex(ValueError, 'SHA_mismatch'):
                failure.main(['--manifest', str(path), '--out', str(out)])
            self.assertEqual((out/'EXIT_CODE').read_text().strip(), '1')
            self.assertFalse((out/'summary.json').exists())

    def test_diagnostic_metadata_without_residual_binding_and_with_actual_checkpoint_binding(self):
        for bound in (False, True):
            with self.subTest(bound=bound), tempfile.TemporaryDirectory() as temporary:
                path, spec, manifest = completed(temporary)
                add_diagnostic(temporary, spec, manifest, bound=bound); write(path, spec)
                report = failure.run(path)
                part = report['families']['factor_R0O0F0']['by_seed']['193']['observed']['strata']
                self.assertEqual(part['semantics']['assigned_labels']['metrics']['queries'], 320)
                self.assertIn('(0.1,0.25]ppm' if bound else 'unknown', part['residual_bin'])

    def test_diagnostic_wrong_forward_checkpoint_or_incomplete_population_is_rejected(self):
        for fault in ('checkpoint', 'population'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                path, spec, manifest = completed(temporary)
                add_diagnostic(temporary, spec, manifest, bound=fault == 'checkpoint',
                    checkpoint_sha='e'*64 if fault == 'checkpoint' else 'f'*64, incomplete=fault == 'population')
                write(path, spec)
                with self.assertRaisesRegex(ValueError, 'binding_mismatch|population_incomplete|observation_or_bank_mismatch'):
                    failure.run(path)

    def test_diagnostic_mask_must_equal_actual_retrieval_cache_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, spec, manifest = completed(temporary)
            entry = add_diagnostic(temporary, spec, manifest)
            items = [json.loads(line) for line in Path(entry['per_query']).read_text().splitlines()]
            items[0]['mask_content_sha256'] = 'd'*64
            Path(entry['per_query']).write_text(''.join(json.dumps(item)+'\n' for item in items))
            summary = results.read(entry['summary'])
            summary.update(per_query_sha256=results.sha(entry['per_query']), **failure.diagnostic_digests(items))
            write(entry['summary'], summary)
            entry['per_query_sha256'] = results.sha(entry['per_query']); entry['summary_sha256'] = results.sha(entry['summary'])
            write(path, spec)
            with self.assertRaisesRegex(ValueError, 'observation_or_bank_mismatch'):
                failure.run(path)

    def test_sealed_diagnostic_path_rejected_before_reading_its_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, spec, manifest = completed(temporary)
            entry = add_diagnostic(temporary, spec, manifest)
            entry['summary'] = str(Path(temporary)/'sealed_final'/'summary.json')
            write(path, spec)
            with self.assertRaisesRegex(ValueError, 'nonvalidation_artifact_path'):
                failure.run(path)

    def test_symlink_to_sealed_artifact_cannot_bypass_validation_path_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'sealed_final'/'rows.jsonl'; target.parent.mkdir()
            target.write_text('opaque sealed sentinel; must not be parsed')
            alias = Path(temporary)/'ordinary_rows.jsonl'; alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'nonvalidation_artifact_path'):
                failure.validation_path(alias)

    def test_disjoint_bank_or_cohort_sources_cannot_bind_diagnostic_to_retrieval(self):
        for changed in ('bank', 'cohorts'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                path, spec, manifest = completed(temporary)
                entry = add_diagnostic(temporary, spec, manifest, bound=True)
                protocol = results.read(entry['protocol']); summary = results.read(entry['summary'])
                original = Path(protocol['args'][changed])
                if changed == 'cohorts':
                    original = original/'cohorts.pt'
                replacement = Path(temporary)/'other_inputs'/original.name; replacement.parent.mkdir()
                replacement.write_bytes(original.read_bytes())
                # 相同字节也不自行认证路径别名；需要另行显式等价凭据，默认拒绝。
                inputs = dict(protocol['inputs_sha256']); inputs.pop(str(original))
                inputs[str(replacement)] = results.sha(replacement)
                protocol['inputs_sha256'] = inputs
                protocol['args'][changed] = str(replacement.parent if changed == 'cohorts' else replacement)
                summary['inputs_start_sha256'] = inputs; summary['inputs_end_sha256'] = inputs
                write(entry['protocol'], protocol); write(entry['summary'], summary)
                entry['protocol_sha256'] = results.sha(entry['protocol']); entry['summary_sha256'] = results.sha(entry['summary'])
                write(path, spec)
                with self.assertRaisesRegex(ValueError, 'required_bank_or_cohort_input_mismatch'):
                    failure.run(path)

    def test_end_sha_guard_rejects_change_during_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = completed(temporary)
            real = failure.analyze
            def mutate(*args, **kwargs):
                output = real(*args, **kwargs)
                path.write_text(path.read_text()+'\n')
                return output
            with patch.object(failure, 'analyze', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'changed_during_read'):
                    failure.run(path)


if __name__ == '__main__':
    unittest.main()

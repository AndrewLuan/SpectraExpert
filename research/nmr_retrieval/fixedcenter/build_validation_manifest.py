#!/usr/bin/env python3
"""功能概述：由正式队列、真实完成回执和旧封存证据构建只含验证集的结果manifest。

本脚本不训练、不执行GPU任务、不访问test/verify或网络。每个预注册家族保留两seed
两cohort的4条planned_entries；只有全部输出和递归祖先有真实双EXIT0、规范完成UTC，
且不晚于显式cutoff时，才复制至results工具消费的entries。未合格家族保留pending。
旧incumbent用已封存summary/EXIT和精确等于冻结blend端点的逐查询排名作内容证明，
完成时间只用旧finalize真实结束UTC上界，绝不使用文件mtime推断历史完成。

成本采用固定增量工作量单位base0+R1+O2，不冒充实际延迟；F的bank预处理离线计费。
所有证据始末SHA核验。父controller在冻结前须再次核验builder_inputs_sha256、cutoff
和完整11家族资格；本脚本成功EXIT0只代表生成快照成功，不代表所有研究作业完成。
"""
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (193, 211)
COHORTS = ('original_val', 'tune')
VARIANTS = ('observed', 'frequency_crop', 'peak_dropout')
LEGACY_BUNDLE_SHA = '42581fcdd7fa342569334949eeb29afe374839bacfcae0638f6170720a7dd9e1'
LEGACY_RECEIPT_SHA = '7d6f7ee9d321d3d9a2b7e937426be19253fb412251d6ac7993b2c8ffe8b7f19f'
LEGACY_FINALIZE_SHA = 'f39108b3264e4677067ce7fa631d45d7338b77f2cf4fe751d9be2ac31e9e98eb'


def canonical(path):
    return str(Path(path).resolve())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'completion_time_requires_timezone')
    return parsed.astimezone(timezone.utc)


def identity_key(row, cohort):
    return (cohort, row['source'], json.dumps(row['record_id'], sort_keys=True), row['smiles'], row['variant'])


def identity_digest(rows, cohort):
    keys = [identity_key(row, cohort) for row in rows]
    require(len(set(keys)) == len(keys), 'duplicate_validation_query_identity')
    return hashlib.sha256(json.dumps([list(k) for k in sorted(keys)], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Snapshot:
    def __init__(self):
        self.inputs = {}

    def pin(self, path, expected=None):
        path = canonical(path); digest = sha(path)
        require(expected is None or digest == expected, 'evidence_SHA_mismatch:' + path)
        require(path not in self.inputs or self.inputs[path] == digest, 'evidence_changed_between_reads:' + path)
        self.inputs[path] = digest
        return digest

    def json(self, path, expected=None):
        self.pin(path, expected)
        return read(path)

    def rows(self, path, cohort, expected=None):
        require(cohort in COHORTS, 'only_original_val_and_tune_allowed')
        parts = Path(path).parts
        require(cohort in parts and not {'test', 'verify', 'sealed'}.intersection(parts), 'nonvalidation_prediction_path_forbidden')
        self.pin(path, expected)
        output = []
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line); row.setdefault('variant', 'observed')
                require(row['variant'] in VARIANTS and row.get('cohort', cohort) == cohort, 'unsupported_validation_variant_or_cohort')
                output.append(row)
        return output

    def finish(self):
        for path, digest in self.inputs.items():
            require(sha(path) == digest, 'builder_input_changed_during_read:' + path)


def ranking_signature(rows, cohort):
    result = {}
    for row in rows:
        key = identity_key(row, cohort)
        require(key not in result, 'duplicate_legacy_endpoint_query')
        result[key] = (row.get('query_order'), row['rank'], row.get('top10', row.get('top10_indices')))
    return result


def legacy_entries(baseline, freeze_dir, cutoff, snapshot, expected_bundle_sha, expected_receipt_sha, expected_finalize_sha):
    folder = Path(freeze_dir)
    bundle = snapshot.json(folder/'freeze_bundle.json', expected_bundle_sha)
    receipt = snapshot.json(folder/'builder_receipt.json', expected_receipt_sha)
    snapshot.pin(folder/'EXIT_CODE')
    require((folder/'EXIT_CODE').read_text().strip() == '0' and receipt['audit'] == 'PASS', 'legacy_freeze_not_complete')
    require(receipt['bundle_sha256'] == expected_bundle_sha and bundle['selection_scope'] == 'train_val_only', 'legacy_freeze_bundle_unbound')
    finalize_path = folder.parent/'finalize_actual_receipt.json'
    finalization = snapshot.json(finalize_path, expected_finalize_sha)
    snapshot.pin(folder.parent/'finalize_actual_EXIT_CODE')
    require((folder.parent/'finalize_actual_EXIT_CODE').read_text().strip() == '0' and
        type(finalization['actual_exit_code']) is int and finalization['actual_exit_code'] == 0 and
        finalization['bundle_sha256'] == expected_bundle_sha, 'legacy_actual_finalize_not_successful')
    completed = utc(finalization['ended_utc'])
    require(utc(finalization['started_utc']) <= utc(bundle['frozen_utc']) <= completed, 'legacy_finalize_time_inconsistent')
    require(completed <= cutoff and completed <= datetime.now(timezone.utc), 'legacy_completion_not_proven_before_cutoff')
    pins = receipt['start_and_end_inputs_sha256']
    fragment = baseline['fragment']
    require(fragment['primary_kind'] == 'dual_route', 'legacy_incumbent_requires_explicit_dual_route')
    owners = {owner['id']: owner for owner in baseline['owners']}
    require(len(owners) == 2 and {owner['seed'] for owner in owners.values()} == set(SEEDS), 'legacy_owner_seed_contract')
    pairs = fragment['pairs']
    require(len(pairs) == 2 and {pair['owner'] for pair in pairs} == set(owners), 'legacy_pair_owner_contract')
    entries, rows_by_cohort, orders, proofs = [], {}, set(), []
    for pair in pairs:
        seed = owners[pair['owner']]['seed']
        for cohort in COHORTS:
            root = Path(pair['dual_route_dir'])/cohort
            summary = snapshot.json(root/'summary.json', pins.get(canonical(root/'summary.json')))
            require(canonical(root/'summary.json') in pins and canonical(root/'EXIT_CODE') in pins, 'legacy_completion_not_in_frozen_evidence')
            snapshot.pin(root/'EXIT_CODE', pins[canonical(root/'EXIT_CODE')])
            require((root/'EXIT_CODE').read_text().strip() == '0' and summary['cohort'] == cohort, 'legacy_validation_incomplete')
            require(summary['normal_gamma'] == fragment['constant_gamma'] and summary['missing_gamma'] == fragment['gamma'], 'legacy_gamma_changed')
            orders.add(summary['candidate_order_sha256'])
            current, endpoint_proofs = [], []
            for name, gamma in (('predictions.jsonl', fragment['constant_gamma']), ('missing_predictions.jsonl', fragment['gamma'])):
                path = root/name; endpoint = Path(summary['blend_dir'])/('gamma_'+str(float(gamma)))/name
                require(canonical(endpoint) in pins, 'legacy_endpoint_not_frozen')
                endpoint_sha = snapshot.pin(endpoint, pins[canonical(endpoint)])
                require(summary['inputs_sha256'].get(canonical(endpoint)) == endpoint_sha, 'legacy_endpoint_not_summary_bound')
                own_rows = snapshot.rows(path, cohort); endpoint_rows = snapshot.rows(endpoint, cohort)
                require(ranking_signature(own_rows, cohort) == ranking_signature(endpoint_rows, cohort), 'legacy_routed_output_differs_from_frozen_endpoint')
                current.extend(own_rows)
                endpoint_proofs.append(dict(path=canonical(endpoint), sha256=endpoint_sha, identity_rank_top10_exact=True))
            for role in ('no_aug', 'aug'):
                checkpoint = canonical(pair[role+'_checkpoint'])
                require(checkpoint == canonical(summary[role+'_checkpoint']) and checkpoint in pins, 'legacy_checkpoint_role_not_frozen')
                snapshot.pin(checkpoint, pins[checkpoint])
                require(summary[role+'_sha256'] == pins[checkpoint], 'legacy_checkpoint_SHA_unbound')
            digest = identity_digest(current, cohort)
            require(cohort not in rows_by_cohort or rows_by_cohort[cohort]['identity_sha256'] == digest, 'legacy_two_seed_validation_identity_changed')
            rows_by_cohort[cohort] = dict(identity_sha256=digest,
                counts={v: sum(r['variant'] == v for r in current) for v in VARIANTS},
                sources=sorted({r['source'] for r in current if r['variant'] == 'observed'}))
            entry = dict(seed=seed, cohort=cohort, adapter='legacy',
                predictions=[canonical(root/name) for name in ('predictions.jsonl', 'missing_predictions.jsonl')],
                summary=canonical(root/'summary.json'), exit_code=canonical(root/'EXIT_CODE'),
                seed_evidence=[dict(role=role, checkpoint=canonical(pair[role+'_checkpoint'])) for role in ('no_aug', 'aug')])
            required = entry['predictions'] + [entry['summary'], entry['exit_code']] + [e['checkpoint'] for e in entry['seed_evidence']]
            entry['inputs_sha256'] = {p: snapshot.inputs[p] for p in required}
            entry['completion_evidence'] = dict(kind='legacy_frozen_content_and_completion_upper_bound',
                completed_not_after_utc=finalization['ended_utc'], finalize_receipt=canonical(finalize_path),
                finalize_receipt_sha256=expected_finalize_sha, bundle_frozen_utc=bundle['frozen_utc'], freeze_bundle=canonical(folder/'freeze_bundle.json'),
                freeze_bundle_sha256=expected_bundle_sha, endpoint_proofs=endpoint_proofs)
            entries.append(entry); proofs.append(entry['completion_evidence'])
    require(len(orders) == 1, 'legacy_candidate_order_changed_between_owners')
    require(rows_by_cohort['original_val']['counts']['observed'] == 320 and rows_by_cohort['tune']['counts']['observed'] == 1024, 'legacy_normal_count_changed')
    require(all(len(value['sources']) == 5 for value in rows_by_cohort.values()) and rows_by_cohort['original_val']['sources'] == rows_by_cohort['tune']['sources'], 'legacy_source_population_changed')
    return entries, rows_by_cohort, orders.pop(), proofs


def job_graph(queue):
    jobs = {job['id']: job for job in queue['jobs']}
    require(len(jobs) == len(queue['jobs']), 'duplicate_queue_job')
    complete, active = set(), set()
    def visit(identifier):
        require(identifier in jobs, 'queue_dependency_missing:' + identifier)
        require(identifier not in active, 'queue_dependency_cycle')
        if identifier in complete: return
        active.add(identifier)
        for dep in jobs[identifier].get('deps', []): visit(dep)
        active.remove(identifier); complete.add(identifier)
    for identifier in jobs: visit(identifier)
    return jobs


def flag(argv, name):
    require(argv.count(name) == 1 and argv.index(name)+1 < len(argv), 'missing_or_duplicate_queue_flag:' + name)
    return argv[argv.index(name)+1]


def proof_jobs(proof):
    return {proof['job']} | set().union(*(proof_jobs(item) for item in proof.get('ancestors', [])))


def completed_job(identifier, jobs, run_root, queue_path, queue_sha, cutoff, snapshot, memo, queue_sources=None, earliest_start=None):
    if identifier in memo: return memo[identifier]
    job = jobs[identifier]; path = Path(run_root)/(identifier+'.receipt.json')
    if not path.is_file():
        result = dict(qualified=False, state='pending', reason='driver_job_receipt_missing', job=identifier)
    else:
        receipt = snapshot.json(path)
        if receipt.get('state') != 'complete':
            result = dict(qualified=False, state='failed' if receipt.get('state') in ('failed', 'blocked', 'deadline') else 'pending', reason='driver_job_not_complete', job=identifier, receipt_state=receipt.get('state'))
        else:
            require(type(receipt.get('external_exit')) is int and receipt['external_exit'] == 0 and
                    type(receipt.get('internal_exit')) is int and receipt['internal_exit'] == 0, 'driver_completion_has_no_dual_actual_EXIT0:' + identifier)
            require(receipt['argv'] == job['argv'] and canonical(receipt['output']) == canonical(job['out']) and
                    canonical(receipt['cwd']) == canonical(job['cwd']), 'driver_receipt_job_mismatch:' + identifier)
            require(receipt['queue_source_sha256'].get(canonical(queue_path)) == queue_sha, 'driver_receipt_queue_SHA_mismatch:' + identifier)
            require(all(receipt['queue_source_sha256'].get(p) == digest for p, digest in (queue_sources or {}).items()), 'driver_receipt_source_pin_mismatch:' + identifier)
            require(receipt.get('requested_inputs_sha256', {}) == job.get('inputs_sha256', {}), 'driver_requested_input_mismatch:' + identifier)
            finish = utc(receipt['finished_utc']); start = utc(receipt['started_utc'])
            require(start <= finish <= datetime.now(timezone.utc), 'driver_job_time_invalid:' + identifier)
            require(earliest_start is None or start >= earliest_start, 'driver_job_predates_actual_driver_launch:' + identifier)
            if finish > cutoff:
                result = dict(qualified=False, state='pending', reason='completed_after_cutoff', job=identifier, finished_utc=receipt['finished_utc'])
            else:
                exit_file = Path(job['out'])/'EXIT_CODE'; summary_file = Path(job['out'])/'summary.json'
                if not exit_file.is_file() or not summary_file.is_file():
                    result = dict(qualified=False, state='pending', reason='completed_receipt_output_artifact_missing', job=identifier)
                else:
                    snapshot.pin(exit_file)
                    require(isinstance(snapshot.json(summary_file), dict), 'completed_output_summary_not_JSON_object:' + identifier)
                    require(exit_file.read_text().strip() == '0', 'driver_receipt_disagrees_with_actual_output_EXIT:' + identifier)
                    for input_path, digest in job.get('inputs_sha256', {}).items(): snapshot.pin(input_path, digest)
                    ancestors = [completed_job(dep, jobs, run_root, queue_path, queue_sha, cutoff, snapshot, memo, queue_sources, earliest_start) for dep in job.get('deps', [])]
                    result = dict(qualified=all(item['qualified'] for item in ancestors), state='complete' if all(item['qualified'] for item in ancestors) else 'pending',
                        reason='actual_wait_and_output_EXIT0' if all(item['qualified'] for item in ancestors) else 'ancestor_not_qualified',
                        job=identifier, finished_utc=receipt['finished_utc'], receipt=canonical(path), ancestors=ancestors)
    memo[identifier] = result
    return result


def build(*, baseline_path, plan_path, queue_path, run_root, arms_path, legacy_freeze_dir,
          cutoff_utc, legacy_bundle_sha256=LEGACY_BUNDLE_SHA, legacy_receipt_sha256=LEGACY_RECEIPT_SHA,
          legacy_finalize_sha256=LEGACY_FINALIZE_SHA):
    cutoff = utc(cutoff_utc); snapshot = Snapshot(); snapshot.pin(__file__)
    baseline = snapshot.json(baseline_path); plan = snapshot.json(plan_path)
    queue = snapshot.json(queue_path); arms = snapshot.json(arms_path); queue_sha = snapshot.inputs[canonical(queue_path)]
    require(queue['schema'] == 'nmr_research_jobs_20261022' and plan['schema'] == 'nmr_research_execution_plan_20261022', 'formal_plan_or_queue_schema')
    require(plan['queue_sha256'] == queue_sha and plan['jobs'] == len(queue['jobs']), 'formal_plan_queue_mismatch')
    require(set(arms['seeds']) == set(SEEDS) and len(arms['families']) == 11, 'preregistered_population_contract')
    names = [family['name'] for family in arms['families']]
    require(len(set(names)) == 11 and 'incumbent_frozen' in names, 'preregistered_family_identity')
    jobs = job_graph(queue)
    require(all(not {'test', 'verify', 'sealed'}.intersection(Path(job['out']).parts) for job in jobs.values()), 'formal_queue_contains_sealed_output')
    old_entries, populations, candidate_order, legacy_proofs = legacy_entries(baseline, legacy_freeze_dir, cutoff, snapshot, legacy_bundle_sha256, legacy_receipt_sha256, legacy_finalize_sha256)
    executions = {}; incumbent = []
    for execution in plan['executions']:
        if execution['family'] == 'incumbent_frozen': incumbent.append(execution)
        else:
            key = (execution['family'], execution['seed'])
            require(key not in executions, 'duplicate_execution_family_seed')
            executions[key] = execution
    expected = {(name, seed) for name in names if name != 'incumbent_frozen' for seed in SEEDS}
    require(set(executions) == expected and len(incumbent) == 1, 'formal_executions_population_mismatch')
    require(incumbent[0]['frozen_incumbent'] == baseline['fragment']['pairs'], 'formal_incumbent_swapped')
    families = []; memo = {}; job_by_out = {canonical(job['out']): job for job in jobs.values()}
    require(len(job_by_out) == len(jobs), 'queue_output_directory_reused')
    driver_start_path = Path(run_root)/'driver_start.json'; launch_path = Path(run_root)/'launch_receipt.json'
    driver_available = driver_start_path.is_file() and launch_path.is_file()
    driver_evidence = dict(available=driver_available, overall_completion_required_for_one_family=False,
        final_driver_double_EXIT0=False, known_failed=False)
    earliest_start = None
    if driver_available:
        start = snapshot.json(driver_start_path); launch = snapshot.json(launch_path)
        require(start['inputs_sha256'].get(canonical(queue_path)) == queue_sha and launch['queue_sha256'].get(canonical(queue_path)) == queue_sha, 'formal_driver_not_bound_to_queue')
        require(canonical(flag(launch['argv'], '--out')) == canonical(run_root) and canonical(flag(launch['argv'], '--queue')) == canonical(queue_path), 'formal_driver_launch_arguments_mismatch')
        require(all(start['inputs_sha256'].get(p) == digest and launch['bundle_sha256'].get(p) == digest
                    for p, digest in queue['source_inputs_sha256'].items()), 'formal_driver_source_pins_mismatch')
        for input_path, digest in queue['source_inputs_sha256'].items(): snapshot.pin(input_path, digest)
        earliest_start = max(utc(start['utc']), utc(launch['started_utc']))
        driver_evidence.update(driver_start=canonical(driver_start_path), launch_receipt=canonical(launch_path), outer_external_exit=launch.get('external_exit'), outer_finished_utc=launch.get('finished_utc'))
        if launch.get('finished_utc'):
            codes = (launch.get('external_exit'), launch.get('internal_exit'))
            driver_evidence['known_failed'] = any(type(code) is not int or code != 0 for code in codes)
            if not driver_evidence['known_failed']:
                actual_exit_paths = [Path(run_root)/name for name in ('EXIT_CODE', 'EXTERNAL_EXIT_CODE')]
                if all(p.is_file() for p in actual_exit_paths):
                    for p in actual_exit_paths: snapshot.pin(p)
                    require(all(p.read_text().strip() == '0' for p in actual_exit_paths), 'formal_outer_receipt_actual_EXIT_mismatch')
                    driver_evidence['final_driver_double_EXIT0'] = True
    for registered in arms['families']:
        name = registered['name']; R = bool(registered.get('R', False)); O = bool(registered.get('O', False))
        family = dict(name=name, inference_cost=int(R)+2*int(O),
            inference_cost_breakdown=dict(common_query_base=0, extra_full_library_channel=int(R), extra_learned_observation_passes=2*int(O), offline_F_bank_cost_excluded=True))
        planned = []; qualification = []
        if name == 'incumbent_frozen':
            planned = copy.deepcopy(old_entries); qualification = [dict(qualified=True, kind='old_frozen_evidence', upper_bound_utc=p['completed_not_after_utc']) for p in legacy_proofs]
        else:
            for seed in SEEDS:
                execution = executions[(name, seed)]
                require(execution['recall'] == ('dual' if R else 'single') and execution['O'] is O and execution['training_size'] == registered['training_size'], 'execution_factor_or_budget_mismatch')
                require(set(execution['validation_outputs']) == set(COHORTS) and plan['profiles'][execution['profile']]['seed'] == seed, 'execution_cohort_or_seed_mismatch')
                expected_kind = 'local' if registered.get('local_slots') else 'physical' if registered.get('F') else 'inherited'
                require(plan['profiles'][execution['profile']]['kind'] == expected_kind, 'execution_forward_profile_mismatch')
                for cohort in COHORTS:
                    folder = Path(execution['validation_outputs'][cohort]); output = canonical(folder)
                    require(output in job_by_out, 'validation_output_has_no_queue_producer')
                    job = job_by_out[output]; argv = job['argv']
                    require(job['kind'] == 'validation' and flag(argv, '-m') == 'nmr_transfer.h_research_ranker' and
                        canonical(flag(argv, '--out')) == output and flag(argv, '--stage') == 'evaluate' and int(flag(argv, '--seed')) == seed and
                        canonical(flag(argv, '--no-aug-checkpoint')) == canonical(execution['no_aug_checkpoint']) and
                        canonical(flag(argv, '--aug-checkpoint')) == canonical(execution['aug_checkpoint']), 'validation_queue_role_or_seed_mismatch')
                    proof = (completed_job(job['id'], jobs, run_root, queue_path, queue_sha, cutoff, snapshot, memo, queue['source_inputs_sha256'], earliest_start)
                             if driver_available else dict(qualified=False, state='pending', reason='formal_driver_receipts_missing', job=job['id']))
                    if driver_evidence['known_failed']:
                        proof = dict(qualified=False, state='failed', reason='formal_driver_final_gate_failed', job=job['id'])
                    entry = dict(seed=seed, cohort=cohort, adapter='research', protocol=canonical(folder/'protocol.json'),
                        predictions=[canonical(folder/'predictions.jsonl')], summary=canonical(folder/'summary.json'), exit_code=canonical(folder/'EXIT_CODE'))
                    required_paths = entry['predictions'] + [entry['summary'], entry['exit_code'], entry['protocol'],
                        canonical(execution['no_aug_checkpoint']), canonical(execution['aug_checkpoint'])]
                    entry['inputs_sha256'] = {p: snapshot.pin(p) for p in required_paths if Path(p).is_file()}
                    if proof['qualified']:
                        require(all(Path(p).is_file() for p in required_paths), 'completed_validation_artifact_missing')
                        protocol = snapshot.json(entry['protocol']); summary = snapshot.json(entry['summary'])
                        require(protocol['args']['seed'] == seed and protocol['args']['stage'] == 'evaluate' and summary['cohort'] == cohort, 'validation_protocol_seed_or_cohort_mismatch')
                        require(summary['candidate_order_sha256'] == candidate_order, 'validation_candidate_order_changed')
                        for role in ('no_aug', 'aug'):
                            checkpoint = canonical(execution[role+'_checkpoint'])
                            require(canonical(protocol['args'][role+'_checkpoint']) == checkpoint and
                                summary['checkpoints'].get(checkpoint) == entry['inputs_sha256'][checkpoint] and
                                protocol['inputs_sha256'].get(checkpoint) == entry['inputs_sha256'][checkpoint], 'validation_checkpoint_not_producer_bound')
                        rows = snapshot.rows(entry['predictions'][0], cohort)
                        require(identity_digest(rows, cohort) == populations[cohort]['identity_sha256'], 'new_validation_identity_differs_from_incumbent')
                        timings = dict(scope='all_validation_query_variants', records=sum(populations[cohort]['counts'].values()),
                            evaluator={key: summary[key] for key in ('elapsed_seconds', 'reranking_seconds_mean', 'reranking_seconds_p95') if key in summary},
                            full_end_to_end_latency_available=False)
                        if '--cache' in argv:
                            cache = canonical(flag(argv, '--cache'))
                            require(canonical(protocol['args']['cache']) == cache and canonical(Path(cache).parent) in job_by_out,
                                'validation_timing_cache_not_queue_bound')
                            cache_job = job_by_out[canonical(Path(cache).parent)]
                            require(cache_job['id'] in proof_jobs(proof) and flag(cache_job['argv'], '-m') == 'nmr_transfer.h_research_extract',
                                'validation_timing_cache_not_qualified_extractor_ancestor')
                            cache_summary = snapshot.json(Path(cache).parent/'summary.json')
                            require(cache_summary['cohort'] == cohort, 'validation_timing_cache_cohort_mismatch')
                            timings['extractor'] = {key: cache_summary[key] for key in ('records', 'base_queries', 'extraction_seconds', 'matching_seconds_mean', 'matching_seconds_p95') if key in cache_summary}
                        entry['validation_timing_components'] = timings
                    entry['completion_evidence'] = proof; planned.append(entry); qualification.append(proof)
        qualified = all(proof['qualified'] for proof in qualification) and len(planned) == 4
        family.update(builder_state='complete' if qualified else 'pending', cutoff_qualified=qualified,
            planned_entries=planned, entries=copy.deepcopy(planned) if qualified else [], completion_evidence=qualification)
        families.append(family)
    snapshot.finish()
    manifest = dict(schema='nmr_research_results_20261022', scope='train_val_only', seeds=list(SEEDS), candidate_count=114982,
        preregistered_arms=canonical(arms_path), preregistered_arms_sha256=snapshot.inputs[canonical(arms_path)],
        incumbent='incumbent_frozen', candidate_order_sha256=candidate_order,
        expected_sources=populations['original_val']['sources'],
        expected_counts={cohort: populations[cohort]['counts'] for cohort in COHORTS},
        expected_query_identity_sha256={cohort: populations[cohort]['identity_sha256'] for cohort in COHORTS},
        families=families, cutoff_utc=cutoff.isoformat(), inference_cost_unit='fixed_incremental_query_work_v1_R1_O2_base0',
        inference_cost_definition='base common coarse/matching/twoMLP excluded; R adds one full-library channel unit; O adds two learned-observation matching-pass units; F offline bank processing excluded; not latency',
        builder_inputs_sha256=snapshot.inputs, builder_selection_provisional=not driver_evidence['final_driver_double_EXIT0'],
        all_families_cutoff_qualified=all(f['cutoff_qualified'] for f in families), controller_requirements=[
            'Recheck every builder_inputs_sha256 file before final selection/freeze.',
            'Require all11 families cutoff_qualified and results all_families_complete before formal freeze.',
            'Require the formal driver final internal/external EXIT0 and completed launch receipt before formal freeze.',
            'Do not move planned_entries into entries without a new builder snapshot and real pre-cutoff receipts.',
            'Keep pending/late/failed diagnostics; overall driver running does not establish all-family completion.',
            'Inference work units are static tie costs; actual end-to-end latency and offline F costs are separate.'],
        external_api_calls=0, test_or_verify_read=False)
    report = dict(cutoff_utc=cutoff.isoformat(), qualified_families=[f['name'] for f in families if f['cutoff_qualified']],
        pending_families=[f['name'] for f in families if not f['cutoff_qualified']],
        planned_entries=sum(len(f['planned_entries']) for f in families), active_entries=sum(len(f['entries']) for f in families),
        driver_evidence=driver_evidence, completed_job_proofs=memo, builder_inputs_sha256=snapshot.inputs,
        queue_jobs=len(jobs), family_seed_runs=len(executions)+2,
        all_families_cutoff_qualified=all(f['cutoff_qualified'] for f in families),
        historical_completion_uses_mtime=False, test_or_verify_read=False, external_api_calls=0)
    return manifest, report


def main():
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'plan', 'queue', 'run-root', 'arms', 'legacy-freeze-dir', 'cutoff-utc', 'out'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    try:
        manifest, report = build(baseline_path=args.baseline, plan_path=args.plan, queue_path=args.queue,
            run_root=args.run_root, arms_path=args.arms, legacy_freeze_dir=args.legacy_freeze_dir, cutoff_utc=args.cutoff_utc)
        for name, value in (('manifest.json', manifest), ('builder_report.json', report)):
            temporary = out/(name+'.tmp'); temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n'); temporary.replace(out/name)
        (out/'EXIT_CODE').write_text('0\n')
    except Exception:
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

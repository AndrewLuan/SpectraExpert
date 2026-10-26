"""功能概述：独立核对唯一 C10/C5 训练队列的协议、源码和元数据闭包。

只读取本轮文件及上一诊断已复制的 train/validation 小型元数据。张量 SHA
使用本轮有真实 SSH 退出的流式库存证据，不打开张量或旧 test 内容。
本工具不导入生产训练/派发模块，不执行模型、不调用 SSH、不派发任务。
输出模型实现、通用 operator 和具体 train admission 三份独立审查 JSON。
"""
import hashlib
import json
from pathlib import Path
import shlex
from datetime import datetime, timezone

M = Path(__file__).resolve().parent
N = M.parent / 'dual-tail-diagnosis-20261024T145544Z'
MR = '/home/user/IRfingerprints/SpectrExpert/outputs/nmr_cap10_controlled_20261024T163633Z'
SOURCE = '/home/user/IRfingerprints/SpectrExpert'
NR = SOURCE + '/outputs/nmr_pool_ranking_20261023'
seen = {}
checks = []


def digest(path):
    p = Path(path).resolve(strict=True)
    assert p.suffix not in ('.pt', '.pth', '.ckpt', '.safetensors')
    assert p.is_relative_to(M) or p.is_relative_to(N)
    raw = p.read_bytes()
    value = hashlib.sha256(raw).hexdigest()
    seen[str(p)] = value
    return value


def load(path):
    digest(path)
    return json.loads(Path(path).read_text())


def require(value, label):
    assert value, label
    checks.append(label)


def save(name, record):
    path = M / name
    with path.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(name, hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    q = load(M / 'train_queue.json')
    b = load(M / 'execution_bindings.json')
    p = load(M / 'protocol.json')
    disposition = load(M / 'planner_disposition.json')
    digest(M / 'planner_decision_message.txt')
    tests = load(M / 'implementation_synthetic_review_receipt.json')
    inventory = load(M / 'current_scientific_inputs.inventory.json')
    inventory_execution = load(M / 'current_scientific_inputs.inventory.execution.json')
    old = load(N / 'execution_bindings.json')
    mapping = load(N / 'collected/artifact_map.json')['files']
    raw_review = load(N / 'diagnosis_independent_review.json')
    require(raw_review['status'] == 'PASS' and raw_review['independent'] is True, 'prior_trainval_diagnostic_review')
    require(q['schema'] == 'cap10_controlled_queue_20261024' and q['phase'] == 'train', 'queue_train_only_schema')
    require(b['schema'] == 'cap10_execution_bindings_20261024' and p['schema'] == 'cap10_controlled_protocol_20261024', 'binding_protocol_schema')
    require(p['remote_root'] == q['root'] == MR and q['local_root'] == str(M), 'canonical_new_roots')
    require(p['planner_disposition']['sha256'] == digest(M / 'planner_disposition.json') and disposition['message']['sha256'] == digest(M / 'planner_decision_message.txt'), 'explicit_planner_authority_bytes')
    require(disposition['exposure_actor'] == 'implementation_session_root' and disposition['exposure_actor_is_original_planner'] is False and disposition['original_readiness_actual_exit'] == 1 and disposition['original_predicate_may_change'] is False, 'exposure_actor_and_prior_failure_preserved')
    require(p['test_blind'] is False and p['independent_generalization_claim_allowed'] is False and p['promotion'] is False and p['independent_acceptance_status'] == 'unavailable', 'controlled_research_not_blind_generalization')
    require(p['training_allowed'] is True and p['test_allowed'] is False and p['only_eligible_new_config'] == 'cap10_dual', 'train_admission_no_test_admission')
    required_budget = dict(fits=8, epochs=64, samples=8192, samples_per_epoch=8192, batch_size=32,
        lr=.001, weight_decay=.001, anchor=.1, temperature=.5, clip_grad_norm=2.,
        draws_per_fit=524288, optimizer_steps=16384, total_draws=4194304,
        total_optimizer_steps=131072, alpha=1.)
    require(all(p['budget'][k] == v and type(p['budget'][k]) is not bool for k, v in required_budget.items()), 'complete_equal_budget_optimizer_contract')
    require(p['final_epoch'] == 64 and p['alpha'] == 1 and p['epoch_offset'] == 0 and p['rng']['cap_enters_seed'] is False and p['rng']['paired_stream_hash_equality_required'] is True, 'final64_alpha1_offset0_paired_RNG')
    require(p['budget']['early_stopping'] is False and p['budget']['cap_grid_allowed'] is False and p['budget']['new_forward_bank_extractions'] == 0, 'no_grid_checkpoint_search_or_forward_extraction')
    require(tests['actual_exit'] == 0 and type(tests['actual_exit']) is int and tests['tests'] == 49 and tests['inputs_unchanged'] is True and tests['source_before_sha256'] == tests['source_after_sha256'], '49_local_actual_tests_with_source_begin_end')
    for path, value in tests['source_after_sha256'].items():
        require(digest(path) == value, 'tested_source:' + Path(path).name)
    require(type(inventory_execution['ssh_exit']) is int and inventory_execution['ssh_exit'] == 0 and inventory['status'] == 'PASS' and inventory['count'] == 69 and inventory['tensor_values_loaded'] is False and inventory['model_jobs_started'] == 0, 'fresh_remote_69_streamhash_evidence')
    remote_argv = shlex.split(inventory_execution['argv'][-1])
    require(json.loads(remote_argv[-1]) == inventory['input_sha256'] == old['input_sha256'], 'remote_invocation_exact_69_requested_pins')
    require(b['caches'] == old['caches'] and b['cohort_contract'] == old['cohort_contract'], 'same_twelve_cache_identities_and_cohort_contract')
    for seed, roles in old['roles'].items():
        for role, original in roles.items():
            current = b['roles'][seed][role]
            require(all(current[k] == v for k, v in original.items()) and set(current) - set(original) == {'center_rows_sha256', 'producer_summary_sha256'}, 'warm_role_values_preserved_with_two_explicit_SHA_fields')
            require(current['center_rows_sha256'] == inventory['input_sha256'][current['center_rows']] and current['producer_summary_sha256'] == inventory['input_sha256'][current['producer_summary']], 'additional_role_SHA_fields_match_actual_inventory')
    require(q['input_sha256'] == b['input_sha256'] and len(q['input_sha256']) == 76 and len(q['source_sha256']) == 16, 'exact_76_input_16_source_closure')
    require(all(q['input_sha256'].get(k) == v for k, v in inventory['input_sha256'].items()), 'prior_69_unchanged_subset')
    additions = {'cap_ranker.py', 'train_cap_ranker.py', 'protocol.json', 'planner_disposition.json',
                 'planner_decision_message.txt', 'fixed_train_pairs.json', 'baseline_validation_reference.json'}
    require(set(q['input_sha256']) - set(inventory['input_sha256']) == {MR + '/' + x for x in additions}, 'only_seven_declared_new_inputs')
    reviewed = dict(q['input_sha256'])
    reviewed.update(q['source_sha256'])
    reviewed[MR + '/train_queue.json'] = digest(M / 'train_queue.json')
    reviewed[MR + '/execution_bindings.json'] = digest(M / 'execution_bindings.json')
    for remote, value in reviewed.items():
        require('/sealed' not in remote and '/test/' not in remote and Path(remote).name not in ('cohorts.pt', 'test.pt', 'predictions.jsonl'), 'no_test_input:' + Path(remote).name)
        if remote.startswith(MR + '/'):
            local = M / Path(remote).relative_to(MR)
            require(digest(local) == value, 'new_local_bytes:' + local.name)
        elif not remote.endswith(('.pt', '.pth', '.ckpt')):
            local = Path(mapping[remote])
            require(digest(local) == value, 'old_copied_metadata:' + Path(remote).name)
        else:
            require(inventory['input_sha256'][remote] == value, 'tensor_SHA_from_remote_stream_inventory')
    require(all(q['source_sha256'].get(k) == v for k, v in b['source_sha256'].items()), 'binding_full_science_source_values_equal_queue')
    mandatory = {MR + '/' + x for x in ('protocol.json', 'execution_bindings.json', 'isolation_entry.py',
                 'supervise_controlled_queue.py', 'launch_controlled_stage.py', 'cap_ranker.py', 'train_cap_ranker.py')}
    require(set(q['source_sha256']) == set(b['source_sha256']) | mandatory, 'source_full_mandatory_exact_set')
    require(q['review_paths'] == [MR + '/train_admission_independent_review.json'] and q['bindings'] == MR + '/execution_bindings.json', 'unique_review_binding_paths')
    require(q['python'] == '/home/user/.venvs/spectrexpert/bin/python' and q['max_parallel'] == 8 and q['maximum_seconds'] == 10800, 'runtime_parallel_budget')
    matrix = set()
    for job in q['jobs']:
        argv = job['command']
        values = dict(zip(argv[1::2], argv[2::2]))
        cap, seed, role = int(values['--cap']), int(values['--seed']), values['--role']
        matrix.add((cap, seed, role))
        output = MR + f'/rankers/C{cap}/s{seed}/{role}'
        expected = [MR + '/train_cap_ranker.py', '--source', SOURCE, '--fixed-module', NR + '/h_pool_fixed_center_ranker.py',
                    '--protocol', MR + '/protocol.json', '--bindings', MR + '/execution_bindings.json',
                    '--cap', str(cap), '--seed', str(seed), '--role', role, '--device', 'cuda:0', '--out', output]
        require(argv == expected and job['id'] == f'fit_C{cap}_s{seed}_{role}' and job['job_out'] == output and job['cwd'] == SOURCE, 'exact_fit_argv:' + job['id'])
        expected_reads = [b['roles'][str(seed)][role]['checkpoint']] + [b['caches'][str(seed)]['train'][pool]['path'] for pool in ('single', 'dual')]
        require(job['allowed_reads'] == expected_reads and all(k in q['input_sha256'] for k in expected_reads), 'only_role_best_plus_two_train_tensors:' + job['id'])
        expected_outer = [q['python'], '-B', MR + '/isolation_entry.py', '--bindings', q['bindings'], '--phase', 'train',
            '--access-log', job['access_log'], '--output-root', output]
        for path in expected_reads:
            expected_outer += ['--allow-read', path]
        require(job['argv'] == expected_outer + ['--'] + expected and job['access_log'] == MR + '/access_logs/' + job['id'] + '.json', 'exact_isolation_argv:' + job['id'])
    require(matrix == {(c, s, r) for c in (5, 10) for s in (193, 211) for r in ('no_aug', 'aug')} and len(q['jobs']) == 8, 'complete_unique_eight_fit_matrix')
    require(sorted(j['gpu'] for j in q['jobs']) == list(range(8)) and all(type(j['gpu']) is int for j in q['jobs']), 'eight_unique_physical_GPUs_fresh_idle_runtime_gate')
    for seed in ('193', '211'):
        for pool in ('single', 'dual'):
            cache = b['caches'][seed]['train'][pool]
            require(cache['manifest']['cohort'] == 'train' and cache['manifest']['recall'] == pool and cache['sha256'] == inventory['input_sha256'][cache['path']], 'bound_training_cache:' + seed + pool)
            producer = load(mapping[cache['producer']])
            require(producer == cache['manifest'] and producer['feature_cache_sha256'] == cache['sha256'], 'actual_old_producer_schema:' + seed + pool)
            require(Path(mapping[str(Path(cache['producer']).parent / 'EXIT_CODE')]).read_text().strip() == '0', 'cache_producer_exit0')
        for role in ('no_aug', 'aug'):
            record = b['roles'][seed][role]
            center = load(mapping[record['centers']])
            producer = load(mapping[record['producer_summary']])
            require(record['checkpoint_sha256'] == inventory['input_sha256'][record['checkpoint']] and center['center33'] == record['center33'] and center['O_center3'] == record['O_center3'], 'fixed_role_tensor_and_36center_identity:' + seed + role)
            require(center['row_audit_sha256'] == inventory['input_sha256'][record['center_rows']] and producer['best'] == record['best'] and record['best']['alpha'] == 1, 'role_producer_and_centerrows_closure:' + seed + role)
            require(record['qualification']['candidate_order_sha256'] == b['candidate_order_sha256'] and record['qualification']['historical_heldout_sha256'] == b['heldout_groups_sha256'], 'candidate_order_and_heldout_identity')
    reference = load(M / 'baseline_validation_reference.json')
    require(p['validation']['baseline_signals'] == {k: v['value'] for k, v in reference['signals'].items()}, 'exact_five_validation_reference_signals')
    require(p['validation']['guards'] == ['top1', 'frequency_crop_MRR', 'peak_dropout_MRR', 'macro_MRR'] and p['validation']['test_outcomes_in_selection'] is False, 'validation_guards_and_no_test_selection')
    require(p['test_preregistration']['matrix_cells'] == 12 and len(p['test_preregistration']['comparisons']) == 10 and p['test_preregistration']['comparison_strata'] == 180 and 'No partial reuse' in p['test_preregistration']['baseline_reuse_policy'], 'prospective_test_matrix_only_all_or_none')
    common = dict(schema='cap10_independent_implementation_review_20261025', status='PASS', independent=True,
        utc=datetime.now(timezone.utc).isoformat(), synthetic_actual_exit=0, synthetic_tests=49,
        synthetic_receipt_sha256=digest(M / 'implementation_synthetic_review_receipt.json'),
        reviewed_sha256=dict(seen), actual_training_started=False, actual_test_data_opened=False,
        boundary='Only source and train/validation metadata reviewed. Python open wrappers are not an OS/C++ I/O sandbox; all executed commands must remain exact-reviewed. No claim of test blindness or independent generalization.')
    for name, scope, keys in [
        ('operator_independent_review.json', 'generic_controlled_operator_implementation_not_specific_phase_admission', ('isolation_entry.py', 'supervise_controlled_queue.py', 'launch_controlled_stage.py', 'test_controlled_operator_independent.py')),
        ('cap_training_implementation_independent_review.json', 'cap_aware_model_and_matched_continuation_implementation', ('cap_ranker.py', 'train_cap_ranker.py', 'test_cap_ranker.py', 'test_cap_training_independent.py'))]:
        record = dict(common, scope=scope)
        record['reviewed_sha256'] = {str(M / k): digest(M / k) for k in keys}
        record['reviewed_sha256'].update({MR + '/' + k: digest(M / k) for k in keys})
        record['checks'] = checks
        save(name, record)
    record = dict(common, schema='cap10_train_admission_independent_review_20261025', admission_phase='train',
        scope='exact_eight_fit_protocol_queue_sources_and_76_input_metadata_closure',
        queue_sha256=digest(M / 'train_queue.json'), protocol_sha256=digest(M / 'protocol.json'),
        bindings_sha256=digest(M / 'execution_bindings.json'), scientific_inputs_sha256=q['input_sha256'],
        scientific_inputs_count=76, runtime_source_count=16, checks=checks,
        admitted_matrix=sorted(matrix), tensor_values_loaded_in_review=False,
        remote_tensor_SHA_verification='Fresh actual SSH0 streamed inventory of69 inputs; reviewer never opens tensor payloads.',
        phase_training_dispatch_after_root_source_only_deployment_and_fresh_idle_gate=True,
        validation_and_test_dispatch_admitted=False)
    record['reviewed_sha256'] = dict(seen, **reviewed)
    for path, value in list(seen.items()):
        require(digest(path) == value, 'review_input_unchanged:' + Path(path).name)
    record['check_count'] = len(checks)
    record['reviewed_sha256'][str(Path(__file__))] = digest(Path(__file__))
    save('train_admission_independent_review.json', record)


if __name__ == '__main__':
    main()

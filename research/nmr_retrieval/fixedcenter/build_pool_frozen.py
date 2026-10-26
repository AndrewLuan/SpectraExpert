"""功能概述：在完整验证与独立审核通过后，生成候选排序研究的冻结清单和封存测试队列。

build只消费summarize_pool_validation的train/val选型、48/12任务双退出证据、训练审计、
资格审计及策划者显式test-plan；不决定测试范围，不运行模型，不打开测试cache或预测。
validate_plan检查八主配置与直接基准的双seed、固定114982候选与1000/200/171身份，
legacy_cache_registry从旧frozen/selection.json绑定其封存队列SHA与四个继承extract job，
cache_references绑定每seed验证producer和真实sealed/features路径；cache_info仅读producer summary/EXIT，
同时核对forward checkpoint身份，模型SHA取自验证摘要并实际复核。测试cache及
其科学来源SHA使用producer声明，留给h_research_driver启动前逐字节复核；该边界写入
输出receipt。read_checked/review/check_audit均失败即停；merge拒绝冲突指纹。
main独占新目录，写满足h_pool_ranker.check_frozen的frozen_selection.json、sealed_queue.json、
构建receipt和内部EXIT_CODE。冻结绑定当前工具、审核、计划和输入SHA，保留唯一验证
preselected，独立验收固定unavailable且promotion=false。无网络、部署或真实测试副作用。
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import summarize_pool_validation as validation

SCHEMA = 'nmr_pool_ranker_20261023'
PLAN_SCHEMA = 'pool_sealed_test_plan_20261023'
VARIANTS = {'observed': 1000, 'frequency_crop': 200, 'peak_dropout': 171}
SOURCE_NAMES = ('h_pool_ranker.py', 'h_research_ranker.py', 'h_research_observation.py',
                'h_research_contract.py', 'h_global_ranker.py', 'h_global_features.py',
                'h_research_driver.py')
AUDIT_CHECKS = ('same_initial_tensors', 'frozen_observation', 'matched_exposure', 'finite', 'full_budget')


def sha(path):
    return validation.sha(path)


def merge(target, incoming):
    for path, digest in incoming.items():
        if (not Path(path).is_absolute() or str(Path(path).resolve()) != path
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('invalid_absolute_input_pin:' + str(path))
        if path in target and target[path] != digest:
            raise ValueError('conflicting_input_pin:' + path)
        target[path] = digest


def read_checked(path, pins):
    return validation.read_json(path, pins)


def exit0(path, pins):
    validation.exit0(path, pins)


def checked_pins(expected, pins):
    merge({}, expected)
    validation.require_sha(expected, pins)


def review(record, required):
    if (record.get('status') != 'PASS' or record.get('independent') is not True
            or not record.get('reviewer') or record.get('scope') != 'freeze_plan_and_implementation'):
        raise ValueError('independent_freeze_review_required')
    reviewed = record.get('reviewed_sha256', {})
    if any(reviewed.get(path) != digest for path, digest in required.items()):
        raise ValueError('review_does_not_bind_current_inputs')


def check_audit(audit, formal_jobs, formal_queue_sha, pins):
    if (audit.get('status') != 'PASS' or audit.get('formal_queue_sha256') != formal_queue_sha
            or any(audit.get('checks', {}).get(k) is not True for k in AUDIT_CHECKS)):
        raise ValueError('complete_training_audit_required')
    inputs = audit.get('input_sha256', {})
    producers = {str(Path(j['out']).resolve()/name)
                 for j in formal_jobs.values() if j['kind'] == 'training'
                 for name in ('summary.json', 'EXIT_CODE')}
    if len(producers) != 32 or not producers <= inputs.keys():
        raise ValueError('training_audit_missing_fit_producers')
    checked_pins(inputs, pins)


def weights_for(summary, config, seed):
    cohorts = summary['metadata'][config][str(seed)]
    first = cohorts['original_val']['checkpoints']
    if first != cohorts['tune']['checkpoints'] or len(first) != 2:
        raise ValueError('validation_checkpoint_pair_mismatch')
    roles = {}
    for path, digest in first.items():
        role = Path(path).parent.name
        if role not in ('no_aug', 'aug') or role in roles:
            raise ValueError('checkpoint_role_ambiguous')
        roles[role] = (path, digest)
    if set(roles) != {'no_aug', 'aug'}:
        raise ValueError('checkpoint_pair_incomplete')
    return roles


def legacy_cache_registry(root, pins):
    old = root.parent/'nmr_global_retrieval_20261022'
    frozen_path = old/'frozen/selection.json'
    frozen = read_checked(frozen_path, pins)
    if (frozen.get('schema') != 'nmr_research_20261022'
            or frozen.get('selection_scope') != 'train_val_only'
            or frozen.get('candidate_count') != 114982 or not frozen.get('frozen_utc')
            or frozen.get('selection_target') != str(frozen_path)):
        raise ValueError('legacy_frozen_registration_invalid')
    queue_path = Path(frozen['sealed_queue_path']).resolve()
    if (str(queue_path) != frozen['sealed_queue_path'] or queue_path.name != 'sealed_queue.json'
            or not queue_path.is_relative_to(old/'seal_preparations')):
        raise ValueError('legacy_sealed_queue_path_invalid')
    queue = read_checked(queue_path, pins)
    if pins[str(queue_path)] != frozen.get('sealed_queue_sha256'):
        raise ValueError('legacy_sealed_queue_sha_changed')
    jobs = queue['jobs']; by_id = {job['id']: job for job in jobs}
    if queue.get('schema') != 'nmr_research_jobs_20261022' or len(by_id) != len(jobs):
        raise ValueError('legacy_sealed_queue_invalid')
    result = {}
    for seed in validation.SEEDS:
        for pool in ('single', 'dual'):
            name = f'extract_inherited_s{seed}_{pool}_test'
            job = by_id.get(name)
            folder = old/'sealed/features'/f'inherited_s{seed}'/pool
            if job is None or job.get('kind') != 'sealed' or job.get('out') != str(folder):
                raise ValueError('legacy_extract_output_registration_mismatch')
            argv = job['argv']; options = {}; index = 3
            if argv[:3] != ['/home/user/.venvs/spectrexpert/bin/python', '-m', 'nmr_transfer.h_research_extract']:
                raise ValueError('legacy_extract_command_mismatch')
            while index < len(argv):
                key = argv[index]
                if not key.startswith('--') or key in options:
                    raise ValueError('legacy_extract_option_ambiguous')
                if key == '--augment':
                    options[key] = True; index += 1
                elif index+1 < len(argv) and not argv[index+1].startswith('--'):
                    options[key] = argv[index+1]; index += 2
                else:
                    raise ValueError('legacy_extract_option_invalid')
            bank = str(old/'banks'/f'inherited_s{seed}'/'bank.pt')
            expected = {'--bank': bank, '--cohort': 'test', '--recall': pool, '--augment': True,
                        '--out': str(folder), '--frozen-selection': str(frozen_path),
                        '--cohorts': str(old/'cohorts'),
                        '--fixed-identities': str(old/'audit/fixed_evaluation_identities.json')}
            if any(options.get(key) != value for key, value in expected.items()):
                raise ValueError('legacy_extract_scientific_registration_mismatch')
            allowed = dict(bank=bank, extension=None, recall=pool, augment=True, pool_size=512, cohort='test')
            if allowed not in frozen.get('allowed_extract_configs', []):
                raise ValueError('legacy_extract_not_frozen')
            result[(seed, pool)] = dict(cache=str(folder/'features.pt'))
    return result


def cache_references(root, summary, pins, registry):
    old = root.parent/'nmr_global_retrieval_20261022'
    references = {}
    for config in list(validation.CONFIGS)+['frozen_single', 'frozen_dual']:
        family, pool = config.rsplit('_', 1)
        base = root/'diagnostics/frozen' if family == 'frozen' else root/'formal'/family
        for seed in validation.SEEDS:
            reference = None
            for cohort in validation.COHORTS:
                protocol_path = base/f's{seed}'/pool/cohort/'protocol.json'
                protocol = read_checked(protocol_path, pins)
                if summary['input_sha256'].get(str(protocol_path)) != pins[str(protocol_path)]:
                    raise ValueError('validation_cache_protocol_not_selection_bound')
                args = protocol['args']; cache = Path(args['cache']).resolve()
                expected_cache = old/'features'/f'inherited_s{seed}'/pool/cohort/'features.pt'
                if (args.get('seed') != seed or args.get('stage') != 'evaluate'
                        or cache != expected_cache or str(cache) != args['cache']):
                    raise ValueError('validation_cache_seed_or_cohort_mismatch')
                producer_path = cache.parent/'summary.json'
                producer = read_checked(producer_path, pins)
                if summary['input_sha256'].get(str(producer_path)) != pins[str(producer_path)]:
                    raise ValueError('validation_cache_producer_not_selection_bound')
                digest = producer.get('checkpoint_sha256')
                if (producer.get('cohort') != cohort or producer.get('recall') != pool
                        or not isinstance(digest, str) or len(digest) != 64
                        or any(c not in '0123456789abcdef' for c in digest)):
                    raise ValueError('validation_cache_scientific_identity_missing')
                value = dict(cache=registry[(seed, pool)]['cache'], checkpoint_sha256=digest)
                if reference is not None and reference != value:
                    raise ValueError('validation_cache_cohort_science_disagrees')
                reference = value
            references[(config, seed)] = reference
    return references


def cache_info(entry, identity, candidate_sha, pins, deferred, reference):
    cache = Path(entry['cache']).resolve()
    if str(cache) != entry['cache'] or cache.name != 'features.pt' or not cache.is_file():
        raise ValueError('noncanonical_or_missing_cache')
    if str(cache) != reference['cache']:
        raise ValueError('test_cache_seed_or_inherited_path_mismatch')
    folder = cache.parent
    exit0(folder/'EXIT_CODE', pins)
    producer = read_checked(folder/'summary.json', pins)
    expected = dict(cohort='test', recall=entry['config'].rsplit('_', 1)[1],
                    candidate_count=114982, pool_size=512, variants=VARIANTS,
                    base_queries=1000, records=1371, appended_training_positives=0,
                    candidate_order_sha256=candidate_sha)
    expected.update(identity)
    expected['checkpoint_sha256'] = reference['checkpoint_sha256']
    if any(producer.get(k) != value for k, value in expected.items()):
        raise ValueError('test_cache_identity_or_scope_mismatch')
    if producer.get('feature_cache_sha256') != entry['cache_sha256']:
        raise ValueError('test_cache_plan_sha_mismatch')
    # 仅producer身份；不反序列化cache，也不逐字节读取其中的测试谱值。
    merge(deferred, {str(cache): entry['cache_sha256']})
    science = producer.get('source_inputs_sha256', {})
    if not science:
        raise ValueError('test_cache_missing_scientific_provenance')
    merge(deferred, science)
    return producer


def validate_plan(plan, summary, candidate_sha, pins, deferred, references):
    if (plan.get('schema') != PLAN_SCHEMA or plan.get('selection_scope') != 'train_val_only'
            or plan.get('preselected') != summary['selection']['preselected']
            or plan.get('purpose') != 'sealed_test_ablation_only'
            or plan.get('promotion') is not False):
        raise ValueError('explicit_frozen_test_plan_required')
    if plan.get('python') != '/home/user/.venvs/spectrexpert/bin/python':
        raise ValueError('test_python_must_match_protocol')
    identity = plan.get('fixed_test_identity', {})
    if set(identity) != {'query_sha256', 'identity_content_sha256', 'variant_mask_content_sha256'}:
        raise ValueError('fixed_test_identity_required')
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
           for v in identity.values()):
        raise ValueError('invalid_test_identity_digest')
    entries = plan.get('evaluations', [])
    configs = set(validation.CONFIGS) | {'frozen_single'}
    requested = {e['config'] for e in entries}
    if requested not in (configs, configs | {'frozen_dual'}):
        raise ValueError('complete_main_matrix_and_direct_baseline_required')
    keys = [(e['config'], e['seed']) for e in entries]
    if (len(keys) != len(set(keys)) or set(keys) != {(c, s) for c in requested for s in validation.SEEDS}):
        raise ValueError('test_plan_incomplete_seed_matrix')
    for entry in entries:
        cache_info(entry, identity, candidate_sha, pins, deferred, references[(entry['config'], entry['seed'])])
        weights = weights_for(summary, entry['config'], entry['seed'])
        for role in ('no_aug', 'aug'):
            path, digest = weights[role]
            checked_pins({path: digest}, pins)
            exit0(Path(path).parent/'EXIT_CODE', pins)
            read_checked(Path(path).parent/'summary.json', pins)
    return entries


def build(args, out):
    root = Path(args.root).resolve(); source = Path(args.source).resolve()
    pins = {}; deferred = {}
    summary_path = Path(args.validation_summary).resolve()
    summary = read_checked(summary_path, pins)
    if sha(summary_path) != args.summary_sha256:
        raise ValueError('validation_summary_sha_mismatch')
    exit0(summary_path.parent/'EXIT_CODE', pins)
    if (summary.get('schema') != validation.SCHEMA
            or summary.get('selection') != validation.choose(summary.get('signals', {}))
            or summary['selection']['promotion'] is not False):
        raise ValueError('validation_selection_not_complete_or_reproducible')
    checked_pins(summary['source_sha256'], pins)
    if summary['source_sha256'].get(str(Path(validation.__file__).resolve())) != sha(validation.__file__):
        raise ValueError('validation_summarizer_source_not_bound')
    # verify_run仅处理train/val及诊断任务；再次要求真实内外EXIT0和完整48/12矩阵。
    formal = validation.verify_run(args.formal_run or root/'runs/formal', 'formal', pins, {})
    validation.verify_run(args.diagnostics_run or root/'runs/diagnostics', 'diagnostics', pins, {})
    if not all(path in summary['input_sha256'] and summary['input_sha256'][path] == digest
               for path, digest in pins.items() if path != str(summary_path)
               and path != str(summary_path.parent/'EXIT_CODE')
               and path not in summary['source_sha256']):
        raise ValueError('validation_summary_missing_current_run_evidence')
    registration = read_checked(args.preregistration, pins)
    if ({f['family'] for f in registration['families']} != set(validation.CONFIGS)
            or registration['seeds'] != list(validation.SEEDS)):
        raise ValueError('preregistration_matrix_mismatch')
    eligibility = read_checked(args.eligibility, pins)
    if (eligibility.get('status') != 'COMPLETE_FIXED_CANDIDATE_ELIGIBILITY_UNAVAILABLE'
            or eligibility.get('qualified_independent_structures') != 0
            or eligibility.get('promotion_allowed') is not False
            or eligibility.get('candidate_count') != 114982):
        raise ValueError('expected_unavailable_eligibility_audit')
    candidate_sha = eligibility['candidate_order_sha256']
    if any(meta['candidate_order_sha256'] != candidate_sha
           for seeds in summary['metadata'].values() for cohorts in seeds.values() for meta in cohorts.values()):
        raise ValueError('validation_candidate_library_changed')
    audit = read_checked(args.train_audit, pins)
    formal_launch = read_checked(Path(args.formal_run or root/'runs/formal','launch_receipt.json'), pins)
    check_audit(audit, formal, next(iter(formal_launch['queue_sha256'].values())), pins)
    plan = read_checked(args.test_plan, pins)
    legacy_pins = {}
    registry = legacy_cache_registry(root, legacy_pins); merge(pins, legacy_pins)
    references = cache_references(root, summary, pins, registry)
    entries = validate_plan(plan, summary, candidate_sha, pins, deferred, references)
    sources = {str(source/'nmr_transfer'/name): sha(source/'nmr_transfer'/name) for name in SOURCE_NAMES}
    # 明确测试运行使用的同一科学源码；计划审核必须绑定当前pin。
    if plan.get('source_sha256') != sources:
        raise ValueError('test_plan_scientific_source_changed')
    checked_pins(sources, pins)
    tool_sources = {str(Path(__file__).resolve()): sha(__file__),
                    str(Path(validation.__file__).resolve()): sha(validation.__file__),
                    str(Path(args.test_summarizer).resolve()): sha(args.test_summarizer)}
    checked_pins(tool_sources, pins)
    approval = read_checked(args.review, pins)
    review_required = {str(Path(p).resolve()): sha(p) for p in
                       (summary_path, args.test_plan, args.train_audit, args.eligibility, args.preregistration)}
    review_required.update(tool_sources); review_required.update(sources)
    review_required.update(legacy_pins)
    review(approval, review_required)
    frozen_now = dt.datetime.now(dt.timezone.utc)
    if frozen_now > dt.datetime(2026, 10, 23, 14, 35, 41, tzinfo=dt.timezone.utc):
        raise ValueError('freeze_deadline_passed')
    end = dt.datetime.fromisoformat(plan['end_utc'].replace('Z', '+00:00'))
    if (end.tzinfo is None or end > dt.datetime(2026, 10, 23, 16, 35, 41, tzinfo=dt.timezone.utc)
            or end <= frozen_now):
        raise ValueError('test_deadline_outside_protocol_or_expired')
    evaluation_pins = {}; merge(evaluation_pins, pins); merge(evaluation_pins, deferred)
    frozen = dict(schema=SCHEMA, frozen_utc=frozen_now.isoformat(),
        selection_scope='train_val_only', preselected=summary['selection']['preselected'],
        baseline='frozen_single', verify_eligibility='unavailable', promotion=False,
        independent_acceptance_status='unavailable', allowed_ranker_configs=[], ranker_configurations={},
        candidate_count=114982, candidate_order_sha256=candidate_sha, test_variants=VARIANTS,
        fixed_test_identity=plan['fixed_test_identity'], configurations=sorted({e['config'] for e in entries}),
        evaluation_inputs_sha256=evaluation_pins, tool_sha256=tool_sources,
        validation_summary_sha256=args.summary_sha256, reviewed_inputs_sha256=review_required,
        validation_summary_path=str(summary_path), test_plan_path=str(Path(args.test_plan).resolve()),
        freeze_review=dict(path=str(Path(args.review).resolve()), sha256=pins[str(Path(args.review).resolve())]))
    frozen_path = out/'frozen_selection.json'; jobs = []
    for entry in sorted(entries, key=lambda e: (e['config'], e['seed'])):
        config, seed = entry['config'], entry['seed']; weights = weights_for(summary, config, seed)
        allowed = dict(no_aug_checkpoint=weights['no_aug'][0], aug_checkpoint=weights['aug'][0],
                       seed=seed, cohort='test', recall=config.rsplit('_', 1)[1], cache_sha256=entry['cache_sha256'])
        if allowed in frozen['allowed_ranker_configs']:
            raise ValueError('duplicate_effective_ranker_configuration')
        frozen['allowed_ranker_configs'].append(allowed)
        frozen['ranker_configurations'].setdefault(config, {})[str(seed)] = allowed
        output = root/'sealed'/config/f's{seed}'/'test'
        if output.exists():
            raise ValueError('sealed_output_already_exists:' + str(output))
        argv = [plan['python'], '-m', 'nmr_transfer.h_pool_ranker', '--stage', 'evaluate',
                '--cache', entry['cache'], '--no-aug-checkpoint', weights['no_aug'][0],
                '--aug-checkpoint', weights['aug'][0], '--seed', str(seed), '--out', str(output),
                '--frozen-selection', str(frozen_path)]
        direct_paths = [str(source/'nmr_transfer'/n) for n in SOURCE_NAMES[:-1]]
        direct_paths += [entry['cache']]
        direct_paths += [str(Path(entry['cache']).parent/name) for name in ('summary.json', 'EXIT_CODE')]
        for role in ('no_aug', 'aug'):
            direct_paths.append(weights[role][0])
            direct_paths += [str(Path(weights[role][0]).parent/name) for name in ('summary.json', 'EXIT_CODE')]
        job_pins = {p: evaluation_pins[p] for p in direct_paths}
        jobs.append(dict(id=f'sealed_{config}_s{seed}', kind='sealed', gpu=True, argv=argv,
                         cwd=str(source), out=str(output), deps=[], inputs_sha256=job_pins))
    frozen_path.write_text(json.dumps(frozen, indent=2)+'\n')
    bundle = dict(evaluation_pins); merge(bundle, {str(frozen_path): sha(frozen_path)})
    for job in jobs:
        merge(job['inputs_sha256'], {str(frozen_path): sha(frozen_path)})
    queue = dict(schema='nmr_research_jobs_20261022', end_utc=plan['end_utc'],
                 source_inputs_sha256=bundle, jobs=jobs)
    queue_path = out/'sealed_queue.json'; queue_path.write_text(json.dumps(queue, indent=2)+'\n')
    checked_pins(pins, {})
    receipt = dict(schema='pool_freeze_build_20261023', status='BUILT_NOT_EXECUTED', jobs=len(jobs),
        preselected=frozen['preselected'], promotion=False, independent_acceptance_status='unavailable',
        input_sha256=pins, deferred_to_sealed_driver_sha256=deferred, tool_sha256=tool_sources,
        output_sha256={str(p): sha(p) for p in (frozen_path, queue_path)}, external_api_calls=0,
        test_cache_values_read=False, test_predictions_read=False)
    (out/'build_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'source', 'validation-summary', 'summary-sha256', 'test-plan', 'review',
                 'train-audit', 'eligibility', 'preregistration', 'test-summarizer', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--formal-run'); p.add_argument('--diagnostics-run')
    args = p.parse_args(); out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    try:
        receipt = build(args, out); (out/'EXIT_CODE').write_text('0\n')
        print(json.dumps(dict(status=receipt['status'], jobs=receipt['jobs'], output_sha256=receipt['output_sha256'])))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc)))+'\n')
        (out/'EXIT_CODE').write_text('1\n'); raise


if __name__ == '__main__':
    main()

"""功能概述：用临时合成证据验证build_pool_frozen的冻结边界与driver兼容性。
Fixture生成48正式任务和12诊断任务的真实文件契约、验证选型、16 fit训练审计、
测试cache producer身份及独立审核，完全不使用真实数据或运行训练/测试。
FrozenTests覆盖八配置双seed、直接基准、审核SHA、预算审计、双退出、资格不可用、
测试身份、模型与缓存pin、重入输出和冻结allowlist；open_guard证明构建不读取测试
cache字节或predictions。make/build_case辅助函数只写临时目录，清理由TemporaryDirectory负责。
新增回归验证seed缓存交换、forward checkpoint身份、验证producer指纹，以及配置到完整
allowlist项的一一映射和freeze_review文件SHA；全部缓存仍为临时合成内容。
scientific_source优先读取本机唯一checkout，缺失时读取JJ唯一checkout；两者均缺失会失败，
不跳过实际check_frozen/driver函数契约验证。
旧轮fixture使用真实目录契约：features/inherited_s{seed}/{pool}/{cohort}用于验证，
sealed/features/inherited_s{seed}/{pool}用于封存test；legacy_queue在seal_preparations下，
由frozen/selection.json的queue SHA绑定，回归拒绝错误sibling路径及旧extract身份漂移。
"""
import copy
import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import build_pool_frozen as m


def scientific_source(name):
    for root in (Path('/Users/andrewluan/Desktop/Projects/IRfingerprints/SpectrExpert'),
                 Path('/home/user/IRfingerprints/SpectrExpert')):
        path = root/'nmr_transfer'/name
        if path.is_file():
            return path
    raise FileNotFoundError('scientific_checkout_source_missing:' + name)


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return str(path.resolve())


class Fixture:
    def __init__(self, folder):
        self.base = Path(folder).resolve(); self.root = self.base/'root'; self.source = self.base/'source'
        self.old = self.base/'nmr_global_retrieval_20261022'
        self.candidate = 'c'*64; self.caches = []; self.cache_reference_pins = {}
        self.test_summarizer = Path(write(self.base/'summarize_pool_test.py', '# synthetic test summarizer'))
        self.create_legacy_registry()
        for name in m.SOURCE_NAMES:
            write(self.source/'nmr_transfer'/name, '# synthetic '+name)
        self.jobs = []
        for family in m.validation.FAMILIES:
            for seed in m.validation.SEEDS:
                for role in ('no_aug', 'aug'):
                    folder = self.root/'formal/rankers'/family/f's{seed}'/role
                    write(folder/'best.pt', family+str(seed)+role)
                    self.jobs.append(self.job(folder, 'training', f'{family}_{seed}_{role}'))
        for config in m.validation.CONFIGS:
            for seed in m.validation.SEEDS:
                for cohort in m.validation.COHORTS:
                    self.jobs.append(self.validation_job(config, seed, cohort))
        self.run('formal', self.jobs)
        diagnostics = [self.validation_job(config, seed, cohort)
                       for config in ('frozen_single', 'frozen_dual') for seed in m.validation.SEEDS
                       for cohort in m.validation.COHORTS]
        diagnostics += [self.job(self.root/'diagnostics'/str(i), 'audit', 'diag'+str(i)) for i in range(4)]
        self.run('diagnostics', diagnostics)
        for seed in m.validation.SEEDS:
            for role in ('no_aug', 'aug'):
                folder = self.root/'old/frozen'/f's{seed}'/role
                write(folder/'best.pt', 'frozen'+str(seed)+role)
                write(folder/'summary.json', {'synthetic': True}); write(folder/'EXIT_CODE', '0\n')
        signal = dict(MRR=.5, top1=.4, frequency_crop_MRR=.3, peak_dropout_MRR=.2, macro_MRR=.5, cost_seconds=1.)
        signals = {c: dict(signal) for c in list(m.validation.CONFIGS)+['frozen_single', 'frozen_dual']}
        signals[m.validation.CONFIGS[0]]['MRR'] = .51
        metadata = {}
        for config in signals:
            family = config.rsplit('_', 1)[0]; metadata[config] = {}
            for seed in m.validation.SEEDS:
                folder = self.root/'old/frozen'/f's{seed}' if family == 'frozen' else self.root/'formal/rankers'/family/f's{seed}'
                ckpts = {str(folder/role/'best.pt'): m.sha(folder/role/'best.pt') for role in ('no_aug', 'aug')}
                metadata[config][str(seed)] = {c: dict(candidate_order_sha256=self.candidate, checkpoints=ckpts)
                                             for c in m.validation.COHORTS}
        run_pins = {}
        for phase in ('formal', 'diagnostics'):
            m.validation.verify_run(self.root/'runs'/phase, phase, run_pins, {})
        run_pins.update(self.cache_reference_pins)
        self.summary = dict(schema=m.validation.SCHEMA, signals=signals, selection=m.validation.choose(signals),
            metadata=metadata, input_sha256=run_pins,
            source_sha256={str(Path(m.validation.__file__).resolve()): m.sha(m.validation.__file__)})
        self.summary_path = Path(write(self.base/'validation/summary.json', self.summary))
        write(self.summary_path.parent/'EXIT_CODE', '0\n')
        self.registration_path = Path(write(self.base/'registration.json', dict(
            families=[dict(family=c) for c in m.validation.CONFIGS], seeds=list(m.validation.SEEDS))))
        self.eligibility = dict(status='COMPLETE_FIXED_CANDIDATE_ELIGIBILITY_UNAVAILABLE',
            qualified_independent_structures=0, promotion_allowed=False, candidate_count=114982,
            candidate_order_sha256=self.candidate)
        self.eligibility_path = Path(write(self.base/'eligibility.json', self.eligibility))
        audit_inputs = {str(Path(j['out'])/name): m.sha(Path(j['out'])/name)
                        for j in self.jobs if j['kind'] == 'training' for name in ('summary.json', 'EXIT_CODE')}
        self.audit = dict(status='PASS', checks=dict.fromkeys(m.AUDIT_CHECKS, True), input_sha256=audit_inputs,
                          formal_queue_sha256=m.sha(self.root/'formal_queue.json'))
        self.audit_path = Path(write(self.base/'audit.json', self.audit))
        identity = dict(query_sha256='1'*64, identity_content_sha256='2'*64, variant_mask_content_sha256='3'*64)
        entries = []
        for config in list(m.validation.CONFIGS)+['frozen_single']:
            for seed in m.validation.SEEDS:
                pool = config.rsplit('_', 1)[1]; folder = self.old/'sealed/features'/f'inherited_s{seed}'/pool
                cache = folder/'features.pt'; write(cache, 'SYNTHETIC_CACHE_NEVER_OPENED_BY_BUILDER')
                science_path = write(self.base/'science.json', '{}')
                producer = dict(identity, cohort='test', recall=pool, candidate_count=114982, pool_size=512,
                    candidate_order_sha256=self.candidate, variants=m.VARIANTS, base_queries=1000, records=1371,
                    appended_training_positives=0, feature_cache_sha256=m.sha(cache), checkpoint_sha256=f'{seed:064x}',
                    source_inputs_sha256={science_path: m.sha(science_path)})
                write(folder/'summary.json', producer); write(folder/'EXIT_CODE', '0\n')
                self.caches.append(cache)
                entries.append(dict(config=config, seed=seed, cache=str(cache), cache_sha256=m.sha(cache)))
        self.plan = dict(schema=m.PLAN_SCHEMA, selection_scope='train_val_only',
            preselected=self.summary['selection']['preselected'], purpose='sealed_test_ablation_only', promotion=False,
            fixed_test_identity=identity, evaluations=entries, python='/home/user/.venvs/spectrexpert/bin/python', end_utc='2026-10-23T16:35:41Z',
            source_sha256={str(self.source/'nmr_transfer'/n): m.sha(self.source/'nmr_transfer'/n) for n in m.SOURCE_NAMES})
        self.plan_path = Path(write(self.base/'plan.json', self.plan))
        self.review_path = self.base/'review.json'; self.approve()

    def job(self, folder, kind, name):
        write(folder/'summary.json', {'synthetic': True}); write(folder/'EXIT_CODE', '0\n')
        return dict(id=name, kind=kind, gpu=True, argv=['synthetic', name], cwd=str(self.source),
                    out=str(folder), deps=[], inputs_sha256={})

    def validation_job(self, config, seed, cohort):
        family, pool = config.rsplit('_', 1)
        base = self.root/'diagnostics/frozen' if family == 'frozen' else self.root/'formal'/family
        folder = base/f's{seed}'/pool/cohort
        job = self.job(folder, 'validation', f'eval_{config}_{seed}_{cohort}')
        cache = self.old/'features'/f'inherited_s{seed}'/pool/cohort/'features.pt'
        protocol = Path(write(folder/'protocol.json', dict(args=dict(stage='evaluate', seed=seed, cache=str(cache)))))
        producer = Path(write(cache.parent/'summary.json', dict(cohort=cohort, recall=pool,
                                                              checkpoint_sha256=f'{seed:064x}')))
        self.cache_reference_pins.update({str(p): m.sha(p) for p in (protocol, producer)})
        return job

    def create_legacy_registry(self):
        self.legacy_selection_path = self.old/'frozen/selection.json'
        self.legacy_queue_path = self.old/'seal_preparations/20261022T153236Z/sealed_queue.json'
        self.legacy_jobs = []; allowed = []
        for seed in m.validation.SEEDS:
            for pool in ('single', 'dual'):
                folder = self.old/'sealed/features'/f'inherited_s{seed}'/pool
                bank = str(self.old/'banks'/f'inherited_s{seed}'/'bank.pt')
                argv = ['/home/user/.venvs/spectrexpert/bin/python', '-m', 'nmr_transfer.h_research_extract',
                    '--bank', bank, '--cohort', 'test', '--recall', pool, '--augment', '--out', str(folder),
                    '--frozen-selection', str(self.legacy_selection_path), '--cohorts', str(self.old/'cohorts'),
                    '--fixed-identities', str(self.old/'audit/fixed_evaluation_identities.json')]
                self.legacy_jobs.append(dict(id=f'extract_inherited_s{seed}_{pool}_test', kind='sealed',
                                             out=str(folder), argv=argv))
                allowed.append(dict(bank=bank, extension=None, recall=pool, augment=True, pool_size=512, cohort='test'))
        write(self.legacy_queue_path, dict(schema='nmr_research_jobs_20261022', jobs=self.legacy_jobs))
        self.legacy_frozen = dict(schema='nmr_research_20261022', selection_scope='train_val_only',
            candidate_count=114982, frozen_utc='2026-10-22T23:46:03Z', selection_target=str(self.legacy_selection_path),
            sealed_queue_path=str(self.legacy_queue_path), sealed_queue_sha256=m.sha(self.legacy_queue_path),
            allowed_extract_configs=allowed)
        write(self.legacy_selection_path, self.legacy_frozen)

    def update_legacy_queue(self):
        write(self.legacy_queue_path, dict(schema='nmr_research_jobs_20261022', jobs=self.legacy_jobs))
        self.legacy_frozen['sealed_queue_sha256'] = m.sha(self.legacy_queue_path)
        write(self.legacy_selection_path, self.legacy_frozen)
        self.approve()

    def run(self, phase, jobs):
        queue = dict(source_inputs_sha256={}, jobs=jobs)
        queue_path = Path(write(self.root/(phase+'_queue.json'), queue)); run = self.root/'runs'/phase
        completion = {}
        for job in jobs:
            receipt = dict(state='complete', external_exit=0, internal_exit=0,
                           started_utc='2026-10-23T05:52:00Z', finished_utc='2026-10-23T05:53:00Z',
                           argv=job['argv'], output=job['out'], requested_inputs_sha256={}, cwd=job['cwd'],
                           queue_source_sha256={str(queue_path): m.sha(queue_path)})
            write(run/(job['id']+'.receipt.json'), receipt); completion[job['id']] = receipt
        write(run/'completion.json', dict(success=True, inputs_unchanged=True, jobs=completion, utc='2026-10-23T05:54:00Z'))
        write(run/'driver_start.json', dict(pid=1234, argv=['driver', '--queue', str(queue_path)],
            utc='2026-10-23T05:51:00Z', inputs_sha256={str(queue_path): m.sha(queue_path)}))
        write(run/'launch_receipt.json', dict(external_exit=0, internal_exit=0,
            started_utc='2026-10-23T05:50:00Z', finished_utc='2026-10-23T05:55:00Z',
            child_pid=1234, argv=['python', '-m', 'driver', '--queue', str(queue_path)],
            queue_sha256={str(queue_path): m.sha(queue_path)}, source_sha256={}, bundle_sha256={}))
        write(run/'EXIT_CODE', '0\n'); write(run/'EXTERNAL_EXIT_CODE', '0\n')

    def approve(self):
        required = {str(p): m.sha(p) for p in (self.summary_path, self.plan_path, self.audit_path,
                                               self.eligibility_path, self.registration_path,
                                               self.legacy_selection_path, self.legacy_queue_path)}
        required.update({str(Path(p).resolve()): m.sha(p) for p in (m.__file__, m.validation.__file__, self.test_summarizer)})
        required.update(self.plan['source_sha256'])
        write(self.review_path, dict(status='PASS', independent=True, reviewer='synthetic_independent',
                                    scope='freeze_plan_and_implementation', reviewed_sha256=required))

    def args(self):
        return SimpleNamespace(root=str(self.root), source=str(self.source), validation_summary=str(self.summary_path),
            summary_sha256=m.sha(self.summary_path), test_plan=str(self.plan_path), review=str(self.review_path),
            train_audit=str(self.audit_path), eligibility=str(self.eligibility_path), preregistration=str(self.registration_path),
            test_summarizer=str(self.test_summarizer),
            formal_run=None, diagnostics_run=None)

    def build(self, frozen_time=(6, 0, 0)):
        out = self.base/'freeze'; out.mkdir()
        class Clock(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 23, *frozen_time, tzinfo=datetime.timezone.utc)
        with patch.object(m.dt, 'datetime', Clock):
            result = m.build(self.args(), out)
        return result, json.loads((out/'frozen_selection.json').read_text()), json.loads((out/'sealed_queue.json').read_text())


class FrozenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.f = Fixture(self.temp.name)

    def test_full_freeze_contract_and_no_test_cache_open(self):
        original = Path.open
        def guard(path, *args, **kwargs):
            if path.name in ('features.pt', 'predictions.jsonl'):
                raise AssertionError('forbidden_test_data_read')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guard):
            receipt, frozen, queue = self.f.build()
        self.assertEqual(receipt['jobs'], 18); self.assertEqual(len(frozen['allowed_ranker_configs']), 18)
        self.assertEqual(frozen['preselected'], self.f.summary['selection']['preselected'])
        self.assertFalse(frozen['promotion']); self.assertEqual(frozen['verify_eligibility'], 'unavailable')
        self.assertEqual(frozen['fixed_test_identity'], self.f.plan['fixed_test_identity'])
        self.assertEqual(frozen['configurations'], sorted(set(e['config'] for e in self.f.plan['evaluations'])))
        self.assertEqual(frozen['freeze_review'], dict(path=str(self.f.review_path), sha256=m.sha(self.f.review_path)))
        self.assertEqual(frozen['test_plan_path'], str(self.f.plan_path))
        for entry in self.f.plan['evaluations']:
            mapped = frozen['ranker_configurations'][entry['config']][str(entry['seed'])]
            self.assertIn(mapped, frozen['allowed_ranker_configs'])
            self.assertEqual(mapped['cache_sha256'], entry['cache_sha256'])
            self.assertEqual(mapped['seed'], entry['seed'])
            weights = m.weights_for(self.f.summary, entry['config'], entry['seed'])
            self.assertEqual(mapped['no_aug_checkpoint'], weights['no_aug'][0])
        self.assertTrue(all(j['kind'] == 'sealed' and '--frozen-selection' in j['argv'] for j in queue['jobs']))
        self.assertIn(str(self.f.audit_path), queue['source_inputs_sha256'])
        self.assertTrue(all(str(self.f.audit_path) not in j['inputs_sha256'] for j in queue['jobs']))
        self.assertTrue(all(len(j['inputs_sha256']) == 16 for j in queue['jobs']))
        self.assertEqual(frozen['test_variants'], {'observed': 1000, 'frequency_crop': 200, 'peak_dropout': 171})
        driver_spec = importlib.util.spec_from_file_location('driver', scientific_source('h_research_driver.py'))
        # driver validate_queue has no scientific dependencies; compile only that AST function.
        import ast
        tree = ast.parse(Path(driver_spec.origin).read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_queue')
        namespace = {'Path': Path}; exec(compile(ast.Module(body=[fn], type_ignores=[]), 'driver_contract', 'exec'), namespace)
        namespace['validate_queue'](queue)

    def test_generated_selection_passes_actual_check_frozen_function(self):
        import ast
        _, frozen, queue = self.f.build()
        path = scientific_source('h_pool_ranker.py')
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_frozen')
        namespace = dict(Path=Path, json=json, SCHEMA=m.SCHEMA,
            source_paths=lambda: [self.f.source/'nmr_transfer'/n for n in m.SOURCE_NAMES[:-1]],
            input_fingerprints=lambda paths: {str(Path(p).resolve()): m.sha(p) for p in paths})
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'actual_check_frozen', 'exec'), namespace)
        argv = queue['jobs'][0]['argv']; args = SimpleNamespace(
            frozen_selection=argv[argv.index('--frozen-selection')+1],
            no_aug_checkpoint=argv[argv.index('--no-aug-checkpoint')+1],
            aug_checkpoint=argv[argv.index('--aug-checkpoint')+1],
            cache=argv[argv.index('--cache')+1], seed=int(argv[argv.index('--seed')+1]))
        manifest = json.loads((Path(args.cache).parent/'summary.json').read_text())
        namespace['check_frozen'](args, dict(manifest=manifest), m.sha(args.cache))
        with self.assertRaisesRegex(ValueError, 'not_frozen'):
            args.seed = 99; namespace['check_frozen'](args, dict(manifest=manifest), m.sha(args.cache))

    def test_optional_frozen_dual_requires_two_seeds(self):
        for seed in m.validation.SEEDS:
            e = next(e for e in self.f.plan['evaluations'] if e['config'].endswith('_dual') and e['seed'] == seed)
            self.f.plan['evaluations'].append(dict(e, config='frozen_dual'))
        write(self.f.plan_path, self.f.plan); self.f.approve()
        result, frozen, _ = self.f.build()
        self.assertEqual(result['jobs'], 20); self.assertEqual(len(frozen['allowed_ranker_configs']), 20)

    def test_missing_baseline_rejected(self):
        self.f.plan['evaluations'] = [e for e in self.f.plan['evaluations'] if e['config'] != 'frozen_single']
        write(self.f.plan_path, self.f.plan); self.f.approve()
        with self.assertRaisesRegex(ValueError, 'direct_baseline'): self.f.build()

    def test_partial_seed_rejected(self):
        self.f.plan['evaluations'].pop(); write(self.f.plan_path, self.f.plan); self.f.approve()
        with self.assertRaisesRegex(ValueError, 'seed_matrix'): self.f.build()

    def test_incomplete_main_configuration_rejected(self):
        config = m.validation.CONFIGS[0]
        self.f.plan['evaluations'] = [e for e in self.f.plan['evaluations'] if e['config'] != config]
        write(self.f.plan_path, self.f.plan); self.f.approve()
        with self.assertRaisesRegex(ValueError, 'complete_main_matrix'): self.f.build()

    def test_incumbent_cannot_enter_plan(self):
        self.f.plan['evaluations'][0]['config'] = 'incumbent_single'; write(self.f.plan_path, self.f.plan)
        with self.assertRaisesRegex(ValueError, 'complete_main_matrix'): self.f.build()

    def test_plan_cannot_reselect(self):
        self.f.plan['preselected'] = m.validation.CONFIGS[1]; write(self.f.plan_path, self.f.plan)
        with self.assertRaisesRegex(ValueError, 'explicit_frozen'): self.f.build()

    def test_failed_outer_exit_rejected(self):
        write(self.f.root/'runs/formal/EXTERNAL_EXIT_CODE', '1\n')
        with self.assertRaisesRegex(ValueError, 'nonzero'): self.f.build()

    def test_incomplete_training_audit_rejected(self):
        self.f.audit['checks']['full_budget'] = False; write(self.f.audit_path, self.f.audit); self.f.approve()
        with self.assertRaisesRegex(ValueError, 'complete_training_audit'): self.f.build()

    def test_train_audit_all_fit_producers_required(self):
        self.f.audit['input_sha256'].popitem(); write(self.f.audit_path, self.f.audit); self.f.approve()
        with self.assertRaisesRegex(ValueError, 'missing_fit_producers'): self.f.build()

    def test_review_must_bind_current_inputs(self):
        approval = json.loads(self.f.review_path.read_text()); approval['reviewed_sha256'].pop(str(self.f.audit_path))
        write(self.f.review_path, approval)
        with self.assertRaisesRegex(ValueError, 'review_does_not_bind'): self.f.build()

    def test_review_must_be_independent(self):
        approval = json.loads(self.f.review_path.read_text()); approval['independent'] = False; write(self.f.review_path, approval)
        with self.assertRaisesRegex(ValueError, 'independent_freeze_review'): self.f.build()

    def test_verify_eligibility_not_fabricated(self):
        self.f.eligibility['promotion_allowed'] = True; write(self.f.eligibility_path, self.f.eligibility)
        with self.assertRaisesRegex(ValueError, 'unavailable_eligibility'): self.f.build()

    def test_test_producer_sha_mismatch_rejected(self):
        self.f.plan['evaluations'][0]['cache_sha256'] = 'f'*64; write(self.f.plan_path, self.f.plan)
        with self.assertRaisesRegex(ValueError, 'plan_sha'): self.f.build()

    def test_fixed_test_count_changed_rejected(self):
        producer_path = self.f.caches[0].parent/'summary.json'; producer = json.loads(producer_path.read_text())
        producer['variants']['observed'] = 999; write(producer_path, producer)
        with self.assertRaisesRegex(ValueError, 'identity_or_scope'): self.f.build()

    def test_checkpoint_changed_rejected(self):
        path = next(iter(self.f.summary['metadata'][m.validation.CONFIGS[0]]['193']['original_val']['checkpoints']))
        write(path, 'tampered')
        with self.assertRaisesRegex(ValueError, 'input_sha_changed'): self.f.build()

    def test_sealed_output_reuse_rejected(self):
        e = self.f.plan['evaluations'][0]; (self.f.root/'sealed'/e['config']/f"s{e['seed']}"/'test').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'sealed_output_already'): self.f.build()

    def test_freeze_deadline_enforced(self):
        with self.assertRaisesRegex(ValueError, 'freeze_deadline_passed'): self.f.build((14, 35, 42))

    def test_fixed_python_enforced(self):
        self.f.plan['python'] = '/some/other/python'; write(self.f.plan_path, self.f.plan); self.f.approve()
        with self.assertRaisesRegex(ValueError, 'python_must_match_protocol'): self.f.build()

    def test_final_test_summarizer_review_bound(self):
        write(self.f.test_summarizer, '# changed')
        with self.assertRaisesRegex(ValueError, 'review_does_not_bind'): self.f.build()

    def test_scientific_end_deadline_enforced(self):
        self.f.plan['end_utc'] = '2026-10-23T16:35:42Z'; write(self.f.plan_path, self.f.plan); self.f.approve()
        with self.assertRaisesRegex(ValueError, 'test_deadline_outside_protocol'): self.f.build()

    def test_seed_swapped_cache_rejected(self):
        first = self.f.plan['evaluations'][0]
        other = next(e for e in self.f.plan['evaluations'] if e['config'] == first['config'] and e['seed'] != first['seed'])
        first.update(cache=other['cache'], cache_sha256=other['cache_sha256'])
        write(self.f.plan_path, self.f.plan); self.f.approve()
        with self.assertRaisesRegex(ValueError, 'seed_or_inherited_path'): self.f.build()

    def test_forward_checkpoint_identity_rejected(self):
        path = Path(self.f.plan['evaluations'][0]['cache']).parent/'summary.json'
        producer = json.loads(path.read_text()); producer['checkpoint_sha256'] = 'f'*64; write(path, producer)
        with self.assertRaisesRegex(ValueError, 'identity_or_scope'): self.f.build()

    def test_validation_cache_producer_bound_to_selection(self):
        path = self.f.old/'features/inherited_s193/single/original_val/summary.json'
        producer = json.loads(path.read_text()); producer['checkpoint_sha256'] = 'f'*64; write(path, producer)
        with self.assertRaisesRegex(ValueError, 'producer_not_selection_bound'): self.f.build()

    def test_empty_fixed_test_identity_rejected(self):
        self.f.plan['fixed_test_identity'] = {}; write(self.f.plan_path, self.f.plan); self.f.approve()
        with self.assertRaisesRegex(ValueError, 'fixed_test_identity_required'): self.f.build()

    def test_nonhex_fixed_test_identity_rejected(self):
        self.f.plan['fixed_test_identity']['query_sha256'] = 'z'*64; write(self.f.plan_path, self.f.plan); self.f.approve()
        with self.assertRaisesRegex(ValueError, 'invalid_test_identity_digest'): self.f.build()

    def test_actual_legacy_sealed_paths_used(self):
        _, frozen, _ = self.f.build()
        for entry in self.f.plan['evaluations']:
            expected = self.f.old/'sealed/features'/f"inherited_s{entry['seed']}"/entry['config'].rsplit('_', 1)[1]/'features.pt'
            self.assertEqual(entry['cache'], str(expected))
        for path in (self.f.legacy_selection_path, self.f.legacy_queue_path):
            self.assertEqual(frozen['reviewed_inputs_sha256'][str(path)], m.sha(path))

    def test_invented_validation_sibling_test_cache_rejected(self):
        entry = self.f.plan['evaluations'][0]
        fake = self.f.old/'features'/f"inherited_s{entry['seed']}"/entry['config'].rsplit('_', 1)[1]/'test/features.pt'
        write(fake, 'synthetic wrong sibling cache')
        entry.update(cache=str(fake), cache_sha256=m.sha(fake))
        write(self.f.plan_path, self.f.plan); self.f.approve()
        with self.assertRaisesRegex(ValueError, 'seed_or_inherited_path'): self.f.build()

    def test_legacy_queue_sha_binding_required(self):
        self.f.legacy_jobs[0]['out'] += '_changed'
        write(self.f.legacy_queue_path, dict(schema='nmr_research_jobs_20261022', jobs=self.f.legacy_jobs))
        self.f.approve()
        with self.assertRaisesRegex(ValueError, 'legacy_sealed_queue_sha_changed'): self.f.build()

    def test_legacy_extract_seed_output_binding_required(self):
        self.f.legacy_jobs[0]['out'] = self.f.legacy_jobs[2]['out']; self.f.update_legacy_queue()
        with self.assertRaisesRegex(ValueError, 'legacy_extract_output_registration'): self.f.build()

    def test_legacy_extract_test_cohort_required(self):
        argv = self.f.legacy_jobs[0]['argv']; argv[argv.index('--cohort')+1] = 'tune'; self.f.update_legacy_queue()
        with self.assertRaisesRegex(ValueError, 'legacy_extract_scientific_registration'): self.f.build()

    def test_legacy_extract_frozen_selection_binding_required(self):
        argv = self.f.legacy_jobs[0]['argv']; argv[argv.index('--frozen-selection')+1] = '/unrelated/selection.json'
        self.f.update_legacy_queue()
        with self.assertRaisesRegex(ValueError, 'legacy_extract_scientific_registration'): self.f.build()


if __name__ == '__main__':
    unittest.main()

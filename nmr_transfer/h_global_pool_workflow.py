"""功能概述：有界比较固定完整候选库上的 1024/512 重排短名单。

arms/reference_paths/extract_command 固定 physical_base_s193/211、atom_count_1.0、
同 bank/width/extension 和 train2048/original_val320/tune1024。read_reference 读取
旧对应 MLP 权重的超参；新 fit 只替换训练缓存与输出，64epoch/2048samples、初始化、
结构划分及 alpha/epoch 选择仍由未修改的 h_global_ranker 完成。validate_cache_pair
核对查询/预测/粗排/来源、公共候选分数、train 抽样选择及验证不补正例；validate_fit
再核对实际初始化、结构曝光和训练参数。完整 114982 候选始终先由旧 extractor 评分。

Workflow 管理自己的新进程组、真实 EXIT_CODE 与必要产物；先等待 forward 和
observation_calibration 成功，再仅在 GPU2/3 各空闲70GiB时运行四查询 smoke、六份
augment 缓存和四个配对 MLP。CPU blend 保留五 gamma 和两个单模型端点；--metadata-route
可再复用 blend。汇总 normal1344 microMRR，并以两seed原val+tune所有crop/dropout
查询共同 microMRR 选择一个 gamma；不按 variant 或有利 seed 分别选择。

Budget、SIGALRM 和每次轮询共同限制到 UTC2026-10-20 19:50，预留15秒清理。正式
抽取前必须尚余90分钟；不足或截止写 pending/EXIT_CODE124，保留部分产物，terminate/
wait/kill/wait 自身子进程组。父任务失败明确失败，不启动后续GPU，不重启/清理他人。
源码单独 SHA 守卫、旧/新实际输入始末 SHA 均留档。main 只接收研究路径和 gate，
不接收任意命令/候选/测试配置，不访问 verify/test、网络或任何外部 API。输出目录
固定 root/pool1024 且必须新建；本模块不修改任何旧源码、缓存、结果或 sealed 门禁。
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

import torch

from .h_global_protocol import CRITICAL_SOURCES
from .h_global_ranker_blend import GAMMAS
from .prepare import sha


HARD_DEADLINE = '2026-10-20T19:50:00+00:00'
GPUS = (2, 3)
MIN_FREE_MIB = 70 * 1024
FIT_FIELDS = ('features', 'architecture', 'shared', 'augment', 'seed', 'anchor', 'epochs', 'samples')
NORMAL_COUNTS = {'original_val': 320, 'tune': 1024}
COHORT_COUNTS = dict(train=2048, **NORMAL_COUNTS)


class DeadlineReached(TimeoutError):
    pass


class Budget:
    """绝对 UTC 上限不允许延后；work 截止提前15秒供自己的子进程收尾。"""
    def __init__(self, deadline=HARD_DEADLINE, clock=time.time):
        parsed = dt.datetime.fromisoformat(deadline.replace('Z', '+00:00'))
        maximum = dt.datetime.fromisoformat(HARD_DEADLINE)
        if parsed.tzinfo is None or parsed > maximum:
            raise ValueError('pool_deadline_must_be_aware_and_no_later_than_1950UTC')
        self.deadline, self.clock = parsed.timestamp(), clock

    def remaining(self):
        return self.deadline - 15 - self.clock()

    def check(self, admission=False):
        remaining = self.remaining()
        if remaining <= 0 or admission and remaining < 90 * 60:
            raise DeadlineReached('pool_deadline_or_90min_admission_budget_exhausted')
        return remaining

    def pause(self, seconds):
        time.sleep(min(seconds, self.check()))


def arms():
    return [dict(name=f'physical_mlp_{kind}_s{seed}', owner=f'physical_base_s{seed}',
                 seed=seed, augment=kind == 'aug')
            for seed in (193, 211) for kind in ('no_aug', 'aug')]


def reference_paths(root, arm):
    name = f'physical_mlp_s{arm["seed"]}' if arm['augment'] else f'physical_mlp_no_augmentation_s{arm["seed"]}'
    folder = root / ('calibration' if arm['augment'] else 'mlp_ablation') / 'training' / name
    return folder / 'best.pt', folder / 'initial.pt', folder / 'summary.json'


def read_reference(root, arm):
    checkpoint, _, summary = reference_paths(root, arm)
    if (checkpoint.parent / 'EXIT_CODE').read_text().strip() != '0':
        raise ValueError('pool_reference_fit_not_complete:' + arm['name'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    expected = dict(features='all', architecture='mlp', shared=False, augment=arm['augment'],
                    seed=arm['seed'], anchor=.1, epochs=64, samples=2048)
    if {field: saved['args'][field] for field in FIT_FIELDS} != expected or saved['pool_size'] != 512:
        raise ValueError('pool_reference_fit_protocol_changed:' + arm['name'])
    return saved, json.loads(summary.read_text()), expected


def extract_command(args, owner, cohort, out, limit=0):
    if cohort not in ('train', 'original_val', 'tune') or limit not in (0, 4):
        raise ValueError('pool_unregistered_extract_scope')
    bank = Path(args.previous) / 'study/evaluation' / owner / 'bank.pt'
    cmd = [sys.executable, '-m', 'nmr_transfer.h_global_extract', '--base', args.base,
        '--bank', str(bank), '--cohorts', str(Path(args.root) / 'cohorts'),
        '--extension', str(Path(args.root) / 'phase_a/extensions' / owner / 'extension.pt'),
        '--widths', str(bank.parent / 'val/width_fit.json'), '--cohort', cohort,
        '--coarse', 'atom_count_1.0', '--pool-size', '1024', '--augment', '--out', str(out)]
    return cmd + (['--limit', str(limit)] if limit else [])


def fit_command(train, out, settings):
    cmd = [sys.executable, '-m', 'nmr_transfer.h_global_ranker', '--stage', 'fit',
           '--train', str(train), '--out', str(out)]
    for field in ('features', 'architecture', 'seed', 'anchor', 'epochs', 'samples'):
        cmd += ['--' + field, str(settings[field])]
    if settings['augment']:
        cmd.append('--augment')
    if settings['shared']:
        cmd.append('--shared')
    return cmd


def sampler_signature(rows, augment):
    """复用 ranker 的 valid-group 与 variant 口径，比较真实结构和内部划分。"""
    valid = {r['group'] for r in rows if r['variant'] == 'observed' and r.get('rerank_eligible', True)}
    selected = [(r['group'], r['query_order'], r['variant']) for r in rows
                if r['group'] in valid and r.get('rerank_eligible', True)
                and (augment or r['variant'] == 'observed')]
    internal = {g: int(hashlib.sha256(('ranker_internal_317' + str(g)).encode()).hexdigest()[:12], 16) % 5 == 0
                for g, _, _ in selected}
    return selected, internal


def validate_cache_pair(old, new, cohort, limit=0):
    """只允许 pool 变化；验证绝不补正例，训练结构/variant曝光保持对应。"""
    left, right = old['manifest'], new['manifest']
    if left['cohort'] != cohort or right['cohort'] != cohort or left['pool_size'] != 512 or right['pool_size'] != 1024:
        raise ValueError('pool_cache_scope_or_size_mismatch')
    fields = ('feature_names', 'coarse_config', 'candidate_count', 'checkpoint_sha256',
              'candidate_order_sha256', 'source_inputs_sha256', 'feature_code_sha256')
    for field in fields:
        if left[field] != right[field]:
            raise ValueError('pool_cache_protocol_changed:' + field)
    if (left['candidate_count'] != 114982 or len(new['smiles']) != 114982
            or old['smiles'] != new['smiles']):
        raise ValueError('pool_complete_candidate_library_changed')
    old_rows = [r for r in old['rows'] if not limit or r['query_order'] < limit]
    if len(old_rows) != len(new['rows']):
        raise ValueError('pool_query_or_variant_count_changed')
    if not limit and (left['query_sha256'] != right['query_sha256'] or left['variants'] != right['variants']):
        raise ValueError('pool_base_queries_changed')
    expected_observed = limit or COHORT_COUNTS[cohort]
    actual_variants = dict(Counter(r['variant'] for r in new['rows']))
    if (right['variants']['observed'] != expected_observed or right['variants'] != actual_variants
            or right['queries'] != len(new['rows'])):
        raise ValueError('pool_base_query_count_changed')
    if cohort != 'train' and right['appended_training_positives']:
        raise ValueError('pool_validation_positive_injection')
    for a, b in zip(old_rows, new['rows']):
        for field in ('query_order', 'variant', 'source', 'record_id', 'smiles', 'group',
                      'coarse_rank', 'baseline_rank', 'coarse_top10', 'baseline_top10', 'query_observation'):
            if a[field] != b[field]:
                raise ValueError('pool_query_or_full_library_score_changed:' + field)
        if cohort != 'train' and (b['target_position'] >= 0) != bool(b['target_in_coarse_pool']):
            raise ValueError('pool_validation_positive_injection')
        if b['target_in_coarse_pool'] and b['target_position'] < 0:
            raise ValueError('pool_recalled_target_missing')
        if cohort == 'train' and b['target_position'] < 0:
            raise ValueError('pool_training_positive_missing')
        old_size = len(a['candidate_indices']) - int(not a['target_in_coarse_pool'] and a['target_position'] >= 0)
        common = {index: j for j, index in enumerate(b['candidate_indices'].tolist())}
        positions = [common.get(index, -1) for index in a['candidate_indices'][:old_size].tolist()]
        if (-1 in positions and a.get('rerank_eligible', True) and b.get('rerank_eligible', True)):
            raise ValueError('pool_larger_shortlist_not_nested')
        old_positions = [j for j, position in enumerate(positions) if position >= 0]
        new_positions = [position for position in positions if position >= 0]
        if old_positions and not torch.allclose(a['coarse_scores'][old_positions],
                b['coarse_scores'][new_positions], atol=1e-6, rtol=1e-6):
            raise ValueError('pool_common_candidate_score_changed')
    if cohort == 'train' and not limit:
        for augmented in (False, True):
            if sampler_signature(old_rows, augmented) != sampler_signature(new['rows'], augmented):
                raise ValueError('pool_training_group_or_variant_exposure_changed')
    diagnostics = {}
    for variant in right['variants']:
        pairs = [(a, b) for a, b in zip(old_rows, new['rows']) if b['variant'] == variant]
        diagnostics[variant] = dict(queries=len(pairs),
            recall512=sum(a['target_in_coarse_pool'] for a, _ in pairs) / len(pairs),
            recall1024=sum(b['target_in_coarse_pool'] for _, b in pairs) / len(pairs),
            recovered=sum(not a['target_in_coarse_pool'] and b['target_in_coarse_pool'] for a, b in pairs))
    return diagnostics


def validate_fit(root, out, arm, settings):
    reference, old_initial, old_summary = reference_paths(root, arm)
    saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    if {field: saved['args'][field] for field in FIT_FIELDS} != settings or saved['pool_size'] != 1024:
        raise ValueError('pool_actual_fit_settings_changed')
    old = json.loads(old_summary.read_text())
    new = json.loads((out / 'summary.json').read_text())
    for field in ('group_exposure_sha256', 'training_groups', 'internal_groups', 'training_rows', 'internal_rows'):
        if old[field] != new[field]:
            raise ValueError('pool_actual_training_exposure_changed:' + field)
    a = torch.load(old_initial, map_location='cpu', weights_only=False)['state_dict']
    b = torch.load(out / 'initial.pt', map_location='cpu', weights_only=False)['state_dict']
    if a.keys() != b.keys() or any(not torch.equal(a[key], b[key]) for key in a):
        raise ValueError('pool_initial_model_changed')
    return dict(reference_checkpoint=str(reference), best_epoch=saved['epoch'], alpha=saved['alpha'],
                group_exposure_sha256=new['group_exposure_sha256'], settings=settings)


def terminate_owned(active):
    """仅清理 start_new_session 创建的自身进程组，等待每个子进程确实收敛。"""
    for _, process, _ in active:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    end = time.monotonic() + 5
    for _, process, _ in active:
        try:
            process.wait(timeout=max(.01, end - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    end = time.monotonic() + 5
    for _, process, _ in active:
        process.wait(timeout=max(.01, end - time.monotonic()))


class Workflow:
    def __init__(self, args, out, budget):
        self.args, self.out, self.budget = args, out, budget
        self.active, self.started = [], time.monotonic()
        names = set(CRITICAL_SOURCES) | {'h_global_pool_workflow.py', 'h_global_ranker_blend.py',
                                       'h_global_metadata_route.py'}
        self.sources = {str(Path(__file__).with_name(name).resolve()): sha(Path(__file__).with_name(name)) for name in names}
        self.gate_sha = sha(args.gate)

    def status(self, stage, **kw):
        tmp = self.out / 'status.tmp'
        tmp.write_text(json.dumps(dict(stage=stage, seconds=time.monotonic() - self.started,
            remaining_seconds=self.budget.remaining(), deadline_utc=self.args.deadline, **kw)))
        tmp.replace(self.out / 'status.json')

    def guard(self):
        self.budget.check()
        for path, digest in self.sources.items():
            if sha(path) != digest:
                raise ValueError('pool_source_changed:' + path)
        if sha(self.args.gate) != self.gate_sha:
            raise ValueError('pool_gate_changed')

    def screen(self, cmd):
        self.guard()
        checked = subprocess.run(['bash', self.args.gate, 'screen-cmd', shlex.join(cmd)],
            capture_output=True, text=True, timeout=min(10, self.budget.check()))
        if checked.returncode or checked.stdout.strip() != 'ok':
            raise ValueError('pool_command_refused')

    def dependencies(self):
        while True:
            self.guard()
            codes = {name: (Path(self.args.root) / name / 'EXIT_CODE').read_text().strip()
                     if (Path(self.args.root) / name / 'EXIT_CODE').exists() else None
                     for name in ('forward', 'observation_calibration')}
            self.status('waiting_parents', parent_exit_codes=codes)
            if any(code not in (None, '0') for code in codes.values()):
                raise RuntimeError('pool_parent_failed_explicit_diagnosis:' + json.dumps(codes))
            if all(code == '0' for code in codes.values()):
                for name in codes:
                    if not (Path(self.args.root) / name / 'summary.json').exists():
                        raise ValueError('pool_parent_exit0_without_summary:' + name)
                return
            self.budget.pause(10)

    def resources(self):
        while True:
            query = ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits']
            self.screen(query)
            text = subprocess.check_output(query, text=True, timeout=min(10, self.budget.check()))
            free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in text.splitlines()}
            if all(free.get(gpu, 0) >= MIN_FREE_MIB for gpu in GPUS):
                return
            self.status('waiting_GPU2_3', free_MiB=free)
            self.budget.pause(10)

    def run(self, jobs, stage, gpu=False):
        for offset in range(0, len(jobs), 2):
            if gpu:
                self.resources()
                if stage == 'extract' and offset == 0:
                    self.budget.check(admission=True)
            self.active.clear()
            for slot, (name, cmd, folder, required) in enumerate(jobs[offset:offset + 2]):
                self.screen(cmd)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPUS[slot]) if gpu else '',
                           OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
                with (self.out / (stage + '_' + name + '.log')).open('x') as stream:
                    process = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                self.active.append((name, process, folder))
            pids = [dict(name=n, pid=p.pid, gpu=GPUS[i] if gpu else None,
                         command=jobs[offset + i][1], screened=True) for i, (n, p, _) in enumerate(self.active)]
            (self.out / (stage + f'_wave{offset // 2}_pids.json')).write_text(json.dumps(pids, indent=2))
            while True:
                self.guard()
                codes = {n: p.poll() for n, p, _ in self.active}
                self.status(stage, wave=offset // 2, exit_codes=codes)
                if any(code not in (None, 0) for code in codes.values()):
                    raise RuntimeError('pool_child_failed:' + json.dumps(codes))
                if all(code == 0 for code in codes.values()):
                    break
                self.budget.pause(2)
            for _, _, folder, required in jobs[offset:offset + 2]:
                if (folder / 'EXIT_CODE').read_text().strip() != '0':
                    raise ValueError('pool_child_artifact_exit_not_zero:' + str(folder))
                for name in ('summary.json',) + tuple(required):
                    if not (folder / name).is_file():
                        raise ValueError('pool_child_missing_output:' + str(folder / name))
            for _, process, _ in self.active:
                process.wait(timeout=min(1, self.budget.check()))
            self.active.clear()


def aggregate(reports):
    """normal按1344微平均；压力全部query按计数合并，所有seed/类别共选gamma。"""
    normal, numerator, denominator = {}, {str(g): 0. for g in GAMMAS}, {str(g): 0 for g in GAMMAS}
    for owner, cohorts in reports.items():
        normal[owner] = {}
        for gamma in map(str, GAMMAS):
            total = 0.
            for cohort, count in NORMAL_COUNTS.items():
                metric = cohorts[cohort]['metrics'][gamma]
                if metric['overall']['queries'] != count:
                    raise ValueError('pool_normal_query_count_changed')
                total += metric['overall']['MRR'] * count
                for kind in ('frequency_crop', 'peak_dropout'):
                    stress = metric['stress'][kind]
                    n = stress['queries']
                    numerator[gamma] += stress.get('MRR', 0.) * n
                    denominator[gamma] += n
            normal[owner][gamma] = total / 1344
    if len(reports) != 2 or len(set(denominator.values())) != 1 or not denominator['0.0']:
        raise ValueError('pool_missing_two_seed_stress_comparison')
    stress = {g: numerator[g] / denominator[g] for g in numerator}
    selected = max(GAMMAS, key=lambda g: (stress[str(g)], -g))
    return dict(normal1344_MRR=normal, stress_micro_MRR=stress, stress_queries=denominator['0.0'],
                shared_stress_gamma=selected, selection='both seeds; both val cohorts; crop+dropout microMRR; exact ties prefer smaller gamma')


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'base', 'previous', 'gate'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--deadline', default=HARD_DEADLINE)
    parser.add_argument('--metadata-route', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = root / 'pool1024'
    out.mkdir(parents=True, exist_ok=False)
    workflow, inputs, results, diagnostics, fits = None, {}, {}, {}, {}
    old_handlers = {}
    def stop(signum, frame):
        raise DeadlineReached('pool_deadline_or_external_signal:' + str(signum))
    try:
        budget = Budget(args.deadline)
        budget.check()
        workflow = Workflow(args, out, budget)
        for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, budget.check())
        if shutil.disk_usage(out).free < 10 * 2**30:
            raise ValueError('pool_requires_10GiB_output_headroom')
        settings = {}
        for arm in arms():
            saved, _, settings[arm['name']] = read_reference(root, arm)
            train_manifest = json.loads((root / 'calibration/features' / arm['owner'] / 'train/summary.json').read_text())
            if (saved['train_sha256'] != train_manifest['feature_cache_sha256']
                    or saved['prediction_checkpoint_sha256'] != train_manifest['checkpoint_sha256']
                    or saved['coarse_config'] != train_manifest['coarse_config']
                    or saved['coarse_config']['name'] != 'atom_count_1.0'):
                raise ValueError('pool_reference_not_from_corresponding_512_cache:' + arm['name'])
            for path in reference_paths(root, arm):
                inputs[str(path)] = sha(path)
        for seed in (193, 211):
            for cohort in ('train', 'original_val', 'tune'):
                folder = root / 'calibration/features' / f'physical_base_s{seed}' / cohort
                manifest = json.loads((folder / 'summary.json').read_text())
                inputs[str(folder / 'features.pt')] = manifest['feature_cache_sha256']
                inputs[str(folder / 'summary.json')] = sha(folder / 'summary.json')
                for path, digest in manifest['source_inputs_sha256'].items():
                    if path in inputs and inputs[path] != digest:
                        raise ValueError('pool_reference_source_SHA_conflict:' + path)
                    inputs[path] = digest
            for cohort in NORMAL_COUNTS:
                folder = root / 'blend/base' / f'physical_base_s{seed}' / cohort
                for name in ('summary.json', 'EXIT_CODE'):
                    inputs[str(folder / name)] = sha(folder / name)
        for path, digest in inputs.items():
            budget.check()
            if sha(path) != digest:
                raise ValueError('pool_reference_input_changed:' + path)
        plan = dict(pool_size=1024, control_pool_size=512, candidate_count=114982,
            arms=arms(), settings=settings, source_sha256=workflow.sources,
            input_sha256=inputs, gate_sha256=workflow.gate_sha, deadline_utc=args.deadline,
            GPU_partition=list(GPUS), minimum_free_MiB=MIN_FREE_MIB,
            parents=['forward', 'observation_calibration'], formal_admission_remaining_seconds=5400,
            normal_selection='original320+tune1024 microMRR; same no_aug and aug endpoints as512',
            stress_selection='one shared gamma over both seeds and all original_val/tune crop+dropout rows',
            feature_extractor_unchanged=True, no_sealed_access=True, external_api_calls=0)
        (out / 'plan.json').write_text(json.dumps(plan, indent=2))
        workflow.dependencies()
        budget.check(admission=True)
        def extraction(owner, cohort, smoke=False):
            folder = out / ('smoke' if smoke else 'features') / owner / cohort
            return owner + '_' + cohort, extract_command(args, owner, cohort, folder, 4 if smoke else 0), folder, ('features.pt', 'protocol.json')
        def check_cache(owner, cohort, smoke=False):
            folder = out / ('smoke' if smoke else 'features') / owner / cohort
            cached_summary = json.loads((folder / 'summary.json').read_text())
            if sha(folder / 'features.pt') != cached_summary['feature_cache_sha256']:
                raise ValueError('pool_new_cache_digest_mismatch')
            for name in ('features.pt', 'summary.json', 'protocol.json', 'EXIT_CODE'):
                inputs[str(folder / name)] = sha(folder / name)
            old = torch.load(root / 'calibration/features' / owner / cohort / 'features.pt', map_location='cpu', weights_only=False, mmap=True)
            new = torch.load(folder / 'features.pt', map_location='cpu', weights_only=False, mmap=True)
            return validate_cache_pair(old, new, cohort, 4 if smoke else 0)
        workflow.run([extraction(f'physical_base_s{s}', c, True) for c in ('train', 'original_val') for s in (193, 211)], 'smoke', gpu=True)
        for seed in (193, 211):
            for cohort in ('train', 'original_val'):
                check_cache(f'physical_base_s{seed}', cohort, True)
        (out / 'smoke_passed.json').write_text(json.dumps(dict(base_queries_per_job=4, jobs=4, scope='real train/val only')))
        budget.check(admission=True)
        workflow.run([extraction(f'physical_base_s{s}', c) for c in ('train', 'original_val', 'tune') for s in (193, 211)], 'extract', gpu=True)
        for seed in (193, 211):
            owner = f'physical_base_s{seed}'
            diagnostics[owner] = {c: check_cache(owner, c) for c in ('train', 'original_val', 'tune')}
        (out / 'pool_recall.json').write_text(json.dumps(diagnostics, indent=2))
        jobs = []
        for arm in arms():
            folder = out / 'training' / arm['name']
            cmd = fit_command(out / 'features' / arm['owner'] / 'train/features.pt', folder, settings[arm['name']])
            jobs.append((arm['name'], cmd, folder, ('best.pt', 'initial.pt', 'history.jsonl')))
        workflow.run(jobs, 'train', gpu=True)
        for arm in arms():
            folder = out / 'training' / arm['name']
            for name in ('best.pt', 'initial.pt', 'summary.json', 'EXIT_CODE'):
                inputs[str(folder / name)] = sha(folder / name)
            fits[arm['name']] = validate_fit(root, folder, arm, settings[arm['name']])
        jobs = []
        for seed in (193, 211):
            owner = f'physical_base_s{seed}'
            for cohort in NORMAL_COUNTS:
                folder = out / 'blend' / owner / cohort
                cmd = [sys.executable, '-m', 'nmr_transfer.h_global_ranker_blend', '--stage', 'train_val',
                    '--no-aug-checkpoint', str(out / 'training' / f'physical_mlp_no_aug_s{seed}/best.pt'),
                    '--aug-checkpoint', str(out / 'training' / f'physical_mlp_aug_s{seed}/best.pt'),
                    '--cache', str(out / 'features' / owner / cohort / 'features.pt'), '--out', str(folder)]
                jobs.append((owner + '_' + cohort, cmd, folder,
                    tuple(f'gamma_{g}/{name}' for g in GAMMAS for name in ('predictions.jsonl', 'missing_predictions.jsonl', 'summary.json'))))
        workflow.run(jobs, 'blend')
        for _, _, folder, _ in jobs:
            for name in ('summary.json', 'EXIT_CODE'):
                inputs[str(folder / name)] = sha(folder / name)
        reports = {f'physical_base_s{s}': {c: json.loads((out / 'blend' / f'physical_base_s{s}' / c / 'summary.json').read_text()) for c in NORMAL_COUNTS} for s in (193, 211)}
        results['pool1024'] = aggregate(reports)
        controls = {}
        for seed in (193, 211):
            owner = f'physical_base_s{seed}'
            controls[owner] = {}
            for cohort in NORMAL_COUNTS:
                folder = root / 'blend/base' / owner / cohort
                if (folder / 'EXIT_CODE').read_text().strip() != '0':
                    raise ValueError('pool512_blend_control_not_complete')
                controls[owner][cohort] = json.loads((folder / 'summary.json').read_text())
                saved = controls[owner][cohort]
                if (saved['cache_sha256'] != inputs[str(root / 'calibration/features' / owner / cohort / 'features.pt')]
                        or saved['no_aug_sha256'] != inputs[str(reference_paths(root, dict(seed=seed, augment=False))[0])]
                        or saved['aug_sha256'] != inputs[str(reference_paths(root, dict(seed=seed, augment=True))[0])]):
                    raise ValueError('pool512_blend_control_origin_changed')
        results['pool512'] = aggregate(controls)
        if args.metadata_route:
            route_jobs = []
            for name, cmd, folder, _ in jobs:
                routed = out / 'metadata_route' / folder.relative_to(out / 'blend')
                command = list(cmd)
                command[2] = 'nmr_transfer.h_global_metadata_route'
                command[command.index('--out') + 1] = str(routed)
                command += ['--blend-dir', str(folder)]
                route_jobs.append((name, command, routed, ('gamma_0.0/predictions.jsonl',)))
            workflow.run(route_jobs, 'metadata_route')
            for _, _, folder, _ in route_jobs:
                for name in ('summary.json', 'EXIT_CODE'):
                    inputs[str(folder / name)] = sha(folder / name)
            routed = {f'physical_base_s{s}': {c: json.loads((out / 'metadata_route' / f'physical_base_s{s}' / c / 'summary.json').read_text()) for c in NORMAL_COUNTS} for s in (193, 211)}
            results['metadata_route1024'] = aggregate(routed)
        workflow.guard()
        for path, digest in inputs.items():
            budget.check()
            if sha(path) != digest:
                raise ValueError('pool_input_changed_during_run:' + path)
        (out / 'summary.json').write_text(json.dumps(dict(comparison_complete=True, results=results,
            diagnostics=diagnostics, fits=fits, source_sha256=workflow.sources, inputs_sha256=inputs,
            selection_scope='train_val_only', external_api_calls=0), indent=2))
        workflow.status('complete')
        (out / 'EXIT_CODE').write_text('0\n')
        return 0
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_error = None
        if workflow:
            try:
                terminate_owned(workflow.active)
            except Exception as cleanup:
                cleanup_error = repr(cleanup)
        pending = isinstance(exc, DeadlineReached) and cleanup_error is None
        record = dict(comparison_complete=False, state='pending' if pending else 'failed',
            error=repr(exc), cleanup_error=cleanup_error, completed_results=results,
            diagnostics=diagnostics, completed_fits=fits, external_api_calls=0)
        (out / 'partial_summary.json').write_text(json.dumps(record, indent=2))
        if workflow:
            workflow.status(record['state'], error=repr(exc), cleanup_error=cleanup_error)
        (out / 'EXIT_CODE').write_text('124\n' if pending else '1\n')
        if pending:
            return 124
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())

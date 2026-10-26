"""功能概述：独立核验本轮八份 C5/C10 续训产物，不执行模型或训练。

verify_driver 核真实 PID/argv/UTC/双退出及始末指纹；replay_training 从原 train
缓存、候选身份与 heldout 契约重算严格负例资格和 64 epoch 的配对抽样流。
audit_model 对比原 best、initial、final 的 dtype/shape/逐元素值，核 O、36 中心、
完整 scale 固定，核历史/摘要/预算与每 epoch 三类曝光。main 对完整输入闭包
核始末 SHA，并输出 summary.json 和真实 EXIT_CODE。只允许已独审的审计源码。
输入为受审 train queue/bindings/protocol 及既有训练输出；不读取 test 或运行评分。
"""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

SCHEMA = 'cap10_training_audit_20261024'
MODEL_SCHEMA = 'nmr_cap_continuation_20261024'
MASK_POLICY = 'external_heldout_OR_ranker_internal_OR_same_query_group; target_loss_only'
BUDGET = dict(epochs=64, samples=8192, batch_size=32, lr=.001, weight_decay=.001,
              anchor=.1, temperature=.5, clip_grad_norm=2., draws_per_fit=524288,
              optimizer_steps=16384)
FIELDS = ('query_order', 'variant', 'source', 'record_id', 'smiles', 'group')


def require(ok, name):
    if not ok:
        raise ValueError(name)


def sha(path):
    path = safe_input(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(safe_input(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_hashes(state, scale):
    return dict({k: tensor_sha(v) for k, v in state.items()}, __scale__=tensor_sha(scale))


def frozen_hash(state, scale):
    values = {k: v for k, v in state_hashes(state, scale).items() if not k.startswith('network.')}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def ordered_sha(values):
    return hashlib.sha256('\n'.join(values).encode()).hexdigest()


def internal(group):
    return int(hashlib.sha256(('ranker_internal_317' + group).encode()).hexdigest()[:12], 16) % 5 == 0


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0, 'UTC_required')
    return parsed


def zero(value):
    require(type(value) is int and value == 0, 'strict_integer_exit0')


def safe_input(path):
    """每次实际读取前重核词法路径和解析路径；不跟随指向禁区的新 symlink。"""
    raw = Path(path).absolute()
    require('..' not in raw.parts, 'forbidden_audit_input')
    resolved = raw.resolve(strict=True)
    for candidate in (raw, resolved):
        text = str(candidate).lower()
        require('/test/' not in text and '/sealed' not in text and candidate.name.lower() not in
                ('test.pt', 'test.json', 'test.jsonl', 'predictions.jsonl', 'cohorts.pt', 'sealed_test_report.md'),
                'forbidden_audit_input')
    return resolved


class Inputs:
    def __init__(self):
        self.before = {}

    def pin(self, path, expected=None):
        path = safe_input(path)
        text = str(path)
        actual = self.before[text] if text in self.before else sha(path)
        require(expected is None or actual == expected, 'input_SHA:' + text)
        require(text not in self.before or self.before[text] == actual, 'input_changed_before_use:' + text)
        self.before[text] = actual
        return path

    def json(self, path, expected=None):
        path = self.pin(path, expected)
        require(safe_input(path) == path, 'audit_canonical_input_redirected')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == self.before[str(path)], 'JSON_read_race')
        return json.loads(raw)

    def lines(self, path):
        path = self.pin(path)
        require(safe_input(path) == path, 'audit_canonical_input_redirected')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == self.before[str(path)], 'JSONL_read_race')
        return [json.loads(line) for line in raw.splitlines() if line]

    def finish(self):
        # 先准入整个闭包，才允许重新哈希任何文件，避免先打开后发现禁区。
        for p in self.before:
            require(safe_input(p) == Path(p), 'audit_canonical_input_redirected')
        after = {p: sha(p) for p in self.before}
        require(after == self.before, 'audit_input_changed_at_finish')
        return after


def verify_driver(root, queue_path, q, inputs, run=None):
    root = Path(root).resolve()
    run_name = q.get('run_name', q['phase'])
    run = Path(run).resolve() if run is not None else root / 'runs' / run_name
    require(run == root / 'runs' / run_name and q['phase'] == 'train', 'actual_successful_run_identity')
    start = inputs.json(run / 'driver_start.json')
    complete = inputs.json(run / 'completion.json')
    status = inputs.json(run / 'status.json')
    launch = inputs.json(run / 'launch_receipt.json')
    claim = inputs.json(root / (run_name + '.operator_claim.json'))
    launch_claim = inputs.json(root / (run_name + '.launch_claim.json'))
    queue_digest = inputs.before[str(Path(queue_path).resolve())]
    expected = [q['python'], '-B', str(root / 'supervise_controlled_queue.py'), '--queue', str(queue_path),
                '--out', str(run), '--execute']
    require(launch['argv'] == expected and start['argv'] == expected[2:], 'driver_exact_argv')
    require(start['queue_sha256'] == launch['queue_sha256'] == complete['queue_sha256'] == claim['queue_sha256'] == launch_claim['queue_sha256'] == queue_digest, 'driver_queue_binding')
    require(type(start['pid']) is int and start['pid'] > 0 and start['pid'] == launch['child_pid'] == claim['pid'] == complete['pid'], 'driver_PID')
    require(start['parent_pid'] == launch['wrapper_pid'] == launch_claim['wrapper_pid'], 'wrapper_PID')
    require(all(type(x) is int and x>0 for x in (start['parent_pid'],launch['wrapper_pid'],launch_claim['wrapper_pid'])), 'strict_wrapper_PID_type')
    require(all(x['phase']=='train' and x['run_name']==run_name for x in (start,complete,launch)), 'driver_phase_run_name')
    require(start['source_sha256'] == q['source_sha256'] and start['input_sha256'] == q['input_sha256'], 'driver_declared_inputs_sources')
    zero(launch['internal_exit']); zero(launch['external_exit']); zero(complete['external_exit'])
    for name in ('EXIT_CODE', 'EXTERNAL_EXIT_CODE'):
        p = inputs.pin(run / name)
        require(p.read_text().strip() == '0', 'driver_exit_file')
    require(complete['status'] == 'COMPLETE' and complete['inputs_unchanged'] is True and complete['error'] is None, 'driver_complete')
    require(status['phase'] == 'COMPLETE' and status['completed'] == 8 and status['running'] == [] and status['pending'] == 0 and status['error'] is None, 'driver_final_status')
    require(utc(launch['started_utc']) <= utc(start['started_utc']) <= utc(complete['finished_utc']) <= utc(launch['finished_utc']), 'driver_UTC_order')
    require((utc(complete['finished_utc']) - utc(start['started_utc'])).total_seconds() <= q['maximum_seconds'] + 30, 'driver_time_budget')
    before = inputs.json(run / 'input_fingerprints.before.json')
    after = inputs.json(run / 'input_fingerprints.after.json')
    require(before == after == q['input_sha256'], 'driver_input_begin_end')
    small_before = inputs.json(run / 'operator_fingerprints.before.json')
    small_after = inputs.json(run / 'operator_fingerprints.after.json')
    mandatory = set(q['source_sha256']) | {str(queue_path)} | set(q['review_paths'])
    require(small_before == small_after and set(small_before) == mandatory, 'operator_small_begin_end_closure')
    for path, value in small_before.items():
        inputs.pin(path, value)
    by_id = {r['id']: r for r in complete['jobs']}
    require(len(complete['jobs']) == len(by_id) == 8 and set(by_id) == {j['id'] for j in q['jobs']}, 'complete_eight_job_matrix')
    for job in q['jobs']:
        r = inputs.json(run / 'jobs' / (job['id'] + '.receipt.json'))
        started = inputs.json(run / 'jobs' / (job['id'] + '.started.json'))
        access = inputs.json(job['access_log'])
        require(r == by_id[job['id']] and all(r[k] == v for k, v in started.items()), 'job_receipt_start_completion_same')
        zero(r['internal_exit']); zero(r['external_exit']); zero(access['internal_exit'])
        require(type(r['pid']) is int and r['pid'] > 0 and r['pid'] == access['pid'] and r['parent_pid'] == access['parent_pid'] == start['pid'], 'job_PID_parent')
        require(all(type(x) is int and x>0 for x in (access['pid'],r['parent_pid'],access['parent_pid'])), 'strict_child_PID_types')
        require(r['argv'] == job['argv'] and r['cwd'] == job['cwd'] and r['gpu'] == job['gpu'] and r['job_out'] == job['job_out'] and r['access_log'] == job['access_log'], 'job_exact_argv_layout')
        require(r['queue_sha256'] == queue_digest and r['requested_inputs_sha256'] == {p: before[p] for p in job['allowed_reads']}, 'job_input_hashes')
        require(access['phase'] == 'train' and access['argv'] == job['command'] and access['allowed_reads'] == sorted(job['allowed_reads']) and access['rejected_paths'] == [] and access['error'] is None, 'job_isolation_receipt')
        require(utc(start['started_utc']) <= utc(r['started_utc']) <= utc(access['started_utc']) <= utc(access['finished_utc']) <= utc(r['finished_utc']) <= utc(complete['finished_utc']), 'job_UTC_envelope')
        for path in job['allowed_reads']:
            require(access['recorded_reads'].get(path, 0) > 0, 'required_tensor_actually_opened')
        for path in access['recorded_reads']:
            require(path in job['allowed_reads'] or Path(path).resolve().is_relative_to(Path(job['job_out']).resolve()), 'unregistered_recorded_science_read')
        fit_exit = inputs.pin(Path(job['job_out']) / 'EXIT_CODE')
        require(fit_exit.read_text().strip() == '0', 'fit_actual_exit_file')
    return dict(wrapper_pid=launch['wrapper_pid'], driver_pid=start['pid'], jobs=8,
                queue_sha256=queue_digest, started_utc=start['started_utc'], finished_utc=complete['finished_utc'])


def replay_training(single, dual, candidate_groups, heldout, role, seed):
    require(single['smiles'] == dual['smiles'] and len(candidate_groups) == len(single['smiles']), 'candidate_mapping_identity')
    key = lambda r: tuple(r[k] for k in FIELDS)
    other = {key(r): r for r in dual['rows']}
    fit = [r for r in single['rows'] if (role == 'aug' or r['variant'] == 'observed') and not internal(r['group'])]
    require(fit and all(r['group'] not in heldout for r in fit), 'fit_query_eligibility')
    groups = collections.defaultdict(list)
    for row in fit:
        groups[row['group']].append(row)
    negative_records, mask_by_key = [], {}
    for row in sorted(fit, key=key):
        for pool, selected in ((0, row), (1, other[key(row)])):
            indices = selected['candidate_indices'].cpu()
            historical = selected['negative_exclusion_mask'].cpu()
            require(indices.dtype in (torch.int32, torch.int64) and len(indices) in (512,513) and len(indices.unique()) == len(indices), 'audit_candidate_indices')
            require(historical.dtype == torch.bool and historical.shape == indices.shape, 'audit_historical_mask_shape')
            require(bool(((indices >= 0) & (indices < len(candidate_groups))).all()), 'audit_candidate_range')
            names = [candidate_groups[int(i)] for i in indices]
            expected = torch.tensor([g == row['group'] or g in heldout for g in names])
            require(torch.equal(historical, expected), 'audit_historical_mask_semantics')
            strict = expected | torch.tensor([internal(g) for g in names])
            target = selected['target_position']
            require(type(target) is int and 0 <= target < len(indices) and names[target] == row['group'], 'audit_target_identity')
            strict[target] = True
            require(bool((~strict[:512]).any()), 'audit_empty_natural_negatives')
            ids = indices[~strict].tolist()
            record = dict(query_key=list(key(row)), pool=pool, negatives=len(ids),
                ordered_eligible_candidate_sha256=ordered_sha([str(i) for i in ids]), excluded_mask_sha256=tensor_sha(strict))
            negative_records.append(record)
            mask_by_key[(key(row), pool)] = bytes.fromhex(record['excluded_mask_sha256'])
    qualification_sha = hashlib.sha256(json.dumps(negative_records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    exposure, pools, negative = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    records = []
    names = sorted(groups)
    for epoch in range(1, BUDGET['epochs'] + 1):
        grng = np.random.default_rng(seed + 2000 + epoch)
        vrng = np.random.default_rng(seed + 4000 + epoch)
        sequence = grng.integers(0, len(names), size=BUDGET['samples'])
        choices = np.arange(BUDGET['samples']) % 2
        np.random.default_rng(seed + 6000 + epoch).shuffle(choices)
        local = [hashlib.sha256() for _ in range(3)]
        for pos, index in enumerate(sequence):
            variants = groups[names[int(index)]]
            row = variants[int(vrng.integers(len(variants)))]
            pool = int(choices[pos])
            payloads = (json.dumps(key(row), separators=(',', ':')).encode() + b'\n', bytes([pool]), mask_by_key[(key(row), pool)])
            for total, part, payload in zip((exposure,pools,negative),local,payloads):
                total.update(payload); part.update(payload)
        records.append(dict(epoch=epoch, draws=BUDGET['samples'], pool_draws={'0':BUDGET['samples']//2,'1':BUDGET['samples']//2},
            optimizer_steps=BUDGET['samples']//BUDGET['batch_size'], cumulative_draws=epoch*BUDGET['samples'],
            cumulative_optimizer_steps=epoch*BUDGET['samples']//BUDGET['batch_size'],
            exposure_sha256=local[0].hexdigest(), pool_exposure_sha256=local[1].hexdigest(), negative_exposure_sha256=local[2].hexdigest(),
            cumulative_exposure_sha256=exposure.hexdigest(), cumulative_pool_exposure_sha256=pools.hexdigest(), cumulative_negative_exposure_sha256=negative.hexdigest()))
    return dict(records=records, negative_records=negative_records, qualification_sha256=qualification_sha,
                exposure_sha256=exposure.hexdigest(), pool_exposure_sha256=pools.hexdigest(),
                negative_exposure_sha256=negative.hexdigest(), groups=len(groups), rows=len(fit))


def model_provenance(directory, inputs, admitted_pins=None, expected_cli=None):
    """独立列举 trainer 实际必需的完整闭包，不从 checkpoint 自愿保留的键推导。"""
    producer = inputs.json(directory / 'protocol.json')
    args = producer['args']
    require(set(args) == {'source','fixed_module','protocol','bindings','out','seed','cap','role','device'},
            'producer_exact_CLI_fields')
    require(safe_input(args['out']) == directory, 'producer_actual_outpath')
    if expected_cli is not None:
        require(args == expected_cli, 'producer_CLI_exact_actual_job')
    bindings = inputs.json(args['bindings'])
    protocol = inputs.json(args['protocol'])
    require(producer['protocol'] == protocol, 'producer_actual_protocol')
    source = safe_input(args['source'])
    require(source == safe_input(bindings['source_checkout']), 'producer_bound_scientific_source')
    record = bindings['roles'][str(args['seed'])][args['role']]
    caches = bindings['caches'][str(args['seed'])]['train']
    # 与已审 runtime 源列表相同，但此处不导入 trainer/cap，也不信任 saved 的列表。
    names = ('h_pool_ranker.py','h_research_ranker.py','h_research_contract.py',
             'h_research_observation.py','h_global_ranker.py','h_global_features.py','data.py')
    sources = [source/'nmr_transfer'/name for name in names]
    sources += [Path(args['fixed_module']),Path(__file__).resolve().parent/'cap_ranker.py',
                Path(__file__).resolve().parent/'train_cap_ranker.py']
    paths = sources + [Path(args['protocol']),Path(args['bindings'])]
    paths += [Path(record[key]) for key in ('checkpoint','centers','center_rows','producer_summary')]
    paths += [Path(record['checkpoint']).parent/'EXIT_CODE']
    for pool in ('single','dual'):
        entry = caches[pool]
        require(Path(entry['producer']).resolve() == Path(entry['path']).resolve().parent/'summary.json',
                'model_cache_producer_path')
        paths += [Path(entry['path']),Path(entry['producer']),Path(entry['path']).parent/'EXIT_CODE']
    contract = Path(bindings['cohort_contract'])
    paths += [contract,contract.parent/'manifest.json',contract.parent/'EXIT_CODE']
    complete = {}
    for path in paths:
        path = safe_input(path)
        wanted = None if path == safe_input(args['bindings']) else bindings['input_sha256'].get(str(path))
        require(path == safe_input(args['bindings']) or wanted is not None, 'mandatory_model_input_binding')
        if admitted_pins is not None:
            require(str(path) in admitted_pins and (wanted is None or admitted_pins[str(path)] == wanted),
                    'unadmitted_model_required_input')
        complete[str(path)] = inputs.before[str(inputs.pin(path,wanted))]
    require(producer['input_sha256'] == complete, 'producer_complete_input_closure')
    ranker_sources = {str(safe_input(path)):complete[str(safe_input(path))] for path in sources}
    saved_args = dict(args,augment=args['role']=='aug',training_pool='mixed',loss='listwise',
                      **{k:v for k,v in BUDGET.items() if k not in args})
    return complete,ranker_sources,saved_args,bindings


def audit_model(directory, parent_path, record, cap_value, seed, role, expected, inputs,
                admitted_pins=None, expected_cli=None):
    directory = safe_input(directory)
    complete_inputs, complete_sources, complete_args, bindings = model_provenance(
        directory,inputs,admitted_pins,expected_cli)
    summary = inputs.json(directory / 'summary.json')
    history = inputs.lines(directory / 'history.jsonl')
    qualification = inputs.json(directory / 'negative_qualification.json')
    initial_hashes = inputs.json(directory / 'initial_tensor_sha256.json')
    initial = torch.load(inputs.pin(directory / 'initial.pt'), map_location='cpu', weights_only=False, mmap=True)
    final_path = inputs.pin(directory / 'final.pt', summary['final_checkpoint_sha256'])
    final = torch.load(final_path, map_location='cpu', weights_only=False, mmap=True)
    parent = torch.load(inputs.pin(parent_path, record['checkpoint_sha256']), map_location='cpu', weights_only=False, mmap=True)
    state, scale = parent['state_dict'], parent['scale']
    require(scale.shape == (33,) and bool(torch.isfinite(scale).all()) and bool((scale > 0).all()), 'parent_scale_shape_finite')
    expected_initial = state_hashes(state, scale)
    require(initial_hashes == expected_initial, 'initial_tensor_manifest_exact_parent')
    require(summary['initial_checkpoint_sha256'] == inputs.before[str((directory / 'initial.pt').resolve())], 'initial_payload_SHA')
    for saved, epoch, draws, steps in ((initial,0,0,0),(final,64,BUDGET['draws_per_fit'],BUDGET['optimizer_steps'])):
        require(saved['schema'] == MODEL_SCHEMA and saved['observation'] is True and saved['initial_sigmas']==parent['initial_sigmas'] and saved['delta_cap'] == cap_value and type(saved['delta_cap']) is int and saved['seed'] == seed and saved['role'] == role, 'new_checkpoint_schema_cap_role')
        require(type(saved['epoch']) is int and saved['epoch'] == epoch and saved['draws'] == draws and type(saved['draws']) is int and saved['optimizer_steps'] == steps and type(saved['optimizer_steps']) is int, 'checkpoint_budget')
        require(type(saved['alpha']) is not bool and saved['alpha'] == 1 and saved['budget'] == BUDGET and saved['negative_mask_policy'] == MASK_POLICY, 'checkpoint_fixed_contract')
        args = saved['args']
        require(args['seed'] == seed and args['role'] == role and args['augment'] is (role == 'aug') and args['cap'] == cap_value and args['training_pool'] == 'mixed' and args['loss'] == 'listwise', 'checkpoint_args_role')
        require(all(args[k] == v and type(args[k]) is not bool for k,v in BUDGET.items()), 'checkpoint_all_budget_args')
        require(args == complete_args, 'checkpoint_all_args_exact_producer_and_job')
        require(saved['continuation_epoch_offset'] == 0 and saved['optimizer_reset'] is True and saved['selection'] == 'fixed_final_epoch64_alpha1_no_selection', 'no_checkpoint_or_alpha_selection')
        require(Path(saved['parent_checkpoint']).resolve()==Path(parent_path).resolve() and saved['parent_checkpoint_sha256'] == record['checkpoint_sha256'] and saved['parent_epoch'] == parent['epoch'] and saved['parent_alpha'] == parent['alpha'], 'parent_provenance')
        require(saved['initial_tensor_sha256'] == expected_initial and set(saved['state_dict']) == set(state), 'all_state_keys_and_initial_hashes')
        for name, value in saved['state_dict'].items():
            require(value.dtype == state[name].dtype and value.shape == state[name].shape and bool(torch.isfinite(value).all()), 'state_tensor_shape_dtype_finite:' + name)
            if epoch == 0 or not name.startswith('network.'):
                require(torch.equal(value, state[name]), 'initial_or_frozen_state_changed:' + name)
        require(saved['scale'].dtype == scale.dtype and saved['scale'].shape == scale.shape and torch.equal(saved['scale'], scale), 'all33_scale_exact_parent')
        require(saved['frozen_tensor_sha256'] == frozen_hash(state,scale) == frozen_hash(saved['state_dict'], saved['scale']), 'frozen_digest_exact')
        require(saved['strict_negative_qualification_sha256'] == expected['qualification_sha256'], 'strict_qualification_independent')
        require(saved['center_estimate_sha256'] == record['centers_sha256'], 'frozen_center_metadata')
        require(all(saved[key] == parent[key] for key in
                    ('train_sha256','dual_train_sha256','cache_protocol','source_inputs_sha256')),
                'parent_training_and_forward_lineage')
        caches = bindings['caches'][str(seed)]['train']
        require(saved['train_sha256'] == complete_inputs[str(safe_input(caches['single']['path']))] and
                saved['dual_train_sha256'] == complete_inputs[str(safe_input(caches['dual']['path']))],
                'parent_actual_training_cache_pair')
        for path, value in saved['ranker_source_sha256'].items():
            require(admitted_pins is None or admitted_pins.get(path)==value, 'unadmitted_model_source')
        for path, value in saved['input_sha256'].items():
            require(admitted_pins is None or admitted_pins.get(path)==value, 'unadmitted_model_input')
        require(saved['ranker_source_sha256'] == complete_sources, 'complete_model_source_closure')
        require(saved['input_sha256'] == complete_inputs, 'complete_model_input_closure')
    require(all(summary.get(k) == v for k,v in final.items() if k not in ('state_dict','scale','initial_sigmas')), 'summary_all_saved_fields_exact')
    require(summary['source_inputs_start_sha256'] == summary['source_inputs_end_sha256'] == final['input_sha256'], 'producer_input_start_end_exact')
    require(all(summary[k] is True for k in ('centers_frozen','scale_frozen','observation_frozen','initial_tensors_inherited_exact')), 'producer_freeze_flags')
    require(summary['epoch_records'] == history and len(history) == 64, 'complete_real64_history')
    for actual, wanted in zip(history, expected['records']):
        require(all(actual[k] == v for k,v in wanted.items()), 'independent_epoch_exposure_budget')
        require(math_finite(actual['mean_training_loss']) and actual['frozen_tensor_sha256'] == frozen_hash(state,scale), 'history_finite_frozen')
    for key in ('exposure_sha256','pool_exposure_sha256','negative_exposure_sha256'):
        require(final[key] == expected[key], 'independent_final_' + key)
    require(qualification['policy'] == MASK_POLICY and qualification['records'] == expected['negative_records'] and qualification['sha256'] == expected['qualification_sha256'], 'all_negative_qualification_rows')
    require(summary['training_groups'] == expected['groups'] and summary['training_rows'] == expected['rows'], 'fit_group_variant_count')
    center_path = inputs.pin(directory / 'centers.json', record['centers_sha256'])
    rows_path = inputs.pin(directory / 'center_rows.jsonl', final['center_rows_sha256'])
    require(sha(rows_path) == sha(inputs.pin(record['center_rows'])), 'center_row_bytes_inherited')
    center = inputs.json(center_path)
    require(center['row_audit_sha256'] == final['center_rows_sha256'] and torch.equal(state['fixed_x_center'], torch.tensor(center['center33'],dtype=torch.float32)) and torch.equal(state['fixed_o_center'],torch.tensor(center['O_center3'],dtype=torch.float32)), 'center36_JSON_tensor_binding')
    return dict(cap=cap_value,seed=seed,role=role,checkpoint=str(final_path),checkpoint_sha256=sha(final_path),
        centers_sha256=record['centers_sha256'],center_rows_sha256=final['center_rows_sha256'],epochs=64,alpha=1,
        draws=BUDGET['draws_per_fit'],optimizer_steps=BUDGET['optimizer_steps'],parent_checkpoint_sha256=record['checkpoint_sha256'],
        strict_negative_qualification_sha256=expected['qualification_sha256'],initial_tensor_sha256=expected_initial,
        exposure_sha256=expected['exposure_sha256'],pool_exposure_sha256=expected['pool_exposure_sha256'],negative_exposure_sha256=expected['negative_exposure_sha256'])


def math_finite(value):
    return type(value) in (int,float) and np.isfinite(value)


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('root','run','queue','bindings','protocol','source','implementation-review','out'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    inputs = Inputs()
    try:
        review = inputs.json(args.implementation_review)
        own = inputs.pin(__file__)
        require(review['status'] == 'PASS' and review['independent'] is True and review['reviewed_sha256'].get(str(own)) == sha(own), 'audit_implementation_review_required')
        q = inputs.json(args.queue)
        b = inputs.json(args.bindings)
        protocol = inputs.json(args.protocol)
        root = Path(args.root).resolve()
        require(q['phase'] == 'train' and q['root'] == str(root) and q['bindings'] == str(Path(args.bindings).resolve()) and q['input_sha256'] == b['input_sha256'], 'audit_queue_binding')
        require(protocol['budget']['draws_per_fit'] == BUDGET['draws_per_fit'] and protocol['budget']['optimizer_steps'] == BUDGET['optimizer_steps'], 'audit_protocol_budget')
        for path,value in dict(q['input_sha256'],**q['source_sha256']).items():
            inputs.pin(path,value)
        for path in q['review_paths']:
            admission = inputs.json(path)
            require(admission['status']=='PASS' and admission['independent'] is True and admission['admission_phase']=='train', 'actual_training_admission')
            for p,value in dict(q['source_sha256'],**{str(Path(args.queue).resolve()):sha(args.queue)}).items():
                require(admission['reviewed_sha256'].get(p)==value, 'training_admission_byte_binding')
        execution = verify_driver(root,Path(args.queue).resolve(),q,inputs,args.run)
        source = Path(args.source).resolve()
        require(source == Path(b['source_checkout']).resolve(), 'audit_scientific_source')
        sys.path.insert(0,str(source))
        base = importlib.import_module('nmr_transfer.h_pool_ranker')
        identity = importlib.import_module('nmr_transfer.data')
        require(Path(base.__file__).resolve() == source/'nmr_transfer/h_pool_ranker.py' and Path(identity.__file__).resolve() == source/'nmr_transfer/data.py','wrong_audit_source_import')
        contract = inputs.json(b['cohort_contract'])
        heldout = set(contract['training_excluded_groups'])
        require(len(heldout)==len(contract['training_excluded_groups'])==b['heldout_groups'] and ordered_sha(sorted(heldout))==b['heldout_groups_sha256'], 'heldout_connectivity_contract')
        models = []
        for seed in (193,211):
            loaded = {}
            for pool in ('single','dual'):
                entry = b['caches'][str(seed)]['train'][pool]
                value,_ = base.read_feature_cache(entry['path'],known_sha=inputs.before[str(Path(entry['path']).resolve())])
                require(all(entry['manifest'].get(k)==v for k,v in value['manifest'].items()) and value['manifest']['cohort']=='train' and value['manifest']['recall']==pool,'audit_loaded_cache_manifest')
                loaded[pool]=value
            base.validate_caches(loaded['single'],loaded['dual'])
            candidate_groups = [identity.identity(s)[1] for s in loaded['single']['smiles']]
            require(len(candidate_groups)==b['candidate_count']==114982 and ordered_sha(loaded['single']['smiles'])==b['candidate_order_sha256'],'audit_candidate_order_SHA')
            for role in ('no_aug','aug'):
                original = b['roles'][str(seed)][role]
                require(ordered_sha(candidate_groups)==original['qualification']['derived_candidate_groups_sha256'],'audit_candidate_group_SHA')
                expected = replay_training(loaded['single'],loaded['dual'],candidate_groups,heldout,role,seed)
                paired=[]
                for c in (5,10):
                    directory=root/f'rankers/C{c}/s{seed}/{role}'
                    job_id=f'fit_C{c}_s{seed}_{role}'
                    jobs=[job for job in q['jobs'] if job['id']==job_id]
                    require(len(jobs)==1, 'exact_model_job_identity')
                    command=jobs[0]['command']
                    require(len(command)%2==1 and all(command[i].startswith('--') for i in range(1,len(command),2)), 'actual_training_CLI_layout')
                    cli={command[i][2:].replace('-','_'):command[i+1] for i in range(1,len(command),2)}
                    require(len(cli)==(len(command)-1)//2, 'actual_training_CLI_duplicate')
                    for key in ('cap','seed'):cli[key]=int(cli[key])
                    row=audit_model(directory,original['checkpoint'],original,c,seed,role,expected,inputs,
                                    dict(q['input_sha256'],**q['source_sha256']),cli)
                    models.append(row);paired.append(row)
                for key in ('initial_tensor_sha256','strict_negative_qualification_sha256','exposure_sha256','pool_exposure_sha256','negative_exposure_sha256'):
                    require(paired[0][key]==paired[1][key], 'paired_arm_' + key)
        require(len(models)==8 and {(x['cap'],x['seed'],x['role']) for x in models} == {(c,s,r) for c in (5,10) for s in (193,211) for r in ('no_aug','aug')},'eight_model_matrix')
        after = inputs.finish()
        checks = {k:True for k in ('same_initial_tensors','frozen_observation','frozen_centers','frozen_scale',
            'matched_exposure','matched_pool_exposure','matched_negative_exposure','strict_negative_qualification',
            'finite','full_budget','real_driver_exit','source_input_unchanged')}
        result = dict(schema=SCHEMA,status='PASS',independent=True,scope='eight_actual_matched_cap_continuation_fits_training_only',
            utc=datetime.now(timezone.utc).isoformat(),models=models,checks=checks,inputs_unchanged=True,
            input_sha256=inputs.before,source_sha256=dict(q['source_sha256'],**{str(own):inputs.before[str(own)]}),queue_sha256=sha(args.queue),
            protocol_sha256=sha(args.protocol),bindings_sha256=sha(args.bindings),negative_mask_policy=MASK_POLICY,
            training_run=str(Path(args.run).resolve()),input_end_sha256=after,execution=execution,test_data_opened=False,new_model_calls=0,new_training_jobs=0)
        write(out/'summary.json',result)
        (out/'EXIT_CODE').write_text('0\n')
        print(json.dumps({'status':'PASS','models':8,'input_files':len(after)}))
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),input_sha256=inputs.before))
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

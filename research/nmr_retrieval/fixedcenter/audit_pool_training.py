"""功能概述：只读核验本轮16个训练产物，禁止用测试输出作选择或读入测试缓存。
verify_driver核验完整48任务driver/launch/receipt来源、UTC顺序和严格整数双EXIT；
audit从formal_queue定位完整四臂×双seed×增强组合，不读验证预测或测试缓存。
audit_model比较旧initial与新initial的全部tensor、scale及alpha，检查initial/best/last
观测参数完全冻结与数值有限，并核对summary/history及末轮64×8192预算。
recompute_exposure仅从train缓存提取查询身份，独立重放numpy随机流，重算各检查点曝光
摘要和混池计数；不调用训练、评估或优化器。main独占新建输出目录，记录输入始末SHA、
失败和真实内部EXIT。依赖torch/numpy及已封存队列/训练产物，不改变科学源与输入。
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

SCHEMA = 'nmr_pool_ranker_20261023'
BUDGET = dict(epochs=64, samples=8192, batch_size=32, anchor=.1, lr=.001)
CHECK_EPOCHS = list(range(0, 65, 8))
KEY_FIELDS = ('query_order', 'variant', 'source', 'record_id', 'smiles', 'group')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def exact(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def recompute_exposure(rows, seed, augment, epochs=64, samples=8192):
    groups = collections.defaultdict(list)
    internal = set()
    for row in rows:
        if not augment and row['variant'] != 'observed':
            continue
        group = row['group']
        if int(hashlib.sha256(('ranker_internal_317'+str(group)).encode()).hexdigest()[:12], 16) % 5 == 0:
            internal.add(group)
        else:
            groups[group].append(tuple(row[k] for k in KEY_FIELDS))
    require(groups and internal, 'empty_internal_or_training_groups')
    names = sorted(groups)
    exposure = hashlib.sha256()
    pools = {'original': hashlib.sha256(), 'mixed': hashlib.sha256()}
    snapshots = {0: dict(exposure=exposure.hexdigest(), original=pools['original'].hexdigest(),
                        mixed=pools['mixed'].hexdigest(), draws=0, dual_draws=0)}
    dual_draws = 0
    for epoch in range(1, epochs+1):
        sequence = np.random.default_rng(seed+2000+epoch).integers(0, len(names), size=samples)
        variants = np.random.default_rng(seed+4000+epoch)
        pool_ids = np.arange(samples) % 2
        np.random.default_rng(seed+6000+epoch).shuffle(pool_ids)
        for index, pool in zip(sequence, pool_ids):
            group_rows = groups[names[int(index)]]
            row = group_rows[int(variants.integers(len(group_rows)))]
            exposure.update(json.dumps(row, separators=(',', ':')).encode()+b'\n')
            pools['original'].update(b'\0')
            pools['mixed'].update(bytes([int(pool)]))
            dual_draws += int(pool)
        if epoch % 8 == 0 or epoch == epochs:
            snapshots[epoch] = dict(exposure=exposure.hexdigest(),
                original=pools['original'].hexdigest(), mixed=pools['mixed'].hexdigest(),
                draws=epoch*samples, dual_draws=dual_draws)
    return snapshots, len(groups), len(internal)


def audit_model(summary, protocol, old, checkpoints, history, expected, exposure):
    args = protocol['args']
    require(all(args.get(k) == v for k, v in BUDGET.items()), 'protocol_budget')
    require(all(args.get(k) == v for k, v in expected.items()), 'protocol_role')
    require(summary['epochs'] == 64 and summary['samples_per_epoch'] == 8192, 'summary_budget')
    require(summary['training_pool'] == args['training_pool'] and summary['loss'] == args['loss'], 'summary_role')
    require(summary['input_sha256'] == protocol['inputs_sha256'], 'summary_protocol_pins')
    require(finite(summary) and finite(history), 'nonfinite_summary_or_history')
    require([h['epoch'] for h in history] == CHECK_EPOCHS, 'history_incomplete')
    require(history[-1]['best'] == summary['best'], 'history_summary_best')
    require(summary['best']['epoch'] in CHECK_EPOCHS, 'best_epoch')
    initial, best, last = [checkpoints[k] for k in ('initial', 'best', 'last')]
    require(old['args']['seed'] == args['seed'] and old['args']['augment'] is args['augment'], 'old_initial_role')
    require(old['cache_protocol'].get('recall') == 'single' and old['observation'], 'old_initial_not_frozen_R0O1')
    require(exact(initial['state_dict'], old['state_dict']), 'initial_tensors_changed')
    require(exact(initial['scale'], old['scale']) and initial['alpha'] == old['alpha'], 'initial_scale_alpha_changed')
    obs = {k:v for k,v in old['state_dict'].items() if k.startswith('observation_model.')}
    require(bool(obs), 'missing_observation_tensors')
    for label, saved in checkpoints.items():
        require(saved['schema'] == SCHEMA and saved['observation'], 'checkpoint_schema')
        require(saved['args'] == args, 'checkpoint_args')
        require(finite(saved['state_dict']) and finite(saved['scale']) and finite(saved['alpha']), 'nonfinite_checkpoint')
        require(saved['state_dict'].keys() == old['state_dict'].keys(), 'state_keys_changed')
        require(all(isinstance(v, torch.Tensor) and isinstance(old['state_dict'][k], torch.Tensor)
                    and v.dtype == old['state_dict'][k].dtype and v.shape == old['state_dict'][k].shape
                    for k,v in saved['state_dict'].items()), 'state_tensor_capacity_changed')
        require(exact(saved['scale'], old['scale']), 'scale_changed')
        require(exact(saved['initial_sigmas'], old['initial_sigmas']), 'initial_sigmas_changed')
        require(exact({k:saved['state_dict'][k] for k in obs}, obs), 'observation_changed')
        require(saved['source_inputs_sha256'] == old['source_inputs_sha256'], 'science_pins_changed')
        require(saved['ranker_source_sha256'] and all(protocol['inputs_sha256'].get(p) == h
                    for p,h in saved['ranker_source_sha256'].items()), 'checkpoint_source_pins')
        for key, field in (('train', 'train_sha256'), ('dual_train', 'dual_train_sha256'),
                           ('initial_checkpoint', 'initialization_sha256')):
            require(saved[field] == protocol['inputs_sha256'][args[key]], 'checkpoint_input_pin:'+field)
        require(saved['epoch'] in exposure, 'checkpoint_epoch')
        snapshot = exposure[saved['epoch']]
        require(saved['exposure_sha256'] == snapshot['exposure'], 'checkpoint_exposure:'+label)
        require(saved['pool_exposure_sha256'] == snapshot[args['training_pool']], 'checkpoint_pool_exposure:'+label)
    require(initial['epoch'] == 0 and last['epoch'] == 64, 'initial_or_last_epoch')
    require(best['epoch'] == summary['best']['epoch'] and best['alpha'] == summary['best']['alpha'], 'best_checkpoint_mismatch')
    require(best['alpha'] in (0., .25, .5, 1.) and last['alpha'] == best['alpha'], 'alpha_grid')
    require(summary['exposure_sha256'] == exposure[64]['exposure'], 'summary_exposure')
    require(summary['pool_exposure_sha256'] == exposure[64][args['training_pool']], 'summary_pool_exposure')
    changes = summary['observation_parameter_max_changes']
    require(set(changes) == {k.removeprefix('observation_model.') for k in obs}
            and all(v == 0 for v in changes.values()), 'observation_declared_changes')
    require(old['train_sha256'] == initial['train_sha256'], 'inherited_training_cache')
    return dict(best_epoch=best['epoch'], last_epoch=64, draws=exposure[64]['draws'],
                optimizer_steps=exposure[64]['draws']//32, initial_tensor_count=len(initial['state_dict']),
                observation_tensor_count=len(obs), exposure_sha256=exposure[64]['exposure'],
                pool_exposure_sha256=exposure[64][args['training_pool']],
                dual_draws=exposure[64]['dual_draws'] if args['training_pool']=='mixed' else 0)


def utc(value):
    try:
        stamp = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
        require(stamp.tzinfo is not None, 'timezone_required')
        return stamp.astimezone(datetime.timezone.utc)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('invalid_completion_utc') from exc


def double_exit(record, label):
    require(all(type(record.get(k)) is int and record[k] == 0
                for k in ('external_exit','internal_exit')), label+'_not_double_exit0')


def verify_driver(run, queue_path, queue, pin):
    """核验完成的48任务来源闭环；只读receipt/EXIT，不读validation预测。"""
    for filename in ('EXIT_CODE','EXTERNAL_EXIT_CODE'):
        pin(run/filename)
        require((run/filename).read_text().strip() == '0', 'driver_exit_not_zero')
    for filename in ('launch_receipt.json','driver_start.json','completion.json'):
        pin(run/filename)
    launch = read_json(run/'launch_receipt.json')
    driver = read_json(run/'driver_start.json')
    completion = read_json(run/'completion.json')
    double_exit(launch, 'driver')
    start, end = utc(launch['started_utc']), utc(launch['finished_utc'])
    require(start <= end <= datetime.datetime.now(datetime.timezone.utc), 'invalid_driver_time_order')
    require(launch['queue_sha256'] == {str(queue_path): pin(queue_path)}, 'launch_queue_binding')
    require(launch['bundle_sha256'] == queue['source_inputs_sha256'], 'launch_bundle_binding')
    require(bool(launch['source_sha256']), 'missing_launcher_sources')
    for p,digest in launch['source_sha256'].items():
        require(queue['source_inputs_sha256'].get(p) == digest, 'launcher_source_binding')
        pin(p,digest)
    expected_sources = {**queue['source_inputs_sha256'],**launch['queue_sha256']}
    require(driver['inputs_sha256'] == expected_sources, 'driver_start_source_binding')
    require(type(driver['pid']) is int and driver['pid'] > 0 and driver['pid'] == launch['child_pid']
            and type(launch['child_pid']) is int
            and driver['argv'][1:] == launch['argv'][3:], 'driver_launch_identity')
    driver_start, driver_end = utc(driver['utc']), utc(completion['utc'])
    require(start <= driver_start <= driver_end <= end, 'invalid_driver_time_order')
    require(completion.get('success') is True and completion.get('inputs_unchanged') is True, 'driver_incomplete')
    jobs = queue['jobs']
    require(len(jobs) == 48 and len({j['id'] for j in jobs}) == 48
            and collections.Counter(j['kind'] for j in jobs) == {'training':16,'validation':32}, 'formal_48_job_matrix')
    require(set(completion['jobs']) == {j['id'] for j in jobs}, 'completion_job_set')
    require(len({j['out'] for j in jobs}) == 48, 'duplicate_job_output')
    for job in jobs:
        path = run/(job['id']+'.receipt.json')
        pin(path)
        receipt = read_json(path)
        require(receipt == completion['jobs'][job['id']], 'receipt_completion_binding')
        double_exit(receipt, 'job')
        require(receipt.get('state') == 'complete', 'job_incomplete')
        job_start, job_end = utc(receipt['started_utc']), utc(receipt['finished_utc'])
        require(driver_start <= job_start <= job_end <= driver_end, 'invalid_job_time_order')
        require(receipt['argv'] == job['argv'] and receipt['output'] == job['out']
                and receipt['cwd'] == job['cwd'], 'receipt_job_binding')
        require(receipt['queue_source_sha256'] == expected_sources, 'receipt_source_binding')
        require(receipt['requested_inputs_sha256'] == job['inputs_sha256'], 'receipt_input_binding')
        for dependency in job.get('deps', []):
            require(utc(completion['jobs'][dependency]['finished_utc']) <= job_start, 'dependency_time_order')
        pin(Path(job['out'])/'EXIT_CODE')
        require((Path(job['out'])/'EXIT_CODE').read_text().strip() == '0', 'job_producer_exit')


def audit(root, formal_run=None):
    root = Path(root).resolve()
    run = Path(formal_run).resolve() if formal_run else root/'runs/formal'
    queue_path = root/'formal_queue.json'
    pins = {}
    def pin(path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        actual = pins.get(key)
        if actual is None:
            actual = sha(path)
            pins[key] = actual
        require(expected is None or actual == expected, 'input_pin_mismatch:'+key)
        return actual
    pin(queue_path)
    pin(__file__)
    queue = read_json(queue_path)
    verify_driver(run, queue_path, queue, pin)
    jobs = [j for j in queue['jobs'] if j['kind']=='training']
    require(len(jobs)==16 and len({j['id'] for j in jobs})==16, 'expected_16_unique_training_jobs')
    for path, digest in queue['source_inputs_sha256'].items():
        pin(path, digest)
    results, roles, cache_exposures = [], set(), {}
    for job in jobs:
        out = Path(job['out']).resolve()
        require(out.is_relative_to(root/'formal/rankers'), 'training_output_outside_formal')
        receipt_path = run/(job['id']+'.receipt.json')
        pin(receipt_path)
        receipt = read_json(receipt_path)
        require(receipt['state']=='complete' and type(receipt['external_exit']) is int and receipt['external_exit']==0
                and type(receipt['internal_exit']) is int and receipt['internal_exit']==0
                and receipt.get('finished_utc'), 'training_not_real_double_exit0:'+job['id'])
        require(receipt['argv']==job['argv'] and receipt['output']==job['out'], 'receipt_job_mismatch')
        require(receipt['requested_inputs_sha256']==job['inputs_sha256'], 'receipt_requested_pins')
        for path, digest in receipt['queue_source_sha256'].items():
            pin(path, digest)
        for path, digest in job['inputs_sha256'].items():
            pin(path, digest)
        for filename in ('summary.json','protocol.json','history.jsonl','EXIT_CODE','initial.pt','best.pt','last.pt'):
            pin(out/filename)
        require((out/'EXIT_CODE').read_text().strip()=='0', 'producer_exit_not_zero')
        protocol = read_json(out/'protocol.json')
        args = protocol['args']
        require(args['stage']=='fit' and args['out']==job['out'], 'not_fit_output')
        role = (args['training_pool'], args['loss'], args['seed'], args['augment'])
        require(role not in roles, 'duplicate_training_role')
        roles.add(role)
        for path, digest in protocol['inputs_sha256'].items():
            # Training pins contain train caches, source and inherited checkpoints only.
            require('/test/' not in path and '/sealed/' not in path, 'unexpected_test_input')
            pin(path, digest)
        for opt in ('--train','--dual-train','--initial-checkpoint','--out','--training-pool','--loss','--seed',
                    '--epochs','--samples','--batch-size','--anchor','--lr'):
            require(job['argv'][job['argv'].index(opt)+1] == str(args[opt[2:].replace('-','_')]), 'argv_protocol_mismatch:'+opt)
        require(('--augment' in job['argv']) is args['augment'], 'argv_augment_mismatch')
        train = Path(args['train'])
        require(train.parent.name == 'train', 'train_cache_only')
        key = (str(train), args['seed'], args['augment'])
        if key not in cache_exposures:
            cache_summary = read_json(train.parent/'summary.json')
            require(cache_summary['feature_cache_sha256']==pins[str(train.resolve())], 'feature_producer_sha')
            data = torch.load(train, map_location='cpu', weights_only=False, mmap=True)
            require(data['manifest']['cohort']=='train' and data['manifest']['recall']=='single', 'wrong_train_cache')
            identity_rows = [{k:r[k] for k in KEY_FIELDS} for r in data['rows']]
            require(len({tuple(r[k] for k in KEY_FIELDS) for r in identity_rows})==len(identity_rows), 'duplicate_train_identity')
            cache_exposures[key] = recompute_exposure(identity_rows, args['seed'], args['augment'])
            del data
        exposure, training_groups, internal_groups = cache_exposures[key]
        summary = read_json(out/'summary.json')
        require(summary['training_groups']==training_groups and summary['internal_groups']==internal_groups, 'group_count_mismatch')
        old = torch.load(args['initial_checkpoint'], map_location='cpu', weights_only=False)
        checkpoints = {k:torch.load(out/(k+'.pt'), map_location='cpu', weights_only=False) for k in ('initial','best','last')}
        history = [json.loads(line) for line in (out/'history.jsonl').read_text().splitlines() if line]
        result = audit_model(summary, protocol, old, checkpoints, history,
                             dict(training_pool=role[0],loss=role[1],seed=role[2],augment=role[3]), exposure)
        results.append(dict(job_id=job['id'],role=role,output=str(out),**result))
    expected = {(pool,loss,seed,aug) for pool in ('original','mixed') for loss in ('pairwise16','listwise')
                for seed in (193,211) for aug in (False,True)}
    require(roles==expected, 'incomplete_training_matrix')
    for seed in (193,211):
        for aug in (False,True):
            part = [r for r in results if r['role'][2:]==(seed,aug)]
            require(len(part)==4 and len({r['exposure_sha256'] for r in part})==1, 'four_arm_exposure_mismatch')
    after = {p:sha(p) for p in pins}
    require(pins==after, 'audit_input_changed_during_run')
    return dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        formal_queue_sha256=pins[str(queue_path)], scope='training_only_no_test',
        checks=dict(same_initial_tensors=True,frozen_observation=True,matched_exposure=True,finite=True,full_budget=True),
        training_jobs=16,draws_per_job=524288,optimizer_steps_per_job=16384,
        input_sha256=pins,inputs_unchanged=True,models=results,external_api_calls=0,
        budget_evidence='last epoch64 + fixed pinned training code + independently replayed full draw and pool exposure SHA; history is recorded every8 epochs including0',
        best_epoch_zero_allowed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--formal-run')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args.root, args.formal_run)
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(status='FAIL',error=repr(exc)))+'\n')
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

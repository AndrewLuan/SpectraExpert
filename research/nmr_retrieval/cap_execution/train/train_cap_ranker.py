"""功能概述：唯一C5/C10 matched continuation训练入口，预算与最终输出固定。
validate_protocol/check_inputs先核本轮授权、精确role/train缓存及producer/source SHA；
prepare_training复用native双池契约、group→variant抽样与固定中心，不重估中心、不读val。
run_fit只优化network，fresh AdamW，epoch1..64原RNG，8192draw/epoch严格半池、batch32；
新strict mask共同排除external/internal/samegroup负例，target仅进入训练loss。
每epoch记录draw/pool/negative-exposure SHA及更新数，保存initial.pt/final.pt和原中心
两文件字节。最终epoch64/alpha1，无best、earlystop、内部alpha选择或外部验证。
main独占新out，起止核全部实际输入/源码SHA，输出protocol/status/summary/真实EXIT；
失败保留已有输出与EXIT1，不覆盖历史结果。共享cap_ranker供后续evaluator读取新schema。
本模块不派发远端、不改原科学源、不会执行test或向推理候选池注入正例。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from torch import nn

import cap_ranker as cap


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def validate_protocol(protocol, args):
    cap.require(protocol.get('schema') == 'cap10_controlled_protocol_20261024', 'training_protocol_schema')
    cap.require(protocol.get('training_allowed') is True and protocol.get('test_allowed') is False,
                'training_authorization_required')
    cap.require(protocol.get('seeds') == [193, 211] and protocol.get('caps') == [5, 10], 'registered_matrix')
    cap.require(all(protocol.get('budget', {}).get(k) == v and
                    type(protocol['budget'][k]) is not bool for k, v in cap.BUDGET.items()), 'full_budget_protocol')
    cap.require(type(protocol.get('final_epoch')) is int and protocol['final_epoch'] == 64 and
                protocol.get('alpha') == 1 and type(protocol.get('alpha')) is not bool and
                type(protocol.get('epoch_offset')) is int and protocol['epoch_offset'] == 0, 'fixed_final_no_selection')
    cap.require(args.seed in (193, 211) and args.cap in (5, 10) and args.role in ('no_aug', 'aug'), 'registered_fit_role')


def train_path(value):
    path = Path(value).resolve()
    lower = str(path).lower()
    cap.require('/test/' not in lower and '/sealed' not in lower and
                path.name not in ('predictions.jsonl', 'research_report.md'), 'training_forbidden_input')
    return path


def check_inputs(args, runtime):
    protocol_path, binding_path = train_path(args.protocol), train_path(args.bindings)
    protocol_bytes, binding_bytes = protocol_path.read_bytes(), binding_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    bindings = json.loads(binding_bytes)
    validate_protocol(protocol, args)
    cap.require(bindings.get('schema') == 'cap10_execution_bindings_20261024', 'training_bindings_schema')
    cap.require(Path(bindings['source_checkout']).resolve() == runtime.source, 'binding_scientific_source')
    record = bindings['roles'][str(args.seed)][args.role]
    caches = bindings['caches'][str(args.seed)]['train']
    paths = runtime.source_paths() + [Path(__file__).resolve(), protocol_path, binding_path]
    for key in ('checkpoint', 'centers', 'center_rows', 'producer_summary'):
        paths.append(train_path(record[key]))
    paths.append(train_path(record['checkpoint']).parent / 'EXIT_CODE')
    for pool in ('single', 'dual'):
        entry = caches[pool]
        path = train_path(entry['path'])
        producer = train_path(entry['producer'])
        cap.require(producer == path.parent / 'summary.json', 'cache_producer_path')
        cap.require(entry['manifest']['cohort'] == 'train' and entry['manifest']['recall'] == pool,
                    'only_bound_training_cache')
        paths += [path, producer, path.parent / 'EXIT_CODE']
    contract = train_path(bindings['cohort_contract'])
    paths += [contract, contract.parent / 'manifest.json', contract.parent / 'EXIT_CODE']
    paths = list({str(train_path(p)): train_path(p) for p in paths}.values())
    pins = cap.fingerprints(paths)
    cap.require(pins[str(protocol_path)] == hashlib.sha256(protocol_bytes).hexdigest() and
                pins[str(binding_path)] == hashlib.sha256(binding_bytes).hexdigest(), 'JSON_changed_during_read')
    declared = bindings['input_sha256']
    for p, digest in pins.items():
        if p != str(binding_path):
            cap.require(p in declared and declared[p] == digest, 'binding_missing_or_changed_input:' + p)
    cap.require(pins[str(train_path(record['checkpoint']))] == record['checkpoint_sha256'] and
                pins[str(train_path(record['centers']))] == record['centers_sha256'], 'bound_initial_artifact_SHA')
    for pool, entry in caches.items():
        if pool not in ('single', 'dual'):
            continue
        path = train_path(entry['path'])
        cap.require(pins[str(path)] == entry['sha256'], 'cache_actual_SHA')
        producer = json.loads(train_path(entry['producer']).read_text())
        # native read_feature_cache is authoritative for cache manifest/producer byte binding too.
        cap.require((path.parent / 'EXIT_CODE').read_text().strip() == '0', 'cache_producer_exit')
        cap.require(all(producer.get(k) == v for k, v in entry['manifest'].items()), 'cache_producer_manifest_binding')
        cap.require(producer.get('feature_cache_sha256') == pins[str(path)], 'cache_producer_payload_SHA')
    cap.require((train_path(record['checkpoint']).parent / 'EXIT_CODE').read_text().strip() == '0',
                'initial_producer_exit')
    return protocol, bindings, record, caches, pins


def prepare_training(args, runtime, bindings, record, caches, pins, device):
    single, _ = runtime.read_feature_cache(caches['single']['path'], known_sha=pins[str(train_path(caches['single']['path']))])
    dual, _ = runtime.read_feature_cache(caches['dual']['path'], known_sha=pins[str(train_path(caches['dual']['path']))])
    # 旧producer summary在torch.save后追加feature_cache_sha256；缓存内部manifest是其精确子集。
    cap.require(all(caches[pool]['manifest'].get(k) == v for pool, data in (('single', single), ('dual', dual))
                    for k, v in data['manifest'].items()), 'loaded_training_manifest_binding')
    dual_lookup = runtime.base.validate_caches(single, dual)
    contract_path = train_path(bindings['cohort_contract'])
    contract = runtime.fixed.validate_contract_producer(
        argparse.Namespace(cohort_contract=str(contract_path)), single, pins)
    qualification = runtime.fixed.candidate_qualification(single, dual, contract)
    cap.require(qualification['metadata']['derived_candidate_groups_sha256'] ==
                record['qualification']['derived_candidate_groups_sha256'] and
                qualification['metadata']['historical_heldout_sha256'] == bindings['heldout_groups_sha256'],
                'qualification_authority_binding')
    rows = runtime.select_rows(single, args.role == 'aug')
    cap.require(all(r['group'] not in qualification['heldout'] for r in rows), 'external_heldout_query_in_training')
    fit = [r for r in rows if not runtime.internal_group(r['group'])]
    cap.require(fit, 'empty_fit_training_queries')
    groups = collections.defaultdict(list)
    # 原native保留cache中每group的variant行顺序；排序仅用于证据序列化。
    for row in fit:
        groups[row['group']].append(row)
    masks, negative_records = {}, []
    for row in sorted(fit, key=runtime.base.row_key):
        key = runtime.base.row_key(row)
        for pool, selected in ((0, row), (1, dual_lookup[key])):
            excluded = cap.strict_negative_mask(selected, qualification, runtime)
            masks[(key, pool)] = excluded
            ids = selected['candidate_indices'][~excluded].tolist()
            negative_records.append(dict(query_key=list(key), pool=pool, negatives=len(ids),
                ordered_eligible_candidate_sha256=runtime.fixed.ordered_sha([str(i) for i in ids]),
                excluded_mask_sha256=cap.tensor_hash(excluded)))
    qsha = hashlib.sha256(json.dumps(negative_records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    initial_model, scale, initial = runtime.fixed.load_checkpoint(record['checkpoint'], device)
    runtime.base.check_science(initial, single)
    cap.require(initial['args']['seed'] == args.seed and initial['args']['augment'] is (args.role == 'aug') and
                initial['alpha'] == 1 and initial['schema'] == runtime.fixed.SCHEMA, 'initial_role_or_schema')
    cap.require(initial['train_sha256'] == pins[str(train_path(caches['single']['path']))] and
                initial['dual_train_sha256'] == pins[str(train_path(caches['dual']['path']))], 'initial_training_pair')
    cap.require(all(initial['args'][k] == v for k, v in runtime.fixed.BUDGET.items()) and
                initial['args']['training_pool'] == 'mixed' and initial['args']['loss'] == 'listwise', 'initial_full_budget')
    producer = json.loads(train_path(record['producer_summary']).read_text())
    cap.require(producer['best']['epoch'] == initial['epoch'] and producer['best']['alpha'] == initial['alpha'] and
                producer['center_estimate_sha256'] == record['centers_sha256'], 'initial_best_producer_binding')
    cap.require(initial['center_estimate_sha256'] == record['centers_sha256'] and
                torch.equal(initial_model.fixed_x_center.cpu(), torch.tensor(record['center33'], dtype=torch.float32)) and
                torch.equal(initial_model.fixed_o_center.cpu(), torch.tensor(record['O_center3'], dtype=torch.float32)),
                'initial_36centers_binding')
    model = cap.CapRanker(initial['initial_sigmas'], initial_model.fixed_x_center,
                          initial_model.fixed_o_center, args.cap, runtime).to(device)
    model.load_state_dict(initial_model.state_dict(), strict=True)
    inherited = cap.tensor_hashes(initial_model, scale)
    cap.require(cap.tensor_hashes(model, scale) == inherited and
                all(torch.isfinite(v).all() for v in model.state_dict().values()), 'initial_all_tensor_inheritance')
    model.eval()
    if args.cap == 5:
        with torch.no_grad():
            probe = fit[0]
            cap.require(torch.equal(model.row_delta(probe, scale, device), initial_model.row_delta(probe, scale, device)),
                        'C5_initial_native_bitwise_mismatch')
    del initial_model
    return dict(single=single, dual_lookup=dual_lookup, groups=groups, masks=masks,
                negative_records=negative_records, qualification_sha=qsha, qualification=qualification['metadata'],
                model=model, scale=scale, initial=initial, inherited=inherited,
                ranker_sources={str(Path(p).resolve()): pins[str(Path(p).resolve())]
                    for p in runtime.source_paths() + [Path(__file__).resolve()]})


def checkpoint_payload(args, context, pins, record, epoch, draws, updates, exposure, pools, negative):
    model, scale, initial = context['model'], context['scale'], context['initial']
    training_args = dict(vars(args), augment=args.role == 'aug', training_pool='mixed', loss='listwise',
                         **{k: v for k, v in cap.BUDGET.items() if k not in vars(args)})
    return dict(schema=cap.SCHEMA, observation=True, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        scale=scale.detach().cpu(), initial_sigmas=initial['initial_sigmas'], delta_cap=args.cap,
        epoch=epoch, alpha=1., seed=args.seed, role=args.role, args=training_args, budget=dict(cap.BUDGET),
        negative_mask_policy=cap.MASK_POLICY, draws=draws, optimizer_steps=updates,
        exposure_sha256=exposure, pool_exposure_sha256=pools, negative_exposure_sha256=negative,
        strict_negative_qualification_sha256=context['qualification_sha'],
        parent_checkpoint=str(train_path(record['checkpoint'])), parent_checkpoint_sha256=record['checkpoint_sha256'],
        parent_epoch=initial['epoch'], parent_alpha=initial['alpha'], initial_tensor_sha256=context['inherited'],
        center_estimate_sha256=record['centers_sha256'],
        center_rows_sha256=pins[str(train_path(record['center_rows']))], frozen_tensor_sha256=cap.frozen_digest(model, scale),
        train_sha256=initial['train_sha256'], dual_train_sha256=initial['dual_train_sha256'],
        cache_protocol=initial['cache_protocol'], source_inputs_sha256=initial['source_inputs_sha256'],
        ranker_source_sha256=context['ranker_sources'], input_sha256=pins, continuation_epoch_offset=0,
        optimizer_reset=True, selection='fixed_final_epoch64_alpha1_no_selection')


def run_fit(args, out, runtime, record, pins, context, status):
    model, scale = context['model'], context['scale']
    for source, name in ((record['centers'], 'centers.json'), (record['center_rows'], 'center_rows.jsonl')):
        shutil.copyfile(train_path(source), out / name)
    cap.require(cap.sha256(out / 'centers.json') == record['centers_sha256'] and
                cap.sha256(out / 'center_rows.jsonl') == pins[str(train_path(record['center_rows']))], 'copied_center_bytes')
    atomic_json(out / 'initial_tensor_sha256.json', context['inherited'])
    atomic_json(out / 'negative_qualification.json', dict(policy=cap.MASK_POLICY,
        qualification=context['qualification'], records=context['negative_records'], sha256=context['qualification_sha']))
    frozen_before = cap.frozen_digest(model, scale)
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=.001, weight_decay=.001)
    torch.manual_seed(args.seed)
    exposure, pools, negatives = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    epoch_records, draws, updates = [], 0, 0
    torch.save(checkpoint_payload(args, context, pins, record, 0, 0, 0,
        exposure.hexdigest(), pools.hexdigest(), negatives.hexdigest()), out / 'initial.pt')
    negative_by_key = {(tuple(r['query_key']), r['pool']): r['excluded_mask_sha256'] for r in context['negative_records']}
    names = sorted(context['groups'])
    started = time.monotonic()
    for epoch in range(1, cap.BUDGET['epochs'] + 1):
        model.train()
        model.observation_model.eval()
        schedule = runtime.base.training_schedule(names, context['groups'], args.seed, epoch, cap.BUDGET['samples'])
        counts = collections.Counter(pool for _, pool in schedule)
        cap.require(len(schedule) == cap.BUDGET['samples'] and counts == {0: cap.BUDGET['samples']//2,
                    1: cap.BUDGET['samples']//2}, 'exact_half_pool_schedule')
        epoch_draw, epoch_pool, epoch_neg = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        total_loss, epoch_updates = 0., 0
        for begin in range(0, len(schedule), cap.BUDGET['batch_size']):
            batch = schedule[begin:begin + cap.BUDGET['batch_size']]
            cap.require(len(batch) == cap.BUDGET['batch_size'], 'full_batch_required')
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for row, pool in batch:
                key = runtime.base.row_key(row)
                encoded = json.dumps(key, separators=(',', ':')).encode() + b'\n'
                pb = bytes([pool])
                nb = bytes.fromhex(negative_by_key[(key, pool)])
                for cumulative, per_epoch, payload in ((exposure, epoch_draw, encoded),
                        (pools, epoch_pool, pb), (negatives, epoch_neg, nb)):
                    cumulative.update(payload)
                    per_epoch.update(payload)
                selected = context['dual_lookup'][key] if pool else row
                delta = model.row_delta(selected, scale, next(model.parameters()).device)
                losses.append(cap.listwise_loss(selected, delta, scale, context['masks'][(key, pool)], .1))
                draws += 1
            loss = torch.stack(losses).mean()
            cap.require(torch.isfinite(loss), 'nonfinite_training_loss')
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.network.parameters(), 2., error_if_nonfinite=True)
            cap.require(torch.isfinite(grad), 'nonfinite_training_gradient')
            optimizer.step()
            updates += 1
            epoch_updates += 1
            total_loss += float(loss.detach())
        cap.require(cap.frozen_digest(model, scale) == frozen_before, 'frozen_O_centers_scale_changed')
        item = dict(epoch=epoch, draws=len(schedule), pool_draws=dict(counts), optimizer_steps=epoch_updates,
            cumulative_draws=draws, cumulative_optimizer_steps=updates,
            exposure_sha256=epoch_draw.hexdigest(), pool_exposure_sha256=epoch_pool.hexdigest(),
            negative_exposure_sha256=epoch_neg.hexdigest(), cumulative_exposure_sha256=exposure.hexdigest(),
            cumulative_pool_exposure_sha256=pools.hexdigest(), cumulative_negative_exposure_sha256=negatives.hexdigest(),
            mean_training_loss=total_loss / epoch_updates, frozen_tensor_sha256=frozen_before,
            seconds=time.monotonic() - started)
        epoch_records.append(item)
        with (out / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(item, sort_keys=True) + '\n')
        status('training', **item)
        print(json.dumps({k: item[k] for k in ('epoch', 'cumulative_draws', 'seconds')}), flush=True)
    cap.require(draws == cap.BUDGET['draws_per_fit'] and updates == cap.BUDGET['optimizer_steps'], 'full_training_budget_not_completed')
    cap.require(all(torch.isfinite(v).all() for v in model.state_dict().values()), 'nonfinite_final_state')
    final = checkpoint_payload(args, context, pins, record, cap.BUDGET['epochs'], draws, updates,
        exposure.hexdigest(), pools.hexdigest(), negatives.hexdigest())
    torch.save(final, out / 'final.pt')
    summary = {k: v for k, v in final.items() if k not in ('state_dict', 'scale', 'initial_sigmas')}
    summary.update(final_checkpoint_sha256=cap.sha256(out / 'final.pt'),
        initial_checkpoint_sha256=cap.sha256(out / 'initial.pt'), centers_frozen=True, scale_frozen=True,
        observation_frozen=True, initial_tensors_inherited_exact=True, training_groups=len(names),
        training_rows=sum(map(len, context['groups'].values())), epoch_records=epoch_records,
        source_inputs_start_sha256=pins, source_inputs_end_sha256=cap.fingerprints(pins),
        elapsed_seconds=time.monotonic() - started, external_api_calls=0)
    cap.require(summary['source_inputs_end_sha256'] == pins, 'input_changed_during_training')
    atomic_json(out / 'summary.json', summary)
    return summary


def parser():
    result = argparse.ArgumentParser()
    for name in ('source', 'fixed-module', 'protocol', 'bindings', 'out'):
        result.add_argument('--' + name, required=True)
    result.add_argument('--seed', type=int, choices=(193, 211), required=True)
    result.add_argument('--cap', type=int, choices=(5, 10), required=True)
    result.add_argument('--role', choices=('no_aug', 'aug'), required=True)
    result.add_argument('--device', default='cuda:0')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    runtime = cap.configure_runtime(args.source, args.fixed_module)
    protocol, bindings, record, caches, pins = check_inputs(args, runtime)
    out = Path(args.out).resolve()
    expected_out = Path(protocol['remote_root']).resolve() / 'rankers' / ('C' + str(args.cap)) / ('s' + str(args.seed)) / args.role
    cap.require(out == expected_out, 'registered_new_fit_output_required')
    out.mkdir(parents=True, exist_ok=False)
    status = lambda stage, **fields: atomic_json(out / 'status.json', dict(stage=stage, **fields))
    try:
        torch.set_num_threads(4)
        atomic_json(out / 'protocol.json', dict(args=vars(args), protocol=protocol, input_sha256=pins))
        context = prepare_training(args, runtime, bindings, record, caches, pins, torch.device(args.device))
        run_fit(args, out, runtime, record, pins, context, status)
        cap.require(cap.fingerprints(pins) == pins, 'input_changed_at_finish')
        status('complete', epoch=64, alpha=1, delta_cap=args.cap)
        (out / 'EXIT_CODE').write_text('0\n')
    except BaseException as exc:
        status('failed', error=repr(exc))
        (out / 'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

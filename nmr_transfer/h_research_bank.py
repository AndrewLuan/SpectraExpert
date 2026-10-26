"""功能概述：为新轮固定候选库生成与当前前向权重一致的条件化预测bank。

build_bank从旧bank只继承结构、候选顺序和冻结查询容器；所有候选目标置零，按
cohort公开条件和旧条件键统一预测。F模型通过load_checkpoint_model重建，显式
去除旧embedding缓存；其同次分位sigma及物理组输出另存，供观测模型与误差审计。
继承均值模型可仅补新条件，原预测不改写。候选自身标签不参与预测，不执行检索。
main写独占输出、完整权重/代码/输入SHA、逐条件状态及实际EXIT；无外部API。
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .emission_model import make_model, weighted_quantiles
from .h_condinfer import candidate_batches, condition_key
from .h_conditions import CONDITION_DIM
from .h_global_bank import load_bank
from .h_research_contract import atomic_json, input_fingerprints
from .h_uncertainty import quantile_classes


@torch.inference_mode()
def build_bank(args, out, status):
    device = torch.device(args.device)
    bank = load_bank(args.reference_bank, args.reference_extension)
    payload = torch.load(Path(args.cohorts)/'cohorts.pt', map_location='cpu', weights_only=False)
    prepared = torch.load(Path(args.prepared)/'cache.pt', map_location='cpu', weights_only=False, mmap=True)
    checkpoint_sha = input_fingerprints([args.checkpoint])[str(Path(args.checkpoint).resolve())]
    original_sha = bank['checkpoint_sha256']
    if args.research_forward:
        from .h_research_forward import load_checkpoint_model
        model = load_checkpoint_model(args.checkpoint, device=device, trainable=False).eval()
    else:
        if checkpoint_sha != original_sha:
            raise ValueError('research_inherited_extension_checkpoint_changed')
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model = make_model(saved['args']).to(device).eval()
        model.load_state_dict(saved['state_dict'], strict=True)
    keys = set(bank['predictions'])
    for rows in payload['cohorts'].values():
        for row in rows:
            keys.add(condition_key(row, payload['metadata'], payload['key_mode']))
    if args.research_forward:
        bank = dict(bank, predictions={}, fixed_predictions={}, candidate_quantile_classes={},
                    candidate_quantile_sigma={}, checkpoint_path=str(Path(args.checkpoint).resolve()),
                    checkpoint_sha256=checkpoint_sha)
    else:
        bank = dict(bank, predictions=dict(bank['predictions']), fixed_predictions=dict(bank['fixed_predictions']),
                    candidate_quantile_classes=dict(bank['candidate_quantile_classes']))
        keys -= set(bank['predictions'])
    status('building_candidate_batches', conditions=len(keys))
    batches = candidate_batches(bank, prepared['graphs'], device)
    if args.research_forward:
        from .h_research_forward import strip_embedding_caches
        batches = [strip_embedding_caches(batch) for batch in batches]
    elif getattr(model, 'conditioned_messages', False):
        for batch in batches:
            batch.unconditioned_context_embedding = model.unconditioned_embedding(batch).detach()
    elif getattr(model, 'context_depth', 0) > 0:
        for batch in batches:
            batch.final_context_embedding = model.structure_embedding(batch).detach()
    groups_folder = out/'environment_predictions'
    if args.research_forward:
        groups_folder.mkdir()
    started = time.monotonic()
    condition_map = []
    for j, key in enumerate(sorted(keys)):
        qs, fs, ts, ss = [], [], [], []
        groups = {name: [] for name in ('group_centers', 'group_Hmass', 'group_sigma', 'group_trust', 'group_mask')}
        for number, batch in enumerate(batches):
            batch.cond = torch.tensor(key[:4], device=device).reshape(1, 4).expand(batch.num_graphs, -1)
            batch.condition_vector = torch.tensor(key[4:], device=device).reshape(1, CONDITION_DIM).expand(batch.num_graphs, -1)
            details = model(batch, details=True)
            qs.append(details['quantiles'].cpu())
            tags, checked = quantile_classes(details, batch)
            if not torch.equal(checked, details['quantiles']):
                raise ValueError('research_same_forward_quantile_mismatch')
            ts.append(tags.cpu())
            if key[1] > 0:
                keep = ~details['exchangeable']
                fs.append(weighted_quantiles(details['values'][keep], details['weights'][keep],
                                             details['groups'][keep], batch.num_graphs).cpu())
            if args.research_forward:
                ss.append(details['quantile_sigma_ppm'].cpu())
                for name in groups:
                    value = details[name]
                    groups[name].append((value*10 if name in ('group_centers', 'group_sigma') else value).cpu())
            if number % 20 == 0:
                status('predicting', condition=j+1, conditions=len(keys), batch=number+1, batches=len(batches),
                       seconds=time.monotonic()-started)
        bank['predictions'][key] = torch.cat(qs)
        bank['candidate_quantile_classes'][key] = torch.cat(ts)
        if fs:
            bank['fixed_predictions'][key] = torch.cat(fs)
        if args.research_forward:
            bank['candidate_quantile_sigma'][key] = torch.cat(ss)
            path = groups_folder/f'condition_{j:04d}.pt'
            torch.save(dict(condition=key, units='ppm', checkpoint_sha256=checkpoint_sha,
                            **{name: torch.cat(values) for name, values in groups.items()}), path)
            condition_map.append(dict(condition=list(key), path=str(path.resolve())))
    bank['research_protocol'] = dict(schema='nmr_research_20261022', inherited=not args.research_forward,
        reference_bank_sha256=input_fingerprints([args.reference_bank])[str(Path(args.reference_bank).resolve())],
        candidates_use_query_conditions_only=True, candidate_targets_zero=True,
        condition_keys=[list(k) for k in sorted(bank['predictions'])])
    torch.save(bank, out/'bank.pt')
    index = bank['single_indices']
    names = [bank['smiles'][i] for i in index]
    if len(names) != 114982:
        raise ValueError('research_bank_candidate_count_changed')
    atomic_json(out/'summary.json', dict(checkpoint_sha256=checkpoint_sha, added_or_recomputed_conditions=len(keys),
        total_conditions=len(bank['predictions']), candidate_count=len(names),
        candidate_order_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
        bank_sha256=input_fingerprints([out/'bank.pt'])[str((out/'bank.pt').resolve())],
        environment_prediction_files=condition_map, elapsed_seconds=time.monotonic()-started, external_api_calls=0))


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'reference-bank', 'cohorts', 'prepared', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--reference-extension')
    parser.add_argument('--research-forward', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    status = lambda stage, **kw: atomic_json(out/'status.json', dict(stage=stage, **kw))
    paths = [args.checkpoint, args.reference_bank, args.reference_extension,
             Path(args.cohorts)/'cohorts.pt', Path(args.cohorts)/'manifest.json',
             Path(args.prepared)/'cache.pt', Path(args.prepared)/'manifest.json',
             __file__, Path(__file__).with_name('h_research_contract.py')]
    paths += [p for p in Path(__file__).parent.glob('*.py')
              if not p.name.startswith(('test_', 'h_research_'))]
    if args.research_forward:
        paths.append(Path(__file__).with_name('h_research_forward.py'))
    model_config = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['args']
    if model_config.get('equivalence_registry_path'):
        paths.append(model_config['equivalence_registry_path'])
    pinned = input_fingerprints(paths)
    try:
        torch.set_num_threads(4)
        atomic_json(out/'protocol.json', dict(args=vars(args), inputs_sha256=pinned))
        build_bank(args, out, status)
        if input_fingerprints(pinned) != pinned:
            raise ValueError('research_bank_inputs_changed_during_run')
        status('complete'); (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed', error=repr(exc)); (out/'EXIT_CODE').write_text('1\n'); raise


if __name__ == '__main__':
    main()

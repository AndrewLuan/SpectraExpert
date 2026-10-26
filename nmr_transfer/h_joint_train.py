"""功能概述：执行条件位移×报告状态的等曝光微调，不重复理论预训练。

main复用已审核domain监督和同布局既有权重，四臂始终建立相同报告小头；shift臂
解冻已有两层溶剂消息并增加合格跨溶剂差约束，report臂启用整谱报告状态NLL。
prepare_rows只在进程内接入24条既有苯甲酰胺支持、更正训练归属和隔离疑点；
同一原实验val选择epoch，320条检索val由后续流程统一选型。pair抽样连接结构均权。
所有arm同seed读取同样实验顺序，辅助配对另用独立seed；未使用的配对曝光不冒充训练。
冻结旧GINE/深上下文且逐张量核验，只缓存未条件化表示。width拟合由后续阶段执行。
输出独占initial/best/last、数据审计、历史、源码SHA和EXIT_CODE；失败保留现场。
依赖h_joint_conditions、h_reviewed_supervision和原prepared缓存；不改原始谱、旧结果或测试。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.data import Batch

from .emission_model import make_model
from .h_condmodel import training_details
from .h_condtrain import sampling_weights
from .h_count_curriculum import validate_spectrum
from .h_joint_conditions import JointSpectra, build_pairs, joint_objective, pair_loss
from .h_labels import make_key
from .h_message_support import add_support, extra_weights
from .h_metadata import metadata_view, RECOVERED_MODE
from .h_observation_study import stamp_queries
from .h_reviewed_supervision import load_artifact, apply_review, common_validation
from .h_site_quality import apply_quality
from .prepare import sha
from .train import collate, seed_worker


def prepare_rows(base, prepared, cache, metadata, labels, review, device):
    exp = [r for r in cache['rows'] if r['domain'] == 'experimental' and r['split'] == 'train']
    val = [r for r in cache['rows'] if r['domain'] == 'experimental' and r['split'] == 'val']
    weights = sampling_weights(exp, metadata)
    support, extra_val, support_audit = add_support(
        cache, metadata, labels, base/'hydrogen_analysis/condition_response/benzene_formamide_inventory',
        json.loads((prepared/'manifest.json').read_text()),
        base/'hydrogen_analysis/site_supervision/train/best.pt', device)
    weights = extra_weights(weights, support, .05)
    exp += support
    artifact, artifact_meta = load_artifact(review)
    label_path = base/'hydrogen_analysis/labels_topology/annotations.pt'
    if artifact_meta['input_sha256'][str(label_path)] != sha(label_path):
        raise ValueError('joint_review_annotations_version_mismatch')
    exp, audit = apply_review(labels, exp, artifact, 'domain')
    val, va = common_validation(val, artifact)
    quality = apply_quality(labels, exp, 'methyl_record_mask')
    if {r['group'] for r in exp} & {r['group'] for r in val + extra_val}:
        raise ValueError('joint_training_validation_overlap')
    return exp, val, weights, dict(review=audit, common_validation=va, quality=quality,
                                  support=support_audit, review_sha256=artifact_meta['artifact_sha256'])


def main():
    parser = argparse.ArgumentParser()
    for name in ('base', 'prepared', 'init', 'review', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--shift', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--samples', type=int, default=32768)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    def status(stage, **fields):
        tmp = out/'status.tmp'
        tmp.write_text(json.dumps(dict(stage=stage, seconds=time.monotonic()-start, **fields)))
        tmp.replace(out/'status.json')

    try:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('high')
        device = torch.device('cuda')
        base, prepared = Path(args.base), Path(args.prepared)
        status('loading')
        cache = torch.load(prepared/'cache.pt', map_location='cpu', weights_only=False, mmap=True)
        metadata = metadata_view(torch.load(prepared/'metadata.pt', map_location='cpu', weights_only=False), RECOVERED_MODE)
        labels = torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt', map_location='cpu', weights_only=False)
        saved = torch.load(args.init, map_location='cpu', weights_only=False)
        manifest = json.loads((prepared/'manifest.json').read_text())
        if saved['prepared_sha256'] != manifest['cache_sha256'] or saved['args']['model'] != 'preserved_group_atom':
            raise ValueError('joint_prepared_or_model_mismatch')
        torch.manual_seed(317)
        cfg = dict(saved['args'], joint_reporting=True, reporting_enabled=args.report)
        model = make_model(cfg).to(device)
        missing, extra = model.load_state_dict(saved['state_dict'], strict=False)
        if extra or any(not k.startswith('reporting_head.') for k in missing):
            raise ValueError('joint_initialization_mismatch')
        exp, val, weights, audit = prepare_rows(base, prepared, cache, metadata, labels, args.review, device)
        pairs, pair_audit = build_pairs(exp, labels, cache['graphs'], metadata)
        if args.smoke:
            eligible = [r for r in exp if r['source'] == 'HMDB' and 'C=O' in r['smiles']]
            selected = eligible[:64] + exp[:64] + [exp[index] for p in pairs[:2] for index in (p['left'], p['right'])]
            exp = list({make_key(r):r for r in selected}.values())
            weights = sampling_weights(exp, metadata)
            val = val[:32]
            pairs, smoke_pairs = build_pairs(exp, labels, cache['graphs'], metadata)
            pair_audit['smoke_subset'] = smoke_pairs
            args.epochs, args.samples = 1, 128
        dataset = JointSpectra(exp, cache['graphs'], '1H', metadata, annotations=labels, grouped=True)
        pair_dataset = JointSpectra(exp, cache['graphs'], '1H', metadata, annotations=labels, grouped=True)
        required = sorted({r['smiles'] for r in exp + val})
        status('caching_unconditioned_context', structures=len(required))
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(required), 512):
                names = required[offset:offset+512]
                batch = Batch.from_data_list([cache['graphs'][s] for s in names]).to(device)
                h = model.unconditioned_embedding(batch).cpu()
                ptr = batch.ptr.cpu().tolist()
                for j, s in enumerate(names):
                    cache['graphs'][s].unconditioned_context_embedding = h[ptr[j]:ptr[j+1]].clone()
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith('equivalence_head.')
                or (args.shift and name.startswith('solvent_message_'))
                or (args.report and name.startswith('reporting_head.')))
        groups = []
        for prefix, lr in [('equivalence_head.', 3e-5), ('solvent_message_', 1e-5), ('reporting_head.', 1e-3)]:
            params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
            if params:
                groups.append(dict(params=params, lr=lr))
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        parameters = [p for p in model.parameters() if p.requires_grad]
        digest = hashlib.sha256()
        for epoch in range(1, args.epochs + 1):
            seq = torch.multinomial(weights, args.samples, replacement=True,
                                   generator=torch.Generator().manual_seed(args.seed+2000+epoch))
            digest.update(seq.numpy().tobytes())
        config = dict(args=vars(args), model_args=cfg, init_sha256=sha(args.init), audit=audit,
                      pairs=pair_audit, train_queries_sha256=stamp_queries(exp), val_queries_sha256=stamp_queries(val),
                      sampling_sha256=digest.hexdigest(), trainable_parameters=sum(p.numel() for p in parameters),
                      pair_weight=.25, pair_batch=8, pair_every_steps=8,
                      selection='same experimental spectral validation macro including epoch0; no test',
                      source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
        (out/'config.json').write_text(json.dumps(config, indent=2))
        (out/'cross_solvent_pairs.json').write_text(json.dumps(pairs, indent=2))
        def checkpoint(name, epoch, metrics):
            torch.save(dict(state_dict=model.state_dict(), args=cfg, epoch=epoch, validation=metrics,
                            baseline_sha256=saved['baseline_sha256'], prepared_sha256=saved['prepared_sha256']), out/name)
        best = float('inf')
        best_epoch = 0
        pair_counts = collections.Counter(p['group'] for p in pairs)
        pair_weights = torch.tensor([1/pair_counts[p['group']] for p in pairs], dtype=torch.float64)
        for epoch in range(args.epochs + 1):
            totals = np.zeros(3)
            steps = pair_steps = 0
            if epoch:
                torch.manual_seed(args.seed + epoch)
                np.random.seed(args.seed + epoch)
                sampler = WeightedRandomSampler(weights, args.samples, replacement=True,
                                                generator=torch.Generator().manual_seed(args.seed+2000+epoch))
                loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=2,
                    collate_fn=collate, worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(args.seed+2000+epoch))
                pair_rng = torch.Generator().manual_seed(args.seed+4000+epoch)
                model.train()
                for batch in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    details = model(batch, details=True)
                    parts = joint_objective(model, details, training_details(details, batch), batch, args.report)
                    loss = parts['loss'] + parameters[0].sum()*0
                    pl = loss.detach()*0
                    if args.shift and pairs and steps % 8 == 0:
                        chosen = torch.multinomial(pair_weights, 8, replacement=True, generator=pair_rng).tolist()
                        pb = collate([pair_dataset[index] for k in chosen for index in (pairs[k]['left'], pairs[k]['right'])]).to(device)
                        pl = pair_loss(model, pb, [pairs[k] for k in chosen])
                        loss = loss + .25*pl
                        pair_steps += 1
                    if not torch.isfinite(loss):
                        raise ValueError('joint_nonfinite_loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 2.)
                    optimizer.step()
                    totals += [float(loss.detach()), float(parts['report_aux'].detach()), float(pl.detach())]
                    steps += 1
                    if steps % 100 == 0:
                        status('training', epoch=epoch, steps=steps, mean_loss=float(totals[0]/steps))
            status('validation', epoch=epoch)
            metrics = validate_spectrum(model, val, cache['graphs'], metadata, labels, device)
            if epoch == 0:
                checkpoint('initial.pt', epoch, metrics)
            if metrics['macro_objective'] < best:
                best, best_epoch = metrics['macro_objective'], epoch
                checkpoint('best.pt', epoch, metrics)
            checkpoint('last.pt', epoch, metrics)
            record = dict(epoch=epoch, steps=steps, pair_steps=pair_steps, losses=(totals/max(1,steps)).tolist(),
                          validation=metrics, seconds=time.monotonic()-start)
            with (out/'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        for name, value in saved['state_dict'].items():
            if name not in trainable and not torch.equal(value, model.state_dict()[name].cpu()):
                raise ValueError('joint_frozen_parameter_changed:'+name)
        result = dict(best_epoch=best_epoch, best_macro=best, checkpoint_sha256=sha(out/'best.pt'),
                      frozen_core_exact=True, seconds=time.monotonic()-start)
        (out/'complete.json').write_text(json.dumps(result, indent=2))
        status('complete')
        (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed', error=repr(exc))
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

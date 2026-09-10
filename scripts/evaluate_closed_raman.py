"""功能概述：在冻结闭库上配对评测纯光谱 kNN 与 spectrum_graph_v2，不读取实验结构作查询。
主要职责：load_role 按 rows.json 顺序用 mol_id 连接光谱，检查完整来源 hash；summarize
汇总 group rank 的 Top-K/MRR 与逐查询延迟；evaluate 校验训练来源、构建理论分子图
embedding 后复用 SpectralEncoder.retrieve 对两种方法做稳定余弦排序并持久化完整排名。
代码库关系：复用正式 checkpoint loader、分子图特征、训练 split 校验和标准化函数。
查询仅输入实验光谱；真实 SMILES/group 只在排序后计分，不重排、不选模型、不使用 LLM。
输入输出与副作用：只读取指定冻结目录和 best checkpoint，输出到新目录；含协议哈希、
候选嵌入、逐查询 Top10/目标排名与聚合 JSON。拒绝覆盖；失败留下可审计的局部产物。
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from spectraexpert.data.closed_library_split import validate_closed_library_split
from spectraexpert.data.molecular_graph import featurize_smiles, collate_graphs
from spectraexpert.data.structure_dataset import canonicalize_smiles, canonical_smiles_sha256, StructureAwareSpectralDataset
from spectraexpert.data.structure_features import StructureFeatureConfig
from spectraexpert.inference import SpectralEncoder
from spectraexpert.training.checkpoints import read_checkpoint_snapshot, load_checkpoint


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load_role(root, role, sample):
    folder = root / role
    for name, expected in sample['outputs_sha256'][role].items():
        if digest(folder / name) != expected:
            raise ValueError(f'source hash mismatch: {role}/{name}')
    rows = json.loads((folder / 'rows.json').read_text())
    metadata = pd.read_parquet(folder / 'smiles.parquet')
    spectra = pd.read_parquet(folder / 'spectra.parquet')
    ids = [r['mol_id'] for r in rows]
    if len(set(ids)) != len(ids) or metadata['mol_id'].duplicated().any() or spectra['mol_id'].duplicated().any():
        raise ValueError('duplicate mol_id')
    if set(metadata['mol_id']) != set(ids) or set(spectra['mol_id']) != set(ids):
        raise ValueError('mol_id join mismatch')
    metadata = metadata.set_index('mol_id').loc[ids].reset_index()
    if metadata['group_hash'].tolist() != [r['group_hash'] for r in rows]:
        raise ValueError('row/group identity mismatch')
    axis = pd.read_parquet(folder / 'wavenumber.parquet')['wavenumber'].to_numpy()
    if not np.array_equal(axis, np.arange(500., 4001.)):
        raise ValueError('expected frozen 500..4000 axis')
    columns = [f'w_{int(w)}' for w in axis]
    matrix = spectra.set_index('mol_id').loc[ids, columns].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) == 0).any():
        raise ValueError('invalid spectral values')
    return metadata, matrix


def summarize(records):
    ranks = np.asarray([r['group_rank'] for r in records])
    ms = np.asarray([r['latency_ms'] for r in records])
    return dict(queries=len(records), **{f'top{k}': float((ranks <= k).mean()) for k in (1, 5, 10)},
                mrr=float((1. / ranks).mean()), median_rank=float(np.median(ranks)),
                canonical_smiles_top1=float(np.mean([r['canonical_smiles_hit_top1'] for r in records])),
                latency_ms_mean=float(ms.mean()), latency_ms_median=float(np.median(ms)),
                latency_ms_p95=float(np.percentile(ms, 95)), total_query_seconds=float(ms.sum()/1000))


def evaluate(root, checkpoint, output, device='cpu'):
    root, checkpoint, output = Path(root), Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    sample = json.loads((root / 'sample_manifest.json').read_text())
    split = json.loads((root / 'training_split.json').read_text())
    if sample.get('schema_version') != 'raman_closed_library_sample_v1' or split.get('schema_version') != 'raman_closed_library_train_v1':
        raise ValueError('requires actual closed-library manifests')
    candidates, theory = load_role(root, 'theory_train_candidate', sample)
    queries, experiment = load_role(root, 'experimental_test', sample)
    candidate_groups = candidates['group_hash'].tolist()
    query_groups = queries['group_hash'].tolist()
    if candidate_groups != sample['candidate_groups'] or query_groups != sample['test_groups']:
        raise ValueError('frozen query/candidate order mismatch')
    if len(set(candidate_groups)) != len(candidate_groups) or len(set(query_groups)) != len(query_groups):
        raise ValueError('requires one spectrum per group')
    if len(candidates) != sample['candidate_count'] or len(queries) != sample['test_count']:
        raise ValueError('frozen counts mismatch')
    c_by_id = dict(zip(candidates['mol_id'], candidate_groups))
    if any(c_by_id.get(i) != g or i not in split['train'] for i, g in zip(queries['mol_id'], query_groups)):
        raise ValueError('experimental query identities must be protected in theory train')
    if not all('/exp/raman/' in p for p in queries['source_path']):
        raise ValueError('queries must be experimental Raman')
    source_hashes = {n: digest(root / 'theory_train_candidate' / n) for n in ('spectra.parquet', 'smiles.parquet', 'wavenumber.parquet')}
    validate_closed_library_split(root / 'theory_train_candidate', split, source_hashes, candidates)
    snapshot = read_checkpoint_snapshot(checkpoint)
    model, payload = load_checkpoint(snapshot, expected_mode='spectrum_graph_v2', device=device)
    canonical = [canonicalize_smiles(s) for s in candidates['smiles']]
    if payload['source_sha256'] != source_hashes or payload['split_manifest_sha256'] != digest(root / 'training_split.json'):
        raise ValueError('checkpoint does not belong to frozen training data')
    if payload['source_smiles_sha256'] != canonical_smiles_sha256(candidates['mol_id'].tolist(), canonical):
        raise ValueError('checkpoint canonical source mismatch')
    if payload['train_config']['max_samples'] is not None:
        raise ValueError('smoke checkpoint is not a formal result')
    normalize = payload['train_config']['normalize']
    model.eval()
    output.mkdir(parents=True)
    protocol = dict(sample_manifest_sha256=digest(root / 'sample_manifest.json'),
        training_split_sha256=digest(root / 'training_split.json'), checkpoint_sha256=snapshot.sha256,
        checkpoint_epoch_zero_based=payload['epoch'], script_sha256=digest(Path(__file__)),
        candidate_count=len(candidates), query_count=len(queries), candidate_mol_ids=candidates['mol_id'].tolist(),
        query_mol_ids=queries['mol_id'].tolist(), source_sha256=sample['outputs_sha256'],
        knn='raw intensity cosine, L2 normalization; full frozen 3501-point spectra; no coverage masking',
        model='spectrum_graph_v2 spectrum-to-graph cosine, no reranking, no LLM',
        model_spectrum_standardize=normalize, identity='frozen group_hash', exclude_identity=False,
        ties='stable descending score; frozen candidate order', mrr='full candidate ranking, not truncated',
        timing='online query preparation/encoding + CPU cosine normalization/scoring/full sorting; excludes I/O and candidate graph build',
        device=device, torch_version=torch.__version__, numpy_version=np.__version__)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    build_start = time.perf_counter()
    feature_config = StructureFeatureConfig(**payload['feature_config'])
    embedding_batches = []
    with torch.inference_mode():
        for start in range(0, len(candidates), 128):
            graphs = [featurize_smiles(canonical[i], int(candidates.iloc[i]['mol_id']), feature_config)
                      for i in range(start, min(start + 128, len(candidates)))]
            embedding_batches.append(model.encode_graph(collate_graphs(graphs).to(device)).cpu().numpy())
    graph_embeddings = np.concatenate(embedding_batches)
    build_seconds = time.perf_counter() - build_start
    np.savez_compressed(output / 'candidate_graph_embeddings.npz', embeddings=graph_embeddings,
                        mol_ids=candidates['mol_id'].to_numpy(), checkpoint_sha256=np.array(snapshot.sha256))

    def encode(spectrum):
        x = StructureAwareSpectralDataset._normalize_spectrum(spectrum) if normalize else spectrum
        with torch.inference_mode():
            return model.encode_spectrum(torch.from_numpy(x).unsqueeze(0).to(device)).cpu().numpy()[0]

    # 固定首个查询热身，仅用于计时稳定性；不改变参数或选择 checkpoint。
    encode(experiment[0])
    SpectralEncoder.retrieve(experiment[0], theory, topk=len(theory))
    results = {}
    for method, database in [('knn_cosine', theory), ('spectrum_graph_v2', graph_embeddings)]:
        records, rankings = [], []
        for i, spectrum in enumerate(experiment):
            tick = time.perf_counter()
            query = spectrum if method == 'knn_cosine' else encode(spectrum)
            order, scores = SpectralEncoder.retrieve(query, database, topk=len(database))
            elapsed = 1000 * (time.perf_counter() - tick)
            # 完成纯光谱排序后才读取真实查询身份。
            target = query_groups[i]
            rank = next(j + 1 for j, index in enumerate(order) if candidate_groups[index] == target)
            records.append(dict(mol_id=int(queries.iloc[i]['mol_id']), group_hash=target, group_rank=rank,
                latency_ms=elapsed, canonical_smiles_hit_top1=canonical[order[0]] == canonicalize_smiles(queries.iloc[i]['smiles']),
                top10_mol_ids=candidates.iloc[order[:10]]['mol_id'].tolist(), top10_scores=scores[:10].tolist()))
            rankings.append(candidates.iloc[order]['mol_id'].to_numpy())
        (output / f'{method}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        np.save(output / f'{method}_rankings.npy', np.stack(rankings), allow_pickle=False)
        results[method] = summarize(records)
    results['candidate_graph_build_seconds'] = build_seconds
    results['total_seconds'] = time.perf_counter() - started
    (output / 'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    print(json.dumps(results, indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    evaluate(args.data_root, args.checkpoint, args.output_dir, args.device)

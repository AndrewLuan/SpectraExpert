"""功能概述：复核冻结模型与原始谱余弦 kNN 的完整结构 Exact Match@K。
主要职责：main 读取既有选择、checkpoint 和理论索引，只重新编码测试谱；以稳定排序
计算真实结构名次、@1/@5/@10 命中及两方法交集。完整 canonical SMILES 保留立体与同位素。
代码库关系：复用 raman_domain_experiment 的掩码编码和数据哈希验证，以及
run_closed_llm_direct 的测试加载；不训练、不选型、不调用 LLM API。
输入输出与副作用：--run 指向已完成的实验目录，--device 指定推理设备；写入新的
topk_exact 目录，拒绝覆盖。索引身份或既有 Top1 复现失败则报错，不发布评测结果。
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.raman_domain_experiment import MaskedSpectrumEncoder, load_prepared, encode, coverage_mask
from scripts.evaluate_closed_raman import digest
from scripts.run_closed_llm_direct import load_queries, save
from scripts.repair_closed_raman_exact import canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = args.run / 'topk_exact'
    if out.exists():
        raise FileExistsError(out)
    final = args.run / 'final'
    selection = json.loads((final / 'selection.json').read_text())
    assert digest(args.run / 'protocol.json') == selection['protocol_sha256']
    protocol, data, _ = load_prepared(args.run)
    checkpoint = args.run / selection['selected_arm'] / 'best.pt'
    checkpoint_sha = digest(checkpoint)
    assert checkpoint_sha == selection['arm_summaries'][selection['selected_arm']]['checkpoint_sha256']
    index_path = final / 'candidate_spectrum_index.npz'
    with np.load(index_path, allow_pickle=False) as index:
        reference = index['embeddings'].copy()
        names = index['canonical_smiles'].tolist()
        ids = index['mol_ids'].tolist()
        assert index['checkpoint_sha256'].item() == checkpoint_sha
        assert index['sample_manifest_sha256'].item() == protocol['sample_manifest_sha256']
    assert names == protocol['candidate_names'][:10000]
    assert ids == [r['mol_id'] for r in protocol['candidate_metadata'][:10000]]
    assert len(names) == len(set(names)) == 10000
    assert all(canonical(s) == s for s in names)
    root = Path(protocol['data_root'])
    queries, spectra, _ = load_queries(root, protocol['sample_manifest_sha256'])
    rows = json.loads((root / 'experimental_test/rows.json').read_text())
    rows_by_id = {r['mol_id']: r for r in rows}
    masks = np.stack([coverage_mask(rows_by_id[q['mol_id']]['coverage']) for q in queries])
    truth = {s: i for i, s in enumerate(names)}
    targets = np.array([truth[canonical(q['ground_truth_smiles'])] for q in queries])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['prepared_sha256'] == protocol['prepared_sha256']
    model = MaskedSpectrumEncoder(**payload['model_config']).to(args.device)
    model.load_state_dict(payload['state_dict'])
    query = encode(model, spectra, masks, args.device)
    raw = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    scores = {'model': query @ reference.T,
              'raw_knn': raw(spectra) @ raw(data['theory'][:10000]).T}
    report = dict(selected_arm=selection['selected_arm'], query_count=len(queries),
                  candidate_count=len(names), definition='full canonical isomeric SMILES membership in top K',
                  checkpoint_sha256=checkpoint_sha, index_sha256=digest(index_path),
                  sample_manifest_sha256=protocol['sample_manifest_sha256'],
                  script_sha256=digest(Path(__file__)), methods={}, overlap={})
    details = [dict(query_id=q['mol_id'], ground_truth_smiles=names[t]) for q, t in zip(queries, targets)]
    ranks = {}
    for method, matrix in scores.items():
        assert np.isfinite(matrix).all()
        order = np.argsort(-matrix, axis=1, kind='stable')
        old = json.loads((final / method / 'predictions.json').read_text())
        assert old['query_ids'] == [q['mol_id'] for q in queries]
        assert np.array_equal(order[:, 0], old['candidate_indices']), 'Top1 changed'
        rank = np.argmax(order == targets[:, None], axis=1) + 1
        ranks[method] = rank
        report['methods'][method] = {str(k): dict(count=int((rank <= k).sum()),
            rate=float((rank <= k).mean())) for k in (1, 5, 10)}
        for i, row in enumerate(details):
            row[method] = dict(true_rank=int(rank[i]), top10=[dict(mol_id=ids[j],
                canonical_smiles=names[j], score=float(matrix[i, j])) for j in order[i, :10]])
    for k in (1, 5, 10):
        a, b = ranks['model'] <= k, ranks['raw_knn'] <= k
        report['overlap'][str(k)] = dict(both=int((a & b).sum()), model_only=int((a & ~b).sum()),
            knn_only=int((~a & b).sum()), neither=int((~a & ~b).sum()))
    report['top1_reproduced'] = True
    out.mkdir()
    save(out / 'per_query.json', details)
    save(out / 'summary.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

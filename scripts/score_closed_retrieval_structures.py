"""功能概述：将已冻结的检索首选结构按纯 LLM baseline 的同一结构指标重新计分。
主要职责：main 将原版或修复版数据绑定到检索协议中的 manifest hash，校验查询顺序和候选身份，用首选 mol_id 取预测 SMILES；
仅构造解析器输入后复用 run_closed_llm_direct.evaluate，绝不调用 API 或重跑检索。
代码库关系：复用 baseline 的 load_queries、group_identity、严格/原论文解析器以及
版本化 Tanimoto/GED；GED 四线程、每次 10 秒，null 不计均值，保存有效样本数。
输入输出与副作用：只读冻结数据及检索 JSONL，写新输出目录和逐条 metrics；已有目录拒绝。
预测只来自第一候选，不使用真实标签选择候选；source hashes 绑定此前排名与评分代码。
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scripts.run_closed_llm_direct import load_queries, evaluate, digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--retrieval-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol = json.loads((args.retrieval_dir / 'protocol.json').read_text())
    if protocol['sample_manifest_sha256'] != digest(args.data_root / 'sample_manifest.json'):
        raise ValueError('retrieval sample manifest mismatch')
    queries, _, groups = load_queries(args.data_root, protocol['sample_manifest_sha256'])
    if protocol['query_mol_ids'] != [q['mol_id'] for q in queries]:
        raise ValueError('retrieval query order mismatch')
    candidate = pd.read_parquet(args.data_root / 'theory_train_candidate/smiles.parquet').set_index('mol_id')
    sources = {}
    prepared = {}
    for method in ('knn_cosine', 'spectrum_graph_v2'):
        path = args.retrieval_dir / f'{method}.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if [r['mol_id'] for r in records] != [q['mol_id'] for q in queries]:
            raise ValueError('prediction/query order mismatch')
        if [r['group_hash'] for r in records] != [q['group_hash'] for q in queries]:
            raise ValueError('prediction/query group mismatch')
        prepared[method] = [dict(content='PREDICTED_SMILES: ' + candidate.loc[r['top10_mol_ids'][0], 'smiles']) for r in records]
        sources[method] = digest(path)
    args.output_dir.mkdir(parents=True)
    save(args.output_dir / 'protocol.json', dict(prediction='first retrieved candidate only; no oracle selection',
        retrieval_protocol_sha256=digest(args.retrieval_dir / 'protocol.json'), source_predictions_sha256=sources,
        sample_manifest_sha256=protocol['sample_manifest_sha256'], query_ids=protocol['query_mol_ids'],
        evaluator_sha256=digest(Path(__file__).with_name('run_closed_llm_direct.py')),
        metrics_sha256=digest(Path(__file__).resolve().parents[1] / 'spectraexpert/eval/metrics.py'),
        ged_timeout_seconds=10, workers=4, api_calls=0))
    summary = {}
    for method, responses in prepared.items():
        out = args.output_dir / method
        (out / 'metrics').mkdir(parents=True)
        evaluate(queries, responses, groups, out)
        result = json.loads((out / 'results.json').read_text())
        summary[method] = result['summary']
    save(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

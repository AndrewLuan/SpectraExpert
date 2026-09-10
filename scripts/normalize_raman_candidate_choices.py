"""功能概述：对完成的候选选择输出兼容 Markdown 包裹，保留原严格评分不变。
主要职责：parse_markdown_choice 去掉粗体与代码标记后仍要求唯一合法索引；main 对
全部输出应用同一规则，仅重新计算选择实际变化的记录，输出 format_normalized 子目录。
代码库关系：复用 run_raman_candidate_llm 的严格索引契约、既有结构评分和 baseline
聚合。原 API 请求、响应、用量、严格评分不修改，不重新请求模型，不按真实结构选择解析。
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_raman_candidate_llm import parse_choice
from scripts.run_closed_llm_direct import load_queries, evaluate, save
from scripts.refresh_closed_llm_baseline import summarize
from scripts.repair_closed_raman_exact import canonical


def parse_markdown_choice(text):
    return parse_choice(text.replace('**', '').replace('`', ''))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    out = args.out; normalized = out / 'format_normalized'
    normalized.mkdir(exist_ok=False)
    config = json.loads((out / 'config.json').read_text())
    queries, _, groups = load_queries(Path(config['data_root']), config['manifest_sha256'])
    changes = []
    for method in ('model', 'raw_knn'):
        rows = json.loads((out / method / 'selections.json').read_text())
        result = json.loads((out / method / 'results.json').read_text())
        folder = normalized / method; folder.mkdir()
        for i, row in enumerate(rows):
            response = json.loads((out / 'responses' / f"{row['request_index']:04d}.json").read_text())
            choice = parse_markdown_choice(response.get('content', '')) if not response.get('error') else None
            index = choice+1 if choice is not None else None
            if index == row['selected_index']:
                continue
            changes.append(dict(method=method, query_index=i, old_index=row['selected_index'], new_index=index))
            predicted = row['candidates'][choice] if choice is not None else None
            row.update(selected_index=index, predicted_smiles=predicted,
                exact=predicted == canonical(queries[i]['ground_truth_smiles']))
            correction = folder / f'correction_{i:04d}'
            (correction / 'metrics').mkdir(parents=True)
            evaluate([queries[i]], [dict(content='PREDICTED_SMILES: '+(predicted or 'UNKNOWN'))], groups, correction)
            updated = json.loads((correction / 'results.json').read_text())['queries'][0]
            updated['query_index'] = i
            result['queries'][i] = updated
        result['summary'] = summarize(result['queries'])
        save(folder / 'results.json', result)
        save(folder / 'selections.json', rows)
    save(normalized / 'config.json', dict(config, normalization='strip Markdown bold/backtick wrappers, then require unique index 1..10'))
    save(normalized / 'changes.json', changes)
    print(json.dumps(dict(changes=changes, summaries={m:json.loads((normalized/m/'results.json').read_text())['summary'] for m in ('model','raw_knn')})), flush=True)


if __name__ == '__main__':
    main()

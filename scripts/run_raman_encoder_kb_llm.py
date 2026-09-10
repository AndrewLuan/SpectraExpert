"""功能概述：在冻结的 200 条 Encoder Top10 请求上增加 Stage 2 知识库证据，进行配对评测。
主要职责：prepare 保留原峰表与候选，冻结新请求、证据与依赖哈希；run 复用已有 API
限流、重试和完整用量留档；score 使用同一 Markdown 索引解析及结构指标，记录配对得失。
代码库关系：调用 Stage2PeakAnalysis 和 run_raman_candidate_llm 的推理器，使用
run_closed_llm_direct 的冻结查询校验与评价函数，不训练或重新检索 Encoder。
输入输出与副作用：prepare 只读既有实验产物并写新目录；run 从 stdin JSON 读取密钥，
密钥只驻留内存，调用既有 USTC endpoint；恢复时校验请求与依赖，成功响应不重问。
"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_raman_candidate_llm as runner
from scripts import run_closed_llm_direct as base
from scripts.normalize_raman_candidate_choices import parse_markdown_choice
from spectraexpert.pipeline.config import PipelineConfig
from spectraexpert.pipeline.data_types import DetectedPeak, Stage1Result
from spectraexpert.pipeline.stage2_peak_analysis import Stage2PeakAnalysis

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = [
    'scripts/run_raman_encoder_kb_llm.py', 'scripts/run_raman_candidate_llm.py',
    'scripts/run_closed_llm_direct.py', 'scripts/normalize_raman_candidate_choices.py',
    'spectraexpert/pipeline/stage2_peak_analysis.py', 'spectraexpert/pipeline/config.py',
    'spectraexpert/pipeline/data_types.py',
    'spectraexpert/pipeline/knowledge/structured_rule_db.py',
    'spectraexpert/pipeline/knowledge/nakamoto_loader.py',
    'spectraexpert/pipeline/knowledge/peak_matcher.py',
    'knowledge_base/nakamoto_rules_structured.json', 'nakamoto_empirical_rules_dataset.json',
    'spectraexpert/eval/evaluate.py', 'spectraexpert/eval/metrics.py',
]


def prepare(parent, out):
    old = json.loads((parent / 'config.json').read_text())
    source_tasks = json.loads((parent / 'tasks.json').read_text())
    queries = json.loads((parent / 'queries.json').read_text())
    for name, expected in old['files_sha256'].items():
        assert base.digest(parent / name) == expected, name
    tasks = [t for t in source_tasks if t['method'] == 'model']
    assert len(tasks) == len(queries) == 200
    assert [t['query_index'] for t in tasks] == list(range(200))
    analyzer = Stage2PeakAnalysis(PipelineConfig.defaults().to_dict())
    assert analyzer._structured_db is not None
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('requests', 'responses', 'attempts', 'evidence'):
        (out / folder).mkdir()
    frozen_tasks = []
    for i, t in enumerate(tasks):
        request = json.loads((parent / 'requests' / f"{t['request_index']:04d}.json").read_text())
        prompt = request['messages'][1]['content']
        peaks = []
        for line in prompt.splitlines():
            if re.match(r'^\|\s*\d+\s*\|', line):
                fields = [s.strip() for s in line.split('|')[1:-1]]
                position, intensity, prominence, width = map(float, fields[1:])
                peaks.append(DetectedPeak(position, intensity, intensity, prominence, width, 0, 0))
        assert len(peaks) == t['peak_count'] and peaks
        result = analyzer.run(Stage1Result(base.np.zeros(256), peaks, []))
        request['model'] = 'deepseek-v4-flash-ascend'
        request['messages'][1]['content'] += (
            '\n\n## Retrieved spectroscopic knowledge\n'
            'Use these rules as supporting hypotheses and cite relevant rule IDs and source keys. '
            'The score measures interval matching, not a probability of molecular identity. '
            'Rules may mention IR and Raman bands: an IR-only band is not direct Raman evidence. '
            'Check the stated activity, molecular system and phase before using a rule. '
            'Missing or overlapping bands do not by themselves exclude a candidate.\n\n'
            + result.relevant_rules_summary
        )
        base.save(out / 'requests' / f'{i:04d}.json', request)
        base.save(out / 'evidence' / f'{i:04d}.json', asdict(result))
        frozen_tasks.append(dict(t, request_index=i, parent_request_index=t['request_index']))
    base.save(out / 'tasks.json', frozen_tasks)
    base.save(out / 'queries.json', queries)
    config = dict(old, schema='encoder_kb_llm_paired_v1', model='deepseek-v4-flash-ascend',
        parent_run=str(parent.resolve()), reused_request_indices=[],
        input='unchanged frozen peak table and Encoder Top10 + current Stage2 evidence; no truth in prompts',
        knowledge_settings=PipelineConfig.defaults().to_dict()['stage2']['knowledge'],
        knowledge_note='legacy rules mix IR/Raman; prompt requires checking stated activity; no new chemistry validation',
        inference_started=False,
        source_sha256={name: base.digest(ROOT / name) for name in DEPENDENCIES},
        files_sha256={str(f.relative_to(out)): base.digest(f) for folder in ('requests', 'evidence')
                      for f in sorted((out / folder).glob('*.json'))}
                    | {name: base.digest(out / name) for name in ('queries.json', 'tasks.json')})
    base.save(out / 'config.json', config)
    print(json.dumps(dict(prepared=len(frozen_tasks), model=config['model'],
        retrieval_ceiling=sum(t['true_rank'] <= 10 for t in frozen_tasks), out=str(out))), flush=True)


def score(out, config, tasks):
    queries, _, groups = base.load_queries(Path(config['data_root']), config['manifest_sha256'])
    assert queries == json.loads((out / 'queries.json').read_text())
    previous = json.loads((Path(config['parent_run']) / 'format_normalized/model/selections.json').read_text())
    resolved, selections = [], []
    for t, old in zip(tasks, previous):
        assert old['query_index'] == t['query_index'] and old['candidates'] == t['candidates']
        response = json.loads((out / 'responses' / f"{t['request_index']:04d}.json").read_text())
        choice = parse_markdown_choice(response.get('content', '')) if not response.get('error') else None
        smiles = t['candidates'][choice] if choice is not None else None
        exact = smiles == runner.canonical(queries[t['query_index']]['ground_truth_smiles'])
        resolved.append(dict(content='PREDICTED_SMILES: ' + (smiles or 'UNKNOWN')))
        selections.append(dict(t, selected_index=choice+1 if choice is not None else None,
            predicted_smiles=smiles, exact=exact, previous_exact=old['exact'], api_error=response.get('error')))
    folder = out / 'model'; (folder / 'metrics').mkdir(parents=True, exist_ok=True)
    base.save(folder / 'selections.json', selections)
    summary = dict(n=len(tasks), exact=sum(s['exact'] for s in selections),
        valid=sum(s['selected_index'] is not None for s in selections),
        gained=sum(s['exact'] and not s['previous_exact'] for s in selections),
        lost=sum(not s['exact'] and s['previous_exact'] for s in selections),
        previous_exact=sum(s['previous_exact'] for s in selections))
    base.save(out / 'selection_summary.json', summary)
    print(json.dumps(summary), flush=True)
    base.evaluate(queries, resolved, groups, folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'score'))
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.parent, args.out)
        return
    config, tasks, requests = runner.load(args.out)
    code = 1
    try:
        if args.mode == 'run':
            keys = json.loads(sys.stdin.readline())['keys']
            assert 1 <= len(keys) <= 3 and len(keys) == len(set(keys)) and all(keys)
            asyncio.run(runner.infer(keys, requests, args.out))
            keys.clear()
            (args.out / 'inference_finished_at').write_text(str(time.time()))
        runner.usage_report(args.out, tasks)
        score(args.out, config, tasks)
        code = 0 if all(not json.loads(f.read_text()).get('error')
                        for f in (args.out / 'responses').glob('*.json')) else 2
    finally:
        (args.out / 'exit_code').write_text(str(code))
    raise SystemExit(code)


if __name__ == '__main__':
    main()

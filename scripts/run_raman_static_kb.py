"""功能概述：在冻结验证谱和 Encoder Top10 上测试整份原始知识库直接注入提示。
主要职责：prepare 校验父实验请求哈希，向无知识库请求追加未经筛选的完整 Markdown；
run 通过既有限流器调用模型，密钥从 stdin 读入内存；score 计算 exact、Tanimoto、
分子组等权准确率和相对父实验各组的配对得失，保存实际报告的 token 用量。
代码库关系：复用 run_raman_candidate_llm 的推理与哈希校验、run_closed_llm_direct
的指标，以及 normalize_raman_candidate_choices 的索引解析，不调用 Stage2 或 RDKit 筛选。
输入输出与副作用：只消费既有验证请求、任务和原始知识 Markdown，不读取原始数据或
最终测试谱。独占新输出目录；父结果仅只读比较。所有请求、来源和父选择结果冻结哈希；
保留规则编号与原文出处，不改峰表、候选、system prompt、模型参数，不添加候选标注。
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_raman_candidate_llm as runner
from scripts import run_closed_llm_direct as base
from scripts.normalize_raman_candidate_choices import parse_markdown_choice

ROOT = Path(__file__).resolve().parents[1]


def prepare(parent, out, knowledge):
    cfg = json.loads((parent / 'config.json').read_text())
    for name, sha in cfg['files_sha256'].items():
        assert base.digest(parent / name) == sha, name
    source = json.loads((parent / 'tasks.json').read_text())
    rows = [t for t in source if t['method'] == 'no_kb']
    assert len(rows) == 149 and len({t['group_hash'] for t in rows}) == 142
    assert [t['query_index'] for t in rows] == list(range(149))
    kb = knowledge.read_text()
    assert kb and '181' in kb
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('requests', 'responses', 'attempts'):
        (out / folder).mkdir()
    tasks = []
    for i, row in enumerate(rows):
        r = json.loads((parent / 'requests' / f"{row['request_index']:04d}.json").read_text())
        assert r['messages'][0]['content'] == runner.SYSTEM
        assert (r['model'], r['temperature'], r['max_tokens']) == ('deepseek-v4-flash-ascend', .2, 4096)
        r['messages'][1]['content'] += ('\n\n## Complete spectroscopic reference knowledge\n'
            'The following is the complete reference, not evidence preassigned to the observed peaks. '
            'Use applicable knowledge to compare the candidates and cite relevant rule IDs and source sections.\n\n' + kb)
        base.save(out / 'requests' / f'{i:04d}.json', r)
        tasks.append(dict(row, request_index=i, parent_request_index=row['request_index'], method='static_full_kb'))
    base.save(out / 'tasks.json', tasks)
    sources = ['scripts/run_raman_static_kb.py', 'scripts/run_raman_candidate_llm.py',
        'scripts/run_closed_llm_direct.py', 'scripts/normalize_raman_candidate_choices.py',
        'spectraexpert/eval/metrics.py', 'spectraexpert/eval/evaluate.py']
    base.save(out / 'config.json', dict(schema='raman_static_full_kb_v1', parent_run=str(parent.resolve()),
        validation_usage=cfg['validation_usage'], queries=149, groups=142,
        candidate_count=cfg['candidate_count'], knowledge_path=str(knowledge.resolve()),
        knowledge_sha256=base.digest(knowledge), knowledge_chars=len(kb),
        parent_selections_sha256=base.digest(parent / 'selections.json'),
        parent_config_sha256=base.digest(parent / 'config.json'),
        model='deepseek-v4-flash-ascend', temperature=.2, max_tokens=4096,
        comparison='one new generation per query; prior six arms reused as historical paired controls; no seed stability claim',
        source_sha256={name: base.digest(ROOT / name) for name in sources},
        files_sha256={str(f.relative_to(out)): base.digest(f) for f in (out / 'requests').glob('*.json')}
            | {'tasks.json': base.digest(out / 'tasks.json')}))
    print(json.dumps(dict(prepared=len(tasks), knowledge_chars=len(kb), out=str(out))), flush=True)


def score(out, cfg, tasks):
    parent = Path(cfg['parent_run'])
    assert base.digest(parent / 'selections.json') == cfg['parent_selections_sha256']
    prior = json.loads((parent / 'selections.json').read_text())
    details = []
    for t in tasks:
        r = json.loads((out / 'responses' / f"{t['request_index']:04d}.json").read_text())
        choice = parse_markdown_choice(r.get('content', '')) if not r.get('error') else None
        predicted = t['candidates'][choice] if choice is not None else None
        details.append(dict(t, selected_index=choice + 1 if choice is not None else None,
            exact=predicted == t['ground_truth'], valid=choice is not None,
            tanimoto=base.tanimoto_similarity(predicted, t['ground_truth']) if predicted else None))
    pairs = {}
    for arm in sorted({r['method'] for r in prior}):
        controls = {r['query_index']: r for r in prior if r['method'] == arm}
        gained = lost = 0
        for r in details:
            b = controls[r['query_index']]
            assert all(r[k] == b[k] for k in ('candidates', 'ground_truth', 'group_hash', 'true_rank'))
            gained += r['exact'] and not b['exact']
            lost += not r['exact'] and b['exact']
        pairs[arm] = dict(gained=gained, lost=lost, net=gained-lost,
            baseline_exact=sum(r['exact'] for r in controls.values()))
    vals = [r['tanimoto'] for r in details if r['tanimoto'] is not None]
    groups = {r['group_hash'] for r in details}
    summary = dict(n=len(details), exact=sum(r['exact'] for r in details),
        valid=sum(r['valid'] for r in details),
        group_exact=sum(sum(r['exact'] for r in details if r['group_hash']==g) /
            sum(r['group_hash']==g for r in details) for g in groups) / len(groups),
        tanimoto_mean=sum(vals)/len(vals) if vals else None, tanimoto_n=len(vals), comparisons=pairs)
    base.save(out / 'selections.json', details)
    base.save(out / 'summary.json', summary)
    runner.usage_report(out, tasks)
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'run', 'score'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--knowledge', type=Path, default=ROOT / 'nakamoto_empirical_rules_markdown.md')
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.parent, a.out, a.knowledge)
        return
    cfg, tasks, requests = runner.load(a.out)
    assert base.digest(Path(cfg['knowledge_path'])) == cfg['knowledge_sha256']
    code = 1
    try:
        if a.mode == 'run':
            keys = json.loads(sys.stdin.readline())['keys']
            assert 1 <= len(keys) <= 3 and len(set(keys)) == len(keys) and all(keys)
            asyncio.run(runner.infer(keys, requests, a.out))
            keys.clear()
            (a.out / 'inference_finished_at').write_text(str(time.time()))
        score(a.out, cfg, tasks)
        code = 0 if all(not json.loads(f.read_text()).get('error') for f in (a.out / 'responses').glob('*.json')) else 2
    finally:
        (a.out / 'exit_code').write_text(str(code))
    raise SystemExit(code)


if __name__ == '__main__':
    main()

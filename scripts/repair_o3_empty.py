"""功能概述：为已完成的多模型 Agent 实验补跑 o3 或 GPT-5 的空正文截断响应。
主要职责：prepare 验证父实验哈希，以“空正文且 finish_reason=length”为唯一补跑条件，
提高这些请求的 max_tokens，复制其余 o3 成功响应；score 复用未变预测的结构指标并
仅为改变的预测调用现有评分器；main 准备或评分实验，不调用 API。
代码库关系：消费 run_agent_multimodel 的冻结请求，生成同 schema 子实验供其 run/score
使用。候选、峰表、知识、温度及已有答案不变；新 protocol 记录逐条预算与复用来源哈希。
输入输出与副作用：新目录拒绝覆盖，父产物只读；复制原始请求和响应确保不同预算可审计。
API 返回空正文不是成功修复，完成后必须另外检查响应内容，不能仅依赖 HTTP 状态。
"""
import argparse
import copy
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_agent_multimodel as base


def prepare(parent, out, tokens, tag='o3'):
    protocol = base.load(parent)
    models = [m for m in protocol['config']['models'] if m['tag'] == tag]
    if tag not in ('o3','gpt5') or len(models) != 1 or models[0]['model'] != {'o3':'o3','gpt5':'gpt-5'}[tag] or not 4096 < tokens <= 32768:
        raise ValueError('requires o3/gpt5 and 4096 < max_tokens <= 32768')
    replacement, reuse, requests, responses = [], [], [], []
    for i in range(200):
        request = base.read(parent/tag/'requests'/f'{i:04d}.json')
        response = base.read(parent/tag/'responses'/f'{i:04d}.json')
        if response.get('error'):
            raise ValueError('parent contains API errors')
        if not response.get('content', '').strip():
            if response.get('finish_reason') != 'length' or tokens <= request['max_tokens']:
                raise ValueError('empty response is not repairable by this budget increment')
            request['max_tokens'] = tokens
            replacement.append(i)
        else:
            reuse.append(i)
        requests.append(request)
        responses.append(response)
    if not replacement:
        raise ValueError('no empty truncated responses to repair')
    out.mkdir(parents=True, exist_ok=False)
    for sub in ('requests', 'responses', 'attempts'):
        (out/tag/sub).mkdir(parents=True)
    for name in ('queries.json', 'tasks.json'):
        shutil.copy2(parent/name, out/name)
    for i, request in enumerate(requests):
        base.save(out/tag/'requests'/f'{i:04d}.json', request)
        if i in reuse:
            shutil.copy2(parent/tag/'responses'/f'{i:04d}.json', out/tag/'responses'/f'{i:04d}.json')
    child = copy.deepcopy(protocol)
    child['config']['models'] = models
    child['repair'] = dict(parent=str(parent.resolve()), parent_protocol_sha256=base.digest(parent/'protocol.json'),
        replacement_indices=replacement, reused_indices=reuse, max_tokens=tokens,
        criterion='empty content and finish_reason=length; not selected using ground truth or accuracy',
        source_responses_sha256={f'{i:04d}':base.digest(parent/tag/'responses'/f'{i:04d}.json') for i in range(200)})
    child['files_sha256'] = {str(p.relative_to(out)):base.digest(p) for p in out.rglob('*.json')}
    # 复用响应也纳入哈希；任何丢失或改写应阻止续跑，以免增加未经计划的请求。
    base.save(out/'protocol.json', child)
    return dict(reused=len(reuse), pending=len(replacement), max_tokens=tokens)


def score(out, previous_scores, tag='o3'):
    from scripts.normalize_raman_candidate_choices import parse_markdown_choice
    from scripts import run_closed_llm_direct as metrics
    from scripts.refresh_closed_llm_baseline import summarize
    from rdkit import Chem
    protocol=base.read(out/'protocol.json')
    for name,sha in protocol['files_sha256'].items():
        if base.digest(out/name)!=sha:raise ValueError('frozen input mismatch: '+name)
    source=protocol['source_config']
    queries,_,groups=metrics.load_queries(Path(source['data_root']),source['manifest_sha256'])
    if queries!=base.read(out/'queries.json'):raise ValueError('query mismatch')
    old=base.read(previous_scores)['queries'];tasks=base.read(out/'tasks.json')
    if len(old)!=200 or [r['query_index'] for r in old]!=list(range(200)):raise ValueError('invalid previous score order')
    changed=[];resolved=[];selections=[];merged=copy.deepcopy(old)
    for i,(query,task,row) in enumerate(zip(queries,tasks,old)):
        if any(row[k]!=query[k] for k in query):raise ValueError('previous score identity mismatch')
        response=base.read(out/tag/'responses'/f'{i:04d}.json')
        choice=None if response.get('error') else parse_markdown_choice(response.get('content',''))
        smiles=task['candidates'][choice] if choice is not None else None
        canonical=Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if smiles else None
        selections.append(dict(query_index=i,selected_index=choice+1 if choice is not None else None,predicted_smiles=smiles))
        if canonical!=row['strict_smiles'] or response.get('error')!=row.get('api_error'):
            changed.append(i);resolved.append(dict(content='PREDICTED_SMILES: '+(smiles or 'UNKNOWN'),error=response.get('error')))
    target=out/tag/'scored';target.mkdir(exist_ok=False)
    addition=target/'changed';(addition/'metrics').mkdir(parents=True)
    if changed:
        metrics.evaluate([queries[i] for i in changed],resolved,groups,addition)
        for i,row in zip(changed,base.read(addition/'results.json')['queries']):
            row['query_index']=i;merged[i]=row
    base.save(target/'selections.json',selections)
    base.save(target/'results.json',dict(summary=summarize(merged),queries=merged))
    base.save(target/'provenance.json',dict(previous_scores=str(previous_scores),previous_sha256=base.digest(previous_scores),
        rescored_indices=changed,reused_metrics=200-len(changed),
        metric_sha256=base.digest(Path(sys.modules['spectraexpert.eval.metrics'].__file__)),
        scorer_sha256=base.digest(Path(metrics.__file__)),repair_scorer_sha256=base.digest(Path(__file__))))
    print(base.read(target/'results.json')['summary'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--tag',choices=['o3','gpt5','gemini'],default='o3')
    parser.add_argument('--previous-scores',type=Path,help='提供此项时只评分，不准备或调用 API')
    args = parser.parse_args()
    if args.previous_scores:score(args.out,args.previous_scores,args.tag)
    elif args.parent:print(prepare(args.parent, args.out, args.max_tokens,args.tag))
    else:parser.error('requires --parent or --previous-scores')

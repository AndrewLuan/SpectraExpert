"""功能概述：汇总五模型 Agent 结构指标、选择得失、调用成本与 o3 修复边界。
主要职责：summarize 读取统一评分 JSON，计算有效/全样本指标及相对 DeepSeek 的配对区间；
usage 汇总原调用及补跑全部已报告 tokens 和耗时；main 导出可追溯 JSON/中文 Markdown。
代码库关系：结构分数只消费 run_closed_llm_direct 的 strict 字段；复用冻结 tasks 的
候选和 true_rank。chemical nGED 用原 GED 除以两图重原子数和键数之和，不重新搜索。
输入输出与副作用：只读本机请求、响应、评分和远端来源快照；输出新目录，拒绝覆盖。
不调用 API、不修改答案；可消费独立保存的 Gemini 无歧义格式归一化产物。
o3/GPT-5 混合预算、原始格式合规率及 valid-only 偏差单列。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem


def read(path):
    return json.loads(path.read_text())


def observed(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                std=float(a.std(ddof=1)) if len(a)>1 else None)


def interval(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    rng=np.random.default_rng(0)
    samples=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(axis=1)
    return np.quantile(samples,[.025,.975]).tolist()


def usage(folders):
    totals={}; seconds=[]; missing=0; attempts=0
    for folder in folders:
        for f in sorted((folder/'attempts').glob('*.json')):
            r=read(f);attempts+=1
            try:u=json.loads(r.get('response_text','{}')).get('usage')
            except ValueError:u=None
            if not u:missing+=1
            for k,v in (u or {}).items():
                if isinstance(v,(int,float)):totals[k]=totals.get(k,0)+v
            if r.get('http_status')==200 and 'elapsed_seconds' in r:seconds.append(r['elapsed_seconds'])
    return dict(attempts=attempts,attempts_without_usage=missing,tokens=totals,
                successful_attempt_latency=observed(seconds),p95_seconds=float(np.percentile(seconds,95)) if seconds else None,
                monetary_cost=None, monetary_cost_note='No verified pricing/billing supplied')


def summarize(run, repair, scores, out, gpt_repair=None, gemini_normalized=None):
    out.mkdir(parents=True,exist_ok=False)
    tasks=read(run/'tasks.json');queries=read(run/'queries.json')
    assert len(tasks)==len(queries)==200
    assert len({q['group_hash'] for q in queries})==200, 'query bootstrap assumes unique molecular groups'
    tags=['gpt5','o3','claude','gemini','deepseek']
    repair_chain=[];cursor=repair
    while cursor.resolve()!=run.resolve():
        detail=read(cursor/'protocol.json')['repair']
        repair_chain.append((cursor,detail));cursor=Path(detail['parent'])
    gpt_chain=[];cursor=gpt_repair
    while cursor and cursor.resolve()!=run.resolve():
        detail=read(cursor/'protocol.json')['repair']
        gpt_chain.append((cursor,detail));cursor=Path(detail['parent'])
    rows={tag:read(scores/tag/'results.json')['queries'] for tag in tags}
    for tag,items in rows.items():
        assert [r['query_index'] for r in items]==list(range(200)),tag
        assert [r['ground_truth_smiles'] for r in items]==[q['ground_truth_smiles'] for q in queries],tag
    truth=[Chem.MolToSmiles(Chem.MolFromSmiles(q['ground_truth_smiles'])) for q in queries]
    top1=[t['candidates'][0]==gt for t,gt in zip(tasks,truth)]
    ceiling=[t['true_rank']<=10 for t in tasks]
    report=dict(protocol='exact_v2 Test-200; frozen encoder+legacy KB; Markdown-normalized strict index',
                n=200,retrieval_top1_hits=sum(top1),retrieval_top10_hits=sum(ceiling),
                bootstrap='10000 paired query resamples; seed=0; each query is a unique molecular group; exploratory CIs',models={})
    for tag in tags:
        items=rows[tag];values=[r['strict'] for r in items]
        exact=[bool(v['exact_match']) for v in values]
        valid=[bool(v['valid']) for v in values]
        normalized=[]
        for row,v in zip(items,values):
            ged=v['chemical_ged_v2']
            if ged is None:normalized.append(None);continue
            a=Chem.MolFromSmiles(row['ground_truth_smiles']);b=Chem.MolFromSmiles(row['strict_smiles'])
            normalized.append(ged/(a.GetNumHeavyAtoms()+b.GetNumHeavyAtoms()+Chem.RemoveHs(a).GetNumBonds()+Chem.RemoveHs(b).GetNumBonds()))
        result=dict(n=200,api_errors=sum(bool(r.get('api_error')) for r in items),valid=sum(valid),
            valid_rate=sum(valid)/200,exact=sum(exact),exact_rate=sum(exact)/200,exact_ci95=interval(exact),
            group_match_rate=sum(v['group_match'] for v in values)/200,
            selection_accuracy_when_target_in_top10=sum(e and c for e,c in zip(exact,ceiling))/sum(ceiling),
            gained_vs_retrieval=sum(e and not t for e,t in zip(exact,top1)),
            lost_vs_retrieval=sum(t and not e for e,t in zip(exact,top1)),
            retained_correct=sum(t and e for e,t in zip(exact,top1)),
            both_incorrect=sum(not t and not e for e,t in zip(exact,top1)),
            tanimoto_valid_only=observed([v['tanimoto'] for v in values]),
            tanimoto_all_queries_invalid_zero=float(np.mean([v['tanimoto'] if v['tanimoto'] is not None else 0 for v in values])),
            legacy_ged_v1=observed([v['legacy_ged_v1'] for v in values]),
            chemical_ged_v2=observed([v['chemical_ged_v2'] for v in values]),chemical_normalized_ged=observed(normalized))
        if tag!='deepseek':
            folder=repair/'o3' if tag=='o3' else gpt_repair/'gpt5' if tag=='gpt5' and gpt_repair else gemini_normalized/'gemini' if tag=='gemini' and gemini_normalized else run/tag
            responses=[read(folder/'responses'/f'{i:04d}.json') for i in range(200)]
            import re
            original=[read(run/tag/'responses'/f'{i:04d}.json') for i in range(200)]
            raw_valid=0
            for response in original:
                matches=re.findall(r'^\s*SELECTED_INDEX\s*:\s*(\d+)\s*$',response.get('content','').replace('**','').replace('`',''),re.M|re.I)
                raw_valid+=len(matches)==1 and 1<=int(matches[0])<=10
            result['initial_schema_compliant_count']=raw_valid
            result['initial_empty_content_count']=sum(not r.get('content','').strip() for r in original)
            result.update(empty_content=sum(not r.get('content','').strip() for r in responses),
                          truncated=sum(r.get('finish_reason')=='length' for r in responses),
                          returned_models=sorted({str(r.get('response_model')) for r in responses}),
                          usage=usage([run/tag]+[folder/'o3' for folder,_ in repair_chain] if tag=='o3' else [run/tag]+[folder/'gpt5' for folder,_ in gpt_chain] if tag=='gpt5' else [run/tag]))
        else:result['usage']=None
        ref=[r['strict'] for r in rows['deepseek']]
        diff=[float(e)-float(v['exact_match']) for e,v in zip(exact,ref)]
        result['vs_deepseek']=dict(exact_difference=float(np.mean(diff)),exact_difference_ci95=interval(diff))
        for metric in ['tanimoto','chemical_ged_v2']:
            pairs=[(v[metric],r[metric]) for v,r in zip(values,ref) if v[metric] is not None and r[metric] is not None]
            delta=[a-b if metric=='tanimoto' else b-a for a,b in pairs]
            result['vs_deepseek'][metric]=dict(common_n=len(pairs),improvement=observed(delta),improvement_ci95=interval(delta))
        report['models'][tag]=result
    report['o3_repair']=read(repair/'protocol.json')['repair']
    report['o3_repair_history']=[detail for _,detail in reversed(repair_chain)]
    report['gpt5_repair_history']=[detail for _,detail in reversed(gpt_chain)]
    report['gemini_normalization']=read(gemini_normalized/'protocol.json')['normalization'] if gemini_normalized else None
    sources=[scores/tag/'results.json' for tag in tags]+[run/'protocol.json',repair/'protocol.json',run/'tasks.json',run/'queries.json',Path(__file__)]
    if gpt_repair:sources.append(gpt_repair/'protocol.json')
    if gemini_normalized:sources.extend([gemini_normalized/'protocol.json',gemini_normalized/'normalization_audit.json'])
    report['source_sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# 多模型 Agent 评测与 o3 空正文修复','',
        '同一 exact-v2 Test-200、冻结 encoder Top10 与 legacy KB。指标沿用现有候选解析规则；所有模型的分母均为 200。','',
        '| 模型 | Valid | Exact | Tanimoto（有效样本） | Chemical GED（可计算 n） | Legacy GED | nGED |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def number(x):return 'NA' if x is None else f'{x:.4f}'
    for tag,r in report['models'].items():
        t=r['tanimoto_valid_only'];g=r['chemical_ged_v2']
        lines.append(f"| {tag} | {r['valid']}/200 | {r['exact']}/200 ({r['exact_rate']:.1%}) | {number(t['mean'])} ({t['n']}) | {number(g['mean'])} ({g['n']}) | {number(r['legacy_ged_v1']['mean'])} | {number(r['chemical_normalized_ged']['mean'])} |")
    lines+=['','o3 原 78 条空正文的 completion budget 均被 reasoning tokens 用完。仅按空正文截断条件提高预算补跑，122 条原有正文原样复用。最终表为此恢复策略结果，不能称所有 200 条均使用同一较大预算。','',
        'o3 首次补跑上限为 16,384 tokens；仍有一条再次耗尽预算，进一步以 32,768 tokens 补跑。全部尝试和不同预算均保留。','',
        'GPT-5 的 150 条空正文未补跑；Gemini 的字段格式失败保留为无效，未按模型单独放宽解析。有效样本上的 Tanimoto/GED 分母不同，不能直接作为跨模型能力排名。共同可计算子集的配对差异、95% 区间及 invalid=0 的全样本 Tanimoto 见 summary.json。','',
            'JSON 同时包含检索新增/丢失命中、Top10 内条件选择率、group match、指标均值/中位数/标准差、nGED、调用 tokens、延迟中位数/P95 和来源哈希。o3 用量包含首轮及补跑；货币费用因无核实账单保留为空。DeepSeek 用量未采集，不填零。','',
            '这些结果来自已多次观察的固定测试集；置信区间仅为探索性查询配对重采样，不反映模型 API 多次独立采样的不确定性。']
    if gpt_repair and gemini_normalized:
        lines=[line.replace('多模型 Agent 评测与 o3 空正文修复','多模型 Agent 评测：预算修复与最终答案格式归一化')
               .replace('指标沿用现有候选解析规则；所有模型的分母均为 200。','结构指标沿用原评分器；Gemini 先进行无歧义最终答案归一化，所有模型的分母均为 200。')
               .replace('GPT-5 的 150 条空正文未补跑；Gemini 的字段格式失败保留为无效，未按模型单独放宽解析。','GPT-5 保留 50 条成功结果，150 条空正文以 32,768-token 上限补跑；全部均恢复。Gemini 原始严格字段合规为 3/200，经候选索引或唯一候选 SMILES 的确定性格式归一化后为 200/200；未新增 API 请求，未改变原始候选选择。解析器只使用回答和候选，不接触真值；原始表与恢复后表须区分。') for line in lines]
        lines+=['','相同归一化规则已检查其他模型的原始回答，未改变其已有合法选择。GPT-5 和 o3 的恢复属于不同预算的失败恢复策略，不等同于全样本统一较大预算的独立重跑。','',
                '报告区分 initial_schema_compliant_count（初始格式合规）与 valid（恢复后可评分输出）；Gemini 的恢复不表示模型已学会指定格式。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({tag:dict(valid=r['valid'],exact=r['exact'],chemical_ged=r['chemical_ged_v2']) for tag,r in report['models'].items()},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','repair','scores','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpt-repair',type=Path);p.add_argument('--gemini-normalized',type=Path)
    a=p.parse_args();summarize(a.run,a.repair,a.scores,a.out,a.gpt_repair,a.gemini_normalized)

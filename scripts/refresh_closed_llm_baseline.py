"""功能概述：为精确修复版闭库只补跑替换查询，复用旧版其他请求、响应和 GED 结果。
主要职责：prepare 核对父结果/数据/提示词与新版请求，持久化 192 条复用响应及 8 条待请求；
finish 复用原 infer/evaluate，仅评分新增查询，并按原 200 槽位合并指标；summarize 复现原聚合。
代码库关系：复用 run_closed_llm_direct，要求 repair_closed_raman_exact 的可审计新版 manifest。
输入输出与副作用：prepare 不调用 API；run 的 USTC key 仅从 stdin JSON 或 USTC_API_KEY 内存获取。
输出新目录，续跑必须核对 manifest 和请求；不改旧结果，不重复请求成功响应，不记录密钥。
"""
import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_closed_llm_direct as base


def summarize(rows):
    result = dict(n_queries=len(rows), api_errors=sum(bool(r['api_error']) for r in rows))
    for label in ('strict', 'legacy'):
        values = [r[label] for r in rows]
        result[label] = {n+'_rate':sum(v[n] for v in values)/len(values)
                         for n in ('valid','exact_match','group_match')}
        for name in ('tanimoto','legacy_ged_v1','chemical_ged_v2'):
            valid = [v[name] for v in values if v[name] is not None]
            result[label][name] = dict(n=len(valid), mean=float(np.mean(valid)) if valid else None,
                                       median=float(np.median(valid)) if valid else None)
    return result


def prepare(root, parent, previous, output):
    manifest = json.loads((root/'sample_manifest.json').read_text())
    if manifest['parent_manifest_sha256'] != base.digest(parent/'sample_manifest.json'):
        raise ValueError('parent manifest mismatch')
    if manifest['repair_report_sha256'] != base.digest(root/'repair_report.json'):
        raise ValueError('repair report hash mismatch')
    report = json.loads((root/'repair_report.json').read_text())
    replaced = report['replaced_query_indices']
    if len(replaced) != 8 or len(set(replaced)) != 8:
        raise ValueError('expected eight replacement queries')
    old_config = json.loads((previous/'config.json').read_text())
    if old_config['manifest_sha256'] != manifest['parent_manifest_sha256']:
        raise ValueError('baseline parent mismatch')
    # 模型、峰检测、提示词与指标保持原设置；运行器唯一改动是允许显式的新 manifest hash。
    if (old_config['model'],old_config['temperature'],old_config['max_tokens'],old_config['endpoint']) != (base.MODEL,0.2,4096,base.ENDPOINT):
        raise ValueError('baseline inference configuration changed')
    repo = Path(__file__).resolve().parents[1]
    for name in ('spectraexpert/eval/evaluate.py','spectraexpert/eval/metrics.py',
                 'spectraexpert/pipeline/prompts/stage3_direct_identification.py'):
        if base.digest(repo/name) != old_config['source_sha256'][name]:
            raise ValueError('baseline prompt/detector/scorer changed: '+name)
    old_queries, old_spectra, _ = base.load_queries(parent)
    queries, spectra, groups = base.load_queries(root, base.digest(root/'sample_manifest.json'))
    old_results = json.loads((previous/'results.json').read_text())
    if len(old_results['queries']) != 200:
        raise ValueError('incomplete old baseline')
    detector = base._PeakDetector({'stage1':{'data_dir':str(root/'experimental_test')}})
    requests = [dict(model=base.MODEL, temperature=0.2, max_tokens=4096,
                     messages=base.make_messages(s,detector)[0]) for s in spectra]
    preserved = []
    for i in range(200):
        if i in replaced:
            continue
        if queries[i] != old_queries[i] or not np.array_equal(spectra[i],old_spectra[i]):
            raise ValueError('unchanged baseline query differs')
        if requests[i] != json.loads((previous/'requests'/f'{i:04d}.json').read_text()):
            raise ValueError('unchanged baseline request differs')
        row = copy.deepcopy(old_results['queries'][i])
        for k,v in queries[i].items():
            if row[k] != v:
                raise ValueError('old result identity mismatch')
        for label in ('strict','legacy'):
            pred = row[label+'_smiles']
            group = groups.get(base.group_identity(pred)) if pred else None
            row[label]['predicted_group_hash'] = group
            row[label]['group_match'] = group == queries[i]['group_hash']
        preserved.append(row)
    config = dict(manifest_sha256=base.digest(root/'sample_manifest.json'),
                  previous_results_sha256=base.digest(previous/'results.json'),
                  previous_dir=str(previous.resolve()), replaced_query_indices=replaced,
                  reused_queries=192, new_queries=8, model=base.MODEL, endpoint=base.ENDPOINT,
                  temperature=0.2,max_tokens=4096,source_sha256={str(Path(__file__).name):base.digest(Path(__file__)),
                  'run_closed_llm_direct.py':base.digest(Path(base.__file__))})
    if output.exists():
        if json.loads((output/'config.json').read_text()) != config:
            raise ValueError('resume configuration differs')
        for i,request in enumerate(requests):
            if json.loads((output/'requests'/f'{i:04d}.json').read_text()) != request:
                raise ValueError('resume request differs')
    else:
        output.mkdir(parents=True)
        for name in ('requests','responses','metrics','replacement_metrics'):
            (output/name).mkdir()
        (output/'replacement_metrics/metrics').mkdir()
        base.save(output/'config.json',config)
        base.save(output/'reused_metrics.json',preserved)
        for i, request in enumerate(requests):
            base.save(output/'requests'/f'{i:04d}.json',request)
            if i not in replaced:
                response=json.loads((previous/'responses'/f'{i:04d}.json').read_text())
                if 'content' not in response or 'error' in response:
                    raise ValueError('cannot reuse failed old response')
                base.save(output/'responses'/f'{i:04d}.json',response)
        base.save(output/'status.json',dict(state='prepared', reused_responses=192, pending_query_indices=replaced))
    # 续跑前再次检查复用槽位，禁止缺失/损坏的旧响应被 infer 当成新 API 请求。
    for i in range(200):
        if i in replaced:
            continue
        cached=output/'responses'/f'{i:04d}.json'
        expected=json.loads((previous/'responses'/f'{i:04d}.json').read_text())
        if not cached.exists() or json.loads(cached.read_text()) != expected or 'content' not in expected or 'error' in expected:
            raise ValueError('reused response missing or changed; refusing extra API calls')
    return queries, requests, groups, preserved, replaced


def finish(root, parent, previous, output, keys):
    queries, requests, groups, preserved, replaced = prepare(root,parent,previous,output)
    responses=asyncio.run(base.infer(keys,requests,output))
    keys.clear()
    if any(r is None or 'error' in r for r in responses):
        base.save(output/'status.json',dict(state='inference_failed'))
        raise RuntimeError('replacement inference failed; resume only failed queries')
    metric_dir=output/'replacement_metrics'
    base.evaluate([queries[i] for i in replaced],[responses[i] for i in replaced],groups,metric_dir)
    added=json.loads((metric_dir/'results.json').read_text())['queries']
    for index,row in zip(replaced,added):row['query_index']=index
    rows=sorted(preserved+added,key=lambda r:r['query_index'])
    if [r['query_index'] for r in rows]!=list(range(200)):
        raise ValueError('merged result order mismatch')
    for i,row in enumerate(rows):base.save(output/'metrics'/f'{i:04d}.json',row)
    summary=summarize(rows)
    base.save(output/'results.json',dict(summary=summary,queries=rows))
    base.save(output/'status.json',dict(state='complete',reused_queries=192,new_queries=8))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('data-root','parent','previous','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--keys-stdin',action='store_true')
    args=parser.parse_args()
    if not args.run:
        prepare(args.data_root,args.parent,args.previous,args.output)
        print('Prepared 192 reused responses and 8 replacement requests; zero API calls.',flush=True)
    else:
        keys=json.load(sys.stdin)['keys'] if args.keys_stdin else [os.environ.get('USTC_API_KEY','')]
        if not 1<=len(keys)<=3 or not all(isinstance(k,str) and k for k in keys):
            raise SystemExit('Set USTC_API_KEY or supply authorized keys through --keys-stdin; no request sent.')
        finish(args.data_root,args.parent,args.previous,args.output,keys)

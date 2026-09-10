"""功能概述：在独立于最终测试的冻结实验验证谱上进行知识库消融，六组共享候选与峰表。
主要职责：prepare 校验 checkpoint 和输入哈希、验证分子隔离，生成交错的六组请求；run
复用原有限流器与完整 API 留档；score 计算 exact、组等权 exact、Tanimoto、配对得失。
代码库关系：消费已存在的 hard/validation.json 和 prepared.npz；候选为该验证协议的
10704 结构库，不冒充最终测试的 10000 库。验证集曾用于 Encoder 选型，仅作开发消融。
输入输出与副作用：不重训、不读原始大型数据目录、不使用测试结果挑参数；新输出目录
独占创建。密钥由 stdin 进入内存；输出保存提示、证据、源哈希、响应及已报告用量。
"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts import run_raman_candidate_llm as runner
from scripts import run_closed_llm_direct as base
from scripts.normalize_raman_candidate_choices import parse_markdown_choice
from scripts.raman_mode_evidence import retrieve, format_evidence
from spectraexpert.pipeline.config import PipelineConfig
from spectraexpert.pipeline.data_types import Stage1Result
from spectraexpert.pipeline.stage2_peak_analysis import Stage2PeakAnalysis

ARMS = ('no_kb', 'legacy_kb', 'modes_all_full', 'modes_all_short',
        'modes_filtered_full', 'modes_filtered_short')
ROOT = Path(__file__).resolve().parents[1]


def prepare(run, out):
    protocol = json.loads((run/'protocol.json').read_text())
    selection = json.loads((run/'final/selection.json').read_text())
    assert selection['selected_arm'] == 'hard'
    assert selection['protocol_sha256'] == base.digest(run/'protocol.json')
    assert base.digest(run/'hard/best.pt') == selection['summaries']['hard']['checkpoint_sha256']
    assert base.digest(run/'prepared.npz') == protocol['hashes']['prepared.npz']
    v = json.loads((run/'hard/validation.json').read_text())
    metadata = protocol['experimental_metadata']
    excluded = set(protocol['excluded_test_groups'])
    training = set(protocol['train_groups'])
    groups = set(v['groups'])
    assert not groups & (excluded | training)
    assert groups <= set(protocol['validation_groups'])
    with np.load(run/'prepared.npz', allow_pickle=False) as d:
        indices = d['validation'].tolist()
        assert indices == v['experiment_indices']
        spectra = d['experiment'][indices]
        targets = d['targets'][indices]
    names = protocol['candidate_names']
    detector = base._PeakDetector({'stage1': {'data_dir':str(Path(protocol['data_root'])/'experimental_test')}})
    assert np.array_equal(detector._wavenumber, np.arange(500,4001))
    stage2 = Stage2PeakAnalysis(PipelineConfig.defaults().to_dict())
    assert stage2._structured_db is not None
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('requests','responses','attempts','evidence'):
        (out/folder).mkdir()
    tasks = []
    for i, spectrum in enumerate(spectra):
        meta = metadata[indices[i]]
        assert meta['role'] == 'surrogate_validation' and meta['group_hash'] == v['groups'][i]
        assert meta['canonical_smiles'] == names[targets[i]]
        top = v['top10'][i]
        assert len(top) == len(set(top)) == 10
        candidates = [names[j] for j in top]
        assert (int(targets[i]) in top) == (v['ranks'][i] <= 10)
        peaks = detector.detect(spectrum)
        initial = (f'## Experimental Raman peaks ({len(peaks)} peaks)\n'+base.format_peak_table(peaks)
            +'\n\n## Ten candidate SMILES (retrieval order)\n'
            +'\n'.join(f'{j+1}. {s}' for j,s in enumerate(candidates))
            +'\n\nSelect exactly one candidate from this list.')
        legacy = stage2.run(Stage1Result(np.zeros(256),peaks,[]))
        positions = [p.position for p in peaks]
        all_hits = retrieve(positions,candidates,False)
        filtered = retrieve(positions,candidates,True)
        base.save(out/'evidence'/f'{i:04d}.json',dict(legacy=asdict(legacy),all_modes=all_hits,filtered_modes=filtered))
        additions = dict(no_kb='',legacy_kb='\n\nUse rules as hypotheses; distinguish IR-only bands from Raman. Cite rule IDs and source keys.\n'+legacy.relevant_rules_summary,
            modes_all_full=format_evidence(all_hits),modes_all_short=format_evidence(all_hits,True),
            modes_filtered_full=format_evidence(filtered),modes_filtered_short=format_evidence(filtered,True))
        for arm in ARMS:
            index = len(tasks)
            request = dict(model='deepseek-v4-flash-ascend',temperature=.2,max_tokens=4096,
                messages=[dict(role='system',content=runner.SYSTEM),dict(role='user',content=initial+'\n\n'+additions[arm])])
            base.save(out/'requests'/f'{index:04d}.json',request)
            tasks.append(dict(request_index=index,query_index=i,method=arm,group_hash=meta['group_hash'],
                candidates=candidates,ground_truth=meta['canonical_smiles'],true_rank=v['ranks'][i]))
    base.save(out/'tasks.json',tasks)
    dependencies = ['scripts/run_raman_kb_ablation.py','scripts/raman_mode_evidence.py',
        'scripts/run_raman_candidate_llm.py','scripts/run_closed_llm_direct.py',
        'scripts/normalize_raman_candidate_choices.py','spectraexpert/eval/metrics.py',
        'spectraexpert/eval/evaluate.py','spectraexpert/pipeline/stage2_peak_analysis.py',
        'spectraexpert/pipeline/knowledge/structured_rule_db.py',
        'knowledge_base/nakamoto_rules_structured.json','nakamoto_empirical_rules_dataset.json']
    base.save(out/'config.json',dict(schema='raman_kb_ablation_v1',arms=ARMS,queries=len(spectra),
        groups=len(groups),candidate_count=len(names),checkpoint_sha256=base.digest(run/'hard/best.pt'),
        validation_usage='development validation previously used for encoder selection; not an untouched test',
        contrasts={'filtering':'modes_filtered_full versus modes_all_full',
                   'compactness':'short versus full within same filter setting; short also caps at eight modes',
                   'source_schema':'modes_all_full versus legacy_kb changes source and mode encoding together'},
        api_sampling='one generation per query/arm; interleaved; temperature 0.2; no significance claim across model seeds',
        source_sha256={f:base.digest(ROOT/f) for f in dependencies},
        retrieval_sha256={f:base.digest(run/f) for f in ('protocol.json','hard/validation.json','prepared.npz')},
        files_sha256={str(f.relative_to(out)):base.digest(f) for folder in ('requests','evidence') for f in sorted((out/folder).glob('*.json'))}
            | {'tasks.json':base.digest(out/'tasks.json')}))
    print(json.dumps(dict(requests=len(tasks),queries=len(spectra),groups=len(groups),arms=ARMS)),flush=True)


def score(out,tasks):
    details=[]
    for t in tasks:
        r=json.loads((out/'responses'/f"{t['request_index']:04d}.json").read_text())
        choice=parse_markdown_choice(r.get('content','')) if not r.get('error') else None
        predicted=t['candidates'][choice] if choice is not None else None
        details.append(dict(t,selected_index=choice+1 if choice is not None else None,
            exact=predicted==t['ground_truth'],valid=choice is not None,
            tanimoto=base.tanimoto_similarity(predicted,t['ground_truth']) if predicted else None))
    summary={}
    baseline={r['query_index']:r for r in details if r['method']=='no_kb'}
    for arm in ARMS:
        rows=[r for r in details if r['method']==arm]
        groups=sorted({r['group_hash'] for r in rows})
        vals=[r['tanimoto'] for r in rows if r['tanimoto'] is not None]
        summary[arm]=dict(n=len(rows),exact=sum(r['exact'] for r in rows),valid=sum(r['valid'] for r in rows),
            group_exact=float(np.mean([np.mean([r['exact'] for r in rows if r['group_hash']==g]) for g in groups])),
            tanimoto_mean=float(np.mean(vals)) if vals else None,tanimoto_n=len(vals),
            gained_vs_no_kb=sum(r['exact'] and not baseline[r['query_index']]['exact'] for r in rows),
            lost_vs_no_kb=sum(not r['exact'] and baseline[r['query_index']]['exact'] for r in rows))
    base.save(out/'selections.json',details);base.save(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','run','score'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--retrieval-run',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.retrieval_run,a.out);return
    config,tasks,requests=runner.load(a.out)
    code=1
    try:
        if a.mode=='run':
            keys=json.loads(sys.stdin.readline())['keys']
            assert 1<=len(keys)<=3 and len(keys)==len(set(keys)) and all(keys)
            asyncio.run(runner.infer(keys,requests,a.out));keys.clear()
            (a.out/'inference_finished_at').write_text(str(time.time()))
        runner.usage_report(a.out,tasks);score(a.out,tasks)
        code=0 if all(not json.loads(f.read_text()).get('error') for f in (a.out/'responses').glob('*.json')) else 2
    finally:(a.out/'exit_code').write_text(str(code))
    raise SystemExit(code)


if __name__=='__main__':main()

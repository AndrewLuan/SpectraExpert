"""功能概述：补齐冻结 Raman Top10 的非 LLM 重排基线，不训练或改变光谱编码器。
主要职责：features 只由检索名次、查询峰和候选 SMARTS 生成特征；fit_ranker 用训练组
候选间 Tanimoto 差拟合带 L2 的线性排序器；predict 稳定排序；summarize/paired 计算
组等权指标与配对区间。develop 校验分组、编码训练查询、选择正则并冻结；test 仅在
冻结后读取既有 Test-200；ged 对选定结构统一补算版本化 GED 并持久化唯一结构对。
代码库关系：复用 raman_mode_evidence 的 15 条规则、原 _PeakDetector、hard checkpoint、
prepared.npz 及已有知识库验证消融。四组严格消融使用 modes_all_full 的同一规则集；
旧 legacy_kb 只作额外参照，不伪装成同规则对照。所有学习标签仅来自训练实验组。
输入输出与副作用：输出目录独占创建，输入和源码 SHA-256 绑定；不调用 API、不读取
原始大数据目录、不注入真实候选。训练 Top10 来自已见训练谱，明确不属于交叉拟合。
开发验证已用于 encoder 选型；Test-200 也已被历次实验观察，不称为新的盲测。
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.raman_mode_evidence import MODES
from scripts.run_closed_llm_direct import digest, save, load_queries, _PeakDetector
from spectraexpert.eval.metrics import tanimoto_similarity, graph_edit_distance

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)
FEATURE_NAMES = ['reciprocal_rank', 'rule_coverage'] + [m[0] for m in MODES]
ARMS = {'learned_rank_only': [0], 'learned_rules_only': list(range(1, 17)),
        'learned_rank_rules': list(range(17))}


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=20000)
def applicability(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('invalid candidate SMILES')
    return np.array([mol.HasSubstructMatch(Chem.MolFromSmarts(m[4])) for m in MODES], dtype=float)


def features(positions, candidates):
    """规则覆盖为命中模式中候选能解释的比例；无命中时为零，不惩罚缺峰。"""
    if len(candidates) != 10 or len(set(candidates)) != 10:
        raise ValueError('expected ten unique candidates')
    observed = np.array([any(m[2]-20 <= p <= m[3]+20 for p in positions) for m in MODES])
    supports = np.stack([applicability(s) for s in candidates]) * observed
    coverage = supports.sum(1) / max(int(observed.sum()), 1)
    return np.column_stack((1 / np.arange(1, 11), coverage, supports))


def check_split(train, validation, excluded):
    t, v, e = set(train), set(validation), set(excluded)
    if not t or not v or t & v or t & e or v & e:
        raise ValueError('empty or overlapping experimental split')


def fit_ranker(x, utility, groups, columns, regularization):
    """最小化每分子组等权、每查询 45 对等权的平方差损失 + lambda*||w||²。"""
    x = x[:, :, columns]
    counts = Counter(groups)
    qweight = np.array([1 / (len(counts)*counts[g]) for g in groups])
    # 仅训练集估计尺度；按分子组等权，防止重复测量获得额外权重。
    mean = np.einsum('q,qkd->d', qweight, x) / 10
    scale = np.sqrt(np.einsum('q,qkd->d', qweight, (x-mean)**2) / 10)
    scale[scale < 1e-8] = 1
    left, right = np.triu_indices(10, 1)
    delta = ((x[:, left]-x[:, right])/scale).reshape(-1, len(columns))
    target = (utility[:, left]-utility[:, right]).ravel()
    weight = np.repeat(qweight/len(left), len(left))
    w = np.linalg.solve(delta.T @ (weight[:, None]*delta) + regularization*np.eye(len(columns)),
                        delta.T @ (weight*target))
    return dict(columns=columns, scale=scale.tolist(), weights=w.tolist(), regularization=regularization)


def predict(x, model):
    scores = (x[:, :, model['columns']]/np.array(model['scale'])) @ np.array(model['weights'])
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite reranker score')
    return np.argsort(-scores, axis=1, kind='stable')[:, 0]


def rows_from_choices(records, choices):
    rows = []
    for r, j in zip(records, choices, strict=True):
        s = r['candidates'][int(j)] if j is not None else None
        rows.append(dict(query_id=r['query_id'], group_hash=r['group_hash'],
                         ground_truth=r['ground_truth'], predicted_smiles=s,
                         selected_index=int(j)+1 if j is not None else None,
                         exact=s == r['ground_truth'], valid=s is not None,
                         tanimoto=tanimoto_similarity(s, r['ground_truth']) if s else None))
    return rows


def summarize(rows):
    result = dict(n=len(rows), groups=len({r['group_hash'] for r in rows}))
    for key in ('exact', 'valid', 'tanimoto', 'legacy_ged_v1', 'chemical_ged_v2'):
        if not all(key in r for r in rows):
            continue
        valid = [r for r in rows if r[key] is not None]
        groups = sorted({r['group_hash'] for r in valid})
        result[key] = dict(n=len(valid), mean=float(np.mean([r[key] for r in valid])) if valid else None,
                          group_mean=float(np.mean([np.mean([r[key] for r in valid if r['group_hash']==g])
                                                   for g in groups])) if valid else None)
    result['exact_count'] = sum(r['exact'] for r in rows)
    return result


def paired(rows, baseline):
    if [(r['query_id'], r['group_hash']) for r in rows] != [(r['query_id'], r['group_hash']) for r in baseline]:
        raise ValueError('unpaired comparison')
    result = {}
    for key in ('exact', 'tanimoto', 'legacy_ged_v1', 'chemical_ged_v2'):
        pairs = [(a, b) for a, b in zip(rows, baseline) if a.get(key) is not None and b.get(key) is not None]
        if not pairs:
            continue
        gs = sorted({a['group_hash'] for a, _ in pairs})
        sign = -1 if 'ged' in key else 1
        diff = np.array([sign*(float(a[key])-float(b[key])) for a, b in pairs])
        gd = np.array([np.mean([d for d, (a, _) in zip(diff, pairs) if a['group_hash']==g]) for g in gs])
        rng = np.random.default_rng(0)
        boot = rng.choice(gd, size=(5000, len(gd)), replace=True).mean(1)
        result[key] = dict(n=len(pairs), groups=len(gs), improvement=float(gd.mean()),
                           ci95=np.quantile(boot, [.025, .975]).tolist(),
                           improved=int((diff>1e-12).sum()), unchanged=int((abs(diff)<=1e-12).sum()),
                           worsened=int((diff < -1e-12).sum()))
    return result


def sources():
    names = ['scripts/raman_nonllm_rerank.py', 'scripts/raman_mode_evidence.py',
             'scripts/raman_domain_experiment.py', 'scripts/run_closed_llm_direct.py',
             'spectraexpert/eval/evaluate.py', 'spectraexpert/eval/metrics.py',
             'spectraexpert/data/structure_features.py']
    return {n: digest(ROOT/n) for n in names}


def make_records(spectra, groups, truth, candidates, detector, ids):
    result = []
    for spectrum, g, gt, cs, qid in zip(spectra, groups, truth, candidates, ids, strict=True):
        peaks = [float(p.position) for p in detector.detect(spectrum)]
        result.append(dict(query_id=qid, group_hash=g, ground_truth=gt, candidates=cs, positions=peaks))
    return result


def matrices(records):
    return np.stack([features(r['positions'], r['candidates']) for r in records])


def all_predictions(records, x, models):
    choices = dict(retrieval=np.zeros(len(records), dtype=int),
                   rule_coverage=np.argsort(-x[:, :, 1], axis=1, kind='stable')[:, 0])
    choices.update({arm: predict(x, model) for arm, model in models.items()})
    return {arm: rows_from_choices(records, js) for arm, js in choices.items()}


def report(out, predictions):
    save(out/'predictions.json', predictions)
    summary = {arm: summarize(rows) for arm, rows in predictions.items()}
    save(out/'summary.json', summary)
    comparisons = {arm: paired(rows, predictions['retrieval']) for arm, rows in predictions.items()}
    if 'llm_no_rules' in predictions and 'llm_modes_all_full' in predictions:
        comparisons['llm_modes_vs_llm_no_rules'] = paired(predictions['llm_modes_all_full'], predictions['llm_no_rules'])
    save(out/'paired.json', comparisons)
    print(json.dumps(summary), flush=True)


def develop(a):
    import torch
    from scripts.raman_domain_experiment import MaskedSpectrumEncoder, encode
    a.out.mkdir(parents=True, exist_ok=False)
    run = a.retrieval_run
    protocol = read(run/'protocol.json')
    frozen = read(run/'final/selection.json')
    for file, sha in protocol['hashes'].items():
        if digest(run/file) != sha:
            raise ValueError(f'prepared input changed: {file}')
    if frozen['selected_arm'] != 'hard' or frozen['protocol_sha256'] != digest(run/'protocol.json'):
        raise ValueError('unexpected encoder selection')
    if digest(run/'hard/best.pt') != frozen['summaries']['hard']['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    check_split(protocol['train_groups'], protocol['validation_groups'], protocol['excluded_test_groups'])
    config = dict(schema='raman_nonllm_v1', retrieval_run=str(run), llm_validation=str(a.llm_validation),
                  source_sha256=sources(), lambda_grid=LAMBDAS, feature_names=FEATURE_NAMES,
                  objective='training group-equal pairwise Morgan Tanimoto differences; no injected positives',
                  selection='validation group exact, then group Tanimoto, then larger lambda',
                  training_caveat='encoder has seen these training spectra; no out-of-fold retrieval',
                  rules='same 15 MODES as modes_all_full; fixed tolerance 20; uniform observed-mode coverage',
                  tie_break='original retrieval order', api_calls=0,
                  input_sha256={str(p):digest(p) for p in [run/'protocol.json',run/'prepared.npz',run/'hard/best.pt',
                      run/'hard/validation.json',a.llm_validation/'tasks.json',a.llm_validation/'selections.json']})
    save(a.out/'protocol.json', config)
    with np.load(run/'prepared.npz', allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    train, val = d['train'], d['validation']
    meta, names = protocol['experimental_metadata'], protocol['candidate_names']
    for indices, role, allowed in [(train,'surrogate_train',set(protocol['train_groups'])),
                                   (val,'surrogate_validation',set(protocol['validation_groups']))]:
        if any(meta[i]['role'] != role or meta[i]['group_hash'] not in allowed or
               meta[i]['canonical_smiles'] != names[d['targets'][i]] for i in indices):
            raise ValueError('metadata alignment failure')
    torch.set_num_threads(4)
    checkpoint = torch.load(run/'hard/best.pt', map_location='cpu', weights_only=False)
    model = MaskedSpectrumEncoder(**checkpoint['model_config']).to(a.device)
    model.load_state_dict(checkpoint['state_dict'])
    reference = encode(model, d['theory'][:len(names)], d['theory_mask'][:len(names)], a.device)
    detector = _PeakDetector({'stage1': {'data_dir': str(Path(protocol['data_root'])/'experimental_test')}})
    datasets = {}
    for role, indices in [('train',train), ('validation',val)]:
        emb = encode(model, d['experiment'][indices], d['experiment_mask'][indices], a.device)
        top = np.argsort(-(emb @ reference.T), axis=1, kind='stable')[:, :10]
        if role == 'validation':
            cached = read(run/'hard/validation.json')
            if indices.tolist() != cached['experiment_indices'] or top.tolist() != cached['top10']:
                raise ValueError('validation candidates differ from frozen LLM protocol')
        records = make_records(d['experiment'][indices], [meta[i]['group_hash'] for i in indices],
                               [names[d['targets'][i]] for i in indices], [[names[j] for j in row] for row in top],
                               detector, [int(i) for i in indices])
        datasets[role] = records
        save(a.out/f'{role}_inputs.json', records)
        print(f'{role}: {len(records)} spectra, {len(set(r["group_hash"] for r in records))} groups', flush=True)
    tr, va = datasets['train'], datasets['validation']
    x, vx = matrices(tr), matrices(va)
    utility = np.array([[tanimoto_similarity(s,r['ground_truth']) for s in r['candidates']] for r in tr])
    models, search = {}, {}
    for arm, columns in ARMS.items():
        trials = []
        for lam in LAMBDAS:
            m = fit_ranker(x, utility, [r['group_hash'] for r in tr], columns, lam)
            stats = summarize(rows_from_choices(va, predict(vx,m)))
            trials.append(dict(model=m, validation=stats))
        chosen = max(trials, key=lambda t:(t['validation']['exact']['group_mean'],
                     t['validation']['tanimoto']['group_mean'],t['model']['regularization']))
        models[arm] = chosen['model']; search[arm] = trials
    save(a.out/'search.json', search)
    save(a.out/'frozen.json', dict(models=models, created_at=time.time(), protocol_sha256=digest(a.out/'protocol.json'),
         search_sha256=digest(a.out/'search.json'), validation_sha256=digest(a.out/'validation_inputs.json'),
         train_sha256=digest(a.out/'train_inputs.json'), training_top10_exact_coverage=float(np.mean([
             r['ground_truth'] in r['candidates'] for r in tr]))))
    predictions = all_predictions(va, vx, models)
    existing = read(a.llm_validation/'selections.json')
    for source, arm in [('no_kb','llm_no_rules'),('modes_all_full','llm_modes_all_full'),('legacy_kb','llm_legacy_rules')]:
        rows = [r for r in existing if r['method']==source]
        if len(rows) != len(va):
            raise ValueError('LLM validation missing rows')
        js = []
        for i, (r, v) in enumerate(zip(rows, va)):
            if r['query_index'] != i or any(r[k] != v[k] for k in ('candidates','ground_truth','group_hash')):
                raise ValueError('LLM inputs not aligned')
            js.append(r['selected_index']-1 if r['selected_index'] is not None else None)
        predictions[arm] = rows_from_choices(va, js)
    directory = a.out/'validation'; directory.mkdir()
    report(directory, predictions)
    print('frozen_sha256='+digest(a.out/'frozen.json'), flush=True)


def test(a):
    frozen = read(a.out/'frozen.json'); config = read(a.out/'protocol.json')
    if config['source_sha256'] != sources() or frozen['protocol_sha256'] != digest(a.out/'protocol.json'):
        raise ValueError('frozen source/protocol changed')
    for file, sha in config['input_sha256'].items():
        if digest(Path(file)) != sha:
            raise ValueError(f'input changed: {file}')
    for file,key in [('train_inputs.json','train_sha256'),('validation_inputs.json','validation_sha256'),('search.json','search_sha256')]:
        if digest(a.out/file) != frozen[key]:
            raise ValueError('frozen development artifact changed')
    directory=a.out/'test'; directory.mkdir(exist_ok=False)
    save(directory/'frozen_binding.json',dict(frozen_sha256=digest(a.out/'frozen.json'),read_started=time.time()))
    run=Path(config['retrieval_run']); protocol=read(run/'protocol.json')
    queries, spectra, _ = load_queries(Path(protocol['data_root']),protocol['sample_manifest_sha256'])
    cached=read(run/'final/per_query.json')
    if cached['queries'] != queries or cached['candidates'] != protocol['candidate_names'][:10000]:
        raise ValueError('test cache mismatch')
    detector=_PeakDetector({'stage1':{'data_dir':str(Path(protocol['data_root'])/'experimental_test')}})
    records=make_records(spectra,[q['group_hash'] for q in queries],
         [Chem.MolToSmiles(Chem.MolFromSmiles(q['ground_truth_smiles'])) for q in queries],
         [[cached['candidates'][j] for j in top] for top in cached['methods']['model']['top10']],
         detector,[q['mol_id'] for q in queries])
    save(directory/'inputs.json',records)
    report(directory,all_predictions(records,matrices(records),frozen['models']))


def ged(a):
    directory=a.out/a.split
    predictions=read(directory/'predictions.json')
    cache=directory/'ged_pairs'; cache.mkdir(exist_ok=True)
    pairs=sorted({(r['ground_truth'],r['predicted_smiles']) for rs in predictions.values() for r in rs if r['valid']})
    def one(pair):
        key=hashlib.sha256(json.dumps(pair).encode()).hexdigest()
        path=cache/f'{key}.json'
        if not path.exists():
            values={v:graph_edit_distance(*pair,timeout=10,version=v) for v in ('legacy_ged_v1','chemical_ged_v2')}
            save(path,dict(pair=pair,**values))
        return pair,read(path)
    values={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(pair,result) in enumerate(pool.map(one,pairs)):
            values[pair]=result
            if (i+1)%20==0: print(f'GED {a.split}: {i+1}/{len(pairs)}',flush=True)
    for rows in predictions.values():
        for r in rows:
            for version in ('legacy_ged_v1','chemical_ged_v2'):
                r[version]=values[(r['ground_truth'],r['predicted_smiles'])][version] if r['valid'] else None
    dest=directory/'with_ged'; dest.mkdir(exist_ok=False)
    report(dest,predictions)
    save(dest/'protocol.json',dict(versions=['legacy_ged_v1','chemical_ged_v2'],timeout=10,workers=4,
         missing='null; pairwise comparisons use common computable query subset',source_sha256=sources(),
         predictions_sha256=digest(directory/'predictions.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('develop','test','ged'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--retrieval-run',type=Path)
    p.add_argument('--llm-validation',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--split',choices=('validation','test'),default='validation')
    a=p.parse_args()
    {'develop':develop,'test':test,'ged':ged}[a.mode](a)


if __name__=='__main__':
    main()

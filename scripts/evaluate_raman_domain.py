"""功能概述：按独立实验验证结果冻结 A/B/C 选择，再进行固定 10k/200 最终跨域检索评分。
主要职责：paired_bootstrap 给出与 raw LLM 的配对差值区间；validation_diagnostics 比较
原始谱、覆盖归一化谱、旧模型两种检索路径和新模型；finalize 先持久化选择，再加载测试谱，
重建所选模型参考谱索引，复用原结构评分器并汇总 raw kNN、模型和 LLM 的同样本比较。
代码库关系：消费 raman_domain_experiment 的独立 checkpoint/schema，不将旧图模型当作新版训练结果。
输入输出与副作用：无 API 请求；只写新的 final 目录。测试分数不影响模型或候选选择，
理论自查询仅是参考谱自检，不被解释为未见理论数据上的泛化精度。GED 超时保留 null。
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.raman_domain_experiment import (MaskedSpectrumEncoder,MODEL_CONFIG,load_prepared,encode,
                                           rank_metrics,unit,coverage_mask)
from scripts.evaluate_closed_raman import digest
from scripts.run_closed_llm_direct import load_queries,evaluate,save
from spectraexpert.training.checkpoints import load_checkpoint
from spectraexpert.data.molecular_graph import featurize_smiles,collate_graphs
from spectraexpert.data.structure_features import StructureFeatureConfig
from spectraexpert.data.structure_dataset import StructureAwareSpectralDataset


def paired_bootstrap(a,b,larger_better=True,seed=0,groups=None):
    a=np.array([np.nan if x is None else x for x in a],dtype=float)
    b=np.array([np.nan if x is None else x for x in b],dtype=float)
    mask=np.isfinite(a)&np.isfinite(b)
    d=(a[mask]-b[mask])*(1 if larger_better else -1)
    if not len(d):return dict(n=0,improvement=None,ci95=None)
    rng=np.random.default_rng(seed)
    if groups is None:
        boot=np.mean(d[rng.integers(0,len(d),(10000,len(d)))],axis=1)
    else:
        kept=np.asarray(groups)[mask];unique=np.unique(kept)
        sums=np.array([d[kept==g].sum() for g in unique]);counts=np.array([(kept==g).sum() for g in unique])
        sample=rng.integers(0,len(unique),(10000,len(unique)))
        boot=sums[sample].sum(1)/counts[sample].sum(1)
    return dict(n=len(d),model_mean=float(a[mask].mean()),llm_mean=float(b[mask].mean()),
                improvement=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())


def old_encode(model,x,device):
    results=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            batch=np.stack([StructureAwareSpectralDataset._normalize_spectrum(s) for s in x[start:start+128]])
            results.append(model.encode_spectrum(torch.tensor(batch,device=device)).cpu().numpy())
    return np.concatenate(results)


def validation_diagnostics(protocol,data,out,legacy_checkpoint=None):
    t,tm,e,em=data['theory'],data['theory_mask'],data['experiment'],data['experiment_mask']
    val=data['validation'];target=data['targets'][val];names=protocol['candidate_names'];report={}
    raw=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    for name,reference,query in [('raw_knn',raw(t),raw(e[val])),
        ('masked_zscore_knn',np.stack([unit(x,m) for x,m in zip(t,tm)]),np.stack([unit(x,m) for x,m in zip(e[val],em[val])]))]:
        report[name],_=rank_metrics(query,reference,target,names)
    for arm in ('A','B','C'):
        payload=torch.load(out/arm/'best.pt',map_location='cpu',weights_only=False)
        model=MaskedSpectrumEncoder(**MODEL_CONFIG).cuda();model.load_state_dict(payload['state_dict'])
        reference=encode(model,t,tm,'cuda');query=encode(model,e[val],em[val],'cuda')
        metrics,predicted=rank_metrics(query,reference,target,names)
        self_metrics,_=rank_metrics(reference[target],reference,target,names)
        report[arm]=dict(experimental=metrics,theory_reference_self_lookup=self_metrics,
                         self_minus_experiment_gap=self_metrics['exact_match']-metrics['exact_match'],
                         exact_per_query=(predicted==target).tolist())
        del model
    if legacy_checkpoint:
        model,payload=load_checkpoint(legacy_checkpoint,expected_mode='spectrum_graph_v2',device='cuda');model.eval()
        query=old_encode(model,e[val],'cuda');reference=old_encode(model,t,'cuda')
        report['original_frozen_spectrum_to_spectrum'],_=rank_metrics(query,reference,target,names)
        features=StructureFeatureConfig(**payload['feature_config']);graphs=[]
        with torch.inference_mode():
            for start in range(0,len(names),128):
                batch=collate_graphs([featurize_smiles(s,start+i,features) for i,s in enumerate(names[start:start+128])]).to('cuda')
                graphs.append(model.encode_graph(batch).cpu().numpy())
        report['original_frozen_spectrum_to_graph'],_=rank_metrics(query,np.concatenate(graphs),target,names)
        report['original_checkpoint_sha256']=digest(legacy_checkpoint)
        report['original_checkpoint_scope']='diagnostic only; original pre-repair training data'
        del model
    validation_groups=[protocol['experimental_metadata'][i]['group_hash'] for i in val]
    report['B_minus_A_validation_exact']=paired_bootstrap(report['B']['exact_per_query'],report['A']['exact_per_query'],groups=validation_groups)
    report['C_minus_B_validation_exact']=paired_bootstrap(report['C']['exact_per_query'],report['B']['exact_per_query'],groups=validation_groups)
    report['theory_self_lookup_caveat']='identical reference self-query, not unseen-theory generalization'
    save(out/'validation_diagnostics.json',report)
    return report


def finalize(out,baseline,legacy_checkpoint=None):
    torch.set_num_threads(4)
    protocol,data,params=load_prepared(out)
    summaries={a:json.loads((out/a/'summary.json').read_text()) for a in ('A','B','C')}
    if len({r['initial_sha256'] for r in summaries.values()})!=1:raise ValueError('arms did not share initialization')
    for a,r in summaries.items():
        if r['checkpoint_sha256']!=digest(out/a/'best.pt'):raise ValueError('checkpoint changed')
    selected=max(('A','B','C'),key=lambda a:tuple(summaries[a]['best_key']))
    final=out/'final';final.mkdir(exist_ok=False)
    save(final/'selection.json',dict(selected_arm=selected,selection_uses='experimental validation only',
        arm_summaries=summaries,protocol_sha256=digest(out/'protocol.json'),script_sha256=digest(Path(__file__)),
        baseline_results_sha256=digest(baseline/'results.json'),test_candidates=10000))
    validation_diagnostics(protocol,data,out,legacy_checkpoint)
    # 选择已冻结，到此才读取实验测试谱。
    root=Path(protocol['data_root'])
    queries,spectra,groups=load_queries(root,protocol['sample_manifest_sha256'])
    test_rows=json.loads((root/'experimental_test/rows.json').read_text())
    test_masks=np.stack([coverage_mask(r['coverage']) for r in test_rows])
    truth={s:i for i,s in enumerate(protocol['candidate_names'][:10000])}
    from scripts.repair_closed_raman_exact import canonical
    targets=np.array([truth[canonical(q['ground_truth_smiles'])] for q in queries])
    checkpoint=out/selected/'best.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['sample_manifest_sha256']!=protocol['sample_manifest_sha256'] or payload['prepared_sha256']!=protocol['prepared_sha256']:
        raise ValueError('selected model data mismatch')
    model=MaskedSpectrumEncoder(**MODEL_CONFIG).cuda();model.load_state_dict(payload['state_dict'])
    reference=encode(model,data['theory'][:10000],data['theory_mask'][:10000],'cuda')
    query=encode(model,spectra,test_masks,'cuda')
    names=protocol['candidate_names'][:10000]
    model_metrics,model_prediction=rank_metrics(query,reference,targets,names)
    np.savez_compressed(final/'candidate_spectrum_index.npz',embeddings=reference,
        mol_ids=np.array([r['mol_id'] for r in protocol['candidate_metadata'][:10000]]),canonical_smiles=np.array(names),
        checkpoint_sha256=np.array(digest(checkpoint)),sample_manifest_sha256=np.array(protocol['sample_manifest_sha256']))
    raw=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    knn_metrics,knn_prediction=rank_metrics(raw(spectra),raw(data['theory'][:10000]),targets,names)
    save(final/'retrieval_summary.json',dict(selected_arm=selected,model=model_metrics,raw_knn=knn_metrics,
        candidate_count=10000,query_count=200))
    predictions=dict(model=model_prediction,raw_knn=knn_prediction)
    for method,prediction in predictions.items():
        folder=final/method;(folder/'metrics').mkdir(parents=True)
        save(folder/'predictions.json',dict(query_ids=[q['mol_id'] for q in queries],
             candidate_indices=prediction.tolist(),smiles=[names[i] for i in prediction]))
        responses=[dict(content='PREDICTED_SMILES: '+names[i]) for i in prediction]
        evaluate(queries,responses,groups,folder)
    llm=json.loads((baseline/'results.json').read_text())
    if json.loads((baseline/'config.json').read_text())['manifest_sha256']!=protocol['sample_manifest_sha256']:
        raise ValueError('baseline dataset version mismatch')
    for row,q in zip(llm['queries'],queries):
        if any(row[k]!=q[k] for k in q):raise ValueError('baseline query mismatch')
    results={m:json.loads((final/m/'results.json').read_text()) for m in predictions}
    comparisons={}
    for method,result in results.items():
        comparisons[method]={}
        for metric in ('valid','exact_match','tanimoto','legacy_ged_v1','chemical_ged_v2'):
            comparisons[method][metric]=paired_bootstrap([r['strict'][metric] for r in result['queries']],
                [r['strict'][metric] for r in llm['queries']],larger_better='ged' not in metric)
    common={}
    for metric in ('tanimoto','legacy_ged_v1','chemical_ged_v2'):
        triples=[[results[m]['queries'][i]['strict'][metric] for m in ('model','raw_knn')]+[llm['queries'][i]['strict'][metric]] for i in range(200)]
        values=np.array([v for v in triples if all(x is not None for x in v)],dtype=float)
        means=dict(zip(('model','raw_knn','llm'),values.mean(0).tolist())) if len(values) else {}
        common[metric]=dict(n=len(values),means=means)
    save(final/'comparison.json',dict(selected_arm=selected,llm_summary=llm['summary'],common_all_methods=common,
        method_summaries={k:v['summary'] for k,v in results.items()},paired_comparisons=comparisons,
        model_all_metrics_point_estimate_better=all(v['improvement'] is not None and v['improvement']>0 for v in comparisons['model'].values()),
        model_all_ci_lower_bounds_positive=all(v['ci95'] is not None and v['ci95'][0]>0 for v in comparisons['model'].values()),
        selection_frozen_before_test=True,api_calls=0))
    save(final/'status.json',dict(state='complete',selected_arm=selected,query_count=200))
    print(json.dumps(dict(completed=True,selected_arm=selected,model=results['model']['summary'],raw_knn=results['raw_knn']['summary'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--legacy-checkpoint',type=Path)
    args=parser.parse_args();finalize(args.out,args.baseline,args.legacy_checkpoint)

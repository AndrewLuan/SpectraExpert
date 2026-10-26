"""功能概述：在固定库/固定查询上复评联合条件模型，并重拟合训练局部误差宽度。

main的val阶段生成同次发射bank，以已审核train原子标签拟合9类宽度；只在固定320-val
选择0/0.5/1插值，保存每个固定插值结果，随后全局研究流程才能冻结模型并开放test。
rank只在原稀疏归属无显式频窗路由替换密度；积分谱和HMDB仍沿用原评分配置。
HMDB报告臂使用模型共享报告头及h_formyl_workflow的结构未报告核，做整谱两状态
边缘化；完整物理bank不删峰、不设DMF位移或分子特判。显式频窗精确回退原路径。
每臂同时保存width=0与val选择宽度的1000条排名和371条压力变体，防止宽度选择
掩盖位移×报告的原始消融。候选顺序、物理输入、查询、检查点均写SHA并核对。
test另运行原子/DMF诊断；这些输出不参与任何选择。所有目录独占，无网络服务调用。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .emission_model import make_model
from .evaluate import missing_queries, summarize
from .h_condinfer import predict, query_kernels, condition_key, scoring_inputs, diagnostics
from .h_conditions import CONDITION_DIM
from .h_eval import LazyFP
from .h_formyl_reporting import candidate_families, mixture_score
from .h_formyl_workflow import reporting_kernel, selected_hmdb_config
from .h_labels import make_key
from .h_latent_score import eligible
from .h_metadata import metadata_view
from .h_observation_finalize import query_score, visible_lines
from .h_observation_study import entry, write_rows, summarize_sources, stamp_queries, SHIFTS
from .h_observed import line_score, cluster_lines
from .h_reviewed_supervision import load_artifact, apply_review
from .h_score import attach_observation
from .h_uncertainty import class_density
from .h_uncertainty_workflow import fit_widths
from .prepare import sha


@torch.inference_mode()
def rank(bank, queries, metadata, labels, model, widths, inputs, device, out, status, stress=False):
    indices = bank['single_indices']
    smiles = [bank['smiles'][i] for i in indices]
    families = candidate_families(smiles)
    lookup = {s:i for i,s in enumerate(smiles)}
    counts = bank['physics'][indices,0].to(device)*40
    fps = LazyFP()
    frozen, prior, config, priors, raw = inputs
    lambdas = [0., .5, 1.] if not stress else [0., float(json.loads((out.parent/'val/frozen_selection.json').read_text())['width_lambda'])]
    lambdas = list(dict.fromkeys(lambdas))
    records = {str(lam):[] for lam in lambdas}
    missing = {str(lam):[] for lam in lambdas}
    for key in sorted({condition_key(r, metadata, bank['condition_key_mode']) for r in queries}):
        group = [(i,r) for i,r in enumerate(queries) if condition_key(r,metadata,bank['condition_key_mode']) == key]
        kernels = query_kernels(bank, indices, key, device, any(r['source']=='HMDB' for _,r in group))
        tags = bank['candidate_quantile_classes'][key][indices].to(device)
        variable = {str(lam):class_density(kernels['pred'], tags, (.2**(1-lam)*np.asarray(widths)**lam).tolist())
                    for lam in lambdas if lam > 0}
        formyl = None
        if model.reporting_enabled and any(r['source']=='HMDB' for _,r in group):
            cfg = selected_hmdb_config(key, inputs)
            if cfg is not None:
                fixed = bank['fixed_predictions'][key][indices].to(device) if key[1]>0 else kernels['pred']
                formyl, active = reporting_kernel(kernels['pred'], fixed, tags, families, cfg)
                cond = torch.tensor(key[4:], device=device).reshape(1,CONDITION_DIM).expand(len(smiles),-1)
                probabilities = {False:model.reporting_head(families,cond), True:model.reporting_head(families,cond,True)}

        def scores(query):
            full = query_score(query, kernels, counts, prior, config, frozen, priors, raw)
            if query['source']=='HMDB' and formyl is not None and not query['observation']['windows']:
                cfg = selected_hmdb_config(key, inputs)
                xs = visible_lines(query, raw)
                k = max(1,len(cluster_lines(xs,cfg['width'])[0]))
                absent = line_score(formyl,query,xs,cfg['width'],SHIFTS,cfg['shift_sigma'])
                absent = torch.where(active,absent,full)
                full = mixture_score(full,absent,probabilities[bool(query['observation'].get('partial_peaks'))],k)
            values = {str(0.):full}
            for lam in lambdas:
                if lam > 0:
                    values[str(lam)] = query_score(query,dict(kernels,mass=variable[str(lam)]),counts,prior,config,frozen,priors,raw) if eligible(query) else full
            return values

        for order,row in group:
            query = attach_observation(row, labels.get(make_key(row),{}))
            values = scores(query)
            for lam in lambdas:
                records[str(lam)].append(dict(query_order=order,**entry(values[str(lam)],query,lookup,smiles,fps)))
            if stress and order < 200:
                for variant,altered in missing_queries(query,'1H'):
                    values = scores(altered)
                    for lam in lambdas:
                        missing[str(lam)].append(dict(variant=variant,**entry(values[str(lam)],altered,lookup,smiles,fps)))
        status('ranking', ranked=len(records['0.0']), total=len(queries))
    reports = {}
    for lam,rs in records.items():
        rs.sort(key=lambda r:r['query_order'])
        folder = out/('width_'+lam)
        folder.mkdir()
        write_rows(folder/'predictions.jsonl',rs)
        report = dict(metrics=summarize_sources(rs), candidates=len(smiles), query_sha256=stamp_queries(queries),
                      candidate_order_sha256=hashlib.sha256('\n'.join(smiles).encode()).hexdigest(),
                      physics_sha256=hashlib.sha256(bank['physics'][indices].contiguous().cpu().numpy().tobytes()).hexdigest(),
                      checkpoint_sha256=bank['checkpoint_sha256'], width_lambda=float(lam))
        if stress:
            if len(rs)!=1000 or len(missing[lam])!=371:
                raise ValueError('joint_retrieval_stress_cohort_drift')
            write_rows(folder/'missing_predictions.jsonl',missing[lam])
            report['missing_stress'] = {name:summarize([r for r in missing[lam] if r['variant']==name])
                                        for name in ('frequency_crop','peak_dropout')}
        (folder/'summary.json').write_text(json.dumps(report,indent=2))
        reports[lam] = report
    return reports


def main():
    parser = argparse.ArgumentParser()
    for name in ('base','prepared','training','review','out','mode'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--stage',choices=['val','test'],required=True)
    args = parser.parse_args()
    base, prepared, root = Path(args.base), Path(args.prepared), Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    out = root/args.stage
    out.mkdir(exist_ok=False)
    def status(stage,**fields):
        (out/'status.json').write_text(json.dumps(dict(stage=stage,**fields)))
    try:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('high')
        device = torch.device('cuda')
        checkpoint = Path(args.training)/args.mode/'train/best.pt'
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        model = make_model(saved['args']).to(device).eval()
        model.load_state_dict(saved['state_dict'])
        model.reporting_enabled = saved['args']['reporting_enabled']
        key_mode = saved['args']['condition_key_mode']
        metadata = metadata_view(torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False),key_mode)
        labels = torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        if args.stage=='val':
            status('loading_cache')
            cache = torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True)
            bank = torch.load(base/'hydrogen_analysis/evaluation_dropout/site/bank.pt',map_location='cpu',weights_only=False,mmap=True)
            predictions,fixed,tags = predict(model,bank,cache['graphs'],metadata,key_mode,device,status)
            bank.update(predictions=predictions,fixed_predictions=fixed,candidate_quantile_classes=tags,
                        checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),condition_key_mode=key_mode)
            torch.save(bank,root/'bank.pt')
            train = [r for r in cache['rows'] if r['domain']=='experimental' and r['split']=='train']
            artifact,_ = load_artifact(args.review)
            _,audit = apply_review(labels,train,artifact,'domain')
            status('fitting_train_widths')
            widths = fit_widths(model,cache,metadata,labels,device,out,'methyl_record_mask')
            (out/'review_width_audit.json').write_text(json.dumps(audit,indent=2))
            del cache
        else:
            if not (root.parent/'frozen_selection.json').exists():
                raise ValueError('joint_global_selection_not_frozen')
            bank = torch.load(root/'bank.pt',map_location='cpu',weights_only=False,mmap=True)
            if bank['checkpoint_sha256'] != sha(checkpoint):
                raise ValueError('joint_checkpoint_changed_after_selection')
            widths = json.loads((root/'val/width_fit.json').read_text())['selected_ppm']
        reports = rank(bank,bank[args.stage],metadata,labels,model,widths,scoring_inputs(base),device,out,status,args.stage=='test')
        if args.stage=='val':
            chosen = max(reports,key=lambda x:reports[x]['metrics']['overall']['MRR'])
            (out/'frozen_selection.json').write_text(json.dumps(dict(width_lambda=float(chosen),validation=reports,
                selection='same320 val; widths fit reviewed experimental train only; report strength fixed1 for report arms'),indent=2))
        else:
            status('diagnostics')
            cache = torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True)
            diagnostics(model,bank,cache['graphs'],metadata,labels,out,device)
        (out/'summary.json').write_text(json.dumps(reports,indent=2))
        status('complete')
        (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc))
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__=='__main__':
    main()

"""功能概述：固定均值的整体 ¹H 检索观测消融，优先量化共性评分失配。

configs 在看新验证结果前声明原子报告数、公共位移、未知溶剂、积分报告数和 HMDB
分辨率边缘化干预；每项作用于全部同口径查询，禁止分子专用分支。evaluate 使用
完整 114982 候选库，baseline 精确复用 h_joint_evaluate 的宽度 0.5 路由；计数
先验只从既有训练 annotations 拟合，候选的核数只取结构。原子表计数表示标签报告
过程，不把归属稀疏解释为测量缺峰。未知溶剂对已缓存且明确训练支持的条件积分。
main 的 train/original_val/tune 与 verify/test 分入口；verify/test 要求外部冻结
选择先存在。追加条件需先由独立 bank 补预测入口生成，不能偷偷改写查询条件。
输出各干预完整排名、Top1/5/10、MRR、查询和候选 SHA；目录独占且无外部服务。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import missing_queries
from .h_condinfer import condition_key, query_kernels, scoring_inputs
from .h_conditions import SOLVENT_DIM
from .h_eval import LazyFP
from .h_labels import make_key
from .h_latent_score import eligible, fit_reporting_priors, features as latent_features, score as latent_score, prototype_keys
from .h_observation_finalize import query_score, visible_lines
from .h_observation_study import entry, write_rows, summarize_sources, stamp_queries, SHIFTS
from .h_observed import line_score, marginal_shift
from .h_score import attach_observation, score_features
from .h_uncertainty import class_density
from .prepare import sha
from .h_global_bank import load_bank
from .h_global_protocol import actual_inputs


def configs():
    result = [dict(name='baseline', kind='baseline')]
    for beta in (.25, 1., 4.):
        result.append(dict(name='atom_count_'+str(beta), kind='atom', condition='given', shift_sigma=0., beta=beta))
    for sigma in (.1, .2):
        for beta in (0., 1., 4.):
            result.append(dict(name=f'atom_shift{sigma}_count{beta}', kind='atom', condition='given', shift_sigma=sigma, beta=beta))
    for beta in (0., 1., 4.):
        result.append(dict(name='atom_latent_count'+str(beta), kind='atom', condition='train_mixture', shift_sigma=.1, beta=beta))
    for beta in (0., 1., 2., 8.):
        result.append(dict(name='integral_count_'+str(beta), kind='integral_count', beta=beta))
    result.append(dict(name='line_resolution_average', kind='line_resolution'))
    for epsilon in (.01, .03, .1):
        result.append(dict(name='outlier_'+str(epsilon),kind='outlier',epsilon=epsilon))
    return result


def contaminated(kernel, epsilon):
    """对固定全范围添加统一污染密度，CDF 同步，保留窗口条件化契约。"""
    result = dict(kernel)
    if epsilon==0:return result
    points=kernel['logp'].shape[1]
    background = epsilon/(kernel['step']*points)
    result['logp'] = torch.logaddexp(kernel['logp']+np.log1p(-epsilon),torch.full_like(kernel['logp'],np.log(background)))
    cdf = torch.arange(points+1,device=kernel['cdf'].device)[None,:]/points
    result['cdf'] = (1-epsilon)*kernel['cdf']+epsilon*cdf
    return result


def latent_feature_inputs(query, given, prototypes, include_latent):
    """仅构建本次干预所需潜在条件；given-only 不假称条件缺失已恢复。"""
    adapted=dict(query)
    if not include_latent:adapted['_condition_known']=True
    return adapted,dict(prototypes,given=given)


def frozen_choices(selection, required_inputs):
    """核对冻结的配置及每个实际评分输入，防止核验时换 bank 或宽度。"""
    frozen=json.loads(Path(selection).read_text())
    if frozen.get('selection_scope')!='train_val_only' or not frozen.get('frozen_utc'):
        raise ValueError('global_invalid_frozen_selection')
    pinned=frozen['evaluation_inputs_sha256']
    for path in required_inputs:
        path=Path(path).resolve()
        if pinned.get(str(path))!=sha(path):raise ValueError('global_frozen_input_changed:'+str(path))
    selected=frozen['scoring_configs']
    registered={c['name']:c for c in configs()}
    if not selected or any(c!=registered.get(c.get('name')) for c in selected):
        raise ValueError('global_unregistered_frozen_scoring_config')
    if not any(c['kind']=='baseline' for c in selected):raise ValueError('global_frozen_baseline_missing')
    return selected


@torch.inference_mode()
def evaluate(bank, queries, payload, widths, inputs, out, status, selected=None):
    device = torch.device('cuda')
    indices = bank['single_indices']
    smiles = [bank['smiles'][i] for i in indices]
    lookup = {s:i for i,s in enumerate(smiles)}
    counts = bank['physics'][indices,0].to(device)*40
    metadata, labels = payload['metadata'], payload['annotations']
    frozen, prior, config, priors, raw = inputs
    reporting = fit_reporting_priors(labels, metadata)
    choices = configs() if selected is None else selected
    records = {c['name']:[] for c in choices}
    fps = LazyFP()
    prototype = prototype_keys(bank)
    available = all(k in bank['predictions'] for k in prototype.values())
    need_atom=any(c['kind']=='atom' for c in choices)
    need_latent=any(c.get('condition')=='train_mixture' for c in choices)
    if need_latent and not available:raise ValueError('global_missing_training_prototype_conditions')
    prototype_kernels = {}
    if need_latent:
        for name, key in prototype.items():
            tags = bank['candidate_quantile_classes'][key][indices].to(device)
            p = bank['predictions'][key][indices].to(device)
            prototype_kernels[name] = class_density(p, tags, (.2**.5*np.asarray(widths)**.5).tolist())
    for key in sorted({condition_key(r,metadata,payload['key_mode']) for r in queries}):
        if key not in bank['predictions']:
            raise ValueError('global_bank_missing_condition:'+str(key))
        group = [(i,r) for i,r in enumerate(queries) if condition_key(r,metadata,payload['key_mode'])==key]
        kernels = query_kernels(bank,indices,key,device,any(r['source']=='HMDB' for _,r in group))
        tags = bank['candidate_quantile_classes'][key][indices].to(device)
        variable = class_density(kernels['pred'], tags, (.2**.5*np.asarray(widths)**.5).tolist())
        base_kernels=use=latent_kernels=kernel=None
        robust = {c['epsilon']:{name:contaminated(k,c['epsilon']) for name,k in dict(kernels,mass=variable).items() if name!='pred'}
                  for c in choices if c['kind']=='outlier'}
        for order, row in group:
            query = attach_observation(row,labels.get(make_key(row),{}))
            query['_condition_known'] = bool(key[4+SOLVENT_DIM-1]>0)
            base_kernels = dict(kernels, mass=variable) if eligible(query) else kernels
            baseline = query_score(query,base_kernels,counts,prior,config,frozen,priors,raw)
            atomic = None
            if eligible(query) and need_atom:
                latent_query,latent_kernels=latent_feature_inputs(query,variable,prototype_kernels,need_latent)
                atomic = latent_features(latent_query,latent_kernels,counts,prior,reporting)
            integral = None
            if query.get('measure')=='integral_weighted' and not query['observation']['windows'] and not query['observation'].get('partial_peaks') and 'observed_xs' in query:
                integral = score_features(kernels['pred'],kernels['mass'],counts,query,prior)
            resolution = None
            if row['source']=='HMDB' and not query['observation']['windows']:
                cfg = frozen['water_HMDB'] if key[1]>0 and frozen['water_HMDB'] is not None else frozen['HMDB']
                if cfg:
                    from .h_observation_finalize import blend
                    kernel = kernels[cfg['kernel']]
                    if 'visibility' in cfg:
                        kernel = blend(kernel,kernels['fixed_'+cfg['kernel']],cfg['visibility'])
                    xs = visible_lines(query,raw)
                    trial = torch.stack([line_score(kernel,query,xs,w,SHIFTS,cfg['shift_sigma']) for w in (.04,.08,.16)])
                    resolution = trial.mean(0)
            for choice in choices:
                kind = choice['kind']; values = baseline
                if kind=='atom' and atomic is not None:
                    values = latent_score(atomic,choice)
                elif kind=='integral_count' and integral is not None and integral['eligible']:
                    values = integral['transport']+choice['beta']*integral['count']
                elif kind=='line_resolution' and resolution is not None:
                    values = resolution
                elif kind=='outlier':
                    use = dict(kernels)
                    for name,kernel in robust[choice['epsilon']].items():
                        if name!='mass' or eligible(query):use[name]=kernel
                    if not eligible(query):use['mass']=contaminated(kernels['mass'],choice['epsilon'])
                    values=query_score(query,use,counts,prior,config,frozen,priors,raw)
                records[choice['name']].append(dict(query_order=order, **entry(values,query,lookup,smiles,fps)))
            if sum(len(v) for v in records.values())//len(choices)%25==0:
                status('ranking', ranked=len(records[choices[0]['name']]), total=len(queries))
        base_kernels=use=latent_kernels=kernel=None
        del kernels, variable, robust
        torch.cuda.empty_cache()
    report = dict(configs=choices, metrics={}, candidate_count=len(smiles),
        candidate_order_sha256=hashlib.sha256('\n'.join(smiles).encode()).hexdigest(),
        query_sha256=stamp_queries(queries), checkpoint_sha256=bank['checkpoint_sha256'],
        latent_prototypes_available=available, external_api_calls=0)
    for name, rows in records.items():
        rows.sort(key=lambda r:r['query_order'])
        folder = out/name; folder.mkdir()
        write_rows(folder/'predictions.jsonl', rows)
        report['metrics'][name] = summarize_sources(rows)
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ('base','cohorts','bank','widths','out','cohort'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--frozen-selection')
    parser.add_argument('--extension')
    parser.add_argument('--limit',type=int,default=0)
    args = parser.parse_args()
    if args.cohort not in ('train','original_val','tune','verify','test'):
        raise ValueError('unknown_cohort')
    if args.cohort in ('verify','test') and not (args.frozen_selection and Path(args.frozen_selection).exists()):
        raise ValueError('global_selection_must_be_frozen')
    out = Path(args.out); out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):
        (out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
        payload = torch.load(Path(args.cohorts)/'cohorts.pt',map_location='cpu',weights_only=False)
        bank = load_bank(args.bank,args.extension)
        queries = bank['test'] if args.cohort=='test' else payload['cohorts'][args.cohort]
        if args.limit:
            if args.cohort in ('verify','test'):raise ValueError('cannot_limit_locked_evaluation')
            queries=queries[:args.limit]
        widths = json.loads(Path(args.widths).read_text())['selected_ppm']
        (out/'protocol.json').write_text(json.dumps(dict(configs=configs(),cohort=args.cohort,
            candidate_bank=args.bank,bank_sha256=sha(args.bank),cohort_manifest_sha256=sha(Path(args.cohorts)/'manifest.json'),
            selection_scope='train/val only; fixed candidates/test',external_api_calls=0),indent=2))
        selected = None
        if args.cohort in ('verify','test'):
            required=actual_inputs(args.base,args.bank,args.widths,args.cohorts,args.extension)
            selected=frozen_choices(args.frozen_selection,required)
        evaluate(bank,queries,payload,widths,scoring_inputs(Path(args.base)),out,status,selected)
        reference=Path(args.bank).parent/('val' if args.cohort=='original_val' else 'test')/'width_0.5/predictions.jsonl'
        if args.cohort in ('original_val','test') and reference.exists():
            old={(r['source'],r['record_id'],r['smiles']):r for r in map(json.loads,reference.read_text().splitlines())}
            new=[json.loads(l) for l in (out/'baseline/predictions.jsonl').read_text().splitlines()]
            exact=all(r['rank']==old[(r['source'],r['record_id'],r['smiles'])]['rank'] and
                      r['top10']==old[(r['source'],r['record_id'],r['smiles'])]['top10'] for r in new)
            (out/'baseline_control.json').write_text(json.dumps(dict(queries=len(new),ranks_and_top10_exact=exact,reference=str(reference))))
            if not exact:raise ValueError('global_baseline_ranking_drift')
        status('complete'); (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc)); (out/'EXIT_CODE').write_text('1\n'); raise


if __name__=='__main__':
    main()

"""功能概述：在固定完整候选库检索后缓存结构无关的难负样本特征。

Scorer 复用冻结均值/宽度和训练报告先验，支持原基线及 h_global_scoring 预声明干预。
extract 对全部 114982 候选评分，再按评分产生 K 短名单；训练只在真结构不在名单时
额外附加正例以构造训练对，验证/核验/测试绝不补入正确候选。原完整库排名和短名单
召回同时保存，候选库没有缩减。candidate_features 不读取真结构身份；身份只在
feature 生成后用于训练标签及指标。重排仅改短名单内部顺序，名单外排名保持原值。
main 严格门禁 verify/test，需要冻结评分输入及配置。原谱、旧 bank、原条件不改写；
每份特征缓存记录 schema、输入 SHA、查询 SHA 与退出码，供低容量 ranker 读取。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import missing_queries
from .data import identity
from .h_condinfer import condition_key, query_kernels, scoring_inputs
from .h_conditions import SOLVENT_DIM
from .h_global_bank import load_bank
from .h_global_features import FEATURE_NAMES, candidate_features
from .h_global_scoring import configs, contaminated, frozen_choices, latent_feature_inputs
from .h_labels import make_key
from .h_latent_score import eligible, fit_reporting_priors, features as latent_features, score as latent_score, prototype_keys
from .h_observation_finalize import query_score
from .h_observation_study import stamp_queries
from .h_score import attach_observation, score_features
from .h_uncertainty import class_density
from .prepare import sha
from .h_global_protocol import actual_inputs


class Scorer:
    def __init__(self,bank,payload,widths,inputs,choice,device):
        self.bank=bank; self.payload=payload; self.widths=widths; self.inputs=inputs
        self.choice=choice; self.device=device; self.indices=bank['single_indices']
        self.counts=bank['physics'][self.indices,0].to(device)*40
        self.reporting=fit_reporting_priors(payload['annotations'],payload['metadata'])
        self.prototypes={}
        if choice.get('condition')=='train_mixture':
            for name,key in prototype_keys(bank).items():
                p=bank['predictions'][key][self.indices].to(device)
                tags=bank['candidate_quantile_classes'][key][self.indices].to(device)
                self.prototypes[name]=class_density(p,tags,(.2**.5*np.asarray(widths)**.5).tolist())

    def group(self,key,hmdb):
        self.kernels=query_kernels(self.bank,self.indices,key,self.device,hmdb)
        self.tags=self.bank['candidate_quantile_classes'][key][self.indices].to(self.device)
        self.variable=class_density(self.kernels['pred'],self.tags,(.2**.5*np.asarray(self.widths)**.5).tolist())
        self.key=key

    def score(self,query):
        frozen,prior,config,priors,raw=self.inputs; kind=self.choice['kind']
        kernels=dict(self.kernels,mass=self.variable) if eligible(query) else self.kernels
        baseline=query_score(query,kernels,self.counts,prior,config,frozen,priors,raw)
        result=baseline
        if kind=='atom' and eligible(query):
            adapted,ks=latent_feature_inputs(query,self.variable,self.prototypes,self.choice.get('condition')=='train_mixture')
            result=latent_score(latent_features(adapted,ks,self.counts,prior,self.reporting),self.choice)
        elif kind=='integral_count' and query.get('measure')=='integral_weighted' and not query['observation']['windows'] and not query['observation'].get('partial_peaks') and 'observed_xs' in query:
            features=score_features(self.kernels['pred'],self.kernels['mass'],self.counts,query,prior)
            if features['eligible']:result=features['transport']+self.choice['beta']*features['count']
        elif kind=='outlier':
            ks={name:contaminated(v,self.choice['epsilon']) if name!='pred' else v for name,v in kernels.items()}
            result=query_score(query,ks,self.counts,prior,config,frozen,priors,raw)
        elif kind not in ('baseline','atom','integral_count','outlier'):
            raise ValueError('unsupported_global_extraction_coarse_kind')
        return result,baseline


@torch.inference_mode()
def extract(bank,queries,payload,widths,inputs,choice,out,status,cohort,pool_size,augment=False):
    device=torch.device('cuda'); scorer=Scorer(bank,payload,widths,inputs,choice,device)
    indices=bank['single_indices']; smiles=[bank['smiles'][i] for i in indices]; lookup={s:i for i,s in enumerate(smiles)}
    candidate_groups=[identity(s)[1] for s in smiles] if cohort=='train' else None
    rows=[]; raw=inputs[-1]; prior=inputs[1]
    for key in sorted({condition_key(r,payload['metadata'],payload['key_mode']) for r in queries}):
        group=[(i,r) for i,r in enumerate(queries) if condition_key(r,payload['metadata'],payload['key_mode'])==key]
        scorer.group(key,any(r['source']=='HMDB' for _,r in group))
        for order,row in group:
            query=attach_observation(row,payload['annotations'].get(make_key(row),{}))
            query['_condition_known']=bool(key[4+SOLVENT_DIM-1]>0)
            variants=[('observed',query)]
            if augment and (cohort!='test' or order<200):variants+=missing_queries(query,'1H')
            for variant,observed in variants:
                coarse,baseline=scorer.score(observed)
                initial=coarse.topk(min(pool_size,len(smiles)),largest=False).indices
                tied=(coarse<=coarse[initial[-1]]+1e-7).nonzero().reshape(-1)
                rerank_eligible=len(tied)<=pool_size*4
                pool=tied[coarse[tied].argsort()] if rerank_eligible else initial
                target=lookup[row['smiles']]
                present=bool((pool==target).any())
                if cohort=='train' and not present:pool=torch.cat([pool,pool.new_tensor([target])])
                positions=(pool==target).nonzero().reshape(-1)
                features,support=candidate_features(scorer.kernels['pred'][pool],scorer.tags[pool],scorer.counts[pool],coarse[pool],observed,prior,raw)
                rows.append(dict(query_order=order,source=row['source'],record_id=row['record_id'],smiles=row['smiles'],group=row['group'],
                    candidate_indices=pool.cpu(),features=features.cpu(),target_position=int(positions[0]) if len(positions) else -1,
                    coarse_scores=coarse[pool].cpu(),
                    target_in_coarse_pool=present,coarse_rank=int((coarse<=coarse[target]+1e-7).sum()),
                    baseline_rank=int((baseline<=baseline[target]+1e-7).sum()),baseline_top10=baseline.topk(10,largest=False).indices.cpu().tolist(),
                    coarse_top10=coarse.topk(10,largest=False).indices.cpu().tolist(),kind=support['kind'],query_observation=observed['observation'],
                    positions=support['positions'],reported=support['reported'],variant=variant,rerank_eligible=rerank_eligible))
                if cohort=='train':rows[-1]['negative_exclusion_mask']=torch.tensor([candidate_groups[i]==row['group'] for i in pool.cpu().tolist()])
                if len(rows)%25==0:status('extracting',queries=len(rows),base_queries=len(queries))
        scorer.kernels=scorer.tags=scorer.variable=None; torch.cuda.empty_cache()
    rows.sort(key=lambda r:(r['query_order'],r['variant']))
    manifest=dict(cohort=cohort,queries=len(rows),query_sha256=stamp_queries(queries),feature_names=FEATURE_NAMES,
        pool_size=pool_size,coarse_config=choice,candidate_count=len(smiles),checkpoint_sha256=bank['checkpoint_sha256'],
        candidate_order_sha256=hashlib.sha256('\n'.join(smiles).encode()).hexdigest(),
        coarse_recall=float(np.mean([r['target_in_coarse_pool'] for r in rows])),
        appended_training_positives=sum(not r['target_in_coarse_pool'] for r in rows) if cohort=='train' else 0,
        variants={name:sum(r['variant']==name for r in rows) for name in sorted({r['variant'] for r in rows})},
        truth_only_for_training_labels_and_metrics=True,external_api_calls=0)
    manifest['source_inputs_sha256']=json.loads((out/'protocol.json').read_text())['inputs_sha256']
    manifest['feature_code_sha256']=sha(Path(__file__).with_name('h_global_features.py'))
    torch.save(dict(rows=rows,smiles=smiles,manifest=manifest),out/'features.pt')
    manifest['feature_cache_sha256']=sha(out/'features.pt')
    (out/'summary.json').write_text(json.dumps(manifest,indent=2))


def main():
    p=argparse.ArgumentParser()
    for name in ('base','bank','cohorts','widths','cohort','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--extension'); p.add_argument('--coarse',default='baseline')
    p.add_argument('--pool-size',type=int,default=512); p.add_argument('--frozen-selection'); p.add_argument('--limit',type=int,default=0)
    p.add_argument('--augment',action='store_true')
    args=p.parse_args()
    if args.cohort not in ('train','original_val','tune','verify','test'):raise ValueError('invalid_global_cohort')
    choices={c['name']:c for c in configs()}; choice=choices[args.coarse]
    required=actual_inputs(args.base,args.bank,args.widths,args.cohorts,args.extension)
    if args.cohort in ('verify','test'):
        if not args.frozen_selection:raise ValueError('global_selection_not_frozen')
        allowed=frozen_choices(args.frozen_selection,required)
        if choice not in allowed:raise ValueError('global_coarse_not_frozen')
        frozen=json.loads(Path(args.frozen_selection).read_text())
        protocol=frozen['feature_extractor_protocol']
        if args.pool_size!=protocol['pool_size'] or bool(args.augment)!=protocol['augment'] or FEATURE_NAMES!=protocol['feature_names']:
            raise ValueError('global_feature_extractor_protocol_changed')
        if protocol['feature_code_sha256']!=sha(Path(__file__).with_name('h_global_features.py')):
            raise ValueError('global_feature_code_changed_after_freeze')
        if args.limit:raise ValueError('cannot_limit_locked_evaluation')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
        payload=torch.load(Path(args.cohorts)/'cohorts.pt',map_location='cpu',weights_only=False)
        bank=load_bank(args.bank,args.extension)
        queries=bank['test'] if args.cohort=='test' else payload['cohorts'][args.cohort]
        if args.limit:queries=queries[:args.limit]
        widths=json.loads(Path(args.widths).read_text())['selected_ppm']
        (out/'protocol.json').write_text(json.dumps(dict(inputs_sha256={str(Path(p).resolve()):sha(p) for p in required},
            coarse_config=choice,pool_size=args.pool_size,cohort=args.cohort,source_sha256=sha(__file__)),indent=2))
        extract(bank,queries,payload,widths,scoring_inputs(Path(args.base)),choice,out,status,args.cohort,args.pool_size,args.augment)
        status('complete'); (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc)); (out/'EXIT_CODE').write_text('1\n'); raise


if __name__=='__main__':
    main()

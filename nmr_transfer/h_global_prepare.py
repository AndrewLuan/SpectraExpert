"""功能概述：为整体实验 ¹H 检索研究冻结训练、原验证和新增验证查询。

main 只从已有 prepared 缓存的 train/val 构建查询侧表，保留原 320-val，按结构哈希
选择独立的自然分布 tune/verify 支持。train 每结构最多一条且不使用测试标签；审核
暂缓谱排除于拟合与新增验证。freeze 保存查询身份、条件、候选次序、物理输入和受
保护文件哈希；候选库、原测试与压力变体始终引用既有 bank，不改写任何输入。
选择条件不以模型误差、测试来源权重或分子名称为依据。缺失条件不填默认值。
入口不训练、不调用外部服务；h_global_extract 使用本产物生成可复用排名特征。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch

from .h_condinfer import condition_key
from .h_labels import make_key
from .h_metadata import metadata_view
from .h_observation_study import stamp_queries
from .h_reviewed_supervision import load_artifact, apply_review, common_validation
from .prepare import sha


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def unique_structures(rows, salt):
    result = {}
    for row in sorted(rows, key=lambda r: digest((salt, r['source'], r['record_id'], r['smiles']))):
        result.setdefault(row['group'], row)
    return sorted(result.values(), key=lambda r: digest((salt, r['group'])))


def main():
    parser = argparse.ArgumentParser()
    for name in ('base', 'prepared', 'bank', 'review', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--train-size', type=int, default=2048)
    parser.add_argument('--tune-size', type=int, default=1024)
    parser.add_argument('--verify-size', type=int, default=1024)
    args = parser.parse_args()
    out, base, prepared = Path(args.out), Path(args.base), Path(args.prepared)
    out.mkdir(parents=True, exist_ok=False)
    try:
        cache = torch.load(prepared/'cache.pt', map_location='cpu', weights_only=False, mmap=True)
        bank = torch.load(args.bank, map_location='cpu', weights_only=False, mmap=True)
        labels = torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt', map_location='cpu', weights_only=False)
        metadata = metadata_view(torch.load(prepared/'metadata.pt', map_location='cpu', weights_only=False), bank['condition_key_mode'])
        artifact, review_meta = load_artifact(args.review)
        train = [r for r in cache['rows'] if r['domain']=='experimental' and r['split']=='train' and r['nucleus']=='1H']
        native_val = [r for r in cache['rows'] if r['domain']=='experimental' and r['split']=='val' and r['nucleus']=='1H']
        train, review_audit = apply_review(labels, train, artifact, 'domain')
        native_val, val_audit = common_validation(native_val, artifact)
        retained = [r for r in train if r['review_spectral_weight']>0]
        selected_train = unique_structures(retained, 'global_train_317')[:args.train_size]
        original = bank['val']
        if len(selected_train)!=args.train_size:
            raise ValueError('insufficient_unique_training_support')
        if {r['group'] for r in train}&{r['group'] for r in native_val+original}:
            raise ValueError('full_training_validation_overlap')
        excluded = {r['group'] for r in original}
        extra = unique_structures([r for r in native_val if r['group'] not in excluded], 'global_validation_317')
        tune = extra[:args.tune_size]
        verify = extra[args.tune_size:args.tune_size+args.verify_size]
        if len(tune)!=args.tune_size or len(verify)!=args.verify_size:
            raise ValueError('insufficient_unique_validation_support')
        cohorts = dict(train=selected_train, original_val=original, tune=tune, verify=verify)
        groups = {k:{r['group'] for r in rows} for k,rows in cohorts.items()}
        for i, a in enumerate(groups):
            for b in list(groups)[i+1:]:
                if groups[a]&groups[b]:
                    raise ValueError('cohort_structure_overlap:'+a+':'+b)
        candidate_smiles = [bank['smiles'][i] for i in bank['single_indices']]
        if len(candidate_smiles)!=114982 or len(bank['test'])!=1000 or len(original)!=320:
            raise ValueError('frozen_evaluation_cohort_drift')
        known = set(bank['predictions'])
        descriptions = {}
        for name, rows in cohorts.items():
            keys = {condition_key(r, metadata, bank['condition_key_mode']) for r in rows}
            descriptions[name] = dict(queries=len(rows), unique_structures=len(groups[name]),
                sources=dict(collections.Counter(r['source'] for r in rows)), query_sha256=stamp_queries(rows),
                conditions=len(keys), missing_bank_conditions=[list(k) for k in sorted(keys-known)])
        old_protected = json.loads((base.parent/'nmr_joint_conditions_20261020/study/plan.json').read_text()).get('inputs_sha256', {})
        paths = set(old_protected)|{args.bank, str(prepared/'manifest.json'), str(Path(args.review)/'artifact.pt')}
        protected = {str(p):sha(p) for p in sorted(map(Path, paths)) if p.is_file()}
        manifest = dict(cohorts=descriptions, candidate_order_sha256=digest('\n'.join(candidate_smiles)),
            candidate_count=len(candidate_smiles), frozen_test_query_sha256=stamp_queries(bank['test']),
            physics_sha256=hashlib.sha256(bank['physics'][bank['single_indices']].contiguous().numpy().tobytes()).hexdigest(),
            protected_files_sha256=protected, review=review_audit, validation_review=val_audit,
            external_api_calls=0, selection='original320-val + independent hash-selected natural validation; no test selection',
            verify_scope='locked independent validation; open once after all candidate choices frozen',
            train_sampling='one record per connectivity; hash-selected without error/source/test weighting')
        torch.save(dict(cohorts=cohorts, metadata=metadata, annotations=labels,
                        bank_path=args.bank, key_mode=bank['condition_key_mode']), out/'cohorts.pt')
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        (out/'EXIT_CODE').write_text('0\n')
        print(json.dumps(descriptions), flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc))))
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__=='__main__':
    main()

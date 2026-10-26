"""功能概述：汇总已冻结的审核训练消融，量化配对检索变化及DMF反事实。
main只读取已完成paired_study，按预设layout/seed比较标签更正、谱隔离和物理分组。
每个对照核对相同查询身份，cluster_bootstrap按连接结构重采样5000次；变差样例和
改善样例均保留。state_signature核查共同初始化和最终严格别名，避免把不同日志或
checkpoint封装误当不同模型。报告单列17条HMDB及371缺峰变体的真实分母。
DMF结构报告状态oracle仅作为测试后诊断加入；不修改选模、测试排名或生产配置。
输出独占analysis目录及EXIT_CODE，保存来源SHA/梯度隔离曝光；无拟合、训练或API。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch

from .h_count_finalization import cluster_bootstrap, index
from .prepare import sha


def state_signature(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    digest=hashlib.sha256(json.dumps(checkpoint['args'],sort_keys=True).encode())
    for key,value in sorted(checkpoint['state_dict'].items()):
        digest.update(key.encode());digest.update(str(value.dtype).encode())
        digest.update(str(list(value.shape)).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);study=root/'paired_study';out=root/'analysis';out.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text())
    rows=lambda p:[json.loads(line) for line in p.read_text().splitlines()]
    try:
        if (study/'EXIT_CODE').read_text().strip()!='0':raise ValueError('paired_study_not_complete')
        summary=read(study/'summary.json');plan=read(study/'plan.json');arms={r['name']:r for r in plan['arms']}
        predictions={};missing={};training={};initial={};aliases=collections.defaultdict(list)
        for name,arm in arms.items():
            folder=study/'training'/name/'train';config=read(folder/'config.json');done=read(folder/'complete.json')
            if sha(folder/'best.pt')!=summary['tests'][name]['checkpoint_sha256']:raise ValueError('analysis_checkpoint_drift')
            signature=state_signature(folder/'initial.pt')
            if arm['layout'] in initial and initial[arm['layout']]!=signature:raise ValueError('analysis_initial_state_not_paired')
            initial[arm['layout']]=signature;aliases[state_signature(folder/'best.pt')].append(name)
            predictions[name]=rows(study/'evaluation'/name/'test/predictions.jsonl')
            missing[name]=rows(study/'evaluation'/name/'test/missing_predictions.jsonl')
            if len(predictions[name])!=1000 or len(missing[name])!=371:raise ValueError('analysis_query_count_drift')
            training[name]=dict(best_phase=done['best_phase'],best_epoch=done['best_epoch'],
                forward_validation=done['best_experimental_macro'],
                held_spectral_exposures=sum(config['held_spectrum_exposures_by_epoch']),
                review=config['review_audit'],sampling_sha256=config['sampling_sequence_sha256']['experimental'])
        pairs=[]
        for layout in ('local','physical'):
            for seed in (193,211):
                name=lambda mode:f'{layout}_{mode}_s{seed}'
                for new,old in [('corrections','legacy'),('domain','corrections'),('domain','legacy')]:
                    if name(new) in arms and name(old) in arms:pairs.append((name(new),name(old)))
        for mode in ('legacy','corrections','domain'):
            new=f'physical_{mode}_s193';old=f'local_{mode}_s193'
            if new in arms and old in arms:pairs.append((new,old))
        contrasts={}
        for new,old in pairs:
            current=index(predictions[new]);previous=index(predictions[old])
            if current.keys()!=previous.keys():raise ValueError('analysis_query_identity_drift')
            delta=[dict(record_id=k[1],source=k[0],smiles=k[2],before=previous[k]['rank'],after=current[k]['rank'],
                        reciprocal_delta=1/current[k]['rank']-1/previous[k]['rank']) for k in current]
            stress={}
            for variant in ('frequency_crop','peak_dropout'):
                nn=[r for r in missing[new] if r['variant']==variant];oo=[r for r in missing[old] if r['variant']==variant]
                stress[variant]=cluster_bootstrap(nn,oo,True)
            contrasts[new+'__vs__'+old]=dict(new=new,old=old,
                cluster_difference=cluster_bootstrap(predictions[new],predictions[old]),
                improved=sum(r['after']<r['before'] for r in delta),worsened=sum(r['after']>r['before'] for r in delta),
                tied=sum(r['after']==r['before'] for r in delta),
                top1_gains=[r for r in delta if r['before']>1 and r['after']==1],
                top1_losses=[r for r in delta if r['before']==1 and r['after']>1],
                largest_improvements=sorted(delta,key=lambda r:(-r['reciprocal_delta'],r['record_id']))[:5],
                largest_regressions=sorted(delta,key=lambda r:(r['reciprocal_delta'],r['record_id']))[:5],missing=stress)
        dmf={}
        for name in arms:
            diag=read(study/'evaluation'/name/'test/site_diagnostics.json')
            ranks={r['record_id']:r['rank'] for r in predictions[name]}
            dmf[name]=[dict(r,main_test_rank=ranks.get(r['record_id'])) for r in diag['DMF']]
        oracle=root/'DMF_structural_reporting_oracles/summary.json'
        report=dict(selected_by_validation=summary['selection']['selected_variant'],training=training,
            test_metrics=summary['tests'],validation_metrics=summary['selection']['validation'],contrasts=contrasts,
            initial_state_exact_within_layout=True,strict_final_model_aliases=[v for v in aliases.values() if len(v)>1],
            DMF=dmf,DMF_oracles=read(oracle) if oracle.exists() else None,
            source_counts=dict(collections.Counter(r['source'] for r in next(iter(predictions.values())))),
            inference_protocol='fixed legacy mean-model scoring; all choices frozen before test; no production promotion',
            bootstrap_scope='exploratory paired development-test comparisons; all predeclared contrasts retained',
            protected_files_exact=summary['protected_files_exact'],extra_API_calls=0,
            source_sha256=sha(__file__),input_sha256={str(f):sha(f) for f in [study/'summary.json',study/'plan.json',study/'evaluation/frozen_selection.json']})
        (out/'summary.json').write_text(json.dumps(report,indent=2));(out/'EXIT_CODE').write_text('0\n')
        for key,r in contrasts.items():print(key,json.dumps(r['cluster_difference']['metrics']['MRR']),flush=True)
    except Exception:
        (out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

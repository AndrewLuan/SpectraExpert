"""功能概述：联合研究完成后，单独量化甲酰胺/其他酰胺的验证误差及条件支持。

main只读取完成的16臂研究；复用审核规则和原苯支持，按原结构分区
选val，不做参数拟合或检索选型。forward_records逐条保存已观测到预测的匹配误差，
完整积分/原子口径另报分位误差；未知窗口不从峰极值补齐。稀少新增val单列。
对train配对的误差只记为训练拟合诊断；pH/溶剂已知度和registry覆盖单列，避免
把观测口径或未认证解码回退解释成纯条件效应。额外DMF三条件发射按结构母节点
分类，报告概率来自共享头，没有指定的甲基/甲酰位移常数。
输出独占focus_diagnostics及源码/权重SHA；不改模型、数据、全局冻结选择或原排名。
无外部API；所有测试后诊断都明确标记，不能反向用于本轮选模。
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .emission_model import make_model
from .h_acyl import acyl_environment
from .h_condmodel import training_details
from .h_conditions import ConditionsSpectra
from .h_formyl_reporting import candidate_families
from .h_joint_conditions import pair_loss
from .h_labels import make_key
from .h_metadata import metadata_view, RECOVERED_MODE
from .h_uncertainty import node_classes
from .h_reviewed_supervision import load_artifact,apply_review,common_validation
from .h_site_quality import apply_quality
from .h_observation_study import stamp_queries
from .observation import observation_loss
from .prepare import sha
from .train import collate


@torch.inference_mode()
def forward_records(model, rows, graphs, metadata, labels, device):
    dataset = ConditionsSpectra(rows,graphs,'1H',metadata,annotations=labels)
    loader = DataLoader(dataset,batch_size=128,num_workers=2,collate_fn=collate)
    records = []
    offset = 0
    for batch in loader:
        batch = batch.to(device)
        details = model(batch,details=True)
        observed = training_details(details,batch)
        errors = observation_loss(observed['quantiles'],batch)*10
        for j,row in enumerate(rows[offset:offset+batch.num_graphs]):
            meta = metadata.get(make_key(row),{})
            label = labels.get(make_key(row),{})
            family = candidate_families([row['smiles']])[0]
            p = model.reporting_head([family],batch.condition_vector[j:j+1]).item()
            records.append(dict(source=row['source'],record_id=row['record_id'],smiles=row['smiles'],
                group=row['group'],category='formamide' if family=='formamide' else 'other_amide',
                solvent=meta.get('family','unknown'),pH=meta.get('pH'),pH_used=meta.get('pH_used',False),
                observed_error_ppm=float(errors[j]),known_registry=bool(batch.equivalence_known[j]),
                integer_integral_consistent=bool(label.get('integral_consistent')),
                full_atom_labels=bool(label.get('full_atom_labels')),
                full_quantile_error_ppm=float((observed['quantiles'][j]-batch.supervised_q[j]).abs().mean()*10)
                    if bool(label.get('integral_consistent') or label.get('full_atom_labels')) else None,
                reporting_probability=p if model.reporting_enabled and row['source']=='HMDB' else None))
        offset += batch.num_graphs
    return records


def grouped_metrics(records):
    groups = collections.defaultdict(list)
    for r in records:
        groups['all'].append(r)
        groups[r['category']].append(r)
        groups[r['category']+':'+r['solvent']].append(r)
    return {key:dict(records=len(rs),connectivities=len({r['group'] for r in rs}),
                     known_registry_records=sum(r['known_registry'] for r in rs),
                     mean_observed_error_ppm=float(np.mean([r['observed_error_ppm'] for r in rs])),
                     median_observed_error_ppm=float(np.median([r['observed_error_ppm'] for r in rs])))
            for key,rs in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--base',required=True)
    args = parser.parse_args()
    root,base = Path(args.root),Path(args.base)
    study = root/'study'
    if (study/'EXIT_CODE').read_text().strip()!='0':
        raise ValueError('joint_study_not_complete')
    out = root/'focus_diagnostics'
    out.mkdir(exist_ok=False)
    try:
        torch.set_num_threads(4)
        device = torch.device('cuda')
        prepared = base/'hydrogen_analysis/condition_response/solvent_resolved_prepared'
        cache = torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True)
        metadata = metadata_view(torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False),RECOVERED_MODE)
        labels = torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        from .h_message_support import add_support
        support,extra,support_audit = add_support(cache,metadata,labels,base/'hydrogen_analysis/condition_response/benzene_formamide_inventory',
            json.loads((prepared/'manifest.json').read_text()),base/'hydrogen_analysis/site_supervision/train/best.pt',device)
        exp = [r for r in cache['rows'] if r['domain']=='experimental' and r['split']=='train']+support
        artifact,_ = load_artifact(base.parent/'nmr_assignment_training_20261020/review_artifact')
        exp,review_audit = apply_review(labels,exp,artifact,'domain')
        quality = apply_quality(labels,exp,'methyl_record_mask')
        val,validation_audit = common_validation([r for r in cache['rows'] if r['domain']=='experimental' and r['split']=='val'],artifact)
        val = [r for r in val if acyl_environment(r['smiles'])[:,0].any()]
        audit = dict(review=review_audit,validation=validation_audit,quality=quality,support=support_audit)
        first_arm = json.loads((study/'plan.json').read_text())['arms'][0]['name']
        candidate_bank = torch.load(study/'evaluation'/first_arm/'bank.pt',map_location='cpu',weights_only=False,mmap=True)
        dmf_physics = candidate_bank['physics'][candidate_bank['smiles'].index('CN(C)C=O')].clone().numpy()
        del candidate_bank
        results = {}
        for arm in json.loads((study/'plan.json').read_text())['arms']:
            name = arm['name']
            path = study/'training'/name/'train/best.pt'
            saved = torch.load(path,map_location='cpu',weights_only=False)
            model = make_model(saved['args']).to(device).eval()
            model.load_state_dict(saved['state_dict'])
            model.reporting_enabled = saved['args']['reporting_enabled']
            config = json.loads((path.parent/'config.json').read_text())
            if stamp_queries(exp)!=config['train_queries_sha256']:
                raise ValueError('diagnostic_pair_training_order_changed')
            pair_records = []
            pair_dataset = ConditionsSpectra(exp,cache['graphs'],'1H',metadata,annotations=labels)
            with torch.inference_mode():
                for pair in json.loads((path.parent/'cross_solvent_pairs.json').read_text()):
                    batch = collate([pair_dataset[pair['left']],pair_dataset[pair['right']]]).to(device)
                    error = float(pair_loss(model,batch,[pair])*10)
                    pair_records.append(dict(group=pair['group'],smiles=pair['smiles'],solvents=pair['solvents'],
                        category=candidate_families([pair['smiles']])[0],difference_error_ppm=error))
            normal = forward_records(model,val,cache['graphs'],metadata,labels,device)
            additional = forward_records(model,extra,cache['graphs'],metadata,labels,device)
            dmf = []
            for key,meta in metadata.items():
                if key[1]!='HMDB' or key[3]!='CN(C)C=O':continue
                row = dict(smiles=key[3],domain=key[0],source=key[1],record_id=key[2],cond=meta['old_cond'],
                           physics=dmf_physics,q=np.zeros(128,np.float32),observation=dict(kind='unknown',windows=[]))
                batch = collate([ConditionsSpectra([row],cache['graphs'],'1H',metadata)[0]]).to(device)
                details = model(batch,details=True)
                tags = node_classes(batch)[details['owners']]
                peaks = {label:sorted(float(v) for v in (details['values'][tags==tag]*10).unique().cpu())
                         for tag,label in [(0,'methyl'),(4,'formyl')]}
                dmf.append(dict(record_id=key[2],solvent=meta['solvent'],pH=meta['pH'],peaks=peaks,
                    reporting_probability=float(model.reporting_head(['formamide'],batch.condition_vector)[0]) if arm['report'] else 1.))
            results[name] = dict(checkpoint_sha256=sha(path),original_val=grouped_metrics(normal),
                additional_benzene_val=grouped_metrics(additional),DMF_test_diagnostic=dmf,
                training_pair_mean_error_ppm=float(np.mean([r['difference_error_ppm'] for r in pair_records])),
                training_benzene_pair_mean_error_ppm=float(np.mean([r['difference_error_ppm'] for r in pair_records if 'benzene' in r['solvents']])))
            (out/(name+'.json')).write_text(json.dumps(dict(original_val=normal,additional_val=additional,training_pairs=pair_records),indent=2))
            (out/'status.json').write_text(json.dumps(dict(stage='diagnostics',completed=len(results),total=16)))
            del model
            torch.cuda.empty_cache()
        (out/'summary.json').write_text(json.dumps(dict(arms=results,data_audit=audit,source_sha256=sha(__file__),
            scope='observed matching validation diagnostics; additional val separate; DMF test diagnostics after selection; no fit'),indent=2))
        (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        (out/'status.json').write_text(json.dumps(dict(stage='failed',error=repr(exc))))
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__=='__main__':
    main()

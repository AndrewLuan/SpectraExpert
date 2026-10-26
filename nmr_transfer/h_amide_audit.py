"""功能概述：审计训练分区内二甲基酰胺的两组甲基位移与模型分裂幅度。
amide_groups按候选连接图识别同一酰胺N上的两个CH3，不指定二者的原子对应顺序。
main仅用train实验逐H标注且两母节点均有标签的记录，比较目标组中心/间距与冻结
模型的联合发射。甲基内标注跨度超过0.03ppm单独标记，不用于简化组间距汇总。
目标两组甲基联合比较避免二维等价节点的任意映射混淆；结果不是单个H的配对证明。
报告完整/部分母节点标注与酰基类别的误差，供后续分裂门控训练判断，不选test参数。
输出独占记录、统计、source/checkpoint身份及退出码；原始数据、标签与模型均只读。
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .h_labels import make_key
from .h_conditions import ConditionsSpectra,LOCAL_NAMES
from .emission_model import make_model,weighted_quantiles
from .train import collate
from .h_observation_study import write_rows
from .prepare import sha


def amide_groups(graph):
    env=graph.local_environment;z=graph.x[:,:54].argmax(1)+1;src,tgt=graph.edge_index;valid=(env[src,LOCAL_NAMES.index('amide_N_methyl')]>0)&(z[tgt]==7);groups=collections.defaultdict(list)
    for a,b in zip(src[valid].tolist(),tgt[valid].tolist()):groups[b].append(a)
    return [(n,sorted(set(parents))) for n,parents in groups.items() if len(set(parents))==2]


def acyl_kind(graph,n):
    z=graph.x[:,:54].argmax(1)+1;src,tgt=graph.edge_index;neighbors=tgt[src==n];carbonyl=[int(i) for i in neighbors if graph.local_environment[int(i),LOCAL_NAMES.index('carbonyl_C')]>0]
    if len(carbonyl)!=1:return 'other'
    c=carbonyl[0];near=tgt[src==c];zs=z[near].tolist()
    if 1 in zs:return 'formyl'
    if zs.count(7)>1:return 'urea'
    if zs.count(8)>1:return 'carbamate'
    if any(graph.local_environment[int(i),LOCAL_NAMES.index('aromatic')]>0 for i in near):return 'aroyl'
    return 'alkyl_acyl'


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--out',required=True);a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading');cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False);labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        groups_by_smi={};rows=[]
        for r in cache['rows']:
            if r['split']!='train' or r['domain']!='experimental':continue
            if r['smiles'] not in groups_by_smi:groups_by_smi[r['smiles']]=amide_groups(cache['graphs'][r['smiles']])
            sites=labels.get(make_key(r),{}).get('groups',{})
            if any(all(parent in sites for parent in parents) for _,parents in groups_by_smi[r['smiles']]):rows.append(r)
        checkpoint=base/'hydrogen_analysis/site_supervision/train/best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict']);loader=DataLoader(ConditionsSpectra(rows,cache['graphs'],'1H',metadata,annotations=labels),batch_size=128,num_workers=2,collate_fn=collate);records=[];offset=0;status('predicting',rows=len(rows))
        for b in loader:
            b=b.to(device);d=model(b,details=True)
            for j,r in enumerate(rows[offset:offset+b.num_graphs]):
                graph=cache['graphs'][r['smiles']];sites=labels[make_key(r)]['groups'];start=int(b.ptr[j])
                for n,parents in groups_by_smi[r['smiles']]:
                    if not all(parent in sites for parent in parents):continue
                    observed=[sites[parent] for parent in parents];medians=sorted(float(np.median(v)) for v in observed);stable=all(np.ptp(v)<=.03 for v in observed)
                    keep=torch.isin(d['owners'],torch.tensor([p+start for p in parents],device=device));q=weighted_quantiles(d['values'][keep],d['weights'][keep],torch.zeros(int(keep.sum()),device=device,dtype=torch.long),1,points=2)[0]*10;low,high=q.cpu().tolist()
                    records.append(dict(record_id=r['record_id'],source=r['source'],smiles=r['smiles'],amide_N=n,parents=parents,observed=observed,target_pair=medians,predicted_pair=[low,high],target_gap=medians[1]-medians[0],predicted_gap=high-low,center_error_ppm=(low+high-sum(medians))/2,stable_methyl_labels=stable,full_parent_labels=all(len(v)==3 for v in observed),acyl_kind=acyl_kind(graph,n)))
            offset+=b.num_graphs
        stable=[r for r in records if r['stable_methyl_labels']]
        def stats(rs):
            return dict(groups=len(rs),median_target_gap_ppm=float(np.median([r['target_gap'] for r in rs])) if rs else None,median_predicted_gap_ppm=float(np.median([r['predicted_gap'] for r in rs])) if rs else None,mean_absolute_gap_error_ppm=float(np.mean([abs(r['target_gap']-r['predicted_gap']) for r in rs])) if rs else None,median_abs_center_error_ppm=float(np.median([abs(r['center_error_ppm']) for r in rs])) if rs else None)
        summary=dict(records=len(records),stable=stats(stable),by_acyl={k:stats([r for r in stable if r['acyl_kind']==k]) for k in sorted({r['acyl_kind'] for r in stable})},by_label_completeness={str(k):stats([r for r in stable if r['full_parent_labels']==k]) for k in [False,True]},checkpoint_sha256=sha(checkpoint),scope='train-only unordered union of two methyl parent labels; no test fit')
        write_rows(out/'records.jsonl',records);(out/'summary.json').write_text(json.dumps(summary,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

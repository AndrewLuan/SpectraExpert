"""功能概述：在train/val已赋值实验H上检查物理等价绑定是否与报告位移一致。
main只用冻结source-order分区连接原子标签；一组仅在全部成员均被标注时计算跨度，
整分子完整H标注时另比较报告位移簇数与环境数。未知频窗和缺峰不填零、不当负例。
输出来源/分区/完整母节点分组跨度及最大跨度样例，拒绝test目标进入统计。
该审计描述标签与结构假设的一致性，不把NMR数据库标签默认作无误物理真值。
源数据、registry和预测只读，无拟合、GPU或外部服务；依赖h_observed聚类口径。
"""
import argparse,collections,json
from pathlib import Path
import numpy as np
from rdkit import Chem
from .data import identity
from .h_observed import cluster_lines
from .prepare import sha


def main():
    p=argparse.ArgumentParser()
    for k in ['source','registry','out']:p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);root=Path(a.registry);structures={r['record_id']:r for r in map(json.loads,open(root/'structures.jsonl'))};pred={r['record_id']:r for r in map(json.loads,open(root/'predictions.jsonl'))};spans=collections.defaultdict(list);counts=collections.defaultdict(collections.Counter);examples=[];errors=collections.Counter()
    try:
        for r in map(json.loads,open(a.source)):
            if r.get('nucleus')!='1H' or r['record_id'] not in structures:continue
            s=structures[r['record_id']];pr=pred.get(r['record_id'],{}).get('nuclei',{}).get('1H',{})
            if pr.get('status')!='predicted':continue
            try:canonical,group,split=identity(r['smiles'])
            except ValueError as exc:errors[str(exc)]+=1;continue
            if split=='test':continue
            if identity(s['source_smiles'])[0]!=canonical:raise ValueError('group_label_structure_mismatch')
            mask=r['atom_target_mask'];values=r['atom_targets'];order=s['source_to_canonical'];mol=Chem.AddHs(Chem.MolFromSmiles(s['source_smiles']));protons=[i for i,z in enumerate(s['tokens']) if z==1];source=r['source_ref'];key='|'.join([source,split]);counts[key]['records']+=1
            if len(mask)!=len(order) or len(values)!=len(order):errors['source_label_length']+=1;continue
            if all(mask[i] and values[i] is not None for i in protons):
                counts[key]['all_H_assigned']+=1;observed=[float(values[i]) for i in protons];k=len(cluster_lines(observed,.05)[0]);counts[key]['assigned_clusters_eq_environments']+=int(k==pr['count']);counts[key]['assigned_clusters_lt_environments']+=int(k<pr['count']);counts[key]['assigned_clusters_gt_environments']+=int(k>pr['count'])
            for g in pr['groups']:
                if len(g)<2 or not all(mask[i] and values[i] is not None for i in g):continue
                xs=[float(values[i]) for i in g]
                if not np.isfinite(xs).all():errors['nonfinite_group_labels']+=1;continue
                parents=collections.Counter(mol.GetAtomWithIdx(order[i]).GetNeighbors()[0].GetIdx() for i in g);complete=all(n==sum(x.GetAtomicNum()==1 for x in mol.GetAtomWithIdx(parent).GetNeighbors()) for parent,n in parents.items());span=max(xs)-min(xs);spans[key].append(span);spans[key+'|'+('whole_parent' if complete else 'partial_parent')].append(span)
                if span>.2:examples.append(dict(source=source,split=split,record_id=r['record_id'],smiles=canonical,source_H_group=g,reported_ppm=xs,span_ppm=span,whole_parent=complete,structure_features=pred[r['record_id']].get('features',{}),conditions=r.get('experimental_conditions',r.get('conditions',{}))))
        summary={k:dict(groups=len(v),quantiles=np.quantile(v,[0,.5,.9,.95,.99,1]).tolist(),fraction_span_over_005=float(np.mean(np.array(v)>.05)),fraction_span_over_02=float(np.mean(np.array(v)>.2))) for k,v in spans.items()};report=dict(counts={k:dict(v) for k,v in counts.items()},group_spans=summary,errors=dict(errors),test_labels_used=False,source_sha256={str(f):sha(f) for f in [Path(a.source),root/'structures.jsonl',root/'predictions.jsonl']});(out/'summary.json').write_text(json.dumps(report,indent=2));examples.sort(key=lambda r:-r['span_ppm']);(out/'largest_binding_conflicts.json').write_text(json.dumps(examples[:100],indent=2));(out/'EXIT_CODE').write_text('0\n');print(json.dumps(report),flush=True)
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

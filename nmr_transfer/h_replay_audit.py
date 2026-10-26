"""功能概述：仅审计训练重放中的目标校准与甲基等价约束，不使用测试标签。
audit比较q/domain_q的逐分位偏移，统计校准侧表是否与实际训练目标一致；
逐母节点汇总理论和实验CH3标签跨度，并单列同酰胺N的两个甲基。
等价甲基约束来自候选图，标签跨度只是数据诊断，不决定检索参数。
输出独占JSON、状态及退出码；prepared缓存、标签和原始记录均只读。
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import torch
from .h_labels import make_key
from .h_conditions import LOCAL_NAMES
from .h_amide_audit import amide_groups


def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None,mean=float(a.mean()) if len(a) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4);(out/'status.json').write_text(json.dumps(dict(stage='loading')));cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);labels=torch.load(Path(a.base)/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        counts=collections.Counter();deltas=collections.defaultdict(list);spans=collections.defaultdict(list);gaps=collections.defaultdict(list);records=[];graph_cache={}
        for r in cache['rows']:
            if r['split']!='train':continue
            tag=r['domain']+'|'+r['source'];counts[tag]+=1;meta=labels.get(make_key(r),{})
            if 'domain_q' in r:
                counts[tag+'|domain_q']+=1;deltas[tag].append(float(np.mean(np.abs(np.asarray(r['domain_q'])-np.asarray(r['q'])))))
            if meta.get('calibration'):counts[tag+'|calibrated_site_labels']+=1
            sites=meta.get('groups',{})
            if not sites:continue
            if r['smiles'] not in graph_cache:
                g=cache['graphs'][r['smiles']];graph_cache[r['smiles']]=(g.local_environment[:,LOCAL_NAMES.index('CH3')]>0,amide_groups(g))
            methyl,groups=graph_cache[r['smiles']]
            for node,vs in sites.items():
                if methyl[int(node)] and len(vs)>1:spans[tag].append(float(np.ptp(vs)))
            for nitrogen,parents in groups:
                if not all(parent in sites for parent in parents):continue
                vs=[sites[parent] for parent in parents];gap=abs(float(np.median(vs[0])-np.median(vs[1])));gaps[tag].append(gap)
                records.append(dict(source=r['source'],domain=r['domain'],record_id=r['record_id'],smiles=r['smiles'],parents=parents,observed=vs,within_spans=[float(np.ptp(v)) for v in vs],between_gap=gap,full_parent_labels=[len(v)==3 for v in vs]))
        result=dict(training_counts=dict(counts),q_domain_q_mean_abs_ppm={k:stats(v) for k,v in deltas.items()},CH3_within_parent_span_ppm={k:dict(stats(v),over_003=sum(x>.03 for x in v),over_02=sum(x>.2 for x in v)) for k,v in spans.items()},NMe2_between_parent_gap_ppm={k:stats(v) for k,v in gaps.items()},scope='only prepared train rows; no model fitting or test record use')
        (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'amide_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));(out/'status.json').write_text(json.dumps(dict(stage='complete')));(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        (out/'status.json').write_text(json.dumps(dict(stage='failed',error=repr(exc))));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

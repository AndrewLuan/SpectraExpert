"""功能概述：在共同的独立理论val结构上比较多个冻结模型的三种目标误差。
main只读取theory_targets_consistent中同一批val，与raw/calibrated/physical三种
目标分别计算W1近似和连接结构宏误差，防止不同标签口径的分数直接混比。
报告结构认证环境数与模型不同预测位置数的关系；它是容量诊断，不冒充实测峰数。
全部模型从checkpoint factory恢复，缺图用经过SHA核验的原GINE补充缓存，
不复用不同深骨干的最终条件表示。输出逐记录预测和汇总至独占目录；不选模型、
不读test目标、不修改prepared/侧表/权重，无外部服务。依赖h_count_train侧表契约。
--registry为扩展侧表提供独立环境K；冻结输入特征中的未知0仍保持未知，不作为K=0。
记录来源及输入K，分来源输出三种目标误差，避免QM9规模掩盖重算结构的误差。
"""
import argparse,collections,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from .emission_model import make_model
from .h_conditions import ConditionsSpectra,local_environment
from .data import graph
from .h_labels import make_key
from .h_deep_context import verify_cached_backbone
from .train import collate
from .prepare import sha


@torch.inference_mode()
def evaluate(model,rows,graphs,variants,device,registry=None):
    loader=DataLoader(ConditionsSpectra(rows,graphs,'1H',metadata={}),batch_size=128,num_workers=2,collate_fn=collate);offset=0;records=[]
    for b in loader:
        d=model(b.to(device),details=True);q=d['quantiles'].cpu().numpy()*10;values=d['values'].cpu().numpy()*10;ids=d['groups'].cpu().numpy()
        for j,r in enumerate(rows[offset:offset+b.num_graphs]):
            v=variants[make_key(r)];pred=values[ids==j];input_k=round(float(r['physics'][5])*30);k=registry.get(r['smiles'],{}).get('counts',{}).get('1H') if registry is not None else input_k or None;records.append(dict(record_id=r['record_id'],source=r['source'],smiles=r['smiles'],group=r['group'],input_count_feature=input_k,certified_environments=k,distinct_predicted_positions=len(np.unique(pred)),errors={m:float(np.abs(q[j]-v[m]['q']).mean()) for m in ['raw','calibrated','physical']}))
        offset+=b.num_graphs
    metrics={}
    for mode in ['raw','calibrated','physical']:
        groups=collections.defaultdict(list)
        for r in records:groups[r['group']].append(r['errors'][mode])
        x=np.array([r['errors'][mode] for r in records]);metrics[mode]=dict(mean_W1_ppm=float(x.mean()),median_W1_ppm=float(np.median(x)),connectivity_macro_W1_ppm=float(np.mean([np.mean(v) for v in groups.values()])))
    counts=collections.Counter('unknown' if not r['certified_environments'] else 'equal' if r['distinct_predicted_positions']==r['certified_environments'] else 'under' if r['distinct_predicted_positions']<r['certified_environments'] else 'over' for r in records)
    by_source={source:{mode:float(np.mean([r['errors'][mode] for r in records if r['source']==source])) for mode in ['raw','calibrated','physical']} for source in sorted({r['source'] for r in records})}
    return dict(records=len(records),connectivities=len({r['group'] for r in records}),metrics=metrics,by_source_W1_ppm=by_source,capacity_counts=dict(counts)),records


def main():
    p=argparse.ArgumentParser()
    for k in ['base','prepared','sidecar','checkpoints','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--registry')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda')
    try:
        base=Path(a.base);cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);side=torch.load(a.sidecar,map_location='cpu',weights_only=False);rows=side['theory_validation'];variants=side['variants'];owner=base/'hydrogen_analysis/site_supervision/train/best.pt';reference=torch.load(owner,map_location='cpu',weights_only=False);manifest=json.loads((Path(a.prepared)/'manifest.json').read_text())
        if manifest['H_checkpoint_sha256']!=sha(owner):raise ValueError('prepared_embedding_owner_mismatch')
        encoder=make_model(reference['args']).to(device).eval();encoder.load_state_dict(reference['state_dict'])
        for r in rows:
            s=r['smiles']
            if s not in cache['graphs']:
                g=graph(s);g.local_environment=torch.tensor(local_environment(s));g.frozen_embedding=encoder.structure_embedding(Batch.from_data_list([g]).to(device)).cpu();cache['graphs'][s]=g
        del encoder;reports={};checkpoints=json.loads(Path(a.checkpoints).read_text());registry=torch.load(a.registry,map_location='cpu',weights_only=False) if a.registry else None
        for name,path in checkpoints.items():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved.get('prepared_sha256')!=manifest['cache_sha256']:raise ValueError('prepared_checkpoint_mismatch')
            verify_cached_backbone(saved['state_dict'],reference['state_dict']);model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict']);report,records=evaluate(model,rows,cache['graphs'],variants,device,registry);report['checkpoint_sha256']=sha(path);reports[name]=report
            with (out/(name+'.jsonl')).open('x') as f:
                for r in records:f.write(json.dumps(r)+'\n')
            (out/'summary.json').write_text(json.dumps(dict(models=reports,sidecar_sha256=sha(a.sidecar),seconds=time.time()-start),indent=2));print(json.dumps(dict(model=name,report=report)),flush=True);del model;torch.cuda.empty_cache()
        (out/'EXIT_CODE').write_text('0\n')
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

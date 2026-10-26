"""功能概述：在原冻结理论val中按来源和观测完整性比较原始/固定校准目标。
calibrate仅应用既有train配对拟合；认证原子侧表优先，否则用来源正仿射同步
变换q及显式频窗。保留unknown/partial，不从首末峰建立频窗，不补峰或零填充。
evaluate同一次模型前向分别计算observation_loss，完整分布和单向部分匹配分开
汇总；不能把后者当作完整谱W1。该流程只做诊断，不参与epoch或检索模型选择。
main验证输入都是理论val和冻结GINE归属，读取h_count_sidecars目标及明确的权重
列表；每个模型写逐记录误差、来源/覆盖类型和结构宏均值。输入缓存、候选库、
源谱、模型均只读；输出独占，保存checkpoint/侧表SHA及EXIT_CODE，无外部服务。
evaluate默认单进程，支持交互Python及合成测试；命令行入口显式开启两个数据worker。
source_validation_rows从原prepared_coverage缓存读取全部来源val，并拒绝空cohort或
训练来源在val中整体缺失；条件表示缓存只提供图，不能将其仅含理论train的rows
冒充完整val。缺图用认证owner GINE批量编码，保留原H计数输入与观测口径。
"""
import argparse,collections,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from .h_conditions import ConditionsSpectra,local_environment
from .data import graph
from .h_labels import make_key
from .h_deep_context import verify_cached_backbone
from .observation import observation_loss
from .emission_model import make_model
from .train import collate
from .prepare import sha
from .h_observation_study import stamp_queries


def source_validation_rows(all_rows):
    rows=[r for r in all_rows if r['nucleus']=='1H' and r['domain']=='theory' and r['split']=='val']
    if not rows:raise ValueError('empty_theoretical_source_validation')
    expected={r['source'] for r in all_rows if r['nucleus']=='1H' and r['domain']=='theory' and r['split']=='train'}
    if expected-{r['source'] for r in rows}:raise ValueError('theoretical_validation_source_missing')
    return rows


def calibrate(row,variants,fits):
    if row['domain']!='theory' or row['split']!='val':raise ValueError('theory_source_diagnostic_requires_val')
    variant=variants.get(make_key(row))
    if variant is not None and row['observation']['kind']=='complete_atoms' and not row['observation']['windows']:return dict(row,q=variant['calibrated']['q'].copy()),'atom_sidecar'
    fit=fits.get(row['source']+'|1H',{})
    if fit.get('molecules',0)<20:return row,'unchanged_no_fit'
    scale=float(fit['affine_scale']);offset=float(fit['affine_offset_ppm'])
    if scale<=0 or not np.isfinite([scale,offset]).all():raise ValueError('invalid_fixed_source_calibration')
    observation=dict(row['observation'],windows=[[scale*float(l)+offset,scale*float(u)+offset] for l,u in row['observation']['windows']])
    return dict(row,q=(scale*np.asarray(row['q'],np.float64)+offset).astype(np.float32),observation=observation),'source_affine'


@torch.inference_mode()
def evaluate(model,raw,calibrated,graphs,device,workers=0):
    if not raw or len(raw)!=len(calibrated):raise ValueError('empty_or_mismatched_theoretical_source_cohort')
    if any(make_key(r)!=make_key(c) for r,c in zip(raw,calibrated)):raise ValueError('theoretical_source_alignment_identity_changed')
    def loader(rows):return DataLoader(ConditionsSpectra(rows,graphs,'1H',metadata={}),batch_size=128,num_workers=workers,collate_fn=collate)
    records=[];offset=0
    for b,bc in zip(loader(raw),loader(calibrated)):
        b=b.to(device);bc=bc.to(device);prediction=model(b);errors={mode:(observation_loss(prediction,batch)*10).cpu().tolist() for mode,batch in [('raw',b),('calibrated',bc)]}
        for j,r in enumerate(raw[offset:offset+b.num_graphs]):records.append(dict(source=r['source'],record_id=r['record_id'],smiles=r['smiles'],group=r['group'],coverage=r['observation']['kind'],explicit_windows=r['observation']['windows'],errors={mode:v[j] for mode,v in errors.items()}))
        offset+=b.num_graphs
    if offset!=len(raw):raise ValueError('source_diagnostic_row_count_changed')
    buckets=collections.defaultdict(list)
    for r in records:
        for key in ['all','source:'+r['source'],'source_coverage:'+r['source']+'|'+r['coverage']]:buckets[key].append(r)
    summary={}
    for key,rs in buckets.items():
        metrics={}
        for mode in ['raw','calibrated']:
            grouped=collections.defaultdict(list)
            for r in rs:grouped[r['group']].append(r['errors'][mode])
            metrics[mode]=dict(mean_observation_error_ppm=float(np.mean([r['errors'][mode] for r in rs])),connectivity_macro_ppm=float(np.mean([np.mean(xs) for xs in grouped.values()])))
        summary[key]=dict(records=len(rs),connectivities=len({r['group'] for r in rs}),metrics=metrics)
    return summary,records


def main():
    p=argparse.ArgumentParser()
    for name in ['base','prepared','sidecar','checkpoints','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--source-cache')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda')
    try:
        base=Path(a.base);cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);source_cache=Path(a.source_cache) if a.source_cache else base/'prepared_coverage/cache.pt';original=torch.load(source_cache,map_location='cpu',weights_only=False,mmap=True);rows=source_validation_rows(original['rows']);del original;side=torch.load(a.sidecar,map_location='cpu',weights_only=False);fits=json.loads((base/'prepared_coverage/domain_shift.json').read_text())['fits'];converted=[calibrate(r,side['variants'],fits) for r in rows];aligned=[r for r,mode in converted];owner=base/'hydrogen_analysis/site_supervision/train/best.pt';reference=torch.load(owner,map_location='cpu',weights_only=False);manifest=json.loads((Path(a.prepared)/'manifest.json').read_text());reports={}
        if manifest['H_checkpoint_sha256']!=sha(owner):raise ValueError('source_diagnostic_embedding_owner_changed')
        encoder=make_model(reference['args']).to(device).eval();encoder.load_state_dict(reference['state_dict']);missing=sorted({r['smiles'] for r in rows}-cache['graphs'].keys())
        with torch.no_grad():
            for offset in range(0,len(missing),256):
                names=missing[offset:offset+256];gs=[graph(s) for s in names];b=Batch.from_data_list(gs).to(device);h=encoder.structure_embedding(b).cpu();ptr=b.ptr.cpu().tolist()
                for j,s in enumerate(names):g=gs[j];g.local_environment=torch.tensor(local_environment(s));g.frozen_embedding=h[ptr[j]:ptr[j+1]].clone();cache['graphs'][s]=g
        del encoder;protocol=dict(source_cache=str(source_cache),source_cache_sha256=sha(source_cache),query_sha256=stamp_queries(rows),records=len(rows),source_counts=dict(collections.Counter(r['source'] for r in rows)),coverage_counts=dict(collections.Counter(r['source']+'|'+r['observation']['kind'] for r in rows)),added_graphs=len(missing),prepared_manifest=str(Path(a.prepared)/'manifest.json'),selection='diagnostic only, original theoretical val; no model selection');(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        for name,path in json.loads(Path(a.checkpoints).read_text()).items():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved.get('prepared_sha256')!=manifest['cache_sha256']:raise ValueError('source_diagnostic_prepared_mismatch')
            verify_cached_backbone(saved['state_dict'],reference['state_dict']);model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict']);summary,records=evaluate(model,rows,aligned,cache['graphs'],device,workers=2)
            with (out/(name+'.jsonl')).open('x') as f:
                for r in records:f.write(json.dumps(r)+'\n')
            reports[name]=dict(checkpoint_sha256=sha(path),buckets=summary);del model;torch.cuda.empty_cache();(out/'summary.json').write_text(json.dumps(dict(models=reports,calibration_counts=dict(collections.Counter(mode for r,mode in converted)),sidecar_sha256=sha(a.sidecar),metric='conditional quantile L1 for complete/window; directed observed-to-predicted error for unknown/partial; no model selection'),indent=2));print(name,flush=True)
        (out/'EXIT_CODE').write_text('0\n')
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

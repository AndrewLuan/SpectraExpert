"""功能概述：诊断冻结bank与重新发射之间的数值差异，不改变模型或均值。
main固定首个条件、相同候选batch和权重，对high重复与highest矩阵精度分别记录
分位数/显式发射支持的误差，以及最差候选身份。输出用于决定类别重建的数值
边界，不能用于改变检索测试或放宽排名控制；原bank/cache/权重只读。
依赖h_condinfer和h_uncertainty，输出独占JSON/EXIT_CODE，无外部API调用。
"""
import argparse
import json
from pathlib import Path
import torch
from .h_condinfer import candidate_batches
from .h_gapworkflow import resolve_checkpoint
from .h_uncertainty import quantile_classes
from .emission_model import make_model


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);device=torch.device('cuda');evaluation=Path(a.evaluation);variant=json.loads((evaluation/'frozen_selection.json').read_text())['selected_variant'];bank=torch.load(evaluation/variant/'bank.pt',map_location='cpu',weights_only=False,mmap=True);cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);checkpoint=resolve_checkpoint(Path(a.base),evaluation);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict']);key=sorted(bank['predictions'])[0];batches=candidate_batches(bank,cache['graphs'],device);offset=0;results=[]
    for b in batches:
        b.cond=torch.tensor(key[:4],device=device).reshape(1,4).expand(b.num_graphs,-1);b.condition_vector=torch.tensor(key[4:],device=device).reshape(1,-1).expand(b.num_graphs,-1);reference=bank['predictions'][key][offset:offset+b.num_graphs].to(device);preds=[];entries=[]
        for precision in ['high','high','highest']:
            torch.set_float32_matmul_precision(precision);d=model(b,details=True);_,support=quantile_classes(d,b,reference=reference);diff=(d['quantiles']-reference).abs();nearest=(support-reference).abs();i,j=divmod(int(diff.argmax()),diff.shape[1]);entries.append(dict(precision=precision,max_quantile_error_ppm=float(diff.max())*10,max_support_error_ppm=float(nearest.max())*10,off_2e5ppm=int((nearest>2e-6).sum()),worst_smiles=bank['smiles'][offset+i],reference_ppm=float(reference[i,j])*10,recomputed_ppm=float(d['quantiles'][i,j])*10));preds.append(d['quantiles'])
        entries.append(dict(high_repeat_max_error_ppm=float((preds[0]-preds[1]).abs().max())*10));results.append(dict(offset=offset,rows=b.num_graphs,variants=entries));offset+=b.num_graphs
    summary=dict(condition=list(key),checkpoint=str(checkpoint),batches=results,max_by_precision={str(i):max(r['variants'][i]['max_quantile_error_ppm'] for r in results) for i in range(3)},max_support_by_precision={str(i):max(r['variants'][i]['max_support_error_ppm'] for r in results) for i in range(3)},max_same_high_repeat_ppm=max(r['variants'][3]['high_repeat_max_error_ppm'] for r in results),GPU=torch.cuda.get_device_name());(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'EXIT_CODE').write_text('0\n')


if __name__=='__main__':main()

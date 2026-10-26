"""功能概述：在验证选定模型上，用train二甲基酰胺的组间距拟合全局收缩校准。
resolve_checkpoint按验证bank中的SHA在明确训练目录定位权重；不按test表现选模型。
main复用8组稳定train甲基标注，重新计算当前模型间距，每个结构总权重为1，以
alpha=1为岭先验拟合0/0.5/5三种强度，并保留identity校准。中心与核质量完全不改。
新checkpoint仅增加amide_gap_scale配置，参数张量不训练；拟合输入和所有系数保存。
输出按h_condeval的四模型目录约定组织，后续仍只用固定320-val选择，测试同库同集。
参考模型可以是历史验证优胜者，baseline可零残差包装；无测试拟合、无原始数据改写。
"""
import argparse
import collections
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .h_condmodel import ConditionSpecGNN
from .h_conditions import ConditionsSpectra
from .emission_model import make_model,weighted_quantiles
from .h_labels import make_key
from .train import collate
from .prepare import sha

VARIANTS=['gap_identity','gap_unshrunk','gap_ridge_mild','gap_ridge_strong']


def resolve_checkpoint(base,reference):
    mode=json.loads((reference/'frozen_selection.json').read_text())['selected_variant']
    if mode=='baseline':return base/'hydrogen_analysis/site_supervision/train/best.pt'
    bank=torch.load(reference/mode/'bank.pt',map_location='cpu',weights_only=False,mmap=True);expected=bank['checkpoint_sha256']
    if 'checkpoint_path' in bank:
        path=Path(bank['checkpoint_path'])
        if sha(path)==expected:return path
    root=base/'hydrogen_analysis/condition_response';candidates=[base/'hydrogen_analysis/site_supervision/train/best.pt']
    for directory in ['training_visible_mass','paired_training','gap_calibration','replay_training','anchored_training']:
        candidates.extend((root/directory).glob('*/train/best.pt'))
    for p in candidates:
        if sha(p)==expected:return p
    raise ValueError('reference_checkpoint_not_found')


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--reference-evaluation',required=True);p.add_argument('--out',required=True);a=p.parse_args();base=Path(a.base);reference=Path(a.reference_evaluation);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        status('waiting_reference');path=reference/'EXIT_CODE';deadline=time.monotonic()+10800
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('reference_timeout')
            time.sleep(10)
        if path.read_text().strip()!='0':raise RuntimeError('reference_failed')
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading')
        checkpoint=resolve_checkpoint(base,reference);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['args'].get('model') not in ['condition_atom','anchored_atom','deep_condition_atom']:
            model=ConditionSpecGNN('1H',saved['args']['width'],saved['args']['layers'],'control');missing,unexpected=model.load_state_dict(saved['state_dict'],strict=False)
            if unexpected or any(not k.startswith('condition_head.') for k in missing):raise ValueError('baseline_wrap_failed')
            saved=dict(saved,state_dict=model.state_dict(),args=dict(model='condition_atom',nucleus='1H',width=saved['args']['width'],layers=saved['args']['layers'],condition_mode='control'))
        model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict']);cache=torch.load(Path(a.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True);metadata=torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False)
        audit=[r for r in map(json.loads,(base/'hydrogen_analysis/condition_response/amide_gap_diagnosis/records.jsonl').read_text().splitlines()) if r['stable_methyl_labels']];lookup={(r['source'],r['record_id'],r['smiles']):r for r in cache['rows'] if r['split']=='train' and r['domain']=='experimental'};fit=[]
        for rec in audit:
            r=lookup[rec['source'],rec['record_id'],rec['smiles']];b=collate([ConditionsSpectra([r],cache['graphs'],'1H',metadata)[0]]).to(device);d=model(b,details=True);keep=torch.isin(d['owners'],torch.tensor(rec['parents'],device=device));q=weighted_quantiles(d['values'][keep],d['weights'][keep],torch.zeros(int(keep.sum()),dtype=torch.long,device=device),1,points=2)[0]*10
            fit.append(dict(smiles=r['smiles'],source=r['source'],record_id=r['record_id'],target_gap=rec['target_gap'],predicted_gap=float(q[1]-q[0])))
        counts=collections.Counter(r['smiles'] for r in fit);x=np.array([r['predicted_gap'] for r in fit]);y=np.array([r['target_gap'] for r in fit]);w=np.array([1/counts[r['smiles']] for r in fit]);models={};original=float(saved['args'].get('amide_gap_scale',1.))
        for variant,penalty in zip(VARIANTS,[None,0.,.5,5.]):
            alpha=1. if penalty is None else float(np.clip(((w*x*y).sum()+penalty)/((w*x*x).sum()+penalty),0.,2.))
            args=dict(saved['args'],amide_gap_scale=original*alpha);folder=out/variant/'train';folder.mkdir(parents=True)
            payload=dict(saved,args=args,calibration=dict(kind='train_only_NMe_gap_ridge',alpha_multiplier=alpha,ridge_penalty=penalty,reference_checkpoint_sha256=sha(checkpoint)));torch.save(payload,folder/'best.pt');(folder/'complete.json').write_text(json.dumps(dict(fit_only=True,checkpoint_sha256=sha(folder/'best.pt'))));(folder/'EXIT_CODE').write_text('0\n');models[variant]=payload['calibration']
        (out/'fit.json').write_text(json.dumps(dict(training_records=fit,structure_weighting='each structure total1',models=models,reference_checkpoint=str(checkpoint),reference_sha256=sha(checkpoint)),indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

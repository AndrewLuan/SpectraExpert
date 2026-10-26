"""功能概述：固定新骨干，用1条冻结样本或完整原表的独立train结构训练局部苯残差。
prepare_data只读取已验证原表清点中的train/val和原冻结样本，核验连接结构分区；
原候选/测试不加入训练。新增图复用原GINE权重编码；已有物理计数复用原bank，
缺少认证计数时保留physical_features的显式缺失掩码，不把谱峰数当物理计数。
main每组同seed、200步、32样本/步、54参数，按结构均匀抽样，不将重复文献当独立结构。
6条专用val按连接结构宏平均选epoch，包含零残差基线；原320检索val仍由评估入口
保留。条件、缺峰和积分辅助监督沿用ReplaySpectra/grouped_objective，未报告信号
不补零；DMF及所有test结构一律排除。结果记录新增支持/选型身份/权重hash。
"""
import argparse
import collections
import gc
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch
from .data import graph,physical_features,quantiles,identity
from .h_conditions import local_environment
from .h_metadata import raw_conditions,metadata_view,RECOVERED_MODE
from .h_labels import make_key
from .h_replaytrain import ReplaySpectra,grouped_objective
from .h_condmodel import training_details
from .h_deep_context import verify_cached_backbone
from .h_observation_study import stamp_queries
from .emission_model import make_model
from .observation import coverage
from .train import collate
from .prepare import sha


def prepare_data(base,prepared,supervision,reference,out,device):
    manifest=json.loads((prepared/'manifest.json').read_text());inventory=json.loads((supervision/'summary.json').read_text())
    if manifest['H_checkpoint_sha256']!=sha(base/'hydrogen_analysis/site_supervision/train/best.pt') or manifest['raw_recovery_sha256']!=inventory['raw_sha256']:raise ValueError('supervision_or_GINE_owner_mismatch')
    cache=torch.load(prepared/'cache.pt',map_location='cpu',weights_only=False,mmap=True)
    rid='NMRexp:50aec978c57a9570b867bce2f70f671e25808e258401486a867aab0b29a74dd2'
    original=next(r for r in cache['rows'] if r['domain']=='experimental' and r['record_id']==rid)
    bank=torch.load(reference/'bank.pt',map_location='cpu',weights_only=False,mmap=True)
    lookup={s:i for i,s in enumerate(bank['smiles'])};metadata=metadata_view(torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False),RECOVERED_MODE)
    labels=torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
    rows={'train':[original],'val':[]};missing_physics=[]
    if (supervision/'EXIT_CODE').read_text().strip()!='0':raise ValueError('inventory_incomplete')
    for split in ['train','val']:
        for item in map(json.loads,(supervision/('eligible_'+split+'.jsonl')).read_text().splitlines()):
            raw=item['record'];smi,group,actual=identity(item['canonical_smiles'])
            if group!=item['group'] or group!=identity(raw['smiles'])[1] or actual!=split or actual=='test' or smi=='CN(C)C=O':raise ValueError('nontraining_structure_in_supervision')
            physics=bank['physics'][lookup[smi]].numpy().copy() if smi in lookup else physical_features(smi)
            if smi not in lookup:missing_physics.append(smi)
            meta,_=raw_conditions(raw)
            row=dict(domain='experimental',source=raw['source'],record_id=str(raw['record_id']),nucleus='1H',smiles=smi,split=split,group=group,physics=physics,cond=meta['old_cond'],q=quantiles(raw['value']['x'],raw['value']['y']),observation=coverage(raw),measure='integral_weighted')
            total=sum(raw['value']['y']);expected=round(float(physics[0])*40)
            labels[make_key(row)]=dict(groups={},integral_consistent=bool(np.allclose(raw['value']['y'],np.round(raw['value']['y']),atol=.001) and abs(total-expected)<.001))
            metadata[make_key(row)]=meta;rows[split].append(row)
    if not rows['val'] or ({identity(r['smiles'])[1] for r in rows['train']} & {identity(r['smiles'])[1] for r in rows['val']}):raise ValueError('invalid_structural_validation')
    required=sorted({r['smiles'] for rs in rows.values() for r in rs});graphs={s:cache['graphs'][s].clone() if s in cache['graphs'] else graph(s) for s in required}
    del cache,bank;gc.collect()
    owner=torch.load(base/'hydrogen_analysis/site_supervision/train/best.pt',map_location='cpu',weights_only=False);encoder=make_model(owner['args']).to(device).eval();encoder.load_state_dict(owner['state_dict'])
    for s,g in graphs.items():
        if not hasattr(g,'local_environment'):g.local_environment=torch.tensor(local_environment(s))
        if not hasattr(g,'frozen_embedding'):
            with torch.no_grad():g.frozen_embedding=encoder.structure_embedding(Batch.from_data_list([g]).to(device)).cpu().clone()
    del encoder;torch.cuda.empty_cache()
    evidence=dict(train_records=len(rows['train']),train_structures=len({identity(r['smiles'])[1] for r in rows['train']}),validation_records=len(rows['val']),validation_structures=len({identity(r['smiles'])[1] for r in rows['val']}),train_sha256=stamp_queries(rows['train']),validation_sha256=stamp_queries(rows['val']),missing_certified_physics=sorted(set(missing_physics)),raw_sha256=json.loads((supervision/'summary.json').read_text())['raw_sha256'])
    torch.save(dict(rows=rows,graphs=graphs,metadata={make_key(r):metadata[make_key(r)] for rs in rows.values() for r in rs},labels={make_key(r):labels[make_key(r)] for rs in rows.values() for r in rs}),out/'data.pt')
    (out/'data_audit.json').write_text(json.dumps(evidence,indent=2));return rows,graphs,metadata,labels


@torch.no_grad()
def validate(model,rows,graphs,metadata,labels,device):
    model.eval();values=collections.defaultdict(list)
    for r in rows:
        b=collate([ReplaySpectra([r],graphs,'1H',metadata,annotations=labels,grouped=True)[0]]).to(device)
        loss=grouped_objective(training_details(model(b,details=True),b),b)['loss'];values[identity(r['smiles'])[1]].append(float(loss))
    return float(np.mean([np.mean(v) for v in values.values()]))


def main():
    p=argparse.ArgumentParser()
    for name in ['base','prepared','supervision','reference','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--arm',choices=['one','expanded'],required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.manual_seed(127);rng=np.random.default_rng(127);device=torch.device('cuda');base=Path(a.base);prepared=Path(a.prepared);reference=Path(a.reference);status('preparing')
        rows,graphs,metadata,labels=prepare_data(base,prepared,Path(a.supervision),reference,out,device)
        bank=torch.load(reference/'bank.pt',map_location='cpu',weights_only=False,mmap=True);init=Path(bank['checkpoint_path']);saved=torch.load(init,map_location='cpu',weights_only=False);owner=torch.load(base/'hydrogen_analysis/site_supervision/train/best.pt',map_location='cpu',weights_only=False);verify_cached_backbone(saved['state_dict'],owner['state_dict']);del bank,owner
        cfg=dict(saved['args'],model='solvent_local_atom',prediction_reuse_bank=str(reference/'bank.pt'));model=make_model(cfg).to(device);missing,unexpected=model.load_state_dict(saved['state_dict'],strict=False)
        if unexpected or any(not k.startswith('solvent_local_head.') for k in missing):raise ValueError('local_warmstart_mismatch')
        # 冻结结构表示只缓存本批train/val；不写回原始cache。
        with torch.no_grad():
            for s,g in graphs.items():g.final_context_embedding=model.structure_embedding(Batch.from_data_list([g]).to(device)).cpu().clone()
        train=rows['train'][:1] if a.arm=='one' else rows['train'];groups=collections.defaultdict(list)
        for i,r in enumerate(train):groups[identity(r['smiles'])[1]].append(i)
        choices=list(groups.values());optimizer=torch.optim.AdamW(model.solvent_local_head.parameters(),lr=.003,weight_decay=.01);dataset=ReplaySpectra(train,graphs,'1H',metadata,annotations=labels,grouped=True)
        best=validate(model,rows['val'],graphs,metadata,labels,device);baseline=best;best_step=0
        def save(step,score):torch.save(dict(state_dict=model.state_dict(),args=cfg,step=step,validation_objective=score,baseline_sha256=saved['baseline_sha256'],prepared_sha256=saved['prepared_sha256']),out/'best.pt')
        save(0,best)
        for step in range(1,201):
            indices=[int(rng.choice(choices[int(rng.integers(len(choices)))])) for _ in range(32)];b=collate([dataset[i] for i in indices]).to(device);model.train();optimizer.zero_grad(set_to_none=True);loss=grouped_objective(training_details(model(b,details=True),b),b)['loss']
            if not torch.isfinite(loss):raise ValueError('nonfinite_local_loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.solvent_local_head.parameters(),1.);optimizer.step()
            if step%10==0:
                score=validate(model,rows['val'],graphs,metadata,labels,device)
                if score<best:best=score;best_step=step;save(step,best)
                row=dict(step=step,train_loss=float(loss.detach()),validation_macro=score)
                with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                status('training',**row)
        report=dict(arm=a.arm,steps=200,samples_per_step=32,training_records=len(train),training_structures=len(groups),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),baseline_validation=baseline,best_validation=best,best_step=best_step,validation_sha256=stamp_queries(rows['val']),init_sha256=sha(init),checkpoint_sha256=sha(out/'best.pt'),extra_data_sha256=sha(out/'data.pt'),selection='original structural val partition, independent benzene/NMe2 records; macro over connectivity; no test choice',args=vars(a),source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')});(out/'complete.json').write_text(json.dumps(report,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

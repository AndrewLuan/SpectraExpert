"""功能概述：流式审计实验积分峰表并构建纯二维分子前向预测缓存，不将峰表冒充实测曲线。
主要职责：features 从 RDKit heavy-atom 图生成18维节点和5类键特征；convert
核验峰表、计算积分分布分位数、记录独立质量标记；main 全文件统计并按记录哈希采样。
关系与契约：读取 SpecGPT experimental_pretrain/data.jsonl，输出供 train.py 使用的
cache.pt、audit.json、excluded.jsonl。坐标和原子归属完全不作为模型输入。
结构非立体canonical SMILES决定split，相同连接结构不会跨集合；原数据永不修改。
写出路径必须不存在，失败不生成complete.json；hash与采样配置均保留以供复现。
节点索引9/10为CIP R/S，16/17为连接E/Z双键标记，避免丢失二维结构的立体信息。
features 会在输入RDKit分子上重算立体标记；其余训练标签及split保持原有定义。
"""
import argparse, hashlib, json, collections, time, pathlib
import numpy as np
import torch
from rdkit import Chem, RDLogger
try:
    from .source_io import normalize_record
except ImportError:
    from source_io import normalize_record
RDLogger.DisableLog('rdApp.*')


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def features(mol):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    n = mol.GetNumAtoms()
    x = np.zeros((n, 18), np.float32)
    adj = np.zeros((n,n,5), np.float32)
    counts = []
    for a in mol.GetAtoms():
        h = a.GetTotalNumHs(); counts.append(h)
        hy = a.GetHybridization()
        cip = a.GetProp('_CIPCode') if a.HasProp('_CIPCode') else ''
        x[a.GetIdx()] = [a.GetAtomicNum()/100,a.GetDegree()/4,h/4,a.GetFormalCharge()/4,
            a.GetIsAromatic(),a.IsInRing(),hy==Chem.HybridizationType.SP,
            hy==Chem.HybridizationType.SP2,hy==Chem.HybridizationType.SP3,
            cip=='R',cip=='S',
            a.GetMass()/200,*[a.IsInRingSize(k) for k in [3,4,5,6]],0,0]
    bt = {Chem.BondType.SINGLE:0,Chem.BondType.DOUBLE:1,Chem.BondType.TRIPLE:2,Chem.BondType.AROMATIC:3}
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); k=bt.get(b.GetBondType(),4)
        adj[i,j,k]=adj[j,i,k]=1
        stereo=b.GetStereo()
        if stereo==Chem.BondStereo.STEREOE: x[[i,j],16]=1
        elif stereo==Chem.BondStereo.STEREOZ: x[[i,j],17]=1
    return x,adj,np.asarray(counts,np.int64)


def convert(r, max_atoms=96):
    if r.get('nucleus')!='1H': return None,'not_1H'
    if r.get('representation')!='integral_peak_spectrum': return None,'not_integral_peak_spectrum'
    xs=np.asarray(r['value']['x'],float); ys=np.asarray(r['value']['y'],float)
    if not len(xs) or xs.shape!=ys.shape: return None,'invalid_arrays'
    if not np.isfinite(xs).all() or not np.isfinite(ys).all() or (ys<=0).any(): return None,'invalid_numeric'
    if (xs < -5).any() or (xs > 25).any(): return None,'outside_supported_shift_range'
    mol=Chem.MolFromSmiles(r['smiles'])
    if mol is None: return None,'invalid_smiles'
    mol=Chem.RemoveHs(mol)
    if not 1<=mol.GetNumAtoms()<=max_atoms:return None,'atom_count_limit'
    smi=Chem.MolToSmiles(mol,isomericSmiles=True)
    # 统一重解析canonical顺序，保证特征与氢槽的稳定性。
    mol=Chem.MolFromSmiles(smi)
    identity=Chem.MolToSmiles(mol,isomericSmiles=False)
    group=digest(identity)
    split_value=int(group[:8],16)%100
    split='train' if split_value<80 else 'val' if split_value<90 else 'test'
    x,adj,h=features(mol)
    if h.sum()==0 or h.max()>4:return None,'unsupported_hydrogens'
    idx=np.argsort(xs); xs=xs[idx];ys=ys[idx]
    cum=ys.cumsum()/ys.sum(); q=(np.arange(128)+.5)/128
    quant=xs[np.searchsorted(cum,q)]
    flags=list(r.get('normalization_flags',[]))
    if abs(float(ys.sum())-int(h.sum()))>0.25:flags.append('integral_vs_structure_H_mismatch')
    if '.' in smi:flags.append('multicomponent_structure')
    if any(a.GetAtomicNum() not in [6,7,8,9,15,16,17,35,53] for a in mol.GetAtoms()): flags.append('uncommon_element')
    if not r.get('frequency'): flags.append('missing_frequency')
    if np.max(xs)>15 or np.min(xs)<-1:flags.append('unusual_shift')
    solvent=str(r.get('solvent') or '').lower()
    freq=r.get('frequency')
    try: freq=float(freq) if freq else 0.
    except (ValueError,TypeError):freq=0.;flags.append('unparseable_frequency')
    condition=np.array([freq/600.,float('water' in solvent or 'd2o' in solvent),float('cdcl3' in solvent),float('dmso' in solvent)],np.float32)
    return dict(x=x,adj=adj,hcount=h,cond=condition,target=quant.astype(np.float32),
        smiles=smi,group=group,split=split,record_id=r.get('record_id'),
        source=r.get('source'),source_locator=r.get('source_locator'),flags=sorted(set(flags)),normalization_evidence=r.get('normalization_evidence'),
        integral=float(ys.sum()),protons=int(h.sum()),xs=xs.tolist(),weights=ys.tolist(),
        annotations=r.get('annotations',[])),None


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--sample-mod',type=int,default=48);p.add_argument('--max-atoms',type=int,default=96)
    args=p.parse_args();out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=False)
    tally=collections.Counter();source=collections.Counter();reasons=collections.Counter();data=[];seen=set();sha=hashlib.sha256();start=time.time()
    excluded=(out/'excluded.jsonl').open('w')
    with open(args.input,'rb') as f:
        for lineno,line in enumerate(f,1):
            sha.update(line)
            try:r=json.loads(line)
            except Exception: tally['invalid_json']+=1;continue
            tally['records']+=1;source[str(r.get('source'))]+=1
            nucleus=r.get('nucleus') or (r.get('experimental_conditions') or {}).get('nucleus')
            tally['nucleus:'+str(nucleus)]+=1
            tally['representation:'+str(r.get('representation'))]+=1
            if lineno%250000==0:print(json.dumps({'lines':lineno,'selected':len(data),'seconds':time.time()-start}),flush=True)
            if nucleus!='1H':continue
            rid=str(r.get('record_id') or digest(line.decode()))
            if int(digest(rid)[:16],16)%args.sample_mod:continue
            tally['sampled']+=1
            try:
                normalized,reason=normalize_record(r)
                if normalized is not None:
                    normalized['record_id']=rid
                    item,reason=convert(normalized,args.max_atoms)
                else:item=None
            except Exception as e:item=None;reason='conversion_'+type(e).__name__
            if item is None:
                reasons[reason]+=1;excluded.write(json.dumps({'record_id':rid,'line':lineno,'reason':reason})+'\n');continue
            obs=(item['group'],tuple(item['xs']),tuple(item['weights']),tuple(item['cond']))
            if obs in seen:reasons['duplicate_observation']+=1;continue
            seen.add(obs);data.append(item);tally['split:'+item['split']]+=1
            if lineno%100000==0: print(json.dumps({'lines':lineno,'selected':len(data),'seconds':time.time()-start}),flush=True)
    excluded.close()
    groups={s:{d['group'] for d in data if d['split']==s} for s in ['train','val','test']}
    assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
    audit=dict(input=args.input,input_sha256=sha.hexdigest(),args=vars(args),counts=dict(tally),sources=dict(source),exclusions=dict(reasons),
        flags=dict(collections.Counter(f for d in data for f in d['flags'])),
        split_method='sha256(nonstereo canonical SMILES) modulo100:80/10/10',
        target='integral_weighted_chemical_shift_distribution_NOT_sampled_spectrum',feature_dim=18,normalization='source_io_v1',seconds=time.time()-start)
    torch.save(data,out/'cache.pt');(out/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    (out/'complete.json').write_text(json.dumps({'records':len(data),'cache_bytes':(out/'cache.pt').stat().st_size}))
    print(json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

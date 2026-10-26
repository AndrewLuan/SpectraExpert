"""功能概述：为扩展分组预训练合并多源理论监督，剔除结构分区/分组冲突。
augment_theory_rows补充已认证且不在冻结抽样中的QM9训练记录，保持原计数输入
已知/缺失状态；不借用val/test标签补训练。combine只保留扩展registry有H分区的train/val键；跨源重复键拒绝覆盖，独立理论val
按原键排序。main保存新侧表、来源计数、SHA与退出码；输入只读，test目标拒绝。
由h_count_curriculum读取，不改变原候选/测试/物理特征；无拟合和外部服务。
"""
import argparse,collections,json
from pathlib import Path
import torch
import numpy as np
from .data import identity,physical_features
from .h_labels import make_key
from .prepare import sha


def combine(sides,registry):
    variants={};val={};excluded=collections.Counter()
    for side in sides:
        for key,v in side['variants'].items():
            if identity(key[3])[2]=='test':raise ValueError('test_target_in_sidecar')
            if not registry.get(key[3],{}).get('groups'):excluded[key[1]]+=1;continue
            if key in variants:raise ValueError('duplicate_theory_target_key')
            variants[key]=v
        for r in side['theory_validation']:
            if r['split']!='val' or identity(r['smiles'])[2]!='val':raise ValueError('theory_validation_split_mismatch')
            if make_key(r) in variants:val[make_key(r)]=r
    return dict(variants=variants,theory_validation=[val[k] for k in sorted(val)]),dict(excluded)



def augment_theory_rows(rows,side,features):
    result=list(rows);seen={make_key(r) for r in rows};added=0
    for key,v in sorted(side['variants'].items()):
        if key[1]!='QM9-NMR' or key in seen:continue
        smi,group,split=identity(key[3])
        if split!='train':continue
        result.append(dict(domain='theory',source=key[1],record_id=key[2],smiles=smi,group=group,split=split,nucleus='1H',cond=np.zeros(4,np.float32),q=v['raw']['q'].copy(),physics=physical_features(smi,features.get(smi,{}).get('counts',{})),measure='atom_equal',observation=dict(kind='complete_atoms',windows=[],evidence='complete source atom assignment')));seen.add(key);added+=1
    return result,added


def main():
    p=argparse.ArgumentParser();p.add_argument('--sidecars',nargs='+',required=True);p.add_argument('--registry',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    try:
        registry=torch.load(a.registry,map_location='cpu',weights_only=False);side,excluded=combine([torch.load(f,map_location='cpu',weights_only=False) for f in a.sidecars],registry);torch.save(side,out/'variants.pt');counts=collections.Counter((k[1],identity(k[3])[2]) for k in side['variants']);report=dict(counts={'|'.join(k):v for k,v in counts.items()},excluded_unknown_or_conflicting=excluded,theory_validation=len(side['theory_validation']),source_sha256={str(f):sha(f) for f in a.sidecars+[a.registry]},artifact_sha256=sha(out/'variants.pt'));(out/'manifest.json').write_text(json.dumps(report,indent=2));(out/'EXIT_CODE').write_text('0\n');print(json.dumps(report),flush=True)
    except Exception:(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

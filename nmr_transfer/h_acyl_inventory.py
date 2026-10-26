"""功能概述：清点苯中酰胺的酰基类别，寻找可跨结构学习的甲酰胺支持。
main按字节流核验完整原表SHA，仅对明确苯名称行解析；normalize_record后检查
连接结构hash分区。只保存train/val甲酰胺记录；test仅计结构类别数，不保存峰值，
不改变原冻结训练集。新支持需要作为另一个数据增量消融才能进入训练。
--selected-input处理已规范化的eligible_train/val.jsonl，用于本地审计上轮支持；
summarize按类别统计记录和独立连接结构，不将重复文献当独立样本。
所有输出独占；原表hash不符时不发布eligible文件，保留错误和退出码用于排查。
依赖h_acyl纯结构分类、data.identity及nmr_forward.source_io，不调用外部服务。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from .h_acyl import acyl_environment,ACYL_NAMES
from .data import identity


def summarize(items):
    groups=collections.defaultdict(set);counts=collections.Counter()
    for item in items:
        s=item['canonical_smiles'];env=acyl_environment(s)
        for tag in ACYL_NAMES[1:7]:
            if env[:,ACYL_NAMES.index(tag)].any():
                key=(item['split'],tag);counts[key]+=1;groups[key].add(item['group'])
    return [dict(split=k[0],acyl=k[1],records=counts[k],connectivities=len(v)) for k,v in sorted(groups.items())]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,nargs='+');p.add_argument('--out',required=True);p.add_argument('--selected-input',action='store_true');p.add_argument('--expected-sha');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    try:
        items=[];export={'train':[],'val':[]};hashes={};errors=collections.Counter()
        for name in a.input:
            digest=hashlib.sha256()
            with open(name,'rb') as f:
                for i,line in enumerate(f,1):
                    digest.update(line)
                    if a.selected_input:item=json.loads(line)
                    else:
                        lower=line.lower()
                        if not any(t in lower for t in [b'c6d6',b'benzene',b'c6h6']):continue
                        from nmr_forward.source_io import normalize_record
                        from .h_metadata import raw_conditions
                        raw=json.loads(line);r,reason=normalize_record(raw)
                        if reason:errors[str(reason)]+=1;continue
                        if r['nucleus']!='1H' or raw_conditions(r)[0]['family']!='benzene':continue
                        try:s,g,split=identity(r['smiles'])
                        except ValueError as exc:errors[str(exc)]+=1;continue
                        item=dict(canonical_smiles=s,group=g,split=split,raw_line=i)
                        env=acyl_environment(s)
                        if not env[:,0].any():continue
                        if split!='test' and env[:,ACYL_NAMES.index('formyl')].any():export[split].append(dict(item,record=r))
                    # 汇总只保留结构，不把test峰写入内存或输出。
                    items.append({k:item[k] for k in ['canonical_smiles','group','split']})
                    if i%100000==0:(out/'status.json').write_text(json.dumps(dict(stage='scanning',line=i,amide_records=len(items))))
            hashes[str(name)]=digest.hexdigest()
        if not a.selected_input and (len(hashes)!=1 or next(iter(hashes.values()))!=a.expected_sha):raise ValueError('raw_source_sha_mismatch')
        for split,rs in export.items():(out/('eligible_'+split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
        (out/'summary.json').write_text(json.dumps(dict(classes=summarize(items),source_sha256=hashes,errors=dict(errors),exported_train_records=len(export['train']),exported_val_records=len(export['val']),training_data_changed=False,scope='structure support audit; no test labels exported; selected-input contains only previously selected NMe2 amides' if a.selected_input else 'all explicit pure benzene amides, raw source hash checked'),indent=2));(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:(out/'status.json').write_text(json.dumps(dict(stage='failed',error=repr(exc))));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

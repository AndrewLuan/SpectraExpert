"""功能概述：固定独立的双核val配对，用于区分¹H误差宽度和融合权重的作用。
validation_pairs严格复用joint.py的同结构/每核record-id+source最小hash规则，
仅把split从test换为val，按SMILES hash取前320；不按条件缓存可用性或误差筛选。
main保存两核原始观测、身份和所需新增预测条件；候选顺序仍为原138783个结构。
既有792条test、候选预测及C逐查询控制均不修改。后续融合参数只能由本val集合
选择；新增条件需独立预测，不能把查询条件填作训练/测试候选自己的实验条件。
输出独占pairs.pt、protocol.json和退出码；源缓存/元数据均只读，无外部API。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import torch
from .h_latent_workflow import load_reference
from .h_latent_score import bank_key
from .h_diagnose import cond_key
from .prepare import sha


def validation_pairs(rows,maximum=320):
    buckets={}
    for r in rows:
        if r['split']!='val' or r['domain']!='experimental' or r['physics'][0]<=0 or r['physics'][1]<=0:continue
        group=buckets.setdefault(r['smiles'],{});key=hashlib.sha256((r['record_id']+r['source']).encode()).hexdigest();n=r['nucleus']
        if n not in group or key<group[n][0]:group[n]=(key,r)
    pairs=[(s,{n:v[1] for n,v in rs.items()}) for s,rs in buckets.items() if '1H' in rs and '13C' in rs];pairs.sort(key=lambda item:hashlib.sha256(item[0].encode()).hexdigest());return pairs[:maximum]


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--reference-latent',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4);base=Path(a.base);ref=Path(a.reference_latent);evaluation=Path(json.loads((ref/'protocol.json').read_text())['reference_evaluation']);bank,_,_,_,_=load_reference(base,evaluation);metadata=torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False);carbon=torch.load(base/'hydrogen_analysis/observation_comparison/joint/carbon_bank.pt',map_location='cpu',weights_only=False,mmap=True);(out/'status.json').write_text(json.dumps(dict(stage='loading_original_val')));cache=torch.load(base/'prepared_coverage/cache.pt',map_location='cpu',weights_only=False,mmap=True);pairs=validation_pairs(cache['rows']);del cache;smiles=[bank['smiles'][i] for i in bank['joint_indices']]
        if smiles!=carbon['smiles']:raise ValueError('candidate_order_mismatch')
        lookup=set(smiles)
        if any(s not in lookup for s,_ in pairs):raise ValueError('validation_candidate_missing')
        hmissing=sorted({bank_key(bank,rs['1H'],metadata) for _,rs in pairs}-bank['predictions'].keys());cmissing=sorted({cond_key(rs['13C']) for _,rs in pairs}-carbon['predictions'].keys());ids=[dict(smiles=s,record_ids={n:r['record_id'] for n,r in rs.items()},sources={n:r['source'] for n,r in rs.items()}) for s,rs in pairs];torch.save(pairs,out/'pairs.pt');protocol=dict(queries=len(pairs),candidates=len(smiles),validation_pairing='exact joint.py record hash rule in val partition, first320 SMILES hash; no condition/error filtering',identities=ids,query_sha256=hashlib.sha256(json.dumps(ids,sort_keys=True).encode()).hexdigest(),candidate_order_sha256=hashlib.sha256('\n'.join(smiles).encode()).hexdigest(),H_condition_addons=hmissing,C_condition_addons=cmissing,H_sources=dict(collections.Counter(rs['1H']['source'] for _,rs in pairs)),C_sources=dict(collections.Counter(rs['13C']['source'] for _,rs in pairs)),reference_latent=str(ref),H_checkpoint_sha256=bank['checkpoint_sha256'],C_checkpoint_sha256=carbon['checkpoint_sha256'],source_sha256=sha(__file__));(out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'status.json').write_text(json.dumps(dict(stage='complete')));(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:(out/'status.json').write_text(json.dumps(dict(stage='failed',error=repr(exc))));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

"""功能概述：为整体检索训练/验证补充缺失条件的结构预测，保留旧 bank 字节。

load_bank 以基准 SHA/检查点/候选顺序验证增量，只允许添加旧 bank 没有的条件；
main 从冻结 cohort 条件及训练支持溶剂原型列出所需键，按同一模型、同一候选结构
预测完整/非交换谱和同次类别标签。额外 verify 的条件可预生成，观测值与排名不读。
预测只接收候选结构与查询条件，目标谱置零；不读取候选自身实验条件或标注。
新条件另存 extension.pt，不覆盖旧 bank、原谱、模型或候选；无外部服务调用。
"""
import argparse
import json
from pathlib import Path

import torch

from .emission_model import make_model, weighted_quantiles
from .h_condinfer import candidate_batches, condition_key
from .h_conditions import CONDITION_DIM
from .h_latent_score import prototype_keys
from .h_uncertainty import quantile_classes
from .prepare import sha


def load_bank(path, extension=None):
    bank = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if extension:
        extra = torch.load(extension,map_location='cpu',weights_only=False,mmap=True)
        if extra['base_sha256']!=sha(path) or extra['checkpoint_sha256']!=bank['checkpoint_sha256']:
            raise ValueError('global_bank_extension_provenance_mismatch')
        for field in ('predictions','fixed_predictions','candidate_quantile_classes'):
            if set(extra[field])&set(bank[field]):
                raise ValueError('global_bank_extension_overwrites_original_condition')
            bank[field] = dict(bank[field])
            bank[field].update(extra[field])
    return bank


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    for name in ('bank','cohorts','prepared','out'):
        p.add_argument('--'+name,required=True)
    args = p.parse_args(); out = Path(args.out); out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):
        (out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    try:
        torch.set_num_threads(4); torch.set_float32_matmul_precision('high'); device=torch.device('cuda')
        bank = load_bank(args.bank)
        payload = torch.load(Path(args.cohorts)/'cohorts.pt',map_location='cpu',weights_only=False)
        keys = set()
        for rows in payload['cohorts'].values():
            keys.update(condition_key(r,payload['metadata'],payload['key_mode']) for r in rows)
        keys.update(prototype_keys(bank).values())
        keys = sorted(keys-set(bank['predictions']))
        status('loading',missing_conditions=len(keys))
        extra = dict(base_sha256=sha(args.bank),checkpoint_sha256=bank['checkpoint_sha256'],
                     predictions={},fixed_predictions={},candidate_quantile_classes={})
        cache=torch.load(Path(args.prepared)/'cache.pt',map_location='cpu',weights_only=False,mmap=True)
        if sha(bank['checkpoint_path'])!=bank['checkpoint_sha256']:
            raise ValueError('global_bank_checkpoint_changed')
        saved=torch.load(bank['checkpoint_path'],map_location='cpu',weights_only=False)
        model=make_model(saved['args']).to(device).eval(); model.load_state_dict(saved['state_dict'])
        batches=candidate_batches(bank,cache['graphs'],device)
        for b in batches:
            if getattr(model,'conditioned_messages',False):
                b.unconditioned_context_embedding=model.unconditioned_embedding(b).detach()
            elif getattr(model,'context_depth',0)>0:
                b.final_context_embedding=model.structure_embedding(b).detach()
        for j,key in enumerate(keys):
            status('predicting',condition=j+1,total=len(keys)); qs=[]; fs=[]; ts=[]
            for b in batches:
                b.cond=torch.tensor(key[:4],device=device).reshape(1,4).expand(b.num_graphs,-1)
                b.condition_vector=torch.tensor(key[4:],device=device).reshape(1,CONDITION_DIM).expand(b.num_graphs,-1)
                d=model(b,details=True); qs.append(d['quantiles'].cpu()); tags,checked=quantile_classes(d,b)
                if not torch.equal(checked,d['quantiles']):raise ValueError('global_same_forward_quantile_drift')
                ts.append(tags.cpu())
                if key[1]>0:
                    keep=~d['exchangeable']; fs.append(weighted_quantiles(d['values'][keep],d['weights'][keep],d['groups'][keep],b.num_graphs).cpu())
            extra['predictions'][key]=torch.cat(qs); extra['candidate_quantile_classes'][key]=torch.cat(ts)
            if fs:extra['fixed_predictions'][key]=torch.cat(fs)
        torch.save(extra,out/'extension.pt')
        (out/'summary.json').write_text(json.dumps(dict(conditions=[list(k) for k in keys],base_sha256=extra['base_sha256'],
            extension_sha256=sha(out/'extension.pt'),checkpoint_sha256=bank['checkpoint_sha256'],external_api_calls=0),indent=2))
        status('complete'); (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc)); (out/'EXIT_CODE').write_text('1\n'); raise


if __name__=='__main__':
    main()

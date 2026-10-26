"""功能概述：在固定¹H候选预测bank上检查近似谱形混叠，而非只检查精确碰撞。
main对旧1000个真结构的预测谱统计0.05/0.1/0.2/0.5ppm W1邻居数，并加入固定
0.1ppm逐预测峰共享抖动做模型自查询；噪声不拟合test标签，不参与模型或评分选择。
这仅测量当前模型表示的噪声容忍度，不能据此声称真实NMR的物理可辨识极限。
输入既有bank.pt，输出新JSON；只读模型预测，不读真实观测谱或修改任何权重。
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    bank=torch.load(a.bank,map_location='cpu',weights_only=False,mmap=True);device=torch.device('cuda')
    torch.set_num_threads(4);ids=bank['single_indices'];lookup={bank['smiles'][int(i)]:j for j,i in enumerate(ids)}
    predictions={c:q[ids].to(device)*10 for c,q in bank['predictions'].items()};rows=[]
    with torch.inference_mode():
        for r in bank['test']:
            c=tuple([0.,*np.asarray(r['cond'])[1:].tolist()]);q=predictions[c];true=lookup[r['smiles']]
            distance=(q-q[true]).abs().mean(1)
            values,inverse=np.unique(q[true].cpu().numpy(),return_inverse=True)
            rng=np.random.default_rng(int(hashlib.sha256(r['smiles'].encode()).hexdigest()[:8],16))
            noisy=np.sort((values+rng.normal(0,.1,len(values)))[inverse]);error=(q-torch.tensor(noisy,device=device)).abs().mean(1)
            rows.append(dict(smiles=r['smiles'],neighbors={str(t):int((distance<=t).sum()) for t in [.05,.1,.2,.5]},
                noisy_self_rank=int((error<=error[true]+1e-6).sum())))
    report=dict(queries=len(rows),candidate_count=len(ids),thresholds_ppm={str(t):dict(median=float(np.median([r['neighbors'][str(t)] for r in rows])),
        p90=float(np.quantile([r['neighbors'][str(t)] for r in rows],.9))) for t in [.05,.1,.2,.5]},
        noisy_self_query=dict(jitter_sigma_ppm=.1,**{'top'+str(k):sum(r['noisy_self_rank']<=k for r in rows)/len(rows) for k in [1,5,10]}),
        scope='model-prediction self-query diagnostic only; no observed test spectrum used or physical identifiability limit claimed')
    with Path(a.out).open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

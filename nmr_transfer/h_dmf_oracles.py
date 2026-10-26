"""功能概述：在相同候选库中，用明确标记的事后反事实分解DMF失败。
methyl_oracle只修改DMF真候选的两个甲基中心/间距，保留核质量和未报告甲酰预测；
不是可部署评分器，不训练权重、不把测试拟合参数写回任何生产配置。
main另外计算公共位移oracle和“只保留已报告位置”的乐观密度oracle，区分均值/
间距偏差与未观测峰归一化影响。所有竞争候选的评分逐位保持不变。
另以同次分位成分的结构tag去除甲酰CH，分别固定预测甲基位移和校正甲基位移，
形成观测状态×甲基位移的2×2诊断；reported_only_density同时使用真峰位，单独
比较它不能把所有收益归因于缺峰。结构报告状态来自h_formyl_reporting，不按ppm阈值
判断甲酰峰，不更改完整物理预测，不将测试oracle用于训练、选模或评分配置。
两条非单核固定测试的DMF记录只作辅助诊断；不加入1000/792主评测，也不推断实测频窗。
--variant可显式诊断已冻结消融中的单个分支，只控制读取哪个bank，
不改变正式选择文件或排名，不把诊断结果用于生产参数。
输出独占候选/模型hash、原始条件/峰高、逐oracle排名和退出码；源数据只读。
依赖h_condinfer冻结评分、h_conditions条件键及likelihood归一化密度，无外部API。
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import quantiles
from .h_condinfer import condition_key,query_kernels,scoring_inputs
from .h_observation_finalize import query_score
from .h_observed import cluster_lines
from .h_labels import make_key
from .h_observation_study import entry,uniform_peaks
from .h_eval import LazyFP
from .likelihood import density
from .prepare import sha
from .h_formyl_reporting import counterfactual_quantiles,candidate_families


def methyl_oracle(prediction,target,center=False,gap=False):
    x=prediction.clone()*10;original=x.clone();peaks,_=cluster_lines(x[x<6].cpu().numpy(),.01)
    if len(peaks)!=2:raise ValueError('expected_two_DMF_methyl_components')
    before=float(peaks.mean());difference=float(peaks[1]-peaks[0]);desired_center=float(np.mean(target)) if center else before;desired_gap=float(target[1]-target[0]) if gap else difference
    for i,p in enumerate(peaks):x[torch.isclose(original,torch.tensor(p,device=x.device,dtype=x.dtype),atol=.01)]=desired_center+(-.5 if i==0 else .5)*desired_gap
    return x/10


def true_kernels(prediction):
    p=prediction.reshape(1,-1);mass=density(p,'1H',.2);distinct=density(uniform_peaks(p),'1H',.2);return dict(pred=p,mass=mass,distinct=distinct,fixed_mass=mass,fixed_distinct=distinct)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--prepared',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True);p.add_argument('--variant');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4);device=torch.device('cuda');base=Path(a.base);evaluation=Path(a.evaluation);variant=a.variant or json.loads((evaluation/'frozen_selection.json').read_text())['selected_variant'];bank=torch.load(evaluation/variant/'bank.pt',map_location='cpu',weights_only=False,mmap=True);metadata=torch.load(Path(a.prepared)/'metadata.pt',map_location='cpu',weights_only=False);frozen,prior,config,priors,raw=scoring_inputs(base);indices=bank['single_indices'];smiles=[bank['smiles'][i] for i in indices];lookup={s:i for i,s in enumerate(smiles)};true=lookup['CN(C)C=O'];bank_true=int(indices[true]);counts=bank['physics'][indices,0].to(device)*40;fps=LazyFP();template=next(r for r in bank['test'] if r['smiles']=='CN(C)C=O');main_ids={r['record_id'] for r in bank['test']};joint_ids={r['record_id'] for r in bank['paired']};records=[]
        with torch.inference_mode():
            for rid in sorted(raw):
                rr=raw[rid]
                if 'HMDB0001888' not in rid:continue
                meta=metadata['experimental','HMDB',rid,'CN(C)C=O'];q=dict(template,record_id=rid,cond=meta['old_cond'],q=quantiles(rr['value']['x']),observation=dict(kind='unknown',windows=[]));key=condition_key(q,metadata,bank.get('condition_key_mode',bank['condition_mode']));kernels=query_kernels(bank,indices,key,device,True);original=kernels['pred'][true];baseline=query_score(q,kernels,counts,prior,config,frozen,priors,raw);low=np.asarray(rr['value']['x'])<6;xs=np.asarray(rr['value']['x'])[low];ys=np.asarray(rr['value']['y'])[low];unique=np.unique(xs);height=np.array([ys[xs==x].max() for x in unique]);target=sorted(unique[np.argsort(height)[-2:]].tolist());predicted,_=cluster_lines(original[original*10<6].cpu().numpy()*10,.01);before=entry(baseline,q,lookup,smiles,fps)
                def assess(p,**details):
                    replacement=query_score(q,true_kernels(p),counts[true:true+1],prior,config,frozen,priors,raw)[0];scores=baseline.clone();scores[true]=replacement;return dict(**details,**entry(scores,q,lookup,smiles,fps),true_score=float(replacement))
                common=[]
                for shift in np.linspace(-1.,1.,81):common.append((float(query_score(q,true_kernels(original+float(shift)/10),counts[true:true+1],prior,config,frozen,priors,raw)[0]),float(shift)))
                _,best_shift=min(common);oracles=dict(common_offset=assess(original+best_shift/10,shift_ppm=best_shift),methyl_center=assess(methyl_oracle(original,target,center=True)),methyl_gap=assess(methyl_oracle(original,target,gap=True)),methyl_both=assess(methyl_oracle(original,target,center=True,gap=True)));centers,_=cluster_lines(rr['value']['x'],.08);echo=torch.tensor(quantiles(centers)/10,device=device);oracles['reported_only_density']=assess(echo,scope='optimistic report-only density, not a complete physical spectrum or inferred sweep window')
                tags=bank['candidate_quantile_classes'][key][bank_true:bank_true+1].to(device);families=candidate_families(['CN(C)C=O'])
                def unreported_formyl(p):
                    altered,active=counterfactual_quantiles(p.reshape(1,-1),tags,families)
                    if not bool(active[0]):raise ValueError('DMF_structural_formyl_tag_required')
                    return altered[0]
                oracles['unreported_formyl_only']=assess(unreported_formyl(original),scope='true candidate only; original methyl positions retained; structural formyl tag removed from reported density')
                oracles['methyl_both_unreported_formyl']=assess(unreported_formyl(methyl_oracle(original,target,center=True,gap=True)),scope='true candidate only; exact methyl-position oracle plus same structural formyl reporting state')
                records.append(dict(record_id=rid,solvent=meta['solvent'],pH=meta['pH'],raw_x=rr['value']['x'],raw_peak_heights=rr['value']['y'],main_single_test=rid in main_ids,main_joint_test=rid in joint_ids,observed_methyl_diagnostic=target,predicted_methyls=predicted.tolist(),center_bias_ppm=float(np.mean(predicted)-np.mean(target)),gap_bias_ppm=float(np.diff(predicted)[0]-(target[1]-target[0])),baseline=before,oracles=oracles,scope='test-only diagnostic; all competitor scores fixed; no parameter used for training/selection; actual coverage remains unknown'))
        (out/'summary.json').write_text(json.dumps(dict(candidates=len(smiles),checkpoint_sha256=bank['checkpoint_sha256'],evaluation=str(evaluation),variant=variant,cases=records,source_sha256=sha(__file__),scope='three predetermined DMF conditions; only benzene is in fixed1000 single-test,Water in792joint; CDCl3auxiliary; no pooled benchmark change'),indent=2));(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:(out/'error.json').write_text(json.dumps(dict(error=repr(exc))));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

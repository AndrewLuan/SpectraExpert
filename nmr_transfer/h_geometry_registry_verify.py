"""功能概述：将生成构象分区经过冲突排除及模型布局核验后接入独立训练侧表。
check_layout分别调用local/physical的真实equivalence_layout，核验H质量、canonical
节点序、父节点槽与最大组数。main等待原生成任务成功，保留基准H分组与C计数；
此前存在源间/源内冲突且基准仍未知H的结构继续未知，生成稳定性不能覆盖冲突。
不兼容的新分区仅恢复基准项或未知，不重命名候选、不删除候选、不改谱/特征。
输入仅结构registry、已冻结冲突名单及原生成状态，输出独占registry、排除明细、
来源SHA、状态及EXIT_CODE。由生成覆盖四个训练臂等待本核验产物，无位移拟合。
"""
import argparse,collections,concurrent.futures,json,multiprocessing,time
from pathlib import Path
import torch
from .h_equivalence_model import equivalence_layout
from .prepare import sha


def check_layout(item):
    s,entry=item
    try:
        for mode in ['local','physical']:
            if not equivalence_layout(s,entry,mode)[3]:raise ValueError('new_partition_not_known')
        return s,None
    except Exception as exc:return s,repr(exc)


def main():
    p=argparse.ArgumentParser()
    for name in ['base-registry','raw','conflicts','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=64);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(1)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**kw)))
    try:
        raw=Path(a.raw);status('waiting_raw_registry')
        while not (raw/'EXIT_CODE').exists():
            if time.monotonic()-start>14400:raise TimeoutError('raw_registry_timeout')
            time.sleep(30)
        if (raw/'EXIT_CODE').read_text().strip()!='0':raise RuntimeError('raw_registry_failed')
        base=torch.load(a.base_registry,map_location='cpu',weights_only=False);registry=torch.load(raw/'registry.pt',map_location='cpu',weights_only=False);excluded={};conflicts=json.loads(Path(a.conflicts).read_text())['excluded_smiles']
        for s,v in base.items():
            if v.get('groups') and registry.get(s)!=v:raise ValueError('existing_H_partition_changed')
            if v.get('counts',{}).get('13C')!=registry.get(s,{}).get('counts',{}).get('13C'):raise ValueError('existing_C_count_changed')
        def restore(s,reason):
            excluded[s]=reason
            if s in base:registry[s]=base[s]
            else:registry.pop(s,None)
        for s in conflicts:
            if registry.get(s,{}).get('groups') and not base.get(s,{}).get('groups'):restore(s,'previous_source_partition_conflict')
        new=[(s,v) for s,v in registry.items() if v.get('groups') and not base.get(s,{}).get('groups')];status('verifying_layout',new_structures=len(new))
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
            for s,error in pool.map(check_layout,new,chunksize=64):
                if error is not None:restore(s,'decoder_layout:'+error)
        torch.save(registry,out/'registry.pt');report=dict(new_H_partitions=len(new)-sum(reason.startswith('decoder_layout:') for reason in excluded.values()),excluded_by_reason=dict(collections.Counter(excluded.values())),excluded=excluded,base_H_partitions_preserved=True,C_counts_preserved=True,candidate_identities_changed=False,spectral_targets_used=False,source_sha256={str(f):sha(f) for f in [Path(a.base_registry),raw/'registry.pt',Path(a.conflicts),Path(__file__),Path(__file__).parent/'h_equivalence_model.py']},registry_sha256=sha(out/'registry.pt'),scope='finite-conformer partition evidence after conflict and decoder checks; no external physical-accuracy claim');(out/'summary.json').write_text(json.dumps(report,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n');print(json.dumps({k:v for k,v in report.items() if k!='excluded'}),flush=True)
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()

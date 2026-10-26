"""功能概述：编译本轮候选池排序的训练、验证和冻结模型诊断队列。
main读取已审协议与继承输入指纹，仅生成argv DAG，不执行模型、不读取测试结果。
build按phase创建独占输出目录：smoke训练仅用train内部划分；formal为16训练和
32验证；diagnostics为相同冻结权重的双池8验证及4分解。全部源与已有输入绑定SHA，
调度沿用h_research_driver的真实内外退出契约，新产物不覆盖旧轮。
"""
import argparse,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(d,source,phase):
    remote=Path('/home/user/IRfingerprints/SpectrExpert')
    old=remote/'outputs/nmr_global_retrieval_20261022'
    new=remote/'outputs/nmr_pool_ranking_20261023'
    py='/home/user/.venvs/spectrexpert/bin/python'
    inputs=json.loads((d/'inherited_input_pins.json').read_text())['inputs_sha256']
    names=['h_pool_ranker.py','h_research_ranker.py','h_research_observation.py','h_research_contract.py','h_global_ranker.py','h_global_features.py','h_research_driver.py']
    sources={str(remote/'nmr_transfer'/n):sha(source/'nmr_transfer'/n) for n in names}
    sources[str(new/'protocol.md')]=sha(d/'protocol.md')
    sources[str(new/'preregistered_arms.json')]=sha(d/'preregistered_arms.json')
    if phase=='diagnostics':sources[str(new/'pool_diagnostics.py')]=sha(d/'pool_diagnostics.py')
    jobs=[]
    def cache(seed,pool,cohort):return str(old/f'features/inherited_s{seed}'/pool/cohort/'features.pt')
    def init(seed,role):return str(old/f'rankers/factor_R0O1F0/s{seed}'/role/'best.pt')
    def add(name,kind,argv,out,paths,deps=(),gpu=True):
        jobs.append(dict(id=name,kind=kind,gpu=gpu,argv=argv,cwd=str(remote),out=str(out),deps=list(deps),inputs_sha256={p:inputs[p] for p in paths}))
    def evaluate(seed,pool,cohort,weights,family,deps=()):
        out=new/phase/family/f's{seed}'/pool/cohort
        path=cache(seed,pool,cohort)
        argv=[py,'-m','nmr_transfer.h_pool_ranker','--stage','evaluate','--cache',path,'--no-aug-checkpoint',weights[0],'--aug-checkpoint',weights[1],'--seed',str(seed),'--out',str(out)]
        name=f'eval_{family}_s{seed}_{pool}_{cohort}'
        add(name,'validation',argv,out,[path]+[p for p in weights if p in inputs],deps)
        return name,out
    if phase=='diagnostics':
        for seed in [193,211]:
            weights=[init(seed,role) for role in ['no_aug','aug']]
            for cohort in ['original_val','tune']:
                pair=[evaluate(seed,pool,cohort,weights,'frozen') for pool in ['single','dual']]
                out=new/phase/'paired'/f's{seed}'/cohort
                argv=[py,str(new/'pool_diagnostics.py'),'--single',str(pair[0][1]),'--dual',str(pair[1][1]),'--out',str(out)]
                add(f'decompose_s{seed}_{cohort}','audit',argv,out,[],[x[0] for x in pair],False)
    else:
        seeds=[193,211] if phase=='formal' else [211]
        for seed in seeds:
            for pool in ['original','mixed']:
                for loss in ['pairwise16','listwise']:
                    family=pool+'_'+loss;weights=[];deps=[]
                    roles=['no_aug','aug'] if phase=='formal' else ['no_aug']
                    for role in roles:
                        out=new/phase/'rankers'/family/f's{seed}'/role
                        paths=[cache(seed,'single','train'),cache(seed,'dual','train'),init(seed,role)]
                        argv=[py,'-m','nmr_transfer.h_pool_ranker','--stage','fit','--train',paths[0],'--dual-train',paths[1],'--initial-checkpoint',paths[2],'--training-pool',pool,'--loss',loss,'--seed',str(seed),'--epochs','64' if phase=='formal' else '1','--samples','8192' if phase=='formal' else '128','--batch-size','32','--anchor','0.1','--lr','0.001','--out',str(out)]
                        if role=='aug':argv.append('--augment')
                        name=f'fit_{family}_s{seed}_{role}'
                        add(name,'training',argv,out,paths);weights.append(str(out/'best.pt'));deps.append(name)
                    if phase=='formal':
                        for inference in ['single','dual']:
                            for cohort in ['original_val','tune']:evaluate(seed,inference,cohort,weights,family,deps)
    expected={'smoke':4,'formal':48,'diagnostics':12}[phase]
    assert len(jobs)==expected
    return dict(schema='nmr_research_jobs_20261022',training_cutoff_utc='2026-10-23T13:35:41Z',end_utc='2026-10-23T14:29:41Z',source_inputs_sha256=sources,jobs=jobs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--source',required=True);p.add_argument('--phase',choices=['smoke','formal','diagnostics'],required=True)
    a=p.parse_args();d=Path(a.directory);queue=build(d,Path(a.source),a.phase)
    output=d/(a.phase+'_queue.json')
    with output.open('x') as f:json.dump(queue,f,indent=2)
    print(json.dumps(dict(path=str(output),sha256=sha(output),jobs=len(queue['jobs']))))
if __name__=='__main__':main()

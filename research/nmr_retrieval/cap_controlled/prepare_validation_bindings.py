#!/usr/bin/env python3
"""功能概述：在8fit训练及独立审计完成后，生成独立validation_bindings和16val/4机制队列。
主要职责：main逐字节核实原训练binding、成功run、audit及实施review；创建8个final权重
身份、12个已有cache的producer引用和完整SHA闭包。make_queue仅展开协议注册的固定矩阵，
将allowed_reads限定为本job两个final权重和一个cache；不加载张量或执行评价。
代码库关系：只消费cap10本轮protocol/execution_bindings、audit_cap_training.py输出与
独立review。供root在JJ显式运行，然后交独立审查，supervise_controlled_queue.py后续准入。
输入输出与副作用：--root为本轮JJ目录，目标validation_bindings.json及两queue必须不存在。
只新建metadata；不覆盖原训练binding/旧失败run，不读取test，不执行模型或新增选择。
"""
from __future__ import annotations
import argparse,copy,hashlib,itertools,json,pathlib,sys,datetime

EXPECTED_BINDING='972d32f4e8f737fceee24d098f73702a7e5ae1004f0d3f16ba1af122aef797f2'
EVAL_SOURCES=('cap_evaluation_common.py','evaluate_cap_ranker.py','summarize_cap_validation.py','analyze_cap_mechanism.py')
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_text())
def write(path,value):
    with pathlib.Path(path).open('x') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def zero(value):assert type(value) is int and value==0
def ref(path):return {'path':str(path),'sha256':sha(path)}

def make_queue(root,binding,protocol,phase):
    jobs=[]
    combinations=itertools.product((5,10),(193,211),('single','dual'),('original_val','tune')) if phase=='validation' else itertools.product((5,10),(193,211),('dual',),('train',))
    for n,(cap,seed,pool,cohort) in enumerate(combinations):
        jid=f'{phase}_C{cap}_s{seed}_{pool}_{cohort}';out=root/phase/jid
        cache=binding['caches'][str(seed)][cohort][pool]['path']
        reads=[binding['roles'][str(cap)][str(seed)][role]['checkpoint'] for role in ('no_aug','aug')]+[cache]
        command=[str(root/'evaluate_cap_ranker.py'),'--source',protocol['source_checkout'],'--protocol',str(root/'protocol.json'),'--bindings',str(root/'validation_bindings.json'),'--cache',cache,'--cap',str(cap),'--seed',str(seed),'--pool',pool,'--cohort',cohort,'--device','cuda:0','--threads','4','--out',str(out)]
        if phase=='mechanism':command+=['--query-manifest',str(root/'fixed_train_pairs.json')]
        access=root/'access_logs'/phase/(jid+'.json')
        argv=[sys.executable,'-B',str(root/'isolation_entry.py'),'--bindings',str(root/'validation_bindings.json'),'--phase',phase,'--access-log',str(access),'--output-root',str(out)]
        for p in reads:argv+=['--allow-read',p]
        argv+=['--']+command
        jobs.append({'id':jid,'argv':argv,'command':command,'cwd':protocol['source_checkout'],'gpu':n%8,'job_out':str(out),'access_log':str(access),'allowed_reads':reads})
    sources=dict(binding['source_sha256'])
    for name in ('protocol.json','validation_bindings.json','isolation_entry.py','supervise_controlled_queue.py','launch_controlled_stage.py'):
        sources[str(root/name)]=sha(root/name)
    assert len(jobs)==(16 if phase=='validation' else 4)
    return {'schema':'cap10_controlled_queue_20261024','phase':phase,'run_name':phase,'root':str(root),'local_root':protocol['local_root'],'python':sys.executable,'bindings':str(root/'validation_bindings.json'),'jobs':jobs,'source_sha256':sources,'input_sha256':binding['input_sha256'],'review_paths':[str(root/(phase+'_admission_independent_review.json'))],'maximum_seconds':10800,'max_parallel':8,'external_api_calls':0,'promotion':False,'automatic_retry':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=pathlib.Path(a.root).resolve()
    for n in ('validation_bindings.json','validation_queue.json','mechanism_queue.json'):assert not (root/n).exists(),n
    bp=root/'execution_bindings.json';assert sha(bp)==EXPECTED_BINDING
    binding=read(bp);protocol=read(root/'protocol.json')
    assert protocol['test_allowed'] is False and protocol['promotion'] is False
    tq=root/'train_after_import_fix_queue.json';run=root/'runs/train_after_import_fix'
    c=read(run/'completion.json');l=read(run/'launch_receipt.json')
    assert c['status']=='COMPLETE' and c['inputs_unchanged'] is True and len(c['jobs'])==8
    zero(c['external_exit']);zero(l['internal_exit']);zero(l['external_exit'])
    assert (run/'EXIT_CODE').read_text().strip()=='0' and (run/'EXTERNAL_EXIT_CODE').read_text().strip()=='0'
    assert c['queue_sha256']==l['queue_sha256']==sha(tq)
    assert read(run/'input_fingerprints.before.json')==read(run/'input_fingerprints.after.json')==binding['input_sha256']
    for j in c['jobs']:zero(j['internal_exit']);zero(j['external_exit'])
    ap=root/'audits/training/summary.json';audit=read(ap);ir=root/'cap_training_audit_implementation_review.json';review=read(ir)
    assert audit['status']=='PASS' and audit['independent'] is True and audit['inputs_unchanged'] is True
    assert audit['training_run']==str(run) and audit['bindings_sha256']==sha(bp) and audit['queue_sha256']==sha(tq)
    assert (ap.parent/'EXIT_CODE').read_text().strip()=='0'
    assert review['status']=='PASS' and review['independent'] is True
    assert all(review['reviewed_sha256'].get(p)==h for p,h in audit['source_sha256'].items())
    assert len(audit['models'])==8 and all(x is True for x in audit['checks'].values()) and len(audit['checks'])==12
    inputs=dict(binding['input_sha256'])
    def pin(p,expected=None):
        actual=sha(p);assert expected is None or actual==expected
        assert str(p) not in inputs or inputs[str(p)]==actual
        inputs[str(p)]=actual;return actual
    for path,h in audit['input_sha256'].items():pin(path,h)
    for path,h in audit['source_sha256'].items():pin(path,h)
    pin(ap);pin(ap.parent/'EXIT_CODE');pin(ir)
    roles={str(cap):{str(seed):{} for seed in (193,211)} for cap in (5,10)}
    seen=set()
    for m in audit['models']:
        key=(m['cap'],m['seed'],m['role']);assert key not in seen;seen.add(key)
        cap,seed,role=key;r=copy.deepcopy(binding['roles'][str(seed)][role]);d=root/'rankers'/f'C{cap}'/f's{seed}'/role
        assert m['checkpoint']==str(d/'final.pt') and m['epochs']==64 and m['alpha']==1 and m['draws']==524288 and m['optimizer_steps']==16384
        for field,name in [('checkpoint','final.pt'),('centers','centers.json'),('center_rows','center_rows.jsonl'),('producer_summary','summary.json')]:
            r[field]=str(d/name);r[field+'_sha256']=pin(d/name,m.get(field+'_sha256'))
        pin(d/'EXIT_CODE');r['best']={'epoch':64,'alpha':1.0,'selection':'fixed_final_no_selection'}
        roles[str(cap)][str(seed)][role]=r
    assert seen==set(itertools.product((5,10),(193,211),('no_aug','aug')))
    caches=copy.deepcopy(binding['caches'])
    for cohorts in caches.values():
        for pools in cohorts.values():
            for cache in pools.values():
                prod=cache['producer'];cache['producer']={**ref(prod),'exit_path':str(pathlib.Path(prod).parent/'EXIT_CODE')}
                pin(prod);pin(cache['producer']['exit_path'])
    sources=dict(binding['source_sha256'])
    for name in EVAL_SOURCES:sources[str(root/name)]=pin(root/name)
    binding.update(roles=roles,caches=caches,source_sha256=sources,input_sha256=inputs,training_bindings=ref(bp),training_queue=ref(tq),training_run=str(run),training_audit={**ref(ap),'implementation_review':{**ref(ir),'required_scope':review['scope']}},fixed_module=ref(pathlib.Path(binding['evidence_root'])/'h_pool_fixed_center_ranker.py'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_scope='successful cap-continuation train audit plus train/validation evaluation only')
    write(root/'validation_bindings.json',binding)
    for phase in ('validation','mechanism'):write(root/(phase+'_queue.json'),make_queue(root,binding,protocol,phase))
    print(json.dumps({'status':'PREPARED_REQUIRES_INDEPENDENT_ADMISSION','input_count':len(inputs),'test_reads':0,'files':{n:sha(root/n) for n in ('validation_bindings.json','validation_queue.json','mechanism_queue.json')}}))

if __name__=='__main__':main()

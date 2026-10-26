"""功能概述：只读小metadata构造固定400机制恢复的新binding与4job队列。
保持旧213 input和全部训练/验证科学字段；仅新增机制evaluator精确ref及恢复证据。
新输出不可覆盖，必须独立具体准入后由root唯一启动；不读tensor/prediction/指标，
不SSH、不派发。request显式path/SHA，copy_map仅用于canonicalremote到已复制小字节。
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path

OLD_BINDING='2f56b0470a15b9746851aa4779c422ad4a8816af70f532450e18d75a9b3d556d'
OLD_QUEUE='6f8c76a26501d749ad7a72c3f7985ed017112004b3e8650d5c58940b2a1e0711'
FAILURE='d8a330365f7b2ae59c7817d5562b40d1e38ac2f7d8b3036232e32f0d668d843b'
STRATEGY='95b1e45dacf71ef9bbd3b7bd9a831893947db5849ef4aa20229e9226e5af70cc'
READERS=('cap_evaluation_common.py','summarize_cap_validation.py','analyze_cap_mechanism.py')

def need(x,label):
    if not x:raise ValueError(label)
def digest(x):return hashlib.sha256(x).hexdigest()
def encode(x):return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def canonical(x):
    p=Path(x);need(type(x) is str and p.is_absolute() and '..' not in p.parts and str(p)==x,'canonical_path');return p
def hexsha(x):return type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def ref_ok(x):need(type(x) is dict and set(x)=={'path','sha256'} and hexsha(x['sha256']),'path_SHA_ref');canonical(x['path']);return x
def safe(x):
    raw=Path(x).absolute();need('..' not in raw.parts and not raw.is_symlink(),'unsafe_alias');p=raw.resolve(strict=True)
    for a in (raw,p):
        need(a.suffix in ('.json','.py','.log'),'small_metadata_source_only')
        need(not any(s=='test' or s.startswith(('sealed','test-')) for s in a.parts) and 'prediction' not in a.name.lower() and 'report' not in a.name.lower(),'test_prediction_report_forbidden')
    return p

class Reader:
    def __init__(self,ref):
        ref_ok(ref);self.map_path=safe(ref['path']);raw=self.map_path.read_bytes();need(digest(raw)==ref['sha256'],'copy_map_SHA');self.map_sha=ref['sha256'];self.map=json.loads(raw);self.reads={};self.locals={}
        need(set(self.map['files'])==set(self.map['sha256']),'exact_copied_map')
    def raw(self,ref):
        ref_ok(ref);path=ref['path'];need(self.map['sha256'].get(path)==ref['sha256'],'explicit_copied_ref')
        p=safe(self.map['files'][path]);raw=p.read_bytes();need(digest(raw)==ref['sha256'],'actual_copied_SHA')
        need(path not in self.reads or self.reads[path]==ref['sha256'],'input_drift');self.reads[path]=ref['sha256'];self.locals[path]=str(p);return raw
    def ref(self,ref):return json.loads(self.raw(ref))
    def finish(self):
        need(digest(safe(self.map_path).read_bytes())==self.map_sha,'copy_map_changed')
        for p,h in self.reads.items():need(digest(safe(self.locals[p]).read_bytes())==h,'input_changed')

def prepare(reader,request):
    need(request['schema']=='cap10_mechanism_fixed_rows_recovery_request_20261025','request_schema')
    root=canonical(request['root']);refs=request['refs'];needed={'protocol','prior_bindings','prior_queue','failure_review','recovery_strategy','implementation_review','JJ_execution','JJ_log','evaluator','preparer'}
    need(set(refs)==needed,'exact_request_refs')
    for r in refs.values():ref_ok(r)
    need(refs['prior_bindings']==dict(path=str(root/'native_cost_validation_bindings.json'),sha256=OLD_BINDING) and refs['prior_queue']==dict(path=str(root/'mechanism_resource_ready_queue.json'),sha256=OLD_QUEUE),'exact_prior_binding_queue')
    need(refs['failure_review']==dict(path=str(root/'mechanism_identity_failure_independent_review.json'),sha256=FAILURE) and refs['recovery_strategy']==dict(path=str(root/'mechanism_identity_recovery_strategy.json'),sha256=STRATEGY),'exact_preserved_failure_strategy')
    proto=reader.ref(refs['protocol']);old=reader.ref(refs['prior_bindings']);q=reader.ref(refs['prior_queue'])
    need(refs['protocol']['path']==str(root/'protocol.json') and q['source_sha256'].get(refs['protocol']['path'])==refs['protocol']['sha256'],'exact_prior_protocol')
    need(proto['remote_root']==str(root) and proto['test_allowed'] is False and proto['promotion'] is False,'unchanged_protocol')
    need(len(old['input_sha256'])==213 and len(old['source_sha256'])==15 and q['input_sha256']==old['input_sha256'],'unchanged_213_inputs_15_sources')
    need(q['phase']=='mechanism' and len(q['jobs'])==4 and q['bindings']==refs['prior_bindings']['path'],'exact_prior_mechanism_queue')
    failure=reader.ref(refs['failure_review']);strategy=reader.ref(refs['recovery_strategy'])
    need(failure['status']=='PASS' and failure['independent'] is True,'actual_failure_independent_PASS')
    need(strategy.get('new_training') is False and strategy.get('validation_rerun') is False and strategy.get('fixed_train_pairs_unchanged')==400 and strategy.get('fixed_train_pairs_per_seed')==200 and strategy.get('original87_unchanged')==87 and strategy.get('compare_all_prior_complete_mechanism_prediction_lines_to_new_semantic_rows') is True,'mechanism_only_recovery_strategy')
    evaluator=refs['evaluator'];need(evaluator['path']==str(root/'mechanism_fixed_rows/evaluate_cap_ranker.py'),'exact_mechanism_leaf_path')
    sources={evaluator['path']:evaluator['sha256']}
    for name,r in request['reader_sources'].items():
        need(name in READERS and r['path']==str(root/'mechanism_fixed_rows'/name),'exact_reader_path');ref_ok(r);sources[r['path']]=r['sha256']
    need(set(request['reader_sources'])==set(READERS),'three_reader_sources')
    for p,h in sources.items():reader.raw(dict(path=p,sha256=h))
    need(refs['preparer']['path']==str(root/'prepare_mechanism_fixed_rows_recovery.py') and refs['preparer']['sha256']==digest(Path(__file__).read_bytes()),'actual_preparer_source')
    reader.raw(refs['preparer']);sources[refs['preparer']['path']]=refs['preparer']['sha256']
    impl=reader.ref(refs['implementation_review'])
    need(impl.get('status')=='PASS' and impl.get('independent') is True and impl.get('scope')=='cap10_mechanism_fixed_rows_recovery_implementation' and all(impl.get('reviewed_sha256',{}).get(p)==h for p,h in sources.items()),'complete_implementation_review')
    jj=reader.ref(refs['JJ_execution']);log=reader.raw(refs['JJ_log']).decode()
    need(type(jj.get('actual_exit')) is int and jj['actual_exit']==0 and jj['source_sha256']==jj['source_end_sha256'] and all(jj['source_sha256'].get(p)==h for p,h in sources.items()),'actual_JJ_same_sources')
    count=request['JJ_test_count'];need(type(count) is int and count>0,'explicit_JJ_test_count')
    import re
    suite_counts=[int(n) for n in re.findall(r'^Ran ([0-9]+) tests? in [^\n]+$',log,re.M)]
    need(suite_counts and all(n>0 for n in suite_counts) and sum(suite_counts)==count and len(re.findall(r'^OK$',log,re.M))==len(suite_counts) and not re.search(r'^(FAILED|ERROR|FAIL:|ERROR:)',log,re.M),'actual_JJ_suite')
    binding=copy.deepcopy(old);binding['source_sha256']={**old['source_sha256'],evaluator['path']:evaluator['sha256']};binding['mechanism_evaluator']=evaluator
    binding['mechanism_recovery']={k:refs[k] for k in ('prior_bindings','prior_queue','failure_review','recovery_strategy','implementation_review','JJ_execution','JJ_log')}
    binding_path=str(root/'mechanism_fixed_rows_bindings.json');bref=dict(path=binding_path,sha256=digest(encode(binding)))
    new=copy.deepcopy(q);name='mechanism_fixed_rows';new.update(run_name=name,bindings=binding_path,review_paths=[str(root/(name+'_admission_independent_review.json'))])
    new['source_sha256']={**q['source_sha256'],**sources,bref['path']:bref['sha256'],**{refs[k]['path']:refs[k]['sha256'] for k in ('failure_review','recovery_strategy','implementation_review','JJ_execution','JJ_log')}}
    new['analysis_tools']=dict(source_root=str(root/name),implementation_review=refs['implementation_review'])
    new['recovery_from']=dict(prior_queue=refs['prior_queue'],prior_bindings=refs['prior_bindings'],failed_mechanism_run=str(root/'runs/mechanism_resource_ready'),failure_review=refs['failure_review'],strategy=refs['recovery_strategy'],automatic_retry=False,validation_rerun=False)
    wanted={(cap,seed) for cap in (5,10) for seed in (193,211)};seen=set()
    for job in new['jobs']:
        original=copy.deepcopy(job);command=job['command'];cap=int(command[command.index('--cap')+1]);seed=int(command[command.index('--seed')+1]);need((cap,seed) in wanted and (cap,seed) not in seen,'four_exact_mechanism_roles');seen.add((cap,seed))
        need(command[0]==str(root/'native_cost_evaluation/evaluate_cap_ranker.py') and command[command.index('--cohort')+1]=='train' and command[command.index('--pool')+1]=='dual','original_mechanism_only')
        mapping={refs['prior_bindings']['path']:binding_path,command[0]:evaluator['path'],job['job_out']:str(root/name/job['id']),job['access_log']:str(root/'access_logs'/name/(job['id']+'.json'))}
        job.update(command=[mapping.get(x,x) for x in job['command']],argv=[mapping.get(x,x) for x in job['argv']],job_out=mapping[job['job_out']],access_log=mapping[job['access_log']])
        need(job['allowed_reads']==original['allowed_reads'] and len(job['allowed_reads'])==3,'same_three_tensors')
    need(seen==wanted,'complete4_mechanism_matrix')
    qpath=str(root/'mechanism_fixed_rows_queue.json');required={**new['source_sha256'],qpath:digest(encode(new))}
    return binding,new,required

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    own=Path(__file__).resolve();start=digest(own.read_bytes());rp=safe(args.request);raw=rp.read_bytes();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        request=json.loads(raw);reader=Reader(request['copy_map']);binding,q,required=prepare(reader,request);reader.finish()
        need(digest(own.read_bytes())==start and digest(safe(rp).read_bytes())==digest(raw),'source_request_changed')
        outputs={'mechanism_fixed_rows_bindings.json':binding,'mechanism_fixed_rows_queue.json':q,'required_mechanism_admission_sha256.json':required}
        for n,v in outputs.items():(out/n).write_bytes(encode(v))
        (out/'receipt.json').write_bytes(encode(dict(status='PREPARED_NOT_ADMITTED_NOT_EXECUTED',input_sha256=reader.reads,source_sha256={str(own):start},outputs_sha256={n:digest(encode(v)) for n,v in outputs.items()},scientific_dispatch=0)))
        (out/'EXIT_CODE').write_text('0\n')
    except BaseException as exc:(out/'failure.json').write_bytes(encode(dict(error=repr(exc))));(out/'EXIT_CODE').write_text('1\n');raise

if __name__=='__main__':main()

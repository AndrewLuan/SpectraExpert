"""功能概述：封存12项test完成后的独立只读结果审查；当前仅准备源代码。
先核具体冻结独审、完整12项PID/argv/双EXIT/资源及源码输入始末证据，再解锁这12份预测。
本工具禁止读取任何模型或feature tensor，禁止先读预测再检查资格；旧test预测不在解锁名单。
成本和排名用独立5c9算术重算。完整324指标、180比较和5000/317簇bootstrap复用已独审
5a/9d数学helper，并明确这不是新增数学oracle。生产统计6448只复用受审metadata gate。
输入为本地原字节copy map与显式canonical路径request，输出唯一review.json和EXIT_CODE。
不训练、不推理、不修改冻结/预选、不升格正式系统；验收保持unavailable且不声称test-blind。
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCES={
 'resource_ready_analysis/summarize_sealed_cap_results.py':'6448bfd45abce4b26d783dba555265173b9cdcbb7d2a483badce47fb37acc444',
 'native_cost_evaluation/cap_test_evaluation_common.py':'0ec9edde9b5505b959cf21d29b360bd7aff1fd867a2729151e44dfc194465345',
 'native_cost_evaluation/cap_test_evaluate_ranker.py':'0344bbd6f6f12c713ff6003940e45b3bc49c3420dde6cb4beeebe9fc69af3e07',
 'sealed_cap_pairmath.py':'5a03076328e2de15de378ce44041f8063bd24beb771b82592d3989f391836353',
 'sealed_cap_matrix.py':'9d3cf5e529d8c84e71d8d1b5dc2f6290a7d0036cb14b9bd1926244b52d1431a0',
 'review_cap_validation_mechanism_results.py':'6a9d92f49f67a19458a3bae81f3d9c379f723bd5b2eac331e839bf0b8d573686',
 'cap_validation_result_math_independent.py':'5c9c5365d9924408e820790bc5ba10a9a7df2e109106283a036ca8e1eabbeed6'}

def need(x,label):
    if not x:raise ValueError(label)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def zero(x):need(type(x) is int and x==0,'actual_strict_int_EXIT0')
def canonical(path):
    p=Path(path)
    need(p.is_absolute() and '..' not in p.parts,'canonical_absolute_path')
    need(str(p)==str(p.resolve()),'canonical_redirect_forbidden')
    return str(p)
def local(path):
    p=Path(path)
    need(p.is_absolute() and '..' not in p.parts,'local_absolute_no_parent')
    q=p.resolve(strict=True)
    need(p==q,'local_symlink_redirect_forbidden')
    need(q.suffix.lower() not in ('.pt','.pth','.ckpt','.npz','.npy'),'tensor_read_forbidden')
    need(q.name not in ('sealed_test_report.md','research_report.md','supplementary_report.md'),'legacy_report_read_forbidden')
    return q
def payload(path):
    p=Path(path);name=p.name.lower()
    return (p.suffix.lower()=='.jsonl' and name!='resource_samples.jsonl') or (p.suffix.lower()=='.json' and name.startswith(('predictions','ranks','rankings')))

class Copies:
    """metadata门前不hash预测。只核请求到的原字节；tensor只比远端实际SHA库存。"""
    def __init__(self,manifests):
        self.files={};self.inventory={};self.hashes={};self.reads={};self.values={};self.deferred={};self.maps={}
        self.predictions={};self.metadata_complete=False;self.events=[]
        for name in manifests:
            p=local(name);raw=p.read_bytes();self.maps[str(p)]=digest(raw);m=json.loads(raw)
            for key in ('remote_inventory_before_exit','remote_inventory_after_exit','rsync_exit'):zero(m[key])
            before=m['remote_before_sha256'];need(before==m['remote_after_sha256'] and bool(before),'remote_actual_begin_end')
            need(set(m['files'])==set(m['sha256']) and set(before)==set(m['files'])|set(m['hash_only_inputs']),'complete_actual_union')
            for remote,h in before.items():
                canonical(remote);need(type(h) is str and len(h)==64 and set(h)<=set('0123456789abcdef'),'hex_SHA')
                need(remote not in self.inventory or self.inventory[remote]==h,'remote_inventory_conflict');self.inventory[remote]=h
            for remote,path in m['files'].items():
                q=Path(path);need(q.is_absolute() and '..' not in q.parts,'copy_absolute_path')
                need(q.resolve().is_relative_to(p.parent),'copy_parent_containment')
                need(before[remote]==m['sha256'][remote],'copy_actual_inventory_SHA')
                need(remote not in self.hashes or self.hashes[remote]==m['sha256'][remote],'copy_SHA_conflict')
                self.files[remote]=str(q);self.hashes[remote]=m['sha256'][remote]
            need(all(before[k]==h for k,h in m['hash_only_inputs'].items()),'hash_only_exact_actual')
    def pin(self,path,expected):
        key=canonical(path);need(self.inventory.get(key)==expected,'actual_inventory_pin:'+key);return expected
    def raw(self,path,expected=None):
        key=canonical(path);name=Path(key).name
        need(Path(key).suffix.lower() not in ('.pt','.pth','.ckpt','.npz','.npy'),'canonical_tensor_read_forbidden')
        ispred=payload(key)
        if ispred:need(self.metadata_complete and self.predictions.get(key)==expected,'prediction_before_complete12_gate_or_unregistered')
        need(key in self.files,'required_actual_copy:'+key)
        p=local(self.files[key]);need(payload(p)==ispred,'canonical_local_payload_class_mismatch')
        raw=p.read_bytes();h=digest(raw)
        need(h==self.hashes[key] and (expected is None or h==expected),'actual_consumed_bytes_SHA')
        need(key not in self.reads or self.reads[key][1]==h,'consumed_drift')
        self.reads[key]=(str(p),h);self.values[key]=h;self.events.append(('prediction' if ispred else 'metadata',key));return raw
    def json(self,path,expected=None):return json.loads(self.raw(path,expected))
    def add(self,path,expected=None):
        key=canonical(path)
        if Path(key).suffix.lower() in ('.pt','.pth','.ckpt','.npz','.npy'):
            need(expected is not None,'tensor_inventory_requires_exact_SHA');self.pin(key,expected);self.values[key]=expected;self.deferred[key]=expected;return expected
        return digest(self.raw(key,expected))
    def defer(self,path,expected):self.pin(path,expected);self.values[canonical(path)]=expected;self.deferred[canonical(path)]=expected
    def exit0(self,path):need(self.raw(path)==b'0\n','actual_EXIT0_file')
    def unlock(self,refs):
        need(not self.metadata_complete and len(refs)==12 and len({str(path) for _,path,_ in refs})==12,'exact12_prediction_unlock')
        self.predictions={canonical(path):h for _,path,h in refs}
        for path,h in self.predictions.items():need(self.inventory.get(path)==self.hashes.get(path)==h,'prediction_declared_actual_SHA')
        self.metadata_complete=True;self.events.append(('complete12_gate',None))
    def finish(self):
        for p,h in self.maps.items():need(digest(local(p).read_bytes())==h,'map_end_drift')
        for p,h in self.reads.values():need(digest(local(p).read_bytes())==h,'consumed_end_drift')
        return True

def load_helpers():
    for name,h in SOURCES.items():need(digest(local(HERE/name).read_bytes())==h,'locked_helper:'+name)
    def module(name,filename):
        spec=importlib.util.spec_from_file_location(name,HERE/filename);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    stats=module('_cap12_reviewed_statistics','resource_ready_analysis/summarize_sealed_cap_results.py')
    old=module('_cap12_independent_execution','review_cap_validation_mechanism_results.py')
    need(Path(stats.c.__file__).resolve()==HERE/'native_cost_evaluation/cap_test_evaluation_common.py','exact_native_common')
    need(Path(stats.matrix.__file__).resolve()==HERE/'sealed_cap_matrix.py' and Path(stats.pm.__file__).resolve()==HERE/'sealed_cap_pairmath.py','exact_math_matrix_helpers')
    return stats,old

def close(actual,expected,label='result'):
    if type(expected) is dict:
        need(type(actual) is dict and set(actual)==set(expected),'exact_dict_keys:'+label)
        for k,v in expected.items():close(actual[k],v,label+'.'+str(k))
    elif type(expected) is list:
        need(type(actual) is list and len(actual)==len(expected),'exact_list:'+label)
        for n,(a,b) in enumerate(zip(actual,expected)):close(a,b,label+'.'+str(n))
    elif type(expected) is float:
        import math
        need(type(actual) in (int,float) and math.isfinite(actual) and abs(actual-expected)<=2e-12,'numeric_value:'+label)
    else:need(type(actual) is type(expected) and actual==expected,'exact_value:'+label)

def verify(store,request,stats,independent):
    need(request['schema']=='cap10_actual12_test_result_review_request_20261025','request_schema')
    args=argparse.Namespace(**request['args']);root=Path(request['root'])
    fields={'source','protocol','bindings','frozen_selection','freeze_review','queue','run','analysis_review','statistics_output'}
    need(set(vars(args))==fields,'exact_review_request_args')
    for value in vars(args).values():canonical(value)
    own=[root/'resource_ready_analysis/summarize_sealed_cap_results.py',root/'sealed_cap_matrix.py',root/'sealed_cap_pairmath.py',root/'native_cost_evaluation/cap_test_evaluation_common.py']
    with patch.object(stats.c,'Pins',lambda allow_test=False:store):
        _,protocol,bindings,frozen,_=stats.c.sealed_gate(args,own,require_cell=False)
    review=stats.load_metadata(store,args.analysis_review)
    stats.c.reviewed(review,{str(p):store.values[str(p)] for p in own},'analysis_implementation')
    need('implementation' in review['scope'],'analysis_review_scope')
    q,results,refs=stats.execution(store,args,bindings,frozen)
    _,execution=independent.actual_stage(store,root,args.queue,args.run,'test',12)
    for p,h in {**frozen['input_sha256'],**frozen['source_sha256'],**q['input_sha256'],**q['source_sha256']}.items():store.pin(p,h)
    execution_ref=request['statistics_execution'];record=store.json(execution_ref['path'],execution_ref['sha256'])
    for key in ('actual_inner_exit','actual_external_exit','actual_ssh_exit'):zero(record[key])
    argv=record['argv'];script=str(root/'resource_ready_analysis/summarize_sealed_cap_results.py')
    need(script in argv and '--out' in argv and argv[argv.index('--out')+1]==args.statistics_output,'actual_statistics_command')
    for key in ('source','protocol','bindings','frozen_selection','freeze_review','queue','run','analysis_review'):
        flag='--'+key.replace('_','-');need(argv.count(flag)==1 and argv[argv.index(flag)+1]==getattr(args,key),'actual_statistics_CLI:'+key)
    # 唯一预测解锁点：具体冻结、完整12执行、统计真实退出及全部元数据闭合之后。
    store.unlock(refs)
    raw={name:{} for name in stats.matrix.CONFIGS}
    for (config,seed),path,h in refs:
        rows=[json.loads(line) for line in store.raw(path,h).splitlines() if line]
        stats.verify_leaf_values(rows,results[str((config,seed))])
        for row in rows:independent.calc.audit_row(row,row['final_cost_dtype'])
        raw[config][seed]=rows
    expected=stats.matrix.analyze_matrix(raw,protocol['test_preregistration'])
    claimed=store.json(Path(args.statistics_output)/'analysis.json');close(claimed,expected,'full324_180')
    summary=store.json(Path(args.statistics_output)/'summary.json');store.exit0(Path(args.statistics_output)/'EXIT_CODE')
    need(summary['status']=='PASS' and summary['preselected']==frozen['preselected'] and summary['inputs_unchanged'] is True,'summary_frozen_selection_unchanged')
    for key,value in [('test_cells',12),('new_evaluation_jobs',12),('metric_rows',324),('comparison_strata',180),('bootstrap_seed',317),('bootstrap_replicates',5000)]:
        need(type(summary[key]) is int and summary[key]==value,'exact_registered_count:'+key)
    for key in ('test_reselection_performed','selection_performed','promotion','training','official_system_changed','test_blind_claim'):need(summary[key] is False,'no_scope_expansion:'+key)
    need(summary['independent_acceptance_status']=='unavailable' and summary['model_calls']==0,'acceptance_and_zero_models')
    for p,h in summary['input_sha256'].items():store.pin(p,h)
    for p,h in summary['output_sha256'].items():need(store.add(p,h)==h,'statistics_output_SHA')
    table=list(csv.DictReader(store.raw(Path(args.statistics_output)/'metrics.tsv').decode().splitlines(),delimiter='\t'))
    need(len(table)==324,'complete324_export_rows')
    for actual,want in zip(table,expected['metrics']):
        for k,v in want.items():
            text=actual[k]
            if v is None:need(text=='','undefined_export_empty')
            elif type(v) is int:need(int(text)==v,'integer_export')
            elif type(v) is float:close(float(text),v,'TSV.'+k)
            else:need(text==v,'identity_export')
        need(all(actual[k]=='' for k in set(actual)-set(want)),'extra_export_fields_empty')
    store.finish()
    return dict(status='PASS',independent=True,reviewer='evidence_audit independent of evaluation and statistics producers',
      scope='actual12_test_execution_native_rank_and_pre_registered_statistics_using_reviewed_numeric_helpers',
      execution=execution,preselected=frozen['preselected'],metric_rows=324,comparison_strata=180,bootstrap_seed=317,bootstrap_replicates=5000,
      all12_metadata_verified_before_prediction_read=True,model_tensor_reads=0,test_reselection_performed=False,promotion=False,
      official_system_changed=False,test_blind_claim=False,independent_acceptance_status='unavailable',
      numeric_reuse='Reviewed 5a pairmath and 9d matrix recompute all 5000/317 intervals; this review is not a new independent numeric oracle.',
      independent_native_rank_helper_sha256=SOURCES['cap_validation_result_math_independent.py'],inputs_unchanged=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',action='append',required=True);p.add_argument('--request',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    own=local(__file__);own_sha=digest(own.read_bytes());request_path=local(a.request);request_raw=request_path.read_bytes()
    out=Path(a.out).absolute();need(not out.exists(),'unique_result_review_out');out.mkdir(parents=True)
    try:
        stats,independent=load_helpers();store=Copies(a.manifest);result=verify(store,json.loads(request_raw),stats,independent)
        source_sha={str(HERE/n):h for n,h in SOURCES.items()};source_sha[str(own)]=own_sha
        for path,h in source_sha.items():need(digest(local(path).read_bytes())==h,'checker_helper_end_drift')
        need(digest(local(request_path).read_bytes())==digest(request_raw),'request_end_drift')
        result.update(schema='cap10_actual12_test_results_independent_review_20261025',utc=datetime.now(timezone.utc).isoformat(),
          reviewed_sha256={**store.inventory,**store.maps,**source_sha,str(request_path):digest(request_raw)},
          source_sha256=source_sha,consumed_sha256={p:h for p,(_,h) in store.reads.items()})
        target=out/'review.json';target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'EXIT_CODE').write_text('0\n')
        print(json.dumps(dict(status='PASS',review=str(target),sha256=digest(target.read_bytes()))))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc)),indent=2)+'\n');(out/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()

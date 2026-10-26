"""功能概述：只读独审机制固定名单恢复的唯一4格实际队列，保留成功16验证。
复用锁定c09原训练/213输入/两次验证失败闭包；另核机制失败d8a、策略95b、五源独审、
真实三进程JJ61及新binding/queue精确差分。只读取已复制metadata/source和SHA库存。
成功validation仅核16项PID/argv/退出/资源收据；禁止本工具读取预测、指标或张量值。
不prepare、不SSH、不训练/评估/选型/重试。输出只是具体机制恢复准入，实际执行由root。
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

PARENT=Path(__file__).resolve().parent
RESOURCE_SHA='c09a1ad5ac666137ef7cb98f918aa30852c05de55ac7f24c167156b35daf54d4'
RESULT_SHA='6a9d92f49f67a19458a3bae81f3d9c379f723bd5b2eac331e839bf0b8d573686'
BASE_REQUEST_SHA='854a97481401576de31077e2e3dda77ba81850d674fea4b944bc93c0db1d7c46'
VAL_MAP_SHA='65fc5ca37a4313dab3cd4de390406f41c3fa405102e384e206531e16d870cf03'
BINDING_SHA='2f56b0470a15b9746851aa4779c422ad4a8816af70f532450e18d75a9b3d556d'
QUEUE_SHA='6f8c76a26501d749ad7a72c3f7985ed017112004b3e8650d5c58940b2a1e0711'
FAILURE_SHA='d8a330365f7b2ae59c7817d5562b40d1e38ac2f7d8b3036232e32f0d668d843b'
STRATEGY_SHA='95b1e45dacf71ef9bbd3b7bd9a831893947db5849ef4aa20229e9226e5af70cc'

def need(x,label):
    if not x:raise ValueError(label)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def load(name,filename,expected):
    p=PARENT/filename;need(not p.is_symlink() and digest(p.read_bytes())==expected,'locked_checker_source:'+filename)
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

resource=load('_review_locked_resource_admission','review_resource_ready_evaluation_admission.py',RESOURCE_SHA)
results=load('_review_locked_validation_results','review_cap_validation_mechanism_results.py',RESULT_SHA)
base=resource.base

class ValidationMetadata:
    """只读成功val的实际metadata；预测和tensor即使在copy map中也绝不打开。"""
    @staticmethod
    def safe(path):
        raw=Path(path).absolute();resolved=results.safe(raw)
        for p in (raw,resolved):
            need('prediction' not in p.name.lower() and p.suffix.lower() not in ('.pt','.pth','.ckpt','.npz','.npy'),'admission_no_prediction_or_tensor_read')
        return resolved
    def __init__(self,path,current):
        self.path=self.safe(path);raw=self.path.read_bytes();need(digest(raw)==VAL_MAP_SHA,'exact_completed_validation_map')
        self.pins={str(self.path):VAL_MAP_SHA};self.map=json.loads(raw);self.hashes=self.map['sha256'];self.current=current;self.reads={}
        for k in ('remote_inventory_before_exit','remote_inventory_after_exit','rsync_exit'):base.zero(self.map[k])
        need(self.map['remote_before_sha256']==self.map['remote_after_sha256'],'successful_validation_collection_begin_end')
        ref=self.map['source_manifest'];p=self.safe(ref['path']);raw=p.read_bytes();need(digest(raw)==ref['sha256'],'original_completed_validation_manifest_SHA')
        self.pins[str(p)]=ref['sha256'];m=json.loads(raw)
        for k in ('remote_inventory_before_exit','remote_inventory_after_exit','rsync_exit'):
            base.zero(m[k]);need(m[k]==self.map[k],'original_validation_collection_actual_exit')
        need(set(self.map['files'])==set(self.hashes) and set(self.map['remote_before_sha256'])==set(self.map['files'])|set(self.map['hash_only_inputs']),'complete_successful_validation_metadata_map')
        need(m['remote_before_sha256']==m['remote_after_sha256']==self.map['remote_before_sha256'],'validation_view_full_actual_union')
        need(all(m['files'].get(k)==v and m['sha256'][k]==self.hashes[k] for k,v in self.map['files'].items()),'validation_metadata_view_exact_subset')
    def raw(self,path):
        key=str(path);need('prediction' not in Path(key).name.lower() and Path(key).suffix.lower() not in ('.pt','.pth','.ckpt','.npz','.npy'),'admission_no_prediction_or_tensor_read')
        need(key in self.map['files'],'successful_val_metadata_required:'+key)
        p=self.safe(self.map['files'][key]);need(p.is_relative_to(self.path.parent),'validation_copy_containment')
        raw=p.read_bytes();need(digest(raw)==self.hashes[key],'actual_successful_validation_metadata_SHA')
        self.reads[str(p)]=self.hashes[key];return raw
    def json(self,path):return json.loads(self.raw(path))
    def pin(self,path,h):
        key=str(path);need(self.map['remote_before_sha256'].get(key)==h,'successful_validation_actual_pin')
        if key in self.current.remote:need(self.current.remote[key]==h,'successful_validation_current_pin')
    def exit0(self,path):need(self.raw(path)==b'0\n','successful_validation_EXIT0')
    def finish(self):
        for p,h in {**self.pins,**self.reads}.items():need(digest(self.safe(p).read_bytes())==h,'validation_metadata_end_drift')

def verify(bundle,request,prior_request,val):
    need(request['schema']=='cap10_mechanism_fixed_rows_recovery_request_20261025','mechanism_prepare_request_schema')
    root=Path(request['root']);refs=request['refs'];protocol=bundle.ref(refs['protocol'])
    # 完整旧8fit审计与213输入、dtype/资源两旧失败、原16/4登记均保留，不输出旧准入。
    resource.verify(bundle,prior_request)
    need(refs['prior_bindings']==dict(path=str(root/'native_cost_validation_bindings.json'),sha256=BINDING_SHA)
      and refs['prior_queue']==dict(path=str(root/'mechanism_resource_ready_queue.json'),sha256=QUEUE_SHA),'exact_preserved_scientific_registry')
    need(refs['failure_review']==dict(path=str(root/'mechanism_identity_failure_independent_review.json'),sha256=FAILURE_SHA)
      and refs['recovery_strategy']==dict(path=str(root/'mechanism_identity_recovery_strategy.json'),sha256=STRATEGY_SHA),'exact_failed_mechanism_and_strategy_refs')
    old=bundle.ref(refs['prior_bindings']);oq=bundle.ref(refs['prior_queue']);failure=bundle.ref(refs['failure_review']);strategy=bundle.ref(refs['recovery_strategy'])
    need(failure['status']=='PASS' and failure['independent'] is True and failure['scope']=='failed_mechanism_execution_preservation_and_exposure_not_recovery_admission'
      and failure['queue_sha256']==QUEUE_SHA and failure['all_four_dispatched'] is True and failure['completed_cells']==0
      and failure['complete_output_lines_lower_bound']==12 and failure['actual_input_pins_preserved']==213,'actual_failed_four_mechanism_cells')
    need(strategy['recovery']=='all4_fresh_in_one_new_run_after_mechanism_only_adapter_and_actual_queue_independent_PASS'
      and strategy['validation_rerun'] is False and strategy['prior_partial_predictions_used_for_selection'] is False
      and strategy['fixed_train_pairs_unchanged']==400 and strategy['original87_unchanged']==87
      and strategy['compare_all_prior_complete_mechanism_prediction_lines_to_new_semantic_rows'] is True
      and strategy['new_training'] is False and strategy['test_read'] is False,'locked_mechanism_only_semantic_recovery')
    failed=root/'runs/mechanism_resource_ready';done=bundle.json(failed/'completion.json');outer=bundle.json(failed/'launch_receipt.json')
    need(done['status']=='FAILED' and done['pid']==outer['child_pid']==failure['driver_pid']
      and outer['wrapper_pid']==failure['wrapper_pid'] and done['queue_sha256']==outer['queue_sha256']==QUEUE_SHA,'actual_failed_mechanism_PID_queue')
    for x in (done['external_exit'],outer['external_exit'],outer['internal_exit']):need(type(x) is int and x==1,'failed_mechanism_strict_double1')
    for p in (root/'mechanism_resource_ready_queue.json',failed/'completion.json',failed/'launch_receipt.json'):
        suffix='/copied/'+str(p).lstrip('/')
        base.reviewed_preserved_alias(failure,p,bundle.hashes[str(p)],suffix)
    sources={refs['evaluator']['path']:refs['evaluator']['sha256'],refs['preparer']['path']:refs['preparer']['sha256']}
    sources.update({ref['path']:ref['sha256'] for ref in request['reader_sources'].values()})
    expected_paths={str(root/'mechanism_fixed_rows'/n) for n in ('evaluate_cap_ranker.py',*resource.READERS)}|{str(root/'prepare_mechanism_fixed_rows_recovery.py')}
    need(set(sources)==expected_paths and len(sources)==5,'exact_five_recovery_sources')
    impl=bundle.ref(refs['implementation_review']);need(impl.get('scope')=='cap10_mechanism_fixed_rows_recovery_implementation','mechanism_implementation_scope')
    resource.reviewed_alias(impl,sources,root,protocol,'mechanism_implementation')
    jj=bundle.ref(refs['JJ_execution']);base.zero(jj['actual_exit']);base.zero(jj['actual_ssh_exit'])
    need(jj['source_sha256']==jj['source_end_sha256'] and all(jj['source_sha256'].get(p)==h for p,h in sources.items()),'actual_JJ_five_sources_unchanged')
    need(jj['log_sha256']==refs['JJ_log']['sha256'],'actual_JJ_receipt_bound_raw_log')
    need(jj['all_deployed_sha256']==jj['all_deployed_end_sha256'] and
      jj['dependencies_sha256']==jj['dependencies_end_sha256'] and jj['all_sources_unchanged'] is True,'actual_JJ_deployed_dependencies_unchanged')
    log=bundle.raw(refs['JJ_log']['path'],refs['JJ_log']['sha256']).decode();counts=re.findall(r'^Ran ([0-9]+) tests? in [^\n]+$',log,re.M)
    need(counts==['18','12','31'] and request['JJ_test_count']==61 and type(request['JJ_test_count']) is int
      and len(re.findall(r'^OK$',log,re.M))==3 and not re.search(r'^FAILED\b|^ERROR:',log,re.M),'actual_JJ_three_fresh_suites_61')
    bp=root/'mechanism_fixed_rows_bindings.json';b=bundle.json(bp)
    need(set(b)==set(old)|{'mechanism_evaluator','mechanism_recovery'},'exact_binding_extension_keys')
    need(all(b[k]==v for k,v in old.items() if k!='source_sha256'),'all_scientific_binding_fields_unchanged')
    need(b['source_sha256']=={**old['source_sha256'],refs['evaluator']['path']:refs['evaluator']['sha256']}
      and b['mechanism_evaluator']==refs['evaluator'],'runtime_old15_plus_new_leaf')
    need(b['mechanism_recovery']=={k:refs[k] for k in ('prior_bindings','prior_queue','failure_review','recovery_strategy','implementation_review','JJ_execution','JJ_log')},'binding_exact_recovery_evidence')
    qp=root/'mechanism_fixed_rows_queue.json';q=bundle.json(qp);name='mechanism_fixed_rows'
    need(q['run_name']==name and q['bindings']==str(bp) and q['review_paths']==[str(root/(name+'_admission_independent_review.json'))],'new_mechanism_run_binding_review')
    proofs={refs[k]['path']:refs[k]['sha256'] for k in ('failure_review','recovery_strategy','implementation_review','JJ_execution','JJ_log')}
    expected_sources={**oq['source_sha256'],**sources,str(bp):bundle.hashes[str(bp)],**proofs}
    need(q['source_sha256']==expected_sources and len(expected_sources)==36,'exact36_queue_sources')
    need(q['input_sha256']==b['input_sha256']==old['input_sha256'] and len(q['input_sha256'])==213,'unchanged213_scientific_inputs')
    need(q['analysis_tools']==dict(source_root=str(root/name),implementation_review=refs['implementation_review']),'new_pure_reader_namespace')
    need(q['recovery_from']==dict(prior_queue=refs['prior_queue'],prior_bindings=refs['prior_bindings'],
      failed_mechanism_run=str(failed),failure_review=refs['failure_review'],strategy=refs['recovery_strategy'],automatic_retry=False,validation_rerun=False),'transparent_mechanism_only_recovery')
    need(len(q['jobs'])==len(oq['jobs'])==4,'fresh4_registered_mechanism_jobs')
    for a,z in zip(oq['jobs'],q['jobs']):
        mapping={refs['prior_bindings']['path']:str(bp),a['command'][0]:refs['evaluator']['path'],a['job_out']:str(root/name/a['id']),
          a['access_log']:str(root/'access_logs'/name/(a['id']+'.json'))}
        expected=dict(a,command=[mapping.get(x,x) for x in a['command']],argv=[mapping.get(x,x) for x in a['argv']],
          job_out=mapping[a['job_out']],access_log=mapping[a['access_log']])
        need(z==expected,'only_leaf_binding_output_access_change')
    changed={'run_name','bindings','review_paths','source_sha256','analysis_tools','recovery_from','jobs'}
    need({k:v for k,v in q.items() if k not in changed}=={k:v for k,v in oq.items() if k not in changed},'all_other_queue_fields_identical')
    for p,h in q['source_sha256'].items():bundle.raw(p,h);bundle.pin(p,h)
    for p,h in q['input_sha256'].items():bundle.pin(p,h)
    _,validation=results.actual_stage(val,root,root/'validation_resource_ready_queue.json',root/'runs/validation_resource_ready','validation',16)
    return dict(schema='cap10_mechanism_fixed_rows_admission_independent_review_20261025',status='PASS',independent=True,
      reviewer='evidence_audit independent of new leaf, reader and preparer authors; actual metadata and preserved input audit',
      scope='exact_fixed400_mechanism_only_recovery_queue_implementation_and_input_admission',admission_phase='mechanism',
      reviewed_sha256={**q['source_sha256'],str(qp):bundle.hashes[str(qp)]},jobs=4,scientific_input_pins=213,
      training_run=b['training_run'],training_budget_changed=False,new_training_jobs=0,test_reads=0,
      successful_validation16=validation,validation_rerun=False,partial_predictions_used_for_selection=False,
      prior12_complete_lines_semantic_comparison_required=True,implementation_review=refs['implementation_review'],
      JJ_execution=refs['JJ_execution'],JJ_test_count=61,failure_review=refs['failure_review'],recovery_strategy=refs['recovery_strategy'],
      operator_paths=q['operator_paths'],resource_wait=q['resource_wait'],automatic_retry=False,selection_performed=False,promotion=False)

def main():
    ap=argparse.ArgumentParser()
    for key in ('request','manifest','base-request','validation-manifest','out'):ap.add_argument('--'+key,required=True)
    a=ap.parse_args();own=results.safe(__file__);own_sha=digest(own.read_bytes());rp=results.safe(a.request);raw=rp.read_bytes()
    prior=results.safe(a.base_request);priorraw=prior.read_bytes();need(digest(priorraw)==BASE_REQUEST_SHA,'exact_original_resource_request')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        bundle=base.Bundle(a.manifest);val=ValidationMetadata(a.validation_manifest,bundle)
        result=verify(bundle,json.loads(raw),json.loads(priorraw),val);bundle.finish();val.finish()
        pins={str(own):own_sha,str(rp):digest(raw),str(prior):BASE_REQUEST_SHA,
          str(PARENT/'review_resource_ready_evaluation_admission.py'):RESOURCE_SHA,str(PARENT/'review_cap_validation_mechanism_results.py'):RESULT_SHA}
        for p,h in pins.items():need(digest(results.safe(p).read_bytes())==h,'checker_or_request_end_drift')
        result.update(input_sha256={**bundle.reads,**val.pins,**val.reads,**pins},artifact_map_sha256=bundle.map_sha,
          source_sha256={str(own):own_sha},inputs_unchanged=True,utc=datetime.now(timezone.utc).isoformat())
        f=out/'mechanism_fixed_rows_admission_independent_review.json';f.write_text(json.dumps(result,indent=2)+'\n');(out/'EXIT_CODE').write_text('0\n')
        print(json.dumps(dict(status='PASS',review=str(f),sha256=digest(f.read_bytes()))))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc)),indent=2)+'\n');(out/'EXIT_CODE').write_text('1\n');raise

if __name__=='__main__':main()

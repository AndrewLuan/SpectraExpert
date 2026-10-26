"""功能概述：只读独立核验单行validation成本诊断的真实执行与dtype证据。
只解析已经复制的小JSON/源码与退出收据，不运行模型、不解析成本数组、不读取tensor。
逐字节绑定受审探针、旧bindings、内外/SSH真实0、isolation PID/argv/UTC与模型状态不变；
结论局限于指定C5/s193/single/original_val row0，不能替代完整validation恢复准入。
"""
from pathlib import Path
import hashlib,json
from datetime import datetime

ROOT=Path(__file__).resolve().parent
def need(x,label):
    if not x:raise ValueError(label)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    p=ROOT/'diagnostic_cost_evidence';pin={}
    def load(rel):
        f=p/rel;raw=f.read_bytes();pin[str(f)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    summary=load('diagnostics/validation_cost_contract_row0/summary.json')
    receipt=load('actual_validation_cost_diagnostic_execution_receipt.json')
    pre=load('actual_validation_cost_diagnostic_execution_receipt.pre_ssh_ack.json')
    ack=load('actual_validation_cost_diagnostic_ssh_receipt.json')
    access=load('access_logs/diagnostics/validation_cost_contract_row0.json')
    protocol=load('diagnostics/validation_cost_contract_row0/protocol.json')
    review=load('validation_cost_diagnostic_implementation_independent_review.json')
    need(review['status']=='PASS' and review['independent'] is True,'source_review_PASS')
    need(receipt['implementation_review_sha256']==pin[str(p/'validation_cost_diagnostic_implementation_independent_review.json')],'review_SHA')
    need(receipt['summary_sha256']==pin[str(p/'diagnostics/validation_cost_contract_row0/summary.json')],'summary_SHA')
    for k in ('actual_inner_exit','actual_external_exit','actual_ssh_exit'):
        need(type(receipt[k]) is int and receipt[k]==0,'actual_'+k)
    need(pre['actual_ssh_exit'] is None and ack['actual_ssh_exit']==0,'no_prefilled_SSH_success')
    need(ack['previous_receipt_sha256']==pin[str(p/'actual_validation_cost_diagnostic_execution_receipt.pre_ssh_ack.json')],'SSH_ack_previous_bytes')
    need(pre['argv']==receipt['argv'] and pre['pid']==receipt['pid'],'receipt_ack_same_execution')
    need(ack['session_id']==10441 and ack['completion_chunk']=='fb7c88','actual_SSH_execution_binding')
    argv=receipt['argv'];inner=argv[argv.index('--')+1:]
    need(access['argv']==inner and protocol['argv']==inner,'actual_inner_argv')
    need(access['pid']==summary['pid']==receipt['pid']==protocol['pid'],'child_PID')
    need(access['parent_pid']==summary['parent_pid']==receipt['parent_pid'],'parent_PID')
    need(datetime.fromisoformat(receipt['started_utc'])<=datetime.fromisoformat(access['started_utc'])<=datetime.fromisoformat(access['finished_utc'])<=datetime.fromisoformat(receipt['finished_utc']),'same_host_UTC')
    allowed=[argv[i+1] for i,v in enumerate(argv) if v=='--allow-read']
    need(len(allowed)==3 and set(access['recorded_reads'])==set(allowed) and set(access['allowed_reads'])==set(allowed),'only_three_registered_tensor_reads')
    need(access['rejected_paths']==[] and access['internal_exit']==0,'isolation_exit')
    need(all('/test/' not in x and '/sealed' not in x for x in allowed),'trainval_only')
    need((summary['cap'],summary['seed'],summary['pool'],summary['cohort'],summary['row_index'])==(5,193,'single','original_val',0),'fixed_row_role')
    need(summary['model_state_before']==summary['model_state_after'] and summary['inputs_unchanged'] is True,'state_inputs_frozen')
    for k in ('training','test_read','rank_metrics_computed','selection_performed','promotion'):need(summary[k] is False,k)
    need(summary['forward_bank_calls']==summary['external_api_calls']==0,'no_bank_or_API')
    d=summary['diagnostic']
    need(d['legacy_assertions']==dict(shape512=True,dtype_float32=False,all_finite=True),'isolated_failure_branch')
    for k in ('coarse','native_cost'):
        need(d[k]['shape']==[512] and d[k]['dtype']=='torch.float64' and d[k]['finite'] is True and d[k]['nonfinite_count']==0,k)
    need(d['features']['shape']==[512,33] and d['features']['dtype']=='torch.float32','features')
    for item in d['scale']+d['correction']:
        need(item['dtype']=='torch.float32' and item['finite'] is True and item['nonfinite_count']==0,'finite_fp32_scales_corrections')
    for f in p.rglob('*'):
        if f.is_file() and str(f) not in pin:pin[str(f)]=sha(f)
    need((p/'diagnostics/validation_cost_contract_row0/EXIT_CODE').read_text().strip()=='0','real_EXIT0')
    need(sha(p/'actual_validation_cost_diagnostic.log')==receipt['log_sha256'],'log_SHA')
    probe=p/'diagnose_validation_cost_contract.py'
    need(sha(probe)==review['reviewed_sha256'][str(ROOT/'diagnose_validation_cost_contract.py')],'actual_reviewed_probe_SHA')
    original=ROOT/'attempt_history/validation_finite_cost/remote/validation_bindings.json'
    need(summary['input_sha256'][protocol['args']['bindings']]==sha(original),'original_binding_unchanged')
    result=dict(schema='cap10_validation_cost_diagnostic_result_review_20261025',status='PASS',independent=True,
      scope='actual_single_validation_row_native_cost_contract_not_full_validation_or_recovery_admission',
      diagnosis='native_float64_cost_is_finite_shape512; only_original_float32_dtype_assertion_false',
      generalization_scope='C5_seed193_single_original_val_row0_only',
      no_NaN_or_shape_failure_in_this_row=True,source_review_sha256=receipt['implementation_review_sha256'],
      actual_inner_exit=0,actual_external_exit=0,actual_ssh_exit=0,model_state_unchanged=True,
      reviewed_sha256=pin,validation_recovery_admission=False,selection_performed=False,test_payload_read=False,promotion=False)
    for path,h in pin.items():need(sha(Path(path))==h,'review_end_SHA')
    out=ROOT/'validation_cost_diagnostic_result_independent_review.json';need(not out.exists(),'unique_out')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','review':str(out),'sha256':sha(out)}))
if __name__=='__main__':run()

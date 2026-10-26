"""功能概述：只在受审首失败validation cell的row0核native cost shape/dtype/有限性。
核原validation binding/source/cache/C5两final权重SHA，CPU仅运行同一row两role row_delta。
不读test、不调用bank、不训练、不输出任何rank/metric/谱或成本数组值。保留原native
PyTorch promotion，不cast成本。输入模型和源码始末重核，独占新输出保存真实EXIT。
该诊断不恢复validation，不改变任何原科学源、binding、queue或已失败产物。
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
import torch
import evaluate_cap_ranker as old
from cap_evaluation_common import Pins,need,safe,utc,write

def describe(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),finite=bool(torch.isfinite(value).all()),
      nonfinite_count=int((~torch.isfinite(value)).sum()),sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())

def inspect_cost(models,natural,gamma,device):
    with torch.inference_mode():
        corrections=[saved['alpha']*model.row_delta(natural,scale,device)*scale[0] for model,scale,saved in models]
        costs=natural['coarse_scores'].to(device)+(1-gamma)*corrections[0]+gamma*corrections[1]
    return dict(coarse=describe(natural['coarse_scores']),features=describe(natural['features']),
      scale=[describe(scale) for _,scale,_ in models],correction=[describe(v) for v in corrections],
      native_cost=describe(costs),legacy_assertions=dict(shape512=costs.shape==(512,),dtype_float32=costs.dtype==torch.float32,all_finite=bool(torch.isfinite(costs).all())),
      formula='coarse_scores.to(device)+(1-gamma)*correction0+gamma*correction1; corrections=alpha*row_delta*scale[0]; no casts')

def run(args,out):
    pins=Pins();pins.add(__file__);pins.add(old.__file__);pins.add(Path(old.__file__).with_name('cap_evaluation_common.py'))
    b=pins.json(args.bindings);need(b['schema']=='cap10_execution_bindings_20261024','actual_validation_binding')
    need(safe(args.source)==safe(b['source_checkout']),'source_checkout')
    for p,h in b['source_sha256'].items():pins.add(p,h)
    cache=b['caches']['193']['original_val']['single'];pins.add(cache['path'],cache['sha256'])
    producer=pins.json(cache['producer']['path'],cache['producer']['sha256']);need(pins.raw(cache['producer']['exit_path']).decode().strip()=='0','cache_producer0')
    module=old.load_module(b['cap_module']['path']);fixed=b['fixed_module'];pins.add(fixed['path'],fixed['sha256'])
    rt=module.configure_runtime(args.source,fixed['path']);torch.set_num_threads(1)
    data,actual=rt.read_feature_cache(cache['path'],known_sha=cache['sha256']);need(actual==cache['sha256'],'cache_SHA')
    old.verify_cache_manifest(data['manifest'],cache['manifest'],producer,actual)
    need(data['manifest']['cohort']=='original_val' and data['manifest']['recall']=='single','val_only')
    models=old.load_models(module,rt,b,5,193,'cpu',pins)
    before=old.model_digest(models);row=data['rows'][0];natural=old.natural_row(row);gamma=old.route_gamma(row['query_observation'])
    value=inspect_cost(models,natural,gamma,'cpu');after=old.model_digest(models);need(before==after,'model_state_changed');pins.finish()
    result=dict(schema='cap10_validation_cost_contract_diagnostic_20261025',status='COMPLETE_DIAGNOSTIC',cap=5,seed=193,pool='single',cohort='original_val',row_index=0,
      diagnostic=value,input_sha256=pins.values,inputs_unchanged=True,model_state_before=before,model_state_after=after,
      pid=os.getpid(),parent_pid=os.getppid(),utc=utc(),training=False,test_read=False,forward_bank_calls=0,rank_metrics_computed=False,external_api_calls=0,selection_performed=False,promotion=False)
    write(out/'summary.json',result);return result

def main():
    p=argparse.ArgumentParser()
    for k in ('source','bindings','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=safe(a.out);out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(args=vars(a),argv=sys.argv,pid=os.getpid(),started_utc=utc()))
    try:run(a,out);(out/'EXIT_CODE').write_text('0\n')
    except BaseException as exc:write(out/'failure.json',dict(error=repr(exc),utc=utc()));(out/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()

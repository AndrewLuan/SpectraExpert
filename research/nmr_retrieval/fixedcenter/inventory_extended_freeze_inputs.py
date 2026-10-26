"""功能概述：在专项审核前，只读收集扩展冻结器实际验证的全部输入SHA。
collect在review调用处捕获闭包并强制停止，阻止写冻结文件；同时拦截测试tensor与预测读取。
此工具不批准方案，不执行freeze、test、训练或模型推理。main只保存独立inventory凭据。
"""
import argparse,builtins,datetime,hashlib,io,json,os,sys
from pathlib import Path
import build_pool_extended_frozen as builder

EXPECTED_BUILDER='d99bdb850771e8afc172ba4d5ff7ebb36c6572bd0d888c49e3c7a197b0aaafe6'
class InventoryStop(Exception):pass
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_path(path):
    p=Path(path).resolve()
    assert p.suffix=='.json' and '/sealed/' not in str(p),'operator_json_path_must_not_be_test_data'
    return p


def collect(args):
    assert digest(builder.__file__)==EXPECTED_BUILDER,'builder_source_drift'
    out=Path(args.out).resolve()
    assert not out.exists(),'freeze_output_must_not_exist'
    pending=json.loads(json_path(args.review).read_text())
    assert pending.get('status')=='PENDING' and pending.get('scope')=='freeze_plan_and_implementation','explicit_pending_review_required'
    captured={}
    original_review=builder.old.review
    original_open=builtins.open;original_io_open=io.open
    def check(path,mode):
        if isinstance(path,(str,bytes,Path)):
            p=Path(path).resolve()
            if ('/sealed/features/' in str(p) or '/test/' in str(p)) and p.suffix=='.pt':
                raise AssertionError('inventory_forbidden_test_tensor_read')
            if ('/sealed/' in str(p) or '/test/' in str(p)) and p.name in ('predictions.jsonl','ranks.jsonl','rankings.jsonl'):
                raise AssertionError('inventory_forbidden_test_prediction_read')
            if any(v in mode for v in ('w','a','x','+')):
                raise AssertionError('inventory_forbidden_write')
    def guarded_open(file,mode='r',*a,**kw):
        check(file,mode);return original_open(file,mode,*a,**kw)
    def guarded_io_open(file,mode='r',*a,**kw):
        check(file,mode);return original_io_open(file,mode,*a,**kw)
    def capture(review,required):
        assert review==pending
        assert isinstance(required,dict) and required
        captured.update(required)
        raise InventoryStop()
    active=[False]
    def audit(event,values):
        if not active[0]:return
        if event=='open':
            path,mode,flags=values
            check(path,mode or '')
            if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                raise AssertionError('inventory_forbidden_write')
        if event in ('os.mkdir','os.remove','os.rename','os.rmdir','os.chmod','os.chown','os.utime','os.symlink','os.link','os.truncate'):
            raise AssertionError('inventory_forbidden_mutation:'+event)
    sys.addaudithook(audit)
    builder.old.review=capture;builtins.open=guarded_open;io.open=guarded_io_open;active[0]=True
    try:
        try:builder.build(args,out)
        except InventoryStop:pass
        else:raise AssertionError('inventory_did_not_stop_before_freeze')
    finally:
        active[0]=False
        builder.old.review=original_review;builtins.open=original_open;io.open=original_io_open
    assert captured and not out.exists()
    return dict(schema='pool_extended_freeze_input_inventory_20261023',status='INVENTORY_ONLY_NOT_REVIEWED',
        collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        builder_sha256=EXPECTED_BUILDER,builder_args=vars(args),
        required_sha256=captured,freeze_output_exists=False,review_not_approved=True,
        test_cache_values_read=False,test_predictions_read=False,actual_freeze=False,actual_test=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--builder-args-file',required=True);p.add_argument('--inventory-output',required=True);a=p.parse_args()
    output=Path(a.inventory_output).resolve();assert not output.exists()
    args_path=json_path(a.builder_args_file);raw=args_path.read_bytes();args=argparse.Namespace(**json.loads(raw))
    result=collect(args);result['args_file_sha256']=hashlib.sha256(raw).hexdigest();result['operator_sha256']=digest(__file__)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'required_inputs':len(result['required_sha256']),'inventory':str(output),'sha256':digest(output)}))
if __name__=='__main__':main()

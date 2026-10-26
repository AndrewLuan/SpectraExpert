"""功能概述：独立复核Mac/JJ已经关闭的定位结果中10个已保存logit的本地原dtype代数。
主要职责：read_exact只读固定JSON并核SHA；typed记录tensor dtype/shape/原始字节；
main核两个结果的同ref/logit身份，运行float32 division/tanh/square/local，比较已报告local。
代码库关系：输入是Mac401245与JJda8ab结果和Mac预检b4ed；不导入作者helper、不读取shard、
模型/缓存/预测test或科学全集，不调用网络或autograd。此复核不能独立重证saved dlogit VJP。
输入输出与副作用：固定读取3个JSON；新建一个独立JSON记录source/runtime/elapsed/RSS/CPU4。
失败立即退出且不补样本、不重试、不放宽容差；差异仅限定当前Mac ARM与当前JJ证据。
"""
from pathlib import Path
import hashlib,json,os,platform,resource,sys,time
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="4"
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
INPUTS={
 "Mac":(ROOT/"saved_jacobian_mac_stage_attempt1/results/mac_localization.json","401245d3606f33ad72bdb5924cee3fda85aae089123800a84cbcbb7fab7ef68a"),
 "JJ":(ROOT/"collected/JJ_algebra_attempt1/results/JJ_algebra.json","da8ab701ed5f473045f17b43e0b7fbcad1249ce72f2b50170d5b8d1061b07a99"),
 "preflight":(ROOT/"Mac_saved_algebra_runtime_preflight.json","b4ed369c30b7ab4f320d83552f7dc984753c3268f00e0882987b0ac675181676")
}
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read_exact(name):
    path,expected=INPUTS[name]
    assert sha(path)==expected
    return json.loads(path.read_text())
def typed(value):
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())
def main():
    started=time.monotonic()
    global torch
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    mac,jj,pre=(read_exact(x)for x in("Mac","JJ","preflight"))
    assert mac["selection"]==jj["selection"] and mac["shard"]==jj["shard"]
    assert mac["selection"]["cap"]==5 and mac["selection"]["positions"]==list(range(10))
    assert str(torch.__version__)==pre["torch"] and sys.executable==pre["python"] and platform.machine()=="arm64"
    me=mac["current_Mac_first_failure"]["evidence"];je=jj["evidence"]
    assert me["saved_logit"]==je["saved_logit"] and len(je["saved_logit"])==10
    z=torch.tensor(je["saved_logit"],dtype=torch.float32)
    assert typed(z)==je["cpu_vs_cuda"]["logit"]["actual"]
    division=z/5
    tanh=torch.tanh(division)
    square=tanh.square()
    local=1-square
    reported_mac=torch.tensor(me["local_native_CPU"],dtype=torch.float32)
    reported_JJ_CPU=torch.tensor(je["native_CPU_local"],dtype=torch.float32)
    reported_JJ_CUDA=torch.tensor(je["native_CUDA_local"],dtype=torch.float32)
    assert typed(local)==typed(reported_mac)
    assert typed(reported_JJ_CPU)==typed(reported_JJ_CUDA)==je["cpu_vs_cuda"]["local"]["actual"]
    unequal=(local!=reported_JJ_CPU).nonzero().flatten().tolist()
    assert unequal==[4,8]
    elapsed=time.monotonic()-started
    peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert elapsed<=30 and peak<=1024**3
    for name in INPUTS:read_exact(name)
    out=ROOT/"independent_saved_logit_backend_review.json"
    assert not out.exists()
    result=dict(schema="saved_logit_backend_independent_algebra_review_v1",status="PASS_CURRENT_MAC_JSON_REPRODUCTION",independent=True,
      source_sha256=sha(Path(__file__)),inputs={name:dict(path=str(path),sha256=expected)for name,(path,expected)in INPUTS.items()},
      runtime=dict(python=sys.executable,python_version=sys.version,torch=str(torch.__version__),platform=platform.platform(),machine=platform.machine(),cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads()),
      resources=dict(elapsed_seconds=elapsed,peak_self_RSS_bytes=peak,max_seconds=30,max_RSS_bytes=1024**3,execution_supervisor="NONE_SMALL_EXPLICITLY_AUTHORIZED_JSON_ONLY_CPU_REVIEW",ru_maxrss_units="bytes_on_Darwin"),
      actual_shard_decodes=0,new_NN_calls=0,model_loads=0,training=False,test_read=False,API_calls=0,autograd_calls=0,
      same_logit_typed=typed(z),current_Mac_primitives={name:typed(value)for name,value in dict(division=division,tanh=tanh,square=square,local=local).items()},
      current_Mac_local=local.tolist(),reported_Mac_local=reported_mac.tolist(),reported_JJ_CPU_local=reported_JJ_CPU.tolist(),reported_JJ_CUDA_local=reported_JJ_CUDA.tolist(),
      current_Mac_local_bit_equal_reported_Mac=True,JJ_CPU_CUDA_local_bit_equal=True,
      current_Mac_vs_JJ_local_unequal_positions=unequal,
      JJ_CPU_division_hash_matches_current_Mac=typed(division)==je["cpu_vs_cuda"]["division"]["actual"],
      JJ_CUDA_division_hash_matches_current_Mac=typed(division)==je["cpu_vs_cuda"]["division"]["expected"],
      scope="10 saved logits already emitted in frozen result JSON; no new tensor payload reads or population analysis.",
      interpretation="Current Mac ARM float32 local differs at positions4/8 from both current JJ CPU and CUDA. No general CPU-vs-CUDA cause claim; no isolated division/tanh attribution without a same-input Mac/JJ primitive control.",
      original_guard_released=False,original_rtol=2e-5,original_atol=2e-6,saved_network_VJP_recertified=False)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(out);print(sha(out))
if __name__=="__main__":main()

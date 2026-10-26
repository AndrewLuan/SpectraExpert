"""功能概述：非作者合成核验新sealed入口的混合精度与真实runpy导入路径。
只运行人工Tensor/冷Python导入，不读实际test缓存、预测、指标或失败案例。
冷子进程均有30秒显式预算，无固定sleep或自动重跑。
"""
import subprocess,sys,unittest
from pathlib import Path
import torch
import cap_test_evaluate_ranker as e
from test_evaluate_cap_ranker import models,row

HERE=Path(__file__).resolve().parent
class NativeAdapterIndependentTests(unittest.TestCase):
    def test_mixed64_cost_equals_unmodified_torch_formula_bitwise(self):
        _,ms=models();r=row();r['coarse_scores']=r['coarse_scores'].double()+1.234567891e-8
        natural=e.natural_row(r);actual=e.score_natural(ms,natural,.25,'cpu',5)
        with torch.inference_mode():
            corr=[saved['alpha']*model.row_delta(natural,scale,'cpu')*scale[0] for model,scale,saved in ms]
            expected=natural['coarse_scores']+.75*corr[0]+.25*corr[1]
        self.assertEqual(actual['final_cost_dtype'],'torch.float64')
        self.assertTrue(torch.equal(actual['cost'],expected))
        self.assertTrue(torch.any(actual['cost']!=actual['cost'].float().double()))
    def test_runpy_from_root_imports_exact_new_contract(self):
        script=HERE/'cap_test_evaluate_ranker.py'
        code='import runpy,sys;sys.path.insert(0,'+repr(str(HERE.parent))+');d=runpy.run_path('+repr(str(script))+',run_name="synthetic_import_only");print(d["_actual_contract"].__file__)'
        r=subprocess.run([sys.executable,'-B','-c',code],cwd=HERE.parent,capture_output=True,text=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(Path(r.stdout.strip()).resolve(),HERE/'cap_test_evaluation_common.py')
    def test_cached_old_contract_is_rejected_before_metadata_or_tensor(self):
        script=HERE/'cap_test_evaluate_ranker.py'
        code='import runpy,sys;sys.path.insert(0,'+repr(str(HERE.parent))+');import cap_test_evaluation_common;runpy.run_path('+repr(str(script))+',run_name="synthetic_import_only")'
        r=subprocess.run([sys.executable,'-B','-c',code],cwd=HERE.parent,capture_output=True,text=True,timeout=30)
        self.assertNotEqual(r.returncode,0);self.assertIn('native_contract_module_path_mismatch',r.stderr)

if __name__=='__main__':unittest.main()

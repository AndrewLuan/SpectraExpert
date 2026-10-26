"""功能概述：新统计gate的原生32/64精度作者合成核验，不读取真实cache或预测。
EPS对照直接调用torch原始tensor排序；64位微小差异不得被降成32位，dtype声明必须
遵守本轮冻结fp32校正与原始coarse类型的提升。旧8c72源码和原测试不修改。
"""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
import summarize_sealed_cap_results as s


def row(dtype='torch.float64',costs=None,target=0,hit=True):
    dt=np.float64 if dtype=='torch.float64' else np.float32
    values=np.asarray(costs if costs is not None else range(512),dtype=dt)
    ids=list(range(512));order=np.lexsort((np.array(ids),values))
    native=torch.tensor(values,dtype=torch.float64 if dt is np.float64 else torch.float32)
    rank=int((native<=native[target]+1e-7).sum()) if hit else 700
    other=int(order[order!=target][0]);tc=float(values[target])
    return dict(query_identity=dict(query_order=0,variant='observed',source='HMDB',record_id='0',smiles='C',group='g'),
      coarse_dtype=dtype,correction_dtypes=['torch.float32']*2,final_cost_dtype=dtype,
      natural_candidate_indices=ids,natural_candidate_costs=values.tolist(),natural_candidate_coarse_costs=values.tolist(),
      winner_candidate_index=int(order[0]),target_candidate_index=target if hit else None,target_cost=tc if hit else None,
      margin=tc-float(values[other]) if hit else None,target_in_pool=hit,rank=rank,RR=1/rank,
      in_pool_target_rank=rank if hit else None,cached_outside_pool_rank=None if hit else rank,
      top1=rank<=1,top5=rank<=5,top10=rank<=10,mask_content_sha256='a'*64)


class NativeRowTests(unittest.TestCase):
    def test_native32_and_native64_rows_both_exact(self):
        for dtype in ('torch.float32','torch.float64'):s.validate_payload_row(row(dtype,target=7))
    def test_fp64_sub_fp32_difference_preserved(self):
        values=[0.,1.00000002e-7]+[1.]*510
        r64=row('torch.float64',values);r32=row('torch.float32',values)
        self.assertEqual(r64['rank'],1);self.assertEqual(r32['rank'],2)
        s.validate_payload_row(r64);s.validate_payload_row(r32)
    def test_native_tensor_EPS_cast_before_add(self):
        target=float(np.float32(-1e-7));values=[target,0.]+[1.]*510
        self.assertEqual(s.native_threshold(target,np.float32),0.)
        self.assertLess(float(np.float32(target+1e-7)),0.)
        for dtype,expected in [('torch.float32',2),('torch.float64',1)]:
            r=row(dtype,values);self.assertEqual(r['rank'],expected);s.validate_payload_row(r)
    def test_unjustified_double_promotion_rejected(self):
        r=row('torch.float32');r['final_cost_dtype']='torch.float64'
        with self.assertRaisesRegex(ValueError,'native_vector_promotion_dtype'):s.validate_payload_row(r)
    def test_unsupported_dtype_or_non_fp32_frozen_correction_rejected(self):
        for coarse,corr in [('torch.float16',['torch.float32']*2),('torch.float64',['torch.float32','torch.float64'])]:
            r=row();r.update(coarse_dtype=coarse,correction_dtypes=corr)
            with self.assertRaisesRegex(ValueError,'registered_native_operand_dtypes'):s.validate_payload_row(r)
    def test_declared32_nonrepresentable_cost_or_coarse_rejected(self):
        for key in ('natural_candidate_costs','natural_candidate_coarse_costs'):
            r=row('torch.float32');r[key][2]=1/3
            with self.assertRaisesRegex(ValueError,'exact_native_declared'):s.validate_payload_row(r)
    def test_cost_dtype_fields_required(self):
        r=row();del r['final_cost_dtype']
        with self.assertRaisesRegex(ValueError,'native_vector_promotion_dtype'):s.validate_payload_row(r)
    def test_rank_lie_rejected_and_outside_fallback_preserved(self):
        r=row();r.update(rank=2,RR=.5,in_pool_target_rank=2,top1=False)
        with self.assertRaisesRegex(ValueError,'native_rank_from_declared_dtype_costs'):s.validate_payload_row(r)
        s.validate_payload_row(row(hit=False))


def native_full_fixture(root):
    spec=importlib.util.spec_from_file_location('_native_statistics_registered_fixture',HERE.parent/'test_summarize_sealed_cap_results_independent.py')
    fx=importlib.util.module_from_spec(spec);spec.loader.exec_module(fx)
    child=fx.CHILD.replace('natural_candidate_costs=[0.]*512,',"natural_candidate_costs=[0.]*512,natural_candidate_coarse_costs=[0.]*512,coarse_dtype='torch.float64',correction_dtypes=['torch.float32']*2,final_cost_dtype='torch.float64',")
    child=child.replace("(out/'summary.json').write_text(json.dumps(s));", "s['evaluator']=b['evaluator']\n(out/'summary.json').write_text(json.dumps(s));")
    with patch.object(fx,'CHILD',child):f=fx.ExecutionFixture(root)
    f.bindings['evaluator']=dict(path=str(f.script),sha256=fx.digest(f.script))
    return fx.full_metadata(f).launch()


class NativeFullEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.f=native_full_fixture(cls.tmp.name)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def tearDown(self):self.f.restore()
    def test_real_temporary12_driver_full_metadata_and_native64_payload(self):
        seen=[];original=s.c.Pins.raw
        def read(pins,path,expected=None):
            if Path(path).name=='predictions.jsonl':seen.append(str(path))
            return original(pins,path,expected)
        def matrix(raw,protocol):
            self.assertEqual(set(raw),set(s.matrix.CONFIGS))
            for seeds in raw.values():
                for rows in seeds.values():
                    self.assertEqual(len(rows),1371);self.assertTrue(all(r['final_cost_dtype']=='torch.float64' for r in rows))
            return {'metrics':[dict(synthetic_only=True)]}
        out=self.f.root/'statistics';out.mkdir(exist_ok=True)
        with patch.object(s.c.Pins,'raw',read),patch.object(s.matrix,'analyze_matrix',side_effect=matrix):result=s.run(self.f.args,out)
        self.assertEqual(len(seen),12);self.assertFalse(result['selection_performed'])
    def test_failed_last_job_blocks_first_prediction(self):
        path=Path(self.f.queue['jobs'][-1]['access_log']);value=json.loads(path.read_text());value['internal_exit']=False;path.write_text(json.dumps(value))
        seen=[];original=s.c.Pins.raw
        def read(pins,path,expected=None):
            if Path(path).name=='predictions.jsonl':seen.append(str(path));raise AssertionError('first_payload_before_all_metadata')
            return original(pins,path,expected)
        with patch.object(s.c.Pins,'raw',read),self.assertRaisesRegex(ValueError,'strict_int0'):s.run(self.f.args,self.f.root/'unused')
        self.assertEqual(seen,[])


class ColdNativeImportTests(unittest.TestCase):
    def test_runpy_from_parent_resolves_exact_new_common(self):
        script=HERE/'summarize_sealed_cap_results.py'
        code='import runpy,sys;sys.path.insert(0,'+repr(str(HERE.parent))+');d=runpy.run_path('+repr(str(script))+',run_name="synthetic_import_only");print(d["c"].__file__)'
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=HERE.parent,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(Path(result.stdout.strip()).resolve(),HERE/'cap_test_evaluation_common.py')
    def test_cached_old_common_refused_before_metadata(self):
        script=HERE/'summarize_sealed_cap_results.py'
        code='import runpy,sys;sys.path.insert(0,'+repr(str(HERE.parent))+');import cap_test_evaluation_common;runpy.run_path('+repr(str(script))+',run_name="synthetic_import_only")'
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=HERE.parent,capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0);self.assertIn('exact_native_common_import_required',result.stderr)


class ActualNativeProducerTests(unittest.TestCase):
    def test_actual_native_evaluator_artificial_loader64_output_to_statistics_oracle(self):
        from test_cap_test_evaluate_ranker import SealedFixture
        with tempfile.TemporaryDirectory() as tmp:
            fixture=SealedFixture(tmp)
            for row in fixture.data['rows']:row['coarse_scores']=row['coarse_scores'].double()+1.234567891234e-8
            result=fixture.run();rows=[json.loads(x) for x in (fixture.out/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),6)
            for row in rows:
                self.assertEqual(row['final_cost_dtype'],'torch.float64');s.validate_payload_row(row)
            self.assertEqual(result['evaluator'],fixture.binding['evaluator'])


if __name__=='__main__':unittest.main()

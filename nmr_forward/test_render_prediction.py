# 功能概述：验证预测JSON到一阶棒谱/理论曲线的物理单位、质量守恒及输出边界。
# 主要职责：RenderPredictionTests覆盖二重/三重峰、预测积分优先级、未知类别、
# 缺失频率/J、Lorentzian无限域归一与有限窗口尾部、拒绝覆写及可执行CLI。
# 代码库关系：仅调用render_prediction标准库接口；CLI测试用临时目录及当前Python。
# 输入输出与副作用：只写临时合成JSON/CSV，不读取真实标签、checkpoint或服务器。
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from .render_prediction import render_prediction, lorentzian_curve, write_outputs
except ImportError:
    from render_prediction import render_prediction, lorentzian_curve, write_outputs


class RenderPredictionTests(unittest.TestCase):
    def prediction(self, multiplicity='t'):
        return dict(nucleus='1H',checkpoint={'sha256':'a'*64},input={'frequency_mhz':400},
                    proton_sites=[dict(site_id='H1',shift_ppm=1.,structural_integration_h=1.,
                                       multiplicity_argmax=multiplicity,
                                       peak_group_j_candidates_hz=[8.,123.,456.])])

    def test_triplet_area_hz_conversion_and_unused_candidates(self):
        result=render_prediction(self.prediction())
        self.assertEqual([s['ppm'] for s in result['sticks']],[.98,1.,1.02])
        self.assertEqual([s['area_h'] for s in result['sticks']],[.25,.5,.25])
        self.assertEqual(result['resolved_area_h'],1.)
        self.assertEqual(result['source_checkpoint_sha256'],'a'*64)
        self.assertEqual(result['unresolved'],[])
        highfield=render_prediction(self.prediction(),800)
        self.assertEqual([s['ppm'] for s in highfield['sticks']],[.99,1.,1.01])

    def test_predicted_reported_area_takes_precedence(self):
        prediction=self.prediction('d')
        prediction['proton_sites'][0]['predicted_reported_integration_h']=.2
        result=render_prediction(prediction)
        self.assertEqual([s['area_h'] for s in result['sticks']],[.1,.1])
        self.assertAlmostEqual(result['input_area_h'],.2)

    def test_unknown_broad_and_missing_mult_are_unresolved_without_placeholder(self):
        for mult in ('m','br','other',None,'unparsed'):
            result=render_prediction(self.prediction(mult))
            self.assertEqual(result['sticks'],[])
            self.assertEqual(len(result['unresolved']),1)
            self.assertEqual(result['unresolved_area_h'],1.)
            self.assertFalse(result['all_sites_have_supported_first_order_patterns'])

    def test_missing_frequency_rejected_override_accepted(self):
        prediction=self.prediction()
        prediction['input']['frequency_mhz']=None
        with self.assertRaises(ValueError):
            render_prediction(prediction)
        self.assertEqual(render_prediction(prediction,600)['frequency_mhz'],600)
        for bad in (0,-1,float('nan'),True):
            with self.assertRaises(ValueError):
                render_prediction(prediction,bad)

    def test_missing_required_j_unresolved_but_singlet_needs_none(self):
        prediction=self.prediction('dd')
        prediction['proton_sites'][0]['peak_group_j_candidates_hz']=[8.]
        result=render_prediction(prediction)
        self.assertEqual(result['unresolved'][0]['reasons'],['insufficient_J_candidates'])
        prediction['proton_sites'][0].update(multiplicity_argmax='s',peak_group_j_candidates_hz=[])
        self.assertEqual(render_prediction(prediction)['sticks'][0]['area_h'],1)

    def test_resolved_plus_unresolved_mass_conservation(self):
        prediction=self.prediction('d')
        prediction['proton_sites'].append(dict(site_id='H2',shift_ppm=4.,structural_integration_h=2.,
                                               multiplicity_argmax='m'))
        result=render_prediction(prediction)
        self.assertEqual(result['input_area_h'],result['resolved_area_h']+result['unresolved_area_h'])
        self.assertEqual(result['resolved_area_h'],1)

    def test_lorentzian_fwhm_and_infinite_area_without_tail_renormalization(self):
        result=render_prediction(self.prediction('s'))
        # 400MHz处40Hz FWHM=0.1ppm；窗口1±0.05只含无限域面积的一半。
        curve,meta=lorentzian_curve(result,40,.95,1.05,1001)
        self.assertAlmostEqual(meta['analytic_window_area_h'],.5)
        self.assertAlmostEqual(meta['trapezoid_grid_area_h'],.5,places=6)
        center=curve[500][1]
        self.assertAlmostEqual(center,1/(math.pi*.05))
        self.assertAlmostEqual(curve[0][1],center/2)
        self.assertEqual(meta['linewidth_source'],'user_specified_not_learned')

    def test_outputs_refuse_overwrite_and_bad_curve_leaves_no_json(self):
        result=render_prediction(self.prediction())
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'sticks.json'
            csv_path=Path(directory)/'curve.csv'
            with self.assertRaises(ValueError):
                write_outputs(result,out,csv_path)
            self.assertFalse(out.exists())
            self.assertFalse(csv_path.exists())
            write_outputs(result,out,csv_path,1.,0.,2.,1001)
            before=out.read_bytes()
            with self.assertRaises(FileExistsError):
                write_outputs(result,out)
            self.assertEqual(out.read_bytes(),before)
            with self.assertRaises(FileExistsError):
                write_outputs(result,Path(directory)/'other.json',csv_path,1.,0.,2.,1001)
            self.assertFalse((Path(directory)/'other.json').exists())

    def test_cli_creates_json_and_csv_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'prediction.json'
            source.write_text(json.dumps(self.prediction()))
            command=[sys.executable,str(Path(__file__).with_name('render_prediction.py')),
                     '--prediction',str(source),'--out',str(root/'sticks.json'),
                     '--curve-csv',str(root/'curve.csv'),'--linewidth-hz','1',
                     '--ppm-min','0','--ppm-max','2','--points','1001']
            process=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            output=json.loads((root/'sticks.json').read_text())
            self.assertEqual(len(output['source_prediction_sha256']),64)
            self.assertEqual(output['curve']['points'],1001)
            self.assertEqual(len((root/'curve.csv').read_text().splitlines()),1002)


if __name__ == '__main__':
    unittest.main()

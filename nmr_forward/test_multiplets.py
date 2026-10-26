# 功能概述：验证峰表规范化与理论一阶棒状谱的物理和异常边界。
# 主要职责：MultipletTests 检查三重峰、四重峰、dd 的面积、MHz 单位换算、
# 简并合并及异常标签拒绝；评估测试确保 J 次序/符号约定和缺失数据计数正确。
# 代码库关系：只依赖同目录 multiplets.py 与标准库 unittest；在 SpectrExpert 下
# 执行 python -m unittest nmr_forward.test_multiplets，无实测预测准确性声明。
# 输入输出与副作用：使用确定性人工标签，无随机状态、文件修改、网络或训练。
import unittest
try:
    from .multiplets import (parse_annotation, normalize_multiplicity,
                            simulate_sticks, evaluate_matched_annotations)
except ImportError:
    from multiplets import (parse_annotation, normalize_multiplicity,
                           simulate_sticks, evaluate_matched_annotations)


class MultipletTests(unittest.TestCase):
    def annotation(self, multiplicity='t', coupling=(7,), area=3):
        return dict(shift_mid=1.0, shift_min=1.0, shift_max=1.0,
                    integration_h=area, multiplicity=multiplicity,
                    coupling_hz=list(coupling))

    def test_triplet_and_quartet_integrals(self):
        triplet = simulate_sticks(self.annotation(), 400)
        for actual, expected in zip(triplet['x_ppm'], [0.9825, 1., 1.0175]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(triplet['area_h'], [.75, 1.5, .75])
        quartet = simulate_sticks(self.annotation('q', (7,), 2), 400)
        self.assertEqual(quartet['area_h'], [.25, .75, .75, .25])
        self.assertAlmostEqual(quartet['x_ppm'][0], 1 - 10.5/400)
        self.assertAlmostEqual(sum(quartet['area_h']), 2)

    def test_double_doublet_and_degeneracy(self):
        dd = simulate_sticks(self.annotation('dd', (8, 2), 1), 500)
        self.assertEqual(dd['area_h'], [.25]*4)
        for actual, expected in zip(dd['x_ppm'], [.99, .994, 1.006, 1.01]):
            self.assertAlmostEqual(actual, expected)
        degenerate = simulate_sticks(self.annotation('dd', (8, 8), 1), 500)
        self.assertEqual(degenerate['area_h'], [.25, .5, .25])
        zero = simulate_sticks(self.annotation('t', (0,), 3), 500)
        self.assertEqual(zero['area_h'], [3.])
        self.assertEqual(zero['x_ppm'], [1.])

    def test_composites_and_field_scaling(self):
        dt = simulate_sticks(self.annotation('dt', (8, 2), 2), 400)
        td = simulate_sticks(self.annotation('td', (8, 2), 2), 400)
        self.assertEqual(len(dt['area_h']), 6)
        self.assertNotEqual(dt['x_ppm'], td['x_ppm'])
        for m,j in [('tt',(9,2)),('dq',(9,2)),('qd',(9,2)),('quint',(7,)),('sept',(7,))]:
            rendered = simulate_sticks(self.annotation(m,j,5), 600)
            self.assertAlmostEqual(sum(rendered['area_h']),5)
        highfield = simulate_sticks(self.annotation(), 800)
        lowfield = simulate_sticks(self.annotation(), 400)
        self.assertAlmostEqual(1-highfield['x_ppm'][0],(1-lowfield['x_ppm'][0])/2)

    def test_no_invented_internal_peaks(self):
        for m in ['m', 'br s', 'ddd', None]:
            result = simulate_sticks(self.annotation(m, (), 2), 400)
            self.assertEqual(result['x_ppm'], [])
            self.assertIn('unsupported_internal_pattern', result['flags'])
        missing = simulate_sticks(self.annotation('dd', (8,)), 400)
        self.assertEqual(missing['x_ppm'], [])
        self.assertIn('missing_J', missing['flags'])
        for frequency in [None,0,float('nan'),True]:
            self.assertEqual(simulate_sticks(self.annotation(),frequency)['x_ppm'],[])
        for area in [None,0,-1,float('inf')]:
            self.assertEqual(simulate_sticks(self.annotation(area=area),400)['area_h'],[])
        self.assertEqual(simulate_sticks(self.annotation('s',(7,)),400)['area_h'],[])

    def test_normalization_and_interval(self):
        self.assertEqual(normalize_multiplicity('doublet of triplets'),'dt')
        self.assertEqual(normalize_multiplicity('br.s'),'br')
        self.assertEqual(normalize_multiplicity('quintet'),'quint')
        a=self.annotation('s',())
        a.update(shift_mid=None,shift_min=1.,shift_max=2.)
        self.assertEqual(parse_annotation(a)['shift_mid'],1.5)
        self.assertEqual(simulate_sticks(a,400)['x_ppm'],[1.5])
        a.update(shift_mid=3.)
        self.assertIn('invalid_shift_interval',parse_annotation(a)['flags'])
        self.assertEqual(simulate_sticks(a,400)['x_ppm'],[])
        result=simulate_sticks(self.annotation(),400)
        self.assertEqual(result['representation'],'theoretical_first_order_sticks')
        self.assertIn('strong_coupling_unknown',result['flags'])

    def test_matched_metrics(self):
        target=self.annotation('dd',(8,2),2)
        prediction=self.annotation('dd',(-2,-8),2)
        metrics=evaluate_matched_annotations([prediction],[target])
        self.assertEqual(metrics['J_magnitude_mae_hz'],0)
        self.assertEqual(metrics['multiplicity_accuracy'],1)
        self.assertEqual(metrics['counts']['J_magnitude_mae_hz'],2)
        empty=evaluate_matched_annotations([],[])
        self.assertIsNone(empty['shift_mae_ppm'])
        with self.assertRaises(ValueError):
            evaluate_matched_annotations([prediction],[])


if __name__ == '__main__':
    unittest.main()

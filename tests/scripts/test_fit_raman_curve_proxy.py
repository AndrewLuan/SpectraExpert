"""功能概述：验证轻量 Raman 曲线代理的数值方向、覆盖边界与训练角色保护。
主要职责：使用人工单峰验证 identity、已知位移及展宽的积分守恒；检查所有参数边界
共用的评价掩码、分子组等权与禁止用验证/受保护测试拟合的入口断言。
代码库关系：直接加载 scripts/fit_raman_curve_proxy.py，不读服务器数据，不运行正式拟合。
输入输出与副作用：仅构造内存数组，可用 unittest 执行，不写数据或访问网络。
"""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'fit_raman_curve_proxy.py'
SPEC = importlib.util.spec_from_file_location('raman_proxy', SCRIPT)
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


class CurveProxyTests(unittest.TestCase):
    def setUp(self):
        self.axis = np.linspace(0, 4000, 3000)
        self.curve = np.exp(-.5 * ((self.axis - 1200) / 5) ** 2)[None, :]
        self.coverage = np.ones_like(self.curve, dtype=bool)

    def test_known_peak_motion_and_integrated_intensity(self):
        actual, mask = proxy.transform(self.axis, self.curve, self.coverage, proxy.IDENTITY)
        np.testing.assert_allclose(actual, self.curve)
        self.assertTrue(mask.all())
        actual, _ = proxy.transform(self.axis, self.curve, self.coverage, (.98, 20, 3))
        self.assertLess(abs(self.axis[actual.argmax()] - (.98 * 1200 + 20)), 2)
        ratio = np.trapezoid(actual[0], self.axis) / np.trapezoid(self.curve[0], self.axis)
        self.assertAlmostEqual(ratio, 1, delta=.01)

    def test_fixed_support_is_valid_for_all_parameter_corners(self):
        self.coverage[:, :375] = False
        self.coverage[:, 2800:] = False
        support = proxy.fixed_support(self.axis, self.coverage, self.coverage)
        for scale in proxy.BOUNDS[0]:
            for shift in proxy.BOUNDS[1]:
                for width in proxy.BOUNDS[2]:
                    _, valid = proxy.transform(self.axis, self.curve, self.coverage, (scale, shift, width))
                    self.assertTrue(np.all(~support | valid))

    def test_repeat_measurements_do_not_overweight_groups(self):
        weights = proxy.group_weights([{'group_hash': g} for g in ('a', 'a', 'b')])
        np.testing.assert_allclose(weights, [.25, .25, .5])

    def test_fit_rejects_validation_and_protected_roles_before_access(self):
        for role in ('surrogate_validation', 'protected_final_test'):
            with self.assertRaises(AssertionError):
                proxy.fit_source(None, [{'role': role}])


if __name__ == '__main__':
    unittest.main()

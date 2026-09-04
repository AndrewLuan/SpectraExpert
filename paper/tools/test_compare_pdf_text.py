"""功能概述：验证论文 PDF 文本归一化与差异检测，确保排版空白可忽略而数字和化学字符串变化不会被吞掉。

主要职责：测试空白归一化、分页符处理、内容一致性和关键数值差异；不创建或修改论文 PDF。
代码库关系：直接测试 compare_pdf_text.py，供最终稿内容锁定检查使用。
"""

from compare_pdf_text import normalize_text
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class NormalizeTextTest(unittest.TestCase):
    def test_collapses_layout_whitespace_and_page_breaks(self) -> None:
        left = "SpectraExpert\n\nGED 16.93\fAppendix A"
        right = "SpectraExpert GED 16.93 Appendix A"
        self.assertEqual(normalize_text(left), normalize_text(right))

    def test_preserves_numeric_changes(self) -> None:
        self.assertNotEqual(normalize_text("GED 16.93"),
                            normalize_text("GED 16.39"))

    def test_preserves_smiles_changes(self) -> None:
        self.assertNotEqual(
            normalize_text("FC1(F)CCCNC1"),
            normalize_text("CC1CCC2CC2(O)C1"),
        )


if __name__ == "__main__":
    unittest.main()

"""功能概述：提取并规范化两份论文 PDF 的文本，生成可审计的一致性差异。

主要职责：extract_text 调用 pdftotext 读取 PDF；normalize_text 只归一化 Unicode 与空白并保留数字和化学字符串；
compare_pdfs 返回规范化文本是否一致及逐词 unified diff；main 提供命令行入口，并可将差异写入指定路径。
代码库关系：读取用户提供的最终稿基线和 paper/build/main.pdf；差异供 CONTENT_EQUIVALENCE.md 审计，不参与 LaTeX 编译。
输入输出与副作用：输入是两个 PDF 路径；返回比较结果，并在指定 --diff-output 时覆盖该 UTF-8 文本文件；pdftotext 失败时直接返回非零状态。
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import unicodedata
from pathlib import Path


def extract_text(pdf: Path) -> str:
    """Return raw Poppler text for one PDF."""
    result = subprocess.run(
        ["pdftotext", "-raw", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def normalize_text(text: str) -> str:
    """Normalize representation-only differences while preserving content tokens."""
    normalized = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    return " ".join(normalized.replace("\f", " ").split())


def compare_pdfs(baseline: Path, candidate: Path) -> tuple[bool, str]:
    """Compare normalized PDF text and return equality plus a word-level diff."""
    expected = normalize_text(extract_text(baseline)).split()
    actual = normalize_text(extract_text(candidate)).split()
    if expected == actual:
        return True, ""
    diff = "\n".join(
        difflib.unified_diff(
            expected,
            actual,
            fromfile=str(baseline),
            tofile=str(candidate),
            lineterm="",
            n=3,
        )
    )
    return False, diff


def main() -> int:
    """Parse CLI arguments, compare PDFs, optionally persist the diff, and return status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--diff-output", type=Path)
    args = parser.parse_args()

    equal, diff = compare_pdfs(args.baseline, args.candidate)
    if args.diff_output:
        args.diff_output.write_text(diff, encoding="utf-8")
    if equal:
        print("Normalized PDF text is identical.")
        return 0
    print("Normalized PDF text differs; inspect the generated diff.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

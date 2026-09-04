# SpectraExpert Official IEEE Template Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not delegate work unless the user explicitly requests delegation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 IEEE 2024 Conference Template 重建可维护的 SpectraExpert LaTeX 工程，并证明生成稿与用户提供的 10 页最终 PDF 在正文、公式、数值、图表和参考文献上保持一致。

**Architecture:** IEEE 官方模板 ZIP 只提供版式和未经修改的 `IEEEtran.cls`，最终 PDF 只提供内容，旧 LaTeX 只提供可复用结构和图件。重建分为来源冻结、自动比对工具、前三章、方法、实验、参考文献与附录、最终 IEEE/PDF 验收七个任务；每个内容任务都编译并核对章节锚点，完整规范化文本差异在最终任务集中审阅。

**Tech Stack:** IEEEtran LaTeX、BibTeX、latexmk、Python 3.11 标准库、Poppler (`pdfinfo`、`pdffonts`、`pdfdetach`、`pdftotext`、`pdftoppm`)、Git。

**Spec:** `docs/superpowers/specs/2026-09-04-paper-revision-foundation-design.md`

## Global Constraints

- 内容基线固定为 `/Users/andrewluan/Downloads/DM2092_5291 (2).pdf`，SHA-256 必须为 `e7bad853c4cf11fc866680a2cc1a9e06a7c40d04dd34be9bb48984823212cf1f`。
- 格式基线固定为 `/Users/andrewluan/Downloads/conference-latex-template.zip`，SHA-256 必须为 `516252d9ac6e974af3a1d9aa72f97ecc462b9a47556225019fb0adef69bba78a`。
- 工作副本直接采用官方 ZIP 中的 `IEEEtran.cls 2015/08/26 V1.8b`，不修改第三方类文件。
- `IEEE_Conference_Template`、两个 Downloads 来源文件和服务器 `/home/user/repro` 全部保持只读。
- 标题、作者、单位、邮箱、正文句子、公式、图表数值、模型名称、实验参数、31 条参考文献和 Appendix A 不得擅自增删或改写。
- 允许的差异仅限分页、断行、浮动体位置、无边框链接、IEEE 默认 Abstract/Index Terms 样式、PDF metadata 和等义 LaTeX 转义。
- 保持 `\documentclass[conference]{IEEEtran}` 和 US Letter；不得使用 `geometry` 或手工覆盖 IEEE 页边距、栏宽、字号和行距。
- `paper/main.tex` 作为项目源代码文件，必须在 `\documentclass` 后维护中文头部注释；`IEEEtran.cls` 不加项目注释。
- 第一次创建或编辑本任务 PDF 前，必须按 PDF skill 要求执行一次且仅一次 artifact-operation marker。
- 用户已有未跟踪文件 `baseline/ja5c05010.pdf` 和 `baseline/s43588-023-00550-y.pdf` 不得暂存或提交。
- 最终稳定产物固定为 `output/pdf/SpectraExpert_camera_ready.pdf`。

---

### Task 1: 冻结来源并建立官方模板工作副本

**Files:**
- Create: `paper/IEEEtran.cls`
- Create: `paper/main.tex`
- Create: `paper/references.bib`
- Create: `paper/SOURCE_PROVENANCE.md`
- Create: `paper/CONTENT_BASELINE.txt`
- Create: `paper/.gitignore`
- Create: `paper/figures/pipeline.pdf`
- Create: `paper/figures/molecular_complexity.png`
- Create: `paper/figures/peak_intensity_distribution.png`
- Create: `paper/figures/peak_distribution.jpg`

**Interfaces:**
- Consumes: 两个已冻结来源文件和旧稿的四个图件。
- Produces: 后续任务唯一可写的 `paper/` 工程，以及机器可读的 PDF 文本基线。

- [ ] **Step 1: 复核所有来源指纹和 PDF 基本属性**

Run:

```bash
shasum -a 256 \
  '/Users/andrewluan/Downloads/DM2092_5291 (2).pdf' \
  '/Users/andrewluan/Downloads/conference-latex-template.zip' \
  '/Users/andrewluan/Desktop/Projects/IRfingerprints/IEEE_Conference_Template/main/conference_101719.tex' \
  '/Users/andrewluan/Desktop/Projects/IRfingerprints/IEEE_Conference_Template/main/conference_101719.pdf'
pdfinfo '/Users/andrewluan/Downloads/DM2092_5291 (2).pdf'
```

Expected hashes:

```text
e7bad853c4cf11fc866680a2cc1a9e06a7c40d04dd34be9bb48984823212cf1f  DM2092_5291 (2).pdf
516252d9ac6e974af3a1d9aa72f97ecc462b9a47556225019fb0adef69bba78a  conference-latex-template.zip
5e7fe0c7d664d9028132e257fe33343fc5be9beddce569605a2ddba7b916c772  conference_101719.tex
69f1b9b04f0c8902bbdc628a87ca1fd20764c7fee0ee0c6ae82dbe58c1e484b7  conference_101719.pdf
```

Expected PDF properties: `Pages: 10`, `Page size: 612 x 792 pts (letter)`, `Encrypted: no`。

- [ ] **Step 2: 启动 PDF 编辑操作记录**

调用工作区依赖发现工具定位 bundled runtime，然后从该 runtime 根目录只执行一次：

```bash
node container_tools/mark_artifact_operation_started.mjs --operation-kind edit --expected-output-count 1 --output-format pdf
```

- [ ] **Step 3: 创建工程目录并导入官方类文件**

Run:

```bash
mkdir -p paper/build paper/figures paper/tools output/pdf /private/tmp/spectrexpert-paper-rebuild/official-template
unzip -q -n '/Users/andrewluan/Downloads/conference-latex-template.zip' -d /private/tmp/spectrexpert-paper-rebuild/official-template
cp /private/tmp/spectrexpert-paper-rebuild/official-template/IEEE-conference-template-062824/IEEEtran.cls paper/IEEEtran.cls
rg -n '^\\documentclass\[conference\]\{IEEEtran\}' /private/tmp/spectrexpert-paper-rebuild/official-template/IEEE-conference-template-062824/IEEE-conference-template-062824.tex
rg -n 'ProvidesClass\{IEEEtran\}\[2015/08/26 V1.8b' paper/IEEEtran.cls
```

Expected: 两个 `rg` 命令均返回一行；`paper/IEEEtran.cls` 的 SHA-256 为 `c972aca108fda004c3514d63658e02816da2e54d9a1451e870b9bd970e003f55`。

- [ ] **Step 4: 导入旧稿可复用的书目和图件**

Run:

```bash
cp ../IEEE_Conference_Template/main/ref/ref.bib paper/references.bib
cp '../IEEE_Conference_Template/fig/pipeline (3).pdf' paper/figures/pipeline.pdf
cp '../IEEE_Conference_Template/fig/molecular complexity.png' paper/figures/molecular_complexity.png
cp '../IEEE_Conference_Template/fig/peak intensity distribution.png' paper/figures/peak_intensity_distribution.png
cp '../IEEE_Conference_Template/fig/peak distribution.jpg' paper/figures/peak_distribution.jpg
find paper/figures -maxdepth 1 -type f -size +0 -print | sort
```

Expected: 精确列出四个非空图件。

- [ ] **Step 5: 生成只读文本基线**

Run:

```bash
pdftotext -raw '/Users/andrewluan/Downloads/DM2092_5291 (2).pdf' paper/CONTENT_BASELINE.txt
wc -l -w -c paper/CONTENT_BASELINE.txt
rg -n 'SpectraExpert: Expert-Like Reasoning Agent|TABLE I:|TABLE IV:|APPENDIX A|FC1\(F\)CCCNC1' paper/CONTENT_BASELINE.txt
```

Expected: 约 7,143 个英文词，并命中标题、Table I、Table IV、Appendix A 和案例 ground truth。

- [ ] **Step 6: 用 `apply_patch` 创建最小官方模板入口**

`paper/main.tex` 初始内容必须为：

```tex
\documentclass[conference]{IEEEtran}
% 功能概述：本文档是 SpectraExpert 最终稿的唯一 LaTeX 入口，使用 IEEE Conference Template 组织正文、公式、图表和参考文献。
% 主要职责：逐项重建最终 PDF 中可见的论文内容，并仅进行链接、元数据和 IEEE 默认环境样式等格式性修复。
% 代码库关系：引用 figures/、references.bib 和未经修改的 IEEEtran.cls；由 latexmk 编译到 build/，稳定产物复制到 ../output/pdf/。
% 内容锁定：正文、公式、数值、模型名称、参考文献和附录以 DM2092_5291 (2).pdf 为准；旧 LaTeX 只提供结构参考，不得覆盖最终稿内容。
\IEEEoverridecommandlockouts

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{url}
\usepackage{hyperref}

\hypersetup{
  hidelinks,
  pdftitle={SpectraExpert: Expert-Like Reasoning Agent for Spectrum-Based Molecular Identification},
  pdfauthor={Denghong Luan and Zaifan Lu},
  pdfsubject={Raman spectrum-based molecular identification},
  pdfkeywords={Retrieval-augmented reasoning, contrastive learning, raman spectroscopy, molecular identification, large language models}
}

\begin{document}
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

- [ ] **Step 7: 用 `apply_patch` 创建来源说明和忽略规则**

`paper/SOURCE_PROVENANCE.md` 写入：

```markdown
# Paper Source Provenance

The manuscript uses two independent sources of truth.

- Content source: `/Users/andrewluan/Downloads/DM2092_5291 (2).pdf`
- Content SHA-256: `e7bad853c4cf11fc866680a2cc1a9e06a7c40d04dd34be9bb48984823212cf1f`
- Format source: `/Users/andrewluan/Downloads/conference-latex-template.zip`
- Format SHA-256: `516252d9ac6e974af3a1d9aa72f97ecc462b9a47556225019fb0adef69bba78a`
- Official class: `IEEEtran.cls 2015/08/26 V1.8b`
- Imported on: `2026-09-04`

The PDF supplies all visible manuscript content. The official template supplies layout only. `../IEEE_Conference_Template/main/conference_101719.tex` is an older structural reference and must not override the final PDF.
```

`paper/.gitignore` 写入：

```gitignore
build/
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
```

- [ ] **Step 8: 提交来源冻结**

```bash
git add paper/IEEEtran.cls paper/main.tex paper/references.bib paper/SOURCE_PROVENANCE.md paper/CONTENT_BASELINE.txt paper/.gitignore paper/figures
git commit -m "docs: establish official IEEE paper baseline"
```

### Task 2: 建立 PDF 内容比对工具

**Files:**
- Create: `paper/tools/compare_pdf_text.py`
- Create: `paper/tools/test_compare_pdf_text.py`
- Create: `paper/CONTENT_EQUIVALENCE.md`

**Interfaces:**
- Consumes: `Path` 形式的基线 PDF 和候选 PDF。
- Produces: `extract_text(pdf: Path) -> str`、`normalize_text(text: str) -> str`、`compare_pdfs(baseline: Path, candidate: Path) -> tuple[bool, str]`，以及可审计的 unified diff。

- [ ] **Step 1: 用 `apply_patch` 写失败测试**

`paper/tools/test_compare_pdf_text.py`：

```python
"""功能概述：验证论文 PDF 文本归一化与差异检测，确保排版空白可忽略而数字和化学字符串变化不会被吞掉。

主要职责：测试空白归一化、分页符处理、内容一致性和关键数值差异；不创建或修改论文 PDF。
代码库关系：直接测试 compare_pdf_text.py，供最终稿内容锁定检查使用。
"""

import unittest

from compare_pdf_text import normalize_text


class NormalizeTextTest(unittest.TestCase):
    def test_collapses_layout_whitespace_and_page_breaks(self) -> None:
        left = "SpectraExpert\n\nGED 16.93\fAppendix A"
        right = "SpectraExpert GED 16.93 Appendix A"
        self.assertEqual(normalize_text(left), normalize_text(right))

    def test_preserves_numeric_changes(self) -> None:
        self.assertNotEqual(normalize_text("GED 16.93"), normalize_text("GED 16.39"))

    def test_preserves_smiles_changes(self) -> None:
        self.assertNotEqual(
            normalize_text("FC1(F)CCCNC1"),
            normalize_text("CC1CCC2CC2(O)C1"),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认缺少实现**

Run:

```bash
cd paper/tools
python3 -m unittest test_compare_pdf_text.py -v
```

Expected: `ModuleNotFoundError: No module named 'compare_pdf_text'`。

- [ ] **Step 3: 用 `apply_patch` 写最小比对实现**

`paper/tools/compare_pdf_text.py`：

```python
"""功能概述：提取并规范化两份论文 PDF 的文本，生成可审计的一致性差异。

主要职责：调用 pdftotext、只归一化 Unicode 与空白、保留数字和化学字符串，并输出 unified diff。
代码库关系：读取外部最终稿基线和 build/main.pdf；其结果写入 CONTENT_EQUIVALENCE.md，不参与 LaTeX 编译。
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
```

- [ ] **Step 4: 运行单元测试**

Run:

```bash
cd paper/tools
python3 -m unittest test_compare_pdf_text.py -v
```

Expected: 3 tests pass。

- [ ] **Step 5: 用 `apply_patch` 创建差异审计记录**

`paper/CONTENT_EQUIVALENCE.md` 初始内容：

```markdown
# Content Equivalence Record

## Baseline

- PDF: `/Users/andrewluan/Downloads/DM2092_5291 (2).pdf`
- SHA-256: `e7bad853c4cf11fc866680a2cc1a9e06a7c40d04dd34be9bb48984823212cf1f`

## Allowed presentation-only changes

- IEEEtran default styling for Abstract and Index Terms
- Hidden hyperlink borders with links preserved
- Populated PDF metadata
- Page breaks, line wraps, and float placement produced by a clean rebuild

## Required content invariants

- Title, authors, affiliations, and email addresses
- Sections I--VI and Appendix A
- Equations (1)--(3), figures 1--3, and tables I--IV
- All experimental values, model names, thresholds, and case-study values
- References [1]--[31] in the same order

## Final verification result

The result is recorded only after normalized text diff review and page-by-page visual inspection.
```

- [ ] **Step 6: 提交比对工具**

```bash
git add paper/tools/compare_pdf_text.py paper/tools/test_compare_pdf_text.py paper/CONTENT_EQUIVALENCE.md
git commit -m "test: add paper content equivalence checks"
```

### Task 3: 重建首页、Introduction、Related Work 和 Preliminaries

**Files:**
- Modify: `paper/main.tex`
- Test: `paper/build/main.pdf`
- Test: `paper/build/frontmatter.txt`

**Interfaces:**
- Consumes: 最终稿 PDF 第 1--4 页、旧稿对应段落和 Task 2 比对工具。
- Produces: 完整首页与 Sections I--III，可独立编译，并保留最终稿中的标题、作者和正文。

- [ ] **Step 1: 用 `apply_patch` 写入标题和作者信息**

必须生成以下可见内容：

```text
SpectraExpert: Expert-Like Reasoning Agent for Spectrum-Based Molecular Identification
Denghong Luan — School of Artificial Intelligence and Data Science — University of Science and Technology of China — Hefei, China — dhluan@mail.ustc.edu.cn
Zaifan Lu — School of Physical Science — University of Science and Technology of China — Hefei, China — elenwath_lzf@mail.ustc.edu.cn
```

LaTeX 使用两个 `\IEEEauthorblockN`/`\IEEEauthorblockA` 块，不保留旧稿中的 `2\textsuperscript{nd}` 前缀。

- [ ] **Step 2: 用最终稿原文写入 Abstract 和 Index Terms**

Abstract 必须精确为：

```text
Identifying molecular structures from Raman spectra is a high-impact applied data mining problem for real-time chemical analysis, yet it remains a difficult inverse task because spectra are noisy, domain-shifted, and only weakly tied to unique structures. We present SpectraExpert, an applied retrieval-analysis-reasoning system that operationalizes the workflow of a human spectroscopist. Given a query Raman spectrum, the framework first retrieves candidate molecules via a contrastive SpectrumTransformer encoder trained on 129,817 QM9S spectra, then matches detected peaks to 181 empirical Nakamoto group-frequency rules, and finally deploys an LLM to evaluate candidates and propose structural modifications guided by spectroscopic evidence. We evaluate five frontier LLMs across four experimental settings on a reproducible zero-shot cross-domain benchmark using a QM9S training database with $\leq 9$ heavy atoms and a ChEMBL test set with up to 21 heavy atoms. Retrieval baselines achieve mean graph edit distance of 16.93 on 211 test samples. On randomly sampled LLM evaluation batches, the agentic pipeline includes individual predictions with GED as low as 8.0, while Claude Sonnet 4.6 maintains 100% valid-SMILES output in direct and agentic modes and 96% in knowledge mode. The study provides both an end-to-end application prototype and a benchmark for spectroscopic molecular identification. Our analysis shows that structured-output reliability is the dominant deployment bottleneck, and that the framework delivers interpretable, expert-aligned reasoning inaccessible to retrieval alone.
```

Index Terms 必须精确为：

```text
Retrieval-augmented reasoning, contrastive learning, raman spectroscopy, molecular identification, large language models
```

只使用 `abstract` 和 `IEEEkeywords` 环境，不对环境正文施加 `\textbf` 或 `\bfseries`。

- [ ] **Step 3: 从最终稿重建 Section I**

使用最终稿第 1--2 页，从 `Raman spectroscopy [1] is a non-destructive analytical technique` 开始，到 `All code and data are available at https://github.com/AndrewLuan/SpectraExpert.` 结束。保留四条贡献、斜体研究问题以及 `SpectraExpert` 名称；不得引入旧稿中的 `Fingerprint Reasoner`。

Run after patch:

```bash
rg -n 'Can LLMs perform expert-level|The main applied contributions|All code and data are available|Fingerprint Reasoner' paper/main.tex
```

Expected: 前三项各命中，`Fingerprint Reasoner` 不命中。

- [ ] **Step 4: 从最终稿重建 Sections II--III**

保持以下小节和锚点：

```text
II. Related Work
  A. Retrieval-Based Molecular Identification
  B. Large Language Models for Scientific Reasoning
III. Preliminaries
  A. Raman Spectroscopy and Molecular Similarity
  B. Cheminformatics Evaluation Metrics
  C. Applied Data-Mining Setting
```

Section II 必须包含 RamanNet、ChEMBL extension、RamanBench、MatExpert、MRKL、Toolformer、ReAct、ChemLLM、ChemCrow 和 autonomous chemistry agents。Section III 必须定义 GED、Tanimoto、exact match、SMILES validity 和 QM9S-to-ChEMBL domain shift。

- [ ] **Step 5: 编译并检查首页内容**

Run:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
pdftotext -layout build/main.pdf build/frontmatter.txt
rg -n 'Expert-Like Reasoning Agent|Denghong Luan|Zaifan Lu|Retrieval-augmented reasoning|III. PRELIMINARIES' build/frontmatter.txt
rg -n 'This document is a model|component, formatting|IEEE conference templates contain guidance' build/frontmatter.txt
```

Expected: 五个内容锚点存在；三个模板短语不存在。

- [ ] **Step 6: 检查编译诊断并提交**

```bash
rg -n 'Undefined control sequence|LaTeX Error' paper/build/main.log
git add paper/main.tex
git commit -m "docs: rebuild paper front matter and background"
```

Expected: 诊断命令无输出。MRKL、Toolformer、ReAct 和 RDKit 的 citation warning 在 Task 6 添加对应 BibTeX 后消除，不在此处伪造临时书目。

### Task 4: 重建 Method 与全部公式参数

**Files:**
- Modify: `paper/main.tex`
- Test: `paper/build/main.pdf`
- Test: `paper/build/method.txt`

**Interfaces:**
- Consumes: 最终稿 PDF 第 4--6 页、旧稿公式结构和四个图件。
- Produces: Section IV、Equations (1)--(3)、Fig. 1 和完整三阶段方法。

- [ ] **Step 1: 重建 Problem Formulation 与 Overview**

写入 Section IV 的 A/B 小节，保留以下数学定义：

```tex
S = \{(w_i, I_i)\}_{i=1}^{N_s}
```

```tex
f : \mathcal{S} \rightarrow \mathcal{M}, \qquad \hat{M} = f(S)
```

Overview 必须列出 `(1) retrieval, (2) Nakamoto-rule spectral analysis, (3) reasoning`。Fig. 1 使用 `figures/pipeline.pdf`，caption 以 `Illustration of SpectraExpert for identifying unknown chemicals` 开始。

- [ ] **Step 2: 重建 Spectral Representation**

从最终稿恢复以下不可变参数：

```text
input length 3501; range 500--4000 cm^-1; step 1 cm^-1
patch size 16; stride 8; 436 patches
embedding dimension 256; 6 Transformer layers; 8 heads; head dimension 32
projection head 256 -> 256 -> 256; trainable parameters 4.99M
NT-Xent temperature 0.1
AdamW learning rate 1e-4; weight decay 0.05; 5 warmup epochs
micro-batch 128 per GPU; per-step global batch 1024 across 8 GPUs
effective batch 4096; 4 gradient-accumulation steps
80 epochs; early-stopping patience 10
database size 129,817; Top-K retrieval
```

Equation (1) 必须是最终稿中的 NT-Xent loss，并保留 `2B` 分母、正样本 `z_i^+` 和 `j \ne i`。

- [ ] **Step 3: 重建峰检测、规则匹配与 LLM 决策**

保留以下精确设置：

```text
find_peaks: minimum height 5%, prominence 2%, distance 5 pixels
181 Nakamoto-derived group-frequency rules
matching tolerance +/-20 cm^-1
coverage = |explained peaks| / |total peaks|
decisions: SELECT, MODIFY, UNCERTAIN
fallback confidence threshold: 0.6
```

Equation (2) 为 `l - 20 \le p \le h + 20`；Equation (3) 为 coverage fraction。Stage 3 必须保持最终稿的 `Modification Fallback` 描述，不将其改写成已验证改进。

- [ ] **Step 4: 编译并核对方法锚点**

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
pdftotext -layout build/main.pdf build/method.txt
rg -n '4.99M|global per-step batch size of 1024|effective batch size of 4096|181 empirical|20 cm|SELECT, MODIFY, and UNCERTAIN|confidence falls below 0.6' build/method.txt
```

Expected: 七项参数全部命中。

- [ ] **Step 5: 提交方法重建**

```bash
git add paper/main.tex paper/figures
git commit -m "docs: reconstruct SpectraExpert method section"
```

### Task 5: 重建实验、图件和 Tables I--IV

**Files:**
- Modify: `paper/main.tex`
- Test: `paper/build/main.pdf`
- Test: `paper/build/experiments.txt`

**Interfaces:**
- Consumes: 最终稿 PDF 第 6--8 页和工作副本图件。
- Produces: Section V、Figs. 2--3、四张结果表和全部结果解释。

- [ ] **Step 1: 重建数据集与 Experimental Setup**

保留 QM9S/ChEMBL 分工以及下列协议：

```text
Retrieval: all 211 ChEMBL queries, top-10 candidates
LLM Direct: randomly sampled 50-query batch
LLM Knowledge: randomly sampled 50-query batch
Agentic: randomly sampled 20-query batch
N is attempted queries, not valid outputs
Retrieval GED is best@10; LLM/Agentic GED uses one predicted SMILES
```

Fig. 2 caption 以 `Statistical characteristics of the QM9S training dataset` 开始；Fig. 3 caption 以 `Two-dimensional density distribution of Raman spectra in the QM9S training set` 开始。图像组合必须与基线 PDF 第 6 页视觉一致。

- [ ] **Step 2: 重建 Table I 的全部行**

Table I 必须使用以下数据，列顺序为 `Method, Model, GED_mean, GED_median, GED_std, N, Parsed`：

```text
Retrieval|Encoder|16.93|17.0|4.63|211|1.00
Retrieval|KNN|17.04|17.0|3.91|211|1.00
LLM-Direct|GPT-5|22.79|22.0|4.22|50|0.38
LLM-Direct|o3|23.53|24.0|4.38|50|0.94
LLM-Direct|Claude-Sonnet-4-6|23.82|24.0|4.57|50|1.00
LLM-Direct|Gemini-2.5-Flash|23.90|24.5|5.49|50|0.20
LLM-Direct|DeepSeek-v4-Flash|24.68|26.0|5.27|50|0.44
LLM-Know.|o3|13.52|14.0|5.95|50|0.84
LLM-Know.|GPT-5|14.86|16.0|5.60|50|0.42
LLM-Know.|DeepSeek-v4-Flash|15.24|15.0|6.28|50|0.68
LLM-Know.|Claude-Sonnet-4-6|16.90|16.5|13.92|50|0.96
LLM-Know.|Gemini-2.5-Flash|17.70|16.5|7.00|50|0.20
Agentic|Gemini-2.5-Flash|15.67|15.0|5.03|20|1.00
Agentic|DeepSeek-v4-Flash|15.73|17.0|3.86|20|0.85
Agentic|Claude-Sonnet-4-6|16.16|15.0|4.91|20|1.00
Agentic|o3|18.07|18.0|5.15|20|0.95
Agentic|GPT-5|21.00|21.0|0.00|20|0.05
```

保留 `dagger`（parse rate below 0.50）和 `double dagger`（near-zero parse rate）脚注含义。

- [ ] **Step 3: 重建 Tables II--IV**

Table II：

```text
LLM-Direct GPT-5|22.79|0.125
LLM-Direct o3|23.53|0.114
LLM-Direct Claude-S-4.6|23.82|0.117
Encoder top-1|24.48|0.058
KNN top-1|25.92|0.061
Encoder best@10|21.98|omitted
KNN best@10|22.00|omitted
```

Table III 的 `Direct|Knowledge|Agentic`：

```text
Claude-Sonnet-4-6|1.00|0.96|1.00
o3|0.94|0.84|0.95
DeepSeek-v4-Flash|0.44|0.68|0.85
GPT-5|0.38|0.42|0.05
Gemini-2.5-Flash|0.20|0.20|1.00
```

Table IV 的 `SELECT(n)|GED|MODIFY(n)|GED`：

```text
Claude-Sonnet-4-6|6|12.8|14|17.7
DeepSeek-v4-Flash|7|14.1|10|17.1
o3|6|17.3|13|18.6
Gemini-2.5-Flash|0|N/A|20|15.7
GPT-5|0|N/A|1|N/A
```

- [ ] **Step 4: 重建 Results A--H 与 Conclusion**

保持最终稿以下小节：

```text
C. Main Results
D. Retrieval Baselines: Encoder vs. KNN
E. LLM Direct and LLM Knowledge
F. SMILES Output Validity
G. Agentic Pipeline
H. Summary and Practical Recommendations
VI. Conclusion
```

结果解释必须保留：zero exact matches、Table II 的 batch-controlled comparison、o3 Knowledge GED 13.52、Gemini validity `0.20 -> 1.00`、GPT-5 validity `0.38 -> 0.05`、MODIFY 相对 SELECT 的 `+4.86/+2.98/+1.23`。Conclusion 以 `We presented SpectraExpert, an applied data mining system` 开始，以 `chemistry-aware constrained generation` 结束。

- [ ] **Step 5: 编译并核对实验数值**

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
pdftotext -layout build/main.pdf build/experiments.txt
rg -n '16.93|22.79|13.52|15.67|21.00|0.20.*1.00|0.38.*0.05|4.86|2.98|1.23|chemistry-aware constrained generation' build/experiments.txt
```

Expected: 全部数值和 Conclusion 结尾命中。

- [ ] **Step 6: 提交实验重建**

```bash
git add paper/main.tex paper/figures
git commit -m "docs: reconstruct paper experiments and results"
```

### Task 6: 重建 References 与 Appendix A

**Files:**
- Modify: `paper/main.tex`
- Modify: `paper/references.bib`
- Test: `paper/build/main.bbl`
- Test: `paper/build/appendix.txt`

**Interfaces:**
- Consumes: 最终稿 PDF 第 8--10 页和旧书目。
- Produces: 与基线顺序一致的 References [1]--[31] 和完整案例 trace。

- [ ] **Step 1: 补齐工具编排与 RDKit 引用**

向 `paper/references.bib` 添加 `karpas2022mrkl`、`schick2023toolformer`、`yao2023react` 和 `landrum2013rdkit`；正文引用顺序必须使前三项分别成为 [17]、[18]、[19]，RDKit 成为 [23]。

精确条目：

```bibtex
@article{karpas2022mrkl,
  title={MRKL Systems: A Modular, Neuro-Symbolic Architecture That Combines Large Language Models, External Knowledge Sources and Discrete Reasoning},
  author={Karpas, Ehud and Abend, Omri and Belinkov, Yonatan and Lenz, Barak and Lieber, Opher and Ratner, Nir and Shoham, Yoav and Bata, Hofit and Levine, Yoav and Leyton-Brown, Kevin and Muhlgay, Dor and Rozen, Noam and Schwartz, Erez and Shachaf, Gal and Shalev-Shwartz, Shai and Shashua, Amnon and Tenenholtz, Moshe},
  journal={arXiv preprint arXiv:2205.00445},
  year={2022}
}

@inproceedings{schick2023toolformer,
  title={Toolformer: Language Models Can Teach Themselves to Use Tools},
  author={Schick, Timo and Dwivedi-Yu, Jane and Dess{\`i}, Roberto and Raileanu, Roberta and Lomeli, Maria and Hambro, Eric and Zettlemoyer, Luke and Cancedda, Nicola and Scialom, Thomas},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  pages={68539--68551},
  year={2023}
}

@inproceedings{yao2023react,
  title={ReAct: Synergizing Reasoning and Acting in Language Models},
  author={Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@misc{landrum2013rdkit,
  title={RDKit: Open-Source Cheminformatics},
  author={Landrum, Greg},
  howpublished={\url{https://www.rdkit.org}},
  year={2013}
}
```

- [ ] **Step 2: 锁定 References [1]--[31] 顺序**

正文首次引用顺序必须产生以下编号：

```text
[1] Raman 1928
[2] Handzo et al. 2022
[3] Kudelski 2008
[4] Rostron et al. 2016
[5] Zhang et al. 2005
[6] Shipp et al. 2017
[7] Bunaciu et al. 2015
[8] Liang et al. 2021
[9] Zou et al. 2023
[10] Hu et al. 2023
[11] Liang et al. 2025
[12] Ibtehaz et al. 2023
[13] Koddenbrock et al. 2026
[14] Ding et al. 2025
[15] Wei et al. 2022
[16] Pang et al. 2024
[17] Karpas et al. 2022
[18] Schick et al. 2023
[19] Yao et al. 2023
[20] Zhang et al. 2024
[21] Bran et al. 2024
[22] Boiko et al. 2023
[23] Landrum 2013
[24] Chen et al. 2020
[25] Nakamoto 2008
[26] Socrates 2004
[27] OpenAI GPT-5 system card
[28] OpenAI o3 and o4-mini System Card
[29] Anthropic Claude Sonnet 4.6 system card
[30] Google DeepMind Gemini 2.5 technical report
[31] DeepSeek AI DeepSeek-v4 technical report
```

为 [23] 添加 RDKit 条目，显示内容保持 `G. Landrum, "RDKit: Open-source cheminformatics," https://www.rdkit.org, 2013.`。

- [ ] **Step 3: 重建 Appendix A 的输入与候选表**

附录标题为 `SPECTRAEXPERT---FULL PIPELINE TRACE`。保留：

```text
case: 4,4-difluoropiperidine; test index #11
ground truth: FC1(F)CCCNC1; length 12
detected peaks: 2996/0.1351/8.23; 3044/0.1094/4.97; 3435/0.0443/2.70; 1468/0.0229/1.40
candidates:
CC1CCC2CC2(O)C1|0.8521|8.0|0.059
Cc1nn(C)c(=O)o1|0.8248|15.0|0.000
OC1CN2CC2C12CC2|0.7538|14.0|0.091
C1=CC2OCCC2OC1|0.7528|15.0|0.028
O=C1C2CC13NCC3N2|0.7459|15.0|0.081
```

- [ ] **Step 4: 重建规则、决策和最终评估**

保留：

```text
coverage 1.0
rules GFQ-003, GFQ-050, GFQ-015, GFQ-010, GFQ-064
DECISION SELECT
SELECTED_INDEX 1
SELECTED_SMILES CC1CCC2CC2(O)C1
CONFIDENCE 0.72
fallback used No
GED / Tanimoto / exact match 8 / 0.059 / No
processing time approximately 65 s for the Stage 3 LLM call
```

- [ ] **Step 5: 编译并核对引用与附录**

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
pdftotext -layout build/main.pdf build/appendix.txt
rg -n 'Warning--I didn.t find a database entry|Citation .* undefined' build/main.blg build/main.log
rg -n '\[31\]|SPECTRAEXPERT.*FULL PIPELINE TRACE|FC1\(F\)CCCNC1|0.8521|CONFIDENCE.*0.72|65 s' build/appendix.txt
```

Expected: 无缺失 BibTeX/citation 警告；六个输出锚点均存在。

- [ ] **Step 6: 提交 References 和 Appendix**

```bash
git add paper/main.tex paper/references.bib
git commit -m "docs: reconstruct references and pipeline appendix"
```

### Task 7: 完成内容等价、IEEE 格式和逐页视觉验收

**Files:**
- Modify: `paper/main.tex`
- Modify: `paper/CONTENT_EQUIVALENCE.md`
- Create: `output/pdf/SpectraExpert_camera_ready.pdf`
- Test: `paper/build/main.log`
- Test: `paper/build/content.diff`
- Test: `/private/tmp/spectrexpert-paper-rebuild/rendered/*.png`

**Interfaces:**
- Consumes: Tasks 1--6 的完整工程和两个冻结来源。
- Produces: 可交付的 camera-ready PDF 和完整内容一致性记录。

- [ ] **Step 1: 清理 LaTeX 版式违规项**

Run:

```bash
rg -n 'geometry|setlength\{\\textwidth|setlength\{\\columnsep|fontsize|linespread|This document is a model|component, formatting|IEEE conference templates contain guidance|Fingerprint Reasoner' paper/main.tex
rg -n 'hypersetup|hidelinks|documentclass\[conference\]' paper/main.tex
```

Expected: 第一条命令无输出；第二条命中 `conference`、`hypersetup` 和 `hidelinks`。

- [ ] **Step 2: 执行干净构建并复制稳定产物**

```bash
cd paper
latexmk -C -outdir=build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf ../output/pdf/SpectraExpert_camera_ready.pdf
cd ..
```

- [ ] **Step 3: 检查 PDF 结构和编译日志**

```bash
pdfinfo output/pdf/SpectraExpert_camera_ready.pdf
pdffonts output/pdf/SpectraExpert_camera_ready.pdf
pdfdetach -list output/pdf/SpectraExpert_camera_ready.pdf
rg -n 'Undefined control sequence|Citation .* undefined|Reference .* undefined|There were undefined references|LaTeX Error|Overfull \\hbox' paper/build/main.log
```

Expected:

```text
Page size: 612 x 792 pts (letter)
Encrypted: no
Title: SpectraExpert: Expert-Like Reasoning Agent for Spectrum-Based Molecular Identification
Author: Denghong Luan and Zaifan Lu
0 embedded files
```

`pdffonts` 每一行的 `emb` 列均为 `yes`；日志检查无输出。页数可以因浮动体重排变化，但不得通过删减内容满足页数。

- [ ] **Step 4: 生成并审阅规范化文本差异**

```bash
python3 paper/tools/compare_pdf_text.py \
  '/Users/andrewluan/Downloads/DM2092_5291 (2).pdf' \
  output/pdf/SpectraExpert_camera_ready.pdf \
  --diff-output paper/build/content.diff
wc -l paper/build/content.diff
sed -n '1,260p' paper/build/content.diff
```

若命令返回 0，记录 normalized text identical。若返回 1，逐条检查差异；只接受页内断词、字体字符映射、Abstract/Index Terms 样式和浮动体重排造成的差异。数字、SMILES、引用编号或正文 token 差异必须回到 `main.tex` 修复，然后重新构建。

- [ ] **Step 5: 更新内容等价记录**

在 `paper/CONTENT_EQUIVALENCE.md` 的 Final verification result 中写入实际结论，必须包含：

```markdown
- Baseline and candidate were compared with `pdftotext -raw` after Unicode and whitespace normalization.
- Every remaining diff was reviewed against the rendered baseline pages.
- No title, author, sentence, equation, numeric value, model name, table row, reference, or Appendix A case value changed.
- Presentation-only changes: hidden hyperlink borders, IEEEtran default Abstract/Index Terms styling, populated metadata, and natural reflow.
```

- [ ] **Step 6: 渲染并逐页检查全部页面**

```bash
mkdir -p /private/tmp/spectrexpert-paper-rebuild/rendered
pdftoppm -png -r 150 output/pdf/SpectraExpert_camera_ready.pdf /private/tmp/spectrexpert-paper-rebuild/rendered/page
find /private/tmp/spectrexpert-paper-rebuild/rendered -maxdepth 1 -name 'page-*.png' -print | sort
```

使用 `view_image` 逐页查看全部 PNG。检查首页、双栏边界、Fig. 1--3、Equations (1)--(3)、Tables I--IV、References [1]--[31] 和 Appendix A。必须确认无彩色链接框、文字裁切、表格越界、图像模糊、黑块、重叠或异常空白；发现问题后修改、重建并重新渲染受影响页面。

- [ ] **Step 7: 复核来源未改变**

重复 Task 1 Step 1 的四个 `shasum`。Expected: 四个哈希与 Task 1 完全一致。

- [ ] **Step 8: 检查最终差异和仓库状态**

```bash
git diff --check
git status --short
git diff --stat
git diff -- paper/main.tex paper/references.bib paper/CONTENT_EQUIVALENCE.md
```

Expected: 只出现本任务的 `paper/` 和 `output/pdf/SpectraExpert_camera_ready.pdf`；两个用户 baseline PDF 仍为未跟踪且未暂存状态；`tmp/` 不进入提交。

- [ ] **Step 9: 提交最终 camera-ready 重建**

```bash
git add paper/main.tex paper/references.bib paper/CONTENT_EQUIVALENCE.md output/pdf/SpectraExpert_camera_ready.pdf
git commit -m "docs: finalize IEEE camera-ready paper rebuild"
```

## Plan Self-Review

- Spec coverage: 双事实来源、只读原件、官方类文件、内容锁定、允许差异、US Letter、无链接边框、Abstract/Index Terms 样式、metadata、完整章节、三条公式、三幅图、四张表、31 条参考文献、Appendix A、编译检查、文本比对和逐页视觉检查均有对应任务。
- Scope: 本计划不运行服务器实验，不引入新基线结果，不吸收会改变最终稿正文的评审建议，也不猜测具体会议页数和 copyright notice。
- Interface consistency: 所有任务统一使用 `paper/main.tex`、`paper/references.bib`、`paper/build/main.pdf`、`paper/tools/compare_pdf_text.py` 和 `output/pdf/SpectraExpert_camera_ready.pdf`。
- Evidence boundary: 所有具体论文数值来自最终 PDF；旧 LaTeX 只用于恢复 LaTeX 结构；外部模板只决定格式。

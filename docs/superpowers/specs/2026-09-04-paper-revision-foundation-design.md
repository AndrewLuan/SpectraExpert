# SpectraExpert 官方模板重建设计

## 目标与边界

本阶段将 SpectraExpert 最终稿重建为可维护、可重复编译的 IEEE Conference LaTeX 工程。重建遵循两个彼此独立的事实来源：IEEE 官方 Conference Template 决定版式，用户提供的 10 页 PDF 决定论文内容。旧 LaTeX 只用于复用图件、公式和表格结构，不再决定标题、正文、数值或参考文献。

本阶段只允许格式性修复，包括去除链接边框、恢复 IEEE 默认摘要样式、填写 PDF 元数据、消除模板残留和编译警告。未经用户再次明确授权，不改写最终稿的论点、实验结果、章节内容、公式、表格数值和参考文献。原评审驱动的进一步内容修改暂停，避免与“内容和最终稿 PDF 一致”的最新要求冲突。

## 事实来源与优先级

### 内容基线

内容事实来源为：

```text
/Users/andrewluan/Downloads/DM2092_5291 (2).pdf
SHA-256: e7bad853c4cf11fc866680a2cc1a9e06a7c40d04dd34be9bb48984823212cf1f
```

该 PDF 共 10 页，US Letter 页面大小为 612 × 792 pt。它包含标题、两位作者、Abstract、Index Terms、六个主章节、四张结果表、31 条参考文献和 Appendix A。正文约 7,143 个英文词。最终重建稿必须保留其中所有可见语义内容和数值。

### 格式基线

格式事实来源为 IEEE 官方 Conference Template 页面：

- [IEEE Conference Templates](https://www.ieee.org/conferences/publishing/templates)
- [IEEE Authoring Tools and Templates](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/)
- [IEEE Conference Publishing Information](https://events.ieee.org/planning-basics/ieee-conference-publications/publishing-information-for-ieee-conference-authors/)

本机已有从该页面下载的 2024 Conference Template：

```text
/Users/andrewluan/Downloads/conference-latex-template.zip
SHA-256: 516252d9ac6e974af3a1d9aa72f97ecc462b9a47556225019fb0adef69bba78a
Template directory: IEEE-conference-template-062824/
Class: IEEEtran.cls 2015/08/26 V1.8b
```

模板入口使用 `\documentclass[conference]{IEEEtran}` 和 US Letter 页面。官方模板要求提交前删除全部 guidance text。具体会议的页数限制、copyright notice 和 PDF eXpress conference ID 不属于通用模板规则；在用户提供会议名称或 author instructions 前，不对这些项目作无依据判断。

### 旧材料的用途

以下材料只作为辅助资源：

- `IEEE_Conference_Template/main/conference_101719.tex`：旧版文本骨架，共 786 行，不能覆盖最终稿 PDF 中更新后的内容。
- `IEEE_Conference_Template/main/conference_101719.pdf`：只有 3 页，不是完整稿件。
- `IEEE_Conference_Template/main/IEEEtran.cls`：去除 CRLF 后与 2024 官方模板类文件字节一致，可以证明旧工程使用了正确的 IEEEtran 版本，但工作副本仍直接采用官方 ZIP 中的类文件。
- `SpectrExpert/Fingerprint_Reasoner.pdf`：另一版本论文，不作为文本来源。
- `/home/user/repro`：已检查至五层目录，没有发现与最终稿对应的 `.tex`、`.bib` 或模板压缩包。

未找到生成 10 页最终 PDF 的对应 LaTeX 源文件，也未发现 PDF 内嵌附件。因此，工作副本需要从 PDF 内容和旧稿可复用结构进行重建。

## 工作副本架构

原始 PDF、官方模板 ZIP 和旧 LaTeX 目录全部保持只读。唯一修改目标位于 Git 仓库：

```text
SpectrExpert/paper/
  main.tex                 # 唯一论文入口
  references.bib           # 与最终稿 31 条引用一致的书目
  IEEEtran.cls             # 直接来自官方 2024 Conference Template
  SOURCE_PROVENANCE.md     # 记录所有来源、哈希和优先级
  CONTENT_BASELINE.txt     # 最终稿 PDF 的只读文本基线
  figures/                 # 最终稿使用的图件
  build/                   # 未跟踪的编译中间物

SpectrExpert/output/pdf/
  SpectraExpert_camera_ready.pdf
```

`paper/main.tex` 必须在 `\documentclass` 后保留中文头部注释，说明文件职责、输入输出和内容锁定规则。第三方 `IEEEtran.cls` 不添加项目注释，也不作任何修改。

## 内容重建策略

### 章节与元素清单

重建顺序以最终 PDF 为准：

1. 标题、作者、单位和邮箱；
2. Abstract 和 Index Terms；
3. Introduction；
4. Related Work；
5. Preliminaries；
6. Method；
7. Experiments；
8. Conclusion；
9. References；
10. Appendix A。

Fig. 1、Fig. 2、Fig. 3 以及 Table I--IV 的标题、正文引用和数值必须与最终 PDF 一致。公式、单位、上下标、希腊字母和 SMILES 不能只依赖 `pdftotext` 的字符结果，必须同时参考逐页渲染图和旧 LaTeX 的相应结构。

### 内容锁定规则

允许的差异：

- 因重新编译产生的分页、断行和浮动体位置变化；
- 链接从彩色边框改为无边框；
- Abstract 和 Index Terms 从整段粗体恢复为 IEEEtran 默认样式；
- PDF metadata、书签和文档属性的补全；
- 不改变含义的 LaTeX 转义、连字符和排版命令修复。

不允许的差异：

- 改变标题、作者、邮箱或单位；
- 增删正文句子、章节、表格行或参考文献；
- 改变任何实验样本数、指标、模型名称、阈值、训练参数、公式或案例数据；
- 用旧稿数值覆盖最终稿数值；
- 为解决页数问题删减内容。

任何无法从 PDF 清楚恢复的字符或公式都必须回到逐页图像和旧源码交叉核对；仍无法确定时暂停该处，不作猜测。

## IEEE 模板符合性处理

工作副本从官方 ZIP 的示例入口和 `IEEEtran.cls` 建立，保持 `conference` 模式，不引入 `geometry` 或手工修改页边距、栏宽、字号和行距。图表允许使用论文已有的 `booktabs`、`graphicx` 等功能包，但不得覆盖 IEEEtran 核心版式。

已在最终 PDF 中发现并需要修复的格式问题：

- `hyperref` 生成绿色、红色和青色链接边框；改为 `\hypersetup{hidelinks}`，保留链接功能但不显示边框。
- Abstract 和 Index Terms 的正文整段粗体；恢复 IEEEtran 默认环境样式，不改变文字。
- PDF 的 Title、Author、Subject 和 Keywords metadata 为空；由 `hyperref` 写入与首页一致的元数据。

最终工程不得包含官方模板的示例标题、示例作者、指导段落或红色警告。全部字体必须嵌入，PDF 不得加密，不得包含 JavaScript 或附件。

## 一致性验证

内容一致性采用三层验证，而不是只看页数或文件哈希。

### 结构清单

检查最终 PDF 是否包含全部章节、Fig. 1--3、Table I--IV、31 条参考文献和 Appendix A。检查标题、作者、单位和邮箱逐字一致。

### 规范化文本比对

分别从基线 PDF 和重建 PDF 提取 `-layout` 与 `-raw` 文本。比较前只允许规范化空白、分页符和行末断词；不得删除数字、标点、引用编号或化学字符串。差异必须逐条分类为允许的排版差异或内容错误。

### 逐页视觉检查

将两个 PDF 的全部页面渲染为 PNG，检查：

- 双栏边界、页边距、标题区和作者区；
- 图表清晰度、caption、颜色与宽度；
- 公式、单位、上下标和 SMILES；
- Table I--IV 的列对齐与数值；
- 参考文献换栏与 Appendix A；
- 是否存在链接边框、裁切、重叠、黑块、孤行或大面积异常空白。

允许因浮动体重排导致页数变化；如果页数不是 10 页，必须证明规范化内容仍完全一致。若具体会议限制少于生成页数，只报告冲突，不自动删减。

## 构建与提交产物

使用 `latexmk -pdf` 完成 LaTeX、BibTeX 和交叉引用构建。编译日志不得出现 undefined citation、undefined reference、LaTeX error 或未经检查的 overfull box。稳定产物为：

```text
SpectrExpert/output/pdf/SpectraExpert_camera_ready.pdf
```

交付前使用 `pdfinfo`、`pdffonts`、`pdfdetach`、`pdftotext` 和 Poppler 页面渲染进行检查。原始 PDF、官方模板 ZIP 和旧上传目录的 SHA-256 在整个过程中保持不变。

## 暂停的工作

此前设计中的摘要改写、贡献重新定位、Vib2Mol/VibraCLIP/MolSpectLLM 补充、Limitations 新增以及其他评审驱动文字调整，会改变最终稿内容，因此不在本次重建中执行。若用户之后要求继续吸收评审意见，应以完成重建后的 `paper/main.tex` 为新基线，另建一份带内容差异清单的修订计划。

服务器模型运行、统一配对评测和外部基线实验同样不属于本阶段。

## 完成判据

- `SpectrExpert/paper/` 中存在基于 IEEE 官方 Conference Template 的独立可编译工程。
- 标题、作者、正文、公式、表格、参考文献和附录与 10 页最终稿 PDF 语义和数值一致。
- 仅存在已列明的格式性差异，且差异清单可审计。
- 最终 PDF 为 US Letter、IEEE 双栏、字体全部嵌入、无加密、无附件、无模板示例文字和可见链接边框。
- 编译日志没有错误或未解析引用，全部页面通过视觉检查。
- 原始 PDF、官方模板 ZIP 和旧上传目录未被修改。

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

重建稿已于 2026-09-04 完成内容与版式验收。基线 PDF 和重建 PDF 都是 10 页、US Letter（612 × 792 pt）、未加密文档；重建稿使用未经修改的 IEEEtran V1.8b，并写入了标题、作者、主题和关键词元数据。

顺序敏感的规范化文本比较仍会报告差异，原因是 IEEEtran 重新安排了跨栏表格、图件、参考文献和 Appendix A 的分页位置，并将 table caption 按官方样式转换为大写。进一步对两份 PDF 做 Unicode 归一化、行末断词合并、大小写折叠和标点无关的 token 清点后，剩余差异只涉及以下排版表示：

- 数学公式被 Poppler 分割为不同 token，例如 `W_e p_i`、`N_p^{-1}` 和上下标；
- `500 to 4000 cm^-1` 与 `500--4000 cm^-1` 等等义范围写法；
- `GFQ-042`、`test-set`、`neuro-symbolic` 和 URL 在换行处的连字符切分；
- Figure/Table caption 的冒号、句点、大小写和浮动顺序；
- IEEE 默认 Abstract、Index Terms、表格和参考文献样式，以及隐藏的链接边框。

已逐项人工核对标题、作者、邮箱、Sections I--VI、Equations (1)--(3)、Figures 1--3、Tables I--IV、References [1]--[31] 和 Appendix A。实验样本数、GED/Tanimoto/Parsed 数值、训练参数、阈值、模型名称、案例候选、SMILES 和处理时间均与基线一致。全部 10 页渲染图已检查，没有发现裁切、重叠、黑块、缺图或可见链接边框；Appendix A 最后一页保留的空白是双栏内容自然结束，不是丢失内容。

结论：重建稿与用户最终稿保持语义和数值一致；可见差异均属于本文件列明的 IEEE 模板与重新排版差异。

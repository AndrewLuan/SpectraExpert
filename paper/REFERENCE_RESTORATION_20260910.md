# 引用恢复记录

本轮恢复 14 条引用，正文引用总数从 15 增至 29。全部沿用旧文献键，逐项找到出版方、作者预印本或官方模型／软件来源后才恢复。没有使用 `nocite` 凑数，没有改变实验结果。

## 恢复位置与核验来源

| 文献键 | 恢复位置／用途 | 核验来源 |
|---|---|---|
| shipp2017raman | Introduction：Raman 化学与生物应用 | [Optica 出版页](https://opg.optica.org/aop/abstract.cfm?uri=aop-9-2-315) |
| zou2023deep | Related Work：学习式谱预测 | [Nature Computational Science](https://www.nature.com/articles/s43588-023-00550-y) |
| liang2025dataset | Related Work：计算 Raman/IR 数据资源 | [Scientific Data](https://www.nature.com/articles/s41597-025-05289-x)、Crossref DOI 记录 |
| koddenbrock2026ramanbench | Related Work：跨数据集泛化与评测背景 | [作者预印本](https://arxiv.org/abs/2605.02003) |
| wei2022chain | Related Work：中间推理步骤 | [NeurIPS 2022](https://proceedings.neurips.cc/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)、[arXiv](https://arxiv.org/abs/2201.11903) |
| zhang2024chemllm | Related Work：化学知识与对话 | [arXiv v2](https://arxiv.org/abs/2402.06852v2)、[作者仓库](https://github.com/keyhsw/ChemLLM/blob/main/README.md) |
| m2024augmenting | Related Work：ChemCrow 化学工具 | [Nature Machine Intelligence](https://www.nature.com/articles/s42256-024-00832-8)、Crossref DOI 记录 |
| boiko2023autonomous | Related Work：Coscientist 自主化学实验 | [Nature](https://www.nature.com/articles/s41586-023-06792-0)、Crossref DOI 记录 |
| landrum2013rdkit | Metrics：分子解析及指纹计算 | [RDKit 官网](https://www.rdkit.org/)、[官方软件说明](https://github.com/rdkit/rdkit/wiki/FrequentlyAskedQuestions)；本地 `spectraexpert/eval/metrics.py` 和 `scripts/summarize_agent_multimodel.py` 确认使用 RDKit |
| gpt-5 | 主结果表：模型来源 | [OpenAI 系统卡发布页](https://openai.com/index/gpt-5-system-card/) |
| o3_and_o4-mini | 主结果表：模型来源 | [OpenAI 系统卡发布页](https://openai.com/index/o3-o4-mini-system-card/) |
| claude-sonnet-4-6 | 主结果表：模型来源 | [Anthropic 官方发布页](https://www.anthropic.com/news/claude-sonnet-4-6) |
| gemini2-5 | 主结果表：模型系列报告 | [Gemini 2.5 技术报告](https://arxiv.org/abs/2507.06261) |
| deepseek2026deepseek | 主结果表：模型系列来源 | [DeepSeek 官方 V4 发布页](https://deepseek.com/en/news/v4-preview/)；该页明确列出 V4-Flash |

模型来源引用说明所用模型系列的出处，不证明项目服务路由的等价性。正文关于 Flash 与 Flash-Ascend 未独立确认等价的限制保留。背景中的 ChEMBL 谱库明确写为计算资源，没有恢复旧稿将其视为当前实验测试来源的说法。

## 元数据修正与核验边界

CoT 补入遗漏的 Brian Ichter，按 NeurIPS 页面恢复完整作者顺序。ChemLLM 补齐作者，并采用 arXiv v2 作者字段顺序；作者仓库与 arXiv 对 Wanli Ouyang 的排序不同，不能把两份列表拼接使用。

GPT-5 的 arXiv:2601.03267 确实对应原系统卡，不能称为捏造编号。旧条目将 arXiv 版本与 2025 年发布年份混用，本轮改引机构署名、2025 年 8 月 7 日的官方发布页。o3 移除重复 note 字段，补上官方发布年份和链接。Claude 使用可访问的官方发布页，标题相应改为 Introducing Claude Sonnet 4.6。DeepSeek 使用官方发布说明，标题也相应更新，不再保留没有链接的孤立技术报告条目。

RDKit 按软件网站引用，移除未对应具体软件版本的 2013 年份，仅记录实际访问日期；保留旧键是为了兼容历史稿件，不代表引用 2013 版软件。Gemini 报告的已有前十作者及 et al. 与 arXiv 列表核对，增加稳定入口。其余恢复的期刊论文补充 DOI，原有卷页与出版记录一致。

Crossref 成功核对 ChemCrow、Coscientist、ChEMBL 数据论文；Zou 和 Shipp 的 Crossref 请求遇到 429，使用出版方页面核对，未宣称其已经双源验证。未恢复的旧文献没有被删除，也不被判定为虚假；本轮只恢复与现稿直接相关并已完成来源核验的条目。其余原先已引用的 15 条不是本轮全量复核范围。

## 编译与恢复

原稿、原文献库和原 PDF 备份于 `../tmp/pdfs/refs-20260910/before/`。独立构建目录为 `../tmp/pdfs/refs-20260910/build/`。BibTeX 必须读取该目录的 main.aux；从该目录运行 pdfLaTeX 时将 `TEXINPUTS` 设置为 `.:<paper绝对路径>//:`，使新生成的辅助文件优先于源码目录的旧 main.bbl。不能将仍只有 15 条的旧缓存编译结果误认为恢复完成。

增加引用和必要背景后，当前 PDF 为 11 页。保留现有字号、图表和实验论述；未为维持原页数而删去恢复的引用。

最终检查：29 个正文引用键与 29 个编译条目完全一致；所有结果表的数值行与备份相同。渲染检查全稿及参考文献末页，修正了双栏平衡过早触发造成的纵向溢出，最终没有未定义引用或 Overfull。原图的 PDF 版本提示和 caption/IEEEtran 兼容提示仍保留。源码目录的 main.bbl 已备份并更新，防止后续编译误读旧的 15 条缓存。

# 当前五模型 Agent 数据入稿审计

本轮只把当前 exact-v2 Test-200 的已完成结果加入论文，没有重跑模型或检索器，也没有恢复旧 ChEMBL 模型比较。四个替换模型使用本机完成的实验；DeepSeek 使用已完成的同协议 full-Agent 结果。

## 数值来源与口径

权威快照：[summary.json](results/agent-multimodel-20260909/summary.json)；摘要：[report.md](results/agent-multimodel-20260909/report.md)。JSON 与 `outputs/agent_multimodel_repaired_report/summary.json` 完全相同，SHA-256 为 `10cdc6ca400f938df18e9ab4bce5b126d76f2e2441b87e4b975734291ebddeb8`。其中 `source_sha256` 保存上游评分来源及哈希。

| 模型 | 初始严格字段合规 | 最终可评分 | Exact/200 | Tanimoto (n) | Chemical GED (n) |
|---|---:|---:|---:|---:|---:|
| GPT-5 | 50 | 200 | 37 | 0.3082 (200) | 5.4603 (189) |
| o3 | 122 | 200 | 41 | 0.3352 (200) | 5.0890 (191) |
| Claude Sonnet 4.6 | 199 | 199 | 58 | 0.4207 (199) | 4.4404 (193) |
| Gemini 2.5 Flash | 3 | 200 | 33 | 0.2985 (200) | 5.4688 (192) |
| DeepSeek v4 Flash-Ascend | 未在新汇总独立采集 | 200 | 44 | 0.3575 (200) | 5.0785 (191) |

所有行共享冻结 seed-0 encoder、Top10 候选及顺序、legacy KB、峰表和 200 个查询。检索 Top1=41/200，Top10=103/200。只比较 full Agent；没有为替换模型增加未经完成的 direct、no-KB 或其他消融结果。

GPT-5 保留原 50 条结果，仅补跑 150 条空正文，上限 32,768 tokens。o3 保留原 122 条结果，78 条空正文先以 16,384 tokens 补跑，其中一条再以 32,768 tokens 恢复。补跑条件是输出失败，不是预测错误。Gemini 不新增请求；通过明确最终索引或唯一候选 SMILES 恢复原选择，解析器不接触真值。新表中的 Valid 是最终可评分率，不是原始格式合规率，也不是所有样本统一加大预算的独立重复。

Claude 相对 DeepSeek 的 exact 差为 7.0 pp，探索性查询配对 bootstrap 95% 区间 [2.5,12.0] pp。200 个查询各属不同分子组，10,000 次重采样、seed=0；无多重比较校正，不用于声明跨独立 API 采样的稳定优越性。结构均值的可算样本不同；共同子集配对差异、全部模型区间、中位数/标准差、legacy GED、nGED、token 与延迟记录均保留在 JSON。未知费用和未采集 DeepSeek 用量保持为空，不填零。

## 写作分配与术语

| 证据 | 类别 | 位置及作用 |
|---|---|---|
| 五模型完整 Agent 结果 | heterogeneity | 总表 III：展示模型替换的差异 |
| Claude 58/200、检索 41/200 | core_discovery | 摘要定位结果；IV-D 展示配对转移，结论综合边界 |
| 有效率恢复与异质预算 | qualification | 表注及 V-B：限制公平预算解释 |
| 模型输入、哈希、全部扩展统计 | provenance_detail | 本文件和 JSON：保留完整追溯链 |

术语固定：full Agent = frozen hard encoder + legacy KB + model；candidate-only DeepSeek 不含 KB；effective validity 不等于 initial strict-field compliance；chemical GED v2 不等于 legacy GED。其他 DeepSeek 消融不被外推为五模型共同知识收益结论。

最短证据链：固定检索证据 → 五模型最终分数 → 查询配对差异 → 预算与格式恢复边界。主表保留 Valid、Exact、Tanimoto 和 chemical GED；扩展统计不重复堆入正文。

本轮替换“仅一个模型家族”和统一 4,096 tokens 的跨模型暗示；后者仅保留在明确的 DeepSeek 原实验段落。移除旧 MODIFY 数值和旧 ChEMBL 失败重复段，压缩成本/来源描述，完整旧记录保留在备份和既有审计。没有更改竞品数据或训练结果。

空白分词计数（含 LaTeX 命令及表格，不等于自然语言字数）：IV-D 476 → 628；`discussion_repro.tex` 1119 → 1015；`main.tex` 1124 → 1135。新增信息以局部替换和压缩抵消，最终 PDF 为 9 页、5 张编号表。

## 验证与恢复

- 五模型表行已逐字段程序核对最终 JSON，包括结构指标的可计算 n。
- 术语一致性脚本没有报告术语冲突；数值精度提示涉及摘要舍入、表格精度和不同单位的同值数字，经人工检查无数值矛盾。
- 使用 pdfLaTeX、BibTeX、两轮 pdfLaTeX 完成交叉引用；最终检查无未定义引用或 Overfull。IEEEtran 与既有 caption 宏包的兼容提示保留。
- 修改前四个论文源文件及 PDF 保存在 `tmp/pdfs/agent-models/before/`，可以恢复；未删除历史实验产物。
- 工作期间其他修改将 Figure 1 设置为待作者定稿的占位图，并更新其引用/图注。本轮保留这些修改；9 页是包含该占位的当前版本，并非图稿已完成的声明。发布前共享 PDF 的当时版本另存为 `tmp/pdfs/agent-models/before/publish_previous.pdf`。

构建目录为 `tmp/pdfs/agent-models/`。BibTeX 必须在此构建目录运行，并让 `BIBINPUTS`、`BSTINPUTS` 指向 `paper/`，避免读取源码目录中的旧辅助文件。

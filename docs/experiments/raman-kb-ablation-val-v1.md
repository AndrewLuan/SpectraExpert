# Raman 知识库开发验证消融

本实验比较模式级规则、候选子结构过滤和紧凑提示。实现位于 `scripts/raman_mode_evidence.py`
与 `scripts/run_raman_kb_ablation.py`。旧规则库保持为对照；本轮新增的是经来源逐项核对的
15 条模式级记录，不代表已完成全部 181 条旧规则的化学审校。

每条新记录保存一个振动模式、一个频率区间、Raman 相对强度、适用 SMARTS、来源键和
页码。实际核对来源是 [HORIBA RA TN-18](https://static.horiba.com/fileadmin/Horiba/Products/Scientific/Molecular_and_Microanalysis/MacroRAM/RA-TN18-Raman_Data_and_Analysis.pdf)，
该技术说明自身引用 Socrates；不能把本轮核对描述为直接复核了 Socrates 原书。
SMARTS 是项目中的保守结构筛选编码，文献未直接提供。未说明的物相保持 unspecified，
相对强度不转换成观察到该峰的概率。无匹配时不退回混合规则，以免绕过过滤。

## 冻结协议

使用 `raman_top10_surrogate_run01/hard/validation.json` 的 149 条实验验证谱，142 个
分子组，复用对应 Encoder Top10 与 10,704 个候选的验证协议。分子组与训练实验谱和
受保护的最终 200 条测试查询不重叠。但该验证集曾用于 Encoder 选型，本实验只能解释为
开发验证，不能称为从未观察的最终测试。候选理论谱可参与训练是原闭库协议的一部分。

六组均用 deepseek-v4-flash-ascend、temperature=0.2、max_tokens=4096，以及相同峰表、
候选顺序和系统提示。每个查询六组交错排队，合计 894 次请求；已成功请求不因选错而重问。

| 组 | 知识来源 | 候选过滤 | 提示 |
|---|---|---|---|
| no_kb | 无 | 无 | 原峰表与 Top10 |
| legacy_kb | 旧 Stage 2 | 无 | 原规则摘要 |
| modes_all_full | 模式级 Raman 记录 | 关 | 全部命中及解释 |
| modes_all_short | 同上 | 关 | 最多 8 条紧凑证据 |
| modes_filtered_full | 同上 | 开 | 全部适用命中及解释 |
| modes_filtered_short | 同上 | 开 | 最多 8 条紧凑证据 |

过滤组保留至少一个候选满足 SMARTS 的记录，并给出兼容候选索引。该对比同时测试
过滤和显式候选映射，不声称能拆分二者。短提示组同时删减文字与限制八条，不能解释为
纯 token 长度因果实验。规则排序在新库四组间一致：区间距离、区间宽度、规则 ID。
不依据验证标签调整阈值、规则范围或候选。新库和旧库之间的差异同时涉及资料来源与
结构化方式，只作为整体方案比较。

## 核验与产物

已验证普通羰基不会匹配 CO2 的 2349 cm^-1；腈与炔的匹配分别链接正确候选；无对应
结构的模式被排除；输出保留来源键和页码。全部请求和代码依赖保存 SHA-256。

远端输出目录为 `outputs/raman_kb_ablation_val_run01`。`evidence/` 保存检索证据，
`requests/` 保存实际提示，`attempts/` 保存 HTTP 尝试和完整响应，`responses/` 保存结果。
`summary.json` 提供 exact、有效率、分子组等权 exact、Tanimoto 和相对无知识库组的
配对得失；`usage_summary.json` 按组记录接口报告用量。缺失用量保持未知。
每组只有一次生成，不据此宣称跨采样种子的稳定显著改善。

论文引用可使用同目录 `raman-kb-sources.bib` 中的 `horibaRATN18`；新实验结果生成前
不写入论文结果表，也不将旧 44/200 结果归因于此次修复。

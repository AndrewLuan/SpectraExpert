# 最新 encoder 与 kNN 的 Top10＋LLM 选择评测

本轮将同一实验查询的十个检索候选与实验峰表交给 LLM，检验语言模型能否从候选中选出
完整 canonical SMILES 一致的结构。两组各 200 条请求，使用同一峰检测、系统提示、
候选格式和生成参数。这里评估的是“检索＋一次 LLM 选择”，尚未加入知识库、结构修改
或多轮 agent 工具调用。

**评测已完成：encoder＋LLM 的最终 exact match 为 44/200（22%），kNN＋LLM 为
8/200（4%），raw LLM 为 5/200（2.5%）。encoder＋LLM 的结构指标均优于 raw LLM，
但相对直接取 encoder 首位仅净增加 3 条，LLM 选择仍损失了大量已召回的正确候选。**

## 最终结构指标

下表使用三种方法共同可评分的查询；exact 分母为全部 200 条，GED 超时不补零。

| 指标 | 样本数 | raw LLM | kNN＋LLM | encoder＋LLM |
|---|---:|---:|---:|---:|
| 完整 exact match | 200 | 2.5% | 4.0% | **22.0%** |
| Tanimoto ↑ | 198 | 0.19455 | 0.16059 | **0.34996** |
| legacy GED ↓ | 198 | 9.85859 | 11.91919 | **8.00000** |
| chemical GED ↓ | 162 | 6.38272 | 7.54321 | **5.24691** |

格式兼容解析后两种候选选择的有效率均为 100%，raw LLM 的有效 SMILES 比例为 99%。
这些有效率对应不同的输出任务：前者是选择现成结构，后者是生成结构字符串。

完整各方法可评分样本上的 encoder＋LLM 指标为 Tanimoto 0.35146（200）、legacy GED
7.965（200）、chemical GED 5.02577（194）；kNN＋LLM 分别为 0.15983（200）、
11.94（200）、7.42135（178）。不要将这些不同样本集合的均值当作严格配对比较。

encoder＋LLM 相对 raw LLM 的配对结果：

| 指标 | 配对样本数 | 改善 | 95% bootstrap 区间 |
|---|---:|---:|---:|
| exact match | 200 | +19.5 个百分点 | [14.0, 25.5] 个百分点 |
| Tanimoto | 198 | +0.15541 | [0.10753, 0.20474] |
| legacy GED | 198 | 降低 1.85859 | [0.79280, 2.98485] |
| chemical GED | 181 | 降低 1.22099 | [0.54696, 1.90055] |

这些区间是在本次固定查询与单次生成上做配对重采样，不包含训练种子或 API 采样重复
带来的不确定性。kNN＋LLM 的 exact 比 raw LLM 高 1.5 个百分点，但区间 [-2.0, 5.0]
跨零，Tanimoto 和两种 GED 也未优于 raw LLM。

## 召回与选择的损失

| 检索方法 | 正确结构在 Top10 | LLM 最终选对 | 已召回条件下选对比例 |
|---|---:|---:|---:|
| encoder | 103/200 | 44/200 | 44/103（42.72%） |
| kNN | 22/200 | 8/200 | 8/22（36.36%） |

encoder 直接取第一名能命中 41 条；LLM 新增选对 15 条，同时把原本第一名正确的
12 条改错，净增 3 条。LLM 在 59 条已含正确候选的列表中仍选错。kNN 从第一名命中
1 条提高到 8 条，新增 8 条、改错 1 条。这说明当前收益主要依赖 encoder 提供更好的
候选库；“给十个结构和实验峰表”这一选择提示尚未充分利用候选覆盖。

本轮只提供实验峰表，没有提供候选理论峰表；多轮工具、候选谱对齐证据或重排分数融合
尚未评估。下一步设计应在独立验证集检验，不能直接针对这 200 条的错误调整后宣称为
新的未观察测试结果。

## 冻结的检索与提示输入

encoder 使用 `raman_top10_surrogate_run01/hard/best.pt`，SHA-256 为
`023da9724091f58a9ee897ddc07000848f6974093093cfa3cefbbe8f3149b8c5`。索引为修正版的
10,000 个完整结构，查询为固定 200 条实验谱。直接消费上轮已保存的 Top10，避免重新
训练或更换候选。encoder 的候选覆盖为 103/200（51.5%），kNN 为 22/200（11%），
这是只允许在列表内选择时，各组最终 exact match 的上限。

每条提示只包括实验谱峰表及按检索顺序排列的 10 个 SMILES。峰表记录位置、强度、
prominence 与宽度；对全部 200 条查询逐条核对已有 raw LLM baseline 的峰提示一致。
不提供真实结构、命中标记、候选身份 ID、数值检索分数或候选理论谱峰。检索次序作为
先验明确告知模型，不随机打乱，两组保持同一规则。

LLM 输出唯一的 `SELECTED_INDEX: 1..10`，评分器将其映射到原候选 SMILES。没有合法
唯一索引的输出计为无效选择，不自动回退到第一名，也不根据真实结构修复选择。
因此有效率体现选择格式是否有效；不能把映射得到的有效 SMILES 解释为生成语法能力。

原严格解析遗漏了一条整行加粗的 `SELECTED_INDEX`。统一去掉 Markdown 粗体/代码
包裹后，仍要求唯一的 1..10 整数，只改变 kNN 查询索引 60 的格式有效性，不改变 exact
命中数。原严格评分完整保留；本文主表使用 `format_normalized/` 下的格式兼容评分。
该修正未重新调用 API，也没有利用真实结构决定解析规则。

参数沿用 raw baseline：temperature=0.2、max_tokens=4096。初次请求模型名为
`deepseek-v4-flash`，其中两个 key 返回 `key_model_access_denied`。用户明确要求切换
为 `deepseek-v4-flash-ascend` 并确认两者为同一模型；据此复用已有 93 条成功输出，
剩余 307 条使用新模型名补跑。原 93 条的请求文件仍保留原模型名，不伪装为新请求。
这种别名等价关系来自用户确认，实际请求模型名与 API 返回模型名均逐条保留。

两组请求按查询交错进入同一队列，共用三个 key 的限流器。每 key 最多 5 个并发请求，
发送间隔不小于 3.2 秒，最多尝试 5 次；429 按共享冷却等待，403 不对同一请求重试。
所有成功响应均复用，模型选择错误或格式不符合要求不会触发重问。

## 完整输出与用量核算

当前服务器产物目录为：
`/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_llm_selection_ascend_run02`。
旧请求目录 `raman_top10_llm_selection_run01` 完整保留，旧运行器也存为
`runner_source.py`，旧调度已停止，避免与新运行重叠占用 key 并发。

| 产物 | 内容 |
|---|---|
| `requests/` | 每条实际使用的完整请求 JSON，包括提示与生成参数 |
| `attempts/` | 每次 HTTP 尝试的完整响应正文、完整解析 JSON、usage、状态码、耗时及 key 槽位 |
| `responses/` | 每条成功或最终失败响应的可解析索引，链接最后一次尝试 |
| `tasks.json` | 请求到查询、检索方法、十个候选及冻结排名的本地映射 |
| `model/selections.json`、`raw_knn/selections.json` | LLM 所选候选及结构是否精确匹配 |
| 各组 `results.json` | 与旧 baseline 同定义的完整结构指标及逐查询结果 |
| `usage_summary.json` | 按方法及 key 槽位累计 API 报告的数值 usage 字段，包含已收到的重试 |
| `analysis.json` | 共同可评分样本比较、配对区间、检索首位对照及 API 模型名统计 |

上述根目录评分为原严格解析；格式兼容后的完整结果与分析分别保存在
`format_normalized/model/results.json`、`format_normalized/raw_knn/results.json` 和
`format_normalized/analysis.json`，变更审计在 `format_normalized/changes.json`。

完整响应 JSON 包含接口返回的正文、reasoning_content、usage、finish_reason、response ID
等所有字段；仅在错误正文回显已知密钥时将其替换并标注。认证头不落盘，响应头只保留
内容类型、时间、请求 ID 和重试等待等诊断字段。密钥仅通过标准输入进入内存。

新尝试在发送前即持久化 `in_flight` 状态，收到响应后更新同一记录。旧调度切换时，
已收到的 280 条响应全部保留，包括 93 条成功与 187 条 403；未收到响应的在途调用
无法恢复服务端 usage，明确记为未知。汇总仅代表接口已报告的用量，不能承诺等于账单。
缺失 usage 不按零计算，嵌套的 reasoning/cache token 与总 token 可能重叠，不能相加。
不依据未知单价推断货币费用。

调用阶段已完成：400 条请求全部取得 HTTP 200，成功响应均带 usage。共保存 587 条
已收到的 HTTP 尝试，其中 187 条为旧路由的 403；这些错误没有返回 usage。新路由完成
307 次成功调用，另复用 93 次旧路由成功输出。

| 请求组 | 已报告输入 tokens | 已报告输出 tokens | 合计 |
|---|---:|---:|---:|
| encoder＋LLM | 135,452 | 83,386 | 218,838 |
| kNN＋LLM | 136,376 | 82,276 | 218,652 |
| 合计 | **271,828** | **165,662** | **437,490** |

接口另外报告 512 个 cached prompt tokens，属于输入 token 的明细，不在上述总量之外
再次累加。三个 key 槽位的总 token 分别为 220,152、106,277、111,061。旧路由成功
输出集中在槽位 1，因此三个槽位总用量不同，不表示新路由未使用三个 key。

## 评价与复现

报告最终 exact match、Tanimoto、legacy GED、chemical GED、有效选择率，并与同一
200 条 raw LLM baseline 比较。GED 使用原定义及超时策略，缺失值保留，另外比较各方法
共同可评分样本。单独记录“正确结构已在 Top10 时的选对比例”和相对检索第一名的
新增/丢失命中，区分候选召回问题与 LLM 选择问题。

入口为 `scripts/run_raman_candidate_llm.py`，汇总入口为
`scripts/summarize_raman_candidate_llm.py`。测试
`tests/scripts/test_raman_candidate_llm.py` 检查唯一有效索引解析、Markdown 格式兼容、
重试/缺失 usage 汇总及完整 HTTP 响应留档，四项均通过。请求、源码和输入产物哈希写入 config；旧输出不被
新结果覆盖。

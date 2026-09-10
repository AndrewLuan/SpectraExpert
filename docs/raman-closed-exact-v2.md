# Raman 闭库完整结构覆盖修复

本次修复把闭库约束从“同分子组在库”收紧到“完整 canonical SMILES 在库”。
规范化使用 RDKit canonical/isomeric SMILES，保留同位素与立体信息。
这是按已知目标身份构建候选库的闭库实验；测试实验谱不进入模型训练或选型。

新版数据已生成，修复进程退出码为 0。独立读取索引再次确认：10,000 个唯一完整结构、
200/200 精确覆盖、训练 9,500/验证 500、全部测试身份的理论记录在训练集。
新版 sample manifest SHA-256 为
`422e6729cad919b76789608db93faaad89276bb2c2397917d446d8acad41ef4c`。

## 数据版本与修改规则

服务器新版目录：
`/home/user/IRfingerprints/SpectrExpert/outputs/raman_closed_10k_test200_seed0_exact_v2`。
父目录为同级 `raman_closed_10k_test200_seed0`，原数据、checkpoint 和评测产物保留。

对父版本的 21 个完整结构缺口，全量检查了 processed continuous 理论 Raman 的
424,929 条记录，在对应分子组内找到 13 个目标的精确理论记录；另外 8 个没有精确匹配。

- 13 个目标同时替换候选结构及对应理论谱，保留测试实验谱。
- 8 个目标替换候选与测试槽位，保留 mol_id 和测试顺序。新分子组不得占用旧候选组。
- 新组按 `SHA256("0:exact-replacement-groups:" + group_hash)` 排序；组内完整结构及
  记录也按显式命名空间的 seed 哈希排序，重复运行选择相同。过程不读取模型排名或命中率。
- 替代实验谱与理论谱必须完整结构一致，且通过已有的有限值、覆盖范围和非常数谱检查。

`repair_report.json` 保存 21 个修改槽位、8 个替代目标及源分片行号核验信息。
`sample_manifest.json` 保存父版本和输出哈希，`training_split.json` 绑定新版数据。
训练仍为 9,500 条、理论验证为 500 条，全部 200 个测试身份的理论谱受保护进入训练。

## 验证与索引

修复脚本核对父文件哈希，将选中缓存逐条比对原始分片的结构、来源、覆盖和谱值；
缓存不能代替源数据核验。生成后逐条检查 Parquet 与原谱插值一致，并检查保留的
192 条测试谱逐点相等。保留原始数值精度，避免 float32 提前舍入改变插值结果。

发布前必须满足：候选完整结构唯一数 10,000、测试数 200、完整结构覆盖 200/200，
以及正式训练划分校验器通过。所有文件先写 staging，通过验证才发布最终目录。

`knn_candidate_index.npz` 是新版原始强度余弦索引，包含候选 ID、完整结构及 manifest
哈希；`index_status.json` 保存索引摘要。模型图索引需要在新版数据上训练得到的 checkpoint，
本次不把旧模型的索引作为新版训练结果，也不启动后续域适配训练。

后续域适配实验已另行完成，得到独立谱间配对模型及新版模型索引；完整 exact match
为 28/200（14%）。详细协议与限制见 [域偏移实验报告](raman-domain-shift-exact-v2.md)。

## raw LLM baseline

新版 baseline 输出目录为同级
`raman_llm_direct_deepseek_v4_flash_test200_exact_v2_run01`。

补跑入口 `scripts/refresh_closed_llm_baseline.py` 核对原模型、endpoint、温度、token
上限，以及峰检测、提示词和评分器代码哈希。192 条未改查询还必须满足输入谱、真实结构
和实际 API 请求逐条一致，才复用已有响应和结构距离；分子组映射按新版库重新计算。
另外 8 条使用原 `deepseek-v4-flash` Direct 设置补跑。新旧 200 条样本并不完全相同，
新版 baseline 已补跑完成，后续使用新版聚合指标，不能沿用旧版 200 条聚合指标作最终对照。

本次替换的 query_index（从 0 开始）为 38、63、68、76、90、175、195、198；
对应 mol_id 为 3723、5055、9654、3004、5310、8562、3301、4278。
8 条 API 补跑及结构评分已完成，合并后 200 条、API 错误为 0。192 条旧请求与响应逐条核对
完全一致，旧版 baseline results 的 SHA-256 未变。8 条新增预测均有效，但 exact match 为 0/8。

用户提供三个 key，每 key 最多 5 并发。调度改为交错启动，使首批八条分摊为 3/3/2；
模型、请求内容及每 key 3.2 秒发送间隔不变。第 2、3 个 key 的五次请求返回 HTTP 403，
原因未进一步确定；随后仅用第 1 个 key 完成这五条，不重复请求已成功的三条。
共八次成功响应、五次被拒绝尝试。凭据只经 stdin 进入内存，不记录 key 内容。
`invocation.json` 保存调用审计；准备时的旧代码哈希配置保存在
`config_before_key_interleave.json`，当前 `config.json` 绑定交错调度版本。

| 新版口径 | 有效率 | 完整 exact match | Tanimoto ↑ | legacy GED ↓ | chemical GED ↓ |
|---|---:|---:|---:|---:|---:|
| 严格字段解析 | 99% | 2.5%（5/200） | 0.194553（198） | 9.858586（198） | 6.301075（186） |
| 原解析器 | 99.5% | 2.5%（5/200） | 0.193576（199） | 9.859296（199） | 6.320856（187） |

括号为可计算样本数；完整 exact match 的分母固定为 200。GED 沿用 10 秒墙钟时限，
超时为 null，各指标可计算子集不同。与后续模型比较时还需报告相同可计算样本的配对结果。
新版 `results.json` SHA-256：
`8f8960fa9b7f87fd8561b3ba3bd489c55c2c691691ae0d0794a2935b97ffcafe`。

检索结构评分入口已改为绑定检索协议中的 manifest hash，支持新旧两版数据，仍拒绝
将旧版排名与新版数据混合计分。

默认入口仅准备请求，不调用 API。正式补跑在服务器项目目录执行以下命令，所用进程须
通过安全渠道设置 `USTC_API_KEY`，或使用 `--keys-stdin` 传入仅驻留内存的凭据。
不要把密钥写入命令、项目文件或日志。

```bash
/home/user/.venvs/spectrexpert/bin/python -u scripts/refresh_closed_llm_baseline.py \
  --data-root outputs/raman_closed_10k_test200_seed0_exact_v2 \
  --parent outputs/raman_closed_10k_test200_seed0 \
  --previous outputs/raman_llm_direct_deepseek_v4_flash_test200_run01 \
  --output outputs/raman_llm_direct_deepseek_v4_flash_test200_exact_v2_run01 \
  --run
```

续跑校验请求与配置，只发送尚未成功的响应槽位。新增样本复用原评分器计算 GED，
最终按原槽位顺序合并为 200 条；无法计算的 GED 保留 null，不算零距离。

## 本地验证

18 项测试通过，覆盖完整结构与同位素/立体差异、替代槽位与确定性、质量及来源拒绝、
数值精度保留、闭库训练划分、192/8 请求复用、请求漂移拒绝、并发续跑及指标缺失口径。
本次补跑新增八请求分摊到三 key 的测试，baseline 调度相关四项测试通过。
实际数据发布状态和 baseline 准备状态见服务器对应目录中的清单与 `status.json`。

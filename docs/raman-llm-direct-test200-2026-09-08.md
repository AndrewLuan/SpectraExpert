# Raman 200 条实验谱：纯 LLM baseline

本次使用已冻结的 `raman_closed_10k_test200_seed0` 数据，运行原论文的 LLM Direct：
只将检测峰表提供给模型，直接预测一个 SMILES。全部 200 条实验谱均参与，不重新抽样。
模型固定为 `deepseek-v4-flash`，端点为 `https://api.llm.ustc.edu.cn/v1`。

**已于北京时间 2026-09-08 21:25:34 完成，退出码 0；200/200 条 API 请求成功。**
按用户给出的 key 顺序，第 2、3 把没有该模型权限，最终全部使用第 1 把完成。

| 指标 | 严格字段解析 | 原论文解析器 |
|---|---:|---:|
| 有效 SMILES | 198/200（99%） | 199/200（99.5%） |
| 完整 canonical exact | 5/200（2.5%） | 5/200（2.5%） |
| 冻结分子组命中 | 5/200（2.5%） | 5/200（2.5%） |
| Tanimoto 均值（有效预测） | 0.196240（n=198） | 0.195254（n=199） |
| legacy GED 均值 | 9.641414（n=198） | 9.643216（n=199） |
| chemical GED 均值 | 6.256684（n=187） | 6.276596（n=188） |

两种解析器有 1 条预测不同：query_index=139、mol_id=9430 的回答被长度截断，
严格解析没有预测；旧解析器从正文兜底提取了 `CCl`。chemical GED 在 11 条有效预测上没有在时限内完成，
不能把其较低的均值解释为同一批样本上对 legacy GED 的改进；算法和可计算样本均不同。
199 条回答以 `stop` 结束，1 条以 `length` 结束；遵循原 4096-token 预算，没有选择性加长重跑。
200 条最终响应累计 prompt=131167、completion=122609、total=253776 tokens；
此用量不包含单独的 key 连通性探测或未保存的中断请求。
最终 `results.json` SHA-256 为
`dafba8b29140fbe74c9160f7ad68ec5fb2a4835120ad8e89445e395eee6a65be`。
已核验请求、响应、指标各 200 条，query mol_id 唯一，所有返回模型均为 `deepseek-v4-flash`。

## 与原论文代码的对应

内容来源为 `paper/main.tex` 的 LLM Direct 设置；该论文的原始 PDF 来源见
`paper/SOURCE_PROVENANCE.md`。具体推理过程复用
`spectraexpert/eval/evaluate.py` 中 `_PeakDetector`、`parse_llm_smiles`，以及
`spectraexpert/pipeline/prompts/stage3_direct_identification.py` 的原始 system/user prompt。
温度为 0.2，`max_tokens=4096`，每条谱生成一次；不因无效 SMILES 或低置信度重试。

输入为 500–4000 cm⁻¹、间隔 1 cm⁻¹ 的 3501 点谱。检测前按原代码进行逐谱 z-score；
height_rel=0.05、prominence_rel=0.02、distance=5、width=1，按强度降序排列全部峰。
候选库、真实 SMILES、分子式、分子组、来源和 Nakamoto 规则均不进入 prompt。
这是直接结构生成；它没有十个排序候选，因此不报告检索 Top-5/10 或 MRR。

读取时按 mol_id 连接，不将测试行号视为候选 ID。启动前核验采样清单的固定 SHA-256、
使用到的实验文件和候选身份文件哈希，确认 10000 个候选分子组与 200 个实验分子组的映射。
不加载 10000 条理论光谱，也不读取原始大数据目录。

## 解析与指标

原解析器会在没有有效结构字段时尝试将正文单词解析为 SMILES，因此同时记录两种口径：

- `legacy`：复用原论文代码的解析函数，方便追溯历史结果。
- `strict`：只接受 `PREDICTED_SMILES` 字段中的有效结构，作为严格输出契约结果。

两者分别报告有效率、完整 canonical SMILES exact match、冻结分子组命中率、Morgan
Tanimoto、`legacy_ged_v1` 与 `chemical_ged_v2`。前者 GED 是旧算法的第一个优化解；
后者要求搜索完成。每次 GED 搜索最多 10 秒，超时保留 null，并报告可计算数量。
有效率和命中率分母为全部 200 条；Tanimoto/GED 的均值只基于可计算结果。

组匹配将预测结构规范化为最大重原子片段（重原子数相同按总原子数）、去立体化学和
同位素标记，再查冻结候选结构的 group_hash。这一映射已在全部 200 条实验身份上验证。
其中 6 条实验/理论源 SMILES 的同位素标记不同；完整 canonical exact 保留这一差异。

## JJ 启动与恢复记录

运行根目录为：
`/home/user/IRfingerprints/SpectrExpert/outputs/raman_llm_direct_deepseek_v4_flash_test200_run01`。
日志为相邻的 `raman_llm_direct_deepseek_v4_flash_test200_run01.log`。

用户给出的第 1 把 key 已实测可调用模型；第 2、3 把返回 HTTP 403，
`key_model_access_denied`，因此当前仅启用第 1 把。密钥通过 SSH stdin 传给进程，
不写入脚本、命令行参数、配置或日志，也不保存在磁盘。

第 1 把 key 除并发限制外，还有网关实测的 20 requests/minute 限制。
最初进程 PID 50608 在成功保存一批结果后触发 429，已停止。
修复后使用每 key 最多五个 worker、请求间隔 3.2 秒，429 遵守 Retry-After 且共享冷却，
每条请求最多五次尝试；使用 `--resume` 恢复，PID 为 51143。
成功响应不重复调用，失败响应移入 `failed_attempts/` 后重试，初始配置保留不覆盖。
恢复进程于 21:10:10 开始，21:19:58 完成 API 推理，21:25:34 完成指标计算。
`exit_code=0`、`finished_at` 和 `results.json` 已核验，最终结果见本文开头。

运行器为 `scripts/run_closed_llm_direct.py`。各目录含义如下：

- `requests/`：完整逐条 prompt 和推理参数；可审计是否泄漏真实结构。
- `responses/`：模型回答、finish_reason、token usage、请求耗时和匿名 key 槽位。
- `metrics/`：逐条预测、真实身份及两种解析口径的结构指标。
- `config.json`、`resume_config_*.json`：初始和恢复参数、代码 SHA-256、数据哈希和 query 顺序。
- `results.json`：最终聚合与全部逐条指标。
- `inference_finished_at`：API 推理阶段结束时间；GED 计算可能仍在继续。
- `run.pid`、`exit_code`、`finished_at`：进程与完成状态；0 表示无 API 错误，2 表示仍有失败请求。

通过 `subprocess.Popen(start_new_session=True)` 脱离 SSH，CPU 调度 API 与四路 GED，
不使用 GPU。没有修改或停止同时运行的模型训练。

## 本地验证

`python -m pytest tests/scripts/test_run_closed_llm_direct.py -q`：3 项通过。
验证三把模拟 key 各自最多五并发、41 条结果完整保存、成功结果不重复调用、失败记录归档、
同 key 发送间隔，以及严格字段解析和同位素组规范化。服务器 `--prepare-only` 验证 200 条数据通过。
脚本、原峰检测代码、原 prompt、指标模块的本地/远端 SHA-256 一致。
仅同步了新运行脚本，先 rsync dry-run 再实际同步，没有传输或覆盖数据、训练输出或 checkpoint。

# 实验 Raman → 理论 Raman 重合组 kNN pilot

2026-09-08 在服务器完成。余弦相似度的分子组等权 Top-1 为 3.93%，Top-10 为
15.56%。该结果说明，在本次预处理和候选池下，直接谱形匹配的识别率较低；它不能单独
证明模型学习或 LLM 推理能带来改善。运行没有使用训练、GPU 或 LLM。

## 数据与任务

只读索引 `/data/SpecGPT/dataset/data_index/specgpt_dataset_index.sqlite`，筛选
`exp_raman_count > 0 AND theory_raman_count > 0`：950 个分子组，1,018 条实验谱、
1,636 条理论谱。身份沿用索引的最大分子片段与无立体化学 parent connectivity，
因此命中不是完整盐型或立体异构体级别的精确识别。

实际读取 `/data/SpecGPT/dataset/processed_stage_2_v1/continuous/combined/` 下
`exp/raman/{train,valid,test}.jsonl.gz` 和 `theory/raman/{train,valid,test}.jsonl.gz`。
这些是完整数据的处理产物，不是 `/data/SpecGPT/data` 的 tiny 文件。实验扫描 3,534 条记录，
理论扫描 424,929 条记录，仅提取索引中重合组；匹配记录数与索引完全一致。

查询为实验谱，候选为重合组的理论谱。这是含真实目标的闭库、跨来源描述性 pilot，
使用全部原始 split，不是训练后的 held-out 评估，也不是原论文 QM9S→ChEMBL 库外推断。
尚未测量加入全部 390,421 理论分子组后的检索结果。

## 固定协议与清洗

- 消费现成 `continuous_v1` 数据：0–4000 cm⁻¹ 的 3000 点网格，已做 1/99 分位
  robust_minmax。pilot 不拟合额外参数、不平移峰位、不优化频率缩放。
- 比较 500–4000 cm⁻¹ 范围内双方 coverage 的交集，至少覆盖该范围的 50%。
  coverage 按半开区间处理，缺失位置的补零不参与相似度。
- 余弦相似度是主基线，Pearson 去均值相关性是固定的辅助诊断。
- 同一候选分子组的多条理论谱取最高相似度；按分子组排序，并列取保守最差排名。
- 同组多条实验谱先组内平均，再对索引全部 950 组等权平均。没有可用查询或目标谱的组
  计零，保留原始分母。Top-K 表示正确分子组是否出现在前 K 个不同分子组中。
- 排除 5 条覆盖不足的实验谱和 17 条在比较区间全零的理论谱。保留 1,013 条实验谱
  （945 组）与 1,619 条理论谱（944 组）；双方都有可用谱的交集为 941 组。
  候选谱包括 1,150 条 QME14S 与 469 条 fg26，实验谱均为 SDBS。

## 结果

| 方法 | Top-1 | Top-5 | Top-10 | MRR |
|---|---:|---:|---:|---:|
| Cosine kNN | 3.93% | 10.76% | 15.56% | 0.0826 |
| Pearson kNN | 3.09% | 10.29% | 15.72% | 0.0747 |

1,013 条可用查询中，1,009 条能计算目标排名。仅在目标排名可用的查询上，Cosine
排名中位数为 94，Pearson 为 98；此排名中位数采用逐谱口径，不是上表的分子组等权口径。

总墙钟时间 113.82 秒，包含流式解压筛选、缓存与评测；检索及逐样本落盘 50.17 秒。
单查询同时计算两种相似度和分子组排名的耗时中位数 25.75 ms，P95 124.25 ms。
该时间是本次 CPU 实现的观察值，不是完整理论库的性能预测，也不是单独 cosine 的测速。

## 可复现产物与验证

脚本：`scripts/pilot_raman_overlap_knn.py`。
本次运行脚本 SHA-256：`98efcaf09fa19a7388e3d9501d225781d1315608514c3903ffe1bbe8ec97a81f`。
经 rsync dry-run 后只同步该脚本，确认两端 SHA-256 相同，未同步或下载上游数据。

服务器产物目录：
`/home/user/IRfingerprints/SpectrExpert/outputs/raman_overlap_knn_20260908_01/`

- `summary.json`：指标、耗时、排除统计与六个输入解压内容的 SHA-256。
- `protocol.json`：950 组清单、索引计数、上游处理 manifest 和固定评测协议。
- `per_query.jsonl`：逐实验谱的目标排名和前十候选组。
- `records.json`、`overlap_arrays.npz`：来源映射及这次新生成的重合组数组缓存。

日志为相邻的 `raman_overlap_knn_20260908_01.log`。本地与远端均通过脚本自检；
另用独立逐候选点积/去均值实现复核查询 0、500、1012 的两种目标排名，全部一致。

复跑须使用新的输出目录，脚本拒绝覆盖现有结果：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  /home/user/.venvs/spectrexpert/bin/python \
  /home/user/IRfingerprints/SpectrExpert/scripts/pilot_raman_overlap_knn.py \
  --output-dir /home/user/IRfingerprints/SpectrExpert/outputs/raman_overlap_knn_new_run
```

下一步宜先检查少量失败样本的实验谱与对应理论谱，区分峰位偏差、基线/强度差异和上游
归一化影响，再决定是否增加一个固定、简单的预处理基线。当前结果不足以直接归因为某一种原因。

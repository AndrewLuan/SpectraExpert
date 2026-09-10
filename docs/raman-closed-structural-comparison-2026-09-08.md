# 与纯 LLM baseline 对齐的结构指标

本次将 kNN 和 spectrum_graph_v2 各自排名第一的结构作为唯一预测，复用
`scripts/run_closed_llm_direct.py` 的 evaluate 函数重新评分。没有从多个候选中按真实
结构挑选答案，没有调用 API、重新训练或重跑检索。200 条查询、mol_id 顺序及真实 SMILES
与已完成的纯 LLM baseline 逐条一致。

主要比较采用 LLM 严格字段解析口径；检索预测的严格与原论文解析结果完全相同。
Tanimoto 越高越好，两版 GED 越低越好。均值后括号给出可计算样本数。

| 方法 | 有效 SMILES | Tanimoto ↑ | legacy GED ↓ | chemical GED ↓ |
|---|---:|---:|---:|---:|
| 纯 LLM，严格解析 | 99% | 0.196240（198） | 9.641414（198） | 6.256684（187） |
| 纯 LLM，原论文解析 | 99.5% | 0.195254（199） | 9.643216（199） | 6.276596（188） |
| 纯 kNN | 100% | 0.117114（200） | 12.515000（200） | 8.204420（181） |
| 我们的模型 | 100% | 0.099635（200） | 11.430000（200） | 7.689474（190） |

纯 LLM 在本次结构指标上优于两个检索方法。模型相较 kNN 的 GED 较低，但 Tanimoto
也较低，因此不能把模型描述为全面优于 kNN。这里没有进行显著性检验。

## 同一批可计算查询的核对

GED 每次仍使用 baseline 的 10 秒墙钟时限、四个评分线程。chemical GED 在 kNN 的
19 条、模型的 10 条预测上超时；严格 LLM 的 198 条有效预测中有 11 条超时。
超时保留 null，不当作零距离，不进入可计算均值。为避免不同样本子集造成误读，额外
取三种方法均可计算的查询交集，保持同一 mol_id 配对：

| 指标 | 共同查询数 | 纯 LLM，严格解析 | kNN | 模型 |
|---|---:|---:|---:|---:|
| Tanimoto ↑ | 198 | 0.196240 | 0.117468 | 0.100061 |
| legacy GED ↓ | 198 | 9.641414 | 12.449495 | 11.414141 |
| chemical GED ↓ | 164 | 6.201220 | 8.359756 | 7.920732 |

共同查询上的方法顺序与各自可计算均值一致。两版 GED 的算法定义不同，数值不应跨版本
直接比较；legacy 是原算法的首个优化解，chemical 要求搜索完成。

## 数据与方法范围

纯 LLM 为 deepseek-v4-flash 的原论文 Direct 峰表→SMILES 设置。
kNN 为冻结 3501 点原始强度的余弦检索；模型为第 23 轮 best checkpoint 的光谱→分子图
余弦检索，不包含 LLM 或辅助重排。两个检索方法使用同一 10000 候选库。全部 200 个
实验测试组的理论谱均参与过训练，但实验谱未进入训练或模型选择；LLM 没有候选库约束。
本表评价最终结构质量，不能把它解释为各方法具有相同训练资源或相同输出约束。

LLM 结果 SHA-256 已重新确认：
`dafba8b29140fbe74c9160f7ad68ec5fb2a4835120ad8e89445e395eee6a65be`。
本地与服务器的 baseline 运行器、指标模块 SHA-256 一致；此前检索排名保持不变。

服务器新结果目录：
`/home/user/IRfingerprints/SpectrExpert/outputs/raman_closed_structural_metrics_run01/`。
`summary.json` 为聚合结果；每个方法子目录中的 `results.json` 包含全部 200 条预测及指标，
`metrics/` 保存逐条结果。`protocol.json` 记录输入排名与评分代码 hash、查询顺序、10 秒
时限和零 API 调用。日志为同级 `raman_closed_structural_metrics_run01.log`。

复现入口为 `python -m scripts.score_closed_retrieval_structures`，需要 `--data-root`、
`--retrieval-dir` 和新的 `--output-dir`。它复用原 baseline 评分器，避免另写一套 GED/Tanimoto 定义。
